"""
sar_snap.py - Sentinel-1 preprocessing through ESA SNAP's Graph Processing Tool (gpt).

This module performs the standard Sentinel-1 GRD chain with SNAP itself
(no re-implementation of the physics):

    Read -> [Subset] -> Apply-Orbit-File -> ThermalNoiseRemoval -> Calibration (sigma0)
         -> Speckle-Filter -> Terrain-Correction (Range-Doppler) -> LinearToFromdB -> GeoTIFF

It has no Qt dependency so it can be unit-tested and used from scripts.
SNAP must be installed separately (https://step.esa.int). The input must be the
original Sentinel-1 product (.zip, .SAFE directory or manifest.safe); a GeoTIFF
export cannot be orbit-corrected or radiometrically calibrated.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

ProgressCb = Optional[Callable[[int, str], None]]
CancelCb = Optional[Callable[[], bool]]

ORBIT_TYPES = (
    "Sentinel Precise (Auto Download)",
    "Sentinel Restituted (Auto Download)",
)
SPECKLE_FILTERS = ("None", "Boxcar", "Median", "Frost", "Gamma Map", "Lee", "Refined Lee", "Lee Sigma", "IDAN")
DEM_NAMES = ("SRTM 3Sec", "SRTM 1Sec HGT", "Copernicus 30m Global DEM", "Copernicus 90m Global DEM", "External DEM")

_S1_NAME = re.compile(
    r"^(?P<mission>S1[ABCD])_(?P<mode>S[1-6]|IW|EW|WV)_(?P<ptype>GRD|SLC|OCN)(?P<res>[FHM_])_"
    r"(?P<level>\d)(?P<cls>[SA])(?P<pol>SV|SH|DV|DH)_(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})(?:_(?P<orbit>\d{6})_(?P<datatake>[0-9A-F]{6}))?",
    re.IGNORECASE,
)


class SnapError(RuntimeError):
    """Raised for any SNAP/GPT failure with a human-readable message."""


class Cancelled(SnapError):
    pass


# --------------------------------------------------------------------------- discovery
def _candidate_gpt_paths() -> List[Path]:
    exe = "gpt.exe" if os.name == "nt" else "gpt"
    cands: List[Path] = []
    home = Path.home()
    if os.name == "nt":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), str(home)):
            for ver in ("snap", "esa-snap"):
                cands.append(Path(base) / ver / "bin" / exe)
                cands.append(Path(base) / "SNAP" / "bin" / exe)
        cands += [home / "snap" / "bin" / exe]
    elif sys.platform == "darwin":
        cands += [Path("/Applications/snap/bin") / exe, home / "Applications" / "snap" / "bin" / exe,
                  home / "esa-snap" / "bin" / exe]
    else:
        cands += [home / "snap" / "bin" / exe, home / "esa-snap" / "bin" / exe,
                  Path("/opt/snap/bin") / exe, Path("/usr/local/snap/bin") / exe]
    return cands


def find_gpt(user_path: Optional[str] = None) -> str:
    """Locate the gpt executable: explicit path, $SNAP_GPT, $PATH, then standard install folders."""
    tried: List[str] = []
    for p in (user_path, os.environ.get("SNAP_GPT")):
        if p:
            tried.append(p)
            if Path(p).is_file():
                return str(Path(p))
    on_path = shutil.which("gpt")
    if on_path:
        return on_path
    for p in _candidate_gpt_paths():
        tried.append(str(p))
        if p.is_file():
            return str(p)
    raise SnapError(
        "SNAP 'gpt' was not found. Install ESA SNAP (https://step.esa.int) and either set the "
        "SNAP_GPT environment variable to the full path of gpt, or choose it in the settings.\n"
        "Looked in:\n  " + "\n  ".join(tried[:8]))


# --------------------------------------------------------------------------- product checks
@dataclass
class ProductInfo:
    path: str
    mission: str
    mode: str
    product_type: str
    polarisations: List[str]
    start: str
    abs_orbit: Optional[int] = None
    datatake: Optional[str] = None

    @property
    def relative_orbit(self) -> Optional[int]:
        """Relative orbit (track) number, 1..175. Known offsets: S1A 73, S1B 27."""
        off = {"S1A": 73, "S1B": 27}.get(self.mission)
        return None if (off is None or self.abs_orbit is None) else (self.abs_orbit - off) % 175 + 1

    @property
    def is_grd(self) -> bool:
        return self.product_type == "GRD"


def _strip_product_suffix(name: str) -> str:
    """'X.SAFE.zip' / 'X.zip' / 'X.SAFE' -> 'X'."""
    prev = None
    while prev != name:
        prev = name
        name = re.sub(r"\.(zip|safe)$", "", name, flags=re.IGNORECASE)
    return name


def parse_product_name(path: str) -> ProductInfo:
    """Validate that `path` looks like an original Sentinel-1 GRD product and extract its identity."""
    p = Path(str(path).replace("\\", "/"))      # tolerate Windows separators on any OS
    if p.is_dir() and (p / "manifest.safe").is_file():
        p = p / "manifest.safe"                   # SNAP opens an unzipped .SAFE folder through its manifest.safe
    name = p.parent.name if p.name.lower() == "manifest.safe" else p.name
    stem = _strip_product_suffix(name)
    m = _S1_NAME.match(stem)
    if not m:
        raise SnapError(
            f"'{name}' is not an original Sentinel-1 product name (expected S1A_IW_GRDH_1SDV_...). "
            "Orbit correction and sigma0 calibration need the original .zip/.SAFE, not a GeoTIFF export.")
    pol = m.group("pol").upper()
    pols = {"SV": ["VV"], "SH": ["HH"], "DV": ["VV", "VH"], "DH": ["HH", "HV"]}[pol]
    info = ProductInfo(str(p), m.group("mission").upper(), m.group("mode").upper(),
                       m.group("ptype").upper(), pols, m.group("start"),
                       int(m.group("orbit")) if m.group("orbit") else None,
                       m.group("datatake").upper() if m.group("datatake") else None)
    if not info.is_grd:
        raise SnapError(f"Only GRD products are supported by this workflow (found {info.product_type}).")
    return info


# --------------------------------------------------------------------------- parameters
@dataclass
class SnapParams:
    polarisation: str = "VV"
    orbit_type: str = ORBIT_TYPES[0]
    remove_thermal_noise: bool = True
    speckle_filter: str = "Lee"          # "None" to skip
    speckle_window: int = 5
    dem_name: str = "SRTM 3Sec"
    external_dem: Optional[str] = None
    pixel_spacing_m: float = 10.0
    map_projection: str = "WGS84(DD)"    # SNAP's own default, geographic WGS84; also "AUTO:42001" (UTM) or "" to omit
    subset_mode: str = "operator"        # "operator" | "read" (Read.geometryRegion) | "after" (crop after terrain correction)
    nodata_at_sea: bool = False          # keep water pixels near sea level (important for flood mapping)
    subset_wkt: Optional[str] = None     # POLYGON((lon lat, ...)) in WGS84
    subset_stage: str = "start"          # stepwise mode only: "start" (slant range) or "end" (after terrain correction)
    output_db: bool = True

    def validate(self, info: Optional[ProductInfo] = None) -> None:
        if self.polarisation not in ("VV", "VH", "HH", "HV"):
            raise SnapError(f"Invalid polarisation '{self.polarisation}'.")
        if info and self.polarisation not in info.polarisations:
            raise SnapError(f"Polarisation {self.polarisation} is not in this product "
                            f"(available: {', '.join(info.polarisations)}).")
        if self.speckle_filter not in SPECKLE_FILTERS:
            raise SnapError(f"Unknown speckle filter '{self.speckle_filter}'.")
        if self.speckle_window < 3 or self.speckle_window % 2 == 0:
            raise SnapError("Speckle window must be an odd number >= 3.")
        if self.dem_name not in DEM_NAMES:
            raise SnapError(f"Unknown DEM '{self.dem_name}'.")
        if self.dem_name == "External DEM" and not (self.external_dem and Path(self.external_dem).is_file()):
            raise SnapError("External DEM selected but the file does not exist.")
        if self.pixel_spacing_m <= 0:
            raise SnapError("Pixel spacing must be positive.")
        if self.subset_mode not in ("operator", "read", "after"):
            raise SnapError("subset_mode must be 'operator', 'read' or 'after'.")
        if self.subset_stage not in ("start", "end"):
            raise SnapError("subset_stage must be 'start' or 'end'.")
        if self.subset_wkt and not self.subset_wkt.strip().upper().startswith("POLYGON"):
            raise SnapError("Subset region must be a WKT POLYGON in WGS84 (lon lat).")


def sigma0_band(pol: str) -> str:
    return f"Sigma0_{pol}"


def result_band_name(params: SnapParams) -> str:
    return sigma0_band(params.polarisation) + ("_db" if params.output_db else "")


# --------------------------------------------------------------------------- graph XML
def _sub(parent, tag, text=None):
    e = ET.SubElement(parent, tag)
    if text is not None:
        e.text = str(text)
    return e


def build_graph_xml(input_path: str, output_path: str, params: SnapParams) -> str:
    """Return a complete SNAP graph (single gpt call) for the whole chain."""
    params.validate()
    g = ET.Element("graph", id="SpaceGeoVision-S1-GRD")
    _sub(g, "version", "1.0")
    nodes: List[str] = []

    def node(nid, operator, src=None):
        n = ET.SubElement(g, "node", id=nid)
        _sub(n, "operator", operator)
        s = _sub(n, "sources")
        if src:
            ET.SubElement(s, "sourceProduct", refid=src)
        cfg = ET.SubElement(n, "parameters", {"class": "com.bc.ceres.binding.dom.XppDomElement"})
        nodes.append(nid)
        return cfg

    prev = "Read"
    cfg = node("Read", "Read")
    _sub(cfg, "file", input_path)
    _sub(cfg, "formatName", "SENTINEL-1")
    if params.subset_wkt and params.subset_mode == "read":
        _sub(cfg, "geometryRegion", params.subset_wkt)
        _sub(cfg, "useAdvancedOptions", "true")
        _sub(cfg, "copyMetadata", "true")

    if params.subset_wkt and params.subset_mode == "operator":
        cfg = node("Subset", "Subset", prev)
        _sub(cfg, "geoRegion", params.subset_wkt)
        _sub(cfg, "copyMetadata", "true")
        _sub(cfg, "subSamplingX", 1)
        _sub(cfg, "subSamplingY", 1)
        prev = "Subset"

    cfg = node("Apply-Orbit-File", "Apply-Orbit-File", prev)
    _sub(cfg, "orbitType", params.orbit_type)
    _sub(cfg, "polyDegree", 3)
    _sub(cfg, "continueOnFail", "false")
    prev = "Apply-Orbit-File"

    if params.remove_thermal_noise:
        cfg = node("ThermalNoiseRemoval", "ThermalNoiseRemoval", prev)
        _sub(cfg, "selectedPolarisations", params.polarisation)
        _sub(cfg, "removeThermalNoise", "true")
        prev = "ThermalNoiseRemoval"

    cfg = node("Calibration", "Calibration", prev)
    _sub(cfg, "selectedPolarisations", params.polarisation)
    _sub(cfg, "outputSigmaBand", "true")
    _sub(cfg, "outputGammaBand", "false")
    _sub(cfg, "outputBetaBand", "false")
    _sub(cfg, "outputImageInComplex", "false")
    _sub(cfg, "outputImageScaleInDb", "false")
    prev = "Calibration"

    if params.speckle_filter != "None":
        cfg = node("Speckle-Filter", "Speckle-Filter", prev)
        _sub(cfg, "filter", params.speckle_filter)
        _sub(cfg, "filterSizeX", params.speckle_window)
        _sub(cfg, "filterSizeY", params.speckle_window)
        prev = "Speckle-Filter"

    cfg = node("Terrain-Correction", "Terrain-Correction", prev)
    _sub(cfg, "sourceBands", sigma0_band(params.polarisation))
    _sub(cfg, "demName", params.dem_name)
    if params.dem_name == "External DEM":
        _sub(cfg, "externalDEMFile", params.external_dem)
        _sub(cfg, "externalDEMNoDataValue", 0)
        _sub(cfg, "externalDEMApplyEGM", "true")
    _sub(cfg, "demResamplingMethod", "BILINEAR_INTERPOLATION")
    _sub(cfg, "imgResamplingMethod", "BILINEAR_INTERPOLATION")
    _sub(cfg, "pixelSpacingInMeter", params.pixel_spacing_m)
    if params.map_projection:
        _sub(cfg, "mapProjection", params.map_projection)
    _sub(cfg, "nodataValueAtSea", "true" if params.nodata_at_sea else "false")
    prev = "Terrain-Correction"

    if params.subset_wkt and params.subset_mode == "after":
        cfg = node("Subset", "Subset", prev)
        _sub(cfg, "geoRegion", params.subset_wkt)
        _sub(cfg, "copyMetadata", "true")
        prev = "Subset"

    if params.output_db:
        node("LinearToFromdB", "LinearToFromdB", prev)
        prev = "LinearToFromdB"

    cfg = node("Write", "Write", prev)
    _sub(cfg, "file", output_path)
    _sub(cfg, "formatName", "GeoTIFF")

    ET.indent(g)
    return ET.tostring(g, encoding="unicode", xml_declaration=False)


# --------------------------------------------------------------------------- execution
_PROGRESS = re.compile(r"(\d{1,3})\s*%")


_ERR_LINE = re.compile(r"(error|exception|caused by|failed|cannot|could not|not found|no such|refused|timed out|denied|unable)", re.I)


def _key_error_lines(lines: List[str], limit: int = 8) -> List[str]:
    """The informative lines of a gpt failure, without Java stack frames ('at org.esa...')."""
    out: List[str] = []
    for ln in lines:
        t = ln.strip()
        if not t or t.startswith("at ") or t.startswith("..."):
            continue
        if _ERR_LINE.search(t) and t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _run(cmd: List[str], progress: ProgressCb, cancelled: CancelCb, label: str,
         log_path: Optional[str] = None) -> str:
    """Run a gpt command, stream progress, honour cancellation, keep the full log, return the output."""
    lines: List[str] = []
    last = 0
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SnapError(f"Could not start gpt: {exc}") from exc

    def save_log():
        if log_path:
            try:
                Path(log_path).write_text("\n".join(lines), encoding="utf-8")
            except OSError:
                pass

    with proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            m = _PROGRESS.findall(line)
            if m and progress:
                last = max(last, min(100, int(m[-1])))
                progress(last, label)
            if cancelled and cancelled():
                _kill_tree(proc, 30)
                save_log()
                raise Cancelled("Processing cancelled by user.")
        rc = proc.wait()
    save_log()
    out = "\n".join(lines)
    if rc != 0:
        key = _key_error_lines(lines)
        shown = "\n".join(key) if key else "\n".join(lines[-15:])
        where = f"\nFull gpt log: {log_path}" if log_path else ""
        raise SnapError(f"gpt failed (exit code {rc}) during '{label}'.\n{shown}{where}")
    if progress:
        progress(100, label)
    return out


def check_gpt(gpt: str) -> str:
    """Run `gpt -h` and return the first line (SNAP version banner) - a cheap installation test."""
    out = _run([gpt, "-h"], None, None, "gpt -h")
    return out.splitlines()[0] if out else ""


def operator_parameters(gpt: str, operator: str) -> List[str]:
    """Names of the -P parameters that this SNAP version accepts for an operator."""
    out = _run([gpt, operator, "-h"], None, None, f"{operator} -h")
    return re.findall(r"^\s*-P(\w+)=", out, flags=re.MULTILINE)


def verify_parameter_names(gpt: str, params: SnapParams) -> List[str]:
    """Compare the parameters we send with what the installed SNAP advertises; return warnings."""
    expected: Dict[str, List[str]] = {
        "Apply-Orbit-File": ["orbitType", "polyDegree", "continueOnFail"],
        "ThermalNoiseRemoval": ["selectedPolarisations", "removeThermalNoise"],
        "Calibration": ["selectedPolarisations", "outputSigmaBand", "outputImageScaleInDb"],
        "Speckle-Filter": ["filter", "filterSizeX", "filterSizeY"],
        "Terrain-Correction": ["demName", "pixelSpacingInMeter", "mapProjection", "nodataValueAtSea",
                               "sourceBands", "demResamplingMethod", "imgResamplingMethod"],
    }
    warnings: List[str] = []
    for op, names in expected.items():
        try:
            have = set(operator_parameters(gpt, op))
        except SnapError as exc:
            warnings.append(f"{op}: cannot query parameters ({exc})")
            continue
        missing = [n for n in names if n not in have]
        if missing:
            warnings.append(f"{op}: parameter(s) not found in this SNAP version: {', '.join(missing)}")
    return warnings


@dataclass
class SnapResult:
    geotiff: str
    band_name: str
    graph_xml: str
    command: List[str]
    log: str = ""
    params: SnapParams = field(default_factory=SnapParams)


def run_full_chain(input_path: str, out_dir: str, params: SnapParams, *, gpt: Optional[str] = None,
                   progress: ProgressCb = None, cancelled: CancelCb = None,
                   java_heap_gb: Optional[int] = None) -> SnapResult:
    """Run the entire chain in one gpt call and return the resulting GeoTIFF."""
    info = parse_product_name(input_path)
    params.validate(info)
    if params.subset_wkt:
        try:
            check_subset_in_footprint(params.subset_wkt, info.path)
        except SnapError as exc:
            if "does not overlap" in str(exc):
                raise
            # manifest unreadable (unusual layout): let SNAP decide
    check_dem_coverage(params, info.path)
    gpt_exe = find_gpt(gpt)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    base = Path(info.path)
    stem = _strip_product_suffix(base.parent.name if base.name.lower() == "manifest.safe" else base.name)
    out_tif = out_root / f"{stem}_{params.polarisation}_{'dB' if params.output_db else 'sigma0'}.tif"
    graph_path = out_root / f"{stem}_graph.xml"
    xml = build_graph_xml(info.path, str(out_tif), params)
    graph_path.write_text(xml, encoding="utf-8")

    cmd = [gpt_exe, str(graph_path)]
    if java_heap_gb:
        cmd += [f"-J-Xmx{int(java_heap_gb)}G"]
    log = _run(cmd, progress, cancelled, "SNAP Sentinel-1 preprocessing",
               log_path=str(out_root / f"{stem}_gpt.log"))
    # SNAP GeoTIFF writer may append the extension itself; accept either name
    if not out_tif.is_file():
        alt = out_tif.with_suffix(".tif") if out_tif.suffix != ".tif" else out_tif.with_suffix(".tiff")
        if alt.is_file():
            out_tif = alt
        else:
            raise SnapError(f"gpt finished but the output file was not created: {out_tif}\n{log[-500:]}")
    return SnapResult(str(out_tif), result_band_name(params), xml, cmd, log, params)


def read_geotiff(path: str):
    """Load the SNAP GeoTIFF as (array, profile). Imported lazily so the module works without rasterio."""
    import numpy as np
    import rasterio
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        profile = src.profile.copy()
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    data[data == 0.0] = np.nan      # SNAP's no-data fill (exactly 0.0 dB) is often not declared in the GeoTIFF
    return data, profile


def sanity_check_db(data) -> List[str]:
    """Warn if a 'dB' array does not look like calibrated backscatter (catches the DN-in-dB mistake)."""
    import numpy as np
    v = data[np.isfinite(data) & (data != 0.0)]
    if v.size == 0:
        return ["The result contains no valid pixels."]
    p2, p98 = np.percentile(v, [2, 98])
    warn: List[str] = []
    if p2 > -5:
        warn.append(f"2nd percentile is {p2:.1f} dB; calibrated sigma0 in dB is normally well below 0 dB. "
                    "Check that the data are calibrated and in dB.")
    if p98 > 15:
        warn.append(f"98th percentile is {p98:.1f} dB, unusually high for sigma0 (typical range about -30 to +5 dB).")
    return warn


# --------------------------------------------------------------------------- provenance
def snap_version(gpt: str) -> str:
    """Best-effort SNAP version: read VERSION.txt from the installation root (parent of bin/)."""
    root = Path(gpt).resolve().parent.parent
    for name in ("VERSION.txt", "version.txt"):
        f = root / name
        if f.is_file():
            try:
                first = f.read_text(errors="replace").strip().splitlines()[0].strip()
                return first or "unknown"
            except OSError:
                pass
    return "unknown"


def write_provenance(result: "SnapResult", gpt: str, product_path: str) -> str:
    """Write a JSON sidecar next to the GeoTIFF: everything needed to reproduce the run."""
    import dataclasses
    import datetime
    import json
    info = parse_product_name(product_path)
    doc = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "input_product": os.path.basename(product_path),
        "mission": info.mission, "mode": info.mode, "acquisition_start": info.start,
        "output_geotiff": os.path.basename(result.geotiff),
        "output_band": result.band_name,
        "snap_gpt": gpt, "snap_version": snap_version(gpt),
        "parameters": dataclasses.asdict(result.params),
        "graph_xml": result.graph_xml,
        "command": result.command,
    }
    out = Path(result.geotiff).with_suffix(".processing.json")
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return str(out)


# --------------------------------------------------------------------------- footprint
def read_footprint(path: str):
    """Footprint of a Sentinel-1 product from its manifest.safe, WITHOUT extracting the zip.
    Returns a list of (lon, lat) vertices (WGS84)."""
    import zipfile
    p = Path(str(path).replace("\\", "/"))
    text = None
    if p.is_file() and p.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(p) as z:
                member = next((n for n in z.namelist() if n.lower().endswith("manifest.safe")), None)
                if member is None:
                    raise SnapError("manifest.safe not found inside the zip: not a Sentinel-1 product.")
                text = z.read(member).decode("utf-8", errors="replace")
        except zipfile.BadZipFile as exc:
            raise SnapError(f"{p.name} is not a valid zip file (incomplete download?): {exc}") from exc
    else:
        mf = p / "manifest.safe" if p.is_dir() else p
        if mf.is_file():
            text = mf.read_text(encoding="utf-8", errors="replace")
    if text is None:
        raise SnapError(f"Cannot read manifest.safe from {path}")
    m = re.search(r"<gml:coordinates>(.*?)</gml:coordinates>", text, flags=re.S)
    if not m:
        raise SnapError("No footprint (gml:coordinates) found in manifest.safe.")
    pts = []
    for tok in m.group(1).split():
        lat, lon = tok.split(",")[:2]          # SAFE stores "lat,lon"
        pts.append((float(lon), float(lat)))
    if len(pts) < 3:
        raise SnapError("Footprint has fewer than 3 vertices.")
    return pts


def footprint_bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)      # lon_min, lat_min, lon_max, lat_max


def _wkt_bbox(wkt: str):
    nums = re.findall(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", wkt)
    if not nums:
        raise SnapError("Could not read coordinates from the subset WKT.")
    xs, ys = [float(a) for a, _ in nums], [float(b) for _, b in nums]
    return min(xs), min(ys), max(xs), max(ys)


def subset_coverage(subset_wkt: str, product_path: str) -> float:
    """Fraction (0..1) of 9 sample points of the subset box (corners, edge midpoints, centre) that lie
    inside the real footprint polygon. 1.0 = fully inside, 0.0 = no overlap."""
    pts = read_footprint(product_path)
    x0, y0, x1, y1 = _wkt_bbox(subset_wkt)
    xs, ys = (x0, (x0 + x1) / 2, x1), (y0, (y0 + y1) / 2, y1)
    samples = [(x, y) for x in xs for y in ys]
    return sum(footprint_contains(pts, x, y) for x, y in samples) / len(samples)


def check_subset_in_footprint(subset_wkt: str, product_path: str) -> float:
    """Fail early (before a long SNAP run) if the subset does not overlap the product footprint.
    Returns the covered fraction so callers can warn about partial coverage."""
    frac = subset_coverage(subset_wkt, product_path)
    if frac == 0.0:
        fx0, fy0, fx1, fy1 = footprint_bbox(read_footprint(product_path))
        sx0, sy0, sx1, sy1 = _wkt_bbox(subset_wkt)
        raise SnapError(
            f"The subset (lon {sx0:.3f}..{sx1:.3f}, lat {sy0:.3f}..{sy1:.3f}) does not overlap the product "
            f"footprint (bounding box lon {fx0:.3f}..{fx1:.3f}, lat {fy0:.3f}..{fy1:.3f}). "
            "Check the area and that the order is 'lon lat' (longitude first).")
    return frac


def read_pass_direction(path: str) -> Optional[str]:
    """'ASCENDING' / 'DESCENDING' from manifest.safe, or None if it cannot be read."""
    import zipfile
    p = Path(str(path).replace("\\", "/"))
    try:
        if p.is_file() and p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                member = next((n for n in z.namelist() if n.lower().endswith("manifest.safe")), None)
                text = z.read(member).decode("utf-8", errors="replace") if member else ""
        else:
            mf = p / "manifest.safe" if p.is_dir() else p
            text = mf.read_text(encoding="utf-8", errors="replace") if mf.is_file() else ""
    except (OSError, zipfile.BadZipFile):
        return None
    m = re.search(r"<s1:pass>\s*(ASCENDING|DESCENDING)\s*</s1:pass>", text, flags=re.I)
    return m.group(1).upper() if m else None


def footprint_contains(pts, lon: float, lat: float) -> bool:
    """True if (lon, lat) lies inside the footprint polygon (ray casting; lon/lat treated as planar,
    which is accurate enough at the 250 km scale of a Sentinel-1 swath)."""
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x_at = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_at:
                inside = not inside
    return inside


# --------------------------------------------------------------------------- diagnostics
def _kill_tree(proc, grace: float = 10.0) -> None:
    """Stop a subprocess AND its children. On Windows ``gpt`` is a launcher that starts Java (and a .cmd wrapper
    starts yet another process); ``terminate()`` would only stop the launcher and leave the real job running while
    its pipe stays open. ``taskkill /T /F`` ends the whole tree; on POSIX terminate then kill is used."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=grace)
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass



@dataclass
class ProbeResult:
    name: str
    status: str            # "init-ok" | "failed"
    detail: str
    seconds: float


def probe_graph_init(gpt: str, graph_path: str, name: str = "", wait_s: float = 60.0) -> ProbeResult:
    """Start gpt on a graph and decide quickly whether graph INITIALISATION succeeds.

    Initialisation errors (bad parameter values, empty target products, ...) are reported within seconds;
    if gpt is still running after `wait_s` seconds without failing, initialisation succeeded and the
    (long) processing is stopped. Used to find a working configuration cheaply.
    """
    import threading
    import time
    lines: List[str] = []
    try:
        proc = subprocess.Popen([gpt, graph_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
    except OSError as exc:
        return ProbeResult(name, "failed", f"could not start gpt: {exc}", 0.0)

    def reader():
        assert proc.stdout is not None
        for ln in proc.stdout:
            lines.append(ln.rstrip())

    t0 = time.time()
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    status, detail = "init-ok", "still running after %.0f s: graph initialised" % wait_s
    while True:
        rc = proc.poll()
        if rc is not None:
            th.join(timeout=5)
            if rc == 0:
                status, detail = "init-ok", "finished successfully"
            else:
                key = _key_error_lines(lines, 3)
                status, detail = "failed", " | ".join(key) if key else f"exit code {rc}"
            break
        if time.time() - t0 > wait_s:
            break
        time.sleep(0.2)
    _kill_tree(proc)
    th.join(timeout=5)
    if proc.stdout:
        proc.stdout.close()
    return ProbeResult(name, status, detail, time.time() - t0)


PROBE_VARIANTS = (
    ("WGS84(DD) + Subset operator", {"map_projection": "WGS84(DD)", "subset_mode": "operator"}),
    ("AUTO:42001 (UTM) + Subset operator", {"map_projection": "AUTO:42001", "subset_mode": "operator"}),
    ("WGS84(DD) + Read.geometryRegion", {"map_projection": "WGS84(DD)", "subset_mode": "read"}),
    ("EPSG:4326 + Read.geometryRegion", {"map_projection": "EPSG:4326", "subset_mode": "read"}),
    ("WGS84(DD) + crop AFTER terrain correction (slow, full scene)", {"map_projection": "WGS84(DD)", "subset_mode": "after"}),
)


# --------------------------------------------------------------------------- step-by-step execution
def _stepwise_plan(src: str, work: Path, final_tif: Path, params: SnapParams):
    """[(label, operator, {param: value}, out_path, format)] - each step reads the previous output."""
    pol = params.polarisation
    plan = []
    subset = ("Subset", "Subset", {"geoRegion": params.subset_wkt, "copyMetadata": "true"}) if params.subset_wkt else None
    if subset and params.subset_stage == "start":
        plan.append(subset)
    plan.append(("Apply-Orbit-File", "Apply-Orbit-File",
                 {"orbitType": params.orbit_type, "polyDegree": 3, "continueOnFail": "false"}))
    if params.remove_thermal_noise:
        plan.append(("ThermalNoiseRemoval", "ThermalNoiseRemoval",
                     {"selectedPolarisations": pol, "removeThermalNoise": "true"}))
    plan.append(("Calibration", "Calibration",
                 {"selectedPolarisations": pol, "outputSigmaBand": "true", "outputGammaBand": "false",
                  "outputBetaBand": "false", "outputImageInComplex": "false", "outputImageScaleInDb": "false"}))
    if params.speckle_filter != "None":
        plan.append(("Speckle-Filter", "Speckle-Filter",
                     {"filter": params.speckle_filter, "filterSizeX": params.speckle_window,
                      "filterSizeY": params.speckle_window}))
    tc = {"sourceBands": sigma0_band(pol), "demName": params.dem_name,
          "demResamplingMethod": "BILINEAR_INTERPOLATION", "imgResamplingMethod": "BILINEAR_INTERPOLATION",
          "pixelSpacingInMeter": params.pixel_spacing_m, "mapProjection": params.map_projection,
          "nodataValueAtSea": "true" if params.nodata_at_sea else "false"}
    if params.dem_name == "External DEM":
        tc.update(externalDEMFile=params.external_dem, externalDEMNoDataValue=0, externalDEMApplyEGM="true")
    plan.append(("Terrain-Correction", "Terrain-Correction", tc))
    if subset and params.subset_stage == "end":
        plan.append(subset)
    if params.output_db:
        plan.append(("LinearToFromdB", "LinearToFromdB", {}))
    out = []
    for i, (label, op, prm) in enumerate(plan):
        last = i == len(plan) - 1
        path = final_tif if last else work / f"{i + 1:02d}_{label}.dim"
        out.append((label, op, prm, path, "GeoTIFF" if last else "BEAM-DIMAP"))
    return out


def run_stepwise(input_path: str, out_dir: str, params: SnapParams, *, gpt: Optional[str] = None,
                 progress: ProgressCb = None, cancelled: CancelCb = None, java_heap_gb: Optional[int] = None,
                 keep_intermediate: bool = True) -> SnapResult:
    """Same chain as run_full_chain but one gpt call per operator. Slower (intermediate files), but a
    failure names the exact operator, every step has its own log, and finished steps can be inspected in SNAP."""
    info = parse_product_name(input_path)
    params.validate(info)
    if params.subset_wkt:
        check_subset_in_footprint(params.subset_wkt, info.path)
    check_dem_coverage(params, info.path)
    gpt_exe = find_gpt(gpt)
    out_root = Path(out_dir)
    work = out_root / "work"
    work.mkdir(parents=True, exist_ok=True)
    base = Path(info.path.replace("\\", "/"))
    stem = _strip_product_suffix(base.parent.name if base.name.lower() == "manifest.safe" else base.name)
    final = out_root / f"{stem}_{params.polarisation}_{'dB' if params.output_db else 'sigma0'}.tif"
    plan = _stepwise_plan(info.path, work, final, params)

    cmds: List[List[str]] = []
    src = info.path
    n = len(plan)
    for i, (label, op, prm, dst, fmt) in enumerate(plan):
        cmd = [gpt_exe, op, f"-Ssource={src}"] + [f"-P{k}={v}" for k, v in prm.items() if v is not None] \
            + ["-t", str(dst), "-f", fmt]
        if java_heap_gb:
            cmd.append(f"-J-Xmx{int(java_heap_gb)}G")
        cmds.append(cmd)
        step_cb = (lambda v, m, i=i: progress(int((i + v / 100.0) / n * 100), f"[{i + 1}/{n}] {m}")) if progress else None
        try:
            _run(cmd, step_cb, cancelled, f"{label}", log_path=str(work / f"{i + 1:02d}_{label}.log"))
        except Cancelled:
            raise
        except SnapError as exc:
            raise SnapError(f"STEP {i + 1} of {n} FAILED: {label}\n{exc}\n"
                            f"Input of this step: {src}\nCommand: {' '.join(cmd)}") from exc
        if not Path(dst).exists():
            raise SnapError(f"Step {i + 1} ({label}) finished but did not create {dst}")
        src = str(dst)

    if not keep_intermediate:
        shutil.rmtree(work, ignore_errors=True)
    text = "\n".join(" ".join(c) for c in cmds)
    return SnapResult(str(final), result_band_name(params), "stepwise:\n" + text, cmds[-1], "", params)


def box_wkt(lon: float, lat: float, size_km: float) -> str:
    """Square study-area box (WGS84 WKT, 'lon lat' order) of about size_km x size_km around a centre."""
    import math
    if size_km <= 0:
        raise SnapError("The study-area size must be positive.")
    h = size_km / 2 / 111.0
    w = h / max(0.1, math.cos(math.radians(lat)))
    x0, x1, y0, y1 = lon - w, lon + w, lat - h, lat + h
    return f"POLYGON(({x0:.4f} {y0:.4f}, {x1:.4f} {y0:.4f}, {x1:.4f} {y1:.4f}, {x0:.4f} {y1:.4f}, {x0:.4f} {y0:.4f}))"


SRTM_DEMS = ("SRTM 3Sec", "SRTM 1Sec HGT")
SRTM_LAT_RANGE = (-56.0, 60.0)       # SRTM coverage; check against your SNAP version's documentation


def check_dem_coverage(params: "SnapParams", product_path: str) -> None:
    """Stop early if an SRTM DEM is selected for an area outside SRTM's latitude coverage (high latitudes)."""
    if params.dem_name not in SRTM_DEMS:
        return
    if params.subset_wkt:
        _, y0, _, y1 = _wkt_bbox(params.subset_wkt)
    else:
        try:
            _, y0, _, y1 = footprint_bbox(read_footprint(product_path))
        except (SnapError, OSError, ValueError):
            return          # footprint unreadable: let SNAP decide
    if y1 > SRTM_LAT_RANGE[1] or y0 < SRTM_LAT_RANGE[0]:
        raise SnapError(
            f"{params.dem_name} does not cover latitudes above {SRTM_LAT_RANGE[1]:.0f} N or below {abs(SRTM_LAT_RANGE[0]):.0f} S "
            f"(your area spans {y0:.1f} to {y1:.1f}). Choose 'Copernicus 30m Global DEM' or an external DEM.")


# --------------------------------------------------------------------------- flexible study-area selection
SUBSET_MODES = ("full", "bbox", "center", "polygon")


@dataclass
class SubsetSpec:
    """How the user selects the area to process. Independent of any country or scene.

    full    : whole scene (the geographic bounds are detected automatically from the raw product)
    bbox    : (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees
    center  : (lon, lat) and radius_km (half the side of the square box)
    polygon : WKT POLYGON in WGS84, 'lon lat' order
    """
    mode: str = "full"
    bbox: Optional[tuple] = None
    center: Optional[tuple] = None
    radius_km: Optional[float] = None
    wkt: Optional[str] = None


def bbox_wkt(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> str:
    return (f"POLYGON(({lon_min:.4f} {lat_min:.4f}, {lon_max:.4f} {lat_min:.4f}, {lon_max:.4f} {lat_max:.4f}, "
            f"{lon_min:.4f} {lat_max:.4f}, {lon_min:.4f} {lat_min:.4f}))")


def _check_lonlat(lon: float, lat: float, what: str) -> None:
    if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
        raise SnapError(f"{what}: longitude must be within -180..180 and latitude within -90..90 "
                        f"(got lon {lon}, lat {lat}). Order is longitude first.")


def resolve_subset(spec: SubsetSpec, product_path: Optional[str] = None) -> Optional[str]:
    """Validate a study-area selection and return the WKT for SNAP, or None for 'process the whole scene'.

    If a box contains the whole scene, None is returned too: no cut is needed (and SNAP is spared a pointless subset).
    """
    if spec.mode not in SUBSET_MODES:
        raise SnapError(f"Unknown subset mode '{spec.mode}'.")
    if spec.mode == "full":
        return None
    if spec.mode == "bbox":
        if not spec.bbox or len(spec.bbox) != 4:
            raise SnapError("Bounding box needs lon_min, lat_min, lon_max, lat_max.")
        x0, y0, x1, y1 = (float(v) for v in spec.bbox)
        _check_lonlat(x0, y0, "Bounding box corner 1"); _check_lonlat(x1, y1, "Bounding box corner 2")
        if x0 >= x1 or y0 >= y1:
            raise SnapError("Bounding box must have lon_min < lon_max and lat_min < lat_max.")
        if x1 - x0 > 180:
            raise SnapError("Boxes crossing the antimeridian are not supported; split the area in two.")
        wkt = bbox_wkt(x0, y0, x1, y1)
    elif spec.mode == "center":
        if not spec.center or spec.radius_km is None:
            raise SnapError("Centre mode needs a centre (lon, lat) and a radius in km.")
        lon, lat = float(spec.center[0]), float(spec.center[1])
        _check_lonlat(lon, lat, "Centre")
        if spec.radius_km <= 0:
            raise SnapError("The radius must be positive.")
        wkt = box_wkt(lon, lat, 2.0 * float(spec.radius_km))
    else:
        if not spec.wkt or not spec.wkt.strip().upper().startswith("POLYGON"):
            raise SnapError("Polygon mode needs a WKT POLYGON in WGS84 ('lon lat' order).")
        nums = re.findall(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", spec.wkt)
        if len(nums) < 4:
            raise SnapError("A polygon needs at least 3 distinct vertices (4 coordinate pairs, the last equal to the first).")
        for a, b in nums:
            _check_lonlat(float(a), float(b), "Polygon vertex")
        wkt = spec.wkt.strip()
    if product_path:
        try:
            fx0, fy0, fx1, fy1 = footprint_bbox(read_footprint(product_path))
            sx0, sy0, sx1, sy1 = _wkt_bbox(wkt)
            if sx0 <= fx0 and sy0 <= fy0 and sx1 >= fx1 and sy1 >= fy1:
                return None
        except (SnapError, OSError, ValueError):
            pass
    return wkt


def estimate_output_pixels(wkt: Optional[str], product_path: str, spacing_m: float) -> int:
    """Rough number of output pixels (the area is the box clipped to the scene's bounding box)."""
    import math
    fx0, fy0, fx1, fy1 = footprint_bbox(read_footprint(product_path))
    if wkt:
        sx0, sy0, sx1, sy1 = _wkt_bbox(wkt)
        x0, x1, y0, y1 = max(sx0, fx0), min(sx1, fx1), max(sy0, fy0), min(sy1, fy1)
    else:
        x0, x1, y0, y1 = fx0, fx1, fy0, fy1
    if x1 <= x0 or y1 <= y0:
        return 0
    lat = (y0 + y1) / 2
    w_m = (x1 - x0) * 111320.0 * math.cos(math.radians(lat))
    h_m = (y1 - y0) * 110574.0
    return int(w_m * h_m / (spacing_m * spacing_m))


def size_advice(pixels: int) -> str:
    """Plain-language hint about the processing load."""
    mp = pixels / 1e6
    if pixels == 0:
        return "The selected area does not overlap the scene."
    if mp < 60:
        return f"about {mp:,.0f} megapixels: light"
    if mp < 200:
        return f"about {mp:,.0f} megapixels: heavy, needs plenty of RAM (16 GB or more is advisable)"
    return f"about {mp:,.0f} megapixels: VERY large; choose a smaller area or a coarser pixel size"
