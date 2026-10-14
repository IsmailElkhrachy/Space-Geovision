"""Stack single-band Sentinel-2 GeoTIFFs (as downloaded from Copernicus Browser / EO Browser, one file per
band, e.g. ``2020-02-10-00_00_2020-02-10-23_59_Sentinel-2_L2A_B02_(Raw).tiff``) into one multi-band GeoTIFF.

Why: the change-detection tools compare two images band by band, so each date must be ONE file with the same
band order.  Band order of the stack is the order of ``bands`` (default B02,B03,B04,B08,B11).

Pure python + rasterio (imported lazily so the module imports without it).  Also usable from a terminal::

    python bandstack.py FOLDER                      # writes FOLDER/<date>_S2L2A_stack.tif
    python bandstack.py --check before.tif after.tif   # are the two grids identical?
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

DEFAULT_BANDS = ("B02", "B03", "B04", "B08", "B11")
_EXT = (".tif", ".tiff")


def _band_regex(band: str):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(band) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def find_band_files(folder: str, bands: Sequence[str] = DEFAULT_BANDS) -> Dict[str, str]:
    """Return {band: path}.  Raises ValueError if a band is missing or ambiguous."""
    if not os.path.isdir(folder):
        raise ValueError(f"Folder not found: {folder}")
    names = sorted(n for n in os.listdir(folder)
                   if n.lower().endswith(_EXT) and "stack" not in n.lower())
    found: Dict[str, str] = {}
    for b in bands:
        rx = _band_regex(b)
        hits = [n for n in names if rx.search(os.path.splitext(n)[0])]
        if not hits:
            raise ValueError(f"No file for band {b} in {folder}")
        if len(hits) > 1:
            raise ValueError(f"More than one file for band {b}: {hits}")
        found[b] = os.path.join(folder, hits[0])
    return found


def date_from_name(path: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return m.group(1) if m else "scene"


def grid_signature(src) -> dict:
    t = src.transform
    return dict(width=src.width, height=src.height, crs=str(src.crs),
                transform=tuple(round(float(v), 6) for v in tuple(t)[:6]))


def same_grid(a: dict, b: dict) -> Optional[str]:
    """None when identical, else a human-readable reason."""
    for k in ("width", "height"):
        if a[k] != b[k]:
            return f"{k} differs: {a[k]} vs {b[k]}"
    if a["crs"] != b["crs"]:
        return f"CRS differs: {a['crs']} vs {b['crs']}"
    for x, y in zip(a["transform"], b["transform"]):
        if abs(x - y) > 1e-3:
            return f"geotransform differs: {a['transform']} vs {b['transform']}"
    return None


def stack_bands(folder: str, bands: Sequence[str] = DEFAULT_BANDS, out_path: Optional[str] = None) -> dict:
    """Write the multi-band stack and return a report dict (path, shape, crs, resolution, dtype, per-band stats)."""
    import numpy as np
    import rasterio

    files = find_band_files(folder, bands)
    arrays: List = []
    sig0 = None
    dtypes = []
    profile = None
    for b in bands:
        with rasterio.open(files[b]) as src:
            if src.count != 1:
                raise ValueError(f"{os.path.basename(files[b])} has {src.count} bands, expected 1")
            sig = grid_signature(src)
            if sig0 is None:
                sig0, profile = sig, src.profile.copy()
            else:
                why = same_grid(sig0, sig)
                if why:
                    raise ValueError(f"Band {b} is not on the same grid as {bands[0]}: {why}. "
                                     "Download all bands with the same area/resolution.")
            arr = src.read(1)
            dtypes.append(arr.dtype)
            arrays.append(arr)
    dtype = np.result_type(*dtypes)
    out_path = out_path or os.path.join(folder, f"{date_from_name(files[bands[0]])}_S2L2A_stack.tif")
    profile.update(driver="GTiff", count=len(bands), dtype=str(dtype), compress="deflate", nodata=None)
    for k in ("blockxsize", "blockysize", "tiled", "photometric"):
        profile.pop(k, None)
    tmp = out_path + ".part"
    with rasterio.open(tmp, "w", **profile) as dst:
        for i, (b, arr) in enumerate(zip(bands, arrays), start=1):
            dst.write(arr.astype(dtype, copy=False), i)
            try:
                dst.set_band_description(i, b)
            except Exception:
                pass
    os.replace(tmp, out_path)

    stats = {}
    for b, arr in zip(bands, arrays):
        a = arr.astype("float64")
        stats[b] = dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
                        zero_frac=float((a == 0).mean()))
    return dict(path=out_path, bands=list(bands), width=sig0["width"], height=sig0["height"],
                crs=sig0["crs"], transform=sig0["transform"], dtype=str(dtype), stats=stats,
                sources={b: files[b] for b in bands})


def format_report(rep: dict) -> str:
    t = rep["transform"]
    lines = [f"Saved: {rep['path']}",
             f"Size: {rep['height']} rows x {rep['width']} cols, {len(rep['bands'])} bands, {rep['dtype']}",
             f"CRS: {rep['crs']}   pixel size: {abs(t[0]):.2f} x {abs(t[4]):.2f}",
             "Band order: " + ", ".join(f"{i}={b}" for i, b in enumerate(rep["bands"], 1)), ""]
    for b in rep["bands"]:
        s = rep["stats"][b]
        lines.append(f"  {b}: min {s['min']:.0f}  max {s['max']:.0f}  mean {s['mean']:.1f}  zeros {s['zero_frac']*100:.1f}%")
    mx = max(s["max"] for s in rep["stats"].values())
    if mx > 12000:
        lines += ["", "Note: values reach %.0f, so reflectance was scaled to the full 16-bit range "
                      "(not 0-10000). That is fine as long as BOTH dates were downloaded the same way." % mx]
    if any(s["zero_frac"] > 0.2 for s in rep["stats"].values()):
        lines += ["", "Warning: more than 20% zeros in a band (no-data or cloud mask). Check the scene."]
    return "\n".join(lines)


def check_pair(path_a: str, path_b: str) -> str:
    import rasterio
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        sa, sb = grid_signature(a), grid_signature(b)
        ca, cb = a.count, b.count
    why = same_grid(sa, sb)
    if why:
        return f"NOT identical: {why}"
    if ca != cb:
        return f"NOT identical: band count differs ({ca} vs {cb})"
    return f"OK: identical grid ({sa['height']} x {sa['width']}, {sa['crs']}, {ca} bands)."


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--bands", default=",".join(DEFAULT_BANDS))
    ap.add_argument("--out")
    ap.add_argument("--check", nargs=2, metavar=("A", "B"))
    a = ap.parse_args(argv)
    if a.check:
        print(check_pair(*a.check))
        return 0
    if not a.folder:
        ap.error("folder required")
    print(format_report(stack_bands(a.folder, [b.strip().upper() for b in a.bands.split(",")], a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
