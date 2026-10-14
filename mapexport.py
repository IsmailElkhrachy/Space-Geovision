"""Print-quality map export (PDF / PNG / JPEG / TIFF / SVG) without Qt: matplotlib + numpy only, unit-testable.

The map is re-rendered at the requested resolution instead of grabbing the screen: the raster is cropped to the chosen
extent, decimated to the pixel size the page needs at ``dpi`` (so ``dpi`` really controls image resolution and file size,
also in PDF), optionally masked to a polygon, and drawn with title, coordinate grid, scale bar, north arrow, colour bar
and footer.

Conventions: rasters are (rows, cols) or (bands, rows, cols), north-up, NaN = no data; ``transform`` is an affine-like
object with attributes a, b, c, d, e, f; vector geometries are given as lists of (N, 2) coordinate arrays in the same CRS
as the raster.
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

import georef

PAGE_SIZES_MM = {"A5": (148.0, 210.0), "A4": (210.0, 297.0), "A3": (297.0, 420.0), "A2": (420.0, 594.0),
                 "Letter": (215.9, 279.4), "Legal": (215.9, 355.6)}
FORMATS = ("pdf", "png", "jpg", "tif", "svg")
MAX_PIXELS = 150_000_000   # refuse absurd raster outputs (RGBA = 4 bytes/pixel); e.g. A2 at 600 dpi is ~ 200 Mpx


@dataclass
class RasterLayer:
    data: np.ndarray
    transform: object
    geographic: bool = False
    crs_text: str = ""
    cmap: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    nodata: Optional[float] = None
    categorical: bool = False        # nearest-neighbour when decimating (masks, classes)
    name: str = ""
    unit_label: str = ""             # colour-bar label


@dataclass
class VectorLayer:
    name: str
    lines: List[np.ndarray] = field(default_factory=list)      # open polylines
    rings: List[np.ndarray] = field(default_factory=list)      # polygon boundaries (outer rings and holes)
    points: List[Tuple[float, float]] = field(default_factory=list)
    color: str = "#d62728"
    width: float = 1.2


@dataclass
class ExportSpec:
    page: str = "A4"                              # key of PAGE_SIZES_MM or "custom"
    custom_mm: Tuple[float, float] = (210.0, 297.0)   # (width, height) portrait
    orientation: str = "auto"                     # auto | portrait | landscape
    margin_mm: float = 12.0
    dpi: int = 300
    fmt: str = "pdf"
    title: str = ""
    subtitle: str = ""
    note: str = ""
    author: str = ""
    extent: Optional[Tuple[float, float, float, float]] = None   # xmin, ymin, xmax, ymax in map units; None = full layer
    clip_rings: Optional[List[np.ndarray]] = None                # polygon(s): mask pixels outside
    mask_outside: bool = True
    outline_clip: bool = True
    show_grid: bool = True
    show_scalebar: bool = True
    show_north: bool = True
    show_colorbar: bool = True
    show_border: bool = True
    show_footer: bool = True
    stretch: str = "layer"                        # layer | percentile | minmax
    interpolation: str = "nearest"                # nearest | bilinear
    overlay_alpha: float = 0.6
    background: str = "white"


# ----------------------------------------------------------------------------------------------------- geometry
def page_inches(spec: ExportSpec, map_aspect: float) -> Tuple[float, float]:
    """(width, height) of the page in inches for the page size/orientation. ``map_aspect`` = displayed width / height."""
    if spec.page == "custom":
        w, h = spec.custom_mm
    else:
        if spec.page not in PAGE_SIZES_MM:
            raise ValueError(f"Unknown page size {spec.page!r}")
        w, h = PAGE_SIZES_MM[spec.page]
    if w <= 0 or h <= 0:
        raise ValueError("Page size must be positive")
    short, long_ = min(w, h), max(w, h)
    o = spec.orientation
    if o == "auto":
        o = "landscape" if map_aspect > 1.0 else "portrait"
    if o not in ("portrait", "landscape"):
        raise ValueError("orientation must be auto, portrait or landscape")
    w, h = (long_, short) if o == "landscape" else (short, long_)
    return w / 25.4, h / 25.4


def crop_window(transform, shape: Tuple[int, int], extent: Optional[Tuple[float, float, float, float]]):
    """(row0, row1, col0, col1) of the pixels overlapping ``extent`` (clamped to the raster). Raises if no overlap."""
    rows, cols = shape
    if extent is None:
        return 0, rows, 0, cols
    if abs(getattr(transform, "b", 0.0)) > 1e-12 or abs(getattr(transform, "d", 0.0)) > 1e-12:
        raise ValueError("Rotated rasters cannot be cropped by a map extent")
    a, c, e, f = float(transform.a), float(transform.c), float(transform.e), float(transform.f)
    x0, y0, x1, y1 = extent
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid extent (max must exceed min)")
    c0, c1 = sorted(((x0 - c) / a, (x1 - c) / a))
    r0, r1 = sorted(((y1 - f) / e, (y0 - f) / e))
    c0, c1 = int(math.floor(max(c0, 0))), int(math.ceil(min(c1, cols)))
    r0, r1 = int(math.floor(max(r0, 0))), int(math.ceil(min(r1, rows)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("The extent does not overlap the raster")
    return r0, r1, c0, c1


def decimate_to(data: np.ndarray, target_rows: int, target_cols: int, categorical: bool = False) -> Tuple[np.ndarray, int]:
    """Reduce a (rows, cols) / (bands, rows, cols) array so it is not larger than needed for the target pixel size.
    Continuous data is block-averaged (NaN-aware); categorical data uses nearest-neighbour. Returns (array, factor)."""
    rows, cols = data.shape[-2:]
    k = int(max(1, min(rows // max(target_rows, 1), cols // max(target_cols, 1))))
    if k <= 1:
        return data, 1
    if categorical or not np.issubdtype(data.dtype, np.floating):
        if categorical or np.issubdtype(data.dtype, np.integer):
            return data[..., ::k, ::k], k
    r2, c2 = (rows // k) * k, (cols // k) * k
    d = data[..., :r2, :c2].astype(np.float32)
    shp = d.shape[:-2] + (r2 // k, k, c2 // k, k)
    d = d.reshape(shp)
    valid = np.isfinite(d)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(valid, d, 0.0).sum(axis=(-1, -3))
        n = valid.sum(axis=(-1, -3))
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out.astype(np.float32), k


def polygon_mask(rings: Sequence[np.ndarray], transform, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean (rows, cols) mask of pixel centres inside the polygon(s) (even-odd rule across all rings, so holes work)."""
    from matplotlib.path import Path
    rows, cols = shape
    a, b, c, d, e, f = (float(getattr(transform, n)) for n in "abcdef")
    cc, rr = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
    x, y = a * cc + b * rr + c, d * cc + e * rr + f
    pts = np.column_stack([x.ravel(), y.ravel()])
    inside = np.zeros(len(pts), bool)
    for ring in rings:
        ring = np.asarray(ring, dtype=float)
        if len(ring) < 3:
            continue
        inside ^= Path(ring).contains_points(pts)
    return inside.reshape(rows, cols)


def sample_to_grid(src: np.ndarray, src_transform, dst_shape: Tuple[int, int], dst_transform) -> np.ndarray:
    """Nearest-neighbour resampling of ``src`` (2-D) onto the grid (dst_shape, dst_transform); NaN outside the source.
    Both transforms must be north-up and in the same CRS."""
    rows, cols = dst_shape
    cc, rr = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
    x = dst_transform.a * cc + dst_transform.c
    y = dst_transform.e * rr + dst_transform.f
    sc = np.floor((x - src_transform.c) / src_transform.a).astype(int)
    sr = np.floor((y - src_transform.f) / src_transform.e).astype(int)
    ok = (sc >= 0) & (sc < src.shape[-1]) & (sr >= 0) & (sr < src.shape[-2])
    out = np.full(dst_shape, np.nan, np.float32)
    out[ok] = src[sr[ok], sc[ok]]
    return out


def geometry_to_arrays(geom, lines=None, rings=None, points=None):
    """Split a shapely-like geometry (duck-typed: geom_type, exterior/interiors, geoms, coords) into polylines,
    polygon rings and points. Returns (lines, rings, points); appends to the given lists if provided."""
    lines = [] if lines is None else lines
    rings = [] if rings is None else rings
    points = [] if points is None else points
    if geom is None or getattr(geom, "is_empty", False):
        return lines, rings, points
    t = getattr(geom, "geom_type", "")
    if t == "Polygon":
        rings.append(np.asarray(geom.exterior.coords, float)[:, :2])
        for r in geom.interiors:
            rings.append(np.asarray(r.coords, float)[:, :2])
    elif t in ("LineString", "LinearRing"):
        lines.append(np.asarray(geom.coords, float)[:, :2])
    elif t == "Point":
        points.append((float(geom.x), float(geom.y)))
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            geometry_to_arrays(g, lines, rings, points)
    return lines, rings, points


def nice_length(target: float) -> float:
    """1, 2 or 5 x 10^k not larger than ``target`` (> 0)."""
    if target <= 0:
        return 0.0
    p = 10 ** math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * p <= target:
            return m * p
    return p


def metres_per_unit(geographic: bool, lat_mid: float) -> float:
    """Metres per map unit along x: degrees at ``lat_mid`` for geographic CRS, else 1 (projected CRS in metres)."""
    if geographic:
        return 111320.0 * max(math.cos(math.radians(max(-85.0, min(85.0, lat_mid)))), 0.01)
    return 1.0


# ----------------------------------------------------------------------------------------------------- rendering
def _stretch(layer: RasterLayer, band: np.ndarray, mode: str):
    v = band[np.isfinite(band)]
    if mode == "layer" and layer.vmin is not None and layer.vmax is not None:
        lo, hi = layer.vmin, layer.vmax
    elif v.size == 0:
        lo, hi = 0.0, 1.0
    elif mode == "minmax":
        lo, hi = float(v.min()), float(v.max())
    else:
        lo, hi = np.percentile(v, [2, 98])
    if not hi > lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _prepare(layer: RasterLayer, spec: ExportSpec, frame_px: Tuple[int, int]):
    """Crop, mask and decimate. Returns (array (rows, cols) or (3, rows, cols) float32 with NaN no-data, extent
    (left, right, bottom, top), transform of the returned array)."""
    data = np.asarray(layer.data)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim not in (2, 3):
        raise ValueError(f"Cannot export an array of shape {data.shape}")
    rows, cols = data.shape[-2:]
    r0, r1, c0, c1 = crop_window(layer.transform, (rows, cols), spec.extent)
    data = data[..., r0:r1, c0:c1]
    tr = layer.transform
    sub = _Affine(tr.a, tr.b, tr.c + tr.a * c0 + tr.b * r0, tr.d, tr.e, tr.f + tr.d * c0 + tr.e * r0)
    nearest = layer.categorical or not np.issubdtype(data.dtype, np.floating)
    data = data.astype(np.float32)
    if layer.nodata is not None and np.isfinite(layer.nodata):
        data = np.where(data == np.float32(layer.nodata), np.nan, data)
    arr, k = decimate_to(data, frame_px[1], frame_px[0], nearest)
    sub = _Affine(sub.a * k, sub.b, sub.c, sub.d, sub.e * k, sub.f)
    if spec.clip_rings and spec.mask_outside:
        m = polygon_mask(spec.clip_rings, sub, arr.shape[-2:])
        arr = np.where(m, arr, np.nan) if arr.ndim == 2 else np.where(m[None], arr, np.nan)
    h, w = arr.shape[-2:]
    left, top = sub.c, sub.f
    extent = (left, left + sub.a * w, top + sub.e * h, top)
    return arr, extent, sub


@dataclass
class _Affine:
    a: float; b: float; c: float; d: float; e: float; f: float


def _to_rgba(arr: np.ndarray, layer: RasterLayer, spec: ExportSpec):
    """(image to pass to imshow, norm info). 3-band -> RGB float array in 0-1; single band -> masked array."""
    import matplotlib
    if arr.ndim == 3:
        if arr.shape[0] < 3:
            raise ValueError("Need 1 or at least 3 bands")
        rgb = np.zeros(arr.shape[1:] + (3,), np.float32)
        bad = ~np.isfinite(arr[:3]).all(axis=0)
        for i in range(3):
            lo, hi = _stretch(layer, arr[i], spec.stretch)
            rgb[..., i] = np.clip((np.nan_to_num(arr[i], nan=lo) - lo) / (hi - lo), 0, 1)
        alpha = np.where(bad, 0.0, 1.0).astype(np.float32)
        return np.dstack([rgb, alpha]), None
    lo, hi = _stretch(layer, arr, spec.stretch)
    return np.ma.masked_invalid(arr), (lo, hi)


def rings_bbox(rings: Sequence[np.ndarray]) -> Tuple[float, float, float, float]:
    pts = np.vstack([np.asarray(r, float) for r in rings if len(r)])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def compose_figure(base: RasterLayer, spec: ExportSpec, overlay: Optional[RasterLayer] = None,
                   vectors: Sequence[VectorLayer] = ()):
    """Build the matplotlib Figure for the page. Nothing is written; call :func:`save_figure`.
    With a clip polygon and no explicit extent the map is cropped to the polygon's bounding box."""
    import dataclasses
    if spec.clip_rings and spec.extent is None:
        bx0, by0, bx1, by1 = rings_bbox(spec.clip_rings)
        padx, pady = (bx1 - bx0) * 0.02, (by1 - by0) * 0.02
        spec = dataclasses.replace(spec, extent=(bx0 - padx, by0 - pady, bx1 + padx, by1 + pady))
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    if spec.dpi < 36 or spec.dpi > 1200:
        raise ValueError("DPI must be between 36 and 1200")
    if spec.fmt not in FORMATS:
        raise ValueError(f"Format must be one of {FORMATS}")

    # ---- extent / aspect (needs the window before the page size when orientation is 'auto')
    rows, cols = np.asarray(base.data).shape[-2:]
    r0, r1, c0, c1 = crop_window(base.transform, (rows, cols), spec.extent)
    wx, hy = abs(base.transform.a) * (c1 - c0), abs(base.transform.e) * (r1 - r0)
    top_y = base.transform.f + base.transform.e * r0
    lat_mid = top_y + base.transform.e * (r1 - r0) / 2.0
    aspect_y = 1.0 / max(math.cos(math.radians(max(-85.0, min(85.0, lat_mid)))), 0.05) if base.geographic else 1.0
    map_aspect = wx / (hy * aspect_y)
    pw, ph = page_inches(spec, map_aspect)

    m = spec.margin_mm / 25.4
    if m * 2 >= min(pw, ph) * 0.8:
        raise ValueError("Margins are too large for this page")
    fs = max(6.0, 8.0 * min(pw, ph) / 8.27)          # base font size scales with the page (A4 short side = 8.27 in)
    has_title = bool(spec.title or spec.subtitle)
    title_h = (0.9 if spec.subtitle else 0.6) * fs / 72.0 * 3.0 if has_title else 0.0
    foot_h = 2.6 * fs / 72.0 if (spec.show_footer or spec.note) else 0.0
    axis_pad = 4.0 * fs / 72.0 if spec.show_grid else 0.4 * fs / 72.0   # tick labels + axis labels
    _d = np.asarray(base.data)
    single_band = _d.ndim == 2 or (_d.ndim == 3 and _d.shape[0] == 1)
    cb_w = 0.9 if (spec.show_colorbar and single_band) else 0.0
    cb_w = cb_w * (fs / 8.0) if cb_w else 0.0
    avail_l, avail_b = m + axis_pad * 0.95, m + foot_h + axis_pad * 0.55
    avail_w = pw - 2 * m - axis_pad * 0.95 - cb_w
    avail_h = ph - 2 * m - title_h - foot_h - axis_pad * 0.55 - 0.1
    if avail_w <= 0.5 or avail_h <= 0.5:
        raise ValueError("The page is too small for the chosen margins and elements")
    if map_aspect >= avail_w / avail_h:
        fw, fh = avail_w, avail_w / map_aspect
    else:
        fw, fh = avail_h * map_aspect, avail_h
    fx = avail_l + (avail_w - fw) / 2.0
    fy = avail_b + (avail_h - fh) / 2.0

    px_w, px_h = int(fw * spec.dpi), int(fh * spec.dpi)
    page_px = int(pw * spec.dpi) * int(ph * spec.dpi)
    if spec.fmt in ("png", "jpg", "tif") and page_px > MAX_PIXELS:
        raise ValueError(f"The output would be {int(pw * spec.dpi)}x{int(ph * spec.dpi)} pixels "
                         f"({page_px / 1e6:.0f} Mpx); lower the DPI or the page size")

    fig = Figure(figsize=(pw, ph), dpi=spec.dpi, facecolor=spec.background)
    ax = fig.add_axes([fx / pw, fy / ph, fw / pw, fh / ph])
    ax.set_facecolor("white")

    arr, extent, sub = _prepare(base, spec, (px_h, px_w))
    img, rng = _to_rgba(arr, base, spec)
    interp = "nearest" if spec.interpolation == "nearest" else "bilinear"
    kw = dict(extent=extent, interpolation=interp, origin="upper", aspect="auto")
    if rng is not None:
        cm = matplotlib.colormaps[base.cmap] if base.cmap in matplotlib.colormaps else matplotlib.colormaps["viridis"]
        im = ax.imshow(img, cmap=cm, vmin=rng[0], vmax=rng[1], **kw)
    else:
        im = ax.imshow(img, **kw)
    if overlay is not None:
        ov = np.asarray(overlay.data)
        ov = ov[0] if ov.ndim == 3 else ov
        ov_on_grid = sample_to_grid(ov.astype(np.float32), overlay.transform, arr.shape[-2:], sub) if arr.ndim == 2 else None
        if ov_on_grid is not None:
            if overlay.nodata is not None and np.isfinite(overlay.nodata):
                ov_on_grid = np.where(ov_on_grid == np.float32(overlay.nodata), np.nan, ov_on_grid)
            lo, hi = _stretch(overlay, ov_on_grid, spec.stretch)
            cm2 = matplotlib.colormaps[overlay.cmap] if overlay.cmap in matplotlib.colormaps else matplotlib.colormaps["viridis"]
            ax.imshow(np.ma.masked_invalid(ov_on_grid), cmap=cm2, vmin=lo, vmax=hi, alpha=spec.overlay_alpha, **kw)

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    for v in vectors:
        for ln in v.lines:
            ax.plot(ln[:, 0], ln[:, 1], color=v.color, lw=v.width)
        for rg in v.rings:
            ax.plot(np.append(rg[:, 0], rg[0, 0]), np.append(rg[:, 1], rg[0, 1]), color=v.color, lw=v.width)
        if v.points:
            xs, ys = zip(*v.points)
            ax.scatter(xs, ys, s=12, color=v.color, zorder=5)
    if spec.clip_rings and spec.outline_clip:
        for rg in spec.clip_rings:
            ax.plot(np.append(rg[:, 0], rg[0, 0]), np.append(rg[:, 1], rg[0, 1]), color="black", lw=1.0)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    # ---- axes: coordinates in degrees or map units
    if spec.show_grid:
        if base.geographic:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _p: georef.format_degrees(val, False)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _p: georef.format_degrees(val, True)))
            ax.set_xlabel("Longitude", fontsize=fs); ax.set_ylabel("Latitude", fontsize=fs)
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(georef.format_projected))
            ax.yaxis.set_major_formatter(FuncFormatter(georef.format_projected))
            ax.set_xlabel("Easting (m)", fontsize=fs); ax.set_ylabel("Northing (m)", fontsize=fs)
        ax.tick_params(labelsize=fs * 0.85, length=3)
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(90); lbl.set_va("center")
        ax.grid(True, color="white", ls=":", lw=0.6, alpha=0.6)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(bool(spec.show_border)); sp.set_linewidth(0.9)

    # ---- title block
    if has_title:
        ytop = ph - m
        if spec.title:
            fig.text(0.5, (ytop - 0.4 * fs * 1.5 / 72.0) / ph, spec.title, ha="center", va="center", fontsize=fs * 1.9, weight="bold")
        if spec.subtitle:
            fig.text(0.5, (ytop - 1.4 * fs * 1.6 / 72.0 - 0.1) / ph, spec.subtitle, ha="center", va="center", fontsize=fs * 1.1, color="#444")

    # ---- colour bar
    if cb_w > 0 and rng is not None:
        cax = fig.add_axes([(fx + fw + 0.18 * cb_w) / pw, fy / ph, (0.28 * cb_w) / pw, fh / ph])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=fs * 0.85)
        label = base.unit_label or base.name
        if label:
            cb.set_label(label, fontsize=fs * 0.95)

    # ---- scale bar (axes coordinates; length from the frame width)
    if spec.show_scalebar:
        width_m = (extent[1] - extent[0]) * metres_per_unit(base.geographic, (extent[2] + extent[3]) / 2.0)
        if width_m > 0:
            length = nice_length(width_m * 0.22)
            if length > 0:
                frac = length / width_m
                x0, y0, hbar = 0.04, 0.035, 0.012
                unit, div = ("km", 1000.0) if length >= 1000 else ("m", 1.0)
                half = frac / 2.0
                for i, (xa, col) in enumerate(((x0, "black"), (x0 + half, "white"))):
                    ax.add_patch(Rectangle((xa, y0), half, hbar, transform=ax.transAxes, facecolor=col, edgecolor="black", lw=0.7, zorder=10))
                for xa, txt in ((x0, "0"), (x0 + half, f"{length / 2 / div:g}"), (x0 + frac, f"{length / div:g} {unit}")):
                    ax.text(xa, y0 + hbar + 0.008, txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=fs * 0.85, zorder=10,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # ---- north arrow (rasters are north-up)
    if spec.show_north:
        ax.annotate("", xy=(0.95, 0.93), xytext=(0.95, 0.83), xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14), zorder=10)
        ax.text(0.95, 0.945, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=fs * 1.3, weight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    # ---- footer
    foot = []
    if spec.show_footer:
        bits = [t for t in (base.crs_text, f"Layer: {base.name}" if base.name else "", f"Exported {datetime.date.today().isoformat()}",
                            "Space GeoVision") if t]
        foot.append("  |  ".join(bits))
    if spec.note:
        foot.insert(0, spec.note)
    for i, line in enumerate(foot):
        fig.text(m / pw, (m + (len(foot) - 1 - i) * 1.25 * fs / 72.0 + 0.2 * fs / 72.0) / ph, line, ha="left", va="bottom",
                 fontsize=fs * 0.8, color="#333")
    return fig


def save_figure(fig, path: str, spec: ExportSpec) -> None:
    kw = {"dpi": spec.dpi, "facecolor": fig.get_facecolor()}
    if spec.fmt == "pdf":
        meta = {"Title": spec.title or "Map", "Creator": "Space GeoVision"}
        if spec.author:
            meta["Author"] = spec.author
        kw["metadata"] = meta
    elif spec.fmt == "jpg":
        kw["pil_kwargs"] = {"quality": 92}
    elif spec.fmt == "tif":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    elif spec.fmt == "png":
        kw["metadata"] = {"Software": "Space GeoVision"}
    fig.savefig(path, format={"jpg": "jpeg", "tif": "tiff"}.get(spec.fmt, spec.fmt), **kw)


def export_map(path: str, base: RasterLayer, spec: ExportSpec, overlay: Optional[RasterLayer] = None,
               vectors: Sequence[VectorLayer] = ()) -> Tuple[float, float]:
    """Render and write the map. Returns the page size in mm."""
    fig = compose_figure(base, spec, overlay, vectors)
    save_figure(fig, path, spec)
    return fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4
