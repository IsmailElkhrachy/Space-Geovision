"""Map-coordinate helpers for the image display (pure Python: no Qt, no rasterio, unit-testable).

``transform`` is an affine transform with attributes a, b, c, d, e, f (``affine.Affine`` as used by rasterio):
x = a*col + b*row + c ; y = d*col + e*row + f, referring to the OUTER corner of pixel (0, 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AxisSpec:
    extent: Optional[Tuple[float, float, float, float]]  # (left, right, bottom, top) for imshow; None = pixel axes
    xlabel: str
    ylabel: str
    geographic: bool
    aspect: float            # y/x display ratio (1 for projected, 1/cos(lat) for geographic)
    crs_text: str = ""


def _get(t, name, default=0.0):
    try:
        return float(getattr(t, name))
    except (AttributeError, TypeError, ValueError):
        return default


def is_georeferenced(transform, crs=None) -> bool:
    """False for a missing transform and for a placeholder transform that carries no real position.

    Placeholders seen in practice: GDAL's default for files without geotags (identity, y down) and the
    ``from_origin(0, 0, 1, 1)`` transform written by older versions of this program (origin 0, pixel 1, y up).
    A raster whose origin is exactly (0, 0) with a 1-unit pixel carries no real position whatever its CRS says.
    """
    if transform is None:
        return False
    a, b, c, d, e, f = (_get(transform, n) for n in "abcdef")
    if a == 0.0 or e == 0.0:
        return False
    placeholder = (a, b, c, d) == (1.0, 0.0, 0.0, 0.0) and f in (0.0,) and e in (1.0, -1.0)
    return not placeholder


def as_crs(crs):
    """CRS object from a rasterio/pyproj CRS, a string such as 'EPSG:4326' or WKT, or None. Never raises."""
    if crs is None or hasattr(crs, "is_geographic"):
        return crs
    try:
        text = str(crs).strip()
        if not text or text.lower() in ("none", "unknown"):
            return None
        from rasterio.crs import CRS
        return CRS.from_user_input(text)
    except Exception:
        return None


def crs_description(crs) -> Tuple[bool, str]:
    """(is_geographic, short text such as 'EPSG:32648') - tolerant of missing/odd CRS objects."""
    crs = as_crs(crs)
    if crs is None:
        return False, ""
    geographic = False
    try:
        geographic = bool(crs.is_geographic)
    except Exception:
        pass
    text = ""
    try:
        epsg = crs.to_epsg()
        text = f"EPSG:{epsg}" if epsg else ""
    except Exception:
        pass
    if not text:
        try:
            text = str(crs.to_string())[:40]
        except Exception:
            text = ""
    return geographic, text


def axis_spec(transform, crs, height: int, width: int) -> AxisSpec:
    """Extent, axis labels and aspect for displaying a (height x width) raster in map coordinates."""
    crs = as_crs(crs)
    if not is_georeferenced(transform, crs) or height < 1 or width < 1:
        return AxisSpec(None, "Column (pixel)", "Row (pixel)", False, 1.0)
    a, b, c, d, e, f = (_get(transform, n) for n in "abcdef")
    if abs(b) > 1e-12 or abs(d) > 1e-12:           # rotated/sheared grid: axis-aligned map axes would be wrong
        return AxisSpec(None, "Column (pixel)", "Row (pixel)", False, 1.0)
    left, right = c, c + a * width
    top, bottom = f, f + e * height
    geographic, text = crs_description(crs)
    if not crs_is_known(crs):                       # a transform without CRS: show coordinates, say units unknown
        xl, yl = "X (map units)", "Y (map units)"
    elif geographic:
        xl, yl = "Longitude", "Latitude"
    else:
        xl, yl = "Easting (m)", "Northing (m)"
    aspect = 1.0
    if geographic:
        mid_lat = max(-85.0, min(85.0, 0.5 * (top + bottom)))
        aspect = 1.0 / max(math.cos(math.radians(mid_lat)), 0.05)
    suffix = f" - {text}" if text else ""
    return AxisSpec((left, right, bottom, top), xl + suffix, yl, geographic, aspect, text)


def crs_is_known(crs) -> bool:
    return crs is not None


def format_degrees(value: float, is_lat: bool, decimals: int = 3) -> str:
    hemi = ("N" if value >= 0 else "S") if is_lat else ("E" if value >= 0 else "W")
    return f"{abs(value):.{decimals}f}°{hemi}"


def format_projected(value: float, _pos=None) -> str:
    return f"{value:,.0f}"


def pixel_to_map(transform, row: float, col: float) -> Tuple[float, float]:
    a, b, c, d, e, f = (_get(transform, n) for n in "abcdef")
    return a * col + b * row + c, d * col + e * row + f


def map_to_pixel(transform, x: float, y: float) -> Optional[Tuple[int, int]]:
    """(row, col) of the pixel containing map point (x, y) for a north-up grid, else None."""
    a, b, c, d, e, f = (_get(transform, n) for n in "abcdef")
    if a == 0 or e == 0 or abs(b) > 1e-12 or abs(d) > 1e-12:
        return None
    return int(math.floor((y - f) / e)), int(math.floor((x - c) / a))


# ----------------------------------------------------------------------------------------------------------------------
# Drawing and writing helpers shared by every display / export path of the program
# ----------------------------------------------------------------------------------------------------------------------
def profile_transform_crs(profile):
    """(transform, crs) of a rasterio-style profile dict (or None)."""
    if not profile:
        return None, None
    try:
        return profile.get("transform"), profile.get("crs")
    except AttributeError:
        return None, None


def decorate_axes(ax, spec: "AxisSpec") -> None:
    """Axis labels, tick formats and aspect for a raster drawn with ``spec`` (degrees, metres or pixels)."""
    from matplotlib.ticker import FuncFormatter
    ax.set_xlabel(spec.xlabel, fontsize=9)
    ax.set_ylabel(spec.ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    if spec.extent is None:
        return
    if spec.geographic:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: format_degrees(v, False)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: format_degrees(v, True)))
        ax.set_aspect(spec.aspect)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(format_projected))
        ax.yaxis.set_major_formatter(FuncFormatter(format_projected))
        ax.set_aspect("equal")
    ax.tick_params(axis="x", labelrotation=0)


def draw_raster(ax, data, profile=None, transform=None, crs=None, decorate: bool = True, **imshow_kw):
    """``ax.imshow`` of an (H, W) or (H, W, C) array in its real map coordinates.

    The grid comes from ``profile`` (``transform`` / ``crs`` keys) or from the explicit ``transform`` / ``crs``
    arguments. Without usable georeferencing the raster is drawn in pixel coordinates and the axes say so.
    Returns ``(artist, AxisSpec)``.
    """
    if transform is None and crs is None:
        transform, crs = profile_transform_crs(profile)
    shape = getattr(data, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("draw_raster needs an array with at least two dimensions")
    spec = axis_spec(transform, crs, int(shape[0]), int(shape[1]))
    imshow_kw.pop("extent", None)
    artist = ax.imshow(data, extent=spec.extent, **imshow_kw)
    if decorate:
        decorate_axes(ax, spec)
    return artist, spec


def result_profile(profile, count: int = 1, dtype: str = "uint8", nodata=None, height: Optional[int] = None,
                   width: Optional[int] = None) -> dict:
    """Profile of a derived raster (classification, index, mask ...): same grid as the source, new band layout."""
    src = dict(profile) if profile else {}
    out = {k: src[k] for k in ("crs", "transform") if src.get(k) is not None}
    for key, val in (("height", height), ("width", width)):
        if val is None and src.get(key) is not None:
            val = src[key]
        if val is not None:
            out[key] = int(val)
    out.update(driver="GTiff", count=int(count), dtype=str(dtype), nodata=nodata)
    return out


def write_geotiff(path: str, array, profile=None, dtype: Optional[str] = None, nodata=None, compress: str = "lzw") -> dict:
    """Write a 2-D or (bands, H, W) array as GeoTIFF keeping the grid (transform + CRS) of ``profile``.

    Never invents a position: if ``profile`` has no usable georeferencing the file is written without geotags
    and the returned dict says so. Returns ``{"georeferenced": bool, "crs": str, "pixel_size": (dx, dy) or None}``.
    """
    import numpy as np
    import rasterio
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError("array must be 2-D or (bands, rows, cols)")
    transform, crs = profile_transform_crs(profile)
    crs = as_crs(crs)
    geo = is_georeferenced(transform, crs)
    kw = dict(driver="GTiff", height=int(arr.shape[1]), width=int(arr.shape[2]), count=int(arr.shape[0]),
              dtype=str(dtype or arr.dtype), compress=compress)
    if nodata is not None:
        kw["nodata"] = nodata
    if geo:
        kw["transform"] = transform
        if crs is not None:
            kw["crs"] = crs
    with rasterio.open(path, "w", **kw) as dst:
        dst.write(arr.astype(kw["dtype"], copy=False))
    return {"georeferenced": bool(geo), "crs": crs_description(crs)[1] if crs is not None else "",
            "pixel_size": (abs(_get(transform, "a")), abs(_get(transform, "e"))) if geo else None}


def georef_note(info: dict) -> str:
    """One-line text for a message box after saving."""
    if info.get("georeferenced"):
        return f"Georeferenced ({info.get('crs') or 'map units'}, pixel {info['pixel_size'][0]:g} x {info['pixel_size'][1]:g})."
    return "The source image has no georeferencing, so the file carries none (pixel coordinates only)."


def rescaled_profile(profile, old_hw, new_hw) -> dict:
    """Copy of ``profile`` for an array of ``new_hw`` that covers the same ground as one of ``old_hw``
    (after resampling, for instance): the pixel size is scaled, the extent stays the same."""
    out = dict(profile) if profile else {}
    tr = out.get("transform")
    (oh, ow), (nh, nw) = old_hw, new_hw
    if tr is not None and (oh, ow) != (nh, nw) and nh > 0 and nw > 0:
        try:
            from affine import Affine
            out["transform"] = tr * Affine.scale(ow / nw, oh / nh)
        except Exception:
            out["transform"] = None
    out["height"], out["width"] = int(nh), int(nw)
    return out


def inherit_profile(sources, shape) -> dict:
    """Grid for a result array of ``shape`` that has no georeferencing of its own.

    ``sources`` is a sequence of ``(profile, image)`` pairs in order of preference (the active layer first, then the
    loaded image). The first georeferenced source wins; its transform is rescaled when the result has another size.
    Returns ``{}`` when no source is georeferenced: the result is then drawn in pixel coordinates.
    """
    h, w = int(shape[-2]), int(shape[-1])
    for prof, img in sources:
        tr, crs = profile_transform_crs(prof)
        if tr is None or not is_georeferenced(tr, crs):
            continue
        shp = getattr(img, "shape", None)
        if shp is not None and len(shp) >= 2:
            ih, iw = int(shp[-2]), int(shp[-1])
        else:
            ih, iw = prof.get("height"), prof.get("width")
        if not ih or not iw:
            continue
        return rescaled_profile(prof, (ih, iw), (h, w))
    return {}


def draw_classified(ax, classified, profile=None, names=None, palette=None, title=None):
    """Draw a classified raster in map coordinates with a class legend; returns the list of class values drawn.

    ``names``    {class value: name}; a value 0 that is not named is 'no data' and left blank.
    ``palette``  list of (r, g, b) 0-255 tuples used in order of the class values (a default is used if None).
    """
    import numpy as np
    import matplotlib.patches as mpatches
    from matplotlib.colors import BoundaryNorm, ListedColormap
    arr = np.asarray(classified)
    if arr.ndim == 3:
        arr = arr[0]
    names = dict(names or {})
    values = [int(v) for v in np.unique(arr)]
    if 0 not in names:
        values = [v for v in values if v != 0]
    default = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
               (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
    pal = list(palette or default)
    colors = [tuple(c / 255.0 for c in pal[i % len(pal)]) for i in range(len(values))]
    if not values:
        draw_raster(ax, np.ma.masked_all(arr.shape), profile)
        return values
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([v - 0.5 for v in values] + [values[-1] + 0.5], cmap.N)
    shown = np.ma.masked_where(~np.isin(arr, values), arr)
    draw_raster(ax, shown, profile, cmap=cmap, norm=norm, interpolation="nearest")
    ax.legend(handles=[mpatches.Patch(color=colors[i], label=names.get(v, f"Class {v}")) for i, v in enumerate(values)],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="Classes", borderaxespad=0.0)
    if title:
        ax.set_title(title)
    return values
