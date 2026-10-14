"""DEM terrain and hydrology numerics (no Qt, no rasterio: unit-testable).

Conventions
-----------
* Arrays are (rows, cols) with row 0 in the north (north-up rasters).  ``xres``/``yres`` are the cell sizes in the
  same unit as the elevations (use :func:`cell_size`, which converts geographic degrees to metres).
* Elevation nodata is NaN.
* D8 flow direction uses the ESRI codes 1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE.
  ``0`` marks a pit / outlet (no lower neighbour), ``255`` marks nodata.
* Slope, aspect and hillshade use central differences (``numpy.gradient``).  Aspect is the compass direction
  (clockwise from north) the surface faces, -1 on flat cells.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
NODATA_DIR = 255
COMMON_NODATA = (-9999.0, -32768.0, -32767.0, -3.4028234663852886e38, -2147483648.0)


# ---------------------------------------------------------------------------------------------- geometry helpers
def cell_size(transform, shape: Optional[Tuple[int, int]] = None, crs=None) -> Tuple[float, float]:
    """Cell size (xres, yres) in metres when ``crs`` is geographic (degrees converted at the mid latitude),
    otherwise in the CRS unit."""
    xres, yres = abs(float(transform.a)), abs(float(transform.e))
    if crs is not None and getattr(crs, "is_geographic", False):
        rows = shape[0] if shape else 0
        lat = math.radians(float(transform.f) + float(transform.e) * rows / 2.0)
        m_lat = 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
        m_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
        return xres * max(m_lon, 1.0), yres * m_lat
    return xres, yres


def clean_dem(dem: np.ndarray, nodata: Optional[float] = None, extra_nodata: Sequence[float] = COMMON_NODATA) -> np.ndarray:
    """float32 copy with nodata sentinels and non-finite values set to NaN. No elevation range is imposed."""
    z = np.array(dem, dtype=np.float32, copy=True)
    bad = ~np.isfinite(z)
    sentinels = [v for v in (nodata, *extra_nodata) if v is not None and np.isfinite(v)]
    for v in sentinels:
        bad |= (z == np.float32(v)) if abs(v) < 3e38 else (z <= np.float32(-3.0e38))
    z[bad] = np.nan
    return z


def fill_small_voids(dem: np.ndarray, max_px: int = 3) -> np.ndarray:
    """Fill NaN voids narrower than ``max_px`` pixels with the nearest valid value; larger voids stay NaN."""
    from scipy import ndimage as ndi
    nan = np.isnan(dem)
    if not nan.any() or nan.all():
        return dem
    dist, (ri, ci) = ndi.distance_transform_edt(nan, return_indices=True)
    out = dem.copy()
    fill = nan & (dist <= max_px)
    out[fill] = dem[ri[fill], ci[fill]]
    return out


def smooth_nan(dem: np.ndarray, sigma: float = 0.75) -> np.ndarray:
    """Gaussian smoothing that ignores NaN (normalised convolution); NaN cells stay NaN."""
    from scipy import ndimage as ndi
    nan = np.isnan(dem)
    num = ndi.gaussian_filter(np.where(nan, 0.0, dem).astype(np.float64), sigma)
    den = ndi.gaussian_filter((~nan).astype(np.float64), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[nan] = np.nan
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------------- surface derivatives
def _gradients(dem: np.ndarray, xres: float, yres: float, z_factor: float = 1.0):
    """(dz/dx east, dz/dy north) of a north-up DEM."""
    if xres <= 0 or yres <= 0:
        raise ValueError("Cell size must be positive")
    z = np.asarray(dem, dtype=np.float64) * z_factor
    d_row, d_col = np.gradient(z, yres, xres)       # axis 0 = rows (south), axis 1 = columns (east)
    nan = np.isnan(z)
    d_col[nan] = np.nan
    d_row[nan] = np.nan
    return d_col, -d_row


def slope(dem: np.ndarray, xres: float, yres: float, unit: str = "degrees", max_percent: float = 1000.0):
    dzdx, dzdy = _gradients(dem, xres, yres)
    rise = np.hypot(dzdx, dzdy)
    deg = np.degrees(np.arctan(rise))
    pct = np.minimum(rise * 100.0, max_percent)
    if unit == "degrees":
        return deg
    if unit == "percent_rise":
        return pct
    if unit == "both":
        return deg, pct
    raise ValueError("unit must be 'degrees', 'percent_rise' or 'both'")


def aspect(dem: np.ndarray, xres: float, yres: float) -> np.ndarray:
    """Direction the slope faces in degrees clockwise from north (0-360); -1 on flat cells; NaN where DEM is NaN."""
    dzdx, dzdy = _gradients(dem, xres, yres)
    az = (np.degrees(np.arctan2(-dzdx, -dzdy)) + 360.0) % 360.0     # downslope direction
    az = np.where((dzdx == 0) & (dzdy == 0), -1.0, az)
    return np.where(np.isnan(dzdx) | np.isnan(dzdy), np.nan, az)


def hillshade(dem: np.ndarray, xres: float, yres: float, azimuth: float = 315.0, altitude: float = 45.0,
              z_factor: float = 1.0) -> np.ndarray:
    """Lambertian hillshade, uint8 0-255 (0 where the DEM is NaN). ``azimuth``: sun direction, clockwise from north."""
    dzdx, dzdy = _gradients(dem, xres, yres, z_factor)
    az, alt = math.radians(azimuth), math.radians(altitude)
    shade = (math.sin(alt) - dzdx * math.cos(alt) * math.sin(az) - dzdy * math.cos(alt) * math.cos(az)) \
        / np.sqrt(1.0 + dzdx ** 2 + dzdy ** 2)
    out = np.clip(shade, 0.0, 1.0) * 255.0
    out = np.where(np.isnan(shade), 0.0, out)
    return np.rint(out).astype(np.uint8)


def ruggedness(dem: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index (Riley et al. 1999): sqrt of the sum of squared differences to the 8 neighbours.
    Edge cells use replicated edge values."""
    z = np.asarray(dem, dtype=np.float64)
    p = np.pad(z, 1, mode="edge")
    h, w = z.shape
    acc = np.zeros_like(z)
    for di, dj in D8.values():
        acc += (p[1 + di:1 + di + h, 1 + dj:1 + dj + w] - z) ** 2
    return np.sqrt(acc)


def contour_lines(dem: np.ndarray, a: float, c: float, e: float, f: float, interval: float, base: float = 0.0,
                  max_levels: int = 2000) -> List[Tuple[float, np.ndarray]]:
    """Contours at ``base + k*interval`` for every k inside the DEM range.
    (a, c, e, f) are the affine terms: x = c + (col + .5) * a, y = f + (row + .5) * e. Returns [(level, Nx2 xy)]."""
    from skimage import measure
    if interval <= 0:
        raise ValueError("Contour interval must be greater than 0")
    z = np.asarray(dem, dtype=np.float64)
    if not np.isfinite(z).any():
        raise ValueError("DEM has no valid cells")
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    first = math.ceil((lo - base) / interval) * interval + base
    levels = np.arange(first, hi, interval)
    if levels.size == 0:
        raise ValueError(f"No contour level of interval {interval} lies inside the elevation range {lo:.1f}-{hi:.1f}")
    if levels.size > max_levels:
        raise ValueError(f"{levels.size} contour levels requested (limit {max_levels}); use a larger interval")
    mask = np.isnan(z)
    zz = np.where(mask, lo, z)
    out = []
    for lev in levels:
        for ln in measure.find_contours(zz, lev, mask=~mask):
            if len(ln) < 2:
                continue
            xy = np.column_stack([c + (ln[:, 1] + 0.5) * a, f + (ln[:, 0] + 0.5) * e])
            out.append((float(lev), xy))
    return out


# ---------------------------------------------------------------------------------------------- hydrology
def fill_depressions(dem: np.ndarray) -> np.ndarray:
    """Priority-Flood with epsilon (Barnes, Lehman & Mulla 2014): fills depressions and imposes a minimal gradient on
    the filled flats, so D8 routing never stalls. NaN cells are nodata; cells on the grid edge or next to nodata are
    outlets. float64 result. Pure Python (about 10 s per million cells)."""
    z = np.array(dem, dtype=np.float64, copy=True)
    h, w = z.shape
    flat = z.ravel()
    nan = np.isnan(flat)
    n = flat.size
    closed = nan.copy()
    heap: list = []
    pit: deque = deque()
    nanimg = nan.reshape(h, w)
    outlet = np.zeros((h, w), bool)
    outlet[0, :] = outlet[-1, :] = outlet[:, 0] = outlet[:, -1] = True
    if nanimg.any():
        near = np.zeros((h, w), bool)
        for di, dj in D8.values():
            sh = np.roll(np.roll(nanimg, -di, 0), -dj, 1)
            near |= sh
        outlet |= near
    for idx in np.flatnonzero(outlet.ravel() & ~nan):
        closed[idx] = True
        heapq.heappush(heap, (flat[idx], int(idx)))
    zl = flat.tolist()
    cl = closed.tolist()
    offs = [(di, dj) for di, dj in D8.values()]
    nextafter = np.nextafter
    inf = math.inf
    while heap or pit:
        if pit:
            ci = pit.popleft(); zc = zl[ci]
        else:
            zc, ci = heapq.heappop(heap)
        r, c = divmod(ci, w)
        for di, dj in offs:
            nr, nc = r + di, c + dj
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            ni = nr * w + nc
            if cl[ni]:
                continue
            cl[ni] = True
            if zl[ni] <= zc:
                zl[ni] = float(nextafter(zc, inf))
                pit.append(ni)
            else:
                heapq.heappush(heap, (zl[ni], ni))
    out = np.array(zl, dtype=np.float64).reshape(h, w)
    out[nanimg] = np.nan
    return out


def flow_direction(dem: np.ndarray, xres: float, yres: float) -> np.ndarray:
    """D8 flow direction (ESRI codes) towards the steepest descent neighbour; 0 = pit/outlet, 255 = nodata."""
    z = np.asarray(dem, dtype=np.float64)
    h, w = z.shape
    p = np.pad(z, 1, mode="constant", constant_values=np.nan)
    best = np.zeros((h, w))
    code = np.zeros((h, w), np.uint8)
    for k, (di, dj) in D8.items():
        nb = p[1 + di:1 + di + h, 1 + dj:1 + dj + w]
        dist = math.hypot(di * yres, dj * xres)
        with np.errstate(invalid="ignore"):
            drop = (z - nb) / dist
        drop = np.where(np.isnan(drop), -np.inf, drop)
        better = drop > best
        best = np.where(better, drop, best)
        code = np.where(better, k, code).astype(np.uint8)
    code[np.isnan(z)] = NODATA_DIR
    return code


_LUT_DI = np.zeros(256, np.int64)
_LUT_DJ = np.zeros(256, np.int64)
for _k, (_di, _dj) in D8.items():
    _LUT_DI[_k], _LUT_DJ[_k] = _di, _dj


def _topology(flow_dir: np.ndarray):
    """Downstream flat index per cell (-1 = none) and the cells grouped in 'levels' such that every cell appears after
    all of its upstream neighbours (level-synchronous Kahn)."""
    h, w = flow_dir.shape
    fd = flow_dir.ravel()
    n = fd.size
    valid = fd != NODATA_DIR
    rr, cc = np.divmod(np.arange(n), w)
    has = valid & (fd > 0)
    nr = np.where(has, rr + _LUT_DI[fd], -1)
    nc = np.where(has, cc + _LUT_DJ[fd], -1)
    inside = has & (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    down = np.where(inside, nr * w + nc, -1)
    down[inside & (fd[np.where(inside, down, 0)] == NODATA_DIR)] = -1
    indeg = np.bincount(down[down >= 0], minlength=n).astype(np.int64)
    frontier = np.flatnonzero(valid & (indeg == 0))
    levels = []
    while frontier.size:
        levels.append(frontier)
        d = down[frontier]
        d = d[d >= 0]
        if d.size == 0:
            break
        np.subtract.at(indeg, d, 1)
        cand = np.unique(d)
        frontier = cand[indeg[cand] == 0]
    return down, levels, valid


def flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """Number of cells draining through each cell (itself included). Nodata cells get 0."""
    down, levels, valid = _topology(flow_dir)
    acc = valid.astype(np.int64)
    for lvl in levels:
        d = down[lvl]
        m = d >= 0
        if m.any():
            np.add.at(acc, d[m], acc[lvl[m]])
    return acc.reshape(flow_dir.shape).astype(np.int64)


def extract_streams(accumulation: np.ndarray, method: str = "percentile", value: float = 95) -> np.ndarray:
    """uint8 stream mask. ``percentile`` is taken over cells with accumulation > 1; ``fixed`` is a cell count."""
    if method == "percentile":
        pool = accumulation[accumulation > 1]
        if pool.size == 0:
            return np.zeros(accumulation.shape, np.uint8)
        threshold = np.percentile(pool, value)
    elif method == "fixed":
        threshold = value
    else:
        raise ValueError("method must be 'percentile' or 'fixed'")
    return (accumulation >= max(threshold, 2)).astype(np.uint8)


def stream_order(flow_dir: np.ndarray, streams: np.ndarray, method: str = "strahler") -> np.ndarray:
    """Strahler or Shreve order for the stream cells (0 elsewhere)."""
    if method not in ("strahler", "shreve"):
        raise ValueError("method must be 'strahler' or 'shreve'")
    down, levels, _ = _topology(flow_dir)
    s = streams.ravel() > 0
    n = s.size
    order = np.zeros(n, np.int32)
    maxin = np.zeros(n, np.int32)
    cntmax = np.zeros(n, np.int32)
    sumin = np.zeros(n, np.int32)
    for lvl in levels:
        cells = lvl[s[lvl]]
        if cells.size == 0:
            continue
        if method == "strahler":
            o = np.where(maxin[cells] == 0, 1, maxin[cells] + (cntmax[cells] >= 2))
        else:
            o = np.where(sumin[cells] == 0, 1, sumin[cells])
        order[cells] = o
        d = down[cells]
        m = (d >= 0)
        m[m] = s[d[m]]
        for t, v in zip(d[m].tolist(), o[m].tolist()):
            if v > maxin[t]:
                maxin[t], cntmax[t] = v, 1
            elif v == maxin[t]:
                cntmax[t] += 1
            sumin[t] += v
    return order.reshape(flow_dir.shape)


def stream_segments(flow_dir: np.ndarray, order: np.ndarray, a: float, c: float, e: float, f: float):
    """One two-point segment per stream cell towards its downstream stream cell: [(order, (x0,y0), (x1,y1))]."""
    down, _, _ = _topology(flow_dir)
    h, w = flow_dir.shape
    o = order.ravel()
    cells = np.flatnonzero((o > 0) & (down >= 0))
    cells = cells[o[down[cells]] > 0]
    r0, c0 = np.divmod(cells, w)
    r1, c1 = np.divmod(down[cells], w)
    xs0, ys0 = c + (c0 + 0.5) * a, f + (r0 + 0.5) * e
    xs1, ys1 = c + (c1 + 0.5) * a, f + (r1 + 0.5) * e
    return [(int(k), (float(x0), float(y0)), (float(x1), float(y1)))
            for k, x0, y0, x1, y1 in zip(o[cells], xs0, ys0, xs1, ys1)]
