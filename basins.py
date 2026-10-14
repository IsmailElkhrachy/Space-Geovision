"""Watershed delineation, basin morphometrics, HAND, TWI and SPI (numpy only: no Qt, no rasterio; unit-testable).

All functions work on the grids produced by :mod:`terrain`: a D8 ``flow_dir`` (ESRI codes, 0 = pit/outlet,
255 = nodata), the flow ``acc``umulation in cells, a 0/1 ``streams`` mask and a stream ``order`` raster. Arrays are
(rows, cols), north-up. ``xres``/``yres`` are cell sizes in metres (use :func:`terrain.cell_size`).

Methods (so the paper can cite them):
* Watershed: all cells whose D8 flow path reaches the outlet cell (O'Callaghan & Mark 1984).
* Morphometrics: Horton (1945) / Strahler (1952) / Schumm (1956) / Miller (1953) / Gravelius (1914) definitions.
* HAND: Rennó et al. (2008), height above the nearest drainage cell *along the flow path* on a depression-filled DEM.
* TWI: Beven & Kirkby (1979), ln(a / tan b); SPI: Moore et al. (1991), ln(a * tan b + 0.001).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import terrain

_MIN_TAN = math.tan(math.radians(0.1))   # flat cells: avoid division by ~0 in the TWI


# ------------------------------------------------------------------------------------------------- helpers
def _step_lengths(flow_dir: np.ndarray, xres: float, yres: float, down: Optional[np.ndarray] = None) -> np.ndarray:
    """Length (m) of the step from every cell to its D8 downstream neighbour (0 where there is none, e.g. the
    last cell before the edge of the grid)."""
    if down is None:
        down = terrain._topology(flow_dir)[0]
    fd = flow_dir.ravel()
    di = np.abs(terrain._LUT_DI[fd]).astype(np.float64)
    dj = np.abs(terrain._LUT_DJ[fd]).astype(np.float64)
    step = np.hypot(di * yres, dj * xres)
    step[(fd == 0) | (fd == terrain.NODATA_DIR) | (down < 0)] = 0.0
    return step


def snap_to_stream(acc: np.ndarray, row: int, col: int, radius: int = 5, min_acc: int = 2) -> Tuple[int, int]:
    """Move (row, col) to the cell with the largest flow accumulation within ``radius`` cells (pour-point snapping).
    Returns the input cell if nothing in the window reaches ``min_acc``."""
    h, w = acc.shape
    if not (0 <= row < h and 0 <= col < w):
        raise ValueError("The pour point is outside the DEM")
    r0, r1 = max(row - radius, 0), min(row + radius + 1, h)
    c0, c1 = max(col - radius, 0), min(col + radius + 1, w)
    win = acc[r0:r1, c0:c1]
    if win.max() < min_acc:
        return row, col
    # nearest among ties
    rr, cc = np.nonzero(win == win.max())
    d = (rr + r0 - row) ** 2 + (cc + c0 - col) ** 2
    k = int(np.argmin(d))
    return int(rr[k] + r0), int(cc[k] + c0)


# ------------------------------------------------------------------------------------------------- watershed
def delineate_watershed(flow_dir: np.ndarray, row: int, col: int) -> np.ndarray:
    """Boolean mask of every cell that drains through the outlet cell (row, col), the outlet included."""
    h, w = flow_dir.shape
    if not (0 <= row < h and 0 <= col < w):
        raise ValueError("The pour point is outside the DEM")
    if flow_dir[row, col] == terrain.NODATA_DIR:
        raise ValueError("The pour point is on a no-data cell")
    down, levels, _ = terrain._topology(flow_dir)
    mask = np.zeros(h * w, bool)
    mask[row * w + col] = True
    # downstream levels first: a cell belongs to the basin if its downstream neighbour does
    for lvl in reversed(levels):
        d = down[lvl]
        m = d >= 0
        if m.any():
            sel = lvl[m]
            mask[sel] |= mask[d[m]]
    return mask.reshape(h, w)


def flow_length_to_outlet(flow_dir: np.ndarray, mask: np.ndarray, xres: float, yres: float,
                          outlet: Tuple[int, int]) -> np.ndarray:
    """Along-flow distance (m) from each basin cell to the outlet; NaN outside the basin."""
    h, w = flow_dir.shape
    down, levels, _ = terrain._topology(flow_dir)
    step = _step_lengths(flow_dir, xres, yres, down)
    m = mask.ravel()
    dist = np.full(h * w, np.nan)
    dist[outlet[0] * w + outlet[1]] = 0.0
    for lvl in reversed(levels):
        d = down[lvl]
        ok = (d >= 0) & m[lvl]
        if ok.any():
            sel = lvl[ok]
            upd = ~np.isfinite(dist[sel])
            sel, dd = sel[upd], d[ok][upd]
            dist[sel] = dist[dd] + step[sel]
    return dist.reshape(h, w)


# ------------------------------------------------------------------------------------------------- morphometrics
def _count_segments(flow_dir: np.ndarray, order: np.ndarray, mask: np.ndarray) -> Dict[int, int]:
    """Number of stream segments per order inside the basin (a segment ends where the stream joins a higher order or
    leaves the basin)."""
    h, w = flow_dir.shape
    down, _, _ = terrain._topology(flow_dir)
    o = order.ravel()
    m = mask.ravel()
    cells = np.flatnonzero((o > 0) & m)
    d = down[cells]
    nxt = np.where(d >= 0, o[np.where(d >= 0, d, 0)], 0)
    nxt = np.where((d >= 0) & m[np.where(d >= 0, d, 0)], nxt, 0)
    ends = cells[(nxt > o[cells]) | (nxt == 0)]
    counts: Dict[int, int] = {}
    for k in o[ends]:
        counts[int(k)] = counts.get(int(k), 0) + 1
    return dict(sorted(counts.items()))


def basin_morphometrics(dem: np.ndarray, mask: np.ndarray, flow_dir: np.ndarray, streams: np.ndarray,
                        order: np.ndarray, xres: float, yres: float, outlet: Tuple[int, int]) -> Dict[str, object]:
    """Standard basin morphometry. Lengths in km, areas in km2, relief in m. ``order`` is a Strahler order raster."""
    if not mask.any():
        raise ValueError("Empty basin")
    cell_area = xres * yres
    n_cells = int(mask.sum())
    area_km2 = n_cells * cell_area / 1e6

    # perimeter: exposed 4-neighbour cell edges (staircase estimate; it over-estimates a smooth boundary)
    p = np.pad(mask, 1, constant_values=False)
    exp_ns = (mask & ~p[:-2, 1:-1]).sum() + (mask & ~p[2:, 1:-1]).sum()   # north / south edges (length xres)
    exp_ew = (mask & ~p[1:-1, :-2]).sum() + (mask & ~p[1:-1, 2:]).sum()   # west / east edges (length yres)
    perimeter_km = (exp_ns * xres + exp_ew * yres) / 1000.0

    dist = flow_length_to_outlet(flow_dir, mask, xres, yres, outlet)
    basin_length_km = float(np.nanmax(dist)) / 1000.0     # longest flow path (Lb)

    z = np.where(mask, dem, np.nan)
    zmin, zmax, zmean = float(np.nanmin(z)), float(np.nanmax(z)), float(np.nanmean(z))
    relief_m = zmax - zmin

    st = (streams > 0) & mask
    step = _step_lengths(flow_dir, xres, yres).reshape(flow_dir.shape)
    stream_len_by_order: Dict[int, float] = {}
    for k in np.unique(order[st]):
        if k > 0:
            stream_len_by_order[int(k)] = float(step[st & (order == k)].sum()) / 1000.0
    total_stream_km = float(sum(stream_len_by_order.values()))
    seg = _count_segments(flow_dir, order, mask)
    n_total = int(sum(seg.values()))

    orders = sorted(seg)
    ratios = [seg[a] / seg[b] for a, b in zip(orders[:-1], orders[1:]) if seg[b] > 0]
    rb_mean = float(np.mean(ratios)) if ratios else float("nan")
    # weighted bifurcation ratio (Strahler 1953): weights = segments involved in the two orders
    wts = [seg[a] + seg[b] for a, b in zip(orders[:-1], orders[1:])]
    rb_weighted = float(np.average(ratios, weights=wts)) if ratios else float("nan")

    Lb, A, P = basin_length_km, area_km2, perimeter_km
    dd = total_stream_km / A if A > 0 else float("nan")
    out: Dict[str, object] = {
        "Area (km2)": A,
        "Perimeter (km)": P,
        "Basin length, longest flow path (km)": Lb,
        "Min elevation (m)": zmin,
        "Max elevation (m)": zmax,
        "Mean elevation (m)": zmean,
        "Relief (m)": relief_m,
        "Relief ratio (Rr = relief / Lb)": (relief_m / 1000.0) / Lb if Lb > 0 else float("nan"),
        "Maximum stream order (Strahler)": int(max(orders)) if orders else 0,
        "Total stream length (km)": total_stream_km,
        "Number of stream segments": n_total,
        "Drainage density Dd (km/km2)": dd,
        "Stream frequency Fs (segments/km2)": n_total / A if A > 0 else float("nan"),
        "Mean bifurcation ratio Rb": rb_mean,
        "Weighted bifurcation ratio Rbw": rb_weighted,
        "Form factor Rf (A / Lb^2)": A / Lb ** 2 if Lb > 0 else float("nan"),
        "Elongation ratio Re (2 sqrt(A/pi) / Lb)": 2.0 * math.sqrt(A / math.pi) / Lb if Lb > 0 else float("nan"),
        "Circularity ratio Rc (4 pi A / P^2)": 4.0 * math.pi * A / P ** 2 if P > 0 else float("nan"),
        "Compactness (Gravelius) Kc (P / 2 sqrt(pi A))": P / (2.0 * math.sqrt(math.pi * A)) if A > 0 else float("nan"),
        "Constant of channel maintenance C (km2/km)": 1.0 / dd if dd and dd > 0 else float("nan"),
        "Length of overland flow Lo (km)": 1.0 / (2.0 * dd) if dd and dd > 0 else float("nan"),
        "Ruggedness number Hd (Dd x relief in km)": dd * relief_m / 1000.0 if dd == dd else float("nan"),
        "Segments per order": seg,
        "Stream length per order (km)": stream_len_by_order,
    }
    return out


def hypsometric_curve(dem: np.ndarray, mask: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
    """(relative area a/A, relative height h/H, hypsometric integral). The integral is the elevation-relief ratio
    (mean - min) / (max - min) (Pike & Wilson 1971), which equals the area under the curve."""
    z = dem[mask & np.isfinite(dem)]
    if z.size == 0:
        raise ValueError("Empty basin")
    zmin, zmax = float(z.min()), float(z.max())
    if zmax <= zmin:
        raise ValueError("Flat basin: hypsometric curve undefined")
    levels = np.linspace(zmin, zmax, bins + 1)
    rel_h = (levels - zmin) / (zmax - zmin)
    rel_a = np.array([(z >= lv).mean() for lv in levels])
    hi = (float(z.mean()) - zmin) / (zmax - zmin)
    return rel_a, rel_h, hi


# ------------------------------------------------------------------------------------------------- HAND / TWI / SPI
def hand(dem_filled: np.ndarray, flow_dir: np.ndarray, streams: np.ndarray,
         xres: float, yres: float) -> Tuple[np.ndarray, np.ndarray]:
    """Height Above the Nearest Drainage (m) and the along-flow distance (m) to that drainage cell.

    Each cell follows its D8 path downstream until it meets a stream cell; HAND = z(cell) - z(that stream cell).
    Use a depression-filled DEM so the path never climbs. Cells that never reach a stream (they drain out of the DEM
    or into a pit first) and no-data cells are NaN; stream cells are 0.
    """
    h, w = flow_dir.shape
    down, levels, valid = terrain._topology(flow_dir)
    step = _step_lengths(flow_dir, xres, yres, down)
    s = (streams.ravel() > 0) & valid
    ref = np.full(h * w, -1, np.int64)
    dist = np.full(h * w, np.nan)
    ref[s] = np.flatnonzero(s)
    dist[s] = 0.0
    for lvl in reversed(levels):
        cells = lvl[~s[lvl]]
        if cells.size == 0:
            continue
        d = down[cells]
        ok = d >= 0
        cells, d = cells[ok], d[ok]
        r = ref[d]
        good = r >= 0
        cells, d, r = cells[good], d[good], r[good]
        ref[cells] = r
        dist[cells] = dist[d] + step[cells]
    z = np.asarray(dem_filled, dtype=np.float64).ravel()
    out = np.full(h * w, np.nan)
    reached = ref >= 0
    out[reached] = np.maximum(z[reached] - z[ref[reached]], 0.0)
    return out.reshape(h, w).astype(np.float32), dist.reshape(h, w).astype(np.float32)


def _specific_area_and_tan(acc: np.ndarray, dem: np.ndarray, xres: float, yres: float):
    cell_area = xres * yres
    width = math.sqrt(cell_area)                       # effective contour width of a cell
    a = acc.astype(np.float64) * cell_area / width     # specific catchment area (m2/m)
    slope_deg = terrain.slope(dem, xres, yres, "degrees")
    tanb = np.maximum(np.tan(np.radians(slope_deg.astype(np.float64))), _MIN_TAN)
    return a, tanb


def twi(acc: np.ndarray, dem: np.ndarray, xres: float, yres: float) -> np.ndarray:
    """Topographic Wetness Index ln(a / tan b). NaN where the DEM or the accumulation is no-data."""
    a, tanb = _specific_area_and_tan(acc, dem, xres, yres)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(a / tanb)
    out[(acc <= 0) | ~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def spi(acc: np.ndarray, dem: np.ndarray, xres: float, yres: float) -> np.ndarray:
    """Stream Power Index ln(a tan b + 0.001). NaN where no-data."""
    a, tanb = _specific_area_and_tan(acc, dem, xres, yres)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(a * tanb + 0.001)
    out[(acc <= 0) | ~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def format_report(metrics: Dict[str, object], hi: Optional[float] = None) -> str:
    """Plain-text table of the morphometric parameters (for the dialog and the CSV export)."""
    lines: List[str] = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            lines.append(f"{k}: " + ", ".join(f"{a}: {b:.3f}" if isinstance(b, float) else f"{a}: {b}" for a, b in v.items()))
        elif isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")
    if hi is not None:
        lines.append(f"Hypsometric integral HI: {hi:.4f}")
    return "\n".join(lines)
