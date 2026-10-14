"""Bi-temporal change detection with radiometric normalisation and statistical thresholds.

Method (change vector analysis, CVA):
  1. valid mask      : finite, not nodata, not all-zero in both dates
  2. normalisation   : per-band robust linear fit of date 2 onto date 1 (median/IQR), so that
                       illumination/atmosphere/sensor offsets do not become "change"
  3. magnitude       : per-band difference divided by its robust std (1.4826*MAD), combined as the
                       Euclidean norm / sqrt(bands)  -> magnitude in robust-sigma units
  4. thresholding    : T = k robust sigma (default k = 3) or Otsu on the magnitude
  5. classes         : 0 nodata, 1 no change (m < T), 2 change (T <= m < 2T), 3 strong change (m >= 2T)

The result is a *candidate change* map. It does not by itself separate urban change from phenology,
shadows or registration error; report it together with an accuracy assessment (see accuracy.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

CLASS_NAMES = {0: "No data", 1: "No change", 2: "Change", 3: "Strong change"}


class AlignmentError(ValueError):
    pass


def check_alignment(profile1: dict, profile2: dict, tol: float = 1e-6) -> None:
    """Raise AlignmentError unless both rasters share CRS, grid origin, pixel size and shape."""
    c1, c2 = profile1.get("crs"), profile2.get("crs")
    if c1 != c2:
        raise AlignmentError(f"CRS differ: {c1} vs {c2}. Reproject one image first.")
    if (profile1["height"], profile1["width"]) != (profile2["height"], profile2["width"]):
        raise AlignmentError(
            f"Raster sizes differ: {profile1['width']}x{profile1['height']} vs {profile2['width']}x{profile2['height']}. "
            "Clip or resample to the same grid; resizing without georeferencing would misregister the images.")
    t1, t2 = profile1["transform"], profile2["transform"]
    if any(abs(a - b) > tol * max(1.0, abs(a)) for a, b in zip(tuple(t1)[:6], tuple(t2)[:6])):
        raise AlignmentError("Raster grids (origin or pixel size) differ. Co-register the images before comparison.")


def valid_mask(a: np.ndarray, b: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    """Pixels usable in BOTH dates. Shapes are (bands, rows, cols)."""
    ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
    if nodata is not None:
        ok &= ~((a == nodata).any(axis=0) | (b == nodata).any(axis=0))
    ok &= ~((a == 0).all(axis=0) | (b == 0).all(axis=0))   # all-zero pixels are fill values (e.g. L2A nodata)
    return ok


def _robust_std(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def normalize_to_reference(ref: np.ndarray, tgt: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Per-band linear match of `tgt` to `ref` using median and inter-quartile range of valid pixels.

    Uses statistics of the whole valid area; that assumes most of the scene did not change, which is the
    usual situation and is the reason robust statistics (median/IQR) are used instead of mean/std.
    """
    out = np.empty_like(tgt, dtype=np.float32)
    params = []
    for b in range(ref.shape[0]):
        r, t = ref[b][valid], tgt[b][valid]
        r25, r75 = np.percentile(r, [25, 75]); t25, t75 = np.percentile(t, [25, 75])
        gain = (r75 - r25) / (t75 - t25) if (t75 - t25) > 0 else 1.0
        offset = np.median(r) - gain * np.median(t)
        out[b] = gain * tgt[b] + offset
        params.append({"band": b + 1, "gain": float(gain), "offset": float(offset)})
    return out, params


def magnitude(ref: np.ndarray, tgt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """CVA magnitude in robust-sigma units (NaN where not valid)."""
    nb = ref.shape[0]
    acc2 = np.zeros(ref.shape[1:], dtype=np.float64)
    for b in range(nb):
        d = (tgt[b] - ref[b]).astype(np.float64)
        s = _robust_std(d[valid])
        if s == 0:
            s = float(np.std(d[valid])) or 1.0
        acc2 += ((d - np.median(d[valid])) / s) ** 2
    mag = np.sqrt(acc2 / nb)
    mag[~valid] = np.nan
    return mag.astype(np.float32)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0 or np.ptp(v) == 0:
        raise ValueError("Cannot compute an Otsu threshold: the change magnitude has no variation.")
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


@dataclass
class ChangeResult:
    magnitude: np.ndarray
    classes: np.ndarray
    threshold: float
    method: str
    stats: Dict
    normalisation: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def detect_changes(date1: np.ndarray, date2: np.ndarray, *, normalise: bool = True, method: str = "sigma",
                   k: float = 3.0, nodata: Optional[float] = None, pixel_area_m2: Optional[float] = None) -> ChangeResult:
    """Run the full bi-temporal analysis on two co-registered (bands, rows, cols) arrays."""
    a = np.asarray(date1, dtype=np.float32); b = np.asarray(date2, dtype=np.float32)
    if a.ndim == 2: a = a[None]
    if b.ndim == 2: b = b[None]
    if a.shape != b.shape:
        raise AlignmentError(f"Array shapes differ: {a.shape} vs {b.shape}.")
    valid = valid_mask(a, b, nodata)
    n_valid = int(valid.sum())
    warnings: List[str] = []
    if n_valid < 100:
        raise ValueError("Fewer than 100 valid pixels are common to both dates.")
    if n_valid < 0.5 * valid.size:
        warnings.append(f"Only {100 * n_valid / valid.size:.0f}% of pixels are valid in both dates (nodata/fill excluded).")

    params: List[Dict] = []
    if normalise:
        b, params = normalize_to_reference(a, b, valid)
    mag = magnitude(a, b, valid)

    if method == "otsu":
        thr = otsu_threshold(mag[valid])
    elif method == "sigma":
        thr = float(k)
    else:
        raise ValueError("method must be 'sigma' or 'otsu'")
    cls = np.zeros(mag.shape, dtype=np.uint8)
    cls[valid & (mag < thr)] = 1
    cls[valid & (mag >= thr) & (mag < 2 * thr)] = 2
    cls[valid & (mag >= 2 * thr)] = 3

    counts = {c: int((cls == c).sum()) for c in (1, 2, 3)}
    pct = {c: 100.0 * counts[c] / n_valid for c in counts}
    changed = counts[2] + counts[3]
    stats = {
        "valid_pixels": n_valid, "total_pixels": int(valid.size), "counts": counts, "percent": pct,
        "changed_pixels": changed, "changed_percent": 100.0 * changed / n_valid,
        "mean_magnitude": float(np.nanmean(mag)), "median_magnitude": float(np.nanmedian(mag)),
        "threshold": thr, "changed_area_ha": (changed * pixel_area_m2 / 1e4) if pixel_area_m2 else None,
    }
    if stats["changed_percent"] > 50:
        warnings.append("More than half of the valid pixels are flagged as changed. This usually indicates "
                        "misregistration, different radiometry/season, or a threshold that is too low - not real change.")
    if stats["changed_percent"] < 0.01:
        warnings.append("Almost no change detected; check that the two dates really differ and that the threshold is not too high.")
    if not normalise:
        warnings.append("Radiometric normalisation was disabled; illumination/atmosphere differences will appear as change.")
    return ChangeResult(mag, cls, thr, method, stats, params, warnings)


def format_report(res: ChangeResult) -> str:
    s = res.stats
    lines = [f"Method: CVA magnitude, threshold = {res.threshold:.2f} robust-sigma ({res.method})",
             f"Valid pixels: {s['valid_pixels']:,} of {s['total_pixels']:,}"]
    for c in (1, 2, 3):
        lines.append(f"  {CLASS_NAMES[c]:<14}{s['counts'][c]:>12,}  {s['percent'][c]:6.2f} %")
    lines.append(f"Changed (classes 2+3): {s['changed_percent']:.2f} %")
    if s["changed_area_ha"] is not None:
        lines.append(f"Changed area: {s['changed_area_ha']:.2f} ha")
    lines += [f"Band {p['band']}: gain={p['gain']:.4f} offset={p['offset']:.2f}" for p in res.normalisation]
    lines += ["WARNING: " + w for w in res.warnings]
    return "\n".join(lines)


def export_classes(classes: np.ndarray) -> np.ndarray:
    """Class map for export/validation: 1 no change, 2 change, 3 strong change, 255 = no data (0 is never used)."""
    return np.where(np.asarray(classes) == 0, 255, classes).astype(np.uint8)


def change_sensitivity(date1: np.ndarray, date2: np.ndarray, ks=(2.0, 3.0, 4.0, 5.0), nodata: Optional[float] = None) -> List[Dict]:
    """Share of valid pixels flagged as change for several thresholds k (in robust sigma), with and without
    radiometric normalisation. Shows how strongly the reported change area depends on these choices."""
    a = np.asarray(date1, dtype=np.float32); b = np.asarray(date2, dtype=np.float32)
    if a.ndim == 2: a = a[None]
    if b.ndim == 2: b = b[None]
    if a.shape != b.shape:
        raise AlignmentError(f"Array shapes differ: {a.shape} vs {b.shape}.")
    valid = valid_mask(a, b, nodata)
    if int(valid.sum()) < 100:
        raise ValueError("Fewer than 100 valid pixels are common to both dates.")
    rows: List[Dict] = []
    for norm in (True, False):
        bb = normalize_to_reference(a, b, valid)[0] if norm else b
        m = magnitude(a, bb, valid)[valid]
        for k in ks:
            rows.append({"normalise": norm, "k": float(k), "changed_percent": 100.0 * float((m >= k).mean()),
                         "strong_percent": 100.0 * float((m >= 2 * k).mean())})
    return rows


def format_sensitivity(rows: List[Dict]) -> str:
    lines = [f"{'normalisation':<15}{'k (sigma)':>10}{'changed %':>12}{'strong %':>12}"]
    for r in rows:
        lines.append(f"{'on' if r['normalise'] else 'off':<15}{r['k']:>10.1f}{r['changed_percent']:>12.2f}{r['strong_percent']:>12.2f}")
    lines.append("\nA result that depends strongly on k or on normalisation is not robust; report this table with the map.")
    return "\n".join(lines)
