"""Water mapping from calibrated sigma0 in dB (output of sar_snap) + validation against a reference mask."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import accuracy as acc
from change_detection import otsu_threshold


def otsu_water_threshold(db: np.ndarray, sample: int = 2_000_000, seed: int = 0) -> float:
    """Otsu threshold on the dB histogram (valid pixels only). Large scenes are randomly subsampled."""
    v = db[np.isfinite(db) & (db != 0.0)]       # 0.0 dB exactly = SNAP no-data fill outside the swath
    if v.size == 0:
        raise ValueError("No valid pixels.")
    if v.size > sample:
        v = np.random.default_rng(seed).choice(v, sample, replace=False)
    return otsu_threshold(v)


def tile_bimodal_pool(db: np.ndarray, tile: int = 256, ashman_min: float = 2.0, min_class_fraction: float = 0.1,
                      low_mean_max_db: float = -12.0, min_valid_fraction: float = 0.8,
                      sample_per_tile: int = 5000, seed: int = 0):
    """Pixels of the tiles whose histogram is genuinely water/land bimodal (see tile_bimodal_threshold)."""
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed)
    h, w = db.shape
    pooled, kept, tested = [], 0, 0
    for r in range(0, h - tile + 1, tile):
        for c in range(0, w - tile + 1, tile):
            blk = db[r:r + tile, c:c + tile]
            ok = np.isfinite(blk) & (blk != 0.0)
            if ok.mean() < min_valid_fraction:
                continue
            tested += 1
            v = blk[ok].astype(np.float64)
            if v.size > sample_per_tile:
                v = rng.choice(v, sample_per_tile, replace=False)
            try:
                g = GaussianMixture(2, random_state=0).fit(v.reshape(-1, 1))
            except Exception:
                continue
            mu = g.means_.ravel(); sd = np.sqrt(g.covariances_.ravel()); wt = g.weights_
            d = np.sqrt(2.0) * abs(mu[0] - mu[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
            lo = int(np.argmin(mu))
            if d >= ashman_min and wt.min() >= min_class_fraction and mu[lo] < low_mean_max_db:
                kept += 1
                pooled.append(v)
    return pooled, tested, kept


def tile_bimodal_threshold(db: np.ndarray, tile: int = 256, ashman_min: float = 2.0, min_class_fraction: float = 0.1,
                           low_mean_max_db: float = -12.0, min_valid_fraction: float = 0.8, min_tiles: int = 3,
                           sample_per_tile: int = 5000, seed: int = 0):
    """Otsu threshold computed ONLY from tiles whose histogram is genuinely water/land bimodal.

    A global Otsu threshold is unreliable when water is a small fraction of the scene and the land histogram has
    a long bright tail (urban areas): it then separates 'bright' from 'rest'. Following the idea of split-based
    thresholding (Bovolo & Bruzzone 2007; Chini et al. 2017), each tile is fitted with a 2-component Gaussian
    mixture; a tile is kept if (i) Ashman's D >= ashman_min, (ii) both components have weight >= min_class_fraction
    and (iii) the lower component mean is below `low_mean_max_db` (open water in VV is far below -12 dB; this
    prior is stated explicitly because it is an assumption of the method). Otsu is then applied to the pooled
    pixels of the kept tiles. Returns (threshold_db, info).
    """
    kw = dict(tile=tile, ashman_min=ashman_min, min_class_fraction=min_class_fraction, low_mean_max_db=low_mean_max_db,
              min_valid_fraction=min_valid_fraction, sample_per_tile=sample_per_tile, seed=seed)
    pooled, tested, kept = tile_bimodal_pool(db, **kw)
    if kept < min_tiles:
        raise ValueError(
            f"Only {kept} of {tested} tiles are bimodal water/land (need {min_tiles}). The scene probably has too little "
            "open water, or the water is not dark enough; use a fixed threshold (--method fixed) and report a sensitivity analysis.")
    thr = otsu_threshold(np.concatenate(pooled))
    return thr, {"tiles_tested": tested, "tiles_used": kept, "tile_size": tile, "ashman_min": ashman_min,
                 "min_class_fraction": min_class_fraction, "low_mean_max_db": low_mean_max_db}


def common_tile_threshold(dbs, min_tiles: int = 3, **kw):
    """ONE threshold for several dates: Otsu on the pooled bimodal-tile pixels of all images.
    Using a single threshold avoids spurious 'change' caused by two different automatic thresholds."""
    pools, info = [], []
    for k, db in enumerate(dbs):
        pooled, tested, kept = tile_bimodal_pool(db, **kw)
        pools += pooled
        info.append({"image": k, "tiles_tested": tested, "tiles_used": kept})
    used = sum(i["tiles_used"] for i in info)
    if used < min_tiles:
        raise ValueError(f"Only {used} bimodal tiles over all images (need {min_tiles}); give a fixed --threshold.")
    return otsu_threshold(np.concatenate(pools)), info


def threshold_sensitivity(db: np.ndarray, thresholds=(-14.0, -16.0, -18.0, -20.0), min_pixels: int = 100) -> list:
    """Water area (% of valid pixels) for several fixed thresholds: shows how strongly the result depends on the choice."""
    valid = np.isfinite(db) & (db != 0.0)
    n = int(valid.sum())
    out = []
    for t in thresholds:
        m = clean_mask(valid & (db < t), min_pixels) & valid
        out.append({"threshold_db": float(t), "water_percent": 100.0 * float(m.sum()) / max(n, 1)})
    return out


def clean_mask(mask: np.ndarray, min_pixels: int = 100, close_radius: int = 1) -> np.ndarray:
    """Morphological closing then removal of connected components smaller than `min_pixels`."""
    from scipy import ndimage as ndi
    m = mask.astype(bool)
    if close_radius > 0:
        st = ndi.generate_binary_structure(2, 1)
        m = ndi.binary_closing(m, structure=st, iterations=close_radius)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool); keep[1:] = sizes >= min_pixels
    return keep[lab]


def map_water(db: np.ndarray, method: str = "otsu", lower: Optional[float] = None, upper: Optional[float] = None,
              min_pixels: int = 100, threshold: Optional[float] = None, tile: int = 256) -> Tuple[np.ndarray, Dict]:
    """Water = low backscatter on sigma0 VV/VH in dB.
    method 'otsu'  : global Otsu threshold (fails when water is a small fraction - a warning is raised if implausible)
           'tiles' : Otsu on bimodal tiles only (recommended for scenes where water is a minority)
           'fixed' : db < threshold
           'manual': lower <= db <= upper
    Returns (mask, info)."""
    valid = np.isfinite(db) & (db != 0.0)       # SNAP writes 0.0 for no data and the GeoTIFF does not always declare it
    warnings = []
    if method == "otsu":
        thr = otsu_water_threshold(db)
        raw = valid & (db < thr)
        info = {"method": "otsu", "threshold_db": thr}
    elif method == "tiles":
        thr, tinfo = tile_bimodal_threshold(db, tile=tile)
        raw = valid & (db < thr)
        info = {"method": "tiles", "threshold_db": thr, **tinfo}
    elif method == "fixed":
        if threshold is None:
            raise ValueError("Method 'fixed' needs a threshold in dB.")
        raw = valid & (db < threshold)
        info = {"method": "fixed", "threshold_db": float(threshold)}
    elif method == "manual":
        if lower is None or upper is None or lower >= upper:
            raise ValueError("Manual method needs lower < upper (dB).")
        raw = valid & (db >= lower) & (db <= upper)
        info = {"method": "manual", "lower_db": lower, "upper_db": upper}
    else:
        raise ValueError("method must be 'otsu', 'tiles', 'fixed' or 'manual'")
    mask = clean_mask(raw, min_pixels) & valid
    info.update(min_pixels=min_pixels, valid_pixels=int(valid.sum()), water_pixels=int(mask.sum()),
                water_percent=100.0 * mask.sum() / max(int(valid.sum()), 1))
    v = db[valid]
    if v.size:
        p2, p98 = np.percentile(v, [2, 98])
        info["db_p2"], info["db_p98"] = float(p2), float(p98)
        if p98 > 15 or p2 > -5:
            warnings.append("Values do not look like calibrated sigma0 in dB; check the preprocessing.")
    thr_used = info.get("threshold_db", info.get("upper_db"))
    if thr_used is not None and thr_used > -10.0:
        warnings.append(f"Threshold {thr_used:.1f} dB is far above typical open-water backscatter (about -15 to -25 dB in VV). "
                        "It probably separates bright land from other land, not water.")
    if info["water_percent"] > 40.0:
        warnings.append(f"{info['water_percent']:.0f}% of the valid pixels are labelled water; check the threshold and the "
                        "histogram, unless most of the scene really is water.")
    if warnings:
        info["warning"] = " | ".join(warnings)
    return mask, info


def validate_water(mask: np.ndarray, reference: np.ndarray, valid: Optional[np.ndarray] = None) -> Dict:
    """Agreement with a reference water mask (e.g. digitised from same-day optical imagery)."""
    return acc.binary_metrics(mask, reference, valid)


# ------------------------------------------------------------------ flood extent from two water masks
FLOOD_CLASSES = {0: "Land (dry in both)", 1: "Permanent water (both dates)", 2: "Flooded (water only after)",
                 3: "Receded (water only before)", 255: "No data"}


def flood_extent(before: np.ndarray, after: np.ndarray, valid: Optional[np.ndarray] = None,
                 pixel_area_m2: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """Combine a pre-event and a post-event water mask (same grid, same SAR geometry) into flood classes.

    flooded = water after AND NOT water before. Permanent water (rivers, ponds) is excluded from the flood.
    """
    b = np.asarray(before).astype(bool); a = np.asarray(after).astype(bool)
    if b.shape != a.shape:
        raise ValueError("The two masks must have the same shape (same grid).")
    v = np.ones(b.shape, bool) if valid is None else np.asarray(valid).astype(bool)
    cls = np.zeros(b.shape, dtype=np.uint8)
    cls[v & a & b] = 1
    cls[v & a & ~b] = 2
    cls[v & ~a & b] = 3
    cls[~v] = 255
    n = int(v.sum())
    counts = {k: int((cls == k).sum()) for k in (0, 1, 2, 3)}
    stats = {"valid_pixels": n, "counts": counts,
             "percent": {k: (100.0 * c / n if n else float("nan")) for k, c in counts.items()}}
    if pixel_area_m2:
        stats["area_ha"] = {k: c * pixel_area_m2 / 1e4 for k, c in counts.items()}
    return cls, stats


# ------------------------------------------------------------------ flood from two sigma0 dB images
FLOOD_DB_CLASSES = {0: "Land (dry in both)", 1: "Permanent water", 2: "Flooded (new water, backscatter drop)",
                    3: "Receded (water only before)", 4: "Uncertain (new dark pixels without a clear drop)", 255: "No data"}


def flood_from_db(before_db: np.ndarray, after_db: np.ndarray, threshold: float, min_drop_db: float = 3.0,
                  min_pixels: int = 100, pixel_area_m2: Optional[float] = None):
    """Flood extent from two co-registered sigma0 (dB) images of the SAME track.

    Water on each date = backscatter below one common `threshold`. A pixel is FLOODED if it is water after,
    not water before, and its backscatter dropped by at least `min_drop_db` (0 disables the test). Dark
    pixels that are new but show no clear drop are reported separately as UNCERTAIN, because permanently dark
    surfaces (radar shadow, smooth soil) can cross a threshold between dates because of noise alone.
    """
    if before_db.shape != after_db.shape:
        raise ValueError("The two images must be on the same grid (resample first).")
    vb = np.isfinite(before_db) & (before_db != 0.0)
    va = np.isfinite(after_db) & (after_db != 0.0)
    valid = vb & va
    wb = clean_mask(vb & (before_db < threshold), min_pixels) & valid
    wa = clean_mask(va & (after_db < threshold), min_pixels) & valid
    new = wa & ~wb
    drop = np.where(valid, after_db - before_db, np.nan)
    strong = (drop <= -abs(min_drop_db)) if min_drop_db > 0 else np.ones(new.shape, bool)
    flooded = clean_mask(new & strong, min_pixels, close_radius=0)
    cls = np.zeros(before_db.shape, dtype=np.uint8)
    cls[wa & wb] = 1
    cls[flooded] = 2
    cls[wb & ~wa] = 3
    cls[new & ~flooded] = 4
    cls[~valid] = 255
    n = int(valid.sum())
    counts = {k: int((cls == k).sum()) for k in (0, 1, 2, 3, 4)}
    stats = {"threshold_db": float(threshold), "min_drop_db": float(min_drop_db), "valid_pixels": n, "counts": counts,
             "percent": {k: (100.0 * c / n if n else float("nan")) for k, c in counts.items()},
             "water_before_percent": 100.0 * float(wb.sum()) / max(n, 1),
             "water_after_percent": 100.0 * float(wa.sum()) / max(n, 1)}
    if pixel_area_m2:
        stats["area_ha"] = {k: c * pixel_area_m2 / 1e4 for k, c in counts.items()}
    return cls, wb, wa, stats
