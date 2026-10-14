"""Flood mapping from two Sentinel-1 sigma0-dB GeoTIFFs (output of sar_snap) of the SAME track.

Shared by the command line (cli.py flood-db) and the GUI. rasterio is imported lazily.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, Dict, Optional, Sequence

import numpy as np

import water_mapping as wm

CLASS_COLOURS = {0: "#e6e6e6", 1: "#1f4fbf", 2: "#e8302a", 3: "#2aa84a", 4: "#f0a030"}


def pixel_area_m2(transform, crs_is_geographic: bool, mid_lat: float) -> float:
    """Approximate pixel area. Geographic grids: degrees converted at the scene's mid-latitude."""
    if crs_is_geographic:
        return abs(transform.a) * 111320 * math.cos(math.radians(mid_lat)) * abs(transform.e) * 110574
    return abs(transform.a * transform.e)


def report_text(st: Dict, area: float, geographic: bool) -> str:
    lines = [f"Valid pixels {st['valid_pixels']:,}   pixel area ~ {area:.1f} m2" + (" (approximate, from degrees)" if geographic else "")
             + f"   threshold {st['threshold_db']:.2f} dB   min drop {st['min_drop_db']} dB"]
    for k in (1, 2, 3, 4):
        lines.append(f"  {k} {wm.FLOOD_DB_CLASSES[k]:<48}{st['counts'][k]:>11,}  {st['percent'][k]:6.2f} %  {st['area_ha'][k]:>10,.1f} ha")
    lines.append(f"Water before {st['water_before_percent']:.1f} %   water after {st['water_after_percent']:.1f} %")
    lines.append("\nSensitivity to the common threshold:")
    for r in st.get("sensitivity", []):
        lines.append(f"  {r['threshold_db']:>6.1f} dB   flooded {r['flooded_ha']:>10,.1f} ha   uncertain {r['uncertain_ha']:>10,.1f} ha"
                     f"   water after {r['water_after_percent']:5.1f} %")
    lines.append("\nNot validated against a reference. Water under dense vegetation or in built-up areas is not detected.")
    return "\n".join(lines)


def run_flood_db(before_path: str, after_path: str, out_dir: str, threshold: Optional[float] = None,
                 min_drop_db: float = 3.0, min_pixels: int = 100, sensitivity: Sequence[float] = (-12.0, -14.0, -16.0, -18.0),
                 progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    import rasterio
    from rasterio.warp import reproject, Resampling
    import change_detection as cd
    say = progress or (lambda p, m: None)
    say(5, "Reading images")
    with rasterio.open(after_path) as sa, rasterio.open(before_path) as sb:
        after = sa.read(1).astype("float32"); after[after == 0.0] = np.nan
        before_src = sb.read(1).astype("float32"); before_src[before_src == 0.0] = np.nan
        prof = sa.profile.copy(); tr = sa.transform; geographic = bool(sa.crs.is_geographic)
        lat = (sa.bounds.top + sa.bounds.bottom) / 2
        note = ""
        try:
            cd.check_alignment(sb.profile, sa.profile)
            before = before_src
        except cd.AlignmentError as exc:
            note = f"Grids differ ({str(exc)[:70]}...): the BEFORE image was resampled onto the AFTER grid (bilinear)."
            say(15, "Aligning grids")
            before = np.full(after.shape, np.nan, dtype="float32")
            reproject(before_src, before, src_transform=sb.transform, src_crs=sb.crs, src_nodata=np.nan,
                      dst_transform=sa.transform, dst_crs=sa.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    say(30, "Choosing the common threshold")
    tinfo = None
    if threshold is None:
        threshold, tinfo = wm.common_tile_threshold([before, after])
    area = pixel_area_m2(tr, geographic, lat)
    say(55, "Mapping flood extent")
    cls, wb, wa, st = wm.flood_from_db(before, after, threshold, min_drop_db, min_pixels, area)
    st["pixel_area_m2_approx"] = area
    st["threshold_source"] = "fixed (user)" if tinfo is None else "bimodal tiles of both dates"
    st["tile_info"] = tinfo
    st["grid_note"] = note
    say(75, "Sensitivity analysis")
    st["sensitivity"] = []
    for t in sensitivity:
        _, _, _, s2 = wm.flood_from_db(before, after, t, min_drop_db, min_pixels, area)
        st["sensitivity"].append({"threshold_db": float(t), "flooded_ha": s2["area_ha"][2],
                                  "uncertain_ha": s2["area_ha"][4], "water_after_percent": s2["water_after_percent"]})
    say(90, "Writing files")
    os.makedirs(out_dir, exist_ok=True)
    prof.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    paths = {}
    for name, arr in (("flood_classes", cls), ("water_before", np.where(cls == 255, 255, wb)),
                      ("water_after", np.where(cls == 255, 255, wa))):
        p = os.path.join(out_dir, name + ".tif")
        with rasterio.open(p, "w", **prof) as d:
            d.write(arr.astype("uint8"), 1)
        paths[name] = p
    paths["summary"] = os.path.join(out_dir, "flood_summary.json")
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, default=float)
    text = (note + "\n" if note else "") + report_text(st, area, geographic)
    say(100, "Done")
    return {"stats": st, "paths": paths, "classes": cls, "report": text, "profile": prof}
