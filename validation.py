"""Independent accuracy assessment of any classified map (water, flood, change) with a stratified random sample.

Workflow
  1. sample_points   : draw random points inside each MAP class (stratum) -> points_to_label.csv / .geojson (no map class shown)
  2. an analyst labels every point BLIND (without seeing the map) using reference imagery  -> column ref_label
  3. evaluate        : confusion matrix + area-adjusted accuracy and area estimates with confidence intervals (Olofsson et al.)
The numerical core is pure NumPy; raster/CRS access is done in cli.py (rasterio) so everything here is unit-tested.
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import accuracy as acc


def parse_counts(spec: str) -> Dict[int, int]:
    """'1=100,2=100,3=50' -> {1: 100, 2: 100, 3: 50}"""
    out: Dict[int, int] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        m = re.fullmatch(r"\s*(\d+)\s*=\s*(\d+)\s*", part)
        if not m:
            raise ValueError(f"Bad --counts entry '{part}'. Use class=number, e.g. 1=100,2=100")
        out[int(m.group(1))] = int(m.group(2))
    if not out:
        raise ValueError("No classes given.")
    return out


def parse_groups(spec: str) -> Dict[str, List[int]]:
    """'flood:2;not_flood:0,1,3,4' -> {'flood': [2], 'not_flood': [0, 1, 3, 4]}  (map classes grouped into validation categories).
    '=' is accepted instead of ':' ('water=1;land=0'). Names may be anything (water, change, urban, class_3 ...)."""
    out: Dict[str, List[int]] = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        m = re.match(r"\s*([^:=]+?)\s*[:=]\s*(.*)$", part)
        if not m:
            raise ValueError(f"Bad group '{part}'. Use name:class,class;name2:class")
        name, vals = m.group(1), m.group(2)
        try:
            out[name] = [int(v) for v in vals.split(",") if v.strip()]
        except ValueError:
            raise ValueError(f"Bad class list in '{part}': use whole numbers separated by commas.")
        if not out[name]:
            raise ValueError(f"Category '{name}' has no map classes.")
    if len({normalize_label(n) for n in out}) != len(out):
        raise ValueError("Two categories have the same name (upper/lower case and spaces are ignored).")
    seen = [v for vs in out.values() for v in vs]
    if len(seen) != len(set(seen)):
        raise ValueError("A map class appears in more than one group.")
    if len(out) < 2:
        raise ValueError("Need at least two groups.")
    return out


def normalize_label(s: str) -> str:
    """'Not Flood ' -> 'not_flood': labels are compared ignoring case, surrounding spaces and space/hyphen/underscore."""
    return re.sub(r"[\s\-_]+", "_", (s or "").strip().lower())


# Ready-made category schemes (name -> groups spec). Any other scheme can be typed as 'name:class,class;name2:class'.
PRESETS: Dict[str, str] = {
    "Flood map (flood-db)": "flood:2;not_flood:0,1,3,4",
    "Flood map, 4 reference categories": "dry:0,4;permanent_water:1;flooded:2;receded:3",
    "Change map": "change:2,3;no_change:1",
    "Water mask (0 = land, 1 = water)": "water:1;land:0",
}


def groups_for_classes(classes: Sequence[int], names: Optional[Dict[int, str]] = None) -> str:
    """One category per map class, for any classified map (land cover ...): [1,2,3] -> 'class_1:1;class_2:2;class_3:3'."""
    names = names or {}
    return ";".join(f"{names.get(int(c), 'class_' + str(int(c)))}:{int(c)}" for c in classes)


SCHEME_FILE = "validation_scheme.json"


def write_scheme(out_dir: str, groups_spec: str, map_name: str = "") -> str:
    """Remember the category scheme next to the points, so labelling and accuracy use the same names without retyping.
    Holds only category names and which map classes they merge - never a map class of a point (labelling stays blind)."""
    groups = parse_groups(groups_spec)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, SCHEME_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"map": map_name, "groups": groups_spec, "categories": list(groups)}, f, indent=2)
    return path


def read_scheme(folder_or_csv: str) -> Optional[Dict]:
    """The scheme saved beside a points file (or in a folder); None if there is none or it is unreadable."""
    folder = folder_or_csv if os.path.isdir(folder_or_csv) else os.path.dirname(folder_or_csv)
    path = os.path.join(folder, SCHEME_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        parse_groups(d["groups"])
        return d
    except Exception:
        return None


def stratified_points(class_array: np.ndarray, counts: Dict[int, int], seed: int = 42, nodata: int = 255,
                      min_distance_px: int = 5) -> List[Dict]:
    """Random pixels inside each class. Points closer than min_distance_px are rejected so that samples are
    not neighbouring pixels of the same object. Returns dicts with row, col, map_class."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(class_array)
    pts: List[Dict] = []
    cell = max(1, int(min_distance_px))
    occupied: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for cls, n in counts.items():
        idx = np.flatnonzero((arr == cls) & (arr != nodata))
        if idx.size == 0:
            raise ValueError(f"Class {cls} has no pixels in the map.")
        order = rng.permutation(idx)
        got = 0
        for flat in order:
            r, c = divmod(int(flat), arr.shape[1])
            key = (r // cell, c // cell)
            near = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    for (rr, cc) in occupied.get((key[0] + dr, key[1] + dc), ()):
                        if (rr - r) ** 2 + (cc - c) ** 2 < cell * cell:
                            near = True
                            break
                    if near:
                        break
                if near:
                    break
            if near:
                continue
            occupied.setdefault(key, []).append((r, c))
            pts.append({"row": r, "col": c, "map_class": int(cls)})
            got += 1
            if got == n:
                break
        if got < n:
            raise ValueError(f"Only {got} of {n} points fit in class {cls} with the minimum distance of {cell} px; reduce the count or the distance.")
    # Shuffle before numbering: otherwise the id order (class 1 first, then class 2, ...) tells the analyst which map class a
    # point belongs to and the labelling is not blind.
    pts = [pts[i] for i in rng.permutation(len(pts))]
    for i, p in enumerate(pts, start=1):
        p["id"] = i
    return pts


def pixel_xy(transform6: Sequence[float], row: int, col: int) -> Tuple[float, float]:
    """Pixel-centre coordinates from an affine transform (a, b, c, d, e, f) as in rasterio's Affine."""
    a, b, c, d, e, f = transform6
    return c + a * (col + 0.5) + b * (row + 0.5), f + d * (col + 0.5) + e * (row + 0.5)


def write_sampling_files(points: List[Dict], lonlat: List[Tuple[float, float]], out_dir: str) -> Dict[str, str]:
    """points_to_label.csv (+ .geojson) WITHOUT the map class, and points_key.csv WITH it."""
    os.makedirs(out_dir, exist_ok=True)
    p_label = os.path.join(out_dir, "points_to_label.csv")
    p_key = os.path.join(out_dir, "points_key.csv")
    p_geo = os.path.join(out_dir, "points_to_label.geojson")
    with open(p_label, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "lon", "lat", "ref_label", "note"])
        for p, (lon, lat) in zip(points, lonlat):
            w.writerow([p["id"], f"{lon:.6f}", f"{lat:.6f}", "", ""])
    with open(p_key, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "row", "col", "map_class"])
        for p in points:
            w.writerow([p["id"], p["row"], p["col"], p["map_class"]])
    feats = [{"type": "Feature", "properties": {"id": p["id"], "ref_label": ""},
              "geometry": {"type": "Point", "coordinates": [lon, lat]}} for p, (lon, lat) in zip(points, lonlat)]
    with open(p_geo, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    p_kml = os.path.join(out_dir, "points_to_label.kml")
    write_kml(points, lonlat, p_kml)
    return {"to_label": p_label, "key": p_key, "geojson": p_geo, "kml": p_kml}


def write_kml(points: List[Dict], lonlat: List[Tuple[float, float]], path: str) -> str:
    """KML for Google Earth (Pro: use the Historical Imagery slider). Placemarks carry only the id - never the map class."""
    from xml.sax.saxutils import escape
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           "<name>Validation points to label</name>",
           "<description>Label each point from reference imagery (Historical Imagery slider) WITHOUT looking at the map. "
           "Write the label in the ref_label column of points_to_label.csv using the id shown here.</description>",
           '<Style id="p"><IconStyle><scale>0.9</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-blank.png</href></Icon>'
           "</IconStyle><LabelStyle><scale>0.8</scale></LabelStyle></Style>"]
    for p, (lon, lat) in zip(points, lonlat):
        out.append(f'<Placemark><name>{escape(str(p["id"]))}</name><styleUrl>#p</styleUrl>'
                   f'<description>id {escape(str(p["id"]))}</description>'
                   f"<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point></Placemark>")
    out.append("</Document></kml>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


def read_labels(labels_csv: str, key_csv: str) -> List[Tuple[int, str]]:
    """[(map_class, reference_label)] for labelled points; blank/'uncertain' labels are dropped (counted by the caller)."""
    key = {}
    with open(key_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key[int(r["id"])] = int(r["map_class"])
    rows = []
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["id"]), (r.get("ref_label") or "").strip()))
    missing = [i for i, _ in rows if i not in key]
    if missing:
        raise ValueError(f"{len(missing)} labelled ids are not in the key file (e.g. {missing[:3]}).")
    return [(key[i], lab) for i, lab in rows]


def evaluate(map_counts_by_class: Dict[int, int], groups: Dict[str, List[int]], samples: List[Tuple[int, str]],
             pixel_area: float = 1.0) -> Dict:
    """Area-adjusted accuracy of a map whose classes are merged into `groups`.

    The sample was drawn PER MAP CLASS, so every map class is kept as a stratum (own area weight, own sample size) and the
    groups are only unions of strata; pooling merged classes as one simple random sample would bias the result whenever the
    classes have different areas and the same number of points.

    map_counts_by_class : pixels per map class (nodata excluded)
    samples             : (map_class, reference_label) for every labelled point
    """
    names = list(groups)
    norm = {normalize_label(n): n for n in names}
    cls_to_group = {c: g for g, cs in groups.items() for c in cs}
    strata = sorted(c for c in cls_to_group if map_counts_by_class.get(c, 0) > 0)
    if not strata:
        raise ValueError("None of the classes in the categories occurs in the map.")
    k = len(names)
    n_hj = np.zeros((len(strata), k), dtype=int)
    excluded = 0
    unknown = set()
    per_class_total = {c: 0 for c in strata}
    for mc, lab in samples:
        if mc not in cls_to_group:
            continue
        if normalize_label(lab) in ("", "uncertain", "unknown", "?"):
            excluded += 1
            if mc in per_class_total:
                per_class_total[mc] += 1
            continue
        canon = norm.get(normalize_label(lab))
        if canon is None:
            unknown.add(lab)
            continue
        if mc in per_class_total:
            n_hj[strata.index(mc), names.index(canon)] += 1
            per_class_total[mc] += 1
    if unknown:
        raise ValueError(f"These ref_label values are not among the categories {names}: {sorted(unknown)}.\n"
                         "Either correct the labels in the CSV, or change the Categories so that they use the same names.")
    counts = np.array([map_counts_by_class[c] for c in strata], dtype=float)
    grp = np.array([names.index(cls_to_group[c]) for c in strata])
    res = acc.olofsson_grouped(n_hj, counts, grp, pixel_area)
    uncertain_by_class = {c: per_class_total[c] - int(n_hj[i].sum()) for i, c in enumerate(strata)}
    res["metrics"] = acc.grouped_metrics(n_hj, counts, grp)
    res.update(names=names, confusion_map_rows_ref_cols=n_hj, excluded_uncertain=excluded,
               row_labels=[f"class {c} ({cls_to_group[c]})" for c in strata], strata=strata,
               stratum_area=counts * pixel_area, uncertain_by_class=uncertain_by_class)
    return res


def format_report(res: Dict, area_unit: str = "ha") -> str:
    names = res["names"]; cm = res["confusion_map_rows_ref_cols"]
    rows = res.get("row_labels") or names
    w = max(12, max(len(n) for n in names) + 2)
    lw = max(w, max(len(r) for r in rows) + 2)
    lines = ["Confusion matrix of the SAMPLE (rows = map class = sampling stratum, columns = reference)",
             f"{'':<{lw}}" + "".join(f"{n:>{w}}" for n in names) + f"{'total':>{w}}" + f"{'uncertain':>{w}}"]
    unc = res.get("uncertain_by_class", {})
    strata = res.get("strata", [None] * len(rows))
    for i, n in enumerate(rows):
        lines.append(f"{n:<{lw}}" + "".join(f"{int(v):>{w}}" for v in cm[i]) + f"{int(cm[i].sum()):>{w}}"
                     + f"{unc.get(strata[i], 0):>{w}}")
    lo, hi = res["oa_ci95"]
    lines += ["", f"Overall accuracy (area-adjusted): {res['overall_accuracy'] * 100:.1f} %  (95 % CI {lo * 100:.1f} - {hi * 100:.1f} %)", ""]
    lines.append(f"{'class':<{w}}{'UA %':>16}{'PA %':>16}{'map area':>16}{'estimated area (95% CI)':>34}")
    for i, n in enumerate(names):
        a_lo, a_hi = res["area_ci95"][i]
        pa = res["producers_accuracy"][i]
        lines.append(f"{n:<{w}}{res['users_accuracy'][i] * 100:>9.1f} +-{res['users_se'][i] * res['z'] * 100:>4.1f}"
                     f"{pa * 100:>9.1f} +-{res['producers_se'][i] * res['z'] * 100:>4.1f}"
                     f"{res['area_map'][i]:>16,.1f}{res['area_estimated'][i]:>16,.1f}  ({a_lo:,.1f} - {a_hi:,.1f})")
    m = res.get("metrics")
    if m:
        lines += ["", "Area-adjusted agreement measures per category (95 % bootstrap interval; categories with no reference area are blank)",
                  f"{'category':<{w}}{'precision(UA)':>15}{'recall(PA)':>12}{'F1 / Dice':>22}{'IoU / Jaccard':>22}"]
        for i, n in enumerate(names):
            lo, hi = m["f1_ci95"][i]; ilo, ihi = m["iou_ci95"][i]
            lines.append(f"{n:<{w}}{res['users_accuracy'][i] * 100:>14.1f}%{res['producers_accuracy'][i] * 100:>11.1f}%"
                         f"{m['f1'][i] * 100:>10.1f} ({lo * 100:.1f}-{hi * 100:.1f})"
                         f"{m['iou'][i] * 100:>10.1f} ({ilo * 100:.1f}-{ihi * 100:.1f})")
        klo, khi = m["kappa_ci95"]
        lines += [f"Macro-F1 {m['macro_f1'] * 100:.1f} %   mean IoU {m['mean_iou'] * 100:.1f} %   area-adjusted Cohen's kappa {m['kappa']:.3f} "
                  f"({klo:.3f}-{khi:.3f})",
                  f"Disagreement (share of the mapped area): quantity {m['quantity_disagreement'] * 100:.1f} %, allocation "
                  f"{m['allocation_disagreement'] * 100:.1f} %   (sum = 100 - overall accuracy)"]
    lines += ["", f"Areas in {area_unit}. Uncertain/blank reference labels excluded: {res['excluded_uncertain']}.",
              "Estimators: stratified random sampling, strata = map classes (Olofsson et al. 2013, 2014). The interval reflects sampling error only,",
              "not errors of the reference interpretation or a temporal gap between the map and the reference imagery."]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ raster-based wrappers
MAX_CLASSES = 50


def _read_class_raster(path: str, nodata: Optional[int] = 255):
    """(int class array with -1 = no data, rasterio profile). Refuses continuous rasters (e.g. a change *magnitude*)."""
    import rasterio
    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError("The raster has no bands.")
        a = src.read(1)
        prof = dict(src.profile)
        file_nodata = src.nodata
    bad = np.zeros(a.shape, bool)
    if np.issubdtype(a.dtype, np.floating):
        bad |= ~np.isfinite(a)
        finite = a[~bad]
        if finite.size and not np.allclose(finite, np.round(finite)):
            raise ValueError("This raster holds continuous values (decimals), not classes. Choose the CLASS map, e.g. the "
                             "'..._classes.tif' written by Change Detection > Save results, not '..._magnitude.tif'.")
    if file_nodata is not None and np.isfinite(file_nodata):
        bad |= a == file_nodata
    if nodata is not None:
        bad |= a == nodata
    out = np.where(bad, -1, np.nan_to_num(a.astype(np.float64), nan=-1)).astype(np.int64)
    n_cls = len(np.unique(out[out >= 0]))
    if n_cls > MAX_CLASSES:
        raise ValueError(f"The raster has {n_cls} distinct values; a classified map should have only a few classes. "
                         "Choose the class map (not an image, magnitude or index layer).")
    return out, prof


def raster_classes(path: str, nodata: Optional[int] = 255) -> Dict[int, int]:
    """{class value: pixel count} of a classified raster (no-data excluded), sorted by class."""
    arr, _ = _read_class_raster(path, nodata)
    vals, counts = np.unique(arr[arr >= 0], return_counts=True)
    if vals.size == 0:
        raise ValueError("The raster contains no valid pixels.")
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _pixel_area_ha(prof: Dict, lat_mid: float = 0.0) -> float:
    t = prof["transform"]
    px = abs(t.a * t.e)
    crs = prof.get("crs")
    if crs is not None and getattr(crs, "is_geographic", False):
        import math
        return px * (111320.0 ** 2) * math.cos(math.radians(lat_mid)) / 10000.0
    return px / 10000.0


def make_sampling_from_raster(raster: str, out_dir: str, counts: Dict[int, int], seed: int = 42, nodata: int = 255,
                              min_distance_px: int = 5, progress=None, groups_spec: Optional[str] = None) -> Dict:
    """Stratified random points in a classified raster -> points_to_label.csv/.geojson (no class) and points_key.csv."""
    from rasterio.warp import transform as warp_transform
    if not counts:
        raise ValueError("Give the number of points per class, e.g. 1=100,2=100.")
    if progress:
        progress(10, "Reading the classified map")
    arr, prof = _read_class_raster(raster, nodata)
    if prof.get("crs") is None:
        raise ValueError("The map has no CRS, so point coordinates cannot be produced. Georeference it first.")
    if progress:
        progress(40, "Drawing random points")
    pts = stratified_points(arr, {int(k): int(v) for k, v in counts.items()}, seed, nodata=-1, min_distance_px=min_distance_px)
    t = prof["transform"]
    xs, ys = zip(*(pixel_xy((t.a, t.b, t.c, t.d, t.e, t.f), p["row"], p["col"]) for p in pts))
    lons, lats = warp_transform(prof["crs"], "EPSG:4326", list(xs), list(ys))
    if progress:
        progress(80, "Writing files")
    files = write_sampling_files(pts, list(zip(lons, lats)), out_dir)
    if groups_spec:
        files["scheme"] = write_scheme(out_dir, groups_spec, os.path.basename(raster))
    per_class: Dict[int, int] = {}
    for p in pts:
        per_class[p["map_class"]] = per_class.get(p["map_class"], 0) + 1
    if progress:
        progress(100, "Done")
    return {"n": len(pts), "per_class": per_class, "files": files}


def validate_from_files(raster: str, labels_csv: str, key_csv: str, groups: Dict[str, List[int]], out_dir: str,
                        nodata: int = 255, progress=None) -> Dict:
    """Area-adjusted accuracy from the labelled points; writes validation_report.txt and confusion_matrix.csv."""
    if progress:
        progress(10, "Reading the classified map")
    arr, prof = _read_class_raster(raster, nodata)
    vals, cnts = np.unique(arr[arr >= 0], return_counts=True)
    map_counts = {int(v): int(c) for v, c in zip(vals, cnts)}
    if progress:
        progress(40, "Reading the labelled points")
    samples = read_labels(labels_csv, key_csv)
    if not any(lab.strip() for _, lab in samples):
        raise ValueError("No point has a ref_label yet. Fill the ref_label column first.")
    lat_mid = 0.0
    try:
        t = prof["transform"]
        lat_mid = t.f + t.e * arr.shape[0] / 2.0
    except Exception:
        pass
    if progress:
        progress(70, "Computing accuracy")
    res = evaluate(map_counts, groups, samples, _pixel_area_ha(prof, lat_mid))
    report = format_report(res, "ha")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    cm_path = os.path.join(out_dir, "confusion_matrix.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["map class \\ reference"] + res["names"])
        for n, row in zip(res["row_labels"], res["confusion_map_rows_ref_cols"]):
            w.writerow([n] + [int(v) for v in row])
    if progress:
        progress(100, "Done")
    res.update(report=report, report_path=report_path, confusion_csv=cm_path)
    return res


def shuffle_points(labels_csv: str, key_csv: str, out_dir: str, seed: Optional[int] = None) -> Dict:
    """New blind copy of an existing sample: the same points in random order with new ids and EMPTY labels.
    Files written: points_to_label.csv/.geojson/.kml, points_key.csv and id_map.csv (new_id, old_id) so that two labelling
    passes of the same points can be compared. Use it when the original ids were ordered by map class."""
    rows = {}
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[int(r["id"])] = (float(r["lon"]), float(r["lat"]))
    key = {}
    with open(key_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key[int(r["id"])] = (int(r["row"]), int(r["col"]), int(r["map_class"]))
    ids = sorted(rows)
    if not ids or set(ids) != set(key):
        raise ValueError("The points file and the key file do not contain the same ids.")
    rng = np.random.default_rng(seed)
    order = [ids[i] for i in rng.permutation(len(ids))]
    pts, lonlat = [], []
    for new_id, old in enumerate(order, start=1):
        row, col, mc = key[old]
        pts.append({"id": new_id, "row": row, "col": col, "map_class": mc})
        lonlat.append(rows[old])
    files = write_sampling_files(pts, lonlat, out_dir)
    map_path = os.path.join(out_dir, "id_map.csv")
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["new_id", "old_id"])
        for new_id, old in enumerate(order, start=1):
            w.writerow([new_id, old])
    files["id_map"] = map_path
    sch = read_scheme(labels_csv)
    if sch:
        files["scheme"] = write_scheme(out_dir, sch["groups"], sch.get("map", ""))
    return {"n": len(pts), "files": files}


# --------------------------------------------------------------------------------------------------------------------
# Two-phase estimator: part of the sample was labelled automatically, only a subsample (all "uncertain" + a random
# audit of the "decided" ones) was labelled by hand.  Hand labels alone are NOT a random sample of the stratum, so they
# are re-weighted by cell (map class x automatic status) and the interval comes from a bootstrap of both phases.
# --------------------------------------------------------------------------------------------------------------------
def evaluate_two_phase(map_counts: Dict[int, int], groups: Dict[str, List[int]], points: List[Dict],
                       pixel_area: float = 1.0, n_boot: int = 2000, seed: int = 1) -> Dict:
    """points: dicts with map_class, status ('uncertain' | 'decided'), label (group name, or None if not labelled by hand).

    p(reference = j | map class h) = sum over status s of  (n_hs / n_h) * mean(hand labels in cell (h, s)).
    Needs at least one hand label in every non-empty cell."""
    names = list(groups)
    k = len(names)
    cls_to_group = {c: g for g, cs in groups.items() for c in cs}
    strata = sorted(c for c in cls_to_group if map_counts.get(c, 0) > 0)
    if not strata:
        raise ValueError("None of the classes in the categories occurs in the map.")
    area = np.array([map_counts[c] * pixel_area for c in strata], dtype=float)
    gi = np.array([names.index(cls_to_group[c]) for c in strata])
    excluded = 0
    cells: Dict[Tuple[int, str], Dict] = {}
    for p in points:
        h = p["map_class"]
        if h not in strata:
            continue
        c = cells.setdefault((h, p["status"]), {"n": 0, "hand": []})
        c["n"] += 1
        lab = p.get("label")
        if lab is None or str(lab).strip() == "":
            continue
        lab = str(lab).strip()
        if lab.lower() in ("uncertain", "unknown", "?"):
            excluded += 1
            continue
        if lab not in names:
            raise ValueError(f"Reference label '{lab}' is not one of the categories {names}.")
        c["hand"].append(names.index(lab))
    n_h = {h: sum(c["n"] for (hh, _), c in cells.items() if hh == h) for h in strata}
    for (h, s), c in cells.items():
        if c["n"] and not c["hand"]:
            raise ValueError(f"No hand label in map class {h} / {s} points ({c['n']} points): cannot weight them. "
                             "Label a few more of them by hand.")
    for h in strata:
        if n_h[h] == 0:
            raise ValueError(f"No sample point in map class {h}.")

    def p_matrix(counts, pools):
        P = np.zeros((len(strata), k))
        for i, h in enumerate(strata):
            for s in ("uncertain", "decided"):
                key = (h, s)
                if key not in cells or counts[key] == 0:
                    continue
                pool = pools[key]
                P[i] += counts[key] / n_h[h] * np.bincount(pool, minlength=k) / len(pool)
        return P

    def metrics(P):
        A = np.zeros((k, k))                       # area: map group (row) x reference (col)
        for i in range(len(strata)):
            A[gi[i]] += area[i] * P[i]
        tot = A.sum()
        row, col = A.sum(1), A.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ua = np.where(row > 0, np.diag(A) / row, np.nan)
            pa = np.where(col > 0, np.diag(A) / col, np.nan)
        return {"oa": float(np.trace(A) / tot), "ua": ua, "pa": pa, "area_ref": col, "A": A}

    counts0 = {key: c["n"] for key, c in cells.items()}
    pools0 = {key: np.array(c["hand"], dtype=int) for key, c in cells.items()}
    est = metrics(p_matrix(counts0, pools0))

    rng = np.random.default_rng(seed)
    oa_b, ua_b, pa_b, ar_b = [], [], [], []
    keys_by_h = {h: [(h, s) for s in ("uncertain", "decided") if (h, s) in cells] for h in strata}
    for _ in range(n_boot):
        counts: Dict[Tuple[int, str], int] = {}
        pools: Dict[Tuple[int, str], np.ndarray] = {}
        for h in strata:
            ks = keys_by_h[h]
            draw = rng.multinomial(n_h[h], [cells[x]["n"] / n_h[h] for x in ks])
            for x, d in zip(ks, draw):
                counts[x] = int(d)
                pl = pools0[x]
                pools[x] = rng.choice(pl, size=len(pl), replace=True)
        m = metrics(p_matrix(counts, pools))
        oa_b.append(m["oa"]); ua_b.append(m["ua"]); pa_b.append(m["pa"]); ar_b.append(m["area_ref"])

    def ci(a):
        a = np.array(a, dtype=float)
        return np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0)

    return {"names": names, "strata": strata, "oa": est["oa"], "oa_ci95": tuple(float(x) for x in ci(oa_b)),
            "ua": est["ua"], "ua_ci95": ci(ua_b), "pa": est["pa"], "pa_ci95": ci(pa_b),
            "area_ref": est["area_ref"], "area_ref_ci95": ci(ar_b),
            "map_area": np.array([area[gi == j].sum() for j in range(k)]),
            "cells": {key: {"n": c["n"], "hand": len(c["hand"])} for key, c in cells.items()},
            "excluded": excluded, "n_boot": n_boot}


def format_two_phase(res: Dict, area_unit: str = "ha") -> str:
    names = res["names"]
    pc = lambda x: f"{100 * x:5.1f}"
    lo, hi = res["oa_ci95"]
    lines = ["Reference: HAND-labelled subsample re-weighted to the full stratified sample (two-phase estimator).",
             "Cells (map class / automatic status): points in the sample -> labelled by hand"]
    for (h, s), c in sorted(res["cells"].items()):
        lines.append(f"  class {h} / {s:<9}: {c['n']:>4} -> {c['hand']:>4}")
    lines += ["", f"Overall accuracy (area-adjusted): {100 * res['oa']:.1f} %  (95 % bootstrap interval {100 * lo:.1f} - {100 * hi:.1f} %)", "",
              f"{'class':<14}{'UA %':>8}{'(95% interval)':>20}{'PA %':>8}{'(95% interval)':>20}{'map area':>12}{'est. ref area':>16}{'(95% interval)':>24}"]
    for j, n in enumerate(names):
        ul, uh = res["ua_ci95"][0][j], res["ua_ci95"][1][j]
        pl, ph = res["pa_ci95"][0][j], res["pa_ci95"][1][j]
        al, ah = res["area_ref_ci95"][0][j], res["area_ref_ci95"][1][j]
        lines.append(f"{n:<14}{pc(res['ua'][j]):>8}{f'({pc(ul)} - {pc(uh)})':>20}{pc(res['pa'][j]):>8}{f'({pc(pl)} - {pc(ph)})':>20}"
                     f"{res['map_area'][j]:>12,.1f}{res['area_ref'][j]:>16,.1f}{f'({al:,.0f} - {ah:,.0f})':>24}")
    lines += ["", f"Areas in {area_unit}. Points labelled 'uncertain' by hand are excluded: {res['excluded']}.",
              f"Interval: percentile bootstrap ({res['n_boot']} replicates) of both phases; it is conservative and reflects sampling error",
              "only, not errors of the visual reference interpretation or a temporal gap between map and imagery."]
    return "\n".join(lines)


def validate_two_phase_from_files(raster: str, points: List[Dict], groups: Dict[str, List[int]], out_dir: str,
                                  nodata: int = 255, n_boot: int = 2000) -> Dict:
    arr, prof = _read_class_raster(raster, nodata)
    vals, cnts = np.unique(arr[arr >= 0], return_counts=True)
    map_counts = {int(v): int(c) for v, c in zip(vals, cnts)}
    lat_mid = 0.0
    try:
        t = prof["transform"]
        lat_mid = t.f + t.e * arr.shape[0] / 2.0
    except Exception:
        pass
    res = evaluate_two_phase(map_counts, groups, points, _pixel_area_ha(prof, lat_mid), n_boot=n_boot)
    report = format_two_phase(res, "ha")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "validation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    res.update(report=report, report_path=path)
    return res


# --------------------------------------------------------------------------------------------------------------------
# Threshold-free view: ROC / precision-recall curves from a CONTINUOUS layer (SAR backscatter in dB, change magnitude, an index)
# --------------------------------------------------------------------------------------------------------------------
def weighted_curves(scores, positive, weights, low_is_positive: bool = False) -> Dict:
    """ROC and precision-recall curves with sampling weights (weight = stratum pixels / stratum points, so the curves describe
    the whole mapped area, not the sample). scores: continuous layer value at each labelled point; positive: bool reference;
    low_is_positive: True when LOW values mean the target (water has low dB). Returns arrays and AUC / AP / best-F1 threshold."""
    s = np.asarray(scores, dtype=float); y = np.asarray(positive, dtype=bool); w = np.asarray(weights, dtype=float)
    ok = np.isfinite(s) & np.isfinite(w)
    s, y, w = s[ok], y[ok], w[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        raise ValueError("Need reference points of both kinds (target and not target) to draw ROC / precision-recall curves.")
    key = -s if low_is_positive else s
    order = np.argsort(-key, kind="mergesort")
    key_s, y_s, w_s = key[order], y[order], w[order]
    last = np.r_[key_s[1:] != key_s[:-1], True]                    # one curve point per distinct threshold
    tp = np.cumsum(w_s * y_s)[last]; fp = np.cumsum(w_s * ~y_s)[last]
    P, N = float(tp[-1]), float(fp[-1])
    thr = (-key_s[last]) if low_is_positive else key_s[last]       # score at which "score <= thr" (low) / ">= thr" is called positive
    tpr = np.r_[0.0, tp / P]; fpr = np.r_[0.0, fp / N]
    prec = tp / (tp + fp); rec = tp / P
    f1 = 2 * prec * rec / np.where(prec + rec > 0, prec + rec, np.nan)
    auc = float((getattr(np, 'trapezoid', None) or np.trapz)(tpr, fpr))
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    i_f = int(np.nanargmax(f1)); i_y = int(np.argmax(tpr[1:] - fpr[1:]))
    return {"threshold": thr, "fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "average_precision": ap, "best_f1": float(f1[i_f]), "best_f1_threshold": float(thr[i_f]),
            "youden_threshold": float(thr[i_y]), "prevalence": P / (P + N), "n_points": int(len(s)),
            "low_is_positive": bool(low_is_positive)}


def validate_curves_from_files(score_raster: str, labels_csv: str, key_csv: str, map_raster: str, groups: Dict[str, List[int]],
                               positive: Sequence[str], out_dir: str, low_is_positive: bool = False, nodata: int = 255,
                               n_boot: int = 500, seed: int = 1) -> Dict:
    """ROC / PR curves of a continuous layer against the blind reference labels. The score raster must be on the SAME grid as the
    sampled map (same rows x cols; e.g. the sigma0-dB image the water / flood map came from). Writes curves.csv, curves.png and
    curves_report.txt. 'positive' = category names that count as the target (e.g. ['water'])."""
    import rasterio
    arr, _ = _read_class_raster(map_raster, nodata)
    vals, cnts = np.unique(arr[arr >= 0], return_counts=True)
    map_counts = {int(v): int(c) for v, c in zip(vals, cnts)}
    with rasterio.open(score_raster) as src:
        sc = src.read(1).astype("float64")
        snd = src.nodata
    if sc.shape != arr.shape:
        raise ValueError(f"The score raster is {sc.shape[1]} x {sc.shape[0]} px but the sampled map is {arr.shape[1]} x {arr.shape[0]} px. "
                         "Curves need the same grid: use the layer the map was made from (or resample it first).")
    if snd is not None and np.isfinite(snd):
        sc[sc == snd] = np.nan
    if np.isfinite(sc).any() and np.nanmin(sc) < 0:
        sc[sc == 0.0] = np.nan                                         # SNAP dB images use 0 for no data
    pos = {normalize_label(p) for p in positive}
    names = {normalize_label(n): n for n in groups}
    if not pos or not pos <= set(names):
        raise ValueError(f"Target categories {sorted(pos)} are not among {list(groups)}.")
    key = {}
    with open(key_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key[int(r["id"])] = (int(r["row"]), int(r["col"]), int(r["map_class"]))
    rows = []
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab = normalize_label(r.get("ref_label") or "")
            if lab in ("", "uncertain", "unknown", "?"):
                continue
            if lab not in names:
                raise ValueError(f"ref_label '{r.get('ref_label')}' is not among the categories {list(groups)}.")
            rows.append((int(r["id"]), lab))
    n_h: Dict[int, int] = {}
    for i, _ in rows:
        n_h[key[i][2]] = n_h.get(key[i][2], 0) + 1
    s, y, w = [], [], []
    for i, lab in rows:
        rr, cc, mc = key[i]
        if mc not in map_counts:
            continue
        s.append(sc[rr, cc]); y.append(lab in pos); w.append(map_counts[mc] / n_h[mc])
    cur = weighted_curves(s, y, w, low_is_positive)
    # bootstrap of AUC / AP (resample points inside each map class)
    rng = np.random.default_rng(seed)
    s_a, y_a, w_a = np.array(s), np.array(y), np.array(w)
    cls_a = np.array([key[i][2] for i, _ in rows if key[i][2] in map_counts])
    aucs, aps = [], []
    for _ in range(int(n_boot)):
        idx = np.concatenate([rng.choice(np.where(cls_a == c)[0], size=int((cls_a == c).sum())) for c in np.unique(cls_a)])
        try:
            c2 = weighted_curves(s_a[idx], y_a[idx], w_a[idx], low_is_positive)
            aucs.append(c2["auc"]); aps.append(c2["average_precision"])
        except ValueError:
            pass
    cur["auc_ci95"] = tuple(np.percentile(aucs, [2.5, 97.5])) if aucs else (float("nan"),) * 2
    cur["ap_ci95"] = tuple(np.percentile(aps, [2.5, 97.5])) if aps else (float("nan"),) * 2
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "curves.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["threshold", "precision", "recall", "f1", "fpr", "tpr"])
        for k in range(len(cur["threshold"])):
            wr.writerow([f"{cur['threshold'][k]:.4f}", f"{cur['precision'][k]:.4f}", f"{cur['recall'][k]:.4f}",
                         f"{cur['f1'][k]:.4f}", f"{cur['fpr'][k + 1]:.4f}", f"{cur['tpr'][k + 1]:.4f}"])
    rel = "<=" if low_is_positive else ">="
    rep = (f"ROC / precision-recall curves of '{os.path.basename(score_raster)}' for target = {', '.join(sorted(pos))}\n"
           f"{cur['n_points']} labelled points, area-weighted (weight = map-class pixels / points); target prevalence in the map area {cur['prevalence'] * 100:.1f} %\n"
           f"ROC AUC {cur['auc']:.3f} (95 % CI {cur['auc_ci95'][0]:.3f}-{cur['auc_ci95'][1]:.3f})   "
           f"average precision {cur['average_precision']:.3f} ({cur['ap_ci95'][0]:.3f}-{cur['ap_ci95'][1]:.3f})\n"
           f"Best F1 {cur['best_f1'] * 100:.1f} % at threshold {rel} {cur['best_f1_threshold']:.2f};  Youden-optimal threshold {rel} {cur['youden_threshold']:.2f}\n"
           "The threshold used to make the map should be close to these; if it is far, the map is not at its best operating point.\n"
           "Interval = bootstrap inside map classes; reference errors and a time gap between map and reference imagery are not included.")
    with open(os.path.join(out_dir, "curves_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    try:
        import matplotlib
        from matplotlib.figure import Figure
        fig = Figure(figsize=(12, 3.8), dpi=110)
        a1, a2, a3 = fig.subplots(1, 3)
        a1.plot(cur["fpr"], cur["tpr"]); a1.plot([0, 1], [0, 1], "k:", lw=0.8)
        a1.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC (AUC {cur['auc']:.3f})")
        a2.plot(cur["recall"], cur["precision"]); a2.axhline(cur["prevalence"], color="k", ls=":", lw=0.8)
        a2.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall (AP {cur['average_precision']:.3f})", ylim=(0, 1.02))
        a3.plot(cur["threshold"], cur["f1"]); a3.axvline(cur["best_f1_threshold"], color="r", ls="--", lw=0.8)
        a3.set(xlabel="Threshold on the layer", ylabel="F1", title=f"F1 vs threshold (best {cur['best_f1']:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "curves.png"))
    except Exception:
        pass
    cur["report"] = rep
    cur["files"] = {k: os.path.join(out_dir, v) for k, v in (("csv", "curves.csv"), ("png", "curves.png"), ("report", "curves_report.txt"))}
    return cur
