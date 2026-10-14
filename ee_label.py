"""Automatic reference labels for change-detection validation points from Google Earth Engine (Dynamic World).

Reads ``points_to_label.csv`` (id, lon, lat, ref_label, note), looks up the land cover at each point for two dates in the
Dynamic World V1 collection (GOOGLE/DYNAMICWORLD/V1, 10 m, Sentinel-2 based, classified by a different model) and writes a
labelled copy with ref_label = ``change`` / ``no_change`` / ``uncertain``.

IMPORTANT (scientific use)
  * Dynamic World is a PRODUCT with its own errors (it is itself derived from Sentinel-2), not field truth.  Report the result
    as "agreement with an independent land-cover product", and still check a manual subsample (e.g. 100 points) in Google Earth.
  * Desert cities are the hard case: DW often confuses 'built' with 'bare'.  Points where the two dates are not
    clearly one class (probability < --conf) are labelled uncertain, never guessed.

Usage (needs:  pip install earthengine-api   and   earthengine authenticate  once)
    python ee_label.py points_to_label.csv --date1 2020-02-10 --date2 2026-02-08 --project YOUR_GEE_PROJECT
    python ee_label.py points_to_label.csv --date1 ... --date2 ... --project P --key points_key.csv --raster map_classes.tif
  Then label points_to_label_manual_todo.csv by hand (Validation > 1b Label Points) and finish with
    python ee_label.py points_to_label.csv --merge points_to_label_manual_todo.csv [more_hand_files.csv] --key points_key.csv --raster map_classes.tif
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]
GROUPS = {
    "water": ["water"],
    "vegetation": ["trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub"],
    "built": ["built"],
    "bare": ["bare", "snow_and_ice"],
}
CHUNK = 400


def group_probs(p: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Sum the 9 Dynamic World probabilities into 4 groups; None if any value is missing (no image)."""
    if not p or any(p.get(b) is None for b in DW_BANDS):
        return None
    g = {name: sum(float(p[b]) for b in bands) for name, bands in GROUPS.items()}
    s = sum(g.values())
    return {k: v / s for k, v in g.items()} if s > 0 else None


def top(g: Dict[str, float]) -> Tuple[str, float]:
    k = max(g, key=g.get)
    return k, g[k]


def decide(p1: Optional[Dict[str, float]], p2: Optional[Dict[str, float]], conf: float = 0.5) -> Tuple[str, str]:
    """(label, note).  change: group differs and both dates >= conf;  no_change: same group, both >= conf;  else uncertain."""
    g1, g2 = group_probs(p1), group_probs(p2)
    if g1 is None or g2 is None:
        return "uncertain", "no Dynamic World image in the window"
    a, ca = top(g1)
    b, cb = top(g2)
    note = f"DW {a}({ca:.2f}) -> {b}({cb:.2f})"
    if ca < conf or cb < conf:
        return "uncertain", note + f" below conf {conf:g}"
    return ("no_change" if a == b else "change"), note


def _window(date: str, days: int) -> Tuple[str, str]:
    d = dt.date.fromisoformat(date)
    return (d - dt.timedelta(days=days)).isoformat(), (d + dt.timedelta(days=days + 1)).isoformat()


def fetch_probs(ee, rows: List[Dict], date: str, window_days: int, buffer_m: float) -> Tuple[int, Dict[int, Dict]]:
    """Mean Dynamic World probabilities inside a small buffer around every point, for images within +-window_days of date."""
    start, end = _window(date, window_days)
    bounds = ee.Geometry.MultiPoint([[r["lon"], r["lat"]] for r in rows])
    col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate(start, end).filterBounds(bounds)
    n_img = int(col.size().getInfo())
    out: Dict[int, Dict] = {}
    if n_img == 0:
        return 0, out
    img = col.select(DW_BANDS).mean()
    for i in range(0, len(rows), CHUNK):
        part = rows[i:i + CHUNK]
        feats = [ee.Feature(ee.Geometry.Point([r["lon"], r["lat"]]).buffer(buffer_m), {"pid": int(r["id"])}) for r in part]
        res = img.reduceRegions(collection=ee.FeatureCollection(feats), reducer=ee.Reducer.mean(), scale=10).getInfo()
        for f in res.get("features", []):
            pr = f["properties"]
            out[int(pr["pid"])] = {b: pr.get(b) for b in DW_BANDS}
    return n_img, out


def label_rows(rows: List[Dict], probs1: Dict[int, Dict], probs2: Dict[int, Dict], conf: float = 0.5,
               overwrite: bool = False) -> Counter:
    """Fill ref_label/note in place (never replaces a human label unless overwrite)."""
    cnt: Counter = Counter()
    for r in rows:
        if r.get("ref_label") and not overwrite:
            cnt["kept (already labelled)"] += 1
            continue
        lab, note = decide(probs1.get(r["id"]), probs2.get(r["id"]), conf)
        r["ref_label"] = lab
        r["note"] = (r.get("note") + "; " if r.get("note") else "") + note
        cnt[lab] += 1
    return cnt


def uncertain_summary(rows: List[Dict], probs1: Dict[int, Dict], probs2: Dict[int, Dict]) -> str:
    """Which (date1 class -> date2 class) pairs the uncertain points have: shows whether the doubt is built-vs-bare etc."""
    c: Counter = Counter()
    for r in rows:
        if r["ref_label"] != "uncertain":
            continue
        g1, g2 = group_probs(probs1.get(r["id"])), group_probs(probs2.get(r["id"]))
        c[("no data",) if g1 is None or g2 is None else (top(g1)[0], top(g2)[0])] += 1
    return "\n".join(f"  {' -> '.join(k)}: {v}" for k, v in c.most_common(8))


def pick_audit(rows: List[Dict], n: int, seed: int = 1) -> List[int]:
    """Random decided (change/no_change) point ids to re-label by hand, to measure how well the automatic label agrees."""
    import random
    ids = sorted(r["id"] for r in rows if r["ref_label"] in ("change", "no_change"))
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def manual_todo(rows: List[Dict], audit_ids: List[int]) -> List[Dict]:
    """Rows to label by hand: every uncertain point + the audit points.  Blank label and NO automatic hint (blind)."""
    audit = set(audit_ids)
    return [dict(id=r["id"], lon=r["lon"], lat=r["lat"], ref_label="", note="")
            for r in rows if r["ref_label"] == "uncertain" or r["id"] in audit]


def merge_labels(ee_rows: List[Dict], manual_rows: List[Dict], audit_ids: List[int]) -> Tuple[List[Dict], Dict]:
    """Final label = hand label where one was typed, else the automatic one.  Also agreement on the audit points."""
    hand = {r["id"]: r["ref_label"] for r in manual_rows if r["ref_label"]}
    audit = set(audit_ids)
    final, src = [], Counter()
    tab: Counter = Counter()
    for r in ee_rows:
        out = dict(r)
        h = hand.get(r["id"])
        if h:
            out["ref_label"] = h
            out["note"] = (r.get("note", "") + "; " if r.get("note") else "") + "hand label"
            src["hand"] += 1
            if r["id"] in audit and r["ref_label"] in ("change", "no_change"):
                tab[(r["ref_label"], h)] += 1
        else:
            src["automatic"] += 1
        final.append(out)
    n = sum(tab.values())
    agree = sum(v for (a, b), v in tab.items() if a == b)
    return final, dict(sources=dict(src), audit_n=n, audit_agree=agree, table=dict(tab))


def format_audit(st: Dict) -> str:
    if not st["audit_n"]:
        return "Audit: no audit point has a hand label yet."
    labs = ["change", "no_change", "uncertain"]
    lines = [f"Audit of the automatic labels ({st['audit_n']} points labelled by hand): agreement {st['audit_agree']}/{st['audit_n']} "
             f"= {100.0 * st['audit_agree'] / st['audit_n']:.0f} %",
             "automatic \\ hand | " + " | ".join(f"{l:>9}" for l in labs)]
    for a in ("change", "no_change"):
        lines.append(f"{a:>16} | " + " | ".join(f"{st['table'].get((a, l), 0):>9}" for l in labs))
    return "\n".join(lines)


def crosstab(rows: List[Dict], key_csv: str) -> str:
    key = {}
    with open(key_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key[int(r["id"])] = int(r["map_class"])
    labs = ["change", "no_change", "uncertain"]
    t: Dict[int, Counter] = {}
    for r in rows:
        t.setdefault(key[r["id"]], Counter())[r["ref_label"]] += 1
    lines = ["map class | " + " | ".join(f"{l:>9}" for l in labs) + " |  n"]
    for c in sorted(t):
        lines.append(f"{c:>9} | " + " | ".join(f"{t[c][l]:>9}" for l in labs) + f" | {sum(t[c].values()):>3}")
    return "\n".join(lines)


def run_merge(a, labeler, base: str) -> int:
    ee_rows, fields = labeler.read_points(base + "_ee.csv")
    hand_rows = []
    for p in a.merge:
        hand_rows += labeler.read_points(p)[0]
    with open(base + "_ee_audit_ids.csv", newline="", encoding="utf-8") as f:
        audit = [int(r["id"]) for r in csv.DictReader(f)]
    final, st = merge_labels(ee_rows, hand_rows, audit)
    out = base + "_final.csv"
    labeler.write_points(out, final, fields)
    print("Labels from:", st["sources"])
    print(format_audit(st))
    print("Saved:", out)
    handled = {r["id"] for r in hand_rows if r["ref_label"]}
    rest = [dict(id=r["id"], lon=r["lon"], lat=r["lat"], ref_label="", note="") for r in ee_rows if r["id"] not in handled]
    if rest:
        rest_path = base + "_manual_rest.csv"
        labeler.write_points(rest_path, rest, ["id", "lon", "lat", "ref_label", "note"])
        print(f"{len(rest)} points still rely on the automatic label. To make the reference fully visual, label {rest_path} "
              "in the app and run --merge again with BOTH hand-label files.")
    if a.hand_only:
        if not (a.raster and a.key):
            raise SystemExit("--hand-only needs --key and --raster")
        import validation
        key = {}
        with open(a.key, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key[int(r["id"])] = int(r["map_class"])
        hand = {r["id"]: r["ref_label"] for r in hand_rows if r["ref_label"]}
        pts = [dict(map_class=key[r["id"]], status="uncertain" if r["ref_label"] == "uncertain" else "decided",
                    label=hand.get(r["id"])) for r in ee_rows]
        res = validation.validate_two_phase_from_files(
            a.raster, pts, validation.parse_groups(a.groups),
            os.path.join(os.path.dirname(os.path.abspath(out)), "hand_only_validation"))
        print(res["report"])
        print("Saved:", res["report_path"])
        return 0
    if a.raster and a.key:
        import validation
        res = validation.validate_from_files(a.raster, out, a.key, validation.parse_groups(a.groups),
                                             os.path.join(os.path.dirname(os.path.abspath(out)), "final_validation"))
        print(res["report"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("points", help="points_to_label.csv")
    ap.add_argument("--date1", help="EARLIER image date, YYYY-MM-DD")
    ap.add_argument("--date2", help="LATER image date, YYYY-MM-DD")
    ap.add_argument("--project", help="Google Cloud / Earth Engine project id")
    ap.add_argument("--window", type=int, default=45, help="days before/after each date to average Dynamic World (default 45)")
    ap.add_argument("--buffer", type=float, default=15.0, help="buffer radius in metres around each point (default 15)")
    ap.add_argument("--conf", type=float, default=0.5, help="minimum group probability at BOTH dates (default 0.5)")
    ap.add_argument("--out", help="output csv (default: <points>_ee.csv)")
    ap.add_argument("--overwrite", action="store_true", help="also replace labels you already typed")
    ap.add_argument("--key", help="points_key.csv: print a class x label table")
    ap.add_argument("--raster", help="change map *_classes.tif: also compute the accuracy report (needs --key)")
    ap.add_argument("--audit", type=int, default=40, help="also hand-label this many randomly chosen decided points (default 40)")
    ap.add_argument("--merge", metavar="MANUAL_CSV", nargs="+", help="after labelling <points>_manual_todo.csv by hand: merge, report audit agreement, compute accuracy")
    ap.add_argument("--hand-only", action="store_true",
                    help="with --merge: use ONLY the hand labels (re-weighted two-phase estimator, bootstrap interval)")
    ap.add_argument("--groups", default="change:2,3;no_change:1", help="map-class groups for the accuracy report")
    a = ap.parse_args(argv)

    if not a.merge and not (a.date1 and a.date2):
        ap.error("--date1 and --date2 are required")
    import labeler
    base = os.path.splitext(a.points)[0]
    if a.merge:
        return run_merge(a, labeler, base)
    rows, fields = labeler.read_points(a.points)
    import ee
    try:
        ee.Initialize(project=a.project) if a.project else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=a.project) if a.project else ee.Initialize()
    n1, p1 = fetch_probs(ee, rows, a.date1, a.window, a.buffer)
    n2, p2 = fetch_probs(ee, rows, a.date2, a.window, a.buffer)
    print(f"Dynamic World images covering the points: {n1} around {a.date1}, {n2} around {a.date2}")
    cnt = label_rows(rows, p1, p2, a.conf, a.overwrite)
    out = a.out or os.path.splitext(a.points)[0] + "_ee.csv"
    labeler.write_points(out, rows, fields)
    print("Labels:", dict(cnt))
    print("Saved:", out)
    print("Uncertain points by (class at date1 -> class at date2):")
    print(uncertain_summary(rows, p1, p2))
    audit = pick_audit(rows, a.audit)
    todo = manual_todo(rows, audit)
    todo_path, audit_path = base + "_manual_todo.csv", base + "_ee_audit_ids.csv"
    labeler.write_points(todo_path, todo, ["id", "lon", "lat", "ref_label", "note"])
    with open(audit_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id"]); w.writerows([[i] for i in audit])
    print(f"Hand-label list: {todo_path}  ({len(todo)} points = {sum(1 for r in rows if r['ref_label']=='uncertain')} uncertain + {len(audit)} audit)")
    if a.key:
        print(crosstab(rows, a.key))
    if a.raster:
        if not a.key:
            ap.error("--raster needs --key")
        import validation
        res = validation.validate_from_files(a.raster, out, a.key, validation.parse_groups(a.groups),
                                             os.path.join(os.path.dirname(os.path.abspath(out)), "ee_validation"))
        print(open(res["report_path"], encoding="utf-8").read() if isinstance(res, dict) and "report_path" in res else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
