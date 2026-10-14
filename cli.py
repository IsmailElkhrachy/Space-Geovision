"""Headless command line for Space GeoVision - reproduces every experiment without the GUI.

  python cli.py sar-preprocess  S1A_..._GRDH_....zip  out/  --pol VV --subset "POLYGON((...))"
  python cli.py water           out/S1A_..._VV_dB.tif out/water.tif --method otsu [--reference ref_water.tif]
  python cli.py classify        scene.tif train.gpkg class out/ --model rf [--validation val.gpkg]
  python cli.py change          t1.tif t2.tif out/ --k 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _subset_from_args(a):
    """Study area from the command line: --subset WKT | --bbox | --center + --radius-km | (nothing = whole scene)."""
    import sar_snap
    if a.subset:
        spec = sar_snap.SubsetSpec("polygon", wkt=a.subset)
    elif a.bbox:
        spec = sar_snap.SubsetSpec("bbox", bbox=tuple(a.bbox))
    elif a.center:
        if a.radius_km is None:
            raise sar_snap.SnapError("--center needs --radius-km.")
        spec = sar_snap.SubsetSpec("center", center=tuple(a.center), radius_km=a.radius_km)
    else:
        spec = sar_snap.SubsetSpec("full")
    wkt = sar_snap.resolve_subset(spec, a.product)
    px = sar_snap.estimate_output_pixels(wkt, a.product, a.spacing)
    print("Study area:", wkt if wkt else "whole scene (bounds detected automatically)")
    print("Size      :", sar_snap.size_advice(px))
    return wkt


def cmd_sar(a):
    import sar_snap
    a.subset = _subset_from_args(a)
    p = sar_snap.SnapParams(polarisation=a.pol, speckle_filter=a.speckle, speckle_window=a.window, dem_name=a.dem,
                            external_dem=a.external_dem, pixel_spacing_m=a.spacing, subset_wkt=a.subset,
                            map_projection=a.projection, subset_mode=a.subset_mode, remove_thermal_noise=not a.no_thermal)
    gpt = sar_snap.find_gpt(a.gpt)
    if a.verify:
        for w in sar_snap.verify_parameter_names(gpt, p):
            print("WARNING:", w, file=sys.stderr)
    p.subset_stage = a.subset_at
    runner = sar_snap.run_stepwise if a.stepwise else sar_snap.run_full_chain
    res = runner(a.product, a.out, p, gpt=gpt, java_heap_gb=a.heap_gb,
                 progress=lambda v, m: print(f"\r{m}: {v}%".ljust(70), end="", flush=True))
    print()
    data, _ = sar_snap.read_geotiff(res.geotiff)
    for w in sar_snap.sanity_check_db(data):
        print("WARNING:", w, file=sys.stderr)
    print("GeoTIFF:", res.geotiff)
    print("Provenance:", sar_snap.write_provenance(res, gpt, a.product))


def cmd_snapcheck(a):
    """Check the SNAP installation and the parameter names we send. Needs no data."""
    import sar_snap
    gpt = sar_snap.find_gpt(a.gpt)
    print("gpt found      :", gpt)
    print("SNAP version   :", sar_snap.snap_version(gpt))
    print("gpt -h         :", sar_snap.check_gpt(gpt))
    warns = sar_snap.verify_parameter_names(gpt, sar_snap.SnapParams())
    if warns:
        print("\nPARAMETER WARNINGS (send these lines back so the code can be adjusted):")
        for w in warns:
            print(" -", w)
        return 2
    print("\nAll parameter names used by Space GeoVision exist in this SNAP version.")


def cmd_footprint(a):
    """Where on Earth is this Sentinel-1 product? Prints the footprint and a ready-to-use subset."""
    import sar_snap
    info = sar_snap.parse_product_name(a.product)
    pts = sar_snap.read_footprint(a.product)
    x0, y0, x1, y1 = sar_snap.footprint_bbox(pts)
    print(f"Product      : {os.path.basename(a.product)}")
    print(f"Acquired     : {info.start} UTC   Polarisations: {', '.join(info.polarisations)}")
    print(f"Footprint    : lon {x0:.3f} .. {x1:.3f}   lat {y0:.3f} .. {y1:.3f}")
    print("Vertices     :", "; ".join(f"{lo:.3f},{la:.3f}" for lo, la in pts))
    center = a.center or ((x0 + x1) / 2, (y0 + y1) / 2)
    if not a.center:
        print("\n(No --center given: using the middle of the scene. Pick your own flood area for the paper.)")
    if center:
        lon, lat = center
        h = a.size_km / 2 / 111.0
        w = h / max(0.1, __import__("math").cos(__import__("math").radians(lat)))
        wkt = f"POLYGON(({lon-w:.4f} {lat-h:.4f}, {lon+w:.4f} {lat-h:.4f}, {lon+w:.4f} {lat+h:.4f}, {lon-w:.4f} {lat+h:.4f}, {lon-w:.4f} {lat-h:.4f}))"
        print(f"\nSubset {a.size_km:g} km around lon={lon}, lat={lat}:\n  --subset \"{wkt}\"")
        frac = sar_snap.check_subset_in_footprint(wkt, a.product)
        if frac >= 0.999:
            print("  (fully inside the product footprint)")
        else:
            print(f"  WARNING: only {frac*100:.0f}% of the sample points of this box are inside the footprint; "
                  "part of the area has no data in this scene. Move the centre or reduce --size-km.")


def cmd_scenes(a):
    """Compare several Sentinel-1 products: tracks, pass direction, footprints and their common area."""
    import sar_snap
    rows = []
    for path in a.products:
        info = sar_snap.parse_product_name(path)
        pts = sar_snap.read_footprint(path)
        rows.append((path, info, sar_snap.footprint_bbox(pts), sar_snap.read_pass_direction(path)))
    rows.sort(key=lambda r: r[1].start)
    print(f"{'acquired (UTC)':<17}{'abs.orbit':>10}{'rel.orbit':>10}  {'pass':<11}{'datatake':<9}footprint lon / lat")
    for path, i, (x0, y0, x1, y1), pas in rows:
        print(f"{i.start:<17}{i.abs_orbit or '?':>10}{i.relative_orbit or '?':>10}  {pas or '?':<11}{i.datatake or '?':<9}"
              f"{x0:.2f}..{x1:.2f} / {y0:.2f}..{y1:.2f}   {os.path.basename(path)[-24:]}")
    if getattr(a, "point", None):
        pts_by = {p: sar_snap.read_footprint(p) for p, *_ in rows}
        print()
        head = f"{'place':<14}" + "".join(f"{i.start[4:8]}/{i.relative_orbit}{'':<2}" for _, i, *_ in rows)
        print("Which scene covers which place (inside the real footprint polygon):\n" + head)
        for name, lon, lat in a.point:
            marks = "".join(f"{'YES' if sar_snap.footprint_contains(pts_by[p], float(lon), float(lat)) else '-':<10}" for p, *_ in rows)
            print(f"{name:<14}{marks}  ({float(lon):.2f}E {float(lat):.2f}N)")
        print("Column = month-day / relative orbit. Coordinates are approximate city centres.")
    gx0 = max(r[2][0] for r in rows); gy0 = max(r[2][1] for r in rows)
    gx1 = min(r[2][2] for r in rows); gy1 = min(r[2][3] for r in rows)
    print()
    if gx0 < gx1 and gy0 < gy1:
        print(f"Bounding boxes of ALL scenes overlap in: lon {gx0:.3f}..{gx1:.3f}, lat {gy0:.3f}..{gy1:.3f}")
        print(f"  centre for a subset: --center {(gx0+gx1)/2:.3f} {(gy0+gy1)/2:.3f}")
        print("  (bounding-box test only: confirm on a map, footprints are slanted)")
    else:
        print("The scenes do NOT share a common area (bounding boxes). Choose scenes that cover the same study area.")
    tracks = {}
    for path, i, *_ in rows:
        tracks.setdefault(i.relative_orbit, []).append(i.start[:8])
    if len(tracks) > 1:
        print("\nDifferent relative orbits (different viewing geometry): " + "; ".join(f"track {k}: {', '.join(v)}" for k, v in tracks.items()))
        print("Compare water extents scene by scene; do not difference backscatter across tracks without care.")
    for path, i, *_ in rows:
        if i.abs_orbit:
            print(f"Same track as {i.start[:8]} (rel. {i.relative_orbit}): previous cycle abs.orbit {i.abs_orbit-175}, next {i.abs_orbit+175}"
                  f"  (~{(i.start[:8])} -/+ 12 days)")


def cmd_probe(a):
    """Find a configuration SNAP accepts, WITHOUT running the long processing (each trial stops after init)."""
    import dataclasses
    import sar_snap
    info = sar_snap.parse_product_name(a.product)
    gpt = sar_snap.find_gpt(a.gpt)
    if a.subset:
        sar_snap.check_subset_in_footprint(a.subset, a.product)
    os.makedirs(a.out, exist_ok=True)
    base = sar_snap.SnapParams(polarisation=a.pol, subset_wkt=a.subset, speckle_window=a.window,
                               dem_name=a.dem, pixel_spacing_m=a.spacing)
    print(f"SNAP {sar_snap.snap_version(gpt)}  product {os.path.basename(a.product)[:40]}...")
    for n, (name, over) in enumerate(sar_snap.PROBE_VARIANTS, 1):
        if over["subset_mode"] == "after" and not a.include_slow:
            print(f"[{n}] {name}: skipped (use --include-slow)")
            continue
        params = dataclasses.replace(base, **over)
        gp = os.path.join(a.out, f"probe_{n}.xml")
        open(gp, "w", encoding="utf-8").write(sar_snap.build_graph_xml(info.path, os.path.join(a.out, f"probe_{n}.tif"), params))
        print(f"[{n}] {name} ...", end=" ", flush=True)
        r = sar_snap.probe_graph_init(gpt, gp, name, a.wait)
        print(f"{r.status.upper()} ({r.seconds:.0f} s)  {r.detail[:160]}")
        if r.status == "init-ok":
            proj = f'"{over["map_projection"]}"'
            print("\nUse this configuration:\n  python cli.py sar-preprocess <product> <out> --pol", a.pol,
                  "--subset", f'"{a.subset}"' if a.subset else "...", "--projection", proj, "--subset-mode", over["subset_mode"])
            return 0
    print("\nNo variant passed initialisation. Send the output above.")
    return 2


def cmd_flood(a):
    """Flood extent = water after AND NOT water before (masks produced by `water` with identical settings)."""
    import math
    import numpy as np, rasterio
    import change_detection as cd, water_mapping as wm
    from rasterio.warp import reproject, Resampling
    with rasterio.open(a.before) as sb, rasterio.open(a.after) as sa:
        mb = sb.read(1)
        try:
            cd.check_alignment(sb.profile, sa.profile)
            ma = sa.read(1)
        except cd.AlignmentError as exc:
            # Two SNAP runs never give exactly the same pixel grid: resample the AFTER mask onto the BEFORE grid
            # (nearest neighbour keeps the classes; the half-pixel shift is far below SAR geolocation accuracy).
            print(f"Grids differ ({exc.args[0][:70]}...): resampling the 'after' mask onto the 'before' grid (nearest neighbour).")
            ma = np.full(mb.shape, 255, dtype="uint8")
            reproject(rasterio.band(sa, 1), ma, src_transform=sa.transform, src_crs=sa.crs, src_nodata=255,
                      dst_transform=sb.transform, dst_crs=sb.crs, dst_nodata=255, resampling=Resampling.nearest)
        prof = sb.profile.copy(); tr = sb.transform; geographic = sb.crs.is_geographic
        lat = (sb.bounds.top + sb.bounds.bottom) / 2
    valid = (mb != 255) & (ma != 255)
    if geographic:       # pixel area from degrees at the scene's mid-latitude (approximation, stated in the report)
        area = abs(tr.a) * 111320 * math.cos(math.radians(lat)) * abs(tr.e) * 110574
    else:
        area = abs(tr.a * tr.e)
    cls, st = wm.flood_extent(mb == 1, ma == 1, valid, area)
    prof.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(a.out_tif, "w", **prof) as d:
        d.write(cls, 1)
    print(f"Valid pixels: {st['valid_pixels']:,}   pixel area ~ {area:.1f} m2"
          + (" (approximate, from degrees)" if geographic else ""))
    for k in (0, 1, 2, 3):
        print(f"  {k} {wm.FLOOD_CLASSES[k]:<32}{st['counts'][k]:>12,}  {st['percent'][k]:6.2f} %  {st['area_ha'][k]:>10,.1f} ha")
    print("Output:", a.out_tif)


def cmd_flood_db(a):
    """Flood extent from the BEFORE and AFTER sigma0-dB GeoTIFFs of the same Sentinel-1 track."""
    import flood
    res = flood.run_flood_db(a.before_db, a.after_db, a.out, a.threshold, a.min_drop_db, a.min_pixels, a.sensitivity_db,
                             progress=lambda v, m: print(f"\r{m}: {v}%".ljust(60), end="", flush=True))
    print()
    print(res["report"])
    print("\nOutputs:", *res["paths"].values(), sep="\n  ")


def cmd_sample_points(a):
    """Stratified random points inside each class of a classified map, for BLIND labelling by an analyst."""
    import validation as val
    res = val.make_sampling_from_raster(a.raster, a.out, val.parse_counts(a.counts), a.seed, a.nodata, a.min_distance_px,
                                        groups_spec=a.groups)
    print(f"{res['n']} points written {res['per_class']}")
    print("Open in QGIS (Layer > Add Delimited Text, x = lon, y = lat, EPSG:4326) or load the GeoJSON, then fill the column ref_label")
    print("with the category you see on reference imagery WITHOUT looking at the map. Use 'uncertain' when you cannot decide.")
    print("Files:", *res["files"].values(), sep="\n  ")


def cmd_validate_curves(a):
    import validation as val
    spec = a.groups
    if not spec:
        sch = val.read_scheme(a.labels) or val.read_scheme(a.key)
        if not sch:
            raise SystemExit("No --groups given and no validation_scheme.json next to the points.")
        spec = sch["groups"]
    res = val.validate_curves_from_files(a.score, a.labels, a.key, a.raster, val.parse_groups(spec),
                                         [s.strip() for s in a.target.split(",") if s.strip()], a.out,
                                         low_is_positive=a.low_is_target, nodata=a.nodata)
    print(res["report"]); print("\nFiles:", *res["files"].values(), sep="\n  ")


def cmd_validate_points(a):
    import validation as val
    spec = a.groups
    if not spec:
        sch = val.read_scheme(a.labels) or val.read_scheme(a.key)
        if not sch:
            raise SystemExit("No --groups given and no validation_scheme.json next to the points. Give --groups, e.g. \"water:1;land:0\".")
        spec = sch["groups"]
        print("Categories from validation_scheme.json:", spec)
    res = val.validate_from_files(a.raster, a.labels, a.key, val.parse_groups(spec), a.out, a.nodata)
    print(res["report"])
    print("\nReport:", res["report_path"])


def cmd_water(a):
    import numpy as np, rasterio
    import accuracy, water_mapping as wm
    with rasterio.open(a.db_tif) as s:
        db = s.read(1).astype("float32"); prof = s.profile.copy()
        if s.nodata is not None:
            db[db == s.nodata] = np.nan
    zero_frac = float(np.mean(db == 0.0))
    mask, info = wm.map_water(db, a.method, a.lower, a.upper, a.min_pixels, threshold=a.threshold)
    if a.sensitivity:
        info["sensitivity"] = wm.threshold_sensitivity(db, min_pixels=a.min_pixels)
    info["no_data_fraction_percent"] = 100.0 * float(np.mean(~np.isfinite(db) | (db == 0.0)))
    prof.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(a.out_tif, "w", **prof) as d:
        d.write(np.where(np.isfinite(db) & (db != 0.0), mask, 255).astype("uint8"), 1)
    if a.reference:
        with rasterio.open(a.reference) as r:
            ref = r.read(1) > 0
        info["validation"] = wm.validate_water(mask, ref, np.isfinite(db))
    json.dump(info, open(os.path.splitext(a.out_tif)[0] + "_water.json", "w"), indent=2, default=float)
    print(json.dumps(info, indent=2, default=float))


def cmd_classify(a):
    import supervised
    cfg = supervised.ClassifierConfig(model=a.model, n_trees=a.trees, block_px=a.block)
    res = supervised.run_supervised(a.raster, a.train, a.field, a.out, cfg, a.validation)
    print(res["report"]); print("\nFiles:", res["classified_tif"], res["confusion_csv"], res["report_txt"], sep="\n  ")


def cmd_change(a):
    import numpy as np, rasterio
    import change_detection as cd
    with rasterio.open(a.t1) as s1, rasterio.open(a.t2) as s2:
        p1, p2 = s1.profile, s2.profile
        cd.check_alignment(p1, p2)
        x1, x2 = s1.read().astype("float32"), s2.read().astype("float32")
        t = s1.transform
        area = abs(t.a * t.e) if not s1.crs.is_geographic else None
    res = cd.detect_changes(x1, x2, normalise=not a.no_normalise, method=a.method, k=a.k,
                            nodata=p1.get("nodata"), pixel_area_m2=area)
    os.makedirs(a.out, exist_ok=True)
    for name, arr, dt, nd in (("magnitude", res.magnitude, "float32", -9999.0), ("classes", cd.export_classes(res.classes), "uint8", 255)):
        prof = {k: v for k, v in p1.items() if k in ("crs", "transform", "width", "height")}
        prof.update(driver="GTiff", count=1, dtype=dt, nodata=nd, compress="lzw")
        with rasterio.open(os.path.join(a.out, f"change_{name}.tif"), "w", **prof) as d:
            d.write(np.where(np.isfinite(arr), arr, nd).astype(dt), 1)
    text = cd.format_report(res)
    open(os.path.join(a.out, "change_report.txt"), "w", encoding="utf-8").write(text)
    print(text)


def cmd_change_sensitivity(a):
    import numpy as np, rasterio
    import change_detection as cd
    with rasterio.open(a.t1) as s1, rasterio.open(a.t2) as s2:
        cd.check_alignment(s1.profile, s2.profile)
        x1, x2 = s1.read().astype("float32"), s2.read().astype("float32"); nodata = s1.nodata
    print(cd.format_sensitivity(cd.change_sensitivity(x1, x2, a.ks, nodata)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="space-geovision-cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sar-preprocess"); s.set_defaults(fn=cmd_sar)
    s.add_argument("product"); s.add_argument("out")
    s.add_argument("--pol", default="VV"); s.add_argument("--speckle", default="Lee"); s.add_argument("--window", type=int, default=5)
    s.add_argument("--dem", default="SRTM 3Sec"); s.add_argument("--external-dem"); s.add_argument("--spacing", type=float, default=10.0)
    s.add_argument("--projection", default="WGS84(DD)"); s.add_argument("--subset-mode", choices=["operator", "read", "after"], default="operator"); s.add_argument("--subset", help="study area as WKT POLYGON (lon lat order)")
    s.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"), help="study area as a lon/lat box")
    s.add_argument("--center", type=float, nargs=2, metavar=("LON", "LAT"), help="study area centre (use with --radius-km)")
    s.add_argument("--radius-km", type=float, help="half-width of the square box around --center, in km")
    s.add_argument("--no-thermal", action="store_true")
    s.add_argument("--stepwise", action="store_true", help="one gpt call per operator: names the failing step, keeps intermediates in out/work")
    s.add_argument("--subset-at", choices=["start", "end"], default="start", help="(stepwise) cut the area before processing or after terrain correction")
    s.add_argument("--gpt"); s.add_argument("--heap-gb", type=int); s.add_argument("--verify", action="store_true", help="check parameter names against the installed SNAP")
    f = sub.add_parser("footprint", help="show where a Sentinel-1 product lies and build a --subset"); f.set_defaults(fn=cmd_footprint)
    f.add_argument("product"); f.add_argument("--center", type=float, nargs=2, metavar=("LON", "LAT"))
    f.add_argument("--size-km", type=float, default=30.0)
    sc = sub.add_parser("scenes", help="compare several Sentinel-1 products"); sc.set_defaults(fn=cmd_scenes)
    sc.add_argument("products", nargs="+")
    sc.add_argument("--point", nargs=3, action="append", metavar=("NAME", "LON", "LAT"), help="test whether each scene covers a place (repeatable)")
    pr = sub.add_parser("sar-probe", help="find a SNAP configuration that initialises (fast, no full processing)"); pr.set_defaults(fn=cmd_probe)
    pr.add_argument("product"); pr.add_argument("out"); pr.add_argument("--subset"); pr.add_argument("--pol", default="VV")
    pr.add_argument("--window", type=int, default=5); pr.add_argument("--dem", default="SRTM 3Sec"); pr.add_argument("--spacing", type=float, default=10.0)
    pr.add_argument("--wait", type=float, default=60.0, help="seconds to wait before declaring init OK"); pr.add_argument("--gpt")
    pr.add_argument("--include-slow", action="store_true")
    k = sub.add_parser("snap-check", help="test the SNAP installation (no data needed)"); k.set_defaults(fn=cmd_snapcheck)
    k.add_argument("--gpt", help="full path to gpt.exe if it is not found automatically")
    fl = sub.add_parser("flood", help="flood extent from a before and an after water mask"); fl.set_defaults(fn=cmd_flood)
    fl.add_argument("before"); fl.add_argument("after"); fl.add_argument("out_tif")
    fd = sub.add_parser("flood-db", help="flood extent from before/after sigma0 dB images (one common threshold + backscatter drop)"); fd.set_defaults(fn=cmd_flood_db)
    fd.add_argument("before_db"); fd.add_argument("after_db"); fd.add_argument("out")
    fd.add_argument("--threshold", type=float, help="common threshold in dB; default: from bimodal tiles of both images")
    fd.add_argument("--min-drop-db", type=float, default=3.0, help="required backscatter decrease for flooded pixels (0 = off)")
    fd.add_argument("--min-pixels", type=int, default=100)
    fd.add_argument("--sensitivity-db", type=float, nargs="+", default=[-12.0, -14.0, -16.0, -18.0])
    sp = sub.add_parser("sample-points", help="stratified random points in each map class, for blind labelling"); sp.set_defaults(fn=cmd_sample_points)
    sp.add_argument("raster", help="classified map GeoTIFF (e.g. flood_classes.tif)"); sp.add_argument("out")
    sp.add_argument("--counts", required=True, help='points per map class, e.g. "0=100,1=50,2=100,4=50"')
    sp.add_argument("--seed", type=int, default=42); sp.add_argument("--nodata", type=int, default=255)
    sp.add_argument("--min-distance-px", type=int, default=5)
    sp.add_argument("--groups", help='categories to label, saved with the points so labelling and accuracy use the same names, '
                                     'e.g. "water:1;land:0" or "dry:0,4;flooded:2;receded:3"')
    vc = sub.add_parser("validate-curves", help="ROC / precision-recall curves of a continuous layer (dB, magnitude) against the labelled points")
    vc.set_defaults(fn=cmd_validate_curves)
    vc.add_argument("score", help="continuous layer on the SAME grid as the sampled map (e.g. the sigma0-dB GeoTIFF)")
    vc.add_argument("raster", help="the classified map the points were sampled from"); vc.add_argument("labels"); vc.add_argument("key")
    vc.add_argument("out"); vc.add_argument("--target", required=True, help='category name(s) that count as the target, e.g. "water" or "flooded"')
    vc.add_argument("--groups"); vc.add_argument("--nodata", type=int, default=255)
    vc.add_argument("--low-is-target", action="store_true", help="low layer values mean the target (water has low backscatter)")
    vp = sub.add_parser("validate-points", help="accuracy of a classified map from the labelled points"); vp.set_defaults(fn=cmd_validate_points)
    vp.add_argument("raster"); vp.add_argument("labels", help="points_to_label.csv with ref_label filled"); vp.add_argument("key", help="points_key.csv")
    vp.add_argument("out"); vp.add_argument("--groups", help='map classes grouped into categories, e.g. "flood:2;not_flood:0,1,3,4" '
                                                          '(default: the categories saved with the points by sample-points)')
    vp.add_argument("--nodata", type=int, default=255)
    w = sub.add_parser("water"); w.set_defaults(fn=cmd_water)
    w.add_argument("db_tif"); w.add_argument("out_tif"); w.add_argument("--method", choices=["otsu", "tiles", "fixed", "manual"], default="tiles"); w.add_argument("--threshold", type=float, help="dB, for --method fixed"); w.add_argument("--sensitivity", action="store_true", help="also report water area for -14/-16/-18/-20 dB")
    w.add_argument("--lower", type=float); w.add_argument("--upper", type=float); w.add_argument("--min-pixels", type=int, default=100)
    w.add_argument("--reference", help="reference water mask GeoTIFF on the same grid")
    c = sub.add_parser("classify"); c.set_defaults(fn=cmd_classify)
    c.add_argument("raster"); c.add_argument("train"); c.add_argument("field"); c.add_argument("out")
    c.add_argument("--model", choices=["rf", "svm_rbf", "svm_linear"], default="rf"); c.add_argument("--trees", type=int, default=100)
    c.add_argument("--block", type=int, default=100); c.add_argument("--validation")
    cs = sub.add_parser("change-sensitivity", help="changed area for several thresholds, with/without normalisation"); cs.set_defaults(fn=cmd_change_sensitivity)
    cs.add_argument("t1"); cs.add_argument("t2"); cs.add_argument("--ks", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    g = sub.add_parser("change"); g.set_defaults(fn=cmd_change)
    g.add_argument("t1"); g.add_argument("t2"); g.add_argument("out"); g.add_argument("--k", type=float, default=3.0)
    g.add_argument("--method", choices=["sigma", "otsu"], default="sigma"); g.add_argument("--no-normalise", dest="no_normalise", action="store_true")
    a = ap.parse_args(argv)
    try:
        rc = a.fn(a)
        if isinstance(rc, int):
            return rc
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
