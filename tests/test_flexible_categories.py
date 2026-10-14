"""Labelling / accuracy must work for ANY kind of map (water, change, flood, land cover ...), not only change / no change."""
import csv
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import labeler
import validation as val
from test_validation_raster import Crs, Tr, make_fake_rasterio  # noqa: F401  (fake in-memory rasterio)


class Parsing(unittest.TestCase):
    def test_any_names_and_both_separators(self):
        self.assertEqual(val.parse_groups("water=1;land=0"), {"water": [1], "land": [0]})
        self.assertEqual(val.parse_groups("urban:1;forest:2;bare soil:3,4"), {"urban": [1], "forest": [2], "bare soil": [3, 4]})
        self.assertEqual(val.groups_for_classes([1, 2, 5]), "class_1:1;class_2:2;class_5:5")
        self.assertEqual(val.groups_for_classes([1, 2], {1: "water"}), "water:1;class_2:2")

    def test_bad_specs(self):
        for bad in ("water:1;Water:0", "a:1;b", "a:;b:2", "a:x;b:2"):
            with self.assertRaises(ValueError, msg=bad):
                val.parse_groups(bad)

    def test_presets_parse(self):
        for name, spec in val.PRESETS.items():
            self.assertGreaterEqual(len(val.parse_groups(spec)), 2, name)

    def test_normalize(self):
        self.assertEqual(val.normalize_label(" Not-Flood "), "not_flood")
        self.assertEqual(val.normalize_label("permanent  water"), "permanent_water")


class Evaluate(unittest.TestCase):
    def test_labels_case_and_spacing_do_not_matter(self):
        groups = val.parse_groups("water:1;not_water:0")
        counts = {0: 900, 1: 100}
        a = [(1, "water")] * 18 + [(1, "not_water")] * 2 + [(0, "not_water")] * 19 + [(0, "water")]
        b = [(c, {"water": "Water ", "not_water": "NOT WATER"}[l]) for c, l in a]
        self.assertAlmostEqual(val.evaluate(counts, groups, a)["overall_accuracy"],
                               val.evaluate(counts, groups, b)["overall_accuracy"])

    def test_unknown_label_message_names_the_categories(self):
        groups = val.parse_groups("water:1;land:0")
        with self.assertRaisesRegex(ValueError, r"not among the categories .*water.*land.*flooded"):
            val.evaluate({0: 10, 1: 10}, groups, [(1, "water"), (0, "flooded")])

    def test_four_category_map_and_merged_classes(self):
        groups = val.parse_groups("dry:0,4;permanent_water:1;flooded:2;receded:3")
        counts = {0: 5000, 1: 800, 2: 300, 3: 200, 4: 150}
        s = ([(0, "dry")] * 20 + [(1, "permanent_water")] * 10 + [(2, "flooded")] * 8 + [(2, "dry")] * 2
             + [(3, "receded")] * 9 + [(3, "dry")] + [(4, "dry")] * 7 + [(4, "flooded")] * 3)
        r = val.evaluate(counts, groups, s)
        self.assertEqual(r["names"], ["dry", "permanent_water", "flooded", "receded"])
        self.assertEqual(r["strata"], [0, 1, 2, 3, 4])


class Scheme(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def test_roundtrip_and_blindness(self):
        p = val.write_scheme(self.tmp.name, "water:1;land:0", "map.tif")
        s = val.read_scheme(os.path.join(self.tmp.name, "points_to_label.csv"))
        self.assertEqual(s["categories"], ["water", "land"]); self.assertEqual(s["groups"], "water:1;land:0")
        self.assertEqual(val.read_scheme(self.tmp.name)["map"], "map.tif")
        txt = open(p).read()
        self.assertNotIn("row", txt)       # no point-level information

    def test_missing_or_broken_scheme(self):
        self.assertIsNone(val.read_scheme(os.path.join(self.tmp.name, "x.csv")))
        open(os.path.join(self.tmp.name, val.SCHEME_FILE), "w").write("{not json")
        self.assertIsNone(val.read_scheme(self.tmp.name))


class LabelerCategories(unittest.TestCase):
    def rows(self, *labels):
        return [dict(id=i, lon=0.0, lat=0.0, ref_label=l, note="") for i, l in enumerate(labels, 1)]

    def test_priority_scheme_then_file_then_fallback(self):
        self.assertEqual(labeler.choose_categories(["water", "land"], self.rows("", ""), ["a", "b"]), ["water", "land"])
        self.assertEqual(labeler.choose_categories(None, self.rows("urban", "forest", "urban", "uncertain"), ["a"]), ["urban", "forest"])
        self.assertEqual(labeler.choose_categories(None, self.rows("", ""), ["a", "b"]), ["a", "b"])

    def test_labels_in_file_missing_from_scheme_are_kept(self):
        out = labeler.choose_categories(["water", "land"], self.rows("Water", "flooded"), [])
        self.assertEqual(out, ["water", "land", "flooded"])


class RasterRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        a = np.full((80, 80), 255, np.uint8)
        a[:, :20] = 1; a[:, 20:40] = 2; a[:, 40:60] = 3; a[:, 60:75] = 4; a[:4] = 255
        prof = {"crs": Crs(False), "transform": Tr(10, 0, 0, 0, -10, 800)}
        self.files = {"lc.tif": (a, prof, 255)}
        p = mock.patch.dict(sys.modules, make_fake_rasterio(self.files)); p.start(); self.addCleanup(p.stop)

    def test_land_cover_with_four_named_classes(self):
        out = os.path.join(self.tmp.name, "s")
        spec = "urban:1;forest:2;crops:3;bare_soil:4"
        res = val.make_sampling_from_raster("lc.tif", out, {1: 15, 2: 15, 3: 15, 4: 15}, seed=3, min_distance_px=2, groups_spec=spec)
        self.assertIn("scheme", res["files"])
        self.assertEqual(val.read_scheme(out)["categories"], ["urban", "forest", "crops", "bare_soil"])
        self.assertNotIn("map_class", open(res["files"]["to_label"]).read())
        key = {int(r["id"]): int(r["map_class"]) for r in csv.DictReader(open(res["files"]["key"]))}
        names = {1: "Urban", 2: "FOREST", 3: "crops", 4: "bare soil"}      # typed differently on purpose
        lab = os.path.join(out, "lab.csv")
        with open(lab, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "lon", "lat", "ref_label", "note"])
            for r in csv.DictReader(open(res["files"]["to_label"])):
                w.writerow([r["id"], r["lon"], r["lat"], names[key[int(r["id"])]], ""])
        v = val.validate_from_files("lc.tif", lab, res["files"]["key"], val.parse_groups(val.read_scheme(out)["groups"]), out)
        self.assertAlmostEqual(v["overall_accuracy"], 1.0, places=6)

    def test_shuffle_keeps_scheme(self):
        out = os.path.join(self.tmp.name, "s"); out2 = os.path.join(self.tmp.name, "s2")
        res = val.make_sampling_from_raster("lc.tif", out, {1: 5, 2: 5}, seed=1, min_distance_px=2, groups_spec="a:1,3;b:2,4")
        sh = val.shuffle_points(res["files"]["to_label"], res["files"]["key"], out2, seed=2)
        self.assertEqual(val.read_scheme(out2)["groups"], "a:1,3;b:2,4")
        self.assertIn("scheme", sh["files"])

    def test_without_scheme_still_works(self):
        out = os.path.join(self.tmp.name, "n")
        res = val.make_sampling_from_raster("lc.tif", out, {1: 5, 2: 5}, seed=1, min_distance_px=2)
        self.assertNotIn("scheme", res["files"]); self.assertIsNone(val.read_scheme(out))


if __name__ == "__main__":
    unittest.main()
