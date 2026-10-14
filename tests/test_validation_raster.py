"""Raster wrappers of validation.py, run against a fake in-memory rasterio so they work everywhere."""
import csv
import os
import sys
import tempfile
import types
import unittest
from collections import namedtuple
from unittest import mock

import numpy as np

import validation

Tr = namedtuple("Tr", "a b c d e f")


class Crs:
    def __init__(self, geo): self.is_geographic = geo


def make_fake_rasterio(files):
    mod = types.ModuleType("rasterio")

    class Src:
        def __init__(self, p):
            self.arr, self.prof, self.nodata = files[p]
            self.count = 1
            self.profile = self.prof
        def read(self, i): return self.arr
        def __enter__(self): return self
        def __exit__(self, *a): return False
    mod.open = lambda p: Src(p)
    warp = types.ModuleType("rasterio.warp")
    warp.transform = lambda crs, dst, xs, ys: (list(xs), list(ys))
    mod.warp = warp
    return {"rasterio": mod, "rasterio.warp": warp}


def class_map():
    a = np.full((60, 60), 255, np.uint8)
    a[:, :30] = 1
    a[:, 30:55] = 2
    a[:5] = 255
    return a


class RasterWrappers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        prof = {"crs": Crs(False), "transform": Tr(10, 0, 0, 0, -10, 600)}
        self.files = {"cls.tif": (class_map(), prof, 255),
                      "mag.tif": (np.random.default_rng(0).random((60, 60)).astype("float32"), prof, None),
                      "many.tif": (np.arange(3600).reshape(60, 60).astype("int32"), prof, None)}
        p = mock.patch.dict(sys.modules, make_fake_rasterio(self.files)); p.start(); self.addCleanup(p.stop)

    def test_raster_classes(self):
        cls = validation.raster_classes("cls.tif")
        self.assertEqual(set(cls), {1, 2})
        self.assertEqual(cls[1], 55 * 30)

    def test_rejects_continuous_and_many_values(self):
        with self.assertRaisesRegex(ValueError, "continuous"):
            validation.raster_classes("mag.tif")
        with self.assertRaisesRegex(ValueError, "distinct values"):
            validation.raster_classes("many.tif")

    def test_sampling_and_validation_roundtrip(self):
        out = os.path.join(self.tmp.name, "v")
        res = validation.make_sampling_from_raster("cls.tif", out, {1: 20, 2: 20}, seed=1, min_distance_px=3)
        self.assertEqual(res["n"], 40); self.assertEqual(res["per_class"], {1: 20, 2: 20})
        key = {int(r["id"]): int(r["map_class"]) for r in csv.DictReader(open(res["files"]["key"]))}
        # label: class 1 -> "no_change" except 2 errors, class 2 -> "change" except 3 errors
        lab = os.path.join(out, "labelled.csv")
        rows = list(csv.DictReader(open(res["files"]["to_label"])))
        wrong1 = wrong2 = 0
        with open(lab, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "lon", "lat", "ref_label", "note"])
            for r in rows:
                c = key[int(r["id"])]
                if c == 1:
                    ref = "change" if wrong1 < 2 else "no_change"; wrong1 += ref == "change"
                else:
                    ref = "no_change" if wrong2 < 3 else "change"; wrong2 += ref == "no_change"
                w.writerow([r["id"], r["lon"], r["lat"], ref, ""])
        import xml.etree.ElementTree as ET
        root = ET.parse(res["files"]["kml"]).getroot()
        ns = {"k": "http://www.opengis.net/kml/2.2"}
        pms = root.findall(".//k:Placemark", ns)
        self.assertEqual(len(pms), 40)
        lon, lat, _z = pms[0].find(".//k:coordinates", ns).text.split(",")
        self.assertEqual(f"{float(lon):.6f}", rows[0]["lon"])
        with open(res["files"]["kml"], encoding="utf-8") as fh:
            self.assertNotIn("map_class", fh.read())
        # the map-class column must not leak into the file given to the analyst
        self.assertNotIn("map_class", open(res["files"]["to_label"]).read())
        v = validation.validate_from_files("cls.tif", lab, res["files"]["key"],
                                           {"change": [2], "no_change": [1]}, out)
        self.assertTrue(os.path.isfile(v["report_path"])); self.assertTrue(os.path.isfile(v["confusion_csv"]))
        self.assertAlmostEqual(v["overall_accuracy"], 35 / 40, delta=0.05)
        self.assertIn("Overall accuracy", v["report"])

    def test_ids_do_not_reveal_the_class(self):
        out = os.path.join(self.tmp.name, "v3")
        res = validation.make_sampling_from_raster("cls.tif", out, {1: 30, 2: 30}, seed=5, min_distance_px=3)
        with open(res["files"]["key"], newline="") as fh:
            classes = [int(r["map_class"]) for r in csv.DictReader(fh)]
        self.assertNotEqual(classes, sorted(classes))            # ids are not ordered by class
        self.assertEqual(sorted(classes), [1] * 30 + [2] * 30)

    def test_shuffle_existing_points(self):
        out = os.path.join(self.tmp.name, "v4"); out2 = os.path.join(self.tmp.name, "v4s")
        res = validation.make_sampling_from_raster("cls.tif", out, {1: 10, 2: 10}, seed=6, min_distance_px=3)
        sh = validation.shuffle_points(res["files"]["to_label"], res["files"]["key"], out2, seed=1)
        self.assertEqual(sh["n"], 20)
        with open(sh["files"]["id_map"], newline="") as fh:
            mp = {int(r["new_id"]): int(r["old_id"]) for r in csv.DictReader(fh)}
        with open(res["files"]["key"], newline="") as fh:
            old = {int(r["id"]): (r["row"], r["col"], r["map_class"]) for r in csv.DictReader(fh)}
        with open(sh["files"]["key"], newline="") as fh:
            new = {int(r["id"]): (r["row"], r["col"], r["map_class"]) for r in csv.DictReader(fh)}
        self.assertTrue(all(new[n] == old[o] for n, o in mp.items()))     # same points, new ids
        self.assertEqual(sorted(mp.values()), list(range(1, 21)))
        with open(sh["files"]["to_label"], newline="") as fh:
            self.assertTrue(all(r["ref_label"] == "" for r in csv.DictReader(fh)))

    def test_empty_labels_rejected(self):
        out = os.path.join(self.tmp.name, "v2")
        res = validation.make_sampling_from_raster("cls.tif", out, {1: 5, 2: 5}, seed=2, min_distance_px=3)
        with self.assertRaisesRegex(ValueError, "ref_label"):
            validation.validate_from_files("cls.tif", res["files"]["to_label"], res["files"]["key"],
                                           {"change": [2], "no_change": [1]}, out)


if __name__ == "__main__":
    unittest.main()
