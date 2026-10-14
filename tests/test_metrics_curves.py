"""F1 / IoU / kappa / disagreement and ROC / PR curves in the validation report."""
import csv
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import accuracy as acc
import validation as val
from test_validation_raster import Crs, Tr, make_fake_rasterio


class Metrics(unittest.TestCase):
    def test_binary_values_by_hand(self):
        # one stratum per category, equal areas -> P = [[.45,.05],[.10,.40]] after row scaling by W=.5
        n = np.array([[18, 2], [4, 16]], float)
        m = acc.grouped_metrics(n, [100, 100], [0, 1], n_boot=200, seed=1)
        P = np.array([[.45, .05], [.10, .40]])
        np.testing.assert_allclose(m["P"], P)
        tp = np.diag(P); row = P.sum(1); col = P.sum(0)
        np.testing.assert_allclose(m["f1"], 2 * tp / (row + col))
        np.testing.assert_allclose(m["iou"], tp / (row + col - tp))
        self.assertAlmostEqual(m["quantity_disagreement"] + m["allocation_disagreement"], 1 - np.trace(P))
        oa = np.trace(P); pe = (row * col).sum()
        self.assertAlmostEqual(m["kappa"], (oa - pe) / (1 - pe))
        lo, hi = m["f1_ci95"][0]; self.assertLessEqual(lo, m["f1"][0]); self.assertGreaterEqual(hi, m["f1"][0])

    def test_unequal_allocation_is_area_weighted(self):
        # tiny 'water' stratum sampled as heavily as huge 'land': F1 must follow areas, not sample sizes
        n = np.array([[18, 2], [1, 19]], float)
        m = acc.grouped_metrics(n, [1000, 99000], [0, 1], n_boot=50)
        self.assertLess(m["f1"][0], 0.6)           # water is mostly commission error at this prevalence

    def test_perfect_map(self):
        m = acc.grouped_metrics(np.diag([10.0, 10, 10]), [5, 5, 5], [0, 1, 2], n_boot=50)
        self.assertEqual(m["macro_f1"], 1.0); self.assertEqual(m["mean_iou"], 1.0); self.assertAlmostEqual(m["kappa"], 1.0)

    def test_report_contains_the_measures(self):
        g = val.parse_groups("water:1;land:0")
        s = [(1, "water")] * 18 + [(1, "land")] * 2 + [(0, "land")] * 19 + [(0, "water")]
        txt = val.format_report(val.evaluate({0: 900, 1: 100}, g, s))
        for word in ("F1 / Dice", "IoU / Jaccard", "kappa", "quantity", "allocation", "precision(UA)"):
            self.assertIn(word, txt)


class Curves(unittest.TestCase):
    def test_separable_and_random(self):
        c = val.weighted_curves([-20, -19, -5, -4], [1, 1, 0, 0], [1, 1, 1, 1], low_is_positive=True)
        self.assertEqual(c["auc"], 1.0); self.assertEqual(c["best_f1"], 1.0)
        rng = np.random.default_rng(0)
        r = val.weighted_curves(rng.normal(size=4000), rng.random(4000) < .3, np.ones(4000))
        self.assertAlmostEqual(r["auc"], 0.5, delta=0.05)
        self.assertAlmostEqual(r["prevalence"], 0.3, delta=0.03)

    def test_weights_change_precision_not_auc(self):
        rng = np.random.default_rng(1)
        y = np.r_[np.ones(100, bool), np.zeros(100, bool)]; s = np.r_[rng.normal(-18, 2, 100), rng.normal(-8, 3, 100)]
        a = val.weighted_curves(s, y, np.ones(200), True); b = val.weighted_curves(s, y, np.r_[np.full(100, 10.), np.full(100, 1000.)], True)
        self.assertAlmostEqual(a["auc"], b["auc"], places=6)
        self.assertLess(b["average_precision"], a["average_precision"])      # rare target in the area -> lower precision

    def test_needs_both_kinds(self):
        with self.assertRaises(ValueError):
            val.weighted_curves([1, 2], [True, True], [1, 1])


class FromFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        rng = np.random.default_rng(3)
        m = np.zeros((60, 60), np.uint8); m[:, :20] = 1; m[:3] = 255
        db = np.where(m == 1, rng.normal(-18, 2, m.shape), rng.normal(-8, 3, m.shape)).astype("float32")
        prof = {"crs": Crs(False), "transform": Tr(10, 0, 0, 0, -10, 600)}
        self.files = {"map.tif": (m, prof, 255), "db.tif": (db, prof, None), "small.tif": (db[:30], prof, None)}
        p = mock.patch.dict(sys.modules, make_fake_rasterio(self.files)); p.start(); self.addCleanup(p.stop)

    def test_roundtrip(self):
        out = os.path.join(self.tmp.name, "o")
        res = val.make_sampling_from_raster("map.tif", out, {0: 25, 1: 25}, seed=1, min_distance_px=2, groups_spec="water:1;land:0")
        key = {int(r["id"]): int(r["map_class"]) for r in csv.DictReader(open(res["files"]["key"]))}
        lab = os.path.join(out, "lab.csv")
        with open(lab, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "lon", "lat", "ref_label", "note"])
            for r in csv.DictReader(open(res["files"]["to_label"])):
                w.writerow([r["id"], r["lon"], r["lat"], "Water" if key[int(r["id"])] == 1 else "land", ""])
        cur = val.validate_curves_from_files("db.tif", lab, res["files"]["key"], "map.tif", val.parse_groups("water:1;land:0"),
                                             ["water"], os.path.join(out, "c"), low_is_positive=True, n_boot=50)
        self.assertGreater(cur["auc"], 0.95)
        self.assertLessEqual(cur["auc_ci95"][0], cur["auc"] + 1e-9)
        for f in cur["files"].values():
            if not f.endswith(".png"):
                self.assertTrue(os.path.isfile(f), f)
        self.assertLess(cur["best_f1_threshold"], -8)            # a dB threshold between the two populations
        with self.assertRaisesRegex(ValueError, "same grid"):
            val.validate_curves_from_files("small.tif", lab, res["files"]["key"], "map.tif", val.parse_groups("water:1;land:0"),
                                           ["water"], os.path.join(out, "c2"), True)
        with self.assertRaisesRegex(ValueError, "not among"):
            val.validate_curves_from_files("db.tif", lab, res["files"]["key"], "map.tif", val.parse_groups("water:1;land:0"),
                                           ["flooded"], os.path.join(out, "c3"), True)


if __name__ == "__main__":
    unittest.main()
