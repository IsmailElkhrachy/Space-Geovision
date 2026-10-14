import os, sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import accuracy as acc
import validation as val


def population(seed=0, n=400_000):
    """Synthetic population with a KNOWN truth: map classes land / permanent / flood, with realistic errors."""
    rng = np.random.default_rng(seed)
    map_cls = rng.choice([0, 1, 2], size=n, p=[0.86, 0.05, 0.09])
    ref = map_cls.copy()
    flip = rng.random(n)
    ref[(map_cls == 2) & (flip < 0.20)] = 0          # 20 % of mapped flood is really land (commission)
    ref[(map_cls == 0) & (flip < 0.02)] = 2          # 2 % of mapped land is really flood (omission)
    ref[(map_cls == 1) & (flip < 0.05)] = 0
    return map_cls, ref


class TestOlofsson(unittest.TestCase):
    def test_proportional_allocation_equals_simple_estimates(self):
        cm = np.array([[80, 10], [5, 55]])
        res = acc.olofsson(cm, map_counts=cm.sum(axis=1))
        self.assertAlmostEqual(res["overall_accuracy"], (80 + 55) / 150)
        self.assertAlmostEqual(res["users_accuracy"][0], 80 / 90)
        self.assertAlmostEqual(res["producers_accuracy"][0], 80 / 85)

    def test_rare_class_is_weighted_by_area_not_by_sample_size(self):
        cm = np.array([[98, 2], [20, 80]])                   # 100 samples in each stratum
        small_second = acc.olofsson(cm, map_counts=[990_000, 10_000])
        large_second = acc.olofsson(cm, map_counts=[500_000, 500_000])
        self.assertGreater(small_second["overall_accuracy"], large_second["overall_accuracy"])   # dominated by the accurate class

    def test_estimators_are_unbiased_and_variances_are_right(self):
        """Monte Carlo against a population with known truth. The mean of the estimated variances must match the
        empirical variance of the estimates (this is the property that makes the standard errors trustworthy)."""
        map_cls, ref = population()
        true_oa = float((map_cls == ref).mean())
        true_area = float((ref == 2).sum())
        true_pa = float(((map_cls == 2) & (ref == 2)).sum() / (ref == 2).sum())
        counts = np.bincount(map_cls, minlength=3)
        pools = [np.flatnonzero(map_cls == c) for c in range(3)]
        rng = np.random.default_rng(11)
        oa, v_oa, ar, v_ar, pa, v_pa, cover = [], [], [], [], [], [], 0
        reps = 1000
        for _ in range(reps):
            cm = np.zeros((3, 3))
            for c, n in ((0, 100), (1, 50), (2, 100)):
                cm[c] += np.bincount(ref[rng.choice(pools[c], n, replace=False)], minlength=3)
            r = acc.olofsson(cm, counts)
            oa.append(r["overall_accuracy"]); v_oa.append(r["oa_se"] ** 2)
            ar.append(r["area_estimated"][2]); v_ar.append(r["area_se"][2] ** 2)
            pa.append(r["producers_accuracy"][2]); v_pa.append(r["producers_se"][2] ** 2)
            lo, hi = r["oa_ci95"]; cover += lo <= true_oa <= hi
        self.assertLess(abs(np.mean(oa) - true_oa), 0.25 * np.std(oa))             # unbiased
        self.assertLess(abs(np.mean(ar) - true_area), 0.25 * np.std(ar))
        self.assertLess(abs(np.mean(pa) - true_pa), 0.25 * np.std(pa))
        self.assertAlmostEqual(np.mean(v_oa) / np.var(oa), 1.0, delta=0.12)        # variance estimators are right
        self.assertAlmostEqual(np.mean(v_ar) / np.var(ar), 1.0, delta=0.12)
        self.assertTrue(0.70 < np.nanmean(v_pa) / np.var(pa) < 1.25)               # delta-method: slightly optimistic
        # Wald intervals under-cover a little for accuracies near 100 % with ~100 samples: a loose check, documented in the paper
        self.assertGreater(cover / reps, 0.82)

    def test_input_checks(self):
        with self.assertRaises(ValueError):
            acc.olofsson(np.ones((2, 3)), [1, 1])
        with self.assertRaises(ValueError):
            acc.olofsson(np.array([[1, 0], [0, 1]]), [10, 10])     # fewer than 2 samples in a stratum


class TestSampling(unittest.TestCase):
    def test_points_lie_in_the_requested_class_and_keep_distance(self):
        rng = np.random.default_rng(0)
        arr = rng.choice([0, 1, 2, 255], size=(300, 300), p=[0.6, 0.1, 0.2, 0.1]).astype(np.uint8)
        pts = val.stratified_points(arr, {1: 40, 2: 60}, seed=3, min_distance_px=6)
        self.assertEqual(sum(p["map_class"] == 1 for p in pts), 40); self.assertEqual(sum(p["map_class"] == 2 for p in pts), 60)
        self.assertTrue(all(arr[p["row"], p["col"]] == p["map_class"] for p in pts))
        for i, a in enumerate(pts):
            for b in pts[i + 1:]:
                self.assertGreaterEqual((a["row"] - b["row"]) ** 2 + (a["col"] - b["col"]) ** 2, 36)
        self.assertEqual(pts, val.stratified_points(arr, {1: 40, 2: 60}, seed=3, min_distance_px=6))   # reproducible

    def test_errors(self):
        arr = np.zeros((20, 20), np.uint8); arr[5, 5] = 1
        with self.assertRaises(ValueError):
            val.stratified_points(arr, {1: 5})                    # only one pixel
        with self.assertRaises(ValueError):
            val.stratified_points(arr, {7: 1})                    # class absent

    def test_pixel_xy(self):
        x, y = val.pixel_xy((0.001, 0, 106.0, 0, -0.001, 21.0), 10, 20)
        self.assertAlmostEqual(x, 106.0205); self.assertAlmostEqual(y, 20.9895)

    def test_parsers(self):
        self.assertEqual(val.parse_counts("1=100, 2=50"), {1: 100, 2: 50})
        self.assertEqual(val.parse_groups("flood:2;not_flood:0,1,3,4"), {"flood": [2], "not_flood": [0, 1, 3, 4]})
        for bad in ("1-100", ""):
            with self.assertRaises(ValueError):
                val.parse_counts(bad)
        with self.assertRaises(ValueError):
            val.parse_groups("a:1;b:1,2")                          # class in two groups
        with self.assertRaises(ValueError):
            val.parse_groups("a:1")


class TestEndToEnd(unittest.TestCase):
    def test_blind_labelling_workflow_recovers_truth(self):
        rng = np.random.default_rng(5)
        H = W = 600
        map_arr = rng.choice([0, 1, 2], size=(H, W), p=[0.85, 0.06, 0.09]).astype(np.uint8)
        truth = map_arr.copy()
        truth[(map_arr == 2) & (rng.random((H, W)) < 0.25)] = 0           # 25 % of mapped flood is wrong
        pts = val.stratified_points(map_arr, {0: 120, 1: 60, 2: 120}, seed=11, min_distance_px=3)
        with tempfile.TemporaryDirectory() as d:
            ll = [val.pixel_xy((0.0001, 0, 106.0, 0, -0.0001, 21.0), p["row"], p["col"]) for p in pts]
            files = val.write_sampling_files(pts, ll, d)
            text = Path(files["to_label"]).read_text()
            self.assertNotIn("map_class", text.splitlines()[0])                 # the analyst never sees the map class
            self.assertIn("map_class", Path(files["key"]).read_text().splitlines()[0])
            lines = ["id,lon,lat,ref_label,note"]
            for p, (lon, lat) in zip(pts, ll):                                  # the "analyst" labels from the truth
                lab = "flood" if truth[p["row"], p["col"]] == 2 else "not_flood"
                lines.append(f"{p['id']},{lon:.6f},{lat:.6f},{lab},")
            lines[5] = lines[5].rsplit(",", 2)[0] + ",uncertain,"                # one uncertain point
            Path(files["to_label"]).write_text("\n".join(lines))
            samples = val.read_labels(files["to_label"], files["key"])
        groups = {"flood": [2], "not_flood": [0, 1]}
        counts = {int(c): int((map_arr == c).sum()) for c in (0, 1, 2)}
        res = val.evaluate(counts, groups, samples, pixel_area=100.0)
        self.assertEqual(res["excluded_uncertain"], 1)
        true_oa = float(((map_arr == 2) == (truth == 2)).mean())
        lo, hi = res["oa_ci95"]
        self.assertTrue(lo - 0.03 <= true_oa <= hi + 0.03)
        self.assertLess(res["users_accuracy"][0], 0.85)                         # flood class: ~75 % user's accuracy
        self.assertIn("Overall accuracy", val.format_report(res))

    def test_unknown_label_rejected(self):
        with self.assertRaises(ValueError):
            val.evaluate({0: 100, 2: 100}, {"flood": [2], "not_flood": [0]}, [(2, "flod"), (2, "flood"), (0, "not_flood"), (0, "not_flood")])


if __name__ == "__main__":
    unittest.main()


class GroupedStrata(unittest.TestCase):
    def test_reduces_to_olofsson_when_groups_equal_strata(self):
        cm = np.array([[45, 5], [8, 42]])
        counts = [9000, 1000]
        a = acc.olofsson(cm, counts)
        b = acc.olofsson_grouped(cm, counts, [0, 1])
        for key in ("overall_accuracy", "oa_se", "users_accuracy", "users_se", "producers_accuracy", "producers_se",
                    "area_estimated", "area_se"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-9, err_msg=key)

    def test_merged_strata_are_not_pooled(self):
        # map classes 2 (area 1338, 31 % real change) and 3 (area 537, 94 % real change) merged into 'change'
        samples = ([(1, "no_change")] * 95 + [(1, "change")] * 5 + [(2, "change")] * 31 + [(2, "no_change")] * 69
                   + [(3, "change")] * 91 + [(3, "no_change")] * 6 + [(3, "uncertain")] * 3)
        areas = {1: 20127.5, 2: 1338.4, 3: 537.2}
        r = val.evaluate(areas, {"change": [2, 3], "no_change": [1]}, samples)
        i = r["names"].index("change")
        self.assertAlmostEqual(r["users_accuracy"][i], (0.31 * 1338.4 + 91 / 97 * 537.2) / 1875.6, places=6)   # 49.0 %, not the pooled 61.9 %
        self.assertAlmostEqual(r["area_estimated"][i], 0.05 * 20127.5 + 0.31 * 1338.4 + 91 / 97 * 537.2, places=4)
        self.assertEqual(r["uncertain_by_class"][3], 3)
        strong = val.evaluate(areas, {"change": [3], "no_change": [1, 2]}, samples)
        j = strong["names"].index("change")
        self.assertAlmostEqual(strong["users_accuracy"][j], 91 / 97, places=6)
        self.assertAlmostEqual(strong["area_estimated"][j], r["area_estimated"][i], places=6)   # same reference area either way
        self.assertLess(strong["producers_accuracy"][j], 0.3)
        self.assertIn("class 3 (change)", val.format_report(strong))
