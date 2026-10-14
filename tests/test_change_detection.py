import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import change_detection as cd


def scene(seed=0, h=120, w=120, nb=3):
    rng = np.random.default_rng(seed)
    base = rng.uniform(500, 3000, (nb, h, w)).astype(np.float32)
    return base


class TestChange(unittest.TestCase):
    def test_old_formula_gives_100_percent_when_one_date_is_zero(self):
        """Documents the defect of the rejected manuscript: |I1-I2|/(max+eps) is 1 wherever one image is 0."""
        a = scene(); b = np.zeros_like(a)
        old = np.abs(a - b) / (np.maximum(a, b) + 1e-7)
        self.assertAlmostEqual(float(old.mean()), 1.0, places=3)

    def test_zero_fill_is_excluded_not_counted_as_change(self):
        a = scene(); b = a.copy(); b[:, :, :80] = 0          # 2/3 of second image is fill
        r = cd.detect_changes(a, b)
        self.assertEqual(r.stats["valid_pixels"], 120 * 40)
        self.assertLess(r.stats["changed_percent"], 1.0)
        self.assertTrue(any("valid in both dates" in w for w in r.warnings))

    def test_global_brightness_shift_is_not_change(self):
        a = scene(1); b = a * 1.25 + 120                       # illumination/atmosphere-like change
        self.assertLess(cd.detect_changes(a, b).stats["changed_percent"], 1.0)

    def test_normalisation_improves_sensitivity(self):
        a = scene(1); rng = np.random.default_rng(5)
        b = (a * 1.25 + 120 + rng.normal(0, 20, a.shape)).astype(np.float32)   # gain change + sensor noise
        truth = np.zeros((120, 120), bool); truth[30:60, 40:80] = True
        b[:, truth] += 400                                      # modest real change
        rec = lambda r: ((r.classes >= 2) & truth).sum() / truth.sum()
        r_on, r_off = cd.detect_changes(a, b, normalise=True), cd.detect_changes(a, b, normalise=False)
        self.assertGreater(rec(r_on), 0.9)
        self.assertLess(rec(r_off), rec(r_on))
        self.assertTrue(any("normalisation was disabled" in w for w in r_off.warnings))

    def test_real_change_is_found(self):
        a = scene(2); b = a + np.random.default_rng(9).normal(0, 20, a.shape).astype(np.float32)
        truth = np.zeros((120, 120), bool); truth[30:60, 40:80] = True
        b[:, truth] += 1500
        r = cd.detect_changes(a, b, k=3.0)
        det = r.classes >= 2
        recall = (det & truth).sum() / truth.sum()
        precision = (det & truth).sum() / max(det.sum(), 1)
        self.assertGreater(recall, 0.95); self.assertGreater(precision, 0.90)

    def test_otsu_separates_bimodal(self):
        a = scene(3); rng = np.random.default_rng(7)
        b = a + rng.normal(0, 20, a.shape).astype(np.float32)
        b[:, :48, :] += 2500                                     # 40 % of the scene changed
        r = cd.detect_changes(a, b, method="otsu", normalise=False)
        self.assertGreater(r.stats["changed_percent"], 35); self.assertLess(r.stats["changed_percent"], 45)

    def test_otsu_constant_input_gives_clear_error(self):
        with self.assertRaises(ValueError):
            cd.otsu_threshold(np.ones(100))

    def test_area_in_hectares(self):
        a = scene(4); b = a.copy(); b[:, :10, :10] += 4000
        r = cd.detect_changes(a, b, pixel_area_m2=100.0, normalise=False)
        self.assertAlmostEqual(r.stats["changed_area_ha"], r.stats["changed_pixels"] * 100 / 1e4)

    def test_too_few_valid_pixels(self):
        a = np.zeros((1, 5, 5), np.float32)
        with self.assertRaises(ValueError):
            cd.detect_changes(a, a)

    @unittest.skipUnless(__import__("importlib").util.find_spec("affine"), "affine (rasterio dependency) not installed")
    def test_alignment_checks(self):
        from affine import Affine
        p = {"crs": "EPSG:32638", "height": 10, "width": 10, "transform": Affine(10, 0, 500000, 0, -10, 1900000)}
        cd.check_alignment(p, dict(p))
        with self.assertRaises(cd.AlignmentError):
            cd.check_alignment(p, {**p, "width": 11})
        with self.assertRaises(cd.AlignmentError):
            cd.check_alignment(p, {**p, "crs": "EPSG:4326"})
        with self.assertRaises(cd.AlignmentError):
            cd.check_alignment(p, {**p, "transform": Affine(10, 0, 500050, 0, -10, 1900000)})

    def test_shape_mismatch(self):
        with self.assertRaises(cd.AlignmentError):
            cd.detect_changes(np.ones((1, 10, 10)), np.ones((1, 11, 10)))


if __name__ == "__main__":
    unittest.main()


class TestSensitivityAndExport(unittest.TestCase):
    def test_sensitivity_table(self):
        a = scene(8); rng = np.random.default_rng(2)
        b = (a * 1.2 + 80 + rng.normal(0, 20, a.shape)).astype(np.float32)
        b[:, 30:60, 40:80] += 600
        rows = cd.change_sensitivity(a, b)
        self.assertEqual(len(rows), 8)
        on = [r for r in rows if r["normalise"]]
        self.assertTrue(all(x["changed_percent"] >= y["changed_percent"] for x, y in zip(on, on[1:])))   # fewer pixels at higher k
        self.assertIn("normalisation", cd.format_sensitivity(rows))

    def test_exported_classes_never_use_zero(self):
        a = scene(9); b = a.copy(); b[:, :, :60] = 0
        r = cd.detect_changes(a, b)
        out = cd.export_classes(r.classes)
        self.assertEqual(out.dtype, np.uint8)
        self.assertFalse((out == 0).any()); self.assertTrue((out[:, :60] == 255).all())
        self.assertTrue(set(np.unique(out)) <= {1, 2, 3, 255})
