import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import imageops as io_


def cube(bands=3, h=40, w=50, dtype=np.uint16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((bands, h, w)) * 3000 + np.arange(bands)[:, None, None] * 500).astype(dtype)


class TestEnhancement(unittest.TestCase):
    def test_input_is_never_modified_and_dtype_range_kept(self):
        a = cube(); ref = a.copy()
        for fn in (io_.enhance_contrast, io_.smooth, io_.sharpen, io_.median_filter, io_.gaussian_filter):
            out = fn(a)
            self.assertTrue(np.array_equal(a, ref), fn.__name__ + " modified its input")
            self.assertEqual(out.dtype, a.dtype, fn.__name__)
            self.assertEqual(out.shape, a.shape)

    def test_contrast_on_integer_image_is_not_zeroed(self):
        """The old code assigned a 0-1 float result into a uint16 array, giving almost all zeros."""
        a = cube()
        out = io_.enhance_contrast(a)
        for b in range(3):
            self.assertAlmostEqual(float(out[b].min()), float(a[b].min()), delta=1)
            self.assertAlmostEqual(float(out[b].max()), float(a[b].max()), delta=1)
            self.assertGreater(float(out[b].mean()), 0.2 * a[b].mean())

    def test_bands_are_processed_independently(self):
        a = np.zeros((3, 30, 30), np.float32); a[1] = 100.0          # a constant band must stay constant
        for fn in (io_.smooth, io_.median_filter, io_.lee_filter):
            out = fn(a)
            self.assertTrue(np.allclose(out[0], 0)); self.assertTrue(np.allclose(out[1], 100)); self.assertTrue(np.allclose(out[2], 0))

    def test_smoothing_reduces_noise_and_keeps_nan(self):
        rng = np.random.default_rng(1)
        a = (rng.normal(100, 10, (2, 60, 60))).astype(np.float32); a[:, 5, 5] = np.nan
        out = io_.smooth(a, 1.5)
        self.assertTrue(np.isnan(out[:, 5, 5]).all())
        self.assertLess(np.nanstd(out), np.nanstd(a) / 2)
        self.assertFalse(np.isnan(out[:, 5, 6]).any())

    def test_sharpen_increases_edge_contrast(self):
        a = np.zeros((1, 20, 20), np.float32); a[0, :, 10:] = 100
        s = io_.sharpen(a, 1.0, 1.0)
        self.assertGreater(s[0, 10, 10], 100)                         # overshoot at the edge
        self.assertTrue(np.allclose(s[0, :, :4], 0, atol=1e-3))

    def test_median_removes_salt_noise(self):
        a = np.full((1, 21, 21), 50.0, np.float32); a[0, 10, 10] = 5000
        self.assertAlmostEqual(float(io_.median_filter(a, 3)[0, 10, 10]), 50.0)
        with self.assertRaises(ValueError): io_.median_filter(a, 4)

    def test_lee_preserves_edges_better_than_mean(self):
        rng = np.random.default_rng(2)
        a = np.ones((1, 60, 60), np.float32); a[0, :, 30:] = 10
        noisy = a * rng.gamma(20, 1 / 20, a.shape).astype(np.float32)
        out = io_.lee_filter(noisy, 5)
        self.assertLess(float(np.std(out[0, :, 5:25])), float(np.std(noisy[0, :, 5:25])))
        self.assertGreater(float(out[0, 30, 33] - out[0, 30, 26]), 6.0)    # step still visible

    def test_single_band_2d_and_errors(self):
        self.assertEqual(io_.smooth(np.random.rand(10, 10)).shape, (10, 10))
        with self.assertRaises(ValueError): io_.smooth(np.zeros((2, 2, 2, 2)))

    def test_resample(self):
        a = cube(2, 20, 30, np.float32)
        r = io_.resample(a, 10, 15)
        self.assertEqual(r.shape, (2, 10, 15))
        with self.assertRaises(ValueError): io_.resample(a, 0, 5)

    def test_grayscale(self):
        a = np.stack([np.full((4, 4), 100.0), np.full((4, 4), 100.0), np.full((4, 4), 100.0)]).astype(np.float32)
        g = io_.grayscale(a)
        self.assertEqual(g.shape, (1, 4, 4)); self.assertAlmostEqual(float(g[0, 0, 0]), 100.0, places=3)
        self.assertEqual(io_.grayscale(a[:1]).shape, (1, 4, 4))


class TestNodataAndScaling(unittest.TestCase):
    def test_default_nodata_keeps_255_and_masks_zero(self):
        a = np.array([[0, 255, 7], [-9999, 8, 9]], np.int16)
        m = io_.mask_nodata(a)
        self.assertTrue(np.isnan(m[0, 0]) and np.isnan(m[1, 0]))
        self.assertEqual(float(m[0, 1]), 255.0)
        m2 = io_.mask_nodata(np.array([[1.0, -3.4028235e38, np.inf]]), nodata_values=[])
        self.assertEqual(int(np.isnan(m2).sum()), 2)

    def test_outliers_per_band(self):
        a = np.zeros((2, 50, 50), np.float32); a[0] += np.random.default_rng(0).normal(0, 1, (50, 50)); a[0, 3, 3] = 1000
        a[1] = 5000 + np.random.default_rng(1).normal(0, 1, (50, 50))
        out = io_.remove_outliers(a, 3)
        self.assertTrue(np.isnan(out[0, 3, 3])); self.assertLess(int(np.isnan(out[1]).sum()), 20)

    def test_normalize_to_8bit(self):
        a = cube(2, dtype=np.float32)
        n = io_.normalize(a)
        for b in range(2): self.assertEqual((float(n[b].min()), float(n[b].max())), (0.0, 1.0))
        e = io_.to_8bit(a); self.assertEqual(e.dtype, np.uint8); self.assertEqual((int(e.min()), int(e.max())), (0, 255))
        u = np.arange(12, dtype=np.uint8).reshape(1, 3, 4); self.assertTrue(np.array_equal(io_.to_8bit(u), u))
        self.assertEqual(float(io_.normalize(np.full((1, 3, 3), 4.0))[0, 0, 0]), 0.0)


class TestIndices(unittest.TestCase):
    def test_normalized_difference(self):
        r = io_.normalized_difference([0.6, 0, 0.2], [0.2, 0, 0.6])
        self.assertAlmostEqual(float(r[0]), 0.5, places=6); self.assertTrue(np.isnan(r[1])); self.assertAlmostEqual(float(r[2]), -0.5, places=6)

    def test_avi_sign_and_zero(self):
        r = io_.avi([0.5, 0.2], [0.1, 0.4])
        self.assertGreater(float(r[0]), 0); self.assertLess(float(r[1]), 0); self.assertFalse(np.isnan(r).any())
        self.assertAlmostEqual(float(r[0]), (0.5 * 0.9 * 0.4) ** (1 / 3), places=5)


if __name__ == "__main__":
    unittest.main()
