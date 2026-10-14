"""ImageProcessor wrappers (need rasterio, geopandas, shapely, affine: skipped when they are not installed)."""
import importlib.util, sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAVE = all(importlib.util.find_spec(m) for m in ("rasterio", "geopandas", "shapely", "affine"))


@unittest.skipUnless(HAVE, "rasterio/geopandas/shapely/affine not installed")
class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        from affine import Affine
        import helpers
        self.IP = helpers.ImageProcessor
        self.helpers = helpers
        self.tr = Affine(10, 0, 1000, 0, -10, 5000)
        self.img = np.arange(3 * 40 * 50, dtype=np.float32).reshape(3, 40, 50)
        self.prof = {'transform': self.tr, 'width': 50, 'height': 40, 'count': 3, 'dtype': 'float32', 'nodata': None, 'crs': None}

    def test_clip_geographic_window(self):
        c, p, _ = self.IP.clip_raster(self.img, self.prof, (1100, 4800, 1200, 4900))
        self.assertEqual(c.shape, (3, 10, 10))
        self.assertEqual((p['transform'].c, p['transform'].f), (1100, 4900))
        self.assertTrue(np.array_equal(c, self.img[:, 10:20, 10:20]))

    def test_clip_box_outside_is_clamped_not_wrapped(self):
        c, p, _ = self.IP.clip_raster(self.img, self.prof, (900, 4900, 1100, 5100))
        self.assertTrue(np.array_equal(c, self.img[:, 0:10, 0:10]))
        self.assertEqual((p['transform'].c, p['transform'].f), (1000, 5000))
        with self.assertRaises(RuntimeError):
            self.IP.clip_raster(self.img, self.prof, (5000, 5000, 6000, 6000))

    def test_clip_nodata_becomes_nan_with_consistent_profile(self):
        im = self.img.copy(); im[0, 12, 12] = -1.0
        c, p, _ = self.IP.clip_raster(im, dict(self.prof, nodata=-1.0), (1100, 4800, 1200, 4900))
        self.assertTrue(np.isnan(c[0, 2, 2])); self.assertEqual(p['dtype'], 'float32')

    def test_resample_does_not_mutate_profile(self):
        prof = dict(self.prof)
        out, new, _ = self.IP.resample_image(self.img, prof, 0.5, 'f')
        self.assertEqual((prof['width'], new['width'], out.shape), (50, 25, (3, 20, 25)))
        self.assertEqual(new['transform'].a, 20)

    def test_enhancement_does_not_modify_input(self):
        ref = self.img.copy()
        self.IP.enhance_contrast(self.img, 'f'); self.IP.smooth_image(self.img, 1, 'f'); self.IP.sharpen_image(self.img, 1, 1, 'f')
        self.assertTrue(np.array_equal(self.img, ref))

    def test_mean_shift_two_populations(self):
        rng = np.random.default_rng(0)
        two = np.concatenate([rng.normal(10, 1, (1, 30, 15)), rng.normal(1000, 5, (1, 30, 15))], axis=2)
        labels, n = self.IP.mean_shift_segmentation(two, 0.2)
        self.assertEqual(n, 2); self.assertNotEqual(labels[0, 0], labels[0, -1])

    def test_polygons_accept_int64_and_keep_class_zero_when_asked(self):
        arr = np.zeros((10, 10), np.int64); arr[:5] = 1
        g = self.helpers.RasterHelper.raster_to_polygons(arr, self.tr, None)
        self.assertEqual(sorted(g['class'].unique()), [1])
        g2 = self.helpers.RasterHelper.raster_to_polygons(arr, self.tr, None, skip_values=())
        self.assertEqual(sorted(g2['class'].unique()), [0, 1])

    def test_streams_to_vector_features(self):
        fd = np.zeros((4, 5), np.uint8); fd[1, 0:4] = 1
        order = np.zeros((4, 5), np.int32); order[1, :] = 1
        feats = self.IP().streams_to_vector(order, fd, self.tr)
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0]['properties']['order'], 1)
        self.assertAlmostEqual(feats[0]['geometry'].length, 40.0, places=6)


if __name__ == "__main__":
    unittest.main()
