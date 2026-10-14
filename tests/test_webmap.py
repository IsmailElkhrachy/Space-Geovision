import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import webmap as w  # noqa: E402


class TestRender(unittest.TestCase):
    def test_single_band_transparent_where_nan_or_nodata(self):
        a = np.arange(16, dtype=float).reshape(4, 4)
        a[0, 0] = np.nan
        a[1, 1] = -9999
        rgba = w.render_rgba(a, "viridis", nodata=-9999)
        self.assertEqual(rgba.shape, (4, 4, 4)); self.assertEqual(rgba.dtype, np.uint8)
        self.assertEqual(rgba[0, 0, 3], 0); self.assertEqual(rgba[1, 1, 3], 0)
        self.assertEqual(rgba[3, 3, 3], 255)

    def test_stretch_is_applied(self):
        a = np.array([[0.0, 50.0, 100.0]])
        lo = w.render_rgba(a, "gray", vmin=0, vmax=100)
        self.assertEqual(tuple(lo[0, 0, :3]), (0, 0, 0)); self.assertEqual(tuple(lo[0, 2, :3]), (255, 255, 255))

    def test_one_band_3d_equals_2d(self):
        a = np.random.default_rng(0).random((5, 5))
        np.testing.assert_array_equal(w.render_rgba(a[None]), w.render_rgba(a))

    def test_rgb_three_bands(self):
        a = np.random.default_rng(1).random((3, 6, 6)) * 1000
        a[:, 0, 0] = np.nan
        out = w.render_rgba(a)
        self.assertEqual(out.shape, (6, 6, 4)); self.assertEqual(out[0, 0, 3], 0); self.assertEqual(out[3, 3, 3], 255)

    def test_bad_shapes_and_all_nodata(self):
        with self.assertRaises(ValueError):
            w.render_rgba(np.zeros((2, 4, 4)))
        self.assertEqual(int(w.render_rgba(np.full((3, 3), np.nan))[..., 3].max()), 0)
        self.assertEqual(w.render_rgba(np.full((3, 3), 5.0)).shape, (3, 3, 4))   # constant raster does not divide by zero

    def test_unknown_colormap_falls_back(self):
        self.assertEqual(w.render_rgba(np.arange(4.0).reshape(2, 2), "no_such_map").shape, (2, 2, 4))

    def test_decimate(self):
        a = np.zeros((3, 4000, 2000))
        d, s = w.decimate(a, 1600)
        self.assertEqual(s, 3); self.assertEqual(d.shape, (3, 1334, 667))
        d, s = w.decimate(np.zeros((10, 10)), 1600)
        self.assertEqual((s, d.shape), (1, (10, 10)))


class TestHtml(unittest.TestCase):
    def test_png_and_page(self):
        rgba = w.render_rgba(np.arange(9.0).reshape(3, 3))
        uri = w.png_data_uri(rgba)
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        html = w.build_html([{"name": "HAND", "uri": uri, "bounds": (17.0, 44.0, 18.0, 45.0), "checked": True}],
                            [{"name": "Basin", "geojson": {"type": "FeatureCollection", "features": []}}],
                            (17.0, 44.0, 18.0, 45.0), 0.6)
        m = re.search(r"const D = (.*?);\nconst T = 256", html, re.S)
        data = json.loads(m.group(1))
        self.assertEqual(data["overlays"][0]["bounds"], [17.0, 44.0, 18.0, 45.0])
        self.assertEqual(data["view"], [17.0, 44.0, 18.0, 45.0]); self.assertEqual(data["opacity"], 0.6)
        for basemap in ("Esri World Imagery", "OpenStreetMap", "Google Satellite"):
            self.assertIn(basemap, html)

    def test_page_has_no_external_script_or_stylesheet(self):
        html = w.build_html([], [], None)
        self.assertNotIn("<script src", html); self.assertNotIn("unpkg", html); self.assertNotIn("leaflet", html.lower())
        self.assertNotIn("<link", html)

    def test_layer_names_cannot_break_out_of_the_script(self):
        evil = '</script><script>alert(1)</script>'
        html = w.build_html([], [{"name": evil, "geojson": {}}], None)
        self.assertNotIn("</script><script>alert", html)
        self.assertEqual(html.count("</script>"), 1)     # only the page's own script element
        self.assertIn("worldX(0, z)", html)               # world view when there is no extent


try:
    import rasterio  # noqa: F401
    HAVE_RASTERIO = True
except ImportError:
    HAVE_RASTERIO = False


@unittest.skipUnless(HAVE_RASTERIO, "rasterio not installed")
class TestMercator(unittest.TestCase):
    def test_geographic_raster_keeps_its_corners_and_alpha(self):
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
        rgba = np.zeros((20, 30, 4), np.uint8); rgba[..., 0] = 200; rgba[..., 3] = 255
        out, (s, w_, n, e) = w.to_web_mercator(rgba, from_origin(44.0, 18.0, 1 / 30.0, 1 / 20.0), CRS.from_epsg(4326))
        self.assertAlmostEqual(w_, 44.0, places=3); self.assertAlmostEqual(e, 45.0, places=3)
        self.assertAlmostEqual(n, 18.0, places=3); self.assertAlmostEqual(s, 17.0, places=3)
        self.assertEqual(out.shape[2], 4); self.assertGreater(out[..., 3].mean(), 200)


if __name__ == "__main__":
    unittest.main()
