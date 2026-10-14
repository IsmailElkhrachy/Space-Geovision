"""Results must be drawn and written in their real map coordinates (pixel axes only without georeferencing)."""
import ast
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import georef

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from affine import Affine
except Exception:  # pragma: no cover
    Affine = None


class T:  # minimal affine stand-in
    def __init__(self, a, b, c, d, e, f): self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f


class Crs:
    def __init__(self, geo, epsg): self.is_geographic = geo; self._e = epsg
    def to_epsg(self): return self._e


UTM = Crs(False, 32638)
GRID = {"transform": T(10.0, 0, 435768.7, 0, -10.0, 1953819.9), "crs": UTM}


class TestPlaceholderTransform(unittest.TestCase):
    def test_old_from_origin_transform_is_not_a_position(self):
        # what versions before 1.8.1 wrote for every K-Means / segmentation result
        self.assertFalse(georef.is_georeferenced(T(1.0, 0, 0, 0, -1.0, 0), UTM))
        self.assertFalse(georef.is_georeferenced(T(1.0, 0, 0, 0, 1.0, 0), None))
        self.assertFalse(georef.is_georeferenced(None, UTM))

    def test_real_grid_is_georeferenced(self):
        self.assertTrue(georef.is_georeferenced(GRID["transform"], UTM))
        self.assertTrue(georef.is_georeferenced(T(1.0, 0, 500000, 0, -1.0, 4000000), UTM))


@unittest.skipIf(plt is None, "matplotlib missing")
class TestDraw(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_raster_drawn_in_map_coordinates(self):
        fig, ax = plt.subplots()
        arr = np.arange(20, dtype=float).reshape(4, 5)
        _, spec = georef.draw_raster(ax, arr, GRID)
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        self.assertAlmostEqual(left, 435768.7); self.assertAlmostEqual(right, 435768.7 + 50)
        self.assertAlmostEqual(top, 1953819.9); self.assertAlmostEqual(bottom, 1953819.9 - 40)
        self.assertIn("Easting", ax.get_xlabel()); self.assertIn("EPSG:32638", ax.get_xlabel())
        self.assertEqual(ax.get_ylabel(), "Northing (m)")

    def test_no_georeferencing_gives_pixel_axes(self):
        fig, ax = plt.subplots()
        georef.draw_raster(ax, np.zeros((6, 8)), {})
        self.assertEqual(ax.get_xlabel(), "Column (pixel)"); self.assertEqual(ax.get_ylabel(), "Row (pixel)")
        self.assertEqual(ax.get_xlim(), (-0.5, 7.5))

    def test_geographic_grid_uses_degrees(self):
        fig, ax = plt.subplots()
        g = {"transform": T(0.001, 0, 44.4, 0, -0.001, 17.7), "crs": Crs(True, 4326)}
        georef.draw_raster(ax, np.zeros((10, 10)), g)
        self.assertEqual(ax.get_xlabel().split(" ")[0], "Longitude")
        self.assertAlmostEqual(ax.get_xlim()[0], 44.4)

    def test_rgb_array(self):
        fig, ax = plt.subplots()
        georef.draw_raster(ax, np.zeros((4, 5, 3)), GRID)
        self.assertAlmostEqual(ax.get_xlim()[1], 435768.7 + 50)

    def test_classified_legend_names_and_nodata(self):
        fig, ax = plt.subplots()
        arr = np.array([[0, 1, 2], [2, 3, 3]], dtype=np.uint8)
        vals = georef.draw_classified(ax, arr, GRID, names={1: "bare_ground", 2: "built_up", 3: "rock"}, title="t")
        self.assertEqual(vals, [1, 2, 3])                      # 0 = no data is not drawn nor listed
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels, ["bare_ground", "built_up", "rock"])
        self.assertAlmostEqual(ax.get_xlim()[1], 435768.7 + 30)

    def test_classified_kmeans_style_ids_without_names(self):
        fig, ax = plt.subplots()
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        self.assertEqual(georef.draw_classified(ax, arr, {}), [1, 2, 3, 4])
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()][0], "Class 1")


class TestInheritProfile(unittest.TestCase):
    def test_same_size_keeps_grid(self):
        img = np.zeros((3, 40, 50))
        p = georef.inherit_profile([(GRID, img)], (40, 50))
        self.assertIs(p["crs"], UTM)
        self.assertEqual(p["transform"].c, 435768.7)

    def test_skips_ungeoreferenced_source(self):
        bad = {"transform": T(1.0, 0, 0, 0, -1.0, 0), "crs": None}
        p = georef.inherit_profile([(bad, np.zeros((1, 4, 4))), (GRID, np.zeros((1, 4, 4)))], (4, 4))
        self.assertEqual(p["transform"].a, 10.0)

    def test_nothing_known_gives_empty(self):
        self.assertEqual(georef.inherit_profile([({}, None), (None, None)], (4, 4)), {})

    @unittest.skipIf(Affine is None, "affine missing")
    def test_resampled_result_keeps_extent(self):
        g = {"transform": Affine(10.0, 0, 1000.0, 0, -10.0, 5000.0), "crs": UTM}
        p = georef.inherit_profile([(g, np.zeros((1, 100, 200)))], (50, 100))   # half the size
        t = p["transform"]
        self.assertAlmostEqual(t.a, 20.0); self.assertAlmostEqual(t.e, -20.0)
        self.assertAlmostEqual(t.c, 1000.0); self.assertAlmostEqual(t.f, 5000.0)
        self.assertEqual((p["height"], p["width"]), (50, 100))


class _FakeDataset:
    def __init__(self, store, path, kw):
        store["kw"], store["path"] = kw, path
        self.store = store
    def write(self, arr): self.store["arr"] = arr
    def __enter__(self): return self
    def __exit__(self, *a): return False


class TestWriteGeoTiff(unittest.TestCase):
    """rasterio replaced by a recorder: checks WHAT is written, not the GDAL driver."""
    def setUp(self):
        self.store = {}
        fake = types.ModuleType("rasterio")
        fake.open = lambda path, mode, **kw: _FakeDataset(self.store, path, kw)
        self._old = sys.modules.get("rasterio")
        sys.modules["rasterio"] = fake

    def tearDown(self):
        if self._old is None:
            sys.modules.pop("rasterio", None)
        else:
            sys.modules["rasterio"] = self._old

    def test_keeps_transform_and_crs(self):
        arr = np.ones((5, 6), dtype=np.uint8)
        info = georef.write_geotiff("x.tif", arr, GRID, nodata=0)
        kw = self.store["kw"]
        self.assertIs(kw["transform"], GRID["transform"]); self.assertIs(kw["crs"], UTM)
        self.assertEqual((kw["height"], kw["width"], kw["count"]), (5, 6, 1)); self.assertEqual(kw["nodata"], 0)
        self.assertTrue(info["georeferenced"]); self.assertEqual(info["pixel_size"], (10.0, 10.0))
        self.assertIn("Georeferenced", georef.georef_note(info))

    def test_never_invents_a_position(self):
        for prof in (None, {}, {"transform": T(1.0, 0, 0, 0, -1.0, 0), "crs": None}):
            self.store.clear()
            info = georef.write_geotiff("y.tif", np.ones((3, 3), dtype=np.uint8), prof)
            self.assertNotIn("transform", self.store["kw"]); self.assertNotIn("crs", self.store["kw"])
            self.assertFalse(info["georeferenced"])
            self.assertIn("no georeferencing", georef.georef_note(info))

    def test_multiband_and_dtype(self):
        georef.write_geotiff("z.tif", np.zeros((3, 4, 4)), GRID, dtype="float32")
        self.assertEqual(self.store["kw"]["count"], 3); self.assertEqual(self.store["arr"].dtype, np.float32)

    def test_result_profile_copies_grid_only(self):
        src = {"transform": GRID["transform"], "crs": UTM, "count": 5, "dtype": "uint16", "nodata": 0.0, "height": 9, "width": 9}
        r = georef.result_profile(src, count=1, dtype="uint8", height=4, width=5)
        self.assertEqual((r["count"], r["dtype"], r["height"], r["width"], r["nodata"]), (1, "uint8", 4, 5, None))
        self.assertIs(r["transform"], GRID["transform"])


class TestUiSource(unittest.TestCase):
    """Static guards on ui.py (cannot import Qt here): no made-up transform, no raw pixel-axes drawing of rasters."""
    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "..", "ui.py")
        with open(path, encoding="utf-8") as fh:
            cls.src = fh.read()
        cls.tree = ast.parse(cls.src)

    def test_no_placeholder_transform_is_written(self):
        self.assertNotIn("from_origin(", self.src.replace("from rasterio.transform import from_origin", ""))

    def test_raw_imshow_only_where_extent_is_given_or_pixel_previews(self):
        allowed = {"display_raster_layer", "_display_sar_layer",      # pass extent=...
                   "show_automatic_masks", "update_mask_display"}       # SAM mask previews: pixel space, axes hidden
        found = []

        class V(ast.NodeVisitor):
            def __init__(s): s.stack = []
            def visit_FunctionDef(s, n): s.stack.append(n.name); s.generic_visit(n); s.stack.pop()
            def visit_Call(s, n):
                if isinstance(n.func, ast.Attribute) and n.func.attr == "imshow":
                    found.append((s.stack[-1] if s.stack else "?", n.lineno))
                s.generic_visit(n)
        V().visit(self.tree)
        offenders = [f for f in found if f[0] not in allowed]
        self.assertEqual(offenders, [], f"imshow without map coordinates in: {offenders}; use georef.draw_raster")

    def test_save_classified_uses_georef_writer(self):
        i = self.src.index("def save_classified_image")
        body = self.src[i:i + 3500]
        self.assertIn("georef.write_geotiff", body)


if __name__ == "__main__":
    unittest.main()
