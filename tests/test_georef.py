import sys, unittest, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import georef


class T:  # minimal affine stand-in
    def __init__(self, a, b, c, d, e, f): self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f


class Crs:
    def __init__(self, geo, epsg): self.is_geographic = geo; self._e = epsg
    def to_epsg(self): return self._e


class TestGeoref(unittest.TestCase):
    def test_projected_extent(self):
        s = georef.axis_spec(T(30, 0, 500000, 0, -30, 4000000), Crs(False, 32648), 100, 200)
        self.assertEqual(s.extent, (500000, 506000, 3997000, 4000000))
        self.assertIn("Easting", s.xlabel); self.assertIn("EPSG:32648", s.xlabel); self.assertEqual(s.aspect, 1.0)

    def test_geographic_aspect_and_labels(self):
        s = georef.axis_spec(T(0.001, 0, 105.0, 0, -0.001, 21.0), Crs(True, 4326), 100, 100)
        self.assertEqual(s.xlabel.split(" ")[0], "Longitude")
        self.assertAlmostEqual(s.aspect, 1 / math.cos(math.radians(20.95)), places=6)
        self.assertEqual(georef.format_degrees(105.25, False), "105.250°E")
        self.assertEqual(georef.format_degrees(-3.5, True, 1), "3.5°S")

    def test_not_georeferenced_falls_back_to_pixels(self):
        for t, c in ((None, None), (T(1, 0, 0, 0, 1, 0), None)):
            s = georef.axis_spec(t, c, 10, 10)
            self.assertIsNone(s.extent); self.assertIn("pixel", s.xlabel)

    def test_rotated_grid_falls_back(self):
        self.assertIsNone(georef.axis_spec(T(30, 5, 0, 5, -30, 0), Crs(False, 1), 10, 10).extent)

    def test_pixel_map_roundtrip(self):
        t = T(30, 0, 500000, 0, -30, 4000000)
        x, y = georef.pixel_to_map(t, 2.5, 4.5)            # pixel centre
        self.assertEqual((x, y), (500135.0, 3999925.0))
        self.assertEqual(georef.map_to_pixel(t, x, y), (2, 4))
        self.assertIsNone(georef.map_to_pixel(T(30, 1, 0, 0, -30, 0), 0, 0))

    def test_unknown_crs_says_map_units(self):
        s = georef.axis_spec(T(2, 0, 10, 0, -2, 50), None, 5, 5)
        self.assertIsNotNone(s.extent); self.assertIn("map units", s.xlabel)


if __name__ == "__main__":
    unittest.main()
