import math, sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import terrain as T


class Tf:                      # affine-like stand-in (a, e, f only)
    def __init__(self, a, e, f=0.0): self.a, self.e, self.f = a, e, f


class Geo:
    is_geographic = True


class TestSurface(unittest.TestCase):
    def ramp(self, rows=30, cols=30, d_row=0.0, d_col=0.0):
        r, c = np.mgrid[0:rows, 0:cols].astype(float)
        return d_row * r + d_col * c

    def test_slope_uses_correct_axis_for_anisotropic_cells(self):
        dem = self.ramp(d_row=1.0)                 # +1 m per row; cells 30 m wide (x) and 10 m tall (y)
        s = T.slope(dem, xres=30, yres=10)
        self.assertAlmostEqual(float(s[15, 15]), math.degrees(math.atan(0.1)), places=4)
        pct = T.slope(dem, xres=30, yres=10, unit="percent_rise")
        self.assertAlmostEqual(float(pct[15, 15]), 10.0, places=4)

    def test_aspect_compass_directions(self):
        east_facing = self.ramp(d_col=-1.0)        # descends eastwards
        south_facing = self.ramp(d_row=-1.0)       # rows run south, descends southwards
        north_facing = self.ramp(d_row=1.0)
        west_facing = self.ramp(d_col=1.0)
        for dem, want in ((east_facing, 90), (south_facing, 180), (west_facing, 270), (north_facing, 0)):
            a = float(T.aspect(dem, 10, 10)[15, 15])
            self.assertAlmostEqual(min(a, 360 - a) if want == 0 else a, want, places=4)
        self.assertEqual(float(T.aspect(np.ones((5, 5)), 10, 10)[2, 2]), -1.0)

    def test_hillshade_flat_equals_sin_altitude(self):
        hs = T.hillshade(np.zeros((9, 9)), 10, 10, azimuth=315, altitude=45)
        self.assertTrue(np.all(hs == round(255 * math.sin(math.radians(45)))))

    def test_hillshade_is_brighter_on_the_sunlit_side(self):
        east_facing = self.ramp(d_col=-0.5)
        north_facing = self.ramp(d_row=0.5)
        self.assertGreater(T.hillshade(east_facing, 10, 10, azimuth=90)[15, 15], T.hillshade(east_facing, 10, 10, azimuth=270)[15, 15])
        self.assertGreater(T.hillshade(north_facing, 10, 10, azimuth=0)[15, 15], T.hillshade(north_facing, 10, 10, azimuth=180)[15, 15])

    def test_hillshade_nan_is_zero(self):
        dem = np.zeros((9, 9)); dem[4, 4] = np.nan
        self.assertEqual(int(T.hillshade(dem, 10, 10)[4, 4]), 0)

    def test_ruggedness(self):
        z = np.zeros((7, 7)); self.assertEqual(float(T.ruggedness(z).max()), 0.0)
        z[3, 3] = 3.0
        tri = T.ruggedness(z)
        self.assertAlmostEqual(float(tri[3, 3]), math.sqrt(8 * 9))
        self.assertAlmostEqual(float(tri[3, 4]), 3.0)

    def test_cell_size_geographic_is_metres(self):
        xres, yres = T.cell_size(Tf(0.001, -0.001, 45.05), (100, 100), Geo())
        self.assertAlmostEqual(yres, 111.0, delta=1.0)
        self.assertAlmostEqual(xres, 111.3 * math.cos(math.radians(45.0)) * 1.0, delta=1.0)
        self.assertEqual(T.cell_size(Tf(30, -30), (10, 10), None), (30.0, 30.0))

    def test_clean_dem_and_voids(self):
        d = np.array([[1, -9999, 3], [4, 5, -3.4028235e38]], np.float32)
        c = T.clean_dem(d, nodata=None)
        self.assertEqual(int(np.isnan(c).sum()), 2)
        self.assertTrue(np.isnan(T.clean_dem(np.array([[7.0, 7.0]]), nodata=7.0)).all())
        dem = np.arange(100, dtype=np.float32).reshape(10, 10); dem[4, 4] = np.nan
        self.assertFalse(np.isnan(T.fill_small_voids(dem)).any())
        big = dem.copy(); big[2:8, 2:8] = np.nan
        self.assertTrue(np.isnan(T.fill_small_voids(big, max_px=1)[4, 4]))
        sm = T.smooth_nan(big); self.assertTrue(np.isnan(sm[4, 4])); self.assertFalse(np.isnan(sm[0, 0]))


class TestContours(unittest.TestCase):
    def test_all_levels_and_coordinates(self):
        r, c = np.mgrid[0:20, 0:50].astype(float)
        dem = c * 1.0                                   # 0..49
        lines = T.contour_lines(dem, a=10, c=1000, e=-10, f=5000, interval=10)
        levels = sorted({l for l, _ in lines})
        self.assertEqual(levels[-4:], [10.0, 20.0, 30.0, 40.0])
        for lev, xy in lines:
            if lev == 20.0:
                self.assertAlmostEqual(float(xy[:, 0].mean()), 1000 + (20 + 0.5) * 10, places=6)
                self.assertGreater(float(xy[:, 1].max() - xy[:, 1].min()), 100)   # runs across the rows
    def test_errors(self):
        with self.assertRaises(ValueError): T.contour_lines(np.zeros((5, 5)), 1, 0, -1, 0, interval=0)
        with self.assertRaises(ValueError): T.contour_lines(np.zeros((5, 5)), 1, 0, -1, 0, interval=5)
        with self.assertRaises(ValueError): T.contour_lines(np.arange(10000.0).reshape(100, 100), 1, 0, -1, 0, interval=1, max_levels=50)


def valley(rows=21, cols=21):
    r, c = np.mgrid[0:rows, 0:cols].astype(float)
    return 0.5 * (rows - 1 - r) + np.abs(c - (cols - 1) / 2)


class TestHydrology(unittest.TestCase):
    def test_priority_flood_fills_pit(self):
        dem = valley(); dem[8, 3] = -5.0                 # an isolated pit on the valley wall
        filled = T.fill_depressions(dem)
        self.assertGreater(filled[8, 3], dem[8, 3])
        fd = T.flow_direction(filled, 10, 10)
        interior = fd[1:-1, 1:-1]
        self.assertTrue((interior != 0).all(), "every interior cell must drain after filling")
        acc = T.flow_accumulation(fd)
        self.assertEqual(int(acc.max()), dem.size)       # everything drains to one outlet

    def test_flat_filled_area_still_drains(self):
        dem = np.full((15, 15), 5.0); dem[0, :] = 5.0; dem[7, 7] = 1.0
        filled = T.fill_depressions(dem)
        fd = T.flow_direction(filled, 1, 1)
        self.assertTrue((fd[1:-1, 1:-1] != 0).all())

    def test_accumulation_valley_outlet(self):
        dem = valley()
        fd = T.flow_direction(T.fill_depressions(dem), 10, 10)
        acc = T.flow_accumulation(fd)
        self.assertEqual(int(acc[20, 10]), dem.size)
        self.assertEqual(int(acc[0, 0]), 1)

    def test_accumulation_does_not_depend_on_raster_order(self):
        """Flow towards row 0: downstream cells have SMALLER indices. A reverse-raster-order loop gets this wrong."""
        r = np.arange(12, dtype=float)[:, None] * np.ones((1, 3))
        r[:, 0] += 50; r[:, 2] += 50                     # walls: single channel in the middle column
        fd = T.flow_direction(r, 1, 1)
        acc = T.flow_accumulation(fd)
        self.assertEqual(int(acc[0, 1]), 36)             # all 3x12 cells drain north through the middle column
    def test_accumulation_counts(self):
        fd = np.zeros((1, 6), np.uint8); fd[0, :5] = 1   # chain flowing east
        acc = T.flow_accumulation(fd)
        self.assertEqual(acc.tolist(), [[1, 2, 3, 4, 5, 6]])
        fd[0, 2] = T.NODATA_DIR
        acc = T.flow_accumulation(fd)
        self.assertEqual(int(acc[0, 2]), 0)

    def network(self):
        fd = np.zeros((4, 5), np.uint8)
        fd[0, 0] = 2      # SE -> (1,1)
        fd[2, 0] = 128    # NE -> (1,1)
        fd[1, 1] = 1; fd[1, 2] = 1; fd[1, 3] = 1      # east
        fd[0, 3] = 4      # S -> (1,3)
        streams = np.zeros((4, 5), np.uint8)
        for rc in ((0, 0), (2, 0), (1, 1), (1, 2), (1, 3), (1, 4), (0, 3)): streams[rc] = 1
        return fd, streams

    def test_strahler_and_shreve(self):
        fd, streams = self.network()
        s = T.stream_order(fd, streams, "strahler"); h = T.stream_order(fd, streams, "shreve")
        self.assertEqual((int(s[0, 0]), int(s[1, 1]), int(s[1, 3]), int(s[1, 4])), (1, 2, 2, 2))
        self.assertEqual((int(h[0, 0]), int(h[1, 1]), int(h[1, 3]), int(h[1, 4])), (1, 2, 3, 3))
        self.assertEqual(int(s[3, 3]), 0)
        with self.assertRaises(ValueError): T.stream_order(fd, streams, "horton")

    def test_stream_segments(self):
        fd, streams = self.network()
        order = T.stream_order(fd, streams)
        segs = T.stream_segments(fd, order, a=10, c=0, e=-10, f=40)
        self.assertEqual(len(segs), 6)
        o, p0, p1 = [s for s in segs if s[1] == (5.0, 35.0)][0]     # cell (0,0) centre
        self.assertEqual(p1, (15.0, 25.0))                           # -> cell (1,1) centre

    def test_extract_streams(self):
        acc = np.array([[1, 1, 50, 100]])
        self.assertEqual(T.extract_streams(acc, "fixed", 50).tolist(), [[0, 0, 1, 1]])
        self.assertEqual(int(T.extract_streams(acc, "percentile", 0).sum()), 2)
        self.assertEqual(int(T.extract_streams(np.ones((3, 3), int), "percentile", 90).sum()), 0)


if __name__ == "__main__":
    unittest.main()
