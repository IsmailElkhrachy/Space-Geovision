import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import basins as b  # noqa: E402
import terrain as t  # noqa: E402


def valley(h=12, w=21, c0=10):
    """Both sides slope to a central channel that slopes to the south edge (all cells drain to (h-1, c0))."""
    r, c = np.mgrid[0:h, 0:w].astype(float)
    return 0.5 * np.abs(c - c0) + 0.2 * (h - 1 - r) + 1.0


def two_valleys(h=12, w=21):
    """A ridge at column 10 separates two valleys with channels at columns 4 and 16."""
    r, c = np.mgrid[0:h, 0:w].astype(float)
    return np.minimum(np.abs(c - 4), np.abs(c - 16)) * 0.5 + 0.2 * (h - 1 - r) + 1.0


def hydro(dem, thr=10):
    f = t.fill_depressions(dem)
    fd = t.flow_direction(f, 10.0, 10.0)
    acc = t.flow_accumulation(fd)
    streams = (acc >= thr).astype(np.uint8)
    return f, fd, acc, streams


class TestWatershed(unittest.TestCase):
    def test_single_valley_everything_drains_to_the_outlet(self):
        dem = valley()
        f, fd, acc, _ = hydro(dem)
        row, col = b.snap_to_stream(acc, 11, 9, radius=3)
        m = b.delineate_watershed(fd, row, col)
        self.assertEqual(m.sum(), acc[row, col])      # basin area equals the accumulation at the outlet
        self.assertGreater(m.sum(), 0.9 * dem.size)

    def test_ridge_separates_the_basins(self):
        dem = two_valleys()
        f, fd, acc, _ = hydro(dem)
        m = b.delineate_watershed(fd, 11, 4)
        self.assertTrue(m[:, :10].any())
        self.assertFalse(m[:, 11:].any())             # nothing east of the ridge drains to the left outlet
        self.assertEqual(m.sum(), acc[11, 4])

    def test_outlet_on_nodata_and_outside_raise(self):
        fd = np.full((3, 3), 255, np.uint8)
        with self.assertRaises(ValueError):
            b.delineate_watershed(fd, 1, 1)
        with self.assertRaises(ValueError):
            b.delineate_watershed(fd, 5, 5)

    def test_snap_moves_to_the_highest_accumulation(self):
        acc = np.ones((9, 9), np.int64)
        acc[4, 6] = 50
        self.assertEqual(b.snap_to_stream(acc, 4, 4, radius=3), (4, 6))
        self.assertEqual(b.snap_to_stream(acc, 0, 0, radius=1), (0, 0))   # nothing to snap to


class TestMorphometrics(unittest.TestCase):
    def _y_network(self):
        fd = np.full((3, 3), 255, np.uint8)
        fd[0, 0], fd[0, 2], fd[1, 1], fd[2, 1] = 2, 8, 4, 4   # two tributaries join at (1,1); outlet (2,1)
        streams = np.zeros((3, 3), np.uint8)
        streams[0, 0] = streams[0, 2] = streams[1, 1] = streams[2, 1] = 1
        return fd, streams

    def test_orders_segments_and_lengths(self):
        fd, streams = self._y_network()
        order = t.stream_order(fd, streams, "strahler")
        mask = streams > 0
        dem = np.where(mask, 10.0, np.nan)
        dem[0, 0] = dem[0, 2] = 13.0; dem[1, 1] = 11.0; dem[2, 1] = 10.0
        m = b.basin_morphometrics(dem, mask, fd, streams, order, 10.0, 10.0, (2, 1))
        self.assertEqual(m["Segments per order"], {1: 2, 2: 1})
        self.assertAlmostEqual(m["Mean bifurcation ratio Rb"], 2.0)
        self.assertEqual(m["Maximum stream order (Strahler)"], 2)
        self.assertAlmostEqual(m["Total stream length (km)"], (2 * math.hypot(10, 10) + 10) / 1000.0, places=6)
        self.assertAlmostEqual(m["Area (km2)"], 4 * 100 / 1e6)
        self.assertAlmostEqual(m["Basin length, longest flow path (km)"], (math.hypot(10, 10) + 10) / 1000.0, places=6)
        self.assertAlmostEqual(m["Relief (m)"], 3.0)

    def test_perimeter_of_a_rectangle(self):
        h, w = 4, 6
        fd = np.full((h, w), 4, np.uint8)             # everything flows south
        streams = np.zeros((h, w), np.uint8); streams[:, 2] = 1
        order = t.stream_order(fd, streams)
        mask = np.ones((h, w), bool)
        dem = np.arange(h * w, dtype=float).reshape(h, w)[::-1]
        m = b.basin_morphometrics(dem, mask, fd, streams, order, 30.0, 30.0, (h - 1, 2))
        self.assertAlmostEqual(m["Perimeter (km)"], 2 * (4 + 6) * 30 / 1000.0)
        self.assertAlmostEqual(m["Area (km2)"], 24 * 900 / 1e6)
        self.assertAlmostEqual(m["Circularity ratio Rc (4 pi A / P^2)"], 4 * math.pi * m["Area (km2)"] / m["Perimeter (km)"] ** 2)

    def test_hypsometric_integral_bounds(self):
        dem = valley()
        mask = np.ones(dem.shape, bool)
        ra, rh, hi = b.hypsometric_curve(dem, mask)
        self.assertTrue(0 < hi < 1)
        self.assertAlmostEqual(ra[0], 1.0); self.assertAlmostEqual(ra[-1], 0.0, delta=0.01)
        self.assertTrue(np.all(np.diff(ra) <= 1e-12))
        area_under = abs(float(np.sum((rh[1:] + rh[:-1]) / 2.0 * np.diff(ra))))
        self.assertAlmostEqual(area_under, hi, delta=0.03)
        with self.assertRaises(ValueError):
            b.hypsometric_curve(np.ones((4, 4)), np.ones((4, 4), bool))


class TestPipeline(unittest.TestCase):
    def test_full_chain_on_a_valley(self):
        dem = valley()
        f, fd, acc, streams = hydro(dem, thr=8)
        r, c = b.snap_to_stream(acc, 11, 9, radius=3)
        mask = b.delineate_watershed(fd, r, c)
        order = t.stream_order(fd, streams, "strahler")
        m = b.basin_morphometrics(dem, mask, fd, streams, order, 10.0, 10.0, (r, c))
        self.assertAlmostEqual(m["Area (km2)"], mask.sum() * 100 / 1e6)
        self.assertGreater(m["Drainage density Dd (km/km2)"], 0)
        self.assertGreaterEqual(m["Maximum stream order (Strahler)"], 1)
        ra, rh, hi = b.hypsometric_curve(dem, mask)
        text = b.format_report(m, hi)
        self.assertIn("Drainage density", text)
        self.assertIn("Hypsometric integral", text)


class TestHandTwi(unittest.TestCase):
    def test_hand_on_a_tilted_plane_equals_height_above_the_channel(self):
        r, c = np.mgrid[0:6, 0:8].astype(float)
        dem = c * 1.0 + 5.0                           # falls to the west; the stream is column 0
        fd = t.flow_direction(dem, 10.0, 10.0)
        streams = np.zeros(dem.shape, np.uint8); streams[:, 0] = 1
        hnd, dist = b.hand(dem, fd, streams, 10.0, 10.0)
        np.testing.assert_allclose(hnd, np.tile(np.arange(8.0), (6, 1)), atol=1e-5)
        np.testing.assert_allclose(dist, np.tile(np.arange(8.0) * 10.0, (6, 1)), atol=1e-4)

    def test_hand_nan_for_cells_that_never_reach_a_stream(self):
        dem = np.tile(np.arange(8.0)[::-1], (4, 1))   # falls to the east, but the "stream" is at the west edge
        fd = t.flow_direction(dem, 10.0, 10.0)
        streams = np.zeros(dem.shape, np.uint8); streams[:, 0] = 1
        hnd, _ = b.hand(dem, fd, streams, 10.0, 10.0)
        self.assertTrue(np.isnan(hnd[:, 1:]).all())
        self.assertTrue((hnd[:, 0] == 0).all())

    def test_hand_nonnegative_on_a_real_valley(self):
        f, fd, acc, streams = hydro(valley())
        hnd, _ = b.hand(f, fd, streams, 10.0, 10.0)
        v = hnd[np.isfinite(hnd)]
        self.assertTrue((v >= 0).all())
        self.assertTrue((hnd[streams > 0] == 0).all())
        self.assertGreater(np.nanmax(hnd), 2.0)

    def test_twi_higher_in_the_flat_convergent_cell(self):
        dem = valley()
        f, fd, acc, _ = hydro(dem)
        w = b.twi(acc, f, 10.0, 10.0)
        self.assertTrue(np.isfinite(w).all())
        self.assertGreater(w[10, 10], w[0, 0])      # channel near the outlet vs a ridge-top corner
        s = b.spi(acc, f, 10.0, 10.0)
        self.assertGreater(s[10, 10], s[0, 0])

    def test_twi_nan_on_nodata(self):
        dem = valley(); dem[3:5, 3:5] = np.nan
        f = t.fill_depressions(dem)
        fd = t.flow_direction(f, 10, 10); acc = t.flow_accumulation(fd)
        w = b.twi(acc, f, 10.0, 10.0)
        self.assertTrue(np.isnan(w[3:5, 3:5]).all())


if __name__ == "__main__":
    unittest.main()
