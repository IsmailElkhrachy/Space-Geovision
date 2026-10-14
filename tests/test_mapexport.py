import os
import struct
import tempfile
import unittest
from collections import namedtuple

import numpy as np

import mapexport as me

Tr = namedtuple("Tr", "a b c d e f")


def _layer(rows=200, cols=300, geographic=True, **kw):
    yy, xx = np.mgrid[0:rows, 0:cols]
    data = (xx + yy).astype(np.float32)
    tr = Tr(0.001, 0, 44.0, 0, -0.001, 15.0) if geographic else Tr(30, 0, 500000, 0, -30, 3000000)
    return me.RasterLayer(data, tr, geographic=geographic, crs_text="EPSG:4326", name="DEM", **kw)


class Geometry(unittest.TestCase):
    def test_page_inches(self):
        w, h = me.page_inches(me.ExportSpec(page="A4", orientation="portrait"), 1.0)
        self.assertAlmostEqual(w * 25.4, 210); self.assertAlmostEqual(h * 25.4, 297)
        w, h = me.page_inches(me.ExportSpec(page="A4"), 2.0)       # auto, wide map
        self.assertGreater(w, h)
        with self.assertRaises(ValueError):
            me.page_inches(me.ExportSpec(page="B9"), 1.0)

    def test_crop_window(self):
        tr = Tr(1, 0, 0, 0, -1, 100)
        self.assertEqual(me.crop_window(tr, (100, 100), None), (0, 100, 0, 100))
        self.assertEqual(me.crop_window(tr, (100, 100), (10, 20, 30, 50)), (50, 80, 10, 30))
        self.assertEqual(me.crop_window(tr, (100, 100), (-50, -50, 5, 5))[2:], (0, 5))
        with self.assertRaises(ValueError):
            me.crop_window(tr, (100, 100), (200, 200, 300, 300))
        with self.assertRaises(ValueError):
            me.crop_window(tr, (100, 100), (5, 5, 5, 9))

    def test_decimate(self):
        d = np.arange(64, dtype=np.float32).reshape(8, 8)
        out, k = me.decimate_to(d, 4, 4)
        self.assertEqual((k, out.shape), (2, (4, 4)))
        self.assertAlmostEqual(float(out[0, 0]), (0 + 1 + 8 + 9) / 4)
        d[0, 0] = np.nan
        out, _ = me.decimate_to(d, 4, 4)
        self.assertAlmostEqual(float(out[0, 0]), (1 + 8 + 9) / 3)
        lab = np.arange(64, dtype=np.uint8).reshape(8, 8)
        out, k = me.decimate_to(lab, 4, 4, categorical=True)
        self.assertEqual(int(out[1, 1]), int(lab[2, 2]))
        same, k = me.decimate_to(d, 100, 100)
        self.assertEqual(k, 1)

    def test_polygon_mask_with_hole(self):
        tr = Tr(1, 0, 0, 0, -1, 10)
        outer = np.array([[1, 1], [9, 1], [9, 9], [1, 9]], float)
        hole = np.array([[4, 4], [6, 4], [6, 6], [4, 6]], float)
        m = me.polygon_mask([outer, hole], tr, (10, 10))
        self.assertEqual(int(m.sum()), 64 - 4)
        self.assertFalse(m[5, 5]); self.assertTrue(m[2, 2]); self.assertFalse(m[0, 0])

    def test_nice_length_and_metres(self):
        self.assertEqual(me.nice_length(730), 500); self.assertEqual(me.nice_length(2400), 2000)
        self.assertEqual(me.nice_length(1), 1); self.assertEqual(me.nice_length(0), 0)
        self.assertAlmostEqual(me.metres_per_unit(False, 10), 1)
        self.assertLess(me.metres_per_unit(True, 60), me.metres_per_unit(True, 0))

    def test_sample_to_grid(self):
        src = np.arange(16, dtype=np.float32).reshape(4, 4)
        s = Tr(1, 0, 0, 0, -1, 4)
        out = me.sample_to_grid(src, s, (2, 2), Tr(1, 0, 1, 0, -1, 3))
        self.assertEqual(out.tolist(), [[5, 6], [9, 10]])
        out = me.sample_to_grid(src, s, (1, 1), Tr(1, 0, 10, 0, -1, 3))
        self.assertTrue(np.isnan(out).all())


def _png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


class Export(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def p(self, name):
        return os.path.join(self.tmp.name, name)

    def test_png_dpi_controls_pixels(self):
        lay = _layer()
        for dpi in (72, 150):
            path = self.p(f"m{dpi}.png")
            wmm, hmm = me.export_map(path, lay, me.ExportSpec(fmt="png", dpi=dpi, page="A4", orientation="portrait"))
            w, h = _png_size(path)
            self.assertAlmostEqual(w, round(210 / 25.4 * dpi), delta=1)
            self.assertAlmostEqual(h, round(297 / 25.4 * dpi), delta=1)
        self.assertAlmostEqual(wmm, 210, delta=0.5)

    def test_pdf_size_scales_with_dpi(self):
        rng = np.random.default_rng(0)
        data = rng.random((1500, 1500)).astype(np.float32)
        lay = me.RasterLayer(data, Tr(0.001, 0, 44, 0, -0.001, 15), geographic=True, name="noise")
        sizes = []
        for dpi in (60, 300):
            path = self.p(f"m{dpi}.pdf")
            me.export_map(path, lay, me.ExportSpec(fmt="pdf", dpi=dpi, show_grid=False, show_colorbar=False))
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(4), b"%PDF")
            sizes.append(os.path.getsize(path))
        self.assertGreater(sizes[1], sizes[0] * 3)

    def test_other_formats(self):
        lay = _layer()
        for fmt in ("jpg", "tif", "svg", "pdf"):
            path = self.p("m." + fmt)
            me.export_map(path, lay, me.ExportSpec(fmt=fmt, dpi=60, title="T", subtitle="S", note="n", author="me"))
            self.assertGreater(os.path.getsize(path), 500)

    def test_projected_rgb_and_overlay_and_vectors(self):
        lay = _layer(geographic=False)
        rgb = np.stack([lay.data, lay.data[::-1], lay.data[:, ::-1]]).astype(np.float32)
        base = me.RasterLayer(rgb, lay.transform, geographic=False, crs_text="EPSG:32638", name="rgb")
        ov = _layer(geographic=False, cmap="Blues")
        vec = me.VectorLayer("riv", lines=[np.array([[500000, 2999000], [503000, 2996000]])],
                             rings=[np.array([[500500, 2998500], [502000, 2998500], [502000, 2997000]])],
                             points=[(501000, 2998000)])
        me.export_map(self.p("a.png"), base, me.ExportSpec(fmt="png", dpi=60), vectors=[vec])
        me.export_map(self.p("b.png"), lay, me.ExportSpec(fmt="png", dpi=60), overlay=ov, vectors=[vec])

    def test_clip_masks_outside(self):
        lay = _layer()
        ring = np.array([[44.05, 14.95], [44.2, 14.95], [44.2, 14.85], [44.05, 14.85]])
        spec = me.ExportSpec(fmt="png", dpi=100, clip_rings=[ring], show_grid=False, show_scalebar=False,
                             show_north=False, show_colorbar=False, show_border=False, show_footer=False,
                             outline_clip=False, margin_mm=5)
        arr, extent, sub = me._prepare(lay, me.ExportSpec(extent=me.rings_bbox([ring]), clip_rings=[ring]), (400, 400))
        self.assertTrue(np.isnan(arr).sum() == 0 or np.isfinite(arr).any())
        # crop defaults to polygon bbox
        fig = me.compose_figure(lay, spec)
        ax = fig.axes[0]
        x0, x1 = ax.get_xlim()
        self.assertAlmostEqual(x0, 44.05 - 0.003, delta=0.0015); self.assertAlmostEqual(x1, 44.2 + 0.003, delta=0.0015)
        # pixels outside a triangle are masked
        tri = np.array([[44.0, 15.0], [44.3, 15.0], [44.0, 14.8]])
        arr, _, _ = me._prepare(lay, me.ExportSpec(clip_rings=[tri]), (300, 200))
        self.assertTrue(np.isnan(arr).any()); self.assertTrue(np.isfinite(arr).any())
        self.assertTrue(np.isnan(arr[-1, -1])); self.assertTrue(np.isfinite(arr[0, 0]))

    def test_extent_window(self):
        lay = _layer()
        spec = me.ExportSpec(fmt="png", dpi=60, extent=(44.1, 14.8, 44.2, 14.9))
        fig = me.compose_figure(lay, spec)
        x0, x1 = fig.axes[0].get_xlim()
        self.assertAlmostEqual(x0, 44.1, delta=0.0015); self.assertAlmostEqual(x1, 44.2, delta=0.0015)

    def test_nodata_not_averaged(self):
        data = np.full((100, 100), 5.0, np.float32); data[:50] = -9999
        lay = me.RasterLayer(data, Tr(1, 0, 0, 0, -1, 100), nodata=-9999)
        arr, _, _ = me._prepare(lay, me.ExportSpec(), (20, 20))
        self.assertTrue(np.isnan(arr[:10]).all()); self.assertTrue(np.allclose(arr[10:], 5.0))

    def test_errors(self):
        lay = _layer()
        with self.assertRaises(ValueError):
            me.compose_figure(lay, me.ExportSpec(dpi=10))
        with self.assertRaises(ValueError):
            me.compose_figure(lay, me.ExportSpec(fmt="gif"))
        with self.assertRaises(ValueError):
            me.compose_figure(lay, me.ExportSpec(margin_mm=100))
        with self.assertRaises(ValueError):
            me.compose_figure(lay, me.ExportSpec(page="A2", dpi=1200, fmt="png"))
        with self.assertRaises(ValueError):
            me.compose_figure(lay, me.ExportSpec(extent=(0, 0, 1, 1)))


if __name__ == "__main__":
    unittest.main()


class _G:
    def __init__(self, t, **kw):
        self.geom_type = t; self.is_empty = False; self.__dict__.update(kw)


class _C:
    def __init__(self, c): self.coords = c


class GeometryArrays(unittest.TestCase):
    def test_split(self):
        sq = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        poly = _G("Polygon", exterior=_C(sq), interiors=[_C([(1, 1), (2, 1), (2, 2), (1, 1)])])
        line = _G("LineString", coords=[(0, 0, 5), (1, 1, 5)])
        pt = _G("Point", x=1.5, y=2.5)
        multi = _G("GeometryCollection", geoms=[poly, line, pt])
        lines, rings, points = me.geometry_to_arrays(multi)
        self.assertEqual((len(lines), len(rings), len(points)), (1, 2, 1))
        self.assertEqual(lines[0].shape, (2, 2)); self.assertEqual(points[0], (1.5, 2.5))
        self.assertEqual(me.geometry_to_arrays(None), ([], [], []))
