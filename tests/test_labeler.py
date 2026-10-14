import os
import tempfile
import unittest
from collections import namedtuple

import numpy as np

import labeler

Tr = namedtuple("Tr", "a b c d e f")


class Labeler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "p.csv")
        with open(self.path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("id,lon,lat,ref_label,note\n1,44.1,15.2,,\n2,44.2,15.3,flood,ok\n3,44.3,15.4,,\n")

    def test_read_write_roundtrip(self):
        rows, fields = labeler.read_points(self.path)
        self.assertEqual([r["id"] for r in rows], [1, 2, 3]); self.assertEqual(rows[1]["ref_label"], "flood")
        rows[0]["ref_label"] = "not_flood"; rows[2]["note"] = "cloud"
        labeler.write_points(self.path, rows, fields)
        rows2, _ = labeler.read_points(self.path)
        self.assertEqual(rows2[0]["ref_label"], "not_flood"); self.assertEqual(rows2[2]["note"], "cloud")
        self.assertEqual(rows2[0]["lon"], 44.1)
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_next_and_progress(self):
        rows, _ = labeler.read_points(self.path)
        self.assertEqual(labeler.next_unlabeled(rows, 0), 0)
        self.assertEqual(labeler.next_unlabeled(rows, 1), 2)
        rows[0]["ref_label"] = "uncertain"; rows[2]["ref_label"] = "flood"
        self.assertIsNone(labeler.next_unlabeled(rows, 0))
        self.assertEqual(labeler.progress(rows), (2, 1, 3))

    def test_missing_columns(self):
        bad = os.path.join(self.tmp.name, "bad.csv")
        open(bad, "w").write("id,lat\n1,2\n")
        with self.assertRaises(ValueError):
            labeler.read_points(bad)

    def test_chip(self):
        d = np.arange(100 * 100).reshape(100, 100)
        tr = Tr(10, 0, 0, 0, -10, 1000)
        win, (r, c), ext = labeler.chip(d, tr, 505, 505, 5)
        self.assertEqual(win.shape, (11, 11)); self.assertAlmostEqual(r, 5.5, delta=0.1); self.assertAlmostEqual(c, 5.5, delta=0.1)
        self.assertEqual(win[5, 5], d[49, 50]); self.assertEqual(ext, (450.0, 560.0, 450.0, 560.0))
        edge, _, _ = labeler.chip(d, tr, 5, 995, 5)
        self.assertEqual(edge.shape, (6, 6))
        self.assertIsNone(labeler.chip(d, tr, -5, 995, 5))
        self.assertIsNone(labeler.chip(d, Tr(10, 1, 0, 0, -10, 1000), 5, 995, 5))
        rgb = np.zeros((3, 100, 100)); self.assertEqual(labeler.chip(rgb, tr, 505, 505, 5)[0].shape, (3, 11, 11))

    def test_urls(self):
        self.assertIn("15.200000,44.100000", labeler.earth_url(44.1, 15.2))
        self.assertIn("44.100000%2C15.200000", labeler.wayback_url(44.1, 15.2))


if __name__ == "__main__":
    unittest.main()


class Render(unittest.TestCase):
    def test_render_chips(self):
        from matplotlib.figure import Figure
        rgb = np.random.default_rng(0).random((3, 20, 20))
        items = [dict(name="before", window=rgb, point_xy=(10, 10), extent=(0, 20, 0, 20)),
                 dict(name="after", window=rgb[0], point_xy=(10, 10), extent=(0, 20, 0, 20)),
                 dict(name="far", window=None, note="outside")]
        fig = Figure(); labeler.render_chips(fig, items)
        self.assertEqual(len(fig.axes), 3)
        labeler.render_chips(fig, [])           # nothing selected: still no crash
        flat = dict(name="flat", window=np.ones((5, 5)), point_xy=None, extent=None)
        labeler.render_chips(fig, [flat])
