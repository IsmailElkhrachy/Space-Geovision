"""A classified layer must look the same when re-selected from the layer list (colours, legend, coordinates)."""
import ast
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import georef

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    Figure = None

SRC = open(os.path.join(os.path.dirname(__file__), "..", "ui.py"), encoding="utf-8").read()


def _methods(*names):
    tree = ast.parse(SRC)
    ns = {"np": np, "georef": georef}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            code = ast.get_source_segment(SRC, node)
            import textwrap
            exec(textwrap.dedent(code), ns)
    return {n: ns[n] for n in names}


class T:
    def __init__(self): self.a, self.b, self.c, self.d, self.e, self.f = 10.0, 0, 435768.7, 0, -10.0, 1953819.9


class Crs:
    is_geographic = False
    def to_epsg(self): return 32638


class Canvas:
    def draw(self): pass


@unittest.skipIf(Figure is None, "matplotlib missing")
class TestRedraw(unittest.TestCase):
    def setUp(self):
        m = _methods("_redraw_classified_layer", "display_classified_image")

        class Win:
            _redraw_classified_layer = m["_redraw_classified_layer"]
            display_classified_image = m["display_classified_image"]
            def __init__(s):
                s.fig = Figure(); s.canvas = Canvas(); s.layers = {}
            def get_cluster_color(s, k):
                return [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0)][k % 5]
            def _inherited_profile(s, shape): return None
        self.w = Win()
        arr = np.random.default_rng(0).integers(1, 5, (40, 50)).astype(np.uint8)
        prof = {"transform": T(), "crs": Crs()}
        self.layer = {"image": arr[None], "profile": prof, "classified": True,
                      "class_names": {1: "bare_ground", 2: "built_up", 3: "rock", 4: "vegetation"}, "title": "RF"}
        self.w.layers["rf"] = self.layer

    def _state(self):
        ax = self.w.ax
        im = ax.images[0]
        leg = [t.get_text() for lg in (ax.get_legend(),) if lg for t in lg.get_texts()]
        return im.get_extent(), leg, ax.get_title(), np.array(im.get_cmap()(np.linspace(0, 1, 5)))

    def test_same_look_after_redraw(self):
        self.assertTrue(self.w._redraw_classified_layer(self.layer, "rf"))
        first = self._state()
        self.w.fig.clf(); self.w.ax = self.w.fig.add_subplot(111)  # another layer was shown in between
        self.assertTrue(self.w._redraw_classified_layer(self.layer, "rf"))
        again = self._state()
        self.assertEqual(first[0], again[0]); self.assertEqual(first[1], again[1]); self.assertEqual(first[2], again[2])
        np.testing.assert_allclose(first[3], again[3])
        self.assertEqual(first[1], ["bare_ground", "built_up", "rock", "vegetation"])
        self.assertAlmostEqual(first[0][0], 435768.7)  # real coordinates, not pixels

    def test_continuous_layer_not_hijacked(self):
        self.assertFalse(self.w._redraw_classified_layer({"image": np.zeros((1, 3, 3))}, "x"))

    def test_names_default_when_missing(self):
        lay = dict(self.layer); lay.pop("class_names")
        self.assertTrue(self.w._redraw_classified_layer(lay, "rf"))
        self.assertEqual([t.get_text() for t in self.w.ax.get_legend().get_texts()],
                         ["Class 1", "Class 2", "Class 3", "Class 4"])

    def test_producers_flag_their_layers(self):
        # K-Means, Mean-Shift and supervised results must be stored as classified layers
        self.assertGreaterEqual(SRC.count("'classified': True"), 3)

    def test_display_raster_layer_uses_hook_first(self):
        i = SRC.index("def display_raster_layer")
        self.assertLess(SRC.index("_redraw_classified_layer(layer_data, layer_name)", i), SRC.index("self.fig.clf()", i))


if __name__ == "__main__":
    unittest.main()
