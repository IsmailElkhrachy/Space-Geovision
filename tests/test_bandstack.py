import os, sys, tempfile, types, unittest
from collections import namedtuple
from unittest import mock
import numpy as np
import bandstack

Tr = namedtuple("Tr", "a b c d e f g h i")
STORE = {}


def fake_rasterio(sources):
    mod = types.ModuleType("rasterio")

    class Src:
        def __init__(self, p):
            self.arr, self.crs, self.tr = sources[p]
            self.height, self.width = self.arr.shape
            self.transform = self.tr
            self.count = 1
            self.profile = {"driver": "GTiff", "dtype": str(self.arr.dtype), "count": 1, "tiled": True,
                            "width": self.width, "height": self.height, "crs": self.crs, "transform": self.tr}
        def read(self, i): return self.arr
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class Dst:
        def __init__(self, p, **kw):
            self.p, self.kw = p, kw; self.bands = {}; self.desc = {}
            open(p, "wb").close()
        def write(self, a, i): self.bands[i] = a
        def set_band_description(self, i, d): self.desc[i] = d
        def __enter__(self): return self
        def __exit__(self, *a):
            STORE[self.p] = (self.kw, self.bands, self.desc); return False

    def op(p, mode="r", **kw):
        return Dst(p, **kw) if mode == "w" else Src(p)
    mod.open = op
    return {"rasterio": mod}


class Stack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.src = {}
        self.tr = Tr(10, 0, 500000, 0, -10, 1900000, 0, 0, 1)

    def make(self, bands=("B02", "B03", "B04", "B08", "B11"), shape=(6, 8), date="2020-02-10"):
        for k, b in enumerate(bands):
            p = os.path.join(self.tmp.name, f"{date}-00_00_{date}-23_59_Sentinel-2_L2A_{b}_(Raw).tiff")
            open(p, "wb").close()
            self.src[p] = (np.full(shape, 100 * (k + 1), np.uint16), "EPSG:32638", self.tr)
        return mock.patch.dict(sys.modules, fake_rasterio(self.src))

    def test_stack_order_and_report(self):
        with self.make():
            rep = bandstack.stack_bands(self.tmp.name)
        self.assertTrue(rep["path"].endswith("2020-02-10_S2L2A_stack.tif"))
        kw, bands, desc = STORE[rep["path"] + ".part"]
        self.assertEqual(kw["count"], 5)
        self.assertEqual([int(bands[i][0, 0]) for i in range(1, 6)], [100, 200, 300, 400, 500])
        self.assertEqual(desc[4], "B08")
        self.assertIn("B11", bandstack.format_report(rep))
        self.assertFalse(os.path.exists(rep["path"] + ".part"))

    def test_missing_band(self):
        with self.make(bands=("B02", "B03", "B04", "B08")):
            with self.assertRaisesRegex(ValueError, "B11"):
                bandstack.stack_bands(self.tmp.name)

    def test_extra_files_ignored_and_b02_not_b08(self):
        with self.make(bands=("B02", "B03", "B04", "B08", "B11", "SCL", "B8A")):
            rep = bandstack.stack_bands(self.tmp.name)
        self.assertEqual(rep["bands"], list(bandstack.DEFAULT_BANDS))

    def test_grid_mismatch(self):
        with self.make():
            k = [p for p in self.src if "_B11_" in p][0]
            self.src[k] = (np.zeros((7, 8), np.uint16), "EPSG:32638", self.tr)
            with self.assertRaisesRegex(ValueError, "same grid"):
                bandstack.stack_bands(self.tmp.name)

    def test_check_pair(self):
        with self.make():
            a = os.path.join(self.tmp.name, "a.tif"); b = os.path.join(self.tmp.name, "b.tif")
            self.src[a] = (np.zeros((6, 8), np.uint16), "EPSG:32638", self.tr)
            self.src[b] = (np.zeros((6, 9), np.uint16), "EPSG:32638", self.tr)
            self.assertIn("NOT identical", bandstack.check_pair(a, b))
            self.src[b] = (np.zeros((6, 8), np.uint16), "EPSG:32638", self.tr)
            self.assertTrue(bandstack.check_pair(a, b).startswith("OK"))


if __name__ == "__main__":
    unittest.main()
