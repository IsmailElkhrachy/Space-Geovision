import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import raster_expr as rx


class TestExpr(unittest.TestCase):
    def setUp(self):
        self.L = {"NIR": np.array([[0.6, 0.0], [0.4, 0.5]]), "Red": np.array([[0.2, 0.0], [0.4, 0.1]]),
                  "a.tif": np.ones((1, 2, 2))}

    def test_ndvi_and_zero_division_is_nan(self):
        r = rx.evaluate('("NIR" - "Red") / ("NIR" + "Red")', self.L)
        self.assertAlmostEqual(float(r[0, 0]), 0.5, places=6)
        self.assertTrue(np.isnan(r[0, 1]))                       # 0/0
        self.assertEqual(float(r[1, 0]), 0.0)                    # no epsilon bias

    def test_mask_with_where_and_comparison(self):
        r = rx.evaluate('where("NIR" > 0.45, 1, 0)', self.L)
        self.assertEqual(r.tolist(), [[1.0, 0.0], [0.0, 1.0]])
        m = rx.evaluate('("NIR" > 0.3) & ("Red" < 0.3)', self.L)
        self.assertEqual(m.astype(int).tolist(), [[1, 0], [0, 1]])

    def test_single_band_3d_layer_and_layer_names_with_dots(self):
        self.assertEqual(rx.evaluate('"a.tif" * 3', self.L).tolist(), [[3.0, 3.0], [3.0, 3.0]])

    def test_rejects_code_execution(self):
        for bad in ['"NIR".__class__', '__import__("os").system("echo x")', '"NIR"[0]', 'open("f")', 'lambda: 1',
                    '[x for x in "NIR"]', '"NIR" if 1 else 2', 'np.sqrt("NIR")', 'exec("1")', '(1).__class__']:
            with self.assertRaises(rx.FormulaError, msg=bad):
                rx.evaluate(bad, self.L)

    def test_errors_are_friendly(self):
        with self.assertRaises(rx.FormulaError): rx.evaluate("", self.L)
        with self.assertRaises(rx.FormulaError): rx.evaluate("1 + 2", self.L)
        with self.assertRaises(rx.FormulaError): rx.evaluate('"Missing" + 1', self.L)
        with self.assertRaises(rx.FormulaError): rx.evaluate('"NIR" +', self.L)
        with self.assertRaises(rx.FormulaError): rx.evaluate('"NIR" + "Z"', {**self.L, "Z": np.ones((3, 3))})
        with self.assertRaises(rx.FormulaError): rx.evaluate('0 < "NIR" < 1', self.L)

    def test_infinity_becomes_nan(self):
        self.assertTrue(np.isnan(rx.evaluate('log("Red")', self.L)[0, 1]))


if __name__ == "__main__":
    unittest.main()
