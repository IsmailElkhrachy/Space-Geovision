"""Opens the generated web-map page in headless Chromium (Playwright) with the tile requests faked.
Skipped when Playwright / a browser is not installed."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import webmap as w  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "playwright not installed")
class TestPageInBrowser(unittest.TestCase):
    def test_page_runs_without_js_errors_and_lists_layers(self):
        from PIL import Image
        png = io.BytesIO(); Image.new("RGB", (256, 256), (40, 90, 40)).save(png, "PNG")
        o = {"name": "HAND", "uri": w.png_data_uri(w.render_rgba(np.arange(100.0).reshape(10, 10))),
             "bounds": (17.0, 44.0, 18.0, 45.0), "checked": True}
        poly = {"type": "Polygon", "coordinates": [[[44.2, 17.2], [44.8, 17.2], [44.8, 17.8], [44.2, 17.8], [44.2, 17.2]]]}
        html = w.build_html([o], [{"name": "Basin", "geojson": poly}], (17.0, 44.0, 18.0, 45.0))
        with tempfile.TemporaryDirectory() as d:
            page = Path(d) / "m.html"; page.write_text(html, encoding="utf-8")
            try:
                pw = sync_playwright().start()
                browser = pw.chromium.launch()
            except Exception as exc:                      # no browser binary
                self.skipTest(f"no browser: {exc}")
            try:
                pg = browser.new_page(viewport={"width": 800, "height": 500})
                errors = []
                pg.on("pageerror", lambda e: errors.append(str(e)))
                pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                pg.route("**/*", lambda r: r.continue_() if r.request.url.startswith(("file:", "data:"))
                         else r.fulfill(status=200, content_type="image/png", body=png.getvalue()))
                pg.goto(page.as_uri()); pg.wait_for_timeout(800)
                rows = pg.eval_on_selector_all("#layers label", "e=>e.map(x=>x.textContent.trim())")
                pg.mouse.move(400, 250); pg.mouse.wheel(0, -200); pg.wait_for_timeout(200)
            finally:
                browser.close(); pw.stop()
        self.assertEqual(errors, [])
        self.assertIn("HAND", rows); self.assertIn("Basin", rows)


if __name__ == "__main__":
    unittest.main()
