"""Headless GUI smoke test: builds the main window and triggers every menu action with all modal dialogs neutralised.

Opt-in (needs PyQt5, rasterio, geopandas, ...): ``SG_GUI_SMOKE=1 python -m pytest tests/test_gui_smoke.py -s``
It reports every Python exception raised inside a slot. Actions that need data (no layer loaded) should show a
warning, never raise. NOT run by the author of this change (PyQt5 could not be installed in the build sandbox).
"""
import os, sys, unittest, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN = os.environ.get("SG_GUI_SMOKE") == "1" and importlib.util.find_spec("PyQt5") and importlib.util.find_spec("rasterio")


@unittest.skipUnless(RUN, "set SG_GUI_SMOKE=1 and install the GUI requirements to run")
class TestGuiSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "win32":  # offscreen has no fonts on Windows; use the normal platform there
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5 import QtWidgets, QtCore
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
        cls.QtWidgets = QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        W = QtWidgets
        from PyQt5 import QtPrintSupport as P
        for cls_ in (W.QDialog, P.QPageSetupDialog, P.QPrintDialog, P.QPrintPreviewDialog):
            cls_.exec_ = lambda self, *a, **k: 0
        P.QPrinterInfo.availablePrinters = staticmethod(lambda: [])  # never touch native print drivers in a smoke test
        W.QMessageBox.exec_ = lambda self, *a, **k: W.QMessageBox.Cancel
        W.QMenu.exec_ = lambda self, *a, **k: None
        for n in ("warning", "information", "critical", "question"):
            setattr(W.QMessageBox, n, staticmethod(lambda *a, **k: W.QMessageBox.No))
        for n in ("getOpenFileName", "getSaveFileName"):
            setattr(W.QFileDialog, n, staticmethod(lambda *a, **k: ("", "")))
        W.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([], ""))
        W.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
        for n in ("getItem", "getText", "getDouble", "getInt"):
            setattr(W.QInputDialog, n, staticmethod(lambda *a, **k: ("", False)))
        W.QColorDialog.getColor = staticmethod(lambda *a, **k: W.QColorDialog().currentColor())

    def test_every_menu_action_runs_without_an_unhandled_exception(self):
        import traceback
        import ui
        errors = []
        old_hook = sys.excepthook
        sys.excepthook = lambda t, e, tb: errors.append("".join(traceback.format_exception(t, e, tb)))
        try:
            win = ui.SatelliteImageApp()
            actions = []
            for bar in win.findChildren(self.QtWidgets.QMenuBar):
                stack = list(bar.actions())
                while stack:
                    a = stack.pop()
                    if a.menu():
                        stack.extend(a.menu().actions())
                    elif a.isEnabled() and not a.isSeparator():
                        actions.append(a)
            self.assertGreater(len(actions), 20, "menu bar looks empty")
            for a in actions:
                before = len(errors)
                try:
                    a.trigger()
                except Exception:
                    errors.append(f"{a.text()}: " + traceback.format_exc())
                if len(errors) > before:
                    errors[-1] = f"[{a.text()}] " + errors[-1]
            win.close()
        finally:
            sys.excepthook = old_hook
        self.assertEqual(errors, [], "\n\n".join(errors[:5]))


if __name__ == "__main__":
    unittest.main()
