#!/usr/bin/env python3
"""Space GeoVision - application entry point."""
import os

# Windows 11 24H2 and later no longer ship `wmic`. joblib/loky (used by scikit-learn: K-Means, Random Forest, SVM tuning)
# calls it to count physical cores and prints a long, harmless traceback when it is missing. Giving the count up front
# avoids that call. Must be set before scikit-learn / joblib are imported.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import faulthandler
import logging
import logging.handlers
import sys
import traceback
from pathlib import Path

logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

from resources import resource_path
from ui import SatelliteImageApp


def _make_splash():
    """Splash screen from icons/Pr_icon.jpeg if it is bundled; the app starts fine without it."""
    pix = QPixmap(resource_path("Pr_icon.jpeg"))
    if pix.isNull():
        return None
    pix = pix.scaled(800, 1000, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
    splash.show()
    return splash


LOG_DIR = Path.home() / ".space_geovision"


def _setup_logging():
    """Log to console and to ~/.space_geovision/space_geovision.log (rotating); keeps native crash traces too."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "space_geovision.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        faulthandler.enable(open(LOG_DIR / "native_crash.log", "a"))
    except OSError:
        pass  # a read-only home directory must never stop the application


def _install_excepthook():
    """PyQt5 aborts the whole process on an unhandled exception inside a slot. Report it instead."""
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.error("Unhandled exception:\n%s", text)
        try:
            from PyQt5.QtWidgets import QMessageBox
            box = QMessageBox(QMessageBox.Critical, "Unexpected error",
                              f"{exc_type.__name__}: {exc}\n\nYour data in memory is unchanged. "
                              f"Details were written to {LOG_DIR / 'space_geovision.log'}.")
            box.setDetailedText(text)
            box.exec_()
        except Exception:
            pass
    sys.excepthook = hook


def main() -> int:
    _setup_logging()
    # Required by QtWebEngine (the web-map view); must be set BEFORE the QApplication exists.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    _install_excepthook()
    splash = _make_splash()
    app.processEvents()
    window = SatelliteImageApp()
    window.show()
    if splash is not None:
        splash.finish(window)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
