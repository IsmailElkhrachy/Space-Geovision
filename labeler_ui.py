"""Qt window for blind labelling of validation points (see labeler.py)."""
from __future__ import annotations

import os

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QMessageBox, QPushButton, QShortcut, QSpinBox, QVBoxLayout)

import georef
import labeler
import validation


class LabelPointsDialog(QDialog):
    def __init__(self, window, points_csv="", categories=""):
        """categories: comma separated fallback names. Normally they come from the scheme saved with the points
        (validation_scheme.json), or from labels already in the file, so any kind of map can be labelled."""
        super().__init__(window)
        self._fallback = [c.strip() for c in (categories or "").split(",") if c.strip()]
        self.win = window
        self.rows, self.fields, self.idx, self.path = [], [], 0, ""
        self.setWindowTitle("Label validation points (blind)")
        self.setMinimumSize(980, 680)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._shortcuts = []

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.ed_path = QLineEdit(points_csv); self.ed_path.setPlaceholderText("points_to_label.csv")
        b_browse, b_load = QPushButton("Browse…"), QPushButton("Load points")
        b_browse.clicked.connect(self._browse); b_load.clicked.connect(self._load)
        top.addWidget(QLabel("Points file:")); top.addWidget(self.ed_path, 1); top.addWidget(b_browse); top.addWidget(b_load)
        root.addLayout(top)
        root.addWidget(QLabel("Choose the REFERENCE images to show around each point (e.g. the before / after images). "
                              "Do not use the classified map: labelling must be blind."))

        mid = QHBoxLayout()
        self.lst = QListWidget(); self.lst.setMaximumWidth(260)
        for name, layer in window.layers.items():
            if isinstance(layer, dict) and layer.get("image") is not None:
                prof = layer.get("profile") or {}
                if prof.get("crs") is not None and georef.is_georeferenced(prof.get("transform"), prof.get("crs")):
                    it = QListWidgetItem(name); it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Unchecked); self.lst.addItem(it)
        self.lst.itemChanged.connect(lambda *_: self._show())
        left = QVBoxLayout(); left.addWidget(QLabel("Reference images:")); left.addWidget(self.lst)
        self.sp_half = QSpinBox(); self.sp_half.setRange(30, 20000); self.sp_half.setValue(300); self.sp_half.setSuffix(" m")
        self.sp_half.setSingleStep(50); self.sp_half.valueChanged.connect(lambda *_: self._show())
        left.addWidget(QLabel("Half-width of the window:")); left.addWidget(self.sp_half)
        mid.addLayout(left)
        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        mid.addWidget(self.canvas, 1)
        root.addLayout(mid, 1)

        links = QHBoxLayout()
        links.addWidget(QLabel("Open this point in:"))
        for text, fn in (("Google Earth (history slider)", labeler.earth_url), ("Esri Wayback (dated archive)", labeler.wayback_url),
                         ("Google Maps (satellite)", labeler.maps_url)):
            b = QPushButton(text); b.clicked.connect(lambda _=False, f=fn: self._open_web(f)); links.addWidget(b)
        links.addStretch(1)
        root.addLayout(links)

        self.lbl_info = QLabel(""); self.lbl_info.setStyleSheet("font-weight:bold")
        root.addWidget(self.lbl_info)
        cat = QHBoxLayout()
        self.ed_cats = QLineEdit(", ".join(self._fallback)); self.ed_cats.setToolTip("Category names, comma separated, any kind of map (water, change, land cover ...). They must match the Categories of the accuracy step; upper/lower case is ignored. Loaded automatically from the sampling step.")
        b_apply = QPushButton("Apply categories"); b_apply.clicked.connect(self._build_buttons)
        cat.addWidget(QLabel("Categories:")); cat.addWidget(self.ed_cats, 1); cat.addWidget(b_apply)
        root.addLayout(cat)
        self.btn_row = QHBoxLayout(); root.addLayout(self.btn_row)
        nav = QHBoxLayout()
        self.ed_note = QLineEdit(); self.ed_note.setPlaceholderText("note (optional)")
        b_prev, b_next, b_nu = QPushButton("◀ Previous"), QPushButton("Next ▶"), QPushButton("Next unlabelled ⏭")
        b_prev.clicked.connect(lambda: self._go(self.idx - 1)); b_next.clicked.connect(lambda: self._go(self.idx + 1))
        b_nu.clicked.connect(self._go_unlabeled)
        nav.addWidget(b_prev); nav.addWidget(b_next); nav.addWidget(b_nu); nav.addWidget(self.ed_note, 1)
        b_close = QPushButton("Close"); b_close.clicked.connect(self.accept); nav.addWidget(b_close)
        root.addLayout(nav)
        self._build_buttons()
        for key, fn in ((Qt.Key_Left, lambda: self._go(self.idx - 1)), (Qt.Key_Right, lambda: self._go(self.idx + 1))):
            sc = QShortcut(QKeySequence(key), self); sc.activated.connect(fn); self._shortcuts.append(sc)
        if points_csv and os.path.isfile(points_csv):
            self._load()

    # ---- file handling
    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "points_to_label.csv", self.ed_path.text(), "CSV (*.csv)")
        if p:
            self.ed_path.setText(p); self._load()

    def _load(self):
        p = self.ed_path.text().strip()
        try:
            self.rows, self.fields = labeler.read_points(p)
        except Exception as exc:
            QMessageBox.warning(self, "Label points", f"Cannot read the points file:\n{exc}"); return
        if not self.rows:
            QMessageBox.warning(self, "Label points", "The file has no points."); return
        self.path = p
        sch = validation.read_scheme(p)
        cats = labeler.choose_categories(sch["categories"] if sch else None, self.rows, self._fallback)
        if cats:
            self.ed_cats.setText(", ".join(cats)); self._build_buttons()
        nxt = labeler.next_unlabeled(self.rows, 0)
        self.idx = 0 if nxt is None else nxt
        self._show()

    def _save(self):
        try:
            labeler.write_points(self.path, self.rows, self.fields)
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Label points",
                                f"Could not save the labels (is the file open in Excel?):\n{exc}\n\nClose it and label again.")
            return False

    # ---- buttons / labels
    def _categories(self):
        return [c.strip() for c in self.ed_cats.text().split(",") if c.strip()]

    def _build_buttons(self):
        while self.btn_row.count():
            w = self.btn_row.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        for sc in self._shortcuts[2:]:
            sc.setEnabled(False)
        self._shortcuts = self._shortcuts[:2]
        for i, name in enumerate(self._categories() + [labeler.UNCERTAIN], start=1):
            b = QPushButton(f"{i}  {name}")
            b.clicked.connect(lambda _=False, n=name: self._set_label(n))
            self.btn_row.addWidget(b)
            if i <= 9:
                sc = QShortcut(QKeySequence(str(i)), self); sc.activated.connect(lambda n=name: self._set_label(n))
                self._shortcuts.append(sc)

    def _set_label(self, name):
        if not self.rows:
            return
        r = self.rows[self.idx]
        r["ref_label"] = name; r["note"] = self.ed_note.text().strip()
        if not self._save():
            return
        nxt = labeler.next_unlabeled(self.rows, self.idx + 1)
        if nxt is None:
            self._show()
            lab, unc, tot = labeler.progress(self.rows)
            QMessageBox.information(self, "Label points",
                                    f"All {tot} points have a label ({unc} uncertain).\nNext: Validation > 2. Compute Accuracy.")
        else:
            self.idx = nxt; self._show()

    def _go(self, i):
        if self.rows:
            self.idx = i % len(self.rows); self._show()

    def _go_unlabeled(self):
        nxt = labeler.next_unlabeled(self.rows, self.idx + 1) if self.rows else None
        if nxt is None:
            QMessageBox.information(self, "Label points", "No unlabelled points left.")
        else:
            self.idx = nxt; self._show()

    def _open_web(self, fn):
        if self.rows:
            r = self.rows[self.idx]
            QDesktopServices.openUrl(QUrl(fn(r["lon"], r["lat"])))

    # ---- drawing
    def _show(self):
        if not self.rows:
            return
        import numpy as np
        from rasterio.warp import transform as warp_transform
        r = self.rows[self.idx]
        lab, unc, tot = labeler.progress(self.rows)
        cur = r["ref_label"] or "(not labelled)"
        self.lbl_info.setText(f"Point id {r['id']}  ({self.idx + 1} of {tot})   lon {r['lon']:.5f}, lat {r['lat']:.5f}   "
                              f"label: {cur}   |   done {lab + unc}/{tot}, uncertain {unc}")
        self.ed_note.setText(r["note"])
        items = []
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if it.checkState() != Qt.Checked:
                continue
            layer = self.win.layers.get(it.text())
            if not isinstance(layer, dict):
                continue
            prof = layer["profile"]; tr = prof["transform"]
            try:
                xs, ys = warp_transform("EPSG:4326", prof["crs"], [r["lon"]], [r["lat"]])
                x, y = xs[0], ys[0]
                half_m = self.sp_half.value()
                if getattr(prof["crs"], "is_geographic", False):
                    half_px = int(half_m / 111320.0 / abs(tr.a)) + 1
                else:
                    half_px = int(half_m / abs(tr.a)) + 1
                got = labeler.chip(np.asarray(layer["image"]), tr, x, y, max(half_px, 3))
            except Exception as exc:
                items.append(dict(name=it.text(), window=None, note=f"error: {exc}")); continue
            if got is None:
                items.append(dict(name=it.text(), window=None, note="point outside\nthis image"))
            else:
                win, _rc, ext = got
                items.append(dict(name=it.text(), window=win, extent=ext, point_xy=(x, y)))
        if not items:
            items = [dict(name="Tick one or more reference images on the left", window=None, note="")]
        try:
            labeler.render_chips(self.fig, items)
        except Exception as exc:
            self.fig.clear(); self.fig.text(0.1, 0.5, f"Cannot draw: {exc}")
        self.canvas.draw_idle()
