"""Qt dialog for print-quality map export (see mapexport.py for the rendering)."""
from __future__ import annotations

import os
import tempfile
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout)

import georef
import mapexport as mx

EXT = {"pdf": ".pdf", "png": ".png", "jpg": ".jpg", "tif": ".tif", "svg": ".svg"}
FILTERS = {"pdf": "PDF (*.pdf)", "png": "PNG (*.png)", "jpg": "JPEG (*.jpg *.jpeg)", "tif": "TIFF (*.tif *.tiff)",
           "svg": "SVG (*.svg)"}


# ------------------------------------------------------------------------------------------ data from the main window
def raster_from_layer(name, layer):
    """RasterLayer for a stored layer dict, or (None, reason) if it has no usable georeferencing."""
    img = layer.get("image")
    prof = layer.get("profile") or {}
    tr, crs = prof.get("transform"), prof.get("crs")
    if img is None:
        return None, "not a raster"
    if crs is None or not georef.is_georeferenced(tr, crs):
        return None, "not georeferenced"
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[None]
    rgb = img.shape[0] >= 3
    data = img[:3] if rgb else img[0]
    dr = layer.get("display_range") or (layer.get("vmin"), layer.get("vmax"))
    vmin, vmax = (dr if dr and dr[0] is not None and dr[1] is not None else (None, None))
    if rgb and np.issubdtype(img.dtype, np.integer) and img.dtype.itemsize == 1:
        vmin, vmax = 0.0, 255.0
    elif rgb:
        vmin = vmax = None                      # per-band percentile stretch
    nodata = prof.get("nodata")
    return mx.RasterLayer(
        data=data, transform=tr, geographic=bool(getattr(crs, "is_geographic", False)),
        crs_text=str(crs.to_string() if hasattr(crs, "to_string") else crs), cmap=str(layer.get("cmap") or "viridis"),
        vmin=None if vmin is None else float(vmin), vmax=None if vmax is None else float(vmax),
        nodata=None if nodata is None else float(nodata),
        categorical=bool(np.issubdtype(img.dtype, np.integer) or img.dtype == bool), name=name,
        unit_label=str(layer.get("unit") or "")), ""


def vector_arrays(gdf, crs):
    """(lines, rings, points) of a GeoDataFrame reprojected to ``crs``."""
    g = gdf
    if getattr(gdf, "crs", None) is not None and crs is not None and gdf.crs != crs:
        g = gdf.to_crs(crs)
    lines, rings, points = [], [], []
    for geom in g.geometry:
        mx.geometry_to_arrays(geom, lines, rings, points)
    return lines, rings, points


def is_polygon_gdf(gdf):
    try:
        return bool(len(gdf)) and all(t in ("Polygon", "MultiPolygon") for t in gdf.geometry.geom_type.unique())
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------------------ dialog
class ExportMapDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.win = window
        self.setWindowTitle("Export Map (PDF / PNG / JPEG / TIFF / SVG)")
        self.setMinimumWidth(560)
        self.drawn_extent = None          # (xmin, ymin, xmax, ymax) in the CRS of the displayed layer
        self.drawn_crs = None
        self._selector = None
        self._cid = None
        self.settings = QSettings("SpaceGeoVision", "MapExport")
        self._build()
        self._load_settings()

    # ---- layout
    def _build(self):
        w = self.win
        root = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QVBoxLayout(); holder = QGroupBox(); holder.setFlat(True); holder.setLayout(inner)
        scroll.setWidget(holder)
        root.addWidget(scroll)

        rasters, vectors, polys = [], [], []
        for name, layer in w.layers.items():
            if not isinstance(layer, dict):
                continue
            if layer.get("image") is not None:
                rl, why = raster_from_layer(name, layer)
                if rl is not None:
                    rasters.append(name)
            elif layer.get("gdf") is not None and getattr(layer["gdf"], "crs", None) is not None:
                vectors.append(name)
                if is_polygon_gdf(layer["gdf"]):
                    polys.append(name)
        self.raster_names, self.vector_names, self.poly_names = rasters, vectors, polys

        g = QGroupBox("Layers"); f = QFormLayout(g)
        self.cb_base = QComboBox(); self.cb_base.addItems(rasters)
        cur = getattr(w, "current_layer", None)
        if cur in rasters:
            self.cb_base.setCurrentText(cur)
        self.cb_over = QComboBox(); self.cb_over.addItems(["(none)"] + rasters)
        self.sl_alpha = QSlider(Qt.Horizontal); self.sl_alpha.setRange(5, 100); self.sl_alpha.setValue(60)
        self.lst_vec = QListWidget(); self.lst_vec.setMaximumHeight(90)
        for n in vectors:
            it = QListWidgetItem(n); it.setFlags(it.flags() | Qt.ItemIsUserCheckable); it.setCheckState(Qt.Unchecked)
            self.lst_vec.addItem(it)
        f.addRow("Base raster", self.cb_base); f.addRow("Overlay raster", self.cb_over)
        f.addRow("Overlay opacity", self.sl_alpha); f.addRow("Vector layers", self.lst_vec)
        inner.addWidget(g)

        g = QGroupBox("Area to print / clip"); f = QFormLayout(g)
        self.cb_ext = QComboBox()
        self.cb_ext.addItems(["Whole layer", "Current view on the map", "Rectangle drawn on the map", "Custom coordinates"])
        self.btn_draw = QPushButton("Draw rectangle on the map…")
        self.btn_draw.clicked.connect(self._start_draw)
        self.lbl_drawn = QLabel("(none drawn)")
        self.sp_ext = [QDoubleSpinBox() for _ in range(4)]
        for sp in self.sp_ext:
            sp.setRange(-1e12, 1e12); sp.setDecimals(6)
        crow = QHBoxLayout()
        for lab, sp in zip(("xmin", "ymin", "xmax", "ymax"), self.sp_ext):
            crow.addWidget(QLabel(lab)); crow.addWidget(sp)
        self.cb_clip = QComboBox(); self.cb_clip.addItems(["(no clipping)"] + polys)
        self.ck_mask = QCheckBox("Blank out everything outside the polygon"); self.ck_mask.setChecked(True)
        self.ck_outline = QCheckBox("Draw the polygon outline"); self.ck_outline.setChecked(True)
        f.addRow("Extent", self.cb_ext); f.addRow("", self.btn_draw); f.addRow("Drawn", self.lbl_drawn)
        f.addRow("Custom (layer CRS units)", crow)
        f.addRow("Clip to polygon layer", self.cb_clip); f.addRow("", self.ck_mask); f.addRow("", self.ck_outline)
        inner.addWidget(g)

        g = QGroupBox("Page and quality"); f = QFormLayout(g)
        self.cb_page = QComboBox(); self.cb_page.addItems(list(mx.PAGE_SIZES_MM))
        self.cb_orient = QComboBox(); self.cb_orient.addItems(["Automatic", "Portrait", "Landscape"])
        self.sp_margin = QDoubleSpinBox(); self.sp_margin.setRange(0, 50); self.sp_margin.setValue(12); self.sp_margin.setSuffix(" mm")
        self.sp_dpi = QSpinBox(); self.sp_dpi.setRange(36, 1200); self.sp_dpi.setValue(300); self.sp_dpi.setSuffix(" dpi")
        self.cb_fmt = QComboBox(); self.cb_fmt.addItems(["PDF", "PNG", "JPEG", "TIFF", "SVG"])
        self.lbl_size = QLabel(""); self.lbl_size.setWordWrap(True)
        for wd in (self.cb_page, self.cb_orient, self.sp_dpi, self.cb_fmt):
            sig = wd.currentIndexChanged if isinstance(wd, QComboBox) else wd.valueChanged
            sig.connect(self._update_size)
        f.addRow("Page size", self.cb_page); f.addRow("Orientation", self.cb_orient); f.addRow("Margins", self.sp_margin)
        f.addRow("Resolution", self.sp_dpi); f.addRow("Format", self.cb_fmt); f.addRow("", self.lbl_size)
        inner.addWidget(g)

        g = QGroupBox("Text and map elements"); f = QFormLayout(g)
        self.ed_title = QLineEdit(); self.ed_sub = QLineEdit(); self.ed_note = QLineEdit(); self.ed_author = QLineEdit()
        f.addRow("Title", self.ed_title); f.addRow("Subtitle", self.ed_sub); f.addRow("Note", self.ed_note)
        f.addRow("Author (PDF metadata)", self.ed_author)
        self.ck = {}
        row = QHBoxLayout()
        for key, lab in (("grid", "Coordinate grid"), ("scalebar", "Scale bar"), ("north", "North arrow"),
                         ("colorbar", "Colour bar"), ("border", "Border"), ("footer", "Footer")):
            c = QCheckBox(lab); c.setChecked(True); self.ck[key] = c; row.addWidget(c)
        f.addRow("Show", row)
        self.cb_stretch = QComboBox(); self.cb_stretch.addItems(["As displayed", "Percentile 2-98 %", "Min-max"])
        self.cb_interp = QComboBox(); self.cb_interp.addItems(["Nearest (sharp pixels)", "Smooth (bilinear)"])
        f.addRow("Contrast", self.cb_stretch); f.addRow("Resampling", self.cb_interp)
        inner.addWidget(g)

        row = QHBoxLayout()
        b_prev, b_exp, b_close = QPushButton("Preview"), QPushButton("Export…"), QPushButton("Close")
        b_prev.clicked.connect(self.preview); b_exp.clicked.connect(self.export); b_close.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(b_prev); row.addWidget(b_exp); row.addWidget(b_close)
        root.addLayout(row)
        if not rasters:
            QMessageBox.information(self, "Export Map", "Load a georeferenced raster first (a layer with a CRS and transform).")
        self._update_size()

    # ---- settings
    def _load_settings(self):
        s = self.settings
        def txt(key, default): return str(s.value(key, default))
        try:
            self.cb_page.setCurrentText(txt("page", "A4")); self.cb_orient.setCurrentIndex(int(s.value("orient", 0)))
            self.sp_dpi.setValue(int(s.value("dpi", 300))); self.cb_fmt.setCurrentIndex(int(s.value("fmt", 0)))
            self.sp_margin.setValue(float(s.value("margin", 12))); self.ed_author.setText(txt("author", ""))
            self.cb_stretch.setCurrentIndex(int(s.value("stretch", 0))); self.cb_interp.setCurrentIndex(int(s.value("interp", 0)))
            for k, c in self.ck.items():
                c.setChecked(str(s.value("show_" + k, "true")).lower() == "true")
        except Exception:
            pass

    def _save_settings(self):
        s = self.settings
        try:
            s.setValue("page", self.cb_page.currentText()); s.setValue("orient", self.cb_orient.currentIndex())
            s.setValue("dpi", self.sp_dpi.value()); s.setValue("fmt", self.cb_fmt.currentIndex())
            s.setValue("margin", self.sp_margin.value()); s.setValue("author", self.ed_author.text())
            s.setValue("stretch", self.cb_stretch.currentIndex()); s.setValue("interp", self.cb_interp.currentIndex())
            for k, c in self.ck.items():
                s.setValue("show_" + k, "true" if c.isChecked() else "false")
        except Exception:
            pass

    def _update_size(self, *_):
        try:
            w, h = mx.PAGE_SIZES_MM[self.cb_page.currentText()]
            o = self.cb_orient.currentIndex()
            if o == 2:
                w, h = max(w, h), min(w, h)
            elif o == 1:
                w, h = min(w, h), max(w, h)
            dpi = self.sp_dpi.value()
            px = int(w / 25.4 * dpi), int(h / 25.4 * dpi)
            fmt = self.cb_fmt.currentText()
            txt = f"{w:.0f} x {h:.0f} mm" + (" (automatic: follows the map shape)" if o == 0 else "")
            if fmt in ("PNG", "JPEG", "TIFF"):
                txt += f"  →  {px[0]} x {px[1]} pixels ({px[0] * px[1] / 1e6:.0f} Mpx)"
                if px[0] * px[1] > mx.MAX_PIXELS:
                    txt += "  — too large, lower the DPI"
            else:
                txt += "  →  vector page; the image inside is resampled to this DPI"
            self.lbl_size.setText(txt)
        except Exception:
            pass

    # ---- extent helpers
    def _base(self):
        name = self.cb_base.currentText()
        layer = self.win.layers.get(name)
        if not name or layer is None:
            raise ValueError("Choose a base raster")
        rl, why = raster_from_layer(name, layer)
        if rl is None:
            raise ValueError(f"'{name}' cannot be exported: {why}")
        return name, layer, rl

    def _displayed_crs(self):
        cur = self.win.layers.get(getattr(self.win, "current_layer", None)) or {}
        crs = (cur.get("profile") or {}).get("crs")
        tr = (cur.get("profile") or {}).get("transform")
        return crs if (cur.get("image") is not None and crs is not None and georef.is_georeferenced(tr, crs)) else None

    def _to_base_crs(self, box, from_crs, to_crs):
        if from_crs is None or to_crs is None or from_crs == to_crs:
            return tuple(box)
        from rasterio.warp import transform_bounds
        return tuple(transform_bounds(from_crs, to_crs, *box))

    def _start_draw(self):
        from matplotlib.widgets import RectangleSelector
        if self._displayed_crs() is None:
            QMessageBox.information(self, "Export Map", "Display a georeferenced raster first, then draw the print area on it.")
            return
        self.hide()
        self.win.statusBar().showMessage("Drag a rectangle on the map to choose the print area…")

        def done(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return
            x0, x1 = sorted((eclick.xdata, erelease.xdata)); y0, y1 = sorted((eclick.ydata, erelease.ydata))
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                return
            self.drawn_extent = (x0, y0, x1, y1); self.drawn_crs = self._displayed_crs()
            self.lbl_drawn.setText(f"{x0:.5g}, {y0:.5g}  →  {x1:.5g}, {y1:.5g}")
            self.cb_ext.setCurrentIndex(2)
            self._stop_draw()
            self.win.statusBar().showMessage("Print area chosen.", 4000)
            self.show(); self.raise_()

        self._selector = RectangleSelector(self.win.ax, done, useblit=False, button=[1], minspanx=1e-12, minspany=1e-12,
                                           interactive=False)
        self.win.canvas.draw_idle()

    def _stop_draw(self):
        if self._selector is not None:
            try:
                self._selector.set_active(False); self._selector.disconnect_events()
            except Exception:
                pass
            self._selector = None

    def reject(self):
        self._stop_draw()
        super().reject()

    # ---- spec
    def _spec(self, fmt=None, dpi=None):
        name, layer, base = self._base()
        base_crs = layer["profile"]["crs"]
        idx = self.cb_ext.currentIndex()
        extent = None
        if idx == 1:
            crs = self._displayed_crs()
            if crs is None:
                raise ValueError("The displayed layer is not georeferenced; choose another extent option")
            x0, x1 = sorted(self.win.ax.get_xlim()); y0, y1 = sorted(self.win.ax.get_ylim())
            extent = self._to_base_crs((x0, y0, x1, y1), crs, base_crs)
        elif idx == 2:
            if self.drawn_extent is None:
                raise ValueError("Draw a rectangle on the map first")
            extent = self._to_base_crs(self.drawn_extent, self.drawn_crs, base_crs)
        elif idx == 3:
            extent = tuple(sp.value() for sp in self.sp_ext)
        clip = None
        cname = self.cb_clip.currentText()
        if self.cb_clip.currentIndex() > 0:
            _, rings, _ = vector_arrays(self.win.layers[cname]["gdf"], base_crs)
            if not rings:
                raise ValueError(f"'{cname}' has no polygons")
            clip = rings
        fmt = fmt or ("pdf", "png", "jpg", "tif", "svg")[self.cb_fmt.currentIndex()]
        spec = mx.ExportSpec(
            page=self.cb_page.currentText(), orientation=("auto", "portrait", "landscape")[self.cb_orient.currentIndex()],
            margin_mm=self.sp_margin.value(), dpi=dpi or self.sp_dpi.value(), fmt=fmt,
            title=self.ed_title.text().strip(), subtitle=self.ed_sub.text().strip(), note=self.ed_note.text().strip(),
            author=self.ed_author.text().strip(), extent=extent, clip_rings=clip, mask_outside=self.ck_mask.isChecked(),
            outline_clip=self.ck_outline.isChecked(), show_grid=self.ck["grid"].isChecked(),
            show_scalebar=self.ck["scalebar"].isChecked(), show_north=self.ck["north"].isChecked(),
            show_colorbar=self.ck["colorbar"].isChecked(), show_border=self.ck["border"].isChecked(),
            show_footer=self.ck["footer"].isChecked(),
            stretch=("layer", "percentile", "minmax")[self.cb_stretch.currentIndex()],
            interpolation="nearest" if self.cb_interp.currentIndex() == 0 else "bilinear",
            overlay_alpha=self.sl_alpha.value() / 100.0)
        overlay, notes = None, []
        oname = self.cb_over.currentText()
        if self.cb_over.currentIndex() > 0 and oname != name:
            ol, why = raster_from_layer(oname, self.win.layers[oname])
            ocrs = self.win.layers[oname]["profile"]["crs"]
            if ol is None:
                notes.append(f"Overlay '{oname}' skipped: {why}.")
            elif ocrs != base_crs:
                notes.append(f"Overlay '{oname}' skipped: its CRS differs from the base raster (reproject it first).")
            elif base.data.ndim == 3:
                notes.append("Overlay skipped: it can only be drawn over a single-band base raster.")
            else:
                overlay = ol
        vectors = []
        palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        for i in range(self.lst_vec.count()):
            it = self.lst_vec.item(i)
            if it.checkState() == Qt.Checked:
                lines, rings, pts = vector_arrays(self.win.layers[it.text()]["gdf"], base_crs)
                vectors.append(mx.VectorLayer(it.text(), lines, rings, pts, color=palette[len(vectors) % len(palette)]))
        return base, spec, overlay, vectors, notes

    # ---- actions
    def preview(self):
        try:
            base, spec, overlay, vectors, notes = self._spec(fmt="png", dpi=70)
        except Exception as exc:
            QMessageBox.warning(self, "Export Map", str(exc)); return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        tmp = os.path.join(tempfile.gettempdir(), "sgv_map_preview.png")
        try:
            mx.export_map(tmp, base, spec, overlay, vectors)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Export Map", f"Preview failed: {exc}"); return
        QApplication.restoreOverrideCursor()
        dlg = QDialog(self); dlg.setWindowTitle("Preview (low resolution)")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(); pm = QPixmap(tmp)
        lbl.setPixmap(pm.scaledToHeight(min(760, max(200, pm.height())), Qt.SmoothTransformation))
        lay.addWidget(lbl)
        if notes:
            lay.addWidget(QLabel("\n".join(notes)))
        dlg.exec_()

    def export(self):
        try:
            base, spec, overlay, vectors, notes = self._spec()
        except Exception as exc:
            QMessageBox.warning(self, "Export Map", str(exc)); return
        start = self.settings.value("last_dir", os.path.expanduser("~"))
        default = os.path.join(str(start), (spec.title or base.name or "map").replace(os.sep, "_") + EXT[spec.fmt])
        path, _ = QFileDialog.getSaveFileName(self, "Export map", default, FILTERS[spec.fmt])
        if not path:
            return
        if not path.lower().endswith(tuple(EXT.values()) + (".jpeg", ".tiff")):
            path += EXT[spec.fmt]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            wmm, hmm = mx.export_map(path, base, spec, overlay, vectors)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export Map", f"Export failed: {exc}")
            return
        QApplication.restoreOverrideCursor()
        self._save_settings()
        try:
            self.settings.setValue("last_dir", os.path.dirname(path))
        except Exception:
            pass
        msg = f"Saved {path}\nPage {wmm:.0f} x {hmm:.0f} mm at {spec.dpi} dpi."
        if notes:
            msg += "\n\n" + "\n".join(notes)
        QMessageBox.information(self, "Export Map", msg)
