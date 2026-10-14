"""Blind labelling of validation points (no Qt): read/write points_to_label.csv, image chips around a point, web links.

Only ``points_to_label.csv`` is used here - it does not contain the map class, so labelling stays blind.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

UNCERTAIN = "uncertain"


def read_points(path: str) -> Tuple[List[Dict], List[str]]:
    """(rows, fieldnames). Each row: id (int), lon, lat (float), ref_label (str), note (str) + any extra columns."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        for need in ("id", "lon", "lat"):
            if need not in fields:
                raise ValueError(f"Column '{need}' is missing in {os.path.basename(path)}.")
        rows = []
        for r in rd:
            r = dict(r)
            r["id"] = int(r["id"]); r["lon"] = float(r["lon"]); r["lat"] = float(r["lat"])
            r["ref_label"] = (r.get("ref_label") or "").strip()
            r["note"] = (r.get("note") or "").strip()
            rows.append(r)
    for col in ("ref_label", "note"):
        if col not in fields:
            fields.append(col)
    return rows, fields


def write_points(path: str, rows: List[Dict], fields: List[str]) -> None:
    """Atomic write (temp file + replace) so a crash or a file open in Excel cannot truncate the labels."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["lon"] = f"{r['lon']:.6f}"; out["lat"] = f"{r['lat']:.6f}"
            w.writerow(out)
    os.replace(tmp, path)


def categories_in_rows(rows: List[Dict]) -> List[str]:
    """Distinct labels already used in the file (order of first use, 'uncertain' excluded)."""
    seen: List[str] = []
    for r in rows:
        lab = r.get("ref_label", "")
        if lab and lab.lower() != UNCERTAIN and lab not in seen:
            seen.append(lab)
    return seen


def choose_categories(scheme_categories: Optional[List[str]], rows: List[Dict], fallback: List[str]) -> List[str]:
    """Categories to offer as buttons: the scheme saved with the points; otherwise the labels already in the file;
    otherwise the fallback. Labels already in the file but missing from the scheme are appended so none is lost."""
    base = list(scheme_categories) if scheme_categories else (categories_in_rows(rows) or list(fallback))
    norm = {c.strip().lower().replace(" ", "_") for c in base}
    for lab in categories_in_rows(rows):
        if lab.strip().lower().replace(" ", "_") not in norm:
            base.append(lab); norm.add(lab.strip().lower().replace(" ", "_"))
    return base


def next_unlabeled(rows: List[Dict], start: int = 0) -> Optional[int]:
    """Index of the first row at or after ``start`` without a label (wrapping around), or None if all are labelled."""
    n = len(rows)
    for k in range(n):
        i = (start + k) % n
        if not rows[i]["ref_label"]:
            return i
    return None


def progress(rows: List[Dict]) -> Tuple[int, int, int]:
    """(labelled, uncertain, total)."""
    lab = sum(1 for r in rows if r["ref_label"] and r["ref_label"].lower() != UNCERTAIN)
    unc = sum(1 for r in rows if r["ref_label"].lower() == UNCERTAIN)
    return lab, unc, len(rows)


def chip(data: np.ndarray, transform, x: float, y: float, half_px: int):
    """Square window of (rows, cols) / (bands, rows, cols) data centred on map coordinates (x, y).
    Returns (window, (row, col) of the point inside the window, extent (left, right, bottom, top)) or None if outside."""
    if abs(getattr(transform, "b", 0.0)) > 1e-12 or abs(getattr(transform, "d", 0.0)) > 1e-12:
        return None
    col = (x - transform.c) / transform.a
    row = (y - transform.f) / transform.e
    rows, cols = data.shape[-2:]
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    r0, r1 = max(0, int(row) - half_px), min(rows, int(row) + half_px + 1)
    c0, c1 = max(0, int(col) - half_px), min(cols, int(col) + half_px + 1)
    win = data[..., r0:r1, c0:c1]
    left = transform.c + transform.a * c0
    top = transform.f + transform.e * r0
    extent = (left, left + transform.a * (c1 - c0), top + transform.e * (r1 - r0), top)
    return win, (row - r0, col - c0), extent


def earth_url(lon: float, lat: float, altitude_m: float = 400.0) -> str:
    """Google Earth for web (its time slider shows historical imagery)."""
    return f"https://earth.google.com/web/@{lat:.6f},{lon:.6f},0a,{altitude_m:.0f}d,35y,0h,0t,0r"


def wayback_url(lon: float, lat: float, level: int = 17) -> str:
    """Esri World Imagery Wayback viewer (dated archive releases), centred on the point."""
    return f"https://livingatlas.arcgis.com/wayback/#mapCenter={lon:.6f}%2C{lat:.6f}%2C{level}"


def maps_url(lon: float, lat: float) -> str:
    return f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},300m/data=!3m1!1e3"


def _to_display(win: np.ndarray):
    """(image, cmap or None, vmin, vmax) for a chip: 3+ bands -> RGB with a 2-98 % stretch; 1 band -> gray scale."""
    a = np.asarray(win, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] >= 3:
        rgb = np.moveaxis(a[:3], 0, -1)
        out = np.zeros_like(rgb)
        for i in range(3):
            ch = rgb[..., i]; v = ch[np.isfinite(ch)]
            lo, hi = (np.percentile(v, [2, 98]) if v.size else (0.0, 1.0))
            if not hi > lo:
                hi = lo + 1.0
            out[..., i] = np.clip((np.nan_to_num(ch, nan=lo) - lo) / (hi - lo), 0, 1)
        return out, None, None, None
    if a.ndim == 3:
        a = a[0]
    v = a[np.isfinite(a)]
    lo, hi = (np.percentile(v, [2, 98]) if v.size else (0.0, 1.0))
    if not hi > lo:
        hi = lo + 1.0
    return np.ma.masked_invalid(a), "gray", float(lo), float(hi)


def render_chips(fig, items) -> None:
    """Draw one panel per item on ``fig``. item = dict(name, window, point_rc, extent|None, note).
    A reference item with window None is drawn as an explanatory empty panel (point outside that image)."""
    fig.clear()
    n = max(1, len(items))
    for k, it in enumerate(items, start=1):
        ax = fig.add_subplot(1, n, k)
        ax.set_title(it.get("name", ""), fontsize=9)
        if it.get("window") is None:
            ax.text(0.5, 0.5, it.get("note") or "point outside\nthis image", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        img, cmap, lo, hi = _to_display(it["window"])
        ext = it.get("extent")
        kw = dict(extent=ext, origin="upper", interpolation="nearest")
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, **kw)
        else:
            ax.imshow(img, **kw)
        if ext is not None and it.get("point_xy") is not None:
            x, y = it["point_xy"]
            ax.plot([x], [y], marker="+", color="red", markersize=18, markeredgewidth=1.6)
            ax.plot([x], [y], marker="o", markerfacecolor="none", markeredgecolor="yellow", markersize=14)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
