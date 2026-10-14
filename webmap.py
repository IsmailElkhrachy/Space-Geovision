"""Web-map overlay support (Leaflet inside QtWebEngine): renders the application's layers over a satellite basemap.

Pure helpers (numpy / matplotlib / PIL, no Qt): colourise a raster to RGBA, decimate it, encode a PNG data URI and build
the HTML page. Re-projection to Web Mercator needs rasterio and lives in :func:`to_web_mercator`.

Basemaps: Esri World Imagery (default), OpenStreetMap and Google Satellite. The Google tile URL is the unofficial one the
application used before; it is NOT part of a licensed Google API, so check Google's terms before relying on it for
publication-quality or commercial work. Esri/OSM tiles require their attribution, which the page shows.
"""
from __future__ import annotations

import base64
import io
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAX_OVERLAY_PX = 1600


def _cmap(name):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (KeyError, TypeError):
        return matplotlib.colormaps["viridis"]


def decimate(arr: np.ndarray, max_px: int = MAX_OVERLAY_PX) -> Tuple[np.ndarray, int]:
    """Nearest-neighbour decimation of the last two axes so that neither exceeds ``max_px``. Returns (array, stride)."""
    h, w = arr.shape[-2:]
    stride = max(1, int(np.ceil(max(h, w) / float(max_px))))
    return (arr[..., ::stride, ::stride] if stride > 1 else arr), stride


def render_rgba(image: np.ndarray, cmap: str = "viridis", vmin: Optional[float] = None, vmax: Optional[float] = None,
                nodata: Optional[float] = None) -> np.ndarray:
    """Colourise to uint8 RGBA (H, W, 4). NaN / nodata pixels are fully transparent.

    A 2-D array or a one-band 3-D array uses ``cmap`` and the stretch ``vmin``/``vmax`` (default 2nd-98th percentile).
    Three or more bands are shown as RGB (first three, per-band 2-98 % stretch)."""
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim not in (2, 3):
        raise ValueError(f"Cannot render an array of shape {a.shape}")
    af = a.astype(np.float64)
    bad = ~np.isfinite(af)
    if nodata is not None and np.isfinite(nodata):
        bad = bad | (af == nodata)
    if a.ndim == 2:
        valid = af[~bad]
        if valid.size == 0:
            return np.zeros(a.shape + (4,), np.uint8)
        lo, hi = (vmin, vmax)
        if lo is None or hi is None:
            lo, hi = np.percentile(valid, [2, 98])
        if not hi > lo:
            hi = lo + 1.0
        t = np.clip((np.where(bad, lo, af) - lo) / (hi - lo), 0.0, 1.0)
        rgba = (_cmap(cmap)(t) * 255.0 + 0.5).astype(np.uint8)
        rgba[bad, 3] = 0
        return rgba
    if a.shape[0] < 3:
        raise ValueError("Need 1 or at least 3 bands")
    bad = bad[:3].any(axis=0)
    out = np.zeros(a.shape[1:] + (4,), np.uint8)
    for i in range(3):
        band = af[i]
        v = band[~bad]
        if v.size == 0:
            return out
        lo, hi = np.percentile(v, [2, 98])
        if not hi > lo:
            hi = lo + 1.0
        out[..., i] = (np.clip((np.where(bad, lo, band) - lo) / (hi - lo), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    out[..., 3] = np.where(bad, 0, 255)
    return out


def png_data_uri(rgba: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def to_web_mercator(rgba: np.ndarray, transform, crs) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Re-project an RGBA image to EPSG:3857 (what Leaflet's image overlay assumes).

    Returns (rgba_3857, (south, west, north, east)) with the corners in degrees. Requires rasterio."""
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
    h, w = rgba.shape[:2]
    dst_crs = "EPSG:3857"
    left, top = transform.c, transform.f
    right, bottom = transform.c + transform.a * w + transform.b * h, transform.f + transform.d * w + transform.e * h
    dt, dw, dh = calculate_default_transform(crs, dst_crs, w, h, left=min(left, right), bottom=min(top, bottom),
                                             right=max(left, right), top=max(top, bottom))
    out = np.zeros((dh, dw, 4), np.uint8)
    for i in range(4):
        band = np.zeros((dh, dw), np.uint8)               # C-contiguous destination for rasterio
        reproject(np.ascontiguousarray(rgba[..., i]), band, src_transform=transform, src_crs=crs,
                  dst_transform=dt, dst_crs=dst_crs, resampling=Resampling.nearest)
        out[..., i] = band
    x_w, y_n = dt.c, dt.f
    x_e, y_s = dt.c + dt.a * dw, dt.f + dt.e * dh
    deg_w, deg_s, deg_e, deg_n = transform_bounds(dst_crs, "EPSG:4326", x_w, y_s, x_e, y_n)
    return out, (deg_s, deg_w, deg_n, deg_e)


def _js_json(obj) -> str:
    """JSON that is safe inside a <script> element."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def build_html(overlays: Sequence[Dict], vectors: Sequence[Dict], view_bounds: Optional[Tuple[float, float, float, float]],
               opacity: float = 0.7) -> str:
    """Self-contained HTML page (no external script or stylesheet: only the basemap tiles come from the internet).

    ``overlays``: [{name, uri, bounds:(south, west, north, east), checked}]; ``vectors``: [{name, geojson (dict), checked}];
    ``view_bounds``: (south, west, north, east) in degrees or None (world view)."""
    data = {
        "overlays": [{"name": o["name"], "uri": o["uri"], "bounds": [o["bounds"][0], o["bounds"][1], o["bounds"][2], o["bounds"][3]],
                      "checked": bool(o.get("checked"))} for o in overlays],
        "vectors": [{"name": v["name"], "geojson": v["geojson"], "checked": bool(v.get("checked", True))} for v in vectors],
        "view": list(view_bounds) if view_bounds else None,
        "opacity": float(opacity),
    }
    return _TEMPLATE.replace("/*__DATA__*/", _js_json(data))


_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Space GeoVision web map</title>
<style>
html,body{height:100%;margin:0;overflow:hidden;font:12px Arial,sans-serif;background:#222}
#c{position:absolute;left:0;top:0;cursor:grab} #c.drag{cursor:grabbing}
.panel{position:absolute;background:rgba(255,255,255,.95);padding:6px 10px;border-radius:5px;box-shadow:0 1px 5px rgba(0,0,0,.4);z-index:5}
#layers{top:8px;right:8px;max-width:300px;max-height:75%;overflow:auto}
#layers h4{margin:4px 0 2px;font-size:12px} #layers label{display:block;margin:2px 0} #layers input[type=range]{width:90px;vertical-align:middle;margin-left:6px}
#coords{bottom:6px;left:50%;transform:translateX(-50%);pointer-events:none}
#zoom{top:8px;left:8px;padding:0} #zoom button{display:block;width:30px;height:30px;font-size:18px;border:0;background:transparent;cursor:pointer}
#attr{bottom:6px;right:6px;font-size:10px;padding:2px 6px} #msg{bottom:30px;left:8px;color:#a00;font-size:11px}
</style></head><body>
<canvas id="c"></canvas>
<div id="zoom" class="panel"><button id="zi" title="Zoom in">+</button><button id="zo" title="Zoom out">&minus;</button></div>
<div id="layers" class="panel"></div>
<div id="coords" class="panel">move the mouse over the map</div>
<div id="attr" class="panel"></div><div id="msg" class="panel" style="display:none"></div>
<script>
"use strict";
const D = /*__DATA__*/;
const T = 256;
const BASE = [
 {name:'Esri World Imagery', url:'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', max:19, attr:'Tiles \u00a9 Esri'},
 {name:'OpenStreetMap', url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png', max:19, attr:'\u00a9 OpenStreetMap contributors'},
 {name:'Google Satellite (unofficial tiles)', url:'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', max:20, attr:'Google'}
];
let base = 0, z = 2, cx = 0, cy = 0;               // cx, cy: view centre in world pixels at zoom z
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const tiles = new Map(), RAD = Math.PI / 180;
const worldX = (lon, zz) => (lon + 180) / 360 * T * Math.pow(2, zz);
const worldY = (lat, zz) => { const s = Math.sin(Math.max(-85.05, Math.min(85.05, lat)) * RAD); return (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * T * Math.pow(2, zz); };
const lonOf = (x, zz) => x / (T * Math.pow(2, zz)) * 360 - 180;
const latOf = (y, zz) => Math.atan(Math.sinh(Math.PI * (1 - 2 * y / (T * Math.pow(2, zz))))) / RAD;
let queued = false;
function redraw() { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; draw(); }); } }
function resize() { cv.width = innerWidth; cv.height = innerHeight; redraw(); }
function tile(b, zz, x, y) {
  const k = b + '/' + zz + '/' + x + '/' + y; let t = tiles.get(k);
  if (!t) { t = new Image(); t.ok = false; t.onload = () => { t.ok = true; redraw(); }; t.onerror = () => { t.bad = true; };
    t.src = BASE[b].url.replace('{z}', zz).replace('{x}', x).replace('{y}', y); tiles.set(k, t);
    if (tiles.size > 600) { tiles.delete(tiles.keys().next().value); } }
  return t;
}
function draw() {
  const w = cv.width, h = cv.height, x0 = cx - w / 2, y0 = cy - h / 2, n = Math.pow(2, z);
  ctx.fillStyle = '#222'; ctx.fillRect(0, 0, w, h);
  const zt = Math.min(z, BASE[base].max), f = Math.pow(2, z - zt), tz = T * f;   // tiles of zoom zt drawn scaled by f
  const tx0 = Math.floor(x0 / tz), tx1 = Math.floor((x0 + w) / tz), ty0 = Math.floor(y0 / tz), ty1 = Math.floor((y0 + h) / tz), nt = Math.pow(2, zt);
  for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) {
    if (ty < 0 || ty >= nt) continue;
    const t = tile(base, zt, ((tx % nt) + nt) % nt, ty);
    if (t.ok) ctx.drawImage(t, tx * tz - x0, ty * tz - y0, tz, tz);
  }
  layers.forEach(L => { if (!L.on.checked) return;
    ctx.globalAlpha = parseFloat(L.op.value);
    if (L.img) { if (!L.img.ok) return;
      const [s, w_, n_, e] = L.bounds, px0 = worldX(w_, z) - x0, px1 = worldX(e, z) - x0, py0 = worldY(n_, z) - y0, py1 = worldY(s, z) - y0;
      ctx.imageSmoothingEnabled = true; ctx.drawImage(L.img, px0, py0, px1 - px0, py1 - py0);
    } else { ctx.globalAlpha = 1; drawGeo(L.geojson, x0, y0); }
    ctx.globalAlpha = 1; });
  document.getElementById('attr').textContent = BASE[base].attr;
}
function drawGeo(g, x0, y0) {
  if (!g) return; const t = g.type;
  if (t === 'FeatureCollection') return g.features.forEach(f => drawGeo(f.geometry, x0, y0));
  if (t === 'Feature') return drawGeo(g.geometry, x0, y0);
  if (t === 'GeometryCollection') return g.geometries.forEach(q => drawGeo(q, x0, y0));
  const P = c => [worldX(c[0], z) - x0, worldY(c[1], z) - y0];
  const ring = (r, close) => { r.forEach((c, i) => { const p = P(c); i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); }); if (close) ctx.closePath(); };
  ctx.lineWidth = 2; ctx.strokeStyle = '#ff3b30'; ctx.fillStyle = 'rgba(255,59,48,.08)';
  if (t === 'LineString') { ctx.beginPath(); ring(g.coordinates, false); ctx.stroke(); }
  else if (t === 'MultiLineString') { ctx.beginPath(); g.coordinates.forEach(r => ring(r, false)); ctx.stroke(); }
  else if (t === 'Polygon') { ctx.beginPath(); g.coordinates.forEach(r => ring(r, true)); ctx.fill('evenodd'); ctx.stroke(); }
  else if (t === 'MultiPolygon') { ctx.beginPath(); g.coordinates.forEach(pg => pg.forEach(r => ring(r, true))); ctx.fill('evenodd'); ctx.stroke(); }
  else if (t === 'Point' || t === 'MultiPoint') { (t === 'Point' ? [g.coordinates] : g.coordinates).forEach(c => { const p = P(c); ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 7); ctx.fillStyle = '#ff3b30'; ctx.fill(); }); }
}
// ---- layers panel
const layers = [], panel = document.getElementById('layers');
function head(txt) { const h = document.createElement('h4'); h.textContent = txt; panel.appendChild(h); }
head('Base map');
BASE.forEach((b, i) => { const l = document.createElement('label'), r = document.createElement('input'); r.type = 'radio'; r.name = 'base'; r.checked = i === 0;
  r.onchange = () => { base = i; z = Math.min(z, 21); redraw(); }; l.appendChild(r); l.appendChild(document.createTextNode(' ' + b.name)); panel.appendChild(l); });
function addLayer(name, checked, extra) {
  const l = document.createElement('label'), on = document.createElement('input'), op = document.createElement('input');
  on.type = 'checkbox'; on.checked = checked; op.type = 'range'; op.min = 0; op.max = 1; op.step = 0.05; op.value = D.opacity; op.title = 'opacity';
  on.onchange = op.oninput = redraw; l.appendChild(on); l.appendChild(document.createTextNode(' ' + name)); if (extra.img) l.appendChild(op); panel.appendChild(l);
  layers.push(Object.assign({on: on, op: op}, extra));
}
if (D.overlays.length) head('Raster layers (check to show; slider = opacity)');
D.overlays.forEach(o => { const im = new Image(); im.ok = false; im.onload = () => { im.ok = true; redraw(); }; im.onerror = () => { const m = document.getElementById('msg'); m.style.display = 'block'; m.textContent = 'Could not decode overlay: ' + o.name; }; im.src = o.uri; addLayer(o.name, o.checked, {img: im, bounds: o.bounds}); });
if (D.vectors.length) head('Vector layers');
D.vectors.forEach(v => addLayer(v.name, v.checked, {geojson: v.geojson}));
// ---- interaction
function setZoom(nz, px, py) {                       // zoom keeping screen point (px, py) fixed
  nz = Math.max(1, Math.min(21, nz)); if (nz === z) return; px = px == null ? cv.width / 2 : px; py = py == null ? cv.height / 2 : py;
  const lon = lonOf(cx - cv.width / 2 + px, z), lat = latOf(cy - cv.height / 2 + py, z); z = nz;
  cx = worldX(lon, z) - px + cv.width / 2; cy = worldY(lat, z) - py + cv.height / 2; redraw();
}
let drag = null;
cv.onmousedown = e => { drag = [e.clientX, e.clientY, cx, cy]; cv.classList.add('drag'); };
addEventListener('mouseup', () => { drag = null; cv.classList.remove('drag'); });
addEventListener('mousemove', e => {
  if (drag) { cx = drag[2] - (e.clientX - drag[0]); cy = drag[3] - (e.clientY - drag[1]); redraw(); }
  const x = cx - cv.width / 2 + e.clientX, y = cy - cv.height / 2 + e.clientY;
  document.getElementById('coords').textContent = latOf(y, z).toFixed(5) + '\u00b0, ' + lonOf(x, z).toFixed(5) + '\u00b0   zoom ' + z;
});
cv.onwheel = e => { e.preventDefault(); setZoom(z + (e.deltaY < 0 ? 1 : -1), e.clientX, e.clientY); };
cv.ondblclick = e => setZoom(z + 1, e.clientX, e.clientY);
document.getElementById('zi').onclick = () => setZoom(z + 1); document.getElementById('zo').onclick = () => setZoom(z - 1);
function fit(s, w, n, e) {
  let best = 2; for (let k = 2; k <= 19; k++) { if (Math.abs(worldX(e, k) - worldX(w, k)) <= cv.width * 0.9 && Math.abs(worldY(s, k) - worldY(n, k)) <= cv.height * 0.9) best = k; else break; }
  z = best; cx = (worldX(w, z) + worldX(e, z)) / 2; cy = (worldY(n, z) + worldY(s, z)) / 2;
}
addEventListener('resize', resize); resize();
if (D.view) fit(D.view[0], D.view[1], D.view[2], D.view[3]); else { z = 2; cx = worldX(0, z); cy = worldY(20, z); }
redraw();
</script></body></html>
"""
