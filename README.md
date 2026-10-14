# Space GeoVision

Desktop application (PyQt5) for optical, SAR and vector geospatial analysis.
Sentinel-1 preprocessing is performed by **ESA SNAP** (`gpt`); classification, change detection and water
mapping are implemented in GUI-independent, unit-tested Python modules that also run from a command line.

**License:** MIT. **Python:** 3.10+. **Status:** research software; see `KNOWN_ISSUES.md`.

## Install
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # core
pip install -r requirements-optional.txt                 # optional: SAM, web maps, WhiteboxTools
python main.py
```
GDAL is installed through the `rasterio` wheels on Windows, macOS and Linux. On Linux you may need
`sudo apt install libgl1 libxkbcommon-x11-0`.

### Sentinel-1 (SNAP)
1. Install ESA SNAP (https://step.esa.int) including the Sentinel-1 toolbox.
2. Tell the application where `gpt` is: menu *SAR Water Classification → SNAP Settings*, or set `SNAP_GPT`.
3. Use the **original** Sentinel-1 GRD product (`.zip` or `.SAFE`), not a GeoTIFF export.
4. SNAP downloads precise orbit files and DEM tiles automatically: internet access is required.
   Use a *subset* polygon for your study area; full IW scenes at 10 m are very large.

Menu *SAR Water Classification* (step by step):
1. **Select Sentinel-1 GRD Product** (.zip or .SAFE folder; shows the scene footprint)
2. **Run SNAP Preprocessing**: orbit file, thermal-noise removal, σ⁰ calibration, speckle filter, Range-Doppler terrain
   correction, dB. **Study area (any place on Earth):** whole scene (bounds detected automatically from the raw product),
   bounding box (type lon/lat or drag a rectangle on the footprint map), centre point + radius in km, or a pasted WKT polygon.
   The area is checked against the scene footprint and the expected output size is shown.
3. **Choose Water Threshold**: histogram, tile-based bimodal threshold (recommended), fixed value, sensitivity table
4. **Classify Water**: mask, area, warnings, GeoTIFF export (0 land, 1 water, 255 no data)
5. **Flood Mapping (before / after)**: two σ⁰ dB GeoTIFFs of the same track -> permanent / flooded / receded / uncertain
Also: *Load processed σ⁰ dB GeoTIFF* (skip SNAP) and *SNAP Settings*.
Every SNAP run writes `*_graph.xml` and `*.processing.json` (SNAP version, parameters, command) next to the GeoTIFF.
Map projection must be `WGS84(DD)` (default) or `AUTO:42001`; `EPSG:4326` made SNAP 10.0.0 fail on our test scene.

### Segment Anything (optional)
`pip install torch segment-anything`, download a checkpoint from the official repository
(`sam_vit_h_4b8939.pth`, `sam_vit_l_0b3195.pth` or `sam_vit_b_01ec64.pth`) and either put it in `./models/`,
set `SAM_CHECKPOINT_DIR`, or select it when asked (the choice is remembered).

## Command line (reproduces the experiments without the GUI)
```bash
python cli.py sar-preprocess S1A_IW_GRDH_...zip out/ --pol VV --subset "POLYGON((lon lat, ...))" --verify
python cli.py water out/S1A_..._VV_dB.tif out/water.tif --method otsu --reference ref_water.tif
python cli.py classify scene.tif train.gpkg class out/ --model rf --validation val.gpkg
python cli.py change t1.tif t2.tif out/ --k 3
python cli.py flood-db before_dB.tif after_dB.tif out/ [--threshold -14] [--min-drop-db 3]
python cli.py sar-preprocess product.zip out/ --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX      # study area as a box
python cli.py sar-preprocess product.zip out/ --center LON LAT --radius-km 20            # centre + radius
python cli.py sar-preprocess product.zip out/                                           # whole scene (large!)
python cli.py scenes a.zip b.zip --point Name LON LAT        # tracks, footprints, coverage
python cli.py sar-probe product.zip out/ --subset "POLYGON((...))"   # find a SNAP configuration that initialises
```
* `classify` never reports training accuracy. Without `--validation`, whole spatial blocks of the training samples are held out.
* `change` uses robust radiometric normalisation and a statistical threshold; it flags degenerate results (e.g. >50 % changed).
* `--verify` compares the SNAP parameter names we send with those advertised by your installed SNAP version.

## Tests
```bash
python -m unittest discover -s tests -v        # or: pytest tests
```
The tests run without a GUI and without SNAP (a fake `gpt` checks command construction, progress parsing, errors and
cancellation). They verify our code, **not** SNAP's numerical output.

## Layout
| File | Role |
|---|---|
| `main.py`, `ui.py` | GUI (PyQt5) |
| `sar_snap.py` | SNAP/gpt graph builder and runner (no Qt) |
| `water_mapping.py` | tile-based bimodal / fixed / Otsu thresholds on σ⁰ dB, morphology, flood from two dates, validation |
| `flood.py` | before/after flood pipeline shared by GUI and CLI |
| `supervised.py`, `accuracy.py` | RF/SVM with held-out validation, confusion matrix, kappa, IoU/F1 |
| `change_detection.py` | CVA change detection with normalisation |
| `helpers.py`, `polygon_processor.py` | thin GUI wrappers; vector tools |
| `imageops.py`, `terrain.py`, `raster_expr.py`, `georef.py` | GUI-free filters/indices, terrain & hydrology, safe raster calculator, map-coordinate axes (unit-tested) |
| `tests/` | `python -m unittest discover -s tests`; GUI smoke: `SG_GUI_SMOKE=1 pytest tests/test_gui_smoke.py` |
| `cli.py` | headless interface |

**Export Map (v1.6.0, File > Export Map...).** Re-renders the map for print instead of grabbing the screen: choose DPI (36-1200), page size
(A5-A2, Letter, Legal) and orientation, margins, and the format (PDF, PNG, JPEG, TIFF, SVG). The area can be the whole layer, the current view, a
rectangle dragged on the map, or typed coordinates; any polygon layer (e.g. a watershed) can clip the map (pixels outside are blanked and the
outline drawn, and the page is cropped to the polygon). Optional overlay raster with opacity, vector layers, title/subtitle/note, coordinate
grid, scale bar, north arrow, colour bar and footer with the CRS. The raster is resampled to the page pixel size at the chosen DPI, so DPI
changes the PDF size too. Settings are remembered.

**Labelling validation points (v1.7.0).** Validation > 1 now also writes `points_to_label.kml` (open in Google Earth Pro and use its Historical
Imagery slider; placemarks carry only the id). Validation > 1b *Label Points in the App* shows image chips of the reference rasters you tick around
each point, buttons for your categories (keys 1-9, arrows to move), saves after every click and has links that open the point in Google Earth for web,
Esri Wayback or Google Maps in your browser. The classified map is never shown, so labelling stays blind.
