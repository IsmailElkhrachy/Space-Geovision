# Known issues and what has NOT been verified

**Verified in this revision (automated tests, 51):** SNAP graph/command construction, progress/error/cancel handling,
product-name validation, confusion-matrix/kappa/IoU maths, blocked validation split, change detection (incl. regression
tests for the defects of v1.0.0), Otsu/water mapping.

**Verified on real data (author, Windows, SNAP 10.0.0):** `snap-check`, `sar-probe`, `sar-preprocess` (Sentinel-1 GRD, VV, study box,
`WGS84(DD)`) for a before and an after scene, `water --method tiles`. **Not yet run on real data:** `flood-db`, the GUI menu workflow
(rewritten, only compiled and statically checked), `supervised`, `change`, SAM.

**Study-area selection:** the GUI offers whole scene / bounding box (with a draggable rectangle on the footprint map) / centre + radius /
pasted WKT polygon. Vertex-by-vertex polygon drawing is not implemented; boxes crossing the antimeridian are refused. SRTM DEMs are refused above
60 N and below 56 S (use the Copernicus DEM there). Not yet exercised interactively.

**Manual GUI test checklist (do this before relying on the menu):** (1) menu SAR -> 1 select a .zip, footprint shown; (2) step 2 dialog opens,
box inside footprint, run, progress bar, result displayed; (3) step 3 histogram opens, "Compute" gives a threshold, "Use this threshold";
(4) step 4 shows two panels and statistics, save mask; (5) "Load processed σ⁰ dB GeoTIFF" then steps 3-4 again; (6) step 5 with the two dB files.

**Not verified (needs a machine with the GUI and SNAP):**
1. The PyQt5 GUI was only compiled and statically checked; it has not been exercised interactively.
2. SNAP parameter names were written from SNAP documentation; run `python cli.py sar-preprocess ... --verify`
   on your SNAP version (the `Copernicus 30m Global DEM` name and `nodataValueAtSea` are version dependent).
3. No SNAP run, classification or change-detection experiment of the manuscript has been reproduced here.
   All result cells in the manuscript are marked `[TO BE COMPLETED]` and must come from your own runs.

**Code-quality pass (v1.2.0).** ~16-20 % unreachable code was removed (archived in `docs/removed_dead_code.txt`); calls to methods that did
not exist (`remove_layer`, `show_attributes`, `load_vector_file`, `reload_layer`, ...) were fixed or removed; filters, terrain and hydrology
maths were rewritten as GUI-free modules (`imageops.py`, `terrain.py`, `raster_expr.py`) covered by unit tests; a global exception hook
and a safe worker shutdown were added so a failing slot no longer aborts the process. These checks were done **without** PyQt5/rasterio
(static analysis, stub-based import/instantiation, numpy unit tests). Still **not verified on a real machine:** the real GUI, SNAP, Windows.
Run `python -m unittest discover -s tests`, then `SG_GUI_SMOKE=1 pytest tests/test_gui_smoke.py`, then the manual checklist above.
Stream-network vector export is exercised by `tests/test_helpers.py` only when shapely is installed. Slope/aspect use central differences
(not Horn 1981). Some Arabic comments remain. `ui.py` is still a single large class.

**v1.3.x - georeferencing.** Rasters are now drawn in real map coordinates (degrees for geographic CRS, metres for projected CRS; pixel axes
only when a layer has no geotransform) and the status bar shows coordinates, row/col and value under the cursor. Root cause of the earlier
"selected layer has no geotransform / could not get bounds 'transform'" errors: displaying a derived array replaced the stored layer by one
with an empty profile. Derived layers (hillshade, slope, aspect, flow accumulation, streams, clips, filters) now keep their profile.
Rotated rasters fall back to pixel axes. Not verified on real data here.

**v1.4.0 - basins, HAND, TWI/SPI (DEM Processing Tools menu).** Tested with synthetic DEMs only (numpy unit tests and a stubbed GUI run);
not yet run on real data. Methods: D8 watershed (O'Callaghan & Mark 1984), Horton/Strahler/Schumm/Miller/Gravelius morphometrics,
HAND (Renno et al. 2008) on a depression-filled DEM, TWI (Beven & Kirkby 1979), SPI (Moore et al. 1991). Limits to state in the paper:
D8 on flat terrain gives parallel paths; lengths follow D8 paths; the perimeter counts cell edges (over-estimates); the specific catchment
area uses sqrt(cell area) as contour width. HAND and TWI are terrain-only susceptibility indicators, not flood extents; validate against
an independent flood map. Streams come from a flow-accumulation threshold, so every morphometric depends on it (report it).

**v1.5.1 - web map overlay (View > Toggle Google Maps).** A dialog lets you tick which layers to overlay (up to 12 rasters, plus vector layers). The page is self-contained (no Leaflet or other external script; v1.5.0 failed with 'Failed to resolve module specifier leaflet'); only the basemap tiles need the internet. Chosen layers are drawn as overlays on
Esri World Imagery / OpenStreetMap / Google Satellite, opened at the extent shown in the image view; opacity slider and per-layer
check boxes are on the page. Needs internet and PyQtWebEngine. The re-projection to Web Mercator (`webmap.to_web_mercator`) needs rasterio
and was NOT run in the build sandbox (only a guarded test exists). Rasters are decimated to <= 1600 px for display. The Google tile URL is
unofficial (not a licensed Google API): check Google's terms before using it in publications; Esri and OSM require attribution (shown).
The web map is a view only: it does not feed back into the image view.

**Behavioural notes**
* Small-object removal in the legacy `_clean_water_mask` uses 16 pixels; `water_mapping.map_water` (CLI) uses 100 by default.
* Pixels are not independent samples; accuracy confidence intervals are optimistic.
* Change detection produces *candidate* change; it does not distinguish urban change from phenology or shadows.

**Before resubmission**
* Create a git tag, archive it on Zenodo (DOI) and cite that DOI in Table 1 (the v1.0.0 release contained only LICENSE and README).
* Verify every reference against its DOI/Scopus record (several references in the rejected manuscript were wrong).
* Add a small example dataset (or download script) so reviewers can reproduce the case studies.

**Map-coordinate display (v1.3.0).** Georeferenced rasters are drawn in real coordinates: Easting/Northing (m) with the EPSG code for projected
data, Longitude/Latitude in degrees (aspect corrected by cos(latitude)) for geographic data, and the status bar shows coordinates, row/col and
value under the cursor. Rasters without a geotransform, and rotated grids, fall back to pixel axes. The classification, change-map and
composite views still use pixel axes. Not verified on the real GUI by the author of this change.

**v1.6.0 - Export Map.** Rendering (mapexport.py) is unit-tested (page size, DPI -> pixel size, crop window, polygon mask with holes, no-data handling,
all five formats). The dialog (mapexport_ui.py) and the rectangle drawing are only exercised under stubs by the author; not verified on the real GUI.
Limits: the crop snaps to whole pixels; the overlay raster must share the base raster's CRS and the base must be single-band; north-up rasters only;
raster formats are limited to 150 Mpx (e.g. A2 at 600 dpi is refused); the page is not georeferenced (use a GeoTIFF export for GIS use). Clipping uses
polygon layers in the project, not free-hand polygons.

**v1.6.1 - Change detection dialog.** Detect Changes now opens one guided window (explanation, EARLIER and LATER file fields with Browse, swap button, validation) instead of two unlabeled file dialogs. Checked only under stubs.

**v1.6.2.** Fixes a start-up crash in v1.6.1 (`detect_changes` missing: a bad edit removed it). The guided EARLIER/LATER dialog is now shared by Detect Changes and Sensitivity. New test tests/test_ui_wiring.py fails if any menu/button slot does not exist.

**v1.6.3.** Validation > Create sample points failed with "module validation has no attribute raster_classes": raster_classes, make_sampling_from_raster and validate_from_files were called by the GUI and CLI but missing from validation.py. Implemented and unit-tested (against a fake rasterio; not run with real rasterio by the author). Continuous rasters (e.g. change magnitude) are rejected with a hint to use the _classes.tif. New static test checks that every module function called from ui.py/cli.py exists.

**v1.7.0 - KML export and in-app labelling.** The KML writer and the labelling logic (labeler.py) are unit-tested; the labelling window (labeler_ui.py) was only
exercised under stubs, not on a real GUI. Google Earth history is NOT embedded (Google offers no tile/API access to it): the links open the web viewers and
Google Earth Pro opens the KML. The reference must be close in time to the event; the app does not check imagery dates.

**v1.7.1 - accuracy of merged classes corrected.** When the report merged several map classes into one category (e.g. change = classes 2+3), the samples of those classes were pooled as one simple random sample although the classes had different areas and equal numbers of points, which biased user/producer accuracy, overall accuracy and areas. Every map class is now kept as a stratum and categories are unions of strata (accuracy.olofsson_grouped; identical to the old result when categories equal classes). Reports from v1.7.0 and earlier with merged classes must be recomputed. The report also lists the sample per map class and the uncertain points per class.

**v1.7.2 - blind labelling.** Point ids were numbered class by class (1-100 = class 1, 101-200 = class 2, ...), so the id order in points_to_label.csv/.kml revealed the map class and labelling was not fully blind. Ids are now assigned after a random shuffle. Samples created by v1.7.1 or earlier: use Validation > 1c to make a shuffled blind copy and label that one. Old labels stay in the old folder; id_map.csv links the ids.

**v1.7.3 - Stack Sentinel-2 Bands.** Copernicus/EO Browser "Raw" downloads give one GeoTIFF per band. File > "Stack Sentinel-2 Bands (one file per date)..." merges B02,B03,B04,B08,B11 of one date into `<date>_S2L2A_stack.tif` (band order fixed, grids checked). Terminal: `python bandstack.py FOLDER`, and `python bandstack.py --check before.tif after.tif` to confirm both dates share the same grid.

**v1.7.4 - ee_label.py (automatic reference labels).** Standalone script: `python ee_label.py points_to_label.csv --date1 2020-02-10 --date2 2026-02-08 --project <GEE project>` labels every point change / no_change / uncertain from Google Earth Engine Dynamic World (10 m) around the two dates, writes `points_to_label_ee.csv` and can compute the accuracy report (`--key points_key.csv --raster ..._classes.tif`). Dynamic World is a product, not truth: report it as agreement with an independent land-cover product and check a manual subsample. Not tested against live Earth Engine in the development sandbox (logic tested with a fake ee).

**v1.7.5 - ee_label.py:** Earth Engine query now filters Dynamic World to the study points (the image count printed before was global and the mean was computed over the whole collection, which is slow); adds a summary of why points are uncertain.

**v1.7.6 - ee_label.py hand-labelling workflow.** Besides `<points>_ee.csv` the script writes `<points>_manual_todo.csv` (every uncertain point + N random audit points, blank labels and no automatic hint, so labelling stays blind) and `<points>_ee_audit_ids.csv`. Label the todo file in the app (Validation > 1b), then `python ee_label.py <points>.csv --merge <points>_manual_todo.csv --key points_key.csv --raster map_classes.tif` merges (hand label wins), prints the agreement between automatic and hand labels on the audit points and computes the accuracy on the merged labels (`_final.csv`).

**v1.7.7 - ee_label.py:** `--merge` accepts several hand-label files and writes `<points>_manual_rest.csv` (points that still rely on the automatic label) so the reference can be made fully visual in a second pass.

**v1.7.8 - hand-only accuracy.** `ee_label.py --merge <todo.csv> --hand-only --key ... --raster ...` computes the accuracy from the hand labels only (all uncertain points + the random audit), re-weighted by cell (map class x automatic status) with a two-phase bootstrap interval (`validation.evaluate_two_phase`). Using hand labels alone as if they were a stratified random sample would over-represent the uncertain points and bias the result.

**v1.8.0 - validation for any kind of map.** (1) The categories are no longer tied to change / flood: *Validation > 1. Create Sampling Points* asks for the
categories to label (`water:1;land:0`, `urban:1;forest:2;bare:3`, `dry:0,4;flooded:2;receded:3`, a preset, or one per map class) and saves them in
`validation_scheme.json` beside the points (names and merged classes only, never a point's class). The labelling window and *2. Compute Accuracy* read
that file, so nothing is retyped; labels are compared ignoring case/spaces; `=` or `:` both work in the spec; labels already in a CSV become buttons.
(2) The report now also gives, per category, precision, recall, F1 (Dice), IoU (Jaccard) with bootstrap 95 % intervals (points resampled inside each
map class), macro-F1, mean IoU, area-adjusted Cohen's kappa and quantity / allocation disagreement; all are computed from the area-weighted
proportion matrix, so unequal sampling does not bias them. (3) Optional ROC and precision-recall curves, AUC / average precision with intervals,
best-F1 and Youden thresholds, from a continuous layer on the same grid as the map (sigma0 dB, change magnitude): `Compute Accuracy` (layer + target
category) or `python cli.py validate-curves LAYER MAP LABELS KEY OUT --target water --low-is-target`. Curves are area-weighted (stratum pixels / points).
Not exercised on a real GUI (PyQt5 not available where this was written); covered by 22 new unit tests.
Limits: the interval covers sampling error only (not reference errors or a time gap); curves need the layer on the SAME grid as the sampled map.

**v1.8.1 - results are drawn and saved in real map coordinates.** Found while evaluating the program on a Sentinel-2 scene (Najran): (1) *Classify with
K-Means* and *Mean-Shift* wrote their GeoTIFF with the placeholder transform `from_origin(0, 0, 1, 1)` (origin 0, 1-unit pixel, often no CRS), so the
saved map had no real position; (2) the results of K-Means, Mean-Shift, edge detection and the land-use map were drawn on pixel axes because a derived
array without a layer of its own was given an empty profile; (3) the land-use GeoTIFF was never written (it depended on attributes that were never set);
(4) K-Means used only the first three bands (and mis-shaped the data for more than three bands) - it now uses every band of the layer; (5) the "Saved
successfully" message appeared even when the save dialog was cancelled. Now: one set of helpers in `georef.py` (`draw_raster`, `draw_classified`,
`inherit_profile`, `result_profile`, `write_geotiff`) is used by every display and export path - classification (supervised, K-Means, Mean-Shift), land-use
map, change map, water/flood dialogs, DEM preview, overlays and vector-edit view - so a result takes the transform and CRS of the layer it came from
(rescaled if its size differs) and is drawn in degrees or metres; pixel axes are used **only** when the source has no georeferencing, and a file is never written with a
made-up position. K-Means / Mean-Shift ids now start at 1 (0 = no data, as in every other classified output). A raster whose origin is exactly (0, 0)
with a 1-unit pixel (the old placeholder) is treated as not georeferenced. The supervised result is added to the layer list so the cursor read-out shows
map coordinates and the class value. Remaining pixel-space views by design: the SAM mask previews (axes hidden). 19 new unit tests (269 in total, 12 skipped
without rasterio/Qt/affine) cover the drawing extents, the legend, the placeholder transform, profile inheritance and what `write_geotiff` passes to
rasterio (with rasterio replaced by a recorder). **Not verified here:** the GUI itself and the GDAL writer (no PyQt5/rasterio in the test environment) -
run the manual check: classify, look at the axes, save, reopen in QGIS and compare the extent.

**v1.8.2 - classified layers keep their look when re-selected.** Found by the author: after a classification the map was correct (map coordinates, legend, class colours), but selecting another layer and coming back redrew the result with the generic continuous colour stretch and without legend. Fix: supervised (RF/SVM), K-Means and Mean-Shift results are stored as `classified` layers (class names, title, grid); `display_raster_layer` re-draws any such layer through the same routine that drew it the first time. Covered by `tests/test_classified_layer_redraw.py` (behavioural: same extent, legend, title and colours after a redraw). Layers opened from a saved classified GeoTIFF are plain rasters and are shown with the continuous colormap; run the classification inside the program to get the legend.

**v1.8.3 - console noise on Windows 11 24H2+.** A long `subprocess ... CreateProcess` traceback (a "Could not find the number of physical cores" warning from joblib/loky) appeared when scikit-learn ran, because `wmic.exe` no longer exists in recent Windows 11. It is harmless; `main.py` now sets `LOKY_MAX_CPU_COUNT` before scikit-learn is imported.
