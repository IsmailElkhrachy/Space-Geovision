"""Raster helper classes used by the GUI. Numerics are in ``imageops`` and ``terrain`` (unit-tested, GUI-free)."""
import os
import tempfile

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import LineString, shape
from skimage.feature import canny
from sklearn.cluster import KMeans, MeanShift

import imageops
import terrain

try:
    import contextily as ctx     # optional basemaps
except ImportError:
    ctx = None


def save_temp_image(image, profile):
    """
    Save a NumPy array as a temporary GeoTIFF file.

    Args:
        image (numpy.ndarray): The image data as a NumPy array. Expected shape:
            - For single-band: (height, width) or (1, height, width)
            - For multi-band: (bands, height, width)
        profile (dict): The rasterio profile (metadata) for the image. Must include:
            - height, width, count, dtype, crs, transform

    Returns:
        str: Path to the temporary file.
    """
    # Validate input array shape
    if image.ndim == 2:  # Single-band image (height, width)
        image = np.expand_dims(image, axis=0)  # Reshape to (1, height, width)
    elif image.ndim == 3:  # (bands, height, width): already in the correct shape
        pass
    else:
        raise ValueError(
            f"Input image must have shape (height, width) or (bands, height, width). "
            f"Got shape: {image.shape}"
        )

    # Validate profile
    required_keys = ['height', 'width', 'count', 'dtype', 'crs', 'transform']
    for key in required_keys:
        if key not in profile:
            raise ValueError(f"Profile is missing required key: {key}")

    # Ensure the profile count matches the number of bands
    if profile['count'] != image.shape[0]:
        raise ValueError(
            f"Profile 'count' ({profile['count']}) does not match the number of bands in the image ({image.shape[0]})"
        )

    # Save the image as a temporary GeoTIFF file
    fd, path = tempfile.mkstemp(suffix='.tif')
    os.close(fd)                      # Windows cannot reopen a file that is still held open
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(image)
    return path

class ImageProcessor:
    """Raster operations used by the GUI.

    Image convention: 2-D = one band, 3-D = (bands, rows, cols) as read by ``rasterio``. Numerics live in the
    GUI-independent, unit-tested modules ``imageops`` (filters, enhancement) and ``terrain`` (DEM products,
    hydrology); these methods are thin wrappers that keep the older API.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def load_image(file_name):
        with rasterio.open(file_name) as src:
            image = src.read()
            profile = src.profile
        return image, profile

    @staticmethod
    def save_image(image, profile, output_path):
        """Write ``image`` (2-D or bands, rows, cols) with ``profile``; count and dtype are taken from the data."""
        arr = image[np.newaxis] if image.ndim == 2 else image
        prof = dict(profile)
        prof.update(count=arr.shape[0], height=arr.shape[1], width=arr.shape[2], dtype=str(arr.dtype))
        with rasterio.open(output_path, 'w', **prof) as dst:
            dst.write(arr)

    @staticmethod
    def save_image_oneBand(image, profile, file_name):
        prof = dict(profile)
        prof.update(count=1, dtype='float32', height=image.shape[-2], width=image.shape[-1])
        with rasterio.open(file_name, 'w', **prof) as dst:
            dst.write(np.asarray(image, dtype=np.float32).reshape(prof['height'], prof['width']), 1)

    # ------------------------------------------------------------------ enhancement
    @staticmethod
    def resample_image(image, profile, scale_factor, file_name):
        """Bilinear resampling by ``scale_factor``. Returns (image, NEW profile, file_name); the input profile is not changed."""
        new_width = int(round(profile['width'] * scale_factor))
        new_height = int(round(profile['height'] * scale_factor))
        out = imageops.resample(image, new_height, new_width)
        new_profile = dict(profile)
        old = profile['transform']
        new_profile.update({
            'width': new_width,
            'height': new_height,
            'transform': old * old.scale(profile['width'] / new_width, profile['height'] / new_height),
            'dtype': 'float32',
        })
        return out, new_profile, file_name

    @staticmethod
    def enhance_contrast(image, file_name):
        """CLAHE per band (value range and dtype preserved; the input array is not modified)."""
        return imageops.enhance_contrast(image), file_name

    @staticmethod
    def smooth_image(image, sigma=1, file_name=None):
        return imageops.smooth(image, sigma), file_name

    @staticmethod
    def sharpen_image(image, radius=1, amount=1, file_name=None):
        return imageops.sharpen(image, radius, amount), file_name

    @staticmethod
    def convert_to_grayscale(image, file_name=None):
        return imageops.grayscale(image), file_name

    @staticmethod
    def clip_raster(image, profile, bbox, file_name=None, use_pixel_coords=False, preserve_nodata=True):
        """Clip to ``bbox`` = (xmin, ymin, xmax, ymax) in map units (or pixels if ``use_pixel_coords``).

        The window is clamped to the raster, so a box that sticks out of the image gives the overlapping part and a
        box with no overlap raises. Returns (clipped, new_profile, output_path); a one-band result is returned 2-D.
        """
        try:
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Invalid input image - must be numpy array")
            if image.ndim == 2:
                image = image[np.newaxis]
            elif image.ndim != 3:
                raise ValueError("Image must be 2D or 3D array")
            _, rows, cols = image.shape
            xmin, ymin, xmax, ymax = [float(v) for v in bbox]
            tr = profile['transform']
            if use_pixel_coords:
                c0, c1, r0, r1 = xmin, xmax, ymin, ymax
            else:
                if abs(tr.b) > 1e-12 or abs(tr.d) > 1e-12:
                    raise ValueError("Rotated rasters cannot be clipped by a map-coordinate box")
                c0, c1 = sorted(((xmin - tr.c) / tr.a, (xmax - tr.c) / tr.a))
                r0, r1 = sorted(((ymax - tr.f) / tr.e, (ymin - tr.f) / tr.e))
            c0, c1 = int(np.floor(max(c0, 0))), int(np.ceil(min(c1, cols)))
            r0, r1 = int(np.floor(max(r0, 0))), int(np.ceil(min(r1, rows)))
            if c1 <= c0 or r1 <= r0:
                raise ValueError("The clip box does not overlap the raster")
            clipped = image[:, r0:r1, c0:c1].copy()

            new_profile = dict(profile)
            nodata = profile.get('nodata') if preserve_nodata else None
            if nodata is not None and np.issubdtype(clipped.dtype, np.floating) and np.isfinite(nodata):
                clipped = clipped.astype(np.float32)
                clipped[clipped == nodata] = np.nan
                new_profile['nodata'] = float('nan')
            new_profile.update({
                'height': clipped.shape[1], 'width': clipped.shape[2], 'count': clipped.shape[0],
                'dtype': str(clipped.dtype), 'transform': tr * tr.translation(c0, r0),
            })
            output_path = None
            if file_name:
                output_path = os.path.splitext(file_name)[0] + '_clip.tif'
                with rasterio.open(output_path, 'w', **new_profile) as dst:
                    dst.write(clipped)
            if clipped.shape[0] == 1:
                clipped = clipped[0]
            return clipped, new_profile, output_path
        except Exception as e:
            raise RuntimeError(f"Clipping failed: {e}") from e

    @staticmethod
    def apply_canny_edge(image, sigma=1):
        """Canny edges of the first band (or the luminance of an RGB image); uint8 0/255 with a band axis."""
        gray = imageops.grayscale(image)[0] if image.ndim == 3 and image.shape[0] >= 3 else (image[0] if image.ndim == 3 else image)
        gray = np.nan_to_num(gray.astype(np.float64), nan=float(np.nanmean(gray)) if np.isfinite(gray).any() else 0.0)
        return (canny(gray, sigma=sigma) * 255).astype(np.uint8)[np.newaxis, ...]

    @staticmethod
    def mean_shift_segmentation(image, bandwidth=0.2, max_fit_pixels=20000, seed=0):
        """Mean-shift clustering of the pixel values.

        Every band is min-max scaled to 0-1 first, so ``bandwidth`` is relative. Mean shift is O(n^2): the modes are
        fitted on a random sample of at most ``max_fit_pixels`` pixels and every pixel is assigned to the nearest mode.
        Returns (label image, number of clusters).
        """
        try:
            if not isinstance(image, np.ndarray):
                raise ValueError("Input must be a numpy array")
            img = np.squeeze(image)
            if img.ndim == 2:
                img = img[np.newaxis]
            if img.ndim != 3:
                raise ValueError("Unsupported image shape. Must be 2D or 3D array")
            nb, h, w = img.shape
            pixels = imageops.normalize(img).reshape(nb, -1).T
            valid = np.isfinite(pixels).all(axis=1)
            if valid.sum() < 2:
                raise ValueError("Not enough valid pixels")
            rng = np.random.default_rng(seed)
            idx = np.flatnonzero(valid)
            sample = pixels[rng.choice(idx, size=min(max_fit_pixels, idx.size), replace=False)]
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(sample)
            labels = np.full(h * w, -1, dtype=np.int32)
            labels[valid] = ms.predict(pixels[valid])
            return labels.reshape(h, w), int(len(ms.cluster_centers_))
        except Exception as e:
            raise ValueError(f"Segmentation failed: {e}") from e

    # ------------------------------------------------------------------ terrain (DEM)
    @staticmethod
    def _cell(dem, transform, crs):
        return terrain.cell_size(transform, np.asarray(dem).shape[-2:], crs)

    @staticmethod
    def calculate_hillshade(dem, transform, azimuth=315, altitude=45, z_factor=1, crs=None):
        """Hillshade uint8 (0 = no data). With a geographic ``crs`` the cell size is converted to metres."""
        xres, yres = ImageProcessor._cell(dem, transform, crs)
        return terrain.hillshade(np.squeeze(dem), xres, yres, azimuth, altitude, z_factor)

    @staticmethod
    def calculate_slope(dem, transform, output_unit='degrees', max_percent=1000, crs=None):
        xres, yres = ImageProcessor._cell(dem, transform, crs)
        return terrain.slope(np.squeeze(dem), xres, yres, output_unit, max_percent)

    @staticmethod
    def calculate_aspect(dem, transform, crs=None):
        xres, yres = ImageProcessor._cell(dem, transform, crs)
        return terrain.aspect(np.squeeze(dem), xres, yres)

    @staticmethod
    def generate_contours(dem, profile, interval=1, base=0.0):
        """Contour lines at base + k * interval as a GeoDataFrame (columns: elevation, length (CRS units), geometry)."""
        try:
            if dem is None or profile is None:
                raise ValueError("Missing DEM data or profile")
            z = np.squeeze(dem).astype(np.float64)
            if z.ndim != 2:
                raise ValueError(f"Invalid DEM dimensions: {z.shape}. Expected 2D array.")
            tr = profile['transform']
            lines = terrain.contour_lines(z, tr.a, tr.c, tr.e, tr.f, interval, base)
            if not lines:
                raise ValueError("No contour lines were found")
            gdf = gpd.GeoDataFrame({'elevation': [lev for lev, _ in lines],
                                    'geometry': [LineString(xy) for _, xy in lines]}, crs=profile.get('crs'))
            gdf['length'] = gdf.geometry.length
            return gdf
        except Exception as e:
            raise RuntimeError(f"Contour generation failed: {e}") from e

    @staticmethod
    def calculate_terrain_ruggedness(dem):
        """Terrain Ruggedness Index (Riley et al. 1999)."""
        return terrain.ruggedness(np.squeeze(dem))

    # ------------------------------------------------------------------ filters (band-wise, NaN = no data)
    def detect_nodata(self, image):
        """Return the first no-data value found in ``image`` (NaN for floats containing NaN), else None."""
        a = np.asarray(image)
        if np.issubdtype(a.dtype, np.floating):
            if np.any(a <= imageops.NODATA_FLOAT_LIMIT):
                return -3.4028235e+38
            return np.nan if np.isnan(a).any() else None
        for v in imageops.DEFAULT_NODATA:
            if np.any(a == v):
                return v
        return None

    def filter_nodata(self, image, nodata_values=None):
        """float32 copy with no-data values replaced by NaN. Default list: 0, -9999, -32768, -32767, -2^31 and
        anything <= -3e38 (255 is a valid 8-bit value and is not treated as no data)."""
        return imageops.mask_nodata(image, nodata_values)

    def remove_outliers(self, image, threshold=3):
        return imageops.remove_outliers(image, threshold)

    def apply_median_filter(self, image, size=3):
        return imageops.median_filter(image, size)

    def apply_gaussian_filter(self, image, sigma=1):
        return imageops.gaussian_filter(image, sigma)

    def apply_lee_filter(self, image, window_size=3, noise_variance=None):
        return imageops.lee_filter(image, window_size, noise_variance)

    def normalize_image(self, image):
        return imageops.normalize(image)

    def to_8bit(self, image):
        return imageops.to_8bit(image)

    # ------------------------------------------------------------------ hydrology
    def fill_sinks(self, dem, epsilon=None):
        """Depression filling (Priority-Flood with epsilon gradient). ``epsilon`` is ignored (kept for compatibility)."""
        return terrain.fill_depressions(np.squeeze(dem))

    def d8_flow_direction(self, dem, transform, crs=None):
        xres, yres = terrain.cell_size(transform, np.asarray(dem).shape[-2:], crs)
        return terrain.flow_direction(np.squeeze(dem), xres, yres)

    def calculate_flow_accumulation(self, flow_dir):
        return terrain.flow_accumulation(flow_dir)

    def extract_streams(self, flow_accumulation, threshold_method='percentile', threshold_value=95):
        return terrain.extract_streams(flow_accumulation, threshold_method, threshold_value)

    def strahler_order(self, flow_dir, streams):
        return terrain.stream_order(flow_dir, streams, 'strahler')

    def shreve_order(self, flow_dir, streams):
        return terrain.stream_order(flow_dir, streams, 'shreve')

    def streams_to_vector(self, stream_order, flow_dir, transform):
        """Stream lines as [{'geometry': LineString, 'properties': {'order': n}}], one merged feature set per order."""
        from shapely.geometry import MultiLineString
        from shapely.ops import linemerge
        segs = terrain.stream_segments(flow_dir, stream_order, transform.a, transform.c, transform.e, transform.f)
        by_order = {}
        for k, p0, p1 in segs:
            by_order.setdefault(k, []).append((p0, p1))
        features = []
        for k in sorted(by_order):
            merged = linemerge(MultiLineString(by_order[k]))
            for line in getattr(merged, 'geoms', [merged]):
                features.append({'geometry': line, 'properties': {'order': int(k)}})
        return features

    @staticmethod
    def get_raster_profile(dem_path):
        with rasterio.open(dem_path) as src:
            profile = src.profile.copy()
            profile.setdefault('crs', src.crs)
            profile.setdefault('transform', src.transform)
            return profile

    @staticmethod
    def validate_raster_profile(profile):
        """Return a copy of ``profile`` or raise ValueError when required keys are missing."""
        if profile is None:
            raise ValueError("Profile cannot be None")
        validated = dict(profile)
        missing = [k for k in ('crs', 'transform', 'width', 'height', 'dtype') if k not in validated]
        if missing:
            raise ValueError(f"Profile missing required keys: {missing}")
        return validated


class ImageClassifier:
    """Unsupervised classification and spectral indices."""

    @staticmethod
    def preprocess_image_for_classification(image):
        """(bands, rows, cols) -> (pixels, bands)."""
        a = image[np.newaxis] if image.ndim == 2 else image
        return a.reshape(a.shape[0], -1).T

    @staticmethod
    def classify_kmeans(X, n_clusters):
        """K-Means on a (pixels, features) array. Returns (model, labels)."""
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return model, model.fit_predict(X)

    @staticmethod
    def load_specific_band(file_path, band_index):
        """Read one band (1-based) as float32 with a leading band axis; returns (band, profile)."""
        with rasterio.open(file_path) as src:
            if not 1 <= band_index <= src.count:
                raise ValueError(f"Band {band_index} is not available in the file ({src.count} bands).")
            return src.read(band_index).astype('float32')[np.newaxis, ...], src.profile

    @staticmethod
    def calculate_ndvi(nir, red):
        """(NIR - Red) / (NIR + Red); NaN where undefined."""
        return imageops.normalized_difference(nir, red)

    @staticmethod
    def calculate_ndwi(green, nir):
        """McFeeters NDWI: (Green - NIR) / (Green + NIR)."""
        return imageops.normalized_difference(green, nir)

    @staticmethod
    def calculate_ndni(swir1, swir2):
        return imageops.normalized_difference(swir1, swir2)

    @staticmethod
    def calculate_avi(nir, red):
        """Advanced Vegetation Index; inputs must be reflectance (0-1)."""
        return imageops.avi(nir, red)

    @staticmethod
    def save_image(image, profile, output_path):
        prof = dict(profile)
        prof.update(count=1, dtype='float32', height=image.shape[-2], width=image.shape[-1])
        with rasterio.open(output_path, 'w', **prof) as dst:
            dst.write(np.asarray(image, dtype=np.float32).reshape(prof['height'], prof['width']), 1)


class RasterHelper:
    @staticmethod
    def raster_to_polygons(classified_image, transform, crs, skip_values=(0,)):
        """Polygonise a class raster. ``skip_values`` are classes left out (default 0 = background / unclassified)."""
        arr = np.asarray(classified_image)
        arr = arr[0] if arr.ndim == 3 else arr
        arr = arr.astype(np.int32)
        polygons, values = [], []
        for geom, value in shapes(arr, transform=transform):
            if int(value) in skip_values:
                continue
            polygons.append(shape(geom))
            values.append(int(value))
        gdf = gpd.GeoDataFrame({'class': values, 'geometry': polygons}, geometry='geometry', crs=crs)
        return gdf

    @staticmethod
    def save_classified_image_to_shapefile(classified_image, transform, crs, output_path, skip_values=(0,)):
        """Save class polygons with area (m2) and perimeter (m) computed in a local UTM CRS; the file keeps the raster CRS."""
        gdf = RasterHelper.raster_to_polygons(classified_image, transform, crs, skip_values)
        if gdf.empty:
            raise ValueError("No polygons to save (the raster contains only skipped classes)")
        metric = gdf if (gdf.crs is not None and gdf.crs.is_projected) else gdf.to_crs(gdf.estimate_utm_crs())
        gdf['area_sqm'] = metric.geometry.area.values
        gdf['perimeter_m'] = metric.geometry.length.values
        gdf.to_file(output_path, driver='ESRI Shapefile')
        return output_path
