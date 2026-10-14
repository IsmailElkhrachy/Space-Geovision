"""Band-wise raster operations (no Qt, no rasterio: unit-testable).

Convention used everywhere in the application: a 3-D array is (bands, rows, cols) as returned by ``rasterio``;
a 2-D array is a single band. Functions never modify their input, NaN is treated as no-data and kept as NaN, and
the data type of integer images is preserved (values rounded and clipped to the dtype range).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage as ndi

NODATA_FLOAT_LIMIT = -3.0e38
# 0 is a no-data fill for most optical/SAR products; 255 is NOT listed because it is a legitimate 8-bit value.
DEFAULT_NODATA = (0, -9999, -32768, -32767, -2147483648)


def _as3d(image: np.ndarray):
    a = np.asarray(image)
    if a.ndim == 2:
        return a[np.newaxis], True
    if a.ndim == 3:
        return a, False
    raise ValueError(f"Image must be 2-D (band) or 3-D (bands, rows, cols); got shape {a.shape}")


def per_band(image: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], out_dtype=np.float32) -> np.ndarray:
    """Apply ``fn`` (2-D float64 band -> 2-D array) to every band; result has the input's shape."""
    a, single = _as3d(image)
    out = np.empty(a.shape, dtype=out_dtype)
    for i in range(a.shape[0]):
        out[i] = fn(a[i].astype(np.float64))
    return out[0] if single else out


def restore_dtype(result: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cast a float result back to the reference's integer dtype (rounded, clipped); floats become float32."""
    ref = np.asarray(reference)
    if np.issubdtype(ref.dtype, np.integer):
        info = np.iinfo(ref.dtype)
        r = np.nan_to_num(result, nan=0.0)
        return np.clip(np.rint(r), info.min, info.max).astype(ref.dtype)
    return result.astype(np.float32)


def _nan_gaussian(band: np.ndarray, sigma: float) -> np.ndarray:
    nan = np.isnan(band)
    num = ndi.gaussian_filter(np.where(nan, 0.0, band), sigma, mode="reflect")
    den = ndi.gaussian_filter((~nan).astype(np.float64), sigma, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[nan] = np.nan
    return out


# ---------------------------------------------------------------------------------------------- enhancement
def enhance_contrast(image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """CLAHE per band. Each band is rescaled to 0-1, equalised and mapped back to its own original min-max range."""
    from skimage import exposure

    def one(b):
        nan = np.isnan(b)
        if nan.all():
            return b
        lo, hi = np.nanmin(b), np.nanmax(b)
        if hi <= lo:
            return b
        u = np.where(nan, lo, (b - lo) / (hi - lo))
        e = exposure.equalize_adapthist(u, clip_limit=clip_limit)
        out = lo + e * (hi - lo)
        out[nan] = np.nan
        return out
    return restore_dtype(per_band(image, one, np.float64), image)


def smooth(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return restore_dtype(per_band(image, lambda b: _nan_gaussian(b, sigma), np.float64), image)


def sharpen(image: np.ndarray, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask in the data's own units: out = b + amount * (b - gaussian(b, radius))."""
    def one(b):
        blur = _nan_gaussian(b, radius)
        return b + amount * (b - blur)
    return restore_dtype(per_band(image, one, np.float64), image)


def grayscale(image: np.ndarray) -> np.ndarray:
    """Luminance of the first three bands (ITU-R 601). Images with fewer than 3 bands are returned unchanged."""
    a, single = _as3d(image)
    if single or a.shape[0] < 3:
        return np.asarray(image).copy()
    g = 0.2125 * a[0].astype(np.float64) + 0.7154 * a[1] + 0.0721 * a[2]
    return restore_dtype(g[np.newaxis], image)


def resample(image: np.ndarray, new_rows: int, new_cols: int) -> np.ndarray:
    """Bilinear resize of every band to (new_rows, new_cols); NaN no-data is kept."""
    from skimage.transform import resize
    if new_rows < 1 or new_cols < 1:
        raise ValueError(f"Scale factor gives an empty image ({new_rows} x {new_cols})")
    a, single = _as3d(image)
    out = np.empty((a.shape[0], new_rows, new_cols), np.float32)
    for i in range(a.shape[0]):
        out[i] = resize(a[i].astype(np.float64), (new_rows, new_cols), order=1, preserve_range=True, anti_aliasing=False)
    return out[0] if single else out


# ---------------------------------------------------------------------------------------------- filters
def mask_nodata(image: np.ndarray, nodata_values: Optional[Sequence[float]] = None) -> np.ndarray:
    """float32 copy with the listed no-data values (default: DEFAULT_NODATA and anything <= -3e38) set to NaN."""
    vals = list(DEFAULT_NODATA if nodata_values is None else nodata_values)
    out = np.array(image, dtype=np.float32, copy=True)
    bad = ~np.isfinite(out) | (out <= NODATA_FLOAT_LIMIT)
    for v in vals:
        if v is None or not np.isfinite(v):
            continue
        bad |= out == np.float32(v)
    out[bad] = np.nan
    return out


def remove_outliers(image: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Per-band: values further than ``threshold`` standard deviations from the band mean become NaN."""
    def one(b):
        m, s = np.nanmean(b), np.nanstd(b)
        out = b.copy()
        out[(b > m + threshold * s) | (b < m - threshold * s)] = np.nan
        return out
    return per_band(image, one)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("Median window size must be a positive odd number")

    def one(b):
        nan = np.isnan(b)
        fill = np.nanmedian(b) if not nan.all() else 0.0
        out = ndi.median_filter(np.where(nan, fill, b), size=size, mode="reflect")
        out[nan] = np.nan
        return out
    return restore_dtype(per_band(image, one, np.float64), image)


def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return smooth(image, sigma)


def lee_filter(image: np.ndarray, window_size: int = 3, noise_variance: Optional[float] = None) -> np.ndarray:
    """Lee (1980) adaptive speckle filter. ``noise_variance`` defaults to the mean local variance of each band.
    Intended for linear-intensity SAR; do not apply it to dB data."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("Lee window size must be an odd number >= 3")

    def one(b):
        nan = np.isnan(b)
        x = np.where(nan, np.nanmean(b) if not nan.all() else 0.0, b)
        mean = ndi.uniform_filter(x, size=window_size, mode="reflect")
        var = np.maximum(ndi.uniform_filter(x * x, size=window_size, mode="reflect") - mean ** 2, 0.0)
        nv = float(np.mean(var)) if noise_variance is None else float(noise_variance)
        k = var / (var + nv + 1e-12)
        out = mean + k * (x - mean)
        out[nan] = np.nan
        return out
    return per_band(image, one, np.float32)


def normalize(image: np.ndarray) -> np.ndarray:
    """Per-band min-max scaling to 0-1 (float32). A constant band becomes 0."""
    def one(b):
        lo, hi = np.nanmin(b), np.nanmax(b)
        return (b - lo) / (hi - lo) if hi > lo else np.where(np.isnan(b), np.nan, 0.0)
    return per_band(image, one)


def to_8bit(image: np.ndarray) -> np.ndarray:
    """Per-band min-max stretch to 0-255 uint8 (NaN -> 0). uint8 input is returned unchanged."""
    a = np.asarray(image)
    if a.dtype == np.uint8:
        return a.copy()
    return np.rint(np.nan_to_num(normalize(a) * 255.0, nan=0.0)).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- spectral indices
def normalized_difference(a, b) -> np.ndarray:
    """(a - b) / (a + b) as float32, NaN where the denominator is zero or an input is NaN."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0, np.nan, (a - b) / den)
    return out.astype(np.float32)


def avi(nir, red) -> np.ndarray:
    """Advanced Vegetation Index [NIR (1 - RED)(NIR - RED)]^(1/3); inputs must be reflectance (0-1).
    The signed cube root keeps negative values (NIR < RED) instead of producing NaN."""
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    return np.cbrt(nir * (1.0 - red) * (nir - red)).astype(np.float32)
