# Standard Library Imports
import os
import sys
import tempfile
import time
from datetime import datetime  # Import the datetime class

# Third-Party Imports
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import LineString
import contextily as ctx
from skimage import exposure, measure, img_as_float, transform
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian, unsharp_mask
from skimage.transform import resize
from scipy.ndimage import convolve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, MeanShift
from sklearn.neighbors import KNeighborsClassifier
from joblib import Parallel, delayed
from rasterio.features import shapes
from shapely.geometry import shape
from pyproj import CRS
from skimage.filters import median
from skimage.filters import gaussian
from skimage.morphology import dilation, erosion, square
from skimage.filters import median, gaussian, rank
from skimage.morphology import dilation, erosion, square
from skimage.exposure import rescale_intensity
from scipy.ndimage import uniform_filter
from scipy import ndimage
from skimage.morphology import skeletonize
from shapely.geometry import LineString



# Print Python executable path
print(sys.executable)

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
    elif image.ndim == 3 and image.shape[0] > 3:  # Multi-band image (bands, height, width)
        pass  # Already in the correct shape
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
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmpfile:
        with rasterio.open(tmpfile.name, 'w', **profile) as dst:
            dst.write(image)
        return tmpfile.name

class ImageProcessor:
    def __init__(self):
        pass


    @staticmethod    
    def load_image(file_name):
        with rasterio.open(file_name) as src:
            image = src.read()
            profile = src.profile
        return image, profile

    @staticmethod
    def resample_image(image, profile, scale_factor, file_name):
        """
        Resamples the image to a new resolution based on the given scale factor.
        :param image: The input image as a numpy array.
        :param profile: The profile of the input image.
        :param scale_factor: The factor by which to scale the image.
        :param file_name: The name of the file being processed.
        :return: The resampled image, updated profile, and file name.
        """
        new_width = int(profile['width'] * scale_factor)
        new_height = int(profile['height'] * scale_factor)

        resampled_image = np.zeros((image.shape[0], new_height, new_width))
        for i in range(image.shape[0]):
            resampled_image[i] = resize(image[i], (new_height, new_width), order=1, preserve_range=True)

        profile.update({
            'width': new_width,
            'height': new_height,
            'transform': profile['transform'] * profile['transform'].scale(1 / scale_factor, 1 / scale_factor)
        })
        return resampled_image, profile, file_name

    @staticmethod
    def save_image(image, profile, output_path):
        """
        Saves the image to the specified output path.
        :param image: The image data to save.
        :param profile: The profile of the image.
        :param output_path: Path to save the image.
        """
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(image)

    @staticmethod
    def save_image_oneBand(image, profile, file_name):
        # Update profile for single-band image
        profile.update(count=1, dtype=rasterio.float32)
        with rasterio.open(file_name, 'w', **profile) as dst:
            dst.write(image, 1)

    @staticmethod
    def enhance_contrast(image, file_name):
        """
        Enhances the contrast of the image using CLAHE.
        
        :param image: The input image.
        :param file_name: The name of the file being processed.
        :return: The contrast-enhanced image and file name.
        """
        if image.shape[0] == 3:  # RGB image
            for i in range(3):
                image[i] = exposure.equalize_adapthist(image[i], clip_limit=0.03)
        else:  # Single-band image
            image[0] = exposure.equalize_adapthist(image[0], clip_limit=0.03)
        return image, file_name

    @staticmethod
    def smooth_image(image, sigma=1, file_name=None):
        """
        Applies a Gaussian filter to smooth the image.
        :param image: The input image.
        :param sigma: The sigma value for the Gaussian filter.
        :param file_name: The name of the file being processed.
        :return: The smoothed image and file name.
        """
        if image.shape[0] == 3:  # RGB image
            for i in range(3):
                image[i] = gaussian(image[i], sigma=sigma, preserve_range=True)
        else:  # Single-band image
            image[0] = gaussian(image[0], sigma=sigma, preserve_range=True)
        return image, file_name

    @staticmethod
    def sharpen_image(image, radius=1, amount=1, file_name=None):
        """
        Sharpens the image using an unsharp mask.
        
        :param image: The input image.
        :param radius: The radius of the unsharp mask.
        :param amount: The amount of sharpening.
        :return: The sharpened image.
        """
        if image.shape[0] == 3:  # RGB image
            for i in range(3):
                image[i] = unsharp_mask(image[i], radius=radius, amount=amount)
        else:  # Single-band image
            image[0] = unsharp_mask(image[0], radius=radius, amount=amount)
        return image, file_name

    @staticmethod
    def convert_to_grayscale(image, file_name=None):
        """
        Converts the image to grayscale.
        
        :param image: The input image.
        :param file_name: The name of the file being processed.
        :return: The grayscale image and file name.
        """
        if image.shape[0] == 3:  # RGB image
            gray_image = rgb2gray(image.transpose(1, 2, 0))
            return gray_image[np.newaxis, ...], file_name  # Add a new axis to match the original shape
        elif image.shape[0] == 4:  # RGBA image
            gray_image = rgb2gray(image[:3].transpose(1, 2, 0))  # Ignore the alpha channel
            return gray_image[np.newaxis, ...], file_name  # Add a new axis to match the original shape
        else:  # Single-band image
            return image, file_name

    @staticmethod
    def clip_raster(image, profile, bbox, file_name=None, use_pixel_coords=False, preserve_nodata=True):
        """
        Robust raster clipping with support for:
        - Pixel and geographic coordinates
        - NoData value handling
        - Comprehensive error checking
        - Multi-band support
        
        Args:
            image: Input image array (2D or 3D)
            profile: Rasterio profile dictionary
            bbox: (xmin, ymin, xmax, ymax) bounding box
            file_name: Optional output path
            use_pixel_coords: Whether bbox uses pixel coords (False for geographic)
            preserve_nodata: Whether to maintain NoData values
        
        Returns:
            (clipped_image, updated_profile, output_path)
        """
        try:
            # Validate input array
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Invalid input image - must be numpy array")
                
            # Ensure 3D array (bands, height, width)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
            elif image.ndim != 3:
                raise ValueError("Image must be 2D or 3D array")

            # Get NoData value if preserving
            nodata = profile.get('nodata', None) if preserve_nodata else None
            
            # Convert bbox to window
            xmin, ymin, xmax, ymax = bbox
            
            if use_pixel_coords:
                # Pixel coordinates - simple window
                window = Window(
                    col_off=int(max(0, xmin)),
                    row_off=int(max(0, ymin)),
                    width=int(min(image.shape[2] - xmin, xmax - xmin)),
                    height=int(min(image.shape[1] - ymin, ymax - ymin))
                )
            else:
                # Geographic coordinates - transform to pixel window
                transform = profile['transform']
                try:
                    window = rasterio.windows.from_bounds(
                        left=xmin,
                        bottom=ymin,
                        right=xmax,
                        top=ymax,
                        transform=transform
                    )
                    window = window.round_offsets().round_lengths()
                except Exception as e:
                    raise ValueError(f"Failed to convert geographic bounds to window: {str(e)}")

            # Validate window
            if window.width <= 0 or window.height <= 0:
                raise ValueError("Invalid window dimensions - would produce empty output")

            # Perform the clip
            clipped = image[
                :,
                int(window.row_off):int(window.row_off + window.height),
                int(window.col_off):int(window.col_off + window.width)
            ].copy()  # Important to copy to avoid memory issues

            # Handle NoData values
            if nodata is not None:
                if np.issubdtype(clipped.dtype, np.floating):
                    clipped = clipped.astype(np.float32)
                    clipped[clipped == nodata] = np.nan
                elif nodata != 0:
                    clipped[clipped == nodata] = 0

            # Update profile
            new_profile = profile.copy()
            new_profile.update({
                'height': clipped.shape[1],
                'width': clipped.shape[2],
                'transform': rasterio.windows.transform(window, profile['transform'])
            })

            # Save if output path provided
            if file_name:
                output_path = os.path.splitext(file_name)[0] + '_clip.tif'
                with rasterio.open(output_path, 'w', **new_profile) as dst:
                    dst.write(clipped)
            else:
                output_path = None

            # Return same dimensionality as input
            if clipped.shape[0] == 1:
                clipped = np.squeeze(clipped, axis=0)

            return clipped, new_profile, output_path

        except Exception as e:
            raise RuntimeError(f"Clipping failed: {str(e)}")
    
    @staticmethod
    def apply_canny_edge(image, sigma=1):
        """
        Applies Canny edge detection to the image.
        
        :param image: The input image.
        :param sigma: The sigma value for Canny edge detection.
        :return: The edge-detected image.
        """
        if image.shape[0] == 3:  # RGB image
            gray_image = rgb2gray(image.transpose(1, 2, 0))
        else:  # Single-band image
            gray_image = image[0]
        edges = canny(gray_image, sigma=sigma)
        return (edges * 255).astype(np.uint8)[np.newaxis, ...]  # Convert to uint8 and add a new axis

    @staticmethod
    def calculate_statistics(change_result):
        total_pixels = change_result.size
        changed_pixels = np.count_nonzero(change_result)
        total_area_of_change = changed_pixels
        percentage_change = (changed_pixels / total_pixels) * 100
        change_intensity = np.mean(change_result[change_result > 0])
        
        # Categorize changes into different intensity levels
        low_change = np.count_nonzero((change_result > 0) & (change_result <= 0.33))
        medium_change = np.count_nonzero((change_result > 0.33) & (change_result <= 0.66))
        high_change = np.count_nonzero(change_result > 0.66)
        
        low_change_percentage = (low_change / total_pixels) * 100
        medium_change_percentage = (medium_change / total_pixels) * 100
        high_change_percentage = (high_change / total_pixels) * 100
        
        return (total_area_of_change, percentage_change, change_intensity,
                low_change_percentage, medium_change_percentage, high_change_percentage)

    @staticmethod
    def mean_shift_segmentation(image, bandwidth=0.2):
        """
        Perform image segmentation using Mean-Shift algorithm with multi-band support
        
        :param image: Input image (numpy array)
        :param bandwidth: Clustering radius control parameter
        :return: Segmented image, number of clusters
        """
        try:
            # Validate input is a numpy array
            if not isinstance(image, np.ndarray):
                raise ValueError("Input must be a numpy array")
            # Remove singleton dimensions (e.g., grayscale stored as 3D)
            image = np.squeeze(image)
            # Handle 3D images (multi-band)
            if image.ndim == 3:
                # Assume (bands, height, width) format if first dimension <=4
                if image.shape[0] <= 4:
                    num_bands, height, width = image.shape
                    image_data = image.transpose(1, 2, 0)  # Reshape to (height, width, bands)
                else:
                    height, width, num_bands = image.shape
                    image_data = image
                # Flatten to (pixels × bands)
                pixels = image_data.reshape(-1, num_bands)
            # Handle 2D grayscale images
            elif image.ndim == 2:
                height, width = image.shape
                pixels = image.reshape(-1, 1)
                num_bands = 1
            else:
                raise ValueError("Unsupported image shape. Must be 2D or 3D array")
            # Normalize pixel values to [0, 1]
            pixels = img_as_float(pixels)
            # Apply Mean-Shift algorithm
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(pixels)
            labels = ms.labels_
            # Reshape results to original dimensions
            segmented_image = labels.reshape(height, width)
            num_clusters = len(np.unique(labels))
            return segmented_image, num_clusters
        except Exception as e:
            raise ValueError(f"Segmentation failed: {str(e)}")

    @staticmethod        
    def calculate_hillshade(dem, transform, azimuth=315, altitude=45, z_factor=1):
        # Add null value handling
        dem = np.nan_to_num(dem, nan=0.0)
        
        # Calculate gradients using transform values
        x_res = transform.a
        y_res = abs(transform.e)
        
        dx, dy = np.gradient(dem * z_factor, x_res, y_res)
        
        # Calculate slope and aspect
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)
        
        # Convert angles to radians
        azimuth_rad = np.radians(360.0 - azimuth + 90)  # Adjust for conventional direction
        altitude_rad = np.radians(altitude)
        
        # Calculate illumination angles
        shaded = np.sin(altitude_rad) * np.sin(slope) + \
                np.cos(altitude_rad) * np.cos(slope) * \
                np.cos(azimuth_rad - aspect)
        
        # Scale to 0-255 and handle NaN/Inf
        shaded = np.clip((shaded + 1) * 127.5, 0, 255).astype(np.uint8)
        return shaded

    # @staticmethod
    # def calculate_slope(dem, transform):
    #     """
    #     Calculate slope from DEM using gradient operator
        
    #     Parameters:
    #     dem (numpy array): DEM data
    #     transform (affine.Affine): Transform object with DEM's coordinate system
        
    #     Returns:
    #     slope (numpy array): Slope values in degrees
    #     """
    #     # Calculate gradients using transform values
    #     x_res = transform.a
    #     y_res = abs(transform.e)
    #     x, y = np.gradient(dem, x_res, y_res)
        
    #     # Calculate slope using gradient magnitudes
    #     slope = np.degrees(np.arctan(np.sqrt(x**2 + y**2)))
    #     return slope
    @staticmethod
    def calculate_slope(dem, transform, output_unit='degrees', max_percent=1000):
        """
        Calculate slope with realistic percentage constraints
        
        Parameters:
        dem (numpy array): DEM data (must be in elevation units)
        transform (affine.Affine): Proper geotransform
        output_unit (str): Output unit type
        max_percent (float): Maximum allowed percentage (default 1000%)
        
        Returns:
        Slope array(s) with validated values
        """
        # Verify input data
        if np.any(dem > 9000) or np.any(dem < -1000):  # Sanity check for elevation values
            raise ValueError("DEM values appear unrealistic (check units)")
        
        x_cell_size = abs(transform.a)
        y_cell_size = abs(transform.e)
        
        # Calculate gradients with edge_order=2 for better accuracy
        dz_dy, dz_dx = np.gradient(dem, y_cell_size, x_cell_size, edge_order=2)
        
        # Calculate slope magnitude
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        
        if output_unit == 'degrees':
            return np.clip(np.degrees(slope_rad), 0, 90)
        elif output_unit == 'percent_rise':
            percent = np.tan(slope_rad) * 100
            return np.clip(percent, 0, max_percent)
        elif output_unit == 'both':
            degrees = np.clip(np.degrees(slope_rad), 0, 90)
            percent = np.clip(np.tan(slope_rad) * 100, 0, max_percent)
            return (degrees, percent)
        else:
            raise ValueError("Invalid output unit")
 
    @staticmethod
    def calculate_aspect(dem, transform):
        x, y = np.gradient(dem, transform.a, -transform.e)
        aspect = np.degrees(np.arctan2(-x, y))
        aspect = (aspect + 360) % 360
        return aspect

    @staticmethod
    def generate_contours(dem, profile, interval=1):
        """
        Generate contours from DEM with proper coordinate handling
        Returns GeoDataFrame with contour lines and metadata
        """
        try:
            # Validate input data
            if dem is None or profile is None:
                raise ValueError("Missing DEM data or profile")
                
            # Ensure 2D array (squeeze band dimension if needed)
            dem = np.squeeze(dem).astype(np.float32)
            if dem.ndim != 2:
                raise ValueError(f"Invalid DEM dimensions: {dem.shape}. Expected 2D array.")

            # Calculate elevation range
            min_elev = np.nanmin(dem)
            max_elev = np.nanmax(dem)
            elev_range = max_elev - min_elev
            
            # Validate contour interval
            if interval <= 0:
                raise ValueError("Contour interval must be greater than 0")
            if interval > elev_range:
                raise ValueError(f"Interval {interval}m exceeds elevation range {elev_range:.1f}m")

            # Calculate geographic coordinates
            transform = profile['transform']
            x = np.arange(dem.shape[1]) * transform.a + transform.c
            y = np.arange(dem.shape[0]) * transform.e + transform.f

            # Generate contours using skimage
            contours = measure.find_contours(dem, level=interval)
            
            if not contours:
                raise ValueError(f"No contours found between {min_elev}-{max_elev}m with {interval}m interval")

            # Create GeoDataFrame with proper coordinate mapping
            geometries = []
            for contour in contours:
                # Convert array coordinates to geographic coordinates
                coords = [
                    (x[int(col)], y[int(row)])  # x=col*a + c, y=row*e + f
                    for row, col in contour
                ]
                if len(coords) > 1:  # Filter single-point "contours"
                    geometries.append(LineString(coords))

            if not geometries:
                raise ValueError("All contours were invalid (single-point lines)")

            gdf = gpd.GeoDataFrame(
                {'elevation': [interval]*len(geometries), 'geometry': geometries},
                crs=profile['crs']
            )
            
            # Add elevation metadata
            gdf = gdf.set_geometry('geometry')
            gdf['length_m'] = gdf.geometry.length            
            return gdf
        except Exception as e:
            raise RuntimeError(f"Contour generation failed: {str(e)}")

    @staticmethod    
    def calculate_terrain_ruggedness(dem):
        kernel = np.ones((3,3))
        kernel[1,1] = -8
        tri = convolve(dem, kernel, mode='constant', cval=0)
        return np.abs(tri)
    ########################################## Filters 
    def __init__(self):
        self.common_nodata_values = {
            np.int16: -32767,
            np.int32: -2147483648,
            np.float32: -3.4028235e+38,
            np.float64: -3.4028235e+38
        }

    def _validate_image(self, image):
            """Enhanced validation with NoData awareness"""
            if image.ndim not in [2, 3]:
                raise ValueError("Image must be 1-band (2D) or 3-band (3D)")
            
            # Auto-detect potential NoData values
            if np.issubdtype(image.dtype, np.integer):
                if np.min(image) == self.common_nodata_values.get(image.dtype, 0):
                    print(f"Warning: Detected possible NoData value {np.min(image)}")
            return image

    def detect_nodata(self, image):
        """Smart NoData detection for ArcMap/QGIS compatibility"""
        if np.issubdtype(image.dtype, np.floating):
            if np.any(image <= -3.4e+38):  # ArcMap float NoData
                return -3.4e+38
            return np.nan  # Standard float NoData
        
        # Check for common integer NoData values
        for val in [self.common_nodata_values.get(image.dtype, 0), 0, -9999, -32768,-32767 ,255, -3.4e+38]:
            if np.any(image == val):
                return val
        return None

    # def filter_nodata(self, image, nodata_value=None):
    #     """
    #     Enhanced NoData filtering with auto-detection
    #     Args:
    #         image: Input array
    #         nodata_value: Optional override (None for auto-detect)
    #     Returns:
    #         Image with NoData as NaN (converted to float)
    #     """
    #     image = self._validate_image(image)
        
    #     # Auto-detect if not specified
    #     if nodata_value is None:
    #         nodata_value = self.detect_nodata(image)
    #         if nodata_value is None:
    #             return image.astype(np.float32)  # No NoData found
        
    #     # Special handling for extreme float values
    #     if nodata_value == -3.4e+38:
    #         processed = np.where(image <= nodata_value, np.nan, image)
    #     else:
    #         processed = np.where(image == nodata_value, np.nan, image)
        
    #     return processed.astype(np.float32)
    
    def filter_nodata(self, image, nodata_values=None):
        """
        Enhanced NoData filtering with auto-detection.
        
        Args:
            image (np.ndarray): Input array, expected to be a 2D or 3D array.
            nodata_values (list, optional): List of NoData values to check. If None, auto-detect.
            
        Returns:
            np.ndarray: Image with NoData values replaced by NaN (converted to float).
        """
        image = self._validate_image(image)
        
        # Define common NoData values if not specified
        if nodata_values is None:
            nodata_values = [0, -9999, -32768,-32767 ,255, -3.4e+38]  # Add more as needed
        
        # Auto-detect NoData value if not in the list
        detected_nodata = self.detect_nodata(image)
        if detected_nodata is not None and detected_nodata not in nodata_values:
            nodata_values.append(detected_nodata)
        
        # Replace NoData values with NaN
        processed = np.copy(image).astype(np.float32)
        for nodata_value in nodata_values:
            processed = np.where(processed == nodata_value, np.nan, processed)
        
        return processed




    def _prepare_for_filtering(self, image):
        """Prepares image for filtering with NoData handling"""
        # Convert to float if needed
        if not np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32)
            
        # Store original NoData mask
        mask = np.isnan(image)
        
        # Temporary fill for filtering
        filled = image.copy()
        filled[mask] = 0  # Simple fill - could use interpolation
        
        return filled, mask

    def apply_median_filter(self, image, size=3):
        """NoData-aware median filter"""
        image = self._validate_image(image)
        filled, mask = self._prepare_for_filtering(image)
        
        if image.ndim == 3:
            filtered = np.stack([
                median(filled[...,i], footprint=np.ones((size,size))) 
                for i in range(image.shape[-1])
            ], axis=-1)
        else:
            filtered = median(filled, footprint=np.ones((size,size)))
        
        # Restore NoData
        return np.where(mask, np.nan, filtered)

    # def remove_outliers(self, image, threshold=3):
    #         """Outlier removal respecting NoData"""
    #         image = self._validate_image(image)
    #         filled, mask = self._prepare_for_filtering(image)
            
    #         axis = (0,1) if image.ndim == 3 else None
    #         mean = np.nanmean(image, axis=axis)
    #         std = np.nanstd(image, axis=axis)
            
    #         if image.ndim == 3:
    #             mean = mean[None,None,:]
    #             std = std[None,None,:]
            
    #         outliers = (filled > mean + threshold*std) | (filled < mean - threshold*std)
    #         return np.where(outliers | mask, np.nan, image)
    
    def remove_outliers(self, image, threshold=3):
        """
        Outlier removal respecting NoData and ensuring final image is in float32.
        
        Args:
            image (np.ndarray): Input array, expected to be a 2D or 3D array.
            threshold (int, optional): Threshold for outlier detection. Default is 3.
            
        Returns:
            np.ndarray: Image with outliers and NaN values removed, converted to float32.
        """
        image = self._validate_image(image)
        filled, mask = self._prepare_for_filtering(image)
        
        axis = (0,1) if image.ndim == 3 else None
        mean = np.nanmean(image, axis=axis)
        std = np.nanstd(image, axis=axis)
        
        if image.ndim == 3:
            mean = mean[None,None,:]
            std = std[None,None,:]
        
        outliers = (filled > mean + threshold*std) | (filled < mean - threshold*std)
        processed = np.where(outliers | mask, np.nan, image)
        
        # Remove NaN values by replacing them with the mean of the non-NaN values
        nan_mask = np.isnan(processed)
        processed[nan_mask] = np.nanmean(processed)
        
        return processed.astype(np.float32) 
   

    def apply_median_filter(self, image, size=3):
        """
        Apply median filter to reduce noise while preserving edges
        Args:
            image: Input image array
            size: Size of the filter window (default: 3)
        Returns:
            Median-filtered image
        """
        image = self._validate_image(image)
        if image.ndim == 3:
            return np.stack([median(image[..., i], footprint=np.ones((size, size))) 
                           for i in range(image.shape[-1])], axis=-1)
        return median(image, footprint=np.ones((size, size)))

    def apply_gaussian_filter(self, image, sigma=1):
        """
        Apply Gaussian blur for smoothing
        Args:
            image: Input image array
            sigma: Standard deviation for Gaussian kernel (default: 1)
        Returns:
            Gaussian-blurred image
        """
        image = self._validate_image(image)
        return gaussian(image, sigma=sigma, preserve_range=True, 
                       channel_axis=-1 if image.ndim == 3 else None) 
    
    
    
    
    def apply_lee_filter(self, image, window_size=3, noise_variance=None):
        """Vectorized Lee filter implementation using SciPy"""
        # Convert to float32 for memory efficiency
        image = image.astype(np.float32)
        
        # Handle NaN values (replace with mean)
        image = np.nan_to_num(image, nan=np.nanmean(image))
        
        # Compute local statistics
        local_mean = uniform_filter(image, size=window_size, mode='reflect')
        local_square_mean = uniform_filter(image**2, size=window_size, mode='reflect')
        local_variance = local_square_mean - local_mean**2
        
        # Calculate noise variance if not provided
        noise_variance = noise_variance or np.mean(local_variance)
        
        # Compute filter coefficients
        K = local_variance / (local_variance + noise_variance + 1e-6)
        
        # Apply filter
        filtered = local_mean + K * (image - local_mean)
        return filtered.clip(min=0)  # Prevent negative values

    def normalize_image(self, image):
        """
        Normalize image to [0, 1] range
        Args:
            image: Input image array
        Returns:
            Normalized image
        """
        image = self._validate_image(image)
        if image.ndim == 3:
            min_val = np.nanmin(image, axis=(0, 1))
            max_val = np.nanmax(image, axis=(0, 1))
            return (image - min_val) / (max_val - min_val + 1e-8)
        else:
            min_val = np.nanmin(image)
            max_val = np.nanmax(image)
            return (image - min_val) / (max_val - min_val + 1e-8)

    # def to_8bit(self, image):
    #     """
    #     Convert image to 8-bit (0-255) while preserving contrast
    #     Args:
    #         image: Input image array
    #     Returns:
    #         8-bit converted image
    #     """
    #     image = self._validate_image(image)
    #     if image.dtype == np.uint8:
    #         return image
    #     return rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)

    def to_8bit(self, image):
        """
        Convert image to 8-bit (0-255) while preserving contrast.
        
        Args:
            image (np.ndarray): Input image array.
        
        Returns:
            np.ndarray: 8-bit converted image.
        """
        image = self._validate_image(image)
        
        if image.dtype == np.uint8:
            return image
        
        # Rescale intensity to 8-bit range
        return rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    
    import numpy as np
    from scipy.ndimage import generic_filter

    def d8_flow_direction(self, dem, transform):
        """Calculate flow direction using D8 algorithm"""
        offsets = self._d8_offsets()  # Use the method here
        padded = np.pad(dem, 1, mode='edge')
        flow_dir = np.zeros_like(dem, dtype=np.uint8)
        
        for i in range(dem.shape[0]):
            for j in range(dem.shape[1]):
                if np.isnan(dem[i,j]):
                    continue
                    
                window = padded[i:i+3, j:j+3]
                dz = window[1,1] - window
                dz[1,1] = -np.inf  # Ignore center cell
                
                # Calculate slope using offset distances
                slope = np.zeros((3,3))
                for code, (di, dj) in offsets.items():
                    dist = np.sqrt(di**2 + dj**2) if di !=0 and dj !=0 else 1
                    slope[1+di, 1+dj] = dz[1+di, 1+dj] / dist
                
                max_slope = np.nanmax(slope)
                if max_slope > 0:
                    flow_dir[i,j] = 2 ** np.unravel_index(np.nanargmax(slope), slope.shape)[1]
        
        return flow_dir

    def calculate_flow_accumulation(self, flow_dir):
        """ArcGIS-style flow accumulation counting contributing cells"""
        rows, cols = flow_dir.shape
        accumulation = np.ones((rows, cols), dtype=np.int32)
        
        # Create D8 flow direction mapping
        offsets = self._d8_offsets()
        
        # Process cells in reverse topological order
        for i in reversed(range(rows)):
            for j in reversed(range(cols)):
                code = flow_dir[i, j]
                if code == 0:  # Pit cell
                    continue
                    
                di, dj = offsets[code]
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    accumulation[ni, nj] += accumulation[i, j]
        
        return accumulation

    def fill_sinks(self, dem, epsilon=1e-5):
        """
        Fill sinks in DEM using Wang & Liu (2006) method
        Returns: depression-filled DEM
        """
        import numpy as np
        from skimage.morphology import reconstruction
        
        # Create seed array (use np.copy explicitly)
        seed = np.array(dem, copy=True)  # Explicit NumPy copy
        seed[1:-1, 1:-1] = np.nanmax(dem)  # Fill interior with max elevation
        
        # Perform morphological reconstruction
        filled = reconstruction(seed, dem, method='erosion')
        
        # Apply epsilon to filled areas only
        return np.where(dem == filled, dem, filled + epsilon)
    
    def extract_streams(self, flow_accumulation, threshold_method='percentile', threshold_value=95):
        """
        Extract stream network from flow accumulation
        Args:
            threshold_method: 'percentile' or 'fixed'
            threshold_value: Percentile (0-100) or fixed cell count
        Returns:
            stream_network: Binary array (1=stream, 0=non-stream)
        """
        if threshold_method == 'percentile':
            threshold = np.percentile(flow_accumulation[flow_accumulation > 0], threshold_value)
        else:
            threshold = threshold_value
        
        return (flow_accumulation >= threshold).astype(np.uint8)

    def strahler_order(self, flow_dir, streams):
        """Calculate Strahler stream order"""
        offsets = self._d8_offsets()  # Reuse here
        order = np.where(streams, 1, 0).astype(np.uint8)
        rows, cols = flow_dir.shape
        
        for i in range(rows):
            for j in range(cols):
                if not streams[i,j]:
                    continue
                
                # Find upstream contributors
                contributors = []
                for code, (di, dj) in offsets.items():
                    ni, nj = i - di, j - dj  # Upstream direction
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if flow_dir[ni,nj] == code:
                            contributors.append(order[ni,nj])
                
                # Apply Strahler rules
                if len(contributors) == 1:
                    order[i,j] = contributors[0]
                elif len(contributors) > 1:
                    order[i,j] = max(contributors) + (1 if len(set(contributors)) == 1 else 0)
        
        return order

    def streams_to_vector(self, stream_order, flow_dir, transform):
        """Convert raster streams to vector features"""
        offsets = self._d8_offsets()  # Reused here
        skeleton = skeletonize(stream_order > 0)
        lines = []
        
        for i in range(skeleton.shape[0]):
            for j in range(skeleton.shape[1]):
                if skeleton[i,j]:
                    segments = []
                    current = (i, j)
                    
                    # Trace downstream using offsets
                    while True:
                        x, y = transform * (j + 0.5, i + 0.5)
                        segments.append((x, y))
                        
                        code = flow_dir[current]
                        if code == 0:
                            break
                            
                        di, dj = offsets.get(code, (0,0))
                        current = (current[0] + di, current[1] + dj)
                        
                        if not (0 <= current[0] < skeleton.shape[0] and 
                            0 <= current[1] < skeleton.shape[1]):
                            break
                    
                    if len(segments) > 1:
                        lines.append(LineString(segments))
        
        return lines

    def _d8_offsets(self):
        """Standard D8 flow direction offsets (ESRI convention)
        Returns:
            dict: {direction_code: (row_offset, col_offset)}
        """
        return {
            1: (0, 1),     # East
            2: (1, 1),     # Southeast
            4: (1, 0),     # South
            8: (1, -1),    # Southwest
            16: (0, -1),   # West
            32: (-1, -1),  # Northwest
            64: (-1, 0),   # North
            128: (-1, 1)   # Northeast
        }
    
    def get_raster_profile(self, dem_path):
        """Safe profile extraction with validation"""
        with rasterio.open(dem_path) as src:
            profile = src.profile.copy()
            
            # Ensure required keys exist
            if 'crs' not in profile:
                profile['crs'] = src.crs
            if 'transform' not in profile:
                profile['transform'] = src.transform
                
            return profile

    def validate_raster_profile(profile):
        """
        Validate and standardize raster profile
        Returns standardized profile dict or raises ValueError
        """
        if profile is None:
            raise ValueError("Profile cannot be None")
        
        # Handle different profile formats
        if isinstance(profile, dict):
            validated = profile.copy()
        elif hasattr(profile, '__dict__'):  # e.g., rasterio.profiles.Profile
            validated = vars(profile)
        else:
            raise ValueError(f"Expected dict or Profile, got {type(profile)}")
        
        # Required keys check
        required_keys = ['crs', 'transform', 'width', 'height', 'dtype']
        missing = [k for k in required_keys if k not in validated]
        if missing:
            raise ValueError(f"Profile missing required keys: {missing}")
        
        # CRS validation
        if not hasattr(validated['crs'], 'is_projected'):
            raise ValueError("CRS object missing projection information")
        
        return validated
    

  

class ImageClassifier:
    """
    A class to handle image classification tasks using Random Forest and SVM.
    """
    @staticmethod
    def preprocess_image_for_classification(image):
        """
        Prepares the image for classification by flattening it.
        """
        if image.shape[0] == 3:  # If the image is RGB
            flattened_image = image.transpose(1, 2, 0).reshape(-1, 3)
        else:  # If the image is grayscale
            flattened_image = image.reshape(-1, 1)
        return flattened_image

    @staticmethod
    def classify_random_forest(X, y):
        """
        Classifies the image using Random Forest.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, y_pred, accuracy

    @staticmethod
    def classify_svm(X, y):
        """
        Classifies the image using SVM.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(kernel='linear', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, y_pred, accuracy

    @staticmethod
    def classify_kmeans(X, n_clusters):
        """
        Classifies the image using K-Means clustering (Unsupervised).
        """
        model = KMeans(n_clusters=n_clusters, random_state=42)
        y_pred = model.fit_predict(X)
        return model, y_pred

    @staticmethod
    def classify_knn(X, y, n_neighbors=5):
        """
        Classifies the image using K-Nearest Neighbors (KNN).
        
        :param X: The input features (flattened image).
        :param y: The target labels.
        :param n_neighbors: The number of neighbors to use.
        :return: The model, predictions, and accuracy.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, y_pred, accuracy

    @staticmethod
    def load_specific_band(file_path, band_index):
        band_names = {
            1: "Coastal aerosol",
                2: "Blue",
                3: "Green",
                4: "Red",
                5: "NIR",
                6: "SWIR 1",
                7: "SWIR 2",
                8: "Panchromatic",
                9: "Cirrus",
                10: "Thermal 1",
                11: "Thermal 2"
        }
        with rasterio.open(file_path) as src:
            if band_index > src.count:
                raise ValueError(f"Band {band_index} is not available in the file.")
            
            band = src.read(band_index).astype('float32')
            band = band[np.newaxis, ...]  # Add a new axis to match the shape (1, height, width)
            return band, src.profile
    
    @staticmethod
    def calculate_ndvi(nir, red):
        """
        Calculates the NDVI index using the formula: (NIR - Red) / (NIR + Red).

        :param nir: NIR band data.
        :param red: Red band data.
        :return: NDVI image.
        """
        # Convert inputs to float arrays
        nir = np.array(nir, dtype=float)
        red = np.array(red, dtype=float)

        # Calculate NDVI with zero division handling
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red + 1e-10))  # Add epsilon to avoid division by zero
        return ndvi
    
    @staticmethod
    def calculate_ndwi(green, nir):
        # Calculate NDWI using float conversion
        green = np.array(green, dtype=float)
        nir = np.array(nir, dtype=float)
        
        # Calculate NDWI with zero division handling
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = np.where((green + nir) == 0, 0, (green - nir) / (green + nir + 1e-10))
        return ndwi
    
    @staticmethod
    def calculate_ndni(swir1, swir2):
        # Calculate NDNI using float conversion
        swir1 = np.array(swir1, dtype=float)
        swir2 = np.array(swir2, dtype=float)
        
        # Calculate NDNI with zero division handling
        with np.errstate(divide='ignore', invalid='ignore'):
            ndni = np.where((swir1 + swir2) == 0, 0, (swir1 - swir2) / (swir1 + swir2 + 1e-10))
        return ndni
    
    @staticmethod
    def calculate_avi(nir, red):
        # Calculate AVI using float conversion
        nir = np.array(nir, dtype=float)
        red = np.array(red, dtype=float)
        
        # Calculate AVI with zero division handling
        with np.errstate(divide='ignore', invalid='ignore'):
            avi = np.where((nir + red) == 0, 0, (nir * (1 - red) * (nir - red)) ** (1/3))            
        return avi
    
    @staticmethod
    def detect_changes(image1, image2):
        """
        Compute change detection with better dimension handling
        """
        # Ensure same number of bands
        min_bands = min(image1.shape[0], image2.shape[0])
        image1 = image1[:min_bands]
        image2 = image2[:min_bands]
        # Resample if needed using parallel processing
        if image1.shape != image2.shape:
            start_time = time.time()
            def resize_band(band, target_shape):
                return resize(band, target_shape, preserve_range=True, anti_aliasing=True)
            resampled = Parallel(n_jobs=-1)(delayed(resize_band)(image2[band], image1[band].shape) for band in range(min_bands))
            image2 = np.array(resampled)
            print(f"Resampling time: {time.time() - start_time} seconds")
        # Calculate normalized difference
        diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
        return diff / (np.maximum(image1, image2) + 1e-7)  # Prevent division by zero

    @staticmethod
    def save_image(image, profile, output_path):  # Fix parameter order
        # Update profile for single-band image
        profile.update(count=1, dtype=rasterio.float32)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(image, 1)


class RasterHelper:
    @staticmethod
    def raster_to_polygons(classified_image, transform, crs):
        """
        Convert a classified raster image to polygons.
        
        Args:
            classified_image (numpy.ndarray): The classified image (2D array).
            transform (rasterio.transform.Affine): The affine transform of the image.
            crs: The coordinate reference system of the raster.
        
        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame containing the polygons and their attributes.
        """
        polygons = []
        values = []
        
        # Generate shapes from the classified image
        for geom, value in shapes(classified_image, transform=transform):
            if value == 0:  # Skip background (if applicable)
                continue
            polygons.append(shape(geom))  # Convert to Shapely geometry
            values.append(value)  # Store the class value
        
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame({'class': values, 'geometry': polygons})
        gdf.set_crs(crs, inplace=True)  # Set the CRS from the input raster
        
        return gdf

    @staticmethod
    def save_classified_image_to_shapefile(classified_image, transform, crs, output_path):
        """
        Save a classified image as a Shapefile with area and perimeter attributes.
        
        Args:
            classified_image (numpy.ndarray): The classified image (2D array).
            transform (rasterio.transform.Affine): The affine transform of the image.
            crs: The coordinate reference system of the raster.
            output_path (str): The path to save the Shapefile.
        """
        # Convert the classified image to polygons
        gdf = RasterHelper.raster_to_polygons(classified_image, transform, crs)
        
        # Reproject to a projected CRS (e.g., UTM) to calculate area and perimeter in square meters
        if gdf.crs.is_geographic:  # Check if the CRS is geographic (e.g., EPSG:4326)
            # Automatically determine a suitable UTM zone for reprojection
            centroid = gdf.geometry.centroid.iloc[0]  # Use the centroid of the first feature
            utm_zone = int((centroid.x + 180) / 6) + 1
            target_crs = CRS.from_string(f"EPSG:326{utm_zone:02d}")  # Northern hemisphere UTM
            gdf = gdf.to_crs(target_crs)
        else:
            target_crs = gdf.crs  # Use the existing CRS if it's already projected
        
        # Calculate area (in square meters) and perimeter (in meters)
        gdf['area_sqm'] = gdf.geometry.area
        gdf['perimeter_m'] = gdf.geometry.length
        
        # Save the GeoDataFrame to a Shapefile
        gdf.to_file(output_path, driver='ESRI Shapefile')
        print(f"Shapefile saved successfully at: {output_path}")
