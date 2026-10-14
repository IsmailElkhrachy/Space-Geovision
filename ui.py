import os
import cv2
import re
import traceback
import logging
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
import tempfile
from PIL import Image
from skimage import img_as_float, measure, exposure
from skimage.feature import canny
from sklearn.cluster import MeanShift, KMeans
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsRectItem, QMainWindow, QAction, QFileDialog, QMessageBox, QVBoxLayout, QLabel, QFrame, QWidget, QRadioButton, QInputDialog, QProgressBar, QSizePolicy, QHBoxLayout, QListWidget, QListWidgetItem, QButtonGroup, QTabWidget, QPushButton, QColorDialog, QFontDialog, QProgressDialog, QTextEdit, QDialog, QApplication, QToolButton, QSplitter, QMenuBar, QToolBar, QMenu, QDoubleSpinBox, QDialogButtonBox, QStackedWidget, QComboBox, QLineEdit, QCheckBox, QSpinBox, QSlider, QGroupBox, QFormLayout, QScrollArea, QTableWidget, QTableWidgetItem
from PyQt5.QtPrintSupport import QPrinter, QPrintPreviewDialog
from PyQt5.QtCore import Qt, QRectF, QRect, QDate, QThread, pyqtSignal, QDateTime, QMutex, QMutexLocker, QMetaObject, QSize, QSettings, QTimer, QPointF, QStandardPaths
from PyQt5.QtGui import QColor, QFont, QPixmap, QImage, QIcon, QPainter, QDoubleValidator, QPen, QPolygonF, QBrush
from PyQt5.QtPrintSupport import QPrintDialog, QPageSetupDialog, QPrinterInfo
from helpers import ImageProcessor, ImageClassifier, RasterHelper
import sar_snap
import water_mapping
import flood
import validation
import supervised
import change_detection
import resources
import georef
from collections.abc import Mapping
import terrain
import basins
import webmap
import raster_expr
from resources import resource_path
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:  # SAM is optional: `pip install torch segment-anything`
    torch = sam_model_registry = SamPredictor = SamAutomaticMaskGenerator = None
    SAM_AVAILABLE = False
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh, PathCollection
from matplotlib.lines import Line2D
from rasterio import features  # Add this import
import colorsys
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
import shutil
from datetime import datetime

from polygon_processor import PolygonProcessor
from shapely.geometry import Polygon, MultiPolygon, LineString
from matplotlib.colors import Normalize, ListedColormap
import math

from matplotlib import colors
import sys

# Add to your PyQt5 imports at the top of the file
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView
from scipy.ndimage import uniform_filter
from copy import copy
from skimage import filters
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import psutil
from skimage.morphology import binary_closing, remove_small_objects
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices


# ---- optional extras: the core application starts without them --------------------------------
def _optional(name):
    try:
        return __import__(name)
    except ImportError:
        logging.warning("Optional package '%s' is not installed; the related feature is disabled.", name)
        return None


folium = _optional("folium")
leafmap = _optional("leafmap")
whitebox = _optional("whitebox")
ctx = _optional("contextily")
WhiteboxTools = getattr(whitebox, "WhiteboxTools", None)
Map = getattr(folium, "Map", None)
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings, QWebEnginePage

    class _LoggingWebPage(QWebEnginePage):
        """Sends JavaScript console messages of the web map to the application log (otherwise they are lost)."""
        def javaScriptConsoleMessage(self, level, message, line, source):
            logging.warning("web map JS %s:%s: %s", source, line, message)
except ImportError:
    QWebEngineView = None
    QWebEngineSettings = None
    _LoggingWebPage = None

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _get_cmap(cmap, lut=None):
    """Colormap by name or instance. ``plt.cm.get_cmap`` was removed in Matplotlib 3.9."""
    if not isinstance(cmap, mcolors.Colormap):
        cmap = matplotlib.colormaps[cmap]
    return cmap.resampled(lut) if lut else cmap


class Worker(QThread):
    """Runs ``operation(image, *args)`` off the GUI thread.

    A failed operation is reported once through ``error`` (no silent retries) and ``stop()`` is cooperative:
    it never terminates the thread, because killing a thread inside GDAL/numpy can corrupt the process.
    """
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, operation, image, *args):
        super().__init__()
        self.operation = operation
        self.image = image
        self.args = args
        self._is_running = True
        self._lock = QMutex()

    def run(self):
        try:
            self.status.emit("Starting operation...")
            self.progress.emit(-1)
            result = self.operation(self.image, *self.args)
            with QMutexLocker(self._lock):
                running = self._is_running
            if running:
                self.progress.emit(100)
                self.status.emit("Operation completed successfully.")
                self.finished.emit(result)
        except Exception as exc:
            logging.exception("Image operation failed")
            self.progress.emit(-2)
            self.status.emit(f"Error: {exc}")
            self.error.emit(f"Error: {exc}")

    def stop(self):
        with QMutexLocker(self._lock):
            self._is_running = False
        self.quit()
        self.wait(500)



class SnapWorker(QThread):
    """Runs the SNAP gpt chain off the GUI thread. No retries: a failed SNAP job is reported once."""
    progress = pyqtSignal(int, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, product, out_dir, params, gpt, heap_gb):
        super().__init__()
        self._args = (product, out_dir, params, gpt, heap_gb)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        product, out_dir, params, gpt, heap_gb = self._args
        try:
            result = sar_snap.run_full_chain(
                product, out_dir, params, gpt=gpt, java_heap_gb=heap_gb,
                progress=lambda v, m: self.progress.emit(int(v), m),
                cancelled=lambda: self._cancel)
            data, profile = sar_snap.read_geotiff(result.geotiff)
            self.done.emit((result, data, profile))
        except sar_snap.Cancelled:
            self.failed.emit('__cancelled__')
        except Exception as exc:
            self.failed.emit(str(exc))


class FuncWorker(QThread):
    """Runs fn(progress) off the GUI thread. fn gets a progress(percent, message) callback."""
    progress = pyqtSignal(int, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn(lambda v, m: self.progress.emit(int(v), str(m))))
        except Exception as exc:
            logging.exception("Background job failed")
            self.failed.emit(str(exc))


class SatelliteImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_styles()  # تحميل الأنماط
        # Ensure proper dialog parent relationships
        self.hist_dialog = None  # Add this in your class initialization
        # Initialize SAR workflow attributes
        self.current_sar_group = None  # Track current processing group
        self.sar_steps = {}  # Store processing steps data
        self.workflow_actions = []  # SAR workflow menu actions
        self.current_worker = None  # Track active worker
        self.image_processor = ImageProcessor()
        self.image_classifier = ImageClassifier()
        self.map_elements = []
        self.undo_stack = []
        self.redo_stack = []
        self.classified_image = None
        self.layers = {}
        self.current_layer = None
        self.change_result = None
        self.profile = None
        self.analysis_data = None
        self.segmentation_analysis = None
        self.sam_predictor = None
        self.classes = {}
        self.class_masks = {}
        self.class_points = {}
        self.class_boxes = {}
        self.current_class = None
        self.waiting_for_input = False
        self.google_maps_enabled = False
        self.leafmap_initialized = False
        self.current_worker = None  # Track the current worker thread
        self.active_image = None  # For storing mouse-selected images   
        self.polygon_processor = PolygonProcessor(self)  
        self.polygon_processor.setup_ui_connections()
        self.polygon_points = []  # Stores clicked points
        self.temp_polygon = None  # For visualization while drawing
        self.drawing = False      # Flag for drawing state
        self.statusBar().showMessage("Ready", 2000)  # Initial status
        # Initialize message counters
        self.message_counter = 0
        self.active_dialogs = [] 

        # SAR water
        self.error_log = []
        # Initialize workflow actions
        self.workflow_actions = []  # Will be populated with menu actions
        
        self.current_step = 0
        self.workflow_steps = ['load', 'snap', 'analyze', 'classify']

        self.initUI()   
        self.color_ramps = {
        'Spectral': ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', 
                    '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'],
        'RdYlGn': ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', 
                  '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'],
        'RdYlBu': ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf',
                  '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'],
        'PiYG': ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef',
                '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221'],
        'Set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                '#ffff33', '#a65628', '#f781bf', '#999999'],
        'Set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
                '#ffd92f', '#e5c494', '#b3b3b3'],
        'Set3': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']
    }

    def initUI(self):
        # Setup menu bar
        menubar = self.menuBar()
        #file_menu = menubar.addMenu('File')
        file_menu = self.menuBar().addMenu('File')    
        edit_menu = menubar.addMenu('Edit')
        satellite_menu = menubar.addMenu('Raster Processing Tools')
        edge_detection_menu = menubar.addMenu('Edge Detection Tools')
        image_classification_menu = menubar.addMenu('Classification Tools')
        self.sar_menu = self.menuBar().addMenu('SAR Water Classification')
        self._init_sar_workflow_actions()
        
        polygon_menu = menubar.addMenu('Vector Processing Tools') 
       
        # Attribute Table Submenu
        attr_table_menu = polygon_menu.addMenu('Attribute Table')        
        basic_polygon_menu = polygon_menu.addMenu('Basic Operations')

        #prepare_map_menu = menubar.addMenu('Map Setup')
        indices_menu = menubar.addMenu('Spectral Indices')
        change_detection_menu = menubar.addMenu('Change Detection Tools')
        validation_menu = menubar.addMenu('Validation')
        val_points_action = validation_menu.addAction('1. Create Sampling Points (blind labelling)...')
        val_points_action.triggered.connect(self.validation_sample_points)
        val_label_action = validation_menu.addAction('1b. Label Points in the App (blind)...')
        val_label_action.triggered.connect(self.validation_label_points)
        val_shuffle_action = validation_menu.addAction('1c. Shuffle Existing Points (hide class order, new blind copy)...')
        val_shuffle_action.triggered.connect(self.validation_shuffle_points)
        val_acc_action = validation_menu.addAction('2. Compute Accuracy from Labelled Points...')
        val_acc_action.triggered.connect(self.validation_accuracy)
        help_menu = menubar.addMenu('Help')         
        
        
        # Show Attribute Table
        show_attr_action = QAction('Show Table', self)
        show_attr_action.triggered.connect(self.polygon_processor.show_attribute_table)
        attr_table_menu.addAction(show_attr_action)

        
        # Edit Attributes
        edit_attr_action = QAction('Edit Attributes', self)
        edit_attr_action.triggered.connect(self.polygon_processor.edit_attributes)
        attr_table_menu.addAction(edit_attr_action)
        
                
        # Setup toolbar
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)
        self._setup_selection_tools()
        


        #Add Editing Toolbar - INSERT HERE
        self.toolbar_edit = self.addToolBar("Editing")
        self.actionStart_Editing = QAction("Start Editing", self)
        self.actionSave_Edits = QAction("Save Edits", self)
        self.actionStop_Editing = QAction("Stop Editing", self)

        self.toolbar_edit.addAction(self.actionStart_Editing)
        self.toolbar_edit.addAction(self.actionSave_Edits)
        self.toolbar_edit.addAction(self.actionStop_Editing)

        # Initially disable save/stop until editing starts
        self.actionSave_Edits.setEnabled(False)
        self.actionStop_Editing.setEnabled(False)

        # Create main widgets - EXISTING CODE CONTINUES
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.actionStart_Editing.triggered.connect(self.polygon_processor.start_editing)
        self.actionSave_Edits.triggered.connect(self.polygon_processor.save_edits)
        self.actionStop_Editing.triggered.connect(self.polygon_processor.stop_editing)        

        #self._init_sar_workflow_menu()


        # Create main widgets
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Setup vertical splitter
        splitter = QSplitter(Qt.Vertical)

        # Upper part: Image display
        upper_widget = QWidget()
        upper_layout = QHBoxLayout(upper_widget)

        # Right Panel (Layer List)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        layer_list_title = QLabel("Image file / Layer List")
        layer_list_title.setAlignment(Qt.AlignCenter)
        layer_list_title.setStyleSheet("""
        QLabel {
            font-weight: bold;
            font-size: 14px;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 4px;
        }
        """)
        right_layout.addWidget(layer_list_title)

        # Layer List
        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self.switch_layer)
        self.layer_list.itemClicked.connect(self.show_stats)
        self.layer_list.setStyleSheet("""
        QListWidget {border: 1px solid #ddd; border-radius: 4px; padding: 5px; background-color: #f9f9f9; font-size: 14px;}
        """)
        right_layout.addWidget(self.layer_list)

        # # Add Layer Button
        # add_layer_button = QPushButton("Add Layer")
        # add_layer_button.setIcon(QIcon(":/icons/add.png"))
        # add_layer_button.clicked.connect(self.add_layer)
        # right_layout.addWidget(add_layer_button)

        # Delete Layer Button
        delete_layer_button = QPushButton("Delete Layer")
        delete_layer_button.setIcon(QIcon(":/icons/delete.png"))
        delete_layer_button.clicked.connect(self.delete_layer)
        right_layout.addWidget(delete_layer_button)

        # Progress & Stats
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.stats_label = QLabel()
        self.stats_label.setVisible(False)

        # Assemble Right Panel
        right_layout.addWidget(layer_list_title)
        right_layout.addWidget(self.layer_list)
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.stats_label)

        # Setup stacked widget
        self.stacked_widget = QStackedWidget()

        # Image display widget
        self.image_display_widget = QWidget()
        image_layout = QVBoxLayout(self.image_display_widget)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        image_layout.addWidget(self.canvas)
        image_layout.addWidget(self.nav_toolbar)
        self.stacked_widget.addWidget(self.image_display_widget)
        self.setup_mouse_selection()  # Connect mouse events

        # Google Maps widget
        if QWebEngineView is not None:
            self.google_maps_view = QWebEngineView()
            try:
                self.google_maps_view.setPage(_LoggingWebPage(self.google_maps_view))
            except Exception:
                pass
            try:  # the local page loads Leaflet and the basemap tiles from the internet
                self.google_maps_view.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            except Exception:
                pass
        else:  # PyQtWebEngine not installed: the interactive web-map view is disabled
            self.google_maps_view = QLabel("Web map view needs PyQtWebEngine (pip install PyQtWebEngine).")
            self.google_maps_view.setAlignment(Qt.AlignCenter)
        self.stacked_widget.addWidget(self.google_maps_view)

        # Add components to upper layout
        upper_layout.addWidget(right_panel, 15)  # 25% width for layer list
        upper_layout.addWidget(self.stacked_widget, 85)  # 75% width for main display

        # Lower part: Info box
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setStyleSheet("""
        QTextEdit {
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
            font-family: Consolas;
            font-size: 12px;
            padding: 10px;
        }
        """)

        # Configure Splitter
        splitter.addWidget(upper_widget)  # Contains both layer list and image
        splitter.addWidget(self.info_box)  # Info panel at bottom
        splitter.setSizes([850, 150])  # Relative sizes
        self.main_layout.addWidget(splitter)

        # Context menu setup
        self.layer_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.layer_list.customContextMenuRequested.connect(self.show_context_menu)

        # Setup file menu actions
        # open_action = QAction('Open', self)
        # open_action.setShortcut('Ctrl+O')
        # open_action.setToolTip('Open image')
        # open_action.triggered.connect(self.load_image)
        # file_menu.addAction(open_action)


    # Existing raster load
        load_raster_action = QAction('Load Raster (TIFF)', self)
        load_raster_action.triggered.connect(self.load_image)  # Your existing function
        file_menu.addAction(load_raster_action)
        
        # New vector load
        load_vector_action = QAction('Load Vector (Shapefile/GeoJSON)', self)
        load_vector_action.triggered.connect(self.load_vector)
        file_menu.addAction(load_vector_action)


        # export_action = QAction('Save Image as', self)
        # export_action.setShortcut('Ctrl+E')
        # export_action.triggered.connect(self.save_as_image)
        # file_menu.addAction(export_action)

        export_map_action = QAction('Export Map (PDF/PNG/JPEG/TIFF/SVG)...', self)
        export_map_action.setShortcut('Ctrl+Shift+E')
        export_map_action.triggered.connect(self.export_map_dialog)
        file_menu.addAction(export_map_action)

        stack_action = QAction('Stack Sentinel-2 Bands (one file per date)...', self)
        stack_action.triggered.connect(self.stack_bands_dialog)
        file_menu.addAction(stack_action)

        print_action = QAction('Print', self)
        print_action.triggered.connect(self.print_image)
        file_menu.addAction(print_action)

        page_setup_action = QAction('Page Setup', self)
        page_setup_action.triggered.connect(self.page_setup)
        file_menu.addAction(page_setup_action)

        # batch_export_action = QAction('Batch Export', self)
        # batch_export_action.triggered.connect(self.batch_export)
        # file_menu.addAction(batch_export_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Setup edit menu actions
        undo_action = QAction('Undo', self)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction('Redo', self)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        # Setup raster processing tools menu
        # image_enhancements_menu = QMenu('Image Enhancements', self)
        # satellite_menu.addMenu(image_enhancements_menu)

        
        # filter_action =QAction('Image Filters', self) 
        # filter_action.triggered.connect(self.apply_filter)
        # image_enhancements_menu.addAction(filter_action)
        # Add Filters submenu under Image Enhancements
        image_enhancements_menu = satellite_menu.addMenu('Image Enhancements')
        
        # Create filter actions
        filter_actions = [
            ('Remove NoData Values', self.apply_nodata_filter),
            ('Remove Outliers', self.apply_outlier_filter),
            ('Median Filter', self.apply_median_filter),
            ('Gaussian Filter', self.apply_gaussian_filter),
            ('Lee Filter (Speckle Reduction)', self.apply_lee_filter),
            ('Normalize Image', self.apply_normalization),
            ('Convert to 8-bit', self.convert_to_8bit)
        ]
        
        for name, callback in filter_actions:
            action = QAction(name, self)
            action.triggered.connect(callback)
            image_enhancements_menu.addAction(action)


        resample_action = QAction('Resample Image', self)
        resample_action.triggered.connect(self.resample_image)
        image_enhancements_menu.addAction(resample_action)

        enhance_contrast_action = QAction('Enhance Contrast', self)
        enhance_contrast_action.triggered.connect(self.enhance_contrast)
        image_enhancements_menu.addAction(enhance_contrast_action)

        smooth_image_action = QAction('Smooth Image', self)
        smooth_image_action.triggered.connect(self.smooth_image)
        image_enhancements_menu.addAction(smooth_image_action)

        sharpen_image_action = QAction('Sharpen Image', self)
        sharpen_image_action.triggered.connect(self.sharpen_image)
        image_enhancements_menu.addAction(sharpen_image_action)

        # DEM Submenu
        dem_menu = satellite_menu.addMenu('DEM Processing Tools')

        hillshade_action = QAction('Calculate Hillshade', self)
        hillshade_action.triggered.connect(self.calculate_hillshade)
        dem_menu.addAction(hillshade_action)

        slope_action = QAction('Calculate Slope', self)
        slope_action.triggered.connect(self.calculate_slope)
        dem_menu.addAction(slope_action)

        aspect_action = QAction('Calculate Aspect', self)
        aspect_action.triggered.connect(self.calculate_aspect)
        dem_menu.addAction(aspect_action)


        flow_accumulation_action = QAction('Flow Accumulation', self)
        flow_accumulation_action.triggered.connect(self.calculate_flow_accumulation)
        dem_menu.addAction(flow_accumulation_action)

        stream_network_action = QAction('Calculate Stream Network', self)
        stream_network_action.triggered.connect(self.calculate_stream_network)
        dem_menu.addAction(stream_network_action)

        dem_menu.addSeparator()
        watershed_action = QAction('Watershed + Basin Morphometrics (click pour point)', self)
        watershed_action.triggered.connect(self.delineate_watershed_tool)
        dem_menu.addAction(watershed_action)

        hand_action = QAction('HAND (Height Above Nearest Drainage)', self)
        hand_action.triggered.connect(self.calculate_hand)
        dem_menu.addAction(hand_action)

        twi_action = QAction('TWI and SPI (wetness / stream power)', self)
        twi_action.triggered.connect(self.calculate_twi_spi)
        dem_menu.addAction(twi_action)

        # DrainageChannels_action = QAction('Generate Drainage Channels', self)
        # DrainageChannels_action.triggered.connect(self.generate_drainage_channels)
        # dem_menu.addAction(DrainageChannels_action)

        # terrain_menu.addAction("Flow Accumulation", self.calculate_flow_accumulation)
        # terrain_menu.addAction("TRI", self.calculate_tri)
        # terrain_menu.addAction("TPI", self.calculate_tpi)
        # terrain_menu.addAction("Curvature", self.calculate_curvature)
        # terrain_menu.addAction("Flow Accumulation", self.calculate_flow_accumulation)
        # terrain_menu.addAction("Viewshed", self.calculate_viewshed)
    


        raster_clipping_action = QAction('Raster Clipping', self)
        raster_clipping_action.triggered.connect(self.raster_clipping_process)
        satellite_menu.addAction(raster_clipping_action)

        raster_calc_action = QAction('Raster Calculator', self)
        raster_calc_action.triggered.connect(self.raster_calculator)
        satellite_menu.addAction(raster_calc_action)

        raster_calc_action = QAction('Project Raster', self)
        raster_calc_action.triggered.connect(self.project_raster)
        satellite_menu.addAction(raster_calc_action)


        # Setup edge detection tools menu actions
        canny_action = QAction('Canny Edge Detection', self)
        canny_action.triggered.connect(self.canny_edge_detection)
        edge_detection_menu.addAction(canny_action)

        sobel_action = QAction('Sobel Edge Detection', self)
        sobel_action.triggered.connect(self.sobel_edge_detection)
        edge_detection_menu.addAction(sobel_action)

        prewitt_action = QAction('Prewitt Edge Detection', self)
        prewitt_action.triggered.connect(self.prewitt_edge_detection)
        edge_detection_menu.addAction(prewitt_action)

        LaplacianOfGaussian_action = QAction('Laplacian of Gaussian (LoG) Edge Detection', self)
        LaplacianOfGaussian_action.triggered.connect(self.log_edge_detection)
        edge_detection_menu.addAction(LaplacianOfGaussian_action)

        save_edges_action = QAction('Save Edges as Shapefile', self)
        save_edges_action.triggered.connect(self.save_edges_as_shapefile)
        edge_detection_menu.addAction(save_edges_action)

        # Setup image classification tools menu actions
        kmeans_action = QAction('Classify with K-Means', self)
        kmeans_action.triggered.connect(self.classify_with_kmeans)
        image_classification_menu.addAction(kmeans_action)

        supervised_action = QAction('Supervised Classification (RF / SVM)...', self)
        supervised_action.triggered.connect(self.supervised_classification)
        image_classification_menu.addAction(supervised_action)

        mean_shift_action = QAction('Mean-Shift Segmentation', self)
        mean_shift_action.triggered.connect(self.mean_shift_segmentation)
        image_classification_menu.addAction(mean_shift_action)

        sam_action = QAction('Segment Anything (SAM)', self)
        sam_action.triggered.connect(self.sam_segmentation)
        image_classification_menu.addAction(sam_action)

        # Merge Action 
        merge_action = QAction('Merge Polygons', self)        
        merge_action.triggered.connect(self.polygon_processor.merge_polygons)
        basic_polygon_menu.addAction(merge_action)        
        
        # Dissolve Action (now fixed)
        dissolve_action = QAction('Dissolve Polygons', self)
        dissolve_action.triggered.connect(self.polygon_processor.dissolve_polygons)  # Fixed connection
        basic_polygon_menu.addAction(dissolve_action)

        clip_action = QAction('Clip Polygons', self)
        clip_action.triggered.connect(self.polygon_processor.clip_polygons)
        basic_polygon_menu.addAction(clip_action)

        intersect_action = QAction('Intersect Polygons', self)
        intersect_action.triggered.connect(self.polygon_processor.intersect_polygons)
        basic_polygon_menu.addAction(intersect_action)

        union_action = QAction('Union Polygons', self)
        union_action.triggered.connect(self.polygon_processor.union_polygons)
        basic_polygon_menu.addAction(union_action)

        # Advanced Polygon Operations
        advanced_polygon_menu = polygon_menu.addMenu('Advanced Operations')
        buffer_action = QAction('Create Buffers', self)
        buffer_action.triggered.connect(self.polygon_processor.create_buffers)
        advanced_polygon_menu.addAction(buffer_action)

        simplify_action = QAction('Simplify Polygons', self)
        simplify_action.triggered.connect(self.polygon_processor.simplify_polygons)
        advanced_polygon_menu.addAction(simplify_action)

        convex_hull_action = QAction('Create Convex Hull', self)
        convex_hull_action.triggered.connect(self.polygon_processor.create_convex_hull)
        advanced_polygon_menu.addAction(convex_hull_action)

        voronoi_action = QAction('Create Voronoi Polygons', self)
        voronoi_action.triggered.connect(self.polygon_processor.create_voronoi_polygons)
        advanced_polygon_menu.addAction(voronoi_action)

        # Geometry Properties Menu (NEW)
        geometry_menu = polygon_menu.addMenu('Geometry Properties')
        add_geom_fields_action = QAction('Add Geometry Fields', self)
        add_geom_fields_action.triggered.connect(self.polygon_processor.add_geometry_fields)
        geometry_menu.addAction(add_geom_fields_action)

        calculate_geom_action = QAction('Calculate Geometry Properties', self)
        calculate_geom_action.triggered.connect(self.polygon_processor.calculate_geometry_properties)
        geometry_menu.addAction(calculate_geom_action)

        # Polygon Analysis
        analysis_menu = polygon_menu.addMenu('Analysis')
        centroid_action = QAction('Calculate Centroids', self)
        centroid_action.triggered.connect(self.polygon_processor.calculate_centroids)
        analysis_menu.addAction(centroid_action)

        area_action = QAction('Calculate Areas', self)
        area_action.triggered.connect(self.polygon_processor.calculate_areas)
        analysis_menu.addAction(area_action)

        distance_action = QAction('Calculate Distances', self)
        distance_action.triggered.connect(self.polygon_processor.calculate_distances)
        analysis_menu.addAction(distance_action)

        nearest_neighbor_action = QAction('Nearest Neighbor Analysis', self)
        nearest_neighbor_action.triggered.connect(self.polygon_processor.nearest_neighbor_analysis)
        analysis_menu.addAction(nearest_neighbor_action)

        # Polygon Editing
        editing_menu = polygon_menu.addMenu('Editing')
        split_action = QAction('Split Polygons', self)
        split_action.triggered.connect(self.polygon_processor.split_polygons)
        editing_menu.addAction(split_action)



        delete_part_action = QAction('Delete Polygon Parts', self)
        delete_part_action.triggered.connect(self.polygon_processor.delete_polygon_parts)
        editing_menu.addAction(delete_part_action)
        
        
        # Add status bar
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        
        overlay_action = QAction('Overlay', self)
        overlay_action.triggered.connect(self.show_overlay_dialog)
        image_classification_menu.addAction(overlay_action)

        # toggle_order_button = QPushButton("Toggle Layer Order")
        # toggle_order_button.clicked.connect(self.toggle_layer_order)
        # self.toolbar.addWidget(toggle_order_button)

        analyze_land_use_action = QAction('Analyze Land Use', self)
        analyze_land_use_action.triggered.connect(self.analyze_land_use)
        image_classification_menu.addAction(analyze_land_use_action)

        save_classified_action = QAction('Save Classified Image', self)
        save_classified_action.triggered.connect(self.save_classified_image)
        image_classification_menu.addAction(save_classified_action)

        # # Setup prepare map menu actions
        # insert_map_title_action = QAction('Insert Map Title', self)
        # insert_map_title_action.triggered.connect(self.insert_map_title)
        # prepare_map_menu.addAction(insert_map_title_action)

        # insert_text_action = QAction('Insert Text', self)
        # insert_text_action.triggered.connect(self.insert_text)
        # prepare_map_menu.addAction(insert_text_action)

        # insert_legend_action = QAction('Insert Current Legend', self)
        # insert_legend_action.triggered.connect(self.insert_legend)
        # prepare_map_menu.addAction(insert_legend_action)

        # insert_north_arrow_action = QAction('Insert North Arrow', self)
        # insert_north_arrow_action.triggered.connect(self.insert_north_arrow)
        # prepare_map_menu.addAction(insert_north_arrow_action)

        # insert_grid_action = QAction('Insert Grid', self)
        # insert_grid_action.triggered.connect(self.insert_grid)
        # prepare_map_menu.addAction(insert_grid_action)

        # insert_scale_bar_action = QAction('Insert Scale bar', self)
        # insert_scale_bar_action.triggered.connect(self.insert_scale_bar)
        # prepare_map_menu.addAction(insert_scale_bar_action)

        # export_full_action = QAction('Export Map Layout', self)
        # export_full_action.triggered.connect(self.save_combined_layout)
        # prepare_map_menu.addAction(export_full_action)

        # # Add Settings menu
        # settings_menu = menubar.addMenu('Settings')

        # general_settings_action = QAction('General Settings', self)
        # general_settings_action.triggered.connect(self.open_general_settings)
        # settings_menu.addAction(general_settings_action)

                
        # display_settings_action = QAction('Display Settings', self)
        # display_settings_action.triggered.connect(self.open_display_settings)
        # settings_menu.addAction(display_settings_action)

        # processing_settings_action = QAction('Processing Settings', self)
        # processing_settings_action.triggered.connect(self.open_processing_settings)
        # settings_menu.addAction(processing_settings_action)

        # advanced_settings_action = QAction('Advanced Settings', self)
        # advanced_settings_action.triggered.connect(self.open_advanced_settings)
        # settings_menu.addAction(advanced_settings_action)

        # about_action = QAction('About', self)
        # about_action.triggered.connect(self.open_about)
        # settings_menu.addAction(about_action)

        # Setup help menu actions
        help_action = QAction('Help', self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        # Setup spectral indices menu actions
        ndvi_action = QAction('Calculate NDVI', self, text="Normalized difference vegetation index NDVI")
        ndvi_action.triggered.connect(self.load_and_calculate_ndvi)
        indices_menu.addAction(ndvi_action)
        
        ndwi_action = QAction('Calculate NDWI', self, text="Normalized difference water index NDWI")
        ndwi_action.triggered.connect(self.load_and_calculate_ndwi)
        indices_menu.addAction(ndwi_action)

        ndni_action = QAction('Calculate NDNI', self, text="Normalized difference nitrogen index NDNI")
        ndni_action.triggered.connect(self.load_and_calculate_ndni)
        indices_menu.addAction(ndni_action)

        avi_action = QAction('Calculate AVI', self, text="Advanced vegetation index AVI")
        avi_action.triggered.connect(self.load_and_calculate_avi)
        indices_menu.addAction(avi_action)

        ndwi_action = QAction('Calculate NDWI', self, text="Normalized difference water index NDWI")
        ndwi_action.triggered.connect(self.load_and_calculate_ndwi)
        indices_menu.addAction(ndwi_action)

        ndmi_action = QAction('Calculate NDMI', self, text="Normalized difference moisture index NDMI")
        ndmi_action.triggered.connect(self.load_and_calculate_ndmi)
        indices_menu.addAction(ndmi_action)

        save_action = QAction('Save Current Index', self, text="Save Current Index")
        save_action.triggered.connect(self.save_current_index)
        indices_menu.addAction(save_action)

        # إضافة إجراءات قائمة اكتشاف التغيرات
        detect_action = QAction('Detect Changes', self)
        detect_action.triggered.connect(self.detect_changes)
        change_detection_menu.addAction(detect_action)

        save_changes_action = QAction('Save Changes Result', self)
        save_changes_action.triggered.connect(self.save_change_results)
        change_detection_menu.addAction(save_changes_action)
        change_detection_menu.addSeparator()
        sens_action = QAction('Sensitivity Analysis (threshold k, normalisation)...', self)
        sens_action.triggered.connect(self.change_sensitivity)
        change_detection_menu.addAction(sens_action)
        fa_action = QAction('False-Alarm Test (two images without real change)...', self)
        fa_action.triggered.connect(lambda: self.detect_changes(control=True))
        change_detection_menu.addAction(fa_action)

                # تفعيل القائمة السياقية
        self.info_box.setContextMenuPolicy(Qt.CustomContextMenu)
        self.info_box.customContextMenuRequested.connect(self.show_context_menu)

        # Add a toggle action for Google Maps
        toggle_google_maps_action = QAction("Toggle Google Maps", self)
        toggle_google_maps_action.triggered.connect(self.toggle_google_maps)

        # Add the action to a menu or toolbar
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(toggle_google_maps_action)

        self.setWindowTitle('Space GeoVision Software v1.0')
        self.setWindowIcon(QIcon(resource_path("Asset 4@2x.png")))
        self.setIconSize(QSize(30, 60))
        self.setGeometry(200, 200, 1200, 800)
        #self.setGeometry(0, 0, 1920, 1080)
        
    ##########   setStyleSheet     ###################
    def load_styles(self):
        try:
            with open('styles.qss', 'r', encoding='utf-8') as f:
                style = f.read()
                self.setStyleSheet(style)
        except Exception as e:
            logging.info(f"Error loading stylesheet: {str(e)}")
##########################################  The End of def initUI

    # Add these methods to your class:
    
    def show_warning(self, message):
        """Show warning message"""
        QMessageBox.warning(self, "Warning", message)
        self.statusBar().showMessage(f"Warning: {message}", 3000)
    
    def show_info(self, message):
        """Show informational message"""
        QMessageBox.information(self, "Information", message)
        self.statusBar().showMessage(message, 2000)

    def _setup_selection_tools(self):
        """Initialize all selection tools and their connections"""
        # Create selection toolbar
        self.selection_toolbar = QToolBar("Selection Tools")
        #self.addToolBar(Qt.LeftToolBarArea, self.selection_toolbar)
        self.addToolBar(Qt.TopToolBarArea, self.selection_toolbar)  # Explicit area
        
        # Set icon size (adjust as needed)
        self.selection_toolbar.setIconSize(QSize(20, 20))
        
        # # Prevent floating/moving (optional)
        # self.selection_toolbar.setFloatable(False)
        # self.selection_toolbar.setMovable(False)
        
        # Style buttons (icons with text below)
        self.selection_toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        
        # Selection tools
        self.rect_select_action = QAction(QIcon(resource_path("select_rect.png")), "Rectangle Select", self)
        self.poly_select_action = QAction(QIcon(resource_path("select_poly.png")), "Polygon Select", self)
        self.clear_select_action = QAction(QIcon(resource_path("clear_select.png")), "Clear Selection", self)
        
        # Connect signals
        self.rect_select_action.triggered.connect(self._activate_rectangle_selection)
        self.poly_select_action.triggered.connect(self._activate_polygon_selection)
        self.clear_select_action.triggered.connect(self._clear_selection)
        
        # Add to toolbar
        self.selection_toolbar.addAction(self.rect_select_action)
        self.selection_toolbar.addAction(self.poly_select_action)
        self.selection_toolbar.addAction(self.clear_select_action)
        
        # Initialize selection state
        self.selection_mode = None
        self.selection_start = None
        self.selection_points = []
        self.temp_shape = None

    


    def _activate_rectangle_selection(self):
        """Activate rectangle selection mode"""
        self.selection_mode = 'rectangle'
        self.canvas.setCursor(Qt.CrossCursor)
        self.statusBar().showMessage("Rectangle selection: Click and drag to select features")
        
        # Clear any previous selection
        self._clear_selection(update_display=False)
        
        # Show instructions
        QMessageBox.information(self, "Rectangle Selection",
            "Rectangle Selection Mode\n\n"
            "1. Click and hold left mouse button to start selection\n"
            "2. Drag to define selection area\n"
            "3. Release to finalize selection\n"
            "4. Right-click to cancel")

    def _activate_polygon_selection(self):
        """Activate polygon selection mode"""
        self.selection_mode = 'polygon'
        self.canvas.setCursor(Qt.CrossCursor)
        self.statusBar().showMessage("Polygon selection: Click to add points, right-click to finish")
        
        # Clear any previous selection
        self._clear_selection(update_display=False)
        self.selection_points = []
        
        # Show instructions
        QMessageBox.information(self, "Polygon Selection",
            "Polygon Selection Mode\n\n"
            "1. Click to add polygon vertices\n"
            "2. Continue clicking to add more points\n"
            "3. Right-click to complete the selection\n"
            "4. Press ESC to cancel")

    def _clear_selection(self, update_display=True):
        """Clear current selection"""
        if hasattr(self.polygon_processor, 'selected_features'):
            self.polygon_processor.selected_features.clear()
        
        self.selection_mode = None
        self.selection_points = []
        self.temp_shape = None
        
        if update_display:
            self.update_map_display()
        self.statusBar().showMessage("Selection cleared")

    def activate_line_drawing_mode(self, callback):
        """Activate line drawing on the canvas"""
        self.canvas.set_drawing_mode('line', callback)
        
        
    def update_map_display(self):
        """
        Update the map display with:
        - Base layer
        - Selected features (highlighted)
        - Temporary selection/editing shapes
        - Visual feedback for active tools
        """
        try:
            # Clear the current display but maintain settings
            self.ax.clear()
            
            # Get the current active layer
            layer_name = None
            if hasattr(self.polygon_processor, 'get_selected_vector_layer'):
                try:
                    layer_name = self.polygon_processor.get_selected_vector_layer()
                except Exception as e:
                    logging.error(f"Error getting selected layer: {str(e)}")
                    return
            
            # Draw the base layer (if any)
            if layer_name and layer_name in self.layers:
                layer_data = self.layers[layer_name]
                
                # Handle vector layers
                if layer_data.get('type') == 'vector' and 'gdf' in layer_data:
                    gdf = layer_data['gdf']
                    
                    # Draw unselected features
                    if hasattr(self.polygon_processor, 'selected_features'):
                        unselected = gdf[~gdf.index.isin(self.polygon_processor.selected_features)]
                        if not unselected.empty:
                            unselected.plot(
                                ax=self.ax,
                                color='blue',
                                edgecolor='black',
                                alpha=0.5,
                                linewidth=0.5
                            )
                    
                    # Draw selected features with highlight
                    if hasattr(self.polygon_processor, 'selected_features') and self.polygon_processor.selected_features:
                        selected = gdf[gdf.index.isin(self.polygon_processor.selected_features)]
                        if not selected.empty:
                            selected.plot(
                                ax=self.ax,
                                color='yellow',
                                edgecolor='red',
                                alpha=0.7,
                                linewidth=2
                            )
                
                # Handle raster layers
                elif 'image' in layer_data:
                    # Drawn in map coordinates so that vector overlays (also in map coordinates) line up with it
                    lp = layer_data.get('profile') or {}
                    arr = np.asarray(layer_data['image'])
                    if arr.ndim == 2:  # Single band
                        _, spec = georef.draw_raster(self.ax, arr, lp, cmap='gray')
                    elif arr.shape[0] in [3, 4]:  # RGB(A): channels first -> HxWxC
                        _, spec = georef.draw_raster(self.ax, np.transpose(arr, (1, 2, 0)), lp)
                    elif arr.shape[0] == 1:
                        _, spec = georef.draw_raster(self.ax, arr[0], lp, cmap='gray')
                    else:  # more than 3 bands: show the first three
                        _, spec = georef.draw_raster(self.ax, np.transpose(arr[:3], (1, 2, 0)), lp)
            
            # Draw temporary shapes (selection/editing)
            if hasattr(self, 'temp_shape') and self.temp_shape:
                if isinstance(self.temp_shape, (Polygon, MultiPolygon)):
                    xs, ys = self.temp_shape.exterior.xy
                    self.ax.plot(xs, ys, color='green', linewidth=2, linestyle='--')
                elif isinstance(self.temp_shape, LineString):
                    xs, ys = self.temp_shape.xy
                    self.ax.plot(xs, ys, color='green', linewidth=2, linestyle='--')
                elif isinstance(self.temp_shape, QRectF):
                    rect = self.temp_shape
                    self.ax.plot(
                        [rect.left(), rect.right(), rect.right(), rect.left(), rect.left()],
                        [rect.top(), rect.top(), rect.bottom(), rect.bottom(), rect.top()],
                        color='green',
                        linewidth=2,
                        linestyle='--'
                    )
            
            # Draw in-progress polygon selection
            if hasattr(self, 'selection_points') and self.selection_points:
                xs, ys = zip(*self.selection_points)
                self.ax.plot(xs, ys, 'go-', alpha=0.5)  # Green circles with lines
                
                # Close the polygon if enough points
                if len(self.selection_points) > 2:
                    self.ax.plot(
                        [self.selection_points[-1][0], self.selection_points[0][0]],
                        [self.selection_points[-1][1], self.selection_points[0][1]],
                        'g--'
                    )
            
            # Set display properties
            self.ax.set_aspect('equal', adjustable='datalim')
            self.ax.grid(True, linestyle=':', alpha=0.5)
            
            # Redraw the canvas
            self.canvas.draw()            
        except Exception as e:
            logging.error(f"Error updating map display: {str(e)}")
            QMessageBox.critical(self, "Display Error", 
                "Failed to update map display. See logs for details.")


    def mousePressEvent(self, event):
        if self.selection_mode == 'polygon' and event.button() == Qt.LeftButton:
            # Convert mouse position to scene coordinates
            point = self.mapToScene(event.pos())
            self.polygon_points.append(QPointF(point))
            
            # Update visualization
            self.update_polygon_preview()
            
        elif event.button() == Qt.RightButton and self.polygon_points:
            # Finish polygon on right-click
            self.finalize_polygon()


    def mouseReleaseEvent(self, event):
        """Finalize selection on release"""
        if (self.selection_mode == 'rectangle' and 
            event.button() == Qt.LeftButton and 
            hasattr(self, 'selection_start')):
            
            # Finalize rectangle selection
            point = self.mapToScene(event.pos())
            x, y = point.x(), point.y()
            rect = QRectF(QPointF(*self.selection_start), QPointF(x, y)).normalized()
            
            # Convert to Shapely polygon
            selection_poly = Polygon([
                (rect.left(), rect.top()),
                (rect.right(), rect.top()),
                (rect.right(), rect.bottom()),
                (rect.left(), rect.bottom())
            ])
            
            self._select_features_in_polygon(selection_poly)
            del self.selection_start
        
        self.update_map_display()

    def _select_features_in_polygon(self, polygon):
        """Select features intersecting with the given polygon"""
        layer_name = self.polygon_processor.get_selected_vector_layer()
        if not layer_name or layer_name not in self.layers:
            return
            
        gdf = self.layers[layer_name]['gdf']
        selected = gdf[gdf.geometry.intersects(polygon)]
        
        if not selected.empty:
            self.polygon_processor.set_selected_features(selected.index.tolist())
            QMessageBox.information(self, "Selection", 
                f"Selected {len(selected)} features")
        else:
            QMessageBox.warning(self, "Selection", "No features found in selection area")
    def update_polygon_preview(self):
        """Update the visual preview of the polygon being drawn"""
        if not self.polygon_points:
            return
        
        # Create a QPolygonF from collected points
        poly = QPolygonF(self.polygon_points)
        
        # Add temporary last segment if moving mouse
        if hasattr(self, 'temp_end_point'):
            poly.append(self.temp_end_point)
        
        self.temp_polygon = poly
        self.viewport().update()  # Trigger repaint 

    def paintEvent(self, event):
        """Draw the polygon preview"""
        super().paintEvent(event)
        
        if self.temp_polygon:
            painter = QPainter(self.viewport())
            painter.setPen(QPen(QColor(0, 255, 0), 2))  # Green, 2px width
            painter.drawPolygon(self.temp_polygon)
            
            # Draw points
            painter.setPen(QPen(QColor(255, 0, 0), 5))  # Red points
            for point in self.polygon_points:
                painter.drawPoint(point) 

    def mouseMoveEvent(self, event):
        if self.selection_mode == 'polygon' and self.polygon_points:
            # Update temporary last segment while moving mouse
            self.temp_end_point = self.mapToScene(event.pos())
            self.update_polygon_preview()

    def finalize_polygon(self):
        """Complete the polygon and add to shapefile"""
        if len(self.polygon_points) < 3:
            QMessageBox.warning(self, "Invalid Polygon", 
                            "A polygon needs at least 3 points")
            return
        
        try:
            # Convert to Shapely polygon
            from shapely.geometry import Polygon
            poly = Polygon([[p.x(), p.y()] for p in self.polygon_points])
            
            active = getattr(self, 'active_layer', None)
            if active is not None and hasattr(active, 'add_feature'):
                active.add_feature(poly)
                logging.info("Added polygon to active layer")

            # Reset drawing state
            self.polygon_points = []
            self.temp_polygon = None
            self.update()
            
        except Exception as e:
            logging.error("Error finalizing polygon: %s", e)
            QMessageBox.critical(self, "Error", 
                            f"Could not create polygon: {str(e)}")


    
    def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape and self.polygon_points:
                self.polygon_points = []
                self.temp_polygon = None
                self.update() 
    
    def setup_mouse_selection(self):
        """Connect mouse events for image selection"""
        self.canvas.mpl_connect('button_press_event', self.on_image_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    # def on_image_click(self, event):
    #     """Handle mouse clicks on the image display"""
    #     if event.inaxes != self.ax:
    #         return

    #     # Store the clicked image as active image
    #     for artist in self.ax.images:
    #         if artist.contains(event)[0]:
    #             self.active_image = artist.get_array()
    #             if hasattr(artist, 'profile'):
    #                 self.profile = artist.profile
    #             break
    def on_image_click(self, event):
        """Handle mouse clicks on the image display"""
        if event.inaxes != self.ax or not self.layers:
            return

        try:
            # Check if we clicked on an image artist
            for artist in self.ax.images:
                if artist.contains(event)[0]:
                    # Find which layer this image belongs to
                    for layer_name, layer_data in self.layers.items():
                        if hasattr(artist, '_layer_name') and artist._layer_name == layer_name:
                            self.current_layer = layer_name
                            self.active_image = layer_data['image']
                            self.profile = layer_data.get('profile', {})
                            self.display_raster_layer(layer_data, layer_name)
                            self.update_info_box(layer_data)
                            break
                    break
        except Exception as e:
            logging.info(f"Error handling image click: {e}")
    def delete_layer(self):
        """
        Delete the selected layer from the list along with its image and legend
        """
        selected_item = self.layer_list.currentItem()
        if selected_item:
            self.delete_layer_from_list(selected_item)


    def delete_layer_from_list(self, item):
        layer_name = item.text()
        if layer_name in self.layers:
            # التحقق مما إذا كانت الطبقة المحذوفة هي الطبقة الحالية
            if self.current_layer == layer_name:
                self.current_layer = None  # إعادة تعيين الطبقة الحالية
                self.clear_display()  # مسح منطقة العرض

            # حذف الطبقة من القائمة
            del self.layers[layer_name]
            self.layer_list.takeItem(self.layer_list.row(item))
            QMessageBox.information(self, "Success", f"Layer '{layer_name}' deleted.")

    def clear_display(self):
        """
        مسح منطقة عرض الصورة.
        """
        self.ax.clear()  # مسح المحتوى من الرسم البياني
        self.canvas.draw()  # إعادة رسم القماش
        logging.info("Display cleared.")
    def _decorate_map_axes(self, spec):
        """Axis labels/ticks in real coordinates (degrees for geographic CRS, metres otherwise)."""
        georef.decorate_axes(self.ax, spec)

    def _inherited_profile(self, shape):
        """Grid (transform + CRS) for a result array that has no layer of its own: that of the layer it was computed
        from (rescaled when the size differs); ``{}`` - pixel axes - when nothing georeferenced is known."""
        cur = self.layers.get(getattr(self, 'current_layer', None))
        sources = []
        if isinstance(cur, dict):
            sources.append((cur.get('profile'), cur.get('image')))
        sources.append((getattr(self, 'profile', None), getattr(self, 'image', None)))
        return georef.inherit_profile(sources, shape)

    def _on_mouse_move(self, event):
        """Status-bar read-out: map coordinates, pixel (row, col) and value under the cursor."""
        try:
            if event.inaxes is not self.ax or event.xdata is None:
                self.statusBar().clearMessage()
                return
            layer = self.layers.get(getattr(self, 'current_layer', None))
            if not isinstance(layer, dict) or layer.get('image') is None:
                return
            img = np.asarray(layer['image'])
            prof = layer.get('profile') or {}
            tr = prof.get('transform')
            h, w = img.shape[-2:]
            spec = georef.axis_spec(tr, prof.get('crs'), h, w)
            if spec.extent is None:
                col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
                where = f"col {col}, row {row}"
            else:
                rc = georef.map_to_pixel(tr, event.xdata, event.ydata)
                if rc is None:
                    return
                row, col = rc
                if spec.geographic:
                    where = (f"{georef.format_degrees(event.xdata, False, 5)}  "
                             f"{georef.format_degrees(event.ydata, True, 5)}  |  row {row}, col {col}")
                else:
                    where = f"E {event.xdata:,.1f}  N {event.ydata:,.1f}  |  row {row}, col {col}"
            if 0 <= row < h and 0 <= col < w:
                vals = img[..., row, col] if img.ndim == 3 else img[row, col]
                vals = np.atleast_1d(vals)
                txt = ", ".join("nodata" if not np.isfinite(v) else f"{float(v):.4g}" for v in vals[:4])
                self.statusBar().showMessage(f"{where}  |  value: {txt}")
            else:
                self.statusBar().showMessage(where)
        except Exception:
            pass  # a cursor read-out must never raise inside a Qt slot

    def _redraw_classified_layer(self, layer_data, layer_name):
        """Re-draw a categorical layer with its legend. Returns True if the layer was classified (and drawn)."""
        ld = layer_data
        if isinstance(ld, np.ndarray):
            ld = self.layers.get(layer_name)
        if not isinstance(ld, dict) or not ld.get('classified') or ld.get('image') is None:
            return False
        arr = np.asarray(ld['image'])
        if arr.ndim == 3:
            arr = arr[0]
        names = ld.get('class_names')
        if not names:
            vals = [int(v) for v in np.unique(arr) if int(v) != 0]
            names = {v: f"Class {v}" for v in vals}
        profile = ld.get('profile') or None
        self.display_classified_image(arr, profile=profile, names=names, title=ld.get('title') or layer_name)
        return True

    def display_raster_layer(self, layer_data, layer_name, show_grid=True):
        """Display raster layer with proper handling of float64 and NaN values"""
        try:
            # Classified results (supervised, K-Means, Mean-Shift ...) always come back exactly as first shown:
            # class colours, legend, title and real map coordinates -- never a continuous colour stretch.
            if self._redraw_classified_layer(layer_data, layer_name):
                return

            # Clear previous plot completely
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_axis_on()
            
            # Ensure layer_data is properly structured
            if isinstance(layer_data, np.ndarray):
                existing = self.layers.get(layer_name)
                if isinstance(existing, dict):
                    # Keep the stored layer (profile/georeferencing, colormap, ...); only its pixels are refreshed.
                    # Never replace it by a profile-less dict: that loses the geotransform.
                    stored = existing.get('image')
                    if stored is None or getattr(stored, 'shape', None) != layer_data.shape:
                        if stored is not None and np.size(stored) == layer_data.size:
                            layer_data = np.reshape(layer_data, stored.shape)
                        existing.pop('display_range', None)
                        existing.pop('original_dims', None)
                    existing['image'] = layer_data
                    layer_data = existing
                else:
                    layer_data = {'image': layer_data, 'profile': self._inherited_profile(layer_data.shape)}
            if 'display_range' not in layer_data and layer_data.get('vmin') is not None and layer_data.get('vmax') is not None:
                layer_data['display_range'] = (layer_data['vmin'], layer_data['vmax'])
            
            # Safely get image data
            image = layer_data.get('image')
            if image is None:
                raise ValueError("Image data not found in layer_data")
            
            # Make sure we have a clean copy of the image
            image = np.array(image, copy=True)
            
            # Handle NoData values (both profile nodata and NaN values)
            profile = layer_data.get('profile', {})
            nodata = profile.get('nodata', None)
            
            # Create combined mask for nodata and NaN values
            mask = np.zeros_like(image, dtype=bool)
            if nodata is not None:
                mask = mask | (image == nodata)
            mask = mask | np.isnan(image)
            
            # Apply mask to image
            image = np.ma.masked_where(mask, image)

            # Map coordinates: the raster is drawn in its real coordinates (pixel axes only if not georeferenced)
            _hw = image.shape[-2:]
            axis_spec = georef.axis_spec(profile.get('transform'), profile.get('crs'), int(_hw[0]), int(_hw[1]))
            extent = axis_spec.extent
            
            # Store original image dimensions
            if 'original_dims' not in layer_data:
                layer_data['original_dims'] = image.shape
            
            # For single-band images that were previously displayed as RGB
            if image.ndim == 3 and layer_data['original_dims'][0] == 1:
                image = image[0]  # Extract single band
                
            # For true single-band images
            if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
                display_image = image[0] if image.ndim == 3 else image
                
                # Calculate display range if not set
                if 'display_range' not in layer_data:
                    valid_vals = display_image[~display_image.mask] if hasattr(display_image, 'mask') else display_image
                    if valid_vals.size > 0:
                        vmin, vmax = np.percentile(valid_vals, [2, 98])  # Use 2-98% stretch for better contrast
                        # Ensure we don't have NaN in our display range
                        if not np.isnan(vmin) and not np.isnan(vmax):
                            layer_data['display_range'] = (vmin, vmax)
                        else:
                            layer_data['display_range'] = (np.nanmin(valid_vals), np.nanmax(valid_vals))
                    else:
                        layer_data['display_range'] = (0, 1)
                
                # Apply colormap with fallback for NaN values
                cmap = layer_data.get('cmap', 'viridis')
                cmap = _get_cmap(cmap).copy()
                cmap.set_bad(color='red', alpha=0.5)  # Make NaN values red
                
                img = self.ax.imshow(display_image, extent=extent,
                                cmap=cmap,
                                vmin=layer_data['display_range'][0],
                                vmax=layer_data['display_range'][1])
                
                # Add colorbar if needed (uncomment if you want it)
                # if hasattr(self, 'colorbar'):
                #     self.colorbar.remove()
                # self.colorbar = self.fig.colorbar(img, ax=self.ax)
                # self.colorbar.set_label('Value')
            
            # Handle RGB images (3 bands) - same as before
            elif image.ndim == 3 and image.shape[0] in [3, 4]:
                # Ensure we have exactly 3 bands for RGB
                rgb_image = image[:3]
                
                # Normalize each band separately for better contrast
                rgb_image = rgb_image.astype(np.float32)
                
                # Apply 2-98% contrast stretch for each band
                for i in range(3):
                    band = rgb_image[i]
                    if hasattr(band, 'mask'):
                        valid_pixels = band[~band.mask]
                    else:
                        valid_pixels = band
                    
                    if valid_pixels.size > 0:
                        p2, p98 = np.percentile(valid_pixels, [2, 98])
                        rgb_image[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                    else:
                        rgb_image[i] = np.zeros_like(band)
                
                # Transpose to (height, width, bands) and display
                rgb_image = rgb_image.transpose(1, 2, 0)
                self.ax.imshow(rgb_image, extent=extent)
                
                # Remove colorbar for RGB images
                if hasattr(self, 'colorbar'):
                    self.colorbar.remove()
                    del self.colorbar
            
            self._decorate_map_axes(axis_spec)

            # Configure grid if enabled
            if show_grid:
                self.ax.grid(True, color='white', linestyle=':', alpha=0.3)
                self.ax.set_axisbelow(True)
            
            # Set title and refresh
            self.ax.set_title(layer_name)
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Update current layer reference
            self.current_layer = layer_name
            self.layers[layer_name] = layer_data  # Save any modifications
            self.update_info_box(layer_data=layer_data)

        except Exception as e:
            logging.info(f"Error displaying layer {layer_name}: {str(e)}")
            traceback.print_exc()
            self._fallback_display(layer_name)

    def switch_layer(self, item):
        """Improved layer switching with dimension handling"""
        try:
            layer_name = item.text()
            if layer_name not in self.layers:
                QMessageBox.warning(self, "Error", f"Layer '{layer_name}' not found!")
                return
            
            layer_data = self.layers[layer_name]
            
            # Restore original dimensions if needed
            if 'original_dims' in layer_data:
                current_dims = layer_data['image'].shape
                if current_dims != layer_data['original_dims']:
                    if layer_data['original_dims'][0] == 1 and len(current_dims) == 2:
                        layer_data['image'] = layer_data['image'][np.newaxis, ...]
                    elif layer_data['original_dims'][0] == 1 and current_dims[0] != 1:
                        layer_data['image'] = layer_data['image'][0][np.newaxis, ...]
            
            # Display the layer
            if 'image' in layer_data:
                self.display_raster_layer(layer_data, layer_name)
            elif 'gdf' in layer_data:
                self.display_vector_layer(layer_data, layer_name)
                self.update_info_box(layer_data=layer_data)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Layer switch failed: {str(e)}")

    def display_vector_layer(self, layer_data, layer_name, show_grid=True):
        """Display vector layer with configurable label placement"""
        try:
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            
            gdf = layer_data['gdf']
            
            # Plot different geometry types
            for geom_type in ['Point', 'LineString', 'Polygon']:
                if geom_type in gdf.geom_type.values:
                    subset = gdf[gdf.geom_type == geom_type]
                    if geom_type == 'Point':
                        self.ax.scatter(
                            subset.geometry.x,
                            subset.geometry.y,
                            color=layer_data.get('color', 'red'),
                            s=layer_data.get('size', 5)
                        )
                    else:
                        subset.plot(
                            ax=self.ax,
                            color=layer_data.get('color', 'blue'),
                            linewidth=layer_data.get('linewidth', 1)
                        )
            
            # Add labels if enabled
            if 'label_field' in layer_data and layer_data['label_field'] in gdf.columns:
                label_field = layer_data['label_field']
                font_family = layer_data.get('label_font', 'Arial')
                font_size = layer_data.get('label_size', 10)
                label_color = layer_data.get('label_color', '#000000')
                placement = layer_data.get('label_placement', 'Auto')
                offset = layer_data.get('label_offset', 5)
                
                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    label_text = str(row[label_field])
                    
                    if geom.geom_type in ['Point', 'MultiPoint']:
                        if geom.geom_type == 'MultiPoint':
                            centroid = geom.centroid
                        else:
                            centroid = geom
                        
                        x, y = centroid.x, centroid.y
                        
                        # Point label placement options
                        if placement == 'Above':
                            va = 'bottom'
                            ha = 'center'
                            y_offset = offset
                            x_offset = 0
                        elif placement == 'Below':
                            va = 'top'
                            ha = 'center'
                            y_offset = -offset
                            x_offset = 0
                        elif placement == 'Left':
                            va = 'center'
                            ha = 'right'
                            x_offset = -offset
                            y_offset = 0
                        elif placement == 'Right':
                            va = 'center'
                            ha = 'left'
                            x_offset = offset
                            y_offset = 0
                        else:  # Auto or default
                            va = 'center'
                            ha = 'center'
                            x_offset = 0
                            y_offset = 0
                        
                        self.ax.text(x + x_offset, y + y_offset, label_text, 
                                    fontsize=font_size, color=label_color,
                                    fontfamily=font_family, va=va, ha=ha)
                    
                    elif geom.geom_type in ['LineString', 'MultiLineString']:
                        if geom.geom_type == 'MultiLineString':
                            line = geom.geoms[0]
                        else:
                            line = geom
                        
                        # Line label placement options
                        if placement == 'Start':
                            point = line.interpolate(0.1, normalized=True)
                        elif placement == 'End':
                            point = line.interpolate(0.9, normalized=True)
                        else:  # Middle or default
                            point = line.interpolate(0.5, normalized=True)
                        
                        self.ax.text(point.x, point.y, label_text,
                                    fontsize=font_size, color=label_color,
                                    fontfamily=font_family, va='center', ha='center')
                    
                    elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                        if geom.geom_type == 'MultiPolygon':
                            centroid = geom.centroid
                        else:
                            centroid = geom.centroid
                        
                        # Polygon label placement options
                        if placement == 'NW':
                            x_offset = -offset
                            y_offset = offset
                        elif placement == 'NE':
                            x_offset = offset
                            y_offset = offset
                        elif placement == 'SW':
                            x_offset = -offset
                            y_offset = -offset
                        elif placement == 'SE':
                            x_offset = offset
                            y_offset = -offset
                        else:  # Center or default
                            x_offset = 0
                            y_offset = 0
                        
                        self.ax.text(centroid.x + x_offset, centroid.y + y_offset, label_text,
                                    fontsize=font_size, color=label_color,
                                    fontfamily=font_family, va='center', ha='center')
            
            # Configure grid
            if show_grid:
                self.ax.grid(True, color='gray', linestyle='--', alpha=0.5)
                self.ax.set_axisbelow(True)
            
            # Set coordinate display
            bounds = gdf.total_bounds
            self.ax.set_xlim(bounds[0], bounds[2])
            self.ax.set_ylim(bounds[1], bounds[3])
            
            # Finalize display
            self.ax.set_title(layer_name)
            self.ax.set_aspect('equal')
            self.fig.tight_layout(pad=0.1)
            self.canvas.draw()

        except Exception as e:
            logging.info(f"Vector display error: {str(e)}")
            self._fallback_display(layer_name)

    


    
    def _fallback_display(self, layer_name):
        """Simplest possible display when everything fails"""
        try:
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, layer_name, ha='center', va='center')
            self.fig.tight_layout(pad=0)
            self.canvas.draw()
        except Exception:
            self.canvas.draw()    


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 
    # # =========================================
    # # NEW SUPPORTING METHODS FOR ENHANCED DIALOG
    # # =========================================
    def show_layer_properties(self):
        """Display comprehensive layer properties dialog with ArcMap-like symbology options"""
        current_item = self.layer_list.currentItem()
        if not current_item:
            return
            
        layer_name = current_item.text()
        layer_data = self.layers[layer_name]
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Layer Properties - {layer_name}")
        dialog.setMinimumSize(800, 600)
        layout = QVBoxLayout()
        
        # Tab widget for different property categories
        tabs = QTabWidget()
        
        # ======================
        # GENERAL TAB
        # ======================
        general_tab = QWidget()
        general_layout = QFormLayout()
        
        # Layer name
        name_edit = QLineEdit(layer_name)
        general_layout.addRow("Layer Name:", name_edit)
        
        # Layer source
        source_label = QLabel(layer_data.get('source', 'Unknown'))
        source_label.setWordWrap(True)
        general_layout.addRow("Source:", source_label)
        
        # CRS information
        crs_info = self.get_crs_info(layer_data.get('gdf').crs if 'gdf' in layer_data else (layer_data.get('profile') or {}).get('crs'))
        crs_label = QLabel(f"{crs_info.get('name', 'Unknown')} ({crs_info.get('epsg', 'Unknown')})")
        general_layout.addRow("Coordinate System:", crs_label)
        
        # Layer visibility
        visible_check = QCheckBox("Visible")
        visible_check.setChecked(layer_data.get('visible', True))
        visible_check.stateChanged.connect(lambda s: self._set_layer_visibility(layer_name, s == Qt.Checked))
        general_layout.addRow("Visibility:", visible_check)
        
        # Transparency slider
        transparency_slider = QSlider(Qt.Horizontal)
        transparency_slider.setRange(0, 100)
        transparency_slider.setValue(int(layer_data.get('transparency', 0)))
        transparency_slider.valueChanged.connect(lambda v: self._set_transparency(layer_name, v))
        transparency_layout = QHBoxLayout()
        transparency_layout.addWidget(transparency_slider)
        transparency_value = QLabel(f"{transparency_slider.value()}%")
        transparency_slider.valueChanged.connect(lambda v: transparency_value.setText(f"{v}%"))
        transparency_layout.addWidget(transparency_value)
        general_layout.addRow("Transparency:", transparency_layout)
        
        # Metadata display
        metadata_text = QTextEdit()
        metadata_text.setReadOnly(True)
        metadata_text.setPlainText(self._generate_metadata_text(layer_data))
        general_layout.addRow("Metadata:", metadata_text)
        
        general_tab.setLayout(general_layout)
        tabs.addTab(general_tab, "General")

        
        
        # ======================
        # SYMBOLOGY TAB (Professional)
        # ======================
        if 'gdf' in layer_data:
            symbology_tab = QWidget()
            symbology_layout = QVBoxLayout()
            
            # Symbology type selector with icons
            symbology_type = QComboBox()
            symbology_type.addItem(QIcon(":/icons/single_symbol.png"), "Single Symbol")
            symbology_type.addItem(QIcon(":/icons/categorized.png"), "Categorized")
            symbology_type.addItem(QIcon(":/icons/graduated.png"), "Graduated")
            symbology_type.addItem(QIcon(":/icons/heatmap.png"), "Heatmap")
            symbology_type.addItem(QIcon(":/icons/rule_based.png"), "Rule-based")
            symbology_type.setCurrentText(layer_data.get('symbology_type', 'Single Symbol'))
            
            # Stacked widget for different symbology types
            symbology_stack = QStackedWidget()
            
            # ------------------
            # SINGLE SYMBOL PANEL
            # ------------------
            single_symbol_panel = QWidget()
            single_symbol_layout = QFormLayout()
            
            # Symbol preview
            symbol_preview = QLabel()
            symbol_preview.setFixedSize(32, 32)
            self._update_symbol_preview(layer_name, symbol_preview)
            
            symbol_button = QPushButton("Change Symbol...")
            symbol_button.clicked.connect(lambda: self._change_symbol(layer_name, symbol_preview))
            single_symbol_layout.addRow("Symbol:", symbol_button)
            single_symbol_layout.addRow("Preview:", symbol_preview)
            
            # Advanced symbol properties
            advanced_group = QGroupBox("Advanced Symbol Properties")
            advanced_layout = QFormLayout()
            
            # Line properties
            if any(t in layer_data['gdf'].geometry.type for t in ['LineString', 'MultiLineString']):
                width_spin = QDoubleSpinBox()
                width_spin.setRange(0.1, 10.0)
                width_spin.setValue(layer_data.get('line_width', 1.0))
                width_spin.valueChanged.connect(lambda v: self._update_line_width(layer_name, v))
                advanced_layout.addRow("Line Width (mm):", width_spin)
                
                line_style = QComboBox()
                line_style.addItems(["Solid", "Dash", "Dot", "Dash Dot", "Dash Dot Dot", "Custom"])
                line_style.setCurrentText(layer_data.get('line_style', 'Solid'))
                line_style.currentTextChanged.connect(lambda s: self._set_line_style(layer_name, s))
                advanced_layout.addRow("Line Style:", line_style)
                
                self.dash_pattern_edit = QLineEdit(layer_data.get('dash_pattern', '5,2'))
                self.dash_pattern_edit.setVisible(False)
                line_style.currentTextChanged.connect(lambda t: self.dash_pattern_edit.setVisible(t == "Custom"))
                advanced_layout.addRow("Dash Pattern:", self.dash_pattern_edit)
            
            # Point properties
            if any(t in layer_data['gdf'].geometry.type for t in ['Point', 'MultiPoint']):
                size_spin = QDoubleSpinBox()
                size_spin.setRange(1, 50)
                size_spin.setValue(layer_data.get('point_size', 5))
                size_spin.valueChanged.connect(lambda v: self._update_point_size(layer_name, v))
                advanced_layout.addRow("Size (mm):", size_spin)
                
                symbol_combo = QComboBox()
                symbol_combo.addItems(["Circle", "Square", "Triangle", "Cross", "X", "Star", "Arrow", "Font Marker"])
                symbol_combo.setCurrentText(layer_data.get('point_symbol', 'Circle'))
                symbol_combo.currentTextChanged.connect(lambda s: self._set_point_symbol(layer_name, s))
                advanced_layout.addRow("Symbol Type:", symbol_combo)
                
                rotation_spin = QDoubleSpinBox()
                rotation_spin.setRange(0, 360)
                rotation_spin.setValue(layer_data.get('rotation', 0))
                rotation_spin.valueChanged.connect(lambda v: self._set_point_rotation(layer_name, v))
                advanced_layout.addRow("Rotation (°):", rotation_spin)
            
            # Polygon properties
            if any(t in layer_data['gdf'].geometry.type for t in ['Polygon', 'MultiPolygon']):
                fill_style = QComboBox()
                fill_style.addItems(["Solid", "Horizontal", "Vertical", "Cross", "Diagonal", "Dense", "No Brush"])
                fill_style.setCurrentText(layer_data.get('fill_style', 'Solid'))
                fill_style.currentTextChanged.connect(lambda s: self._set_fill_style(layer_name, s))
                advanced_layout.addRow("Fill Style:", fill_style)
                
                outline_width = QDoubleSpinBox()
                outline_width.setRange(0, 5)
                outline_width.setValue(layer_data.get('outline_width', 0.3))
                outline_width.valueChanged.connect(lambda v: self._set_outline_width(layer_name, v))
                advanced_layout.addRow("Outline Width (mm):", outline_width)
            
            advanced_group.setLayout(advanced_layout)
            single_symbol_layout.addRow(advanced_group)
            single_symbol_panel.setLayout(single_symbol_layout)
            symbology_stack.addWidget(single_symbol_panel)
            
            # ------------------
            # CATEGORIZED PANEL
            # ------------------
            categorized_panel = QWidget()
            categorized_layout = QFormLayout()
            
            # Field selection
            field_combo = QComboBox()
            for col in layer_data['gdf'].columns:
                if col != 'geometry' and layer_data['gdf'][col].dtype in [object, str, 'category']:
                    field_combo.addItem(col)
            field_combo.setCurrentText(layer_data.get('categorized_field', ''))
            categorized_layout.addRow("Field:", field_combo)
            
            # Color ramp selection
            ramp_combo = QComboBox()
            ramps = ["Spectral", "RdYlGn", "RdYlBu", "PiYG", "PRGn", "BrBG", "PuOr", "Set1", "Set2", "Set3", "Paired", "Accent"]
            ramp_combo.addItems(ramps)
            ramp_combo.setCurrentText(layer_data.get('color_ramp', 'Spectral'))
            categorized_layout.addRow("Color Ramp:", ramp_combo)
            
            # Classification preview
            preview_table = QTableWidget()
            preview_table.setColumnCount(2)
            preview_table.setHorizontalHeaderLabels(["Value", "Symbol"])
            preview_table.verticalHeader().setVisible(False)
            preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self._update_categorized_preview(layer_name, preview_table)
            categorized_layout.addRow(preview_table)
            
            categorized_panel.setLayout(categorized_layout)
            symbology_stack.addWidget(categorized_panel)
            
            # ------------------
            # GRADUATED PANEL
            # ------------------
            graduated_panel = QWidget()
            graduated_layout = QFormLayout()
            
            # Field selection (numeric only)
            num_field_combo = QComboBox()
            for col in layer_data['gdf'].columns:
                if col != 'geometry' and np.issubdtype(layer_data['gdf'][col].dtype, np.number):
                    num_field_combo.addItem(col)
            num_field_combo.setCurrentText(layer_data.get('graduated_field', ''))
            graduated_layout.addRow("Field:", num_field_combo)
            
            # Classification method
            method_combo = QComboBox()
            method_combo.addItems(["Equal Interval", "Quantile (Equal Count)", "Natural Breaks (Jenks)", "Standard Deviation", "Pretty Breaks"])
            method_combo.setCurrentText(layer_data.get('classification_method', 'Equal Interval'))
            graduated_layout.addRow("Method:", method_combo)
            
            # Classes and color ramp
            classes_spin = QSpinBox()
            classes_spin.setRange(2, 20)
            classes_spin.setValue(layer_data.get('classes', 5))
            graduated_layout.addRow("Classes:", classes_spin)
            
            graduated_ramp_combo = QComboBox()
            graduated_ramp_combo.addItems(ramps)
            graduated_ramp_combo.setCurrentText(layer_data.get('graduated_ramp', 'Spectral'))
            graduated_layout.addRow("Color Ramp:", graduated_ramp_combo)
            
            # Mode (color/size)
            mode_group = QButtonGroup()
            color_radio = QRadioButton("Color")
            size_radio = QRadioButton("Size")
            mode_group.addButton(color_radio)
            mode_group.addButton(size_radio)
            mode_group.setId(color_radio, 0)
            mode_group.setId(size_radio, 1)
            
            mode = layer_data.get('graduated_mode', 0)
            color_radio.setChecked(mode == 0)
            size_radio.setChecked(mode == 1)
            
            mode_layout = QHBoxLayout()
            mode_layout.addWidget(color_radio)
            mode_layout.addWidget(size_radio)
            graduated_layout.addRow("Mode:", mode_layout)
            
            graduated_panel.setLayout(graduated_layout)
            symbology_stack.addWidget(graduated_panel)
            
            # Connect symbology type to stack
            symbology_type.currentIndexChanged.connect(symbology_stack.setCurrentIndex)
            
            # Add to main layout
            symbology_layout.addWidget(QLabel("Symbology Type:"))
            symbology_layout.addWidget(symbology_type)
            symbology_layout.addWidget(symbology_stack)
            
            # Connect field changes to preview updates
            field_combo.currentTextChanged.connect(
                lambda f: self._update_categorized_preview(layer_name, preview_table, f))
            
            symbology_tab.setLayout(symbology_layout)
            tabs.addTab(symbology_tab, "Symbology")
            
            # ======================
            # LABELS TAB (Enhanced)
            # ======================
            labels_tab = QWidget()
            labels_layout = QFormLayout()
            
            # Label enable/disable
            label_enable = QCheckBox("Label features")
            label_enable.setChecked('label_field' in layer_data)
            labels_layout.addRow(label_enable)
            
            # Label field selection
            label_combo = QComboBox()
            label_combo.addItem("(No labels)")
            for col in layer_data['gdf'].columns:
                if col != 'geometry':
                    label_combo.addItem(col)
            
            current_label = layer_data.get('label_field')
            if current_label:
                idx = label_combo.findText(current_label)
                if idx >= 0:
                    label_combo.setCurrentIndex(idx)
            
            label_combo.setEnabled(label_enable.isChecked())
            label_enable.toggled.connect(label_combo.setEnabled)
            labels_layout.addRow("Label Field:", label_combo)
            
            # Label font settings
            font_button = QPushButton("Change Font...")
            current_font = QFont(layer_data.get('label_font', 'Arial'))
            current_font.setPointSize(layer_data.get('label_size', 10))
            font_button.clicked.connect(lambda: self._change_label_font(layer_name, font_button))
            font_button.setFont(current_font)
            labels_layout.addRow("Label Font:", font_button)
            
            # Label color
            label_color_button = QPushButton()
            label_color_button.setStyleSheet(f"background-color: {layer_data.get('label_color', '#000000')}")
            label_color_button.clicked.connect(lambda: self._pick_label_color(layer_name, label_color_button))
            labels_layout.addRow("Label Color:", label_color_button)
            
            # Geometry-specific label placement
            geom_types = layer_data['gdf'].geometry.type.unique()
            
            if any(t in ['Point', 'MultiPoint'] for t in geom_types):
                placement = QComboBox()
                placement.addItems(["Auto", "Above", "Below", "Left", "Right"])
                placement.setCurrentText(layer_data.get('label_placement', 'Auto'))
                placement.currentTextChanged.connect(lambda p: self._set_label_placement(layer_name, p))
                labels_layout.addRow("Point Label Placement:", placement)
                
                offset_spin = QSpinBox()
                offset_spin.setRange(1, 50)
                offset_spin.setValue(layer_data.get('label_offset', 5))
                offset_spin.valueChanged.connect(lambda v: self._set_label_offset(layer_name, v))
                labels_layout.addRow("Label Offset (pixels):", offset_spin)
                
            elif any(t in ['LineString', 'MultiLineString'] for t in geom_types):
                placement = QComboBox()
                placement.addItems(["Middle", "Start", "End"])
                placement.setCurrentText(layer_data.get('label_placement', 'Middle'))
                placement.currentTextChanged.connect(lambda p: self._set_label_placement(layer_name, p))
                labels_layout.addRow("Line Label Placement:", placement)
                
            elif any(t in ['Polygon', 'MultiPolygon'] for t in geom_types):
                placement = QComboBox()
                placement.addItems(["Center", "NW", "NE", "SW", "SE"])
                placement.setCurrentText(layer_data.get('label_placement', 'Center'))
                placement.currentTextChanged.connect(lambda p: self._set_label_placement(layer_name, p))
                labels_layout.addRow("Polygon Label Placement:", placement)
                
                offset_spin = QSpinBox()
                offset_spin.setRange(1, 50)
                offset_spin.setValue(layer_data.get('label_offset', 5))
                offset_spin.valueChanged.connect(lambda v: self._set_label_offset(layer_name, v))
                labels_layout.addRow("Label Offset (pixels):", offset_spin)
            
            # Connect label field changes
            label_combo.currentTextChanged.connect(
                lambda t: self._update_label_field(layer_name, t if t != "(No labels)" else None))
            
            labels_tab.setLayout(labels_layout)
            tabs.addTab(labels_tab, "Labels")
        
        # ======================
        # DISPLAY TAB
        # ======================
        display_tab = QWidget()
        display_layout = QFormLayout()
        
        # Scale range (min/max scale)
        scale_group = QGroupBox("Scale Range")
        scale_layout = QFormLayout()
        
        min_scale = QDoubleSpinBox()
        min_scale.setRange(0, 1e9)
        min_scale.setValue(layer_data.get('min_scale', 0))
        min_scale.setSpecialValueText("(No minimum)")
        min_scale.valueChanged.connect(lambda v: self._set_min_scale(layer_name, v))
        scale_layout.addRow("Minimum Scale (1:x):", min_scale)
        
        max_scale = QDoubleSpinBox()
        max_scale.setRange(0, 1e9)
        max_scale.setValue(layer_data.get('max_scale', 1e9))
        max_scale.setSpecialValueText("(No maximum)")
        max_scale.valueChanged.connect(lambda v: self._set_max_scale(layer_name, v))
        scale_layout.addRow("Maximum Scale (1:x):", max_scale)
        
        scale_group.setLayout(scale_layout)
        display_layout.addRow(scale_group)
        
        # Display expression (for raster)
        if 'image' in layer_data:
            band_combo = QComboBox()
            num_bands = layer_data['image'].shape[2] if len(layer_data['image'].shape) > 2 else 1
            if num_bands >= 3:
                band_combo.addItems(["RGB", "Single Band", "Palette"])
                current_display = layer_data.get('display_mode', 'RGB')
                band_combo.setCurrentText(current_display)
            elif num_bands == 1:
                band_combo.addItems(["Single Band", "Palette"])
                current_display = layer_data.get('display_mode', 'Single Band')
                band_combo.setCurrentText(current_display)
            
            band_combo.currentTextChanged.connect(lambda t: self._set_display_mode(layer_name, t))
            display_layout.addRow("Display Mode:", band_combo)
        
        display_tab.setLayout(display_layout)
        tabs.addTab(display_tab, "Display")
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(lambda: self.redraw_all_layers())
        
        # Add widgets to main layout
        layout.addWidget(tabs)
        layout.addWidget(button_box)
        dialog.setLayout(layout)
        
        if dialog.exec_():
            # Save any changes made in the dialog
            new_name = name_edit.text()
            if new_name != layer_name and new_name not in self.layers:
                self.layers[new_name] = self.layers.pop(layer_name)
                current_item.setText(new_name)
            
            # Update symbology settings
            if 'gdf' in layer_data:
                # Update categorized/graduated fields
                if symbology_type.currentText() == "Categorized":
                    self._update_categorized_field(layer_name, field_combo.currentText())
                elif symbology_type.currentText() == "Graduated":
                    self._update_graduated_field(layer_name, num_field_combo.currentText())
                
                # Update label settings
                if label_enable.isChecked() and label_combo.currentText() != "(No labels)":
                    self._update_label_field(layer_name, label_combo.currentText())
                else:
                    self._update_label_field(layer_name, None)
            
            self.redraw_all_layers()

    def _change_symbol(self, layer_name, preview_label):
        """Open symbol selector dialog and update symbol preview"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Symbol")
        dialog.setMinimumSize(400, 300)
        layout = QVBoxLayout()
        
        # Get current symbol properties
        layer_data = self.layers[layer_name]
        current_color = QColor(layer_data.get('color', '#FF0000'))
        current_size = layer_data.get('size', 5)
        
        # Color selection
        color_button = QPushButton("Select Color...")
        color_button.setStyleSheet(f"background-color: {current_color.name()}")
        color_display = QLabel()
        color_display.setFixedSize(32, 32)
        color_display.setStyleSheet(f"background-color: {current_color.name()}")
        
        def select_color():
            nonlocal current_color  # Add this line to access the outer current_color
            color = QColorDialog.getColor(current_color, self)
            if color.isValid():
                current_color = color  # Update the outer current_color
                color_button.setStyleSheet(f"background-color: {color.name()}")
                color_display.setStyleSheet(f"background-color: {color.name()}")
                update_preview()
        
        color_button.clicked.connect(select_color)
        
        # Size selection
        size_spin = QSpinBox()
        size_spin.setRange(1, 50)
        size_spin.setValue(current_size)
        
        # Symbol type selection (for points)
        if any(t in self.layers[layer_name]['gdf'].geometry.type for t in ['Point', 'MultiPoint']):
            symbol_combo = QComboBox()
            symbol_combo.addItems(["Circle", "Square", "Triangle", "Cross", "X", "Star"])
            symbol_combo.setCurrentText(layer_data.get('point_symbol', 'Circle'))
        
        # Preview area
        preview = QLabel()
        preview.setFixedSize(100, 100)
        preview.setAlignment(Qt.AlignCenter)
        
        def update_preview():
            # Create a pixmap with the current symbol
            pixmap = QPixmap(100, 100)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw the symbol based on geometry type
            if any(t in self.layers[layer_name]['gdf'].geometry.type for t in ['Point', 'MultiPoint']):
                # Draw point symbol
                brush = QBrush(current_color)
                pen = QPen(Qt.black, 1)
                painter.setBrush(brush)
                painter.setPen(pen)
                
                symbol = symbol_combo.currentText()
                size = size_spin.value()
                
                if symbol == "Circle":
                    painter.drawEllipse(50 - size//2, 50 - size//2, size, size)
                elif symbol == "Square":
                    painter.drawRect(50 - size//2, 50 - size//2, size, size)
                elif symbol == "Triangle":
                    polygon = QPolygonF([QPointF(50, 50-size//2), 
                                    QPointF(50+size//2, 50+size//2), 
                                    QPointF(50-size//2, 50+size//2)])
                    painter.drawPolygon(polygon)
                elif symbol == "Cross":
                    painter.drawLine(50-size//2, 50, 50+size//2, 50)
                    painter.drawLine(50, 50-size//2, 50, 50+size//2)
                elif symbol == "X":
                    painter.drawLine(50-size//2, 50-size//2, 50+size//2, 50+size//2)
                    painter.drawLine(50+size//2, 50-size//2, 50-size//2, 50+size//2)
                elif symbol == "Star":
                    # Draw a simple star
                    star = QPolygonF()
                    for i in range(5):
                        star.append(QPointF(50 + size//2 * math.cos(math.pi/2 + 2*math.pi*i/5),
                                        50 - size//2 * math.sin(math.pi/2 + 2*math.pi*i/5)))
                        star.append(QPointF(50 + size//4 * math.cos(math.pi/2 + 2*math.pi*(i+0.5)/5),
                                        50 - size//4 * math.sin(math.pi/2 + 2*math.pi*(i+0.5)/5)))
                    painter.drawPolygon(star)
            
            elif any(t in self.layers[layer_name]['gdf'].geometry.type for t in ['LineString', 'MultiLineString']):
                # Draw line symbol
                pen = QPen(current_color, size_spin.value())
                painter.setPen(pen)
                painter.drawLine(20, 50, 80, 50)
            
            elif any(t in self.layers[layer_name]['gdf'].geometry.type for t in ['Polygon', 'MultiPolygon']):
                # Draw polygon symbol
                brush = QBrush(current_color)
                pen = QPen(Qt.black, 1)
                painter.setBrush(brush)
                painter.setPen(pen)
                painter.drawRect(30, 30, 40, 40)
            
            painter.end()
            preview.setPixmap(pixmap)
        
        # Connect signals for live preview
        size_spin.valueChanged.connect(update_preview)
        if any(t in self.layers[layer_name]['gdf'].geometry.type for t in ['Point', 'MultiPoint']):
            symbol_combo.currentTextChanged.connect(update_preview)
        color_button.clicked.connect(update_preview)
        
        # Form layout for controls
        form = QFormLayout()
        form.addRow("Color:", color_button)
        form.addRow("Preview:", color_display)
        form.addRow("Size:", size_spin)
        if any(t in self.layers[layer_name]['gdf'].geometry.type for t in ['Point', 'MultiPoint']):
            form.addRow("Symbol Type:", symbol_combo)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        # Initial preview
        update_preview()
        
        # Layout
        layout.addLayout(form)
        layout.addWidget(preview)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        
        if dialog.exec_():
            # Save the new symbol properties
            layer_data = self.layers[layer_name]
            layer_data['color'] = current_color.name()
            layer_data['size'] = size_spin.value()
            
            if any(t in layer_data['gdf'].geometry.type for t in ['Point', 'MultiPoint']):
                layer_data['point_symbol'] = symbol_combo.currentText()
            elif any(t in layer_data['gdf'].geometry.type for t in ['LineString', 'MultiLineString']):
                layer_data['line_width'] = size_spin.value()
            elif any(t in layer_data['gdf'].geometry.type for t in ['Polygon', 'MultiPolygon']):
                layer_data['fill_color'] = current_color.name()
            
            # Update the preview in properties dialog
            self._update_symbol_preview(layer_name, preview_label)
            self.redraw_all_layers()


    def _update_symbol_preview(self, layer_name, label):
        """Update the symbol preview thumbnail in properties dialog"""
        layer_data = self.layers[layer_name]
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        color = QColor(layer_data.get('color', '#FF0000'))
        size = min(32, layer_data.get('size', 5) * 3)  # Scale for preview
        
        if any(t in layer_data['gdf'].geometry.type for t in ['Point', 'MultiPoint']):
            brush = QBrush(color)
            pen = QPen(Qt.black, 1)
            painter.setBrush(brush)
            painter.setPen(pen)
            
            symbol = layer_data.get('point_symbol', 'Circle')
            if symbol == "Circle":
                painter.drawEllipse(16 - size//2, 16 - size//2, size, size)
            elif symbol == "Square":
                painter.drawRect(16 - size//2, 16 - size//2, size, size)
            elif symbol == "Triangle":
                polygon = QPolygonF([QPointF(16, 16-size//2), 
                                QPointF(16+size//2, 16+size//2), 
                                QPointF(16-size//2, 16+size//2)])
                painter.drawPolygon(polygon)
            elif symbol == "Cross":
                painter.drawLine(16-size//2, 16, 16+size//2, 16)
                painter.drawLine(16, 16-size//2, 16, 16+size//2)
            elif symbol == "X":
                painter.drawLine(16-size//2, 16-size//2, 16+size//2, 16+size//2)
                painter.drawLine(16+size//2, 16-size//2, 16-size//2, 16+size//2)
        
        elif any(t in layer_data['gdf'].geometry.type for t in ['LineString', 'MultiLineString']):
            pen = QPen(color, layer_data.get('line_width', 1))
            painter.setPen(pen)
            painter.drawLine(8, 16, 24, 16)
        
        elif any(t in layer_data['gdf'].geometry.type for t in ['Polygon', 'MultiPolygon']):
            brush = QBrush(color)
            pen = QPen(Qt.black, 1)
            painter.setBrush(brush)
            painter.setPen(pen)
            painter.drawRect(8, 8, 16, 16)
        
        painter.end()
        label.setPixmap(pixmap)

    def _update_categorized_preview(self, layer_name, table, field=None):
        """Update categorized symbology preview table"""
        layer_data = self.layers[layer_name]
        field = field or layer_data.get('categorized_field')
        if not field or field not in layer_data['gdf'].columns:
            return
        
        # Get unique values (limited to 20 for preview)
        unique_values = layer_data['gdf'][field].unique()[:20]
        table.setRowCount(len(unique_values))
        
        # Get current color ramp
        ramp_name = layer_data.get('color_ramp', 'Spectral')
        
        for i, value in enumerate(unique_values):
            # Add value
            table.setItem(i, 0, QTableWidgetItem(str(value)))
            
            # Add color preview
            color_item = QTableWidgetItem()
            color = self._get_category_color(value, ramp_name)
            color_item.setBackground(QColor(color))
            table.setItem(i, 1, color_item)
        
        table.resizeColumnsToContents()

    def _set_point_rotation(self, layer_name, angle):
        """Set rotation angle for point symbols"""
        self.layers[layer_name]['rotation'] = angle
        self.redraw_all_layers()

    def _set_outline_width(self, layer_name, width):
        """Set polygon outline width"""
        self.layers[layer_name]['outline_width'] = width
        self.redraw_all_layers()


    def _set_label_offset(self, layer_name, offset):
        """Set label offset in pixels"""
        self.layers[layer_name]['label_offset'] = offset
        self.redraw_all_layers()


    def _generate_metadata_text(self, layer_data):
        """Generate comprehensive metadata text for display in properties dialog
        
        Args:
            layer_data (dict): Dictionary containing either 'gdf' (vector) or 'image' (raster) data
            
        Returns:
            str: Formatted metadata string with detailed layer information
        """
        metadata = []
        
        if 'gdf' in layer_data:
            # Vector layer metadata
            gdf = layer_data['gdf']
            
            # Basic information
            metadata.append("=== VECTOR LAYER ===")
            metadata.append(f"Geometry Type: {gdf.geometry.type.iloc[0] if len(gdf) > 0 else 'Unknown'}")
            metadata.append(f"Feature Count: {len(gdf)}")
            
            # CRS details
            crs = gdf.crs
            if crs:
                metadata.append(f"\nCRS:")
                metadata.append(f"  Name: {crs.name if hasattr(crs, 'name') else 'Unknown'}")
                metadata.append(f"  EPSG: {crs.to_epsg() if hasattr(crs, 'to_epsg') else 'Unknown'}")
                metadata.append(f"  Proj4: {crs.to_proj4() if hasattr(crs, 'to_proj4') else 'Unknown'}")
                metadata.append(f"  Is Geographic: {'Yes' if crs.is_geographic else 'No'}")
            
            # Bounding box
            if not gdf.empty:
                bounds = gdf.total_bounds
                metadata.append("\nExtent:")
                metadata.append(f"  Min X: {bounds[0]:.6f}")
                metadata.append(f"  Min Y: {bounds[1]:.6f}")
                metadata.append(f"  Max X: {bounds[2]:.6f}")
                metadata.append(f"  Max Y: {bounds[3]:.6f}")
            
            # Attributes
            metadata.append("\nAttributes:")
            for col in gdf.columns:
                if col != 'geometry':
                    dtype = gdf[col].dtype
                    unique_count = f" ({gdf[col].nunique()} unique)" if dtype == 'object' else ""
                    null_count = f", {gdf[col].isna().sum()} nulls" if gdf[col].isna().any() else ""
                    metadata.append(f"  {col}: {dtype}{unique_count}{null_count}")
            
            # Statistics for numeric columns
            numeric_cols = gdf.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                metadata.append("\nNumeric Attributes Statistics:")
                for col in numeric_cols:
                    stats = gdf[col].describe()
                    metadata.append(f"  {col}:")
                    metadata.append(f"    Min: {stats['min']:.2f}")
                    metadata.append(f"    Max: {stats['max']:.2f}")
                    metadata.append(f"    Mean: {stats['mean']:.2f}")
                    metadata.append(f"    Std Dev: {stats['std']:.2f}")
        
        elif 'image' in layer_data:
            # Raster layer metadata
            img = layer_data['image']
            metadata.append("=== RASTER LAYER ===")
            
            # Basic information
            metadata.append(f"Dimensions: {img.shape} (bands, height, width)")
            metadata.append(f"Data Type: {img.dtype}")
            metadata.append(f"Size: {img.nbytes / (1024*1024):.2f} MB")
            
            # Band information
            if len(img.shape) == 3:
                metadata.append(f"\nNumber of Bands: {img.shape[0]}")
                if img.shape[0] > 1:
                    metadata.append("Band Order: " + ", ".join(str(i) for i in range(img.shape[0])))
            
            # Profile information
            if layer_data.get('profile'):
                profile = layer_data['profile']
                metadata.append("\nProfile Information:")
                metadata.append(f"  Driver: {profile.get('driver', 'Unknown')}")
                metadata.append(f"  CRS: {profile.get('crs', 'Unknown')}")
                metadata.append(f"  Transform: {profile.get('transform', 'Unknown')}")
                metadata.append(f"  NoData Value: {profile.get('nodata', 'None')}")
                metadata.append(f"  Compression: {profile.get('compress', 'None')}")
            
            # Statistics
            metadata.append("\nStatistics:")
            if len(img.shape) == 2:
                # Single band
                valid_pixels = img[~np.isnan(img)] if np.issubdtype(img.dtype, np.floating) else img
                metadata.append(f"  Min: {np.min(valid_pixels):.2f}")
                metadata.append(f"  Max: {np.max(valid_pixels):.2f}")
                metadata.append(f"  Mean: {np.mean(valid_pixels):.2f}")
                metadata.append(f"  Std Dev: {np.std(valid_pixels):.2f}")
            elif len(img.shape) == 3:
                # Multi-band
                for i in range(img.shape[0]):
                    band = img[i]
                    valid_pixels = band[~np.isnan(band)] if np.issubdtype(band.dtype, np.floating) else band
                    metadata.append(f"\nBand {i+1} Statistics:")
                    metadata.append(f"  Min: {np.min(valid_pixels):.2f}")
                    metadata.append(f"  Max: {np.max(valid_pixels):.2f}")
                    metadata.append(f"  Mean: {np.mean(valid_pixels):.2f}")
                    metadata.append(f"  Std Dev: {np.std(valid_pixels):.2f}")
        
        else:
            metadata.append("Unknown layer type - no metadata available")
        
        # Add layer timestamp if available
        if 'timestamp' in layer_data:
            metadata.append(f"\nLast Updated: {layer_data['timestamp']}")
        
        return "\n".join(metadata)


    def _set_layer_visibility(self, layer_name, visible):
        """Set layer visibility"""
        self.layers[layer_name]['visible'] = visible
        self.redraw_all_layers()

    def _set_transparency(self, layer_name, value):
        """Set layer transparency (0-100)"""
        self.layers[layer_name]['transparency'] = value
        self.redraw_all_layers()


    def _set_line_style(self, layer_name, style):
        """Set line style (solid, dash, etc.)"""
        self.layers[layer_name]['line_style'] = style
        self.redraw_all_layers()

    def _set_point_symbol(self, layer_name, symbol):
        """Set point symbol type"""
        self.layers[layer_name]['point_symbol'] = symbol
        self.redraw_all_layers()

    def _set_fill_style(self, layer_name, style):
        """Set polygon fill style"""
        self.layers[layer_name]['fill_style'] = style
        self.redraw_all_layers()


    def _pick_label_color(self, layer_name, button):
        """Pick a color for labels"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.layers[layer_name]['label_color'] = color.name()
            button.setStyleSheet(f"background-color: {color.name()}")
            self.redraw_all_layers()

    def _change_label_font(self, layer_name, button):
        """Change the label font"""
        font, ok = QFontDialog.getFont(button.font())
        if ok:
            self.layers[layer_name]['label_font'] = font.family()
            self.layers[layer_name]['label_size'] = font.pointSize()
            button.setFont(font)
            self.redraw_all_layers()

    def _set_label_placement(self, layer_name, placement):
        """Set label placement for point features"""
        self.layers[layer_name]['label_placement'] = placement
        self.redraw_all_layers()

    def _set_min_scale(self, layer_name, scale):
        """Set minimum visible scale"""
        self.layers[layer_name]['min_scale'] = scale
        self.redraw_all_layers()

    def _set_max_scale(self, layer_name, scale):
        """Set maximum visible scale"""
        self.layers[layer_name]['max_scale'] = scale
        self.redraw_all_layers()

    def _set_display_mode(self, layer_name, mode):
        """Set display mode for raster layers"""
        self.layers[layer_name]['display_mode'] = mode
        self.redraw_all_layers()
    
    def redraw_all_layers(self):
        """Redraw all layers in correct z-order (bottom to top)"""
        # Clear the current axes but keep the figure
        self.ax.clear()
        
        # Draw from bottom up (first in list is bottom)
        for i in range(self.layer_list.count()):
            layer_name = self.layer_list.item(i).text()
            layer_data = self.layers[layer_name]
            
            # Skip if layer is not visible
            if not layer_data.get('visible', True):
                continue
                
            # Apply transparency if set
            alpha = 1 - (layer_data.get('transparency', 0) / 100)
            
            if 'image' in layer_data:
                # Update layer_data with alpha before passing
                layer_data['alpha'] = alpha
                self.display_raster_layer(layer_data['image'], layer_name)
            elif 'gdf' in layer_data:
                # Update layer_data with alpha before passing
                layer_data['alpha'] = alpha
                self.display_vector_layer(layer_data, layer_name)
        
        # Redraw the canvas
        self.ax.autoscale_view()
        self.canvas.draw()


    def _update_line_width(self, layer_name, width):
        self.layers[layer_name]['line_width'] = width
        self.redraw_all_layers()

    def _update_point_size(self, layer_name, size):
        self.layers[layer_name]['point_size'] = size
        self.redraw_all_layers()

    def _update_label_field(self, layer_name, field):
        self.layers[layer_name]['label_field'] = field
        self.redraw_all_layers()

    def _update_graduated_field(self, layer_name, field):
        """Update the graduated symbology field and classify data"""
        if not field:
            return
            
        layer_data = self.layers[layer_name]
        layer_data['graduated_field'] = field
        
        # Get numerical values from the field
        values = layer_data['gdf'][field].dropna().values
        
        if len(values) == 0:
            return
            
        # Default classification - equal interval with 5 classes
        n_classes = layer_data.get('graduated_classes', 5)
        min_val, max_val = np.min(values), np.max(values)
        breaks = np.linspace(min_val, max_val, n_classes + 1)
        
        # Store classification breaks
        layer_data['graduated_breaks'] = breaks.tolist()
        
        # Generate colors from current ramp
        ramp_name = layer_data.get('color_ramp', 'Spectral')
        colors = self._get_color_ramp(ramp_name, n_classes)
        layer_data['graduated_colors'] = colors
        
        self.redraw_all_layers()

    def _get_color_ramp(self, ramp_name, n_classes):
        """Generate a color ramp with specified number of classes"""
        color_ramps = {
            'Spectral': ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', 
                        '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'],
            'RdYlGn': ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', 
                    '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'],
            'RdYlBu': ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf',
                    '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'],
            'Set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                    '#ffff33', '#a65628', '#f781bf', '#999999'],
            'Set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
                    '#ffd92f', '#e5c494', '#b3b3b3'],
            'Set3': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']
        }
        
        ramp = color_ramps.get(ramp_name, color_ramps['Spectral'])
        n_ramp = len(ramp)
        
        # Interpolate colors if needed
        if n_classes <= n_ramp:
            return ramp[:n_classes]
        else:
            # Interpolate for more classes than the ramp provides
            x = np.linspace(0, 1, n_ramp)
            x_new = np.linspace(0, 1, n_classes)
            colors = []
            for i in range(3):  # RGB components
                component = [int(ramp[j][i*2+1:i*2+3], 16) for j in range(n_ramp)]
                interp = np.interp(x_new, x, component)
                colors.append(interp)
            
            # Convert back to hex colors
            hex_colors = []
            for i in range(n_classes):
                r, g, b = [int(colors[j][i]) for j in range(3)]
                hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
            return hex_colors

    def _update_categorized_field(self, layer_name, field):
        """Update the categorized symbology field"""
        if not field:
            return
            
        layer_data = self.layers[layer_name]
        layer_data['categorized_field'] = field
        
        # Get unique values and assign colors
        unique_values = layer_data['gdf'][field].unique()
        ramp_name = layer_data.get('color_ramp', 'Spectral')
        
        # Store color mapping
        layer_data['category_colors'] = {
            val: self._get_category_color(val, ramp_name) 
            for val in unique_values
        }
        
        self.redraw_all_layers()

    def _get_category_color(self, value, ramp_name='Spectral'):
        """Generate a consistent color for a category value"""
        # Create a consistent hash value for this category
        hash_val = hash(str(value)) % 100
        
        # Get the color ramp
        ramp = self._get_color_ramp(ramp_name, 100)  # Use large number for variety
        
        # Select color based on hash
        return ramp[hash_val % len(ramp)]


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def toggle_google_maps(self):
        """Switch between the image view and a web map (Esri / OSM / Google basemap) that shows the application's
        georeferenced layers as overlays, opened at the extent currently displayed."""
        try:
            if self.google_maps_enabled:
                self.google_maps_enabled = False
                self.stacked_widget.setCurrentWidget(self.image_display_widget)
                self.statusBar().showMessage("Image view", 3000)
                return
            if QWebEngineView is None:
                QMessageBox.warning(self, "Web map", "The web map needs PyQtWebEngine:\n    pip install PyQtWebEngine")
                return
            chosen = self._choose_web_layers()
            if chosen is None:
                return                                   # cancelled
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                html, notes, n_over = self._build_web_map_html(selected=chosen)
            finally:
                QApplication.restoreOverrideCursor()
            if n_over == 0 and not chosen:
                notes = ["No layer selected: showing the basemap only."] + notes
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as fh:
                fh.write(html)
                path = fh.name
            self._web_map_file = path
            self.google_maps_view.load(QUrl.fromLocalFile(path))
            self.stacked_widget.setCurrentWidget(self.google_maps_view)
            self.google_maps_enabled = True
            self.statusBar().showMessage(f"Web map: {n_over} layer(s) as overlays (internet needed for the basemap)", 6000)
            if notes:
                QMessageBox.information(self, "Web map", "\n".join(notes[:8]))
        except Exception as e:
            self.google_maps_enabled = False
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Failed to open the web map: {str(e)}")

    def _web_layer_candidates(self):
        """[(name, kind, ok, reason)] for every layer: rasters need a transform and CRS, vectors need a CRS."""
        out = []
        for name, layer in self.layers.items():
            if not isinstance(layer, dict):
                continue
            if layer.get('image') is not None:
                prof = layer.get('profile') or {}
                ok = prof.get('crs') is not None and georef.is_georeferenced(prof.get('transform'), prof.get('crs'))
                out.append((name, 'raster', ok, '' if ok else 'not georeferenced'))
            elif layer.get('gdf') is not None:
                ok = getattr(layer['gdf'], 'crs', None) is not None
                out.append((name, 'vector', ok, '' if ok else 'no CRS'))
        return out

    def _choose_web_layers(self):
        """Dialog with a check box per layer. Returns the set of chosen layer names, or None if cancelled."""
        cands = self._web_layer_candidates()
        dlg = QDialog(self)
        dlg.setWindowTitle("Layers to overlay on the web map")
        dlg.setMinimumWidth(420)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Tick the layers to draw over the basemap (later layers are drawn on top).\n"
                             "You can still hide them or change their opacity on the map page."))
        lst = QListWidget()
        current = getattr(self, 'current_layer', None)
        for name, kind, ok, why in cands:
            it = QListWidgetItem(f"{name}   [{kind}]" + ("" if ok else f"   ({why})"))
            it.setData(Qt.UserRole, name)
            if ok:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if (name == current or kind == 'vector') else Qt.Unchecked)
            else:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
            lst.addItem(it)
        lay.addWidget(lst)
        row = QHBoxLayout()
        b_all, b_none = QPushButton("Select all"), QPushButton("Select none")
        row.addWidget(b_all); row.addWidget(b_none); row.addStretch(1)
        lay.addLayout(row)

        def set_all(state):
            for i in range(lst.count()):
                if lst.item(i).flags() & Qt.ItemIsUserCheckable:
                    lst.item(i).setCheckState(state)
        b_all.clicked.connect(lambda: set_all(Qt.Checked))
        b_none.clicked.connect(lambda: set_all(Qt.Unchecked))
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(dlg.accept); box.rejected.connect(dlg.reject)
        lay.addWidget(box)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return {lst.item(i).data(Qt.UserRole) for i in range(lst.count())
                if (lst.item(i).flags() & Qt.ItemIsUserCheckable) and lst.item(i).checkState() == Qt.Checked}

    def _build_web_map_html(self, selected=None, max_rasters=12, max_features=20000):
        """(html, notes, number of raster overlays). The displayed layer is checked, the others can be ticked on the page."""
        from rasterio.warp import transform_bounds
        notes, overlays, vectors = [], [], []
        current = getattr(self, 'current_layer', None)
        for name in list(self.layers):                         # list order = drawing order (later layers on top)
            layer = self.layers[name]
            if not isinstance(layer, dict) or (selected is not None and name not in selected):
                continue
            if layer.get('image') is not None:
                if len(overlays) >= max_rasters:
                    notes.append(f"Only the first {max_rasters} raster layers are included ({name} skipped).")
                    continue
                prof = layer.get('profile') or {}
                tr, crs = prof.get('transform'), prof.get('crs')
                if crs is None or not georef.is_georeferenced(tr, crs):
                    notes.append(f"'{name}' skipped: not georeferenced (no transform/CRS).")
                    continue
                try:
                    img = np.asarray(layer['image'])
                    dec, stride = webmap.decimate(img)
                    rgba = webmap.render_rgba(dec, layer.get('cmap', 'viridis'),
                                              *(layer.get('display_range') or (layer.get('vmin'), layer.get('vmax'))),
                                              nodata=prof.get('nodata'))
                    rgba, bounds = webmap.to_web_mercator(rgba, tr * tr.scale(stride), crs)
                    overlays.append({'name': name, 'uri': webmap.png_data_uri(rgba), 'bounds': bounds,
                                     'checked': True})
                except Exception as exc:
                    notes.append(f"'{name}' skipped: {exc}")
            elif layer.get('gdf') is not None:
                try:
                    gdf = layer['gdf']
                    if len(gdf) > max_features:
                        notes.append(f"'{name}' skipped: {len(gdf):,} features (limit {max_features:,}).")
                        continue
                    if gdf.crs is None:
                        notes.append(f"'{name}' skipped: no CRS.")
                        continue
                    vectors.append({'name': name, 'geojson': json.loads(gdf.to_crs(4326)[['geometry']].to_json()),
                                    'checked': True})
                except Exception as exc:
                    notes.append(f"'{name}' skipped: {exc}")
        # open at the extent currently shown (zoomed/panned) when the displayed layer is georeferenced
        view = None
        try:
            shown = self.layers.get(current) or {}
            sp = shown.get('profile') or {}
            if shown.get('image') is not None and sp.get('crs') is not None and georef.is_georeferenced(sp.get('transform'), sp.get('crs')):
                x0, x1 = sorted(self.ax.get_xlim())
                y0, y1 = sorted(self.ax.get_ylim())
                w, s, e, n = transform_bounds(sp['crs'], 'EPSG:4326', x0, y0, x1, y1)
                view = (s, w, n, e)
        except Exception:
            view = None
        if view is None and overlays:
            view = (min(o['bounds'][0] for o in overlays), min(o['bounds'][1] for o in overlays),
                    max(o['bounds'][2] for o in overlays), max(o['bounds'][3] for o in overlays))
        return webmap.build_html(overlays, vectors, view, opacity=0.7), notes, len(overlays)

##########################################################################################################3333
    # def validate_hex_color(color):
    #     """
    #     Validates a hex color code.        
    #     Parameters:
    #         color (str): The color code to validate.        
    #     Returns:
    #         bool: True if the color code is valid, False otherwise.        """
    #     if not re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color):
    #         raise ValueError(f"Invalid hex color code: {color}")
    #     return True
    # # Example usage
    # try:
    #     color = "#1a2b3c"
    #     validate_hex_color(color)
    #     print("Valid color code")
    # except ValueError as e:
    #     print(e)
    
################################################################################################################
##########################################################################################################

    # def load_image(self):
    #     file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "GeoTIFF (*.tif *.tiff)")
    #     if file_name:
    #         try:
    #             with rasterio.open(file_name) as src:
    #                 image = src.read()
    #                 profile = dict(src.profile)  # rasterio's Profile is a UserDict, not a dict
    #                 layer_name = os.path.basename(file_name)

    #                 if image.shape[0] > 3:
    #                     selected_bands = self.choose_bands(image.shape[0])
    #                     if not selected_bands:
    #                         return
    #                     image = image[[b-1 for b in selected_bands]]

    #                 self.layers[layer_name] = {
    #                     'image': image,
    #                     'profile': profile,
    #                     'file_name': file_name
    #                 }

    #                 item = QListWidgetItem(layer_name)
    #                 self.layer_list.addItem(item)

    #                 self.image = image
    #                 self.profile = profile
    #                 self.image_path = file_name
    #                 self.active_image = self.image
    #                 self.current_layer = layer_name
    #                 self.display_raster_layer(self.layers[layer_name], layer_name)
                    
    #                 self.update_info_box(layer_data=self.layers[layer_name])

    #         except Exception as e:
    #             QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    def load_image(self):
        # Supported raster and vector formats for GIS and satellite imagery
        file_formats = (
            "All Supported Formats (*.tif *.tiff *.img *.hdf *.h5 *.hdf4 *.hdf5 *.nc *.jp2 *.j2k *.dat *.bsq *.bil *.bip *.png *.jpg *.jpeg *.kmz *.kml);;"
            "GeoTIFF (*.tif *.tiff);;"
            "ERDAS IMG (*.img);;"
            "HDF4/HDF5 (*.hdf *.h4 *.hdf4 *.h5 *.hdf5);;"
            "NetCDF (*.nc);;"
            "JPEG 2000 (*.jp2 *.j2k);;"
            "ENVI (*.dat *.bsq *.bil *.bip);;"
            "Keyhole Markup (*.kmz *.kml);;"
            "Common Image Formats (*.png *.jpg *.jpeg)"
        )
        
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open GIS File", 
            "", 
            file_formats
        )
        
        if not file_name:
            return
            
        try:
            # Handle KMZ/KML files differently as they're vector formats
            if file_name.lower().endswith(('.kml', '.kmz')):
                self.load_vector_file(file_name)
                return
                
            # Original raster handling
            with rasterio.open(file_name) as src:
                image = src.read()
                profile = dict(src.profile)  # rasterio's Profile is a UserDict, not a dict
                layer_name = os.path.basename(file_name)

                # Handle multi-band images
                if len(image.shape) > 2 and image.shape[0] > 3:
                    selected_bands = self.choose_bands(image.shape[0])
                    if not selected_bands:
                        return
                    image = image[[b-1 for b in selected_bands]]
                elif len(image.shape) == 2:
                    # Add channel dimension for single-band images
                    image = image[np.newaxis, ...]

                # Store layer data
                self.layers[layer_name] = {
                    'image': image,
                    'profile': profile,
                    'file_name': file_name,
                    'crs': src.crs.to_string() if src.crs else None,
                    'bounds': src.bounds,
                    'type': 'raster'
                }

                # Update UI
                item = QListWidgetItem(layer_name)
                self.layer_list.addItem(item)

                self.image = image
                self.profile = profile
                self.image_path = file_name
                self.active_image = self.image
                self.current_layer = layer_name
                self.display_raster_layer(self.layers[layer_name], layer_name)
                
                self.update_info_box(layer_data=self.layers[layer_name])

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to load file:\n{str(e)}\n\n"
                f"File: {os.path.basename(file_name)}"
            )


    def load_vector(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Vector File", "",
            "Vector Files (*.shp *.geojson *.json *.gpkg *.kml *.gml);;All Files (*)"
        )
        if file_name:
            self.load_vector_file(file_name)

    def load_vector_file(self, file_name):
        """Read a vector file with GeoPandas and add it as a layer. Shows an error dialog instead of raising."""
        try:
            if file_name.lower().endswith('.kmz'):
                raise ValueError("KMZ is a zipped KML. Unzip it and open the .kml inside "
                                 "(or convert it to GeoPackage in QGIS).")
            gdf = gpd.read_file(file_name)
            if gdf.empty or 'geometry' not in gdf.columns:
                self.show_warning("Invalid vector file: empty or missing geometry")
                return
            self.polygon_processor.current_gdf = gdf
            self.polygon_processor.original_file_path = file_name
            self.polygon_processor.editing_active = False

            layer_name = os.path.basename(file_name)
            layer_data = {
                'type': 'vector',
                'gdf': gdf,
                'file_name': file_name,
                'style': {'color': '#FF0000', 'width': 1}
            }
            self.layers[layer_name] = layer_data
            item = QListWidgetItem(layer_name)
            item.setIcon(QIcon(":/icons/vector.png"))
            self.layer_list.addItem(item)
            self.display_vector_layer(layer_data, layer_name)
            self.update_info_box(layer_data=self.layers[layer_name])
            self.show_info(f"Successfully loaded {layer_name}")
        except Exception as exc:
            logging.exception("Vector load failed")
            self.show_error(f"Load failed: {exc}")


        

 
 ########################################## save image as functions




    
    


 ##################################################################
    
    UNDO_LIMIT = 10

    def _current_layer_name(self):
        name = getattr(self, 'current_layer', None)
        if name:
            return name
        item = self.layer_list.currentItem() if hasattr(self, 'layer_list') else None
        return item.text() if item else "Image"

    def _push_undo(self):
        """Save the active image as (image, layer_name) and invalidate the redo history."""
        image = getattr(self, 'image', None)
        if image is None:
            return
        self.undo_stack.append((np.array(image, copy=True), self._current_layer_name()))
        del self.undo_stack[:-self.UNDO_LIMIT]
        self.redo_stack.clear()

    def _restore_state(self, image, name):
        self.image = image
        layer = self.layers.get(name)
        if isinstance(layer, dict):
            layer['image'] = image
        else:
            self.layers[name] = {'image': image, 'profile': getattr(self, 'profile', None) or {}, 'file_name': name}
        self.display_raster_layer(self.layers[name], name)

    def undo(self):
        if not self.undo_stack:
            return
        image, name = self.undo_stack.pop()
        if getattr(self, 'image', None) is not None:
            self.redo_stack.append((np.array(self.image, copy=True), self._current_layer_name()))
        self._restore_state(image, name)

    def redo(self):
        if not self.redo_stack:
            return
        image, name = self.redo_stack.pop()
        if getattr(self, 'image', None) is not None:
            self.undo_stack.append((np.array(self.image, copy=True), self._current_layer_name()))
        self._restore_state(image, name)


    def resample_image(self):
        if hasattr(self, 'image'):  # التأكد من وجود صورة نشطة
            try:
                scale_factor, ok = QInputDialog.getDouble(self, "Scale Factor", 
                                                        "Enter scale factor (e.g., 2):", 
                                                        1.0, 0.1, 10.0, 1)
                if ok:
                    self.progress_bar.setVisible(True)  # Show progress bar
                    self.worker = Worker(self.image_processor.resample_image, 
                                        self.image, self.profile, scale_factor, self.image_path)  # Pass image_path
                    self.worker.progress.connect(self.update_progress)  # Connect progress signal
                    self.worker.finished.connect(self.on_resample_finished)
                    self.worker.start()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to resample image: {str(e)}")
        else:
            QMessageBox.warning(self, "No Image", "No image loaded. Please load an image first.")

    def on_resample_finished(self, result):
        try:
            self.resampled_image, self.new_profile, file_name = result
            self._push_undo()
            self.display_raster_layer(self.resampled_image, file_name)  # Pass file_name to display_image
            self.progress_bar.setVisible(False)  # Hide progress bar after completion

            # Ask user for confirmation to save the resampled image
            save_reply = QMessageBox.question(
                self, "Save Resampled Image", "Do you want to save the resampled image?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if save_reply == QMessageBox.Yes:
                # Save the resampled image
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Resampled Image", "", "Images (*.tif *.tiff)")
                if file_name:
                    try:
                        self.image_processor.save_image(self.resampled_image, self.new_profile, file_name)
                        QMessageBox.information(self, "Success", f"Resampled image saved as '{file_name}'")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
            else:
                QMessageBox.information(self, "Cancelled", "Resampled image saving cancelled by user")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to finish resampling: {str(e)}")

    
               
    def enhance_contrast(self):
        if hasattr(self, 'image'):  # التأكد من وجود صورة نشطة
            try:
                self.progress_bar.setVisible(True)  # Show progress bar
                self.worker = Worker(self.image_processor.enhance_contrast, self.image, self.image_path)  # Pass image_path
                self.worker.progress.connect(self.update_progress)  # Connect progress signal
                self.worker.finished.connect(self.on_contrast_enhance_finished)
                self.worker.start()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start enhance_contrast operation: {str(e)}")
        else:
            QMessageBox.warning(self, "No Image", "No image loaded. Please load an image first.")

    def on_contrast_enhance_finished(self, result):
        try:
            self.contrast_enhanced_image, file_name = result
            self._push_undo()
            self.display_raster_layer(self.contrast_enhanced_image, file_name)  # Pass file_name to display_image
            self.progress_bar.setVisible(False)  # Hide progress bar after completion

            # Ask user for confirmation to save the contrast-enhanced image
            save_reply = QMessageBox.question(
                self, "Save Contrast-Enhanced Image", "Do you want to save the contrast-enhanced image?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if save_reply == QMessageBox.Yes:
                # Save the contrast-enhanced image
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Contrast-Enhanced Image", "", "Images (*.tif *.tiff)")
                if file_name:
                    try:
                        self.image_processor.save_image(self.contrast_enhanced_image, self.profile, file_name)
                        QMessageBox.information(self, "Success", f"Contrast-enhanced image saved as '{file_name}'")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
            else:
                QMessageBox.information(self, "Cancelled", "Contrast-enhanced image saving cancelled by user")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to finish contrast enhancement: {str(e)}")

    def smooth_image(self):
        if hasattr(self, 'image'):  # التأكد من وجود صورة نشطة
            try:
                self.progress_bar.setVisible(True)  # Show progress bar
                self.worker = Worker(self.image_processor.smooth_image, self.image, 1, self.image_path)  # Pass image_path
                self.worker.progress.connect(self.update_progress)  # Connect progress signal
                self.worker.finished.connect(self.on_smooth_finished)
                self.worker.start()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start smooth_image operation: {str(e)}")
        else:
            QMessageBox.warning(self, "No Image", "No image loaded. Please load an image first.")

    def on_smooth_finished(self, result):
        try:
            self.smoothed_image, file_name = result
            self._push_undo()
            self.display_raster_layer(self.smoothed_image, file_name)  # Pass file_name to display_image
            self.progress_bar.setVisible(False)  # Hide progress bar after completion

            # Ask user for confirmation to save the smoothed image
            save_reply = QMessageBox.question(
                self, "Save Smoothed Image", "Do you want to save the smoothed image?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if save_reply == QMessageBox.Yes:
                # Save the smoothed image
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Smoothed Image", "", "Images (*.tif *.tiff)")
                if file_name:
                    try:
                        self.image_processor.save_image(self.smoothed_image, self.profile, file_name)
                        QMessageBox.information(self, "Success", f"Smoothed image saved as '{file_name}'")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
            else:
                QMessageBox.information(self, "Cancelled", "Smoothed image saving cancelled by user")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to finish smoothing: {str(e)}")

        
        
    def update_progress(self, value):
        if value == -1:  # Indeterminate mode
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setVisible(True)
        elif value == -2:  # Error state
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", "Processing failed")
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)
            if value >= 100:
                self.progress_bar.setVisible(False)

    def sharpen_image(self):
        if hasattr(self, 'image'):  # التأكد من وجود صورة نشطة
            try:
                self.progress_bar.setVisible(True)  # Show progress bar
                self.worker = Worker(self.image_processor.sharpen_image, self.image, 1, 1, self.image_path)  # Pass image_path
                self.worker.progress.connect(self.update_progress)  # Connect progress signal
                self.worker.finished.connect(self.on_sharpen_finished)
                self.worker.start()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start sharpen_image operation: {str(e)}")
        else:
            QMessageBox.warning(self, "No Image", "No image loaded. Please load an image first.")

    def on_sharpen_finished(self, result):
        try:
            self.sharpened_image, file_name = result
            self._push_undo()
            self.display_raster_layer(self.sharpened_image, file_name)  # Pass file_name to display_image
            self.progress_bar.setVisible(False)  # Hide progress bar after completion

            # Ask user for confirmation to save the sharpened image
            save_reply = QMessageBox.question(
                self, "Save Sharpened Image", "Do you want to save the sharpened image?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if save_reply == QMessageBox.Yes:
                # Save the sharpened image
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Sharpened Image", "", "Images (*.tif *.tiff)")
                if file_name:
                    try:
                        self.image_processor.save_image(self.sharpened_image, self.profile, file_name)
                        QMessageBox.information(self, "Success", f"Sharpened image saved as '{file_name}'")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
            else:
                QMessageBox.information(self, "Cancelled", "Sharpened image saving cancelled by user")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to finish sharpening: {str(e)}")

    def validate_dem_layer(self):
        if not self.layer_list.currentItem():
            QMessageBox.warning(self, "Error", "No layer selected!")
            return None, None
            
        layer_name = self.layer_list.currentItem().text()
        if layer_name not in self.layers:
            QMessageBox.warning(self, "Error", "Invalid layer selection!")
            return None, None
            
        layer_data = self.layers[layer_name]['image']
        if layer_data.shape[0] != 1:
            QMessageBox.warning(self, "Error", "DEM must be a single-band layer!")
            return None, None
            
        return layer_data[0], self.layers[layer_name]['profile']
    
 ##################################################### DEM operations
    def _prepare_dem(self, dem_data, profile):
        """float32 DEM with no-data sentinels as NaN and voids of up to 3 pixels filled (vectorised, no elevation limits)."""
        dem = terrain.clean_dem(np.squeeze(dem_data), (profile or {}).get('nodata'))
        if np.isnan(dem).all():
            raise ValueError("The DEM has no valid elevation cells")
        return terrain.fill_small_voids(dem, 3)

    def preprocess_dem(self, dem_data, profile):
        """Clean the DEM and optionally smooth it. Returns (dem, new_profile) or (None, None) after showing the reason."""
        try:
            dem = self._prepare_dem(dem_data, profile)
            if np.nanmax(dem) == np.nanmin(dem):
                QMessageBox.warning(self, "Warning", "The DEM is completely flat")
                return None, None
            smoothing_applied = False
            reply = QMessageBox.question(
                self, 'DEM Smoothing',
                'Apply gentle Gaussian smoothing (sigma = 0.75 pixel) to the DEM?\n\n'
                'Smoothing reduces noise but removes some detail.',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                dem = terrain.smooth_nan(dem, 0.75)
                smoothing_applied = True
            new_profile = dict(profile)
            new_profile.update({'nodata': float('nan'), 'dtype': 'float32', 'processed': True,
                                'smoothing_applied': smoothing_applied})
            return dem, new_profile
        except Exception as exc:
            QMessageBox.critical(self, "Preprocessing Error", f"DEM cleaning failed: {exc}")
            return None, None


    def calculate_hillshade(self):
        """Calculate hillshade with proper visualization"""
        try:
            dem_data, profile = self.validate_dem_layer()
            if dem_data is None:
                return
            
            # Check if CRS is projected
            # Preprocess DEM
            dem_processed, new_profile = self.preprocess_dem(dem_data, profile)
            if dem_processed is None:
                return

            # Get parameters
            azimuth, ok1 = QInputDialog.getDouble(self, "Azimuth", 
                "Sun azimuth angle (degrees):", 315, 0, 360)
            altitude, ok2 = QInputDialog.getDouble(self, "Altitude", 
                "Sun altitude angle (degrees):", 45, 0, 90)
            z_factor, ok3 = QInputDialog.getDouble(self, "Z Factor", 
                "Vertical exaggeration:", 1, 0.1, 10)

            if not all((ok1, ok2, ok3)):
                return

            # Calculate hillshade (0-255 range)
            hillshade = self.image_processor.calculate_hillshade(
                dem_processed, 
                new_profile['transform'], 
                azimuth, 
                altitude, 
                z_factor,
                crs=new_profile.get('crs')
            )
            
            # Create layer with visualization preferences
            layer_name = f"Hillshade_{azimuth}°_{altitude}°"
            self.layers[layer_name] = {
                'image': hillshade[np.newaxis, ...],  # Add batch dimension
                'profile': new_profile,
                'file_name': "Hillshade",
                'cmap': 'gray',  # Store preferred colormap
                'vmin': 0,       # Store value ranges
                'vmax': 255
            }
            
            # Add to layer list
            self.layer_list.addItem(QListWidgetItem(layer_name))
            
            # Display with proper parameters
            self.display_raster_layer(
                hillshade[np.newaxis, ...],  # Ensure 3D array
                layer_name
            )
            # Show statistics and ask about saving
            stats = (f"Hillshade Statistics:\n"
                    f"Min brightness: {np.nanmin(hillshade):.2f}\n"
                    f"Max brightness: {np.nanmax(hillshade):.2f}\n"
                    f"Mean brightness: {np.nanmean(hillshade):.2f}\n\n"
                    f"Parameters:\n"
                    f"Azimuth: {azimuth}°\n"
                    f"Altitude: {altitude}°\n"
                    f"Z-factor: {z_factor}")
            
            msg = QMessageBox()
            msg.setWindowTitle("Hillshade Calculation Complete")
            msg.setText(stats)
            msg.setInformativeText("Would you like to save the hillshade results?")
            msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
            msg.setDefaultButton(QMessageBox.Save)
            
            if msg.exec_() == QMessageBox.Save:
                self.save_raster_results(hillshade, new_profile, "hillshade", 
                                    f"hillshade_az{azimuth}_alt{altitude}_z{z_factor}")
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                f"Hillshade calculation failed:\n{str(e)}\n\n"
                "Common issues:\n"
                "1. Invalid DEM values\n"
                "2. Incorrect CRS units\n"
                "3. Extreme elevation values")
        
    def calculate_slope(self):
        """Calculate slope with proper preprocessing, CRS check, and save options"""
        try:
            dem_raw, new_profile = self.validate_dem_layer()
            if dem_raw is None:
                return
            dem_processed = self._prepare_dem(dem_raw, new_profile)

            # Check if CRS is projected
            # Create parameter selection dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Slope Parameters")
            layout = QVBoxLayout()
            
            # DEM Preview
            fig = Figure(figsize=(7, 6))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            preview, _spec = georef.draw_raster(ax, dem_processed, new_profile, cmap='terrain')
            fig.colorbar(preview, ax=ax, label='Elevation (m)')
            ax.set_title("Processed DEM Preview")
            layout.addWidget(canvas)
            
            # Output type selection
            output_group = QButtonGroup()
            rb_degrees = QRadioButton("Degrees (0-90°)")
            rb_percent = QRadioButton("Percent Rise (0-∞%)")
            rb_both = QRadioButton("Both")
            rb_degrees.setChecked(True)
            
            output_group.addButton(rb_degrees)
            output_group.addButton(rb_percent)
            output_group.addButton(rb_both)
            
            layout.addWidget(QLabel("Output Type:"))
            layout.addWidget(rb_degrees)
            layout.addWidget(rb_percent)
            layout.addWidget(rb_both)
            
            # Dialog buttons
            btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btn_box.accepted.connect(dialog.accept)
            btn_box.rejected.connect(dialog.reject)
            layout.addWidget(btn_box)
            
            dialog.setLayout(layout)
            
            if dialog.exec_() != QDialog.Accepted:
                return
                
            # Determine output type
            output_type = ('degrees' if rb_degrees.isChecked() else
                        'percent_rise' if rb_percent.isChecked() else 
                        'both')

            # Calculate slope
            result = self.image_processor.calculate_slope(
                dem_processed,
                new_profile['transform'],
                output_type,
                crs=new_profile.get('crs')
            )
            
            # Handle output
            if output_type == 'both':
                degrees, percent = result
                
                # Create layers for both outputs
                for name, data, cmap, vmin, vmax in [
                    ("Slope_Degrees", degrees, 'viridis', 0, 90),
                    ("Slope_Percent", percent, 'RdYlGn_r', 0, 1000)
                ]:
                    self.layers[name] = {
                        'image': data[np.newaxis, ...],
                        'profile': new_profile,
                        'file_name': name,
                        'cmap': cmap,
                        'vmin': vmin,
                        'vmax': vmax
                    }
                    self.layer_list.addItem(QListWidgetItem(QIcon(f":/icons/{name.lower()}.png"), name))
                    self.display_raster_layer(data, name)
                    
                # Show statistics
                stats = (f"Slope Statistics:\n"
                        f"Degrees: {np.nanmin(degrees):.2f}° to {np.nanmax(degrees):.2f}°\n"
                        f"Percent: {np.nanmin(percent):.2f}% to {np.nanmax(percent):.2f}%")
                
                # Ask about saving
                msg = QMessageBox()
                msg.setWindowTitle("Slope Calculation Complete")
                msg.setText(stats)
                msg.setInformativeText("Would you like to save the results?")
                msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
                msg.setDefaultButton(QMessageBox.Save)
                
                if msg.exec_() == QMessageBox.Save:
                    self.save_slope_results(degrees, percent, new_profile)
                    
            else:  # Single output
                output_name = "Slope_Degrees" if output_type == 'degrees' else "Slope_Percent"
                output_data = result
                cmap = 'viridis' if output_type == 'degrees' else 'RdYlGn_r'
                vmin, vmax = (0, 90) if output_type == 'degrees' else (0, 1000)
                
                self.layers[output_name] = {
                    'image': output_data[np.newaxis, ...],
                    'profile': new_profile,
                    'file_name': output_name,
                    'cmap': cmap,
                    'vmin': vmin,
                    'vmax': vmax
                }
                self.layer_list.addItem(QListWidgetItem(QIcon(f":/icons/{output_name.lower()}.png"), output_name))
                self.display_raster_layer(output_data, output_name)
                
                # Ask about saving
                msg = QMessageBox()
                msg.setWindowTitle("Slope Calculation Complete")
                msg.setText(f"{output_name.replace('_', ' ')} calculated successfully")
                msg.setInformativeText("Would you like to save the results?")
                msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
                msg.setDefaultButton(QMessageBox.Save)
                
                if msg.exec_() == QMessageBox.Save:
                    self.save_slope_results(output_data, None, new_profile, output_type)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                f"Slope calculation failed:\n{str(e)}\n\n"
                "Possible fixes:\n"
                "1. Check DEM has valid elevation values\n"
                "2. Verify CRS is projected\n"
                "3. Try cleaning DEM first")
            
    def save_slope_results(self, degrees_data, percent_data, profile, output_type='both'):
        """Save slope results to file(s)"""
        try:
            options = QFileDialog.Options()
            base_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Slope Results",
                "",
                "GeoTIFF Files (*.tif);;All Files (*)",
                options=options
            )
            
            if not base_path:
                return  # User cancelled
                
            base_path = os.path.splitext(base_path)[0]  # Remove extension if present
            
            # Prepare the profile for saving
            save_profile = profile.copy()
            save_profile.update({
                'driver': 'GTiff',
                'dtype': degrees_data.dtype if degrees_data is not None else percent_data.dtype,
                'count': 1,
                'nodata': None
            })
            
            save_profile.update(dtype='float32', nodata=float('nan'))
            to_write = []
            if output_type == 'both':
                to_write = [(f"{base_path}_degrees.tif", degrees_data), (f"{base_path}_percent.tif", percent_data)]
            elif output_type == 'degrees':
                to_write = [(f"{base_path}_degrees.tif", degrees_data)]
            else:      # percent_rise: the single result arrives in the first argument
                to_write = [(f"{base_path}_percent.tif", percent_data if percent_data is not None else degrees_data)]
            for path, arr in to_write:
                with rasterio.open(path, 'w', **save_profile) as dst:
                    dst.write(np.asarray(arr, dtype=np.float32), 1)
                
            QMessageBox.information(self, "Success", "Slope results saved successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save results: {str(e)}")

    def calculate_aspect(self):
        """Calculate aspect with proper visualization"""
        try:
            dem_data, profile = self.validate_dem_layer()
            if dem_data is None:
                return
            
            # Check if CRS is projected
            # Preprocess DEM
            dem_processed, new_profile = self.preprocess_dem(dem_data, profile)
            if dem_processed is None:
                return

            # Calculate aspect (0-360 degrees)
            aspect = self.image_processor.calculate_aspect(
                dem_processed, 
                new_profile['transform'],
                crs=new_profile.get('crs')
            )
            
            # Create layer with visualization preferences
            layer_name = "Aspect"
            self.layers[layer_name] = {
                'image': aspect[np.newaxis, ...],  # Add batch dimension
                'profile': new_profile,
                'file_name': "Aspect Directions",
                'cmap': 'twilight_shifted',  # Cyclic colormap for aspect
                'vmin': 0,                    # Minimum degrees
                'vmax': 360                    # Maximum degrees
            }
            
            # Add to layer list with compass icon
            item = QListWidgetItem(QIcon(resource_path("compass.png")), layer_name)
            self.layer_list.addItem(item)
            
            # Display using parameters from layer metadata
            self.display_raster_layer(aspect, layer_name)

            # Show statistics and ask about saving
            stats = (f"Aspect Statistics:\n"
                    f"Minimum: {np.nanmin(aspect):.2f}°\n"
                    f"Maximum: {np.nanmax(aspect):.2f}°\n"
                    f"Mean: {np.nanmean(aspect):.2f}°\n\n"
                    "Note: Aspect is measured clockwise from north")
            
            msg = QMessageBox()
            msg.setWindowTitle("Aspect Calculation Complete")
            msg.setText(stats)
            msg.setInformativeText("Would you like to save the aspect results?")
            msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
            msg.setDefaultButton(QMessageBox.Save)
            
            if msg.exec_() == QMessageBox.Save:
                self.save_raster_results(aspect, new_profile, "aspect", "aspect_directions")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                f"Aspect calculation failed:\n{str(e)}\n\n"
                "Common issues:\n"
                "1. Flat areas in DEM (produces NaN values)\n"
                "2. NoData values not properly handled\n"
                "3. Coordinate system issues (requires projected CRS)")
            
    def save_raster_results(self, data, profile, data_type, default_name):
        """Generic method to save raster results"""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                f"Save {data_type.title()} Results",
                f"{default_name}.tif",
                "GeoTIFF Files (*.tif);;All Files (*)",
                options=options
            )
            
            if not file_path:
                return  # User cancelled
                
            # Prepare the profile for saving
            save_profile = profile.copy()
            save_profile.update({
                'driver': 'GTiff',
                'dtype': data.dtype,
                'count': 1,
                'nodata': None
            })
            
            # Save the data
            with rasterio.open(file_path, 'w', **save_profile) as dst:
                dst.write(data, 1)
                
            QMessageBox.information(self, "Success", 
                                f"{data_type.title()} results saved successfully to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", 
                            f"Failed to save {data_type} results: {str(e)}")
            
    def _dem_for_hydrology(self):
        """(clean DEM, profile) of the selected single-band layer, or (None, None) after showing the reason."""
        dem_raw, profile = self.validate_dem_layer()
        if dem_raw is None:
            return None, None
        if not isinstance(profile, Mapping) or profile.get('transform') is None:
            name = self.layer_list.currentItem().text() if self.layer_list.currentItem() else "?"
            QMessageBox.warning(self, "Georeferencing missing",
                                f"The selected layer '{name}' has no geotransform (it is not georeferenced), so cell sizes "
                                "cannot be derived.\nSelect the original DEM GeoTIFF layer in the layer list and try again.")
            return None, None
        return self._prepare_dem(dem_raw, profile), profile

    @staticmethod
    def _flow_job(dem, transform, crs):
        """Depression filling -> D8 direction -> accumulation. Returns a dict (runs in a worker thread)."""
        def job(progress):
            progress(5, "Filling depressions (Priority-Flood)...")
            filled = terrain.fill_depressions(dem)
            progress(45, "Flow direction (D8)...")
            xres, yres = terrain.cell_size(transform, dem.shape, crs)
            flow_dir = terrain.flow_direction(filled, xres, yres)
            progress(75, "Flow accumulation...")
            return {'filled': filled, 'flow_dir': flow_dir, 'acc': terrain.flow_accumulation(flow_dir)}
        return job

    def calculate_flow_accumulation(self):
        """Flow accumulation (number of upstream cells) from the selected DEM layer."""
        try:
            dem, profile = self._dem_for_hydrology()
            if dem is None:
                return
            if dem.size > 4_000_000:
                if QMessageBox.question(self, "Large DEM", f"{dem.size:,} cells: depression filling may take a minute. Continue?",
                                        QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                    return
            job = self._flow_job(dem, profile['transform'], profile.get('crs'))
            self._start_validation_job(job, lambda res: self._on_flow_done(res, profile), "Flow accumulation")
        except Exception as e:
            self.show_error(f"Flow accumulation failed: {e}")

    def _on_flow_done(self, res, profile):
        self._val_progress.close()
        acc = res['acc']
        log_acc = np.log1p(acc).astype(np.float32)
        new_profile = dict(profile)
        new_profile.update(dtype='float32', nodata=float('nan'), count=1)
        layer_name = "Flow_Accumulation"
        self.layers[layer_name] = {
            'image': log_acc[np.newaxis, ...], 'profile': new_profile, 'file_name': layer_name,
            'cmap': 'Blues', 'vmin': 0, 'vmax': float(np.percentile(log_acc[acc > 0], 99)) if (acc > 0).any() else 1.0,
        }
        self.layer_list.addItem(QListWidgetItem(layer_name))
        self.display_raster_layer(log_acc[np.newaxis, ...], layer_name)
        pool = acc[acc > 1]
        stats = (f"Flow accumulation (D8, Priority-Flood filled)\n"
                 f"Max contributing cells: {int(acc.max()):,}\n"
                 f"Cells draining out of the DEM (pits/outlets): {int((res['flow_dir'] == 0).sum()):,}\n"
                 f"Suggested stream threshold (95th percentile of flowing cells): "
                 f"{int(np.percentile(pool, 95)) if pool.size else 0:,} cells")
        if QMessageBox.question(self, "Flow analysis complete", stats + "\n\nSave the log-scaled raster?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.save_raster_results(log_acc, new_profile, "flow_accumulation", layer_name)

    def calculate_stream_network(self):
        """Stream network: flow accumulation -> threshold -> Strahler/Shreve order -> raster + line layers."""
        try:
            dem, profile = self._dem_for_hydrology()
            if dem is None:
                return
            job = self._flow_job(dem, profile['transform'], profile.get('crs'))
            self._start_validation_job(job, lambda res: self._on_stream_flow_done(res, profile), "Stream network: flow")
        except Exception as e:
            self.show_error(f"Stream extraction failed: {e}")

    def _on_stream_flow_done(self, res, profile):
        self._val_progress.close()
        try:
            flat = float((res['flow_dir'] == 0).sum()) / res['flow_dir'].size * 100.0
            if flat > 30 and QMessageBox.question(
                    self, "Many outlets", f"{flat:.1f}% of cells have no downstream neighbour. Results may be unreliable. Continue?",
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
            params = self._get_stream_parameters(res['acc'])
            if not params:
                return
            transform = profile['transform']

            def job(progress):
                progress(10, "Thresholding...")
                streams = terrain.extract_streams(res['acc'], params['method'], params['threshold'])
                if not streams.any():
                    raise ValueError("No stream cells: the threshold is too high")
                progress(40, f"{params['order_method']} stream order...")
                order = terrain.stream_order(res['flow_dir'], streams, params['order_method'].lower())
                progress(70, "Vectorising...")
                features = self.image_processor.streams_to_vector(order, res['flow_dir'], transform)
                return {'order': order, 'features': features, 'params': params}
            self._start_validation_job(job, lambda r: self._on_streams_done(r, profile), "Stream network")
        except Exception as e:
            self.show_error(f"Stream extraction failed: {e}")

    def _on_streams_done(self, r, profile):
        self._val_progress.close()
        try:
            self._create_stream_outputs(r['order'], r['features'], profile, r['params'])
            lines = r['features']
            length_m = 0.0
            if lines:
                gdf = self.layers["Stream_Lines"]['gdf']
                metric = gdf if (gdf.crs is not None and gdf.crs.is_projected) else gdf.to_crs(gdf.estimate_utm_crs())
                length_m = float(metric.length.sum())
            QMessageBox.information(
                self, "Stream network",
                f"{r['params']['order_method']} order up to {int(r['order'].max())}\n"
                f"{len(lines):,} line features, total length {length_m / 1000.0:,.2f} km")
        except Exception as e:
            self.show_error(f"Could not create the stream layers: {e}")

        

    def _get_stream_parameters(self, accumulation):
        """
        Collect stream extraction parameters via interactive dialog
        with automatic threshold suggestions and validation
        
        Args:
            accumulation: numpy array of flow accumulation values
            
        Returns:
            dict: {
                'method': 'percentile'|'fixed',
                'threshold': int,
                'order_method': 'Strahler'|'Shreve'
            } or None if canceled
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Stream Extraction Parameters")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout()
        
        # 1. Threshold Method Selection
        threshold_group = QGroupBox("Stream Threshold Method")
        threshold_layout = QVBoxLayout()
        
        # Automatic percentile-based
        auto_group = QHBoxLayout()
        rb_percentile = QRadioButton("Automatic (percentile)")
        rb_percentile.setChecked(True)
        self.percentile_spin = QSpinBox()
        self.percentile_spin.setRange(70, 99)
        self.percentile_spin.setValue(90)
        self.percentile_spin.setSuffix("%")
        auto_group.addWidget(rb_percentile)
        auto_group.addWidget(self.percentile_spin)
        
        # Manual fixed threshold
        manual_group = QHBoxLayout()
        rb_fixed = QRadioButton("Manual threshold:")
        self.fixed_spin = QSpinBox()
        self.fixed_spin.setRange(1, 1000000)
        
        # Calculate suggested threshold (95th percentile)
        valid_cells = accumulation[accumulation > 0]
        if len(valid_cells) > 0:
            suggested_threshold = int(np.percentile(valid_cells, 90))
            self.fixed_spin.setValue(suggested_threshold)
        
        manual_group.addWidget(rb_fixed)
        manual_group.addWidget(self.fixed_spin)
        
        threshold_layout.addLayout(auto_group)
        threshold_layout.addLayout(manual_group)
        threshold_group.setLayout(threshold_layout)
        
        # 2. Stream Order Method
        order_group = QGroupBox("Stream Ordering Method")
        order_layout = QVBoxLayout()
        self.order_combo = QComboBox()
        self.order_combo.addItems(["Strahler", "Shreve"])
        order_layout.addWidget(self.order_combo)
        order_group.setLayout(order_layout)
        
        # 3. Preview Statistics
        stats_group = QGroupBox("Basin Characteristics")
        stats_layout = QFormLayout()
        
        # Calculate basic stats
        if len(valid_cells) > 0:
            stats = {
                "Total cells": f"{accumulation.size:,}",
                "Flowing cells": f"{len(valid_cells):,}",
                "Max accumulation": f"{np.max(valid_cells):,}",
                "90th percentile": f"{int(np.percentile(valid_cells, 90)):,}",
                "95th percentile": f"{int(np.percentile(valid_cells, 95)):,}"
            }
            
            for label, value in stats.items():
                stats_layout.addRow(QLabel(label), QLabel(value))
        
        stats_group.setLayout(stats_layout)
        
        # 4. Dialog Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        
        # Layout Assembly
        layout.addWidget(threshold_group)
        layout.addWidget(order_group)
        layout.addWidget(stats_group)
        layout.addWidget(button_box)
        dialog.setLayout(layout)
        
        # Connect Signals
        def update_ui():
            self.percentile_spin.setEnabled(rb_percentile.isChecked())
            self.fixed_spin.setEnabled(rb_fixed.isChecked())
        
        rb_percentile.toggled.connect(update_ui)
        rb_fixed.toggled.connect(update_ui)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Initialize UI state
        update_ui()
        
        # Execute dialog
        if dialog.exec_() == QDialog.Accepted:
            return {
                'method': 'percentile' if rb_percentile.isChecked() else 'fixed',
                'threshold': self.percentile_spin.value() if rb_percentile.isChecked() else self.fixed_spin.value(),
                'order_method': self.order_combo.currentText()
            }
        return None
    
        # Set ArcGIS-like defaults
        self.percentile_spin.setValue(85)  # ArcGIS often uses ~85th percentile
        self.fixed_spin.setValue(1000)     # Common default threshold
        
        # Add validation
        def validate():
            if rb_fixed.isChecked() and self.fixed_spin.value() < 100:
                QMessageBox.warning(dialog, "Invalid Threshold", 
                                "Minimum recommended threshold is 100 cells")
                return False
            return True
            
        button_box.accepted.connect(lambda: dialog.accept() if validate() else None)
    
    def _create_stream_outputs(self, stream_order, features, profile, params):
        """Raster layer with the stream order and a vector layer with the stream lines."""
        raster_profile = dict(profile)
        raster_profile.update(dtype='float32', nodata=0.0, count=1)
        order_f = stream_order.astype(np.float32)
        self.layers["Stream_Order"] = {
            'image': order_f[np.newaxis, ...], 'profile': raster_profile,
            'file_name': f"Stream_Order_{params['method']}_{params['threshold']}",
            'cmap': 'viridis', 'vmin': 1, 'vmax': float(max(1, stream_order.max())),
        }
        self.layer_list.addItem(QListWidgetItem("Stream_Order"))
        self.display_raster_layer(order_f[np.newaxis, ...], "Stream_Order")
        if features:
            gdf = gpd.GeoDataFrame({'order': [f['properties']['order'] for f in features]},
                                   geometry=[f['geometry'] for f in features], crs=profile.get('crs'))
            layer_data = {'type': 'vector', 'gdf': gdf, 'file_name': 'Stream_Lines',
                          'style': {'color': '#1f77b4', 'width': 1}}
            self.layers["Stream_Lines"] = layer_data
            self.layer_list.addItem(QListWidgetItem("Stream_Lines"))
            self.display_vector_layer(layer_data, "Stream_Lines")




    
    
    # ================================================================== watershed / HAND / TWI
    def _add_derived_layer(self, name, data2d, profile, cmap='viridis', vmin=None, vmax=None, nodata=float('nan'), dtype='float32'):
        """Register a single-band result that keeps the DEM's georeferencing; replaces a layer of the same name."""
        prof = dict(profile)
        prof.update(dtype=dtype, nodata=nodata, count=1, height=data2d.shape[0], width=data2d.shape[1])
        layer = {'image': np.asarray(data2d)[np.newaxis, ...], 'profile': prof, 'file_name': name, 'cmap': cmap, 'type': 'raster'}
        if vmin is not None:
            layer['vmin'] = vmin
        if vmax is not None:
            layer['vmax'] = vmax
        is_new = name not in self.layers
        self.layers[name] = layer
        if is_new:
            self.layer_list.addItem(QListWidgetItem(name))
        self.display_raster_layer(layer, name)
        return layer

    def _with_flow(self, title, on_ready):
        """Run (or reuse) depression filling + D8 + accumulation for the selected DEM, then call
        ``on_ready(dem, profile, res)``. The result is cached per layer so several analyses do not repeat it."""
        dem, profile = self._dem_for_hydrology()
        if dem is None:
            return
        name = self.layer_list.currentItem().text()
        key = (name, id(self.layers[name]['image']))
        cache = getattr(self, '_hydro_cache', None)
        if cache and cache['key'] == key:
            on_ready(cache['dem'], profile, cache['res'])
            return
        if dem.size > 4_000_000 and QMessageBox.question(
                self, "Large DEM", f"{dem.size:,} cells: depression filling may take a minute. Continue?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return

        def done(res):
            self._val_progress.close()
            self._hydro_cache = {'key': key, 'dem': dem, 'res': res}
            on_ready(dem, profile, res)
        self._start_validation_job(self._flow_job(dem, profile['transform'], profile.get('crs')), done, title)

    def _stream_mask_from_dialog(self, res):
        params = self._get_stream_parameters(res['acc'])
        if not params:
            return None, None
        streams = terrain.extract_streams(res['acc'], params['method'], params['threshold'])
        if not streams.any():
            QMessageBox.warning(self, "No streams", "The threshold is too high: no stream cells.")
            return None, None
        return streams, params

    # ---- HAND
    def calculate_hand(self):
        try:
            self._with_flow("HAND: flow", self._hand_ready)
        except Exception as e:
            self.show_error(f"HAND failed: {e}")

    def _hand_ready(self, dem, profile, res):
        streams, params = self._stream_mask_from_dialog(res)
        if streams is None:
            return
        xres, yres = terrain.cell_size(profile['transform'], dem.shape, profile.get('crs'))

        def job(progress):
            progress(20, "HAND along the flow paths...")
            hnd, dist = basins.hand(res['filled'], res['flow_dir'], streams, xres, yres)
            return {'hand': hnd, 'dist': dist}

        def done(r):
            self._val_progress.close()
            try:
                hnd = r['hand']
                finite = hnd[np.isfinite(hnd)]
                if finite.size == 0:
                    raise ValueError("No cell drains to a stream cell; lower the stream threshold")
                vmax = float(np.percentile(finite, 99))
                self._add_derived_layer("HAND", hnd, profile, cmap='YlGnBu_r', vmin=0, vmax=max(vmax, 1.0))
                self._add_derived_layer("Distance_to_Drainage", r['dist'], profile, cmap='viridis')
                self.display_raster_layer(self.layers["HAND"], "HAND")
                h, ok = QInputDialog.getDouble(
                    self, "Flood-prone mask",
                    f"HAND median {np.median(finite):.1f} m, 95th percentile {np.percentile(finite, 95):.1f} m.\n"
                    "Create a mask of cells with HAND <= (m)?  (Cancel to skip)", 5.0, 0.1, 1000.0, 1)
                if ok:
                    m = np.where(np.isfinite(hnd), (hnd <= h).astype(np.uint8), 0).astype(np.uint8)
                    self._add_derived_layer(f"HAND_le_{h:g}m", m, profile, cmap='Blues', vmin=0, vmax=1, nodata=0, dtype='uint8')
                    pct = 100.0 * float(m.sum()) / finite.size
                    QMessageBox.information(
                        self, "HAND mask",
                        f"{pct:.1f}% of the cells that drain to a stream are within {h:g} m above it.\n"
                        "HAND is a terrain-only susceptibility indicator, not a flood extent: validate it against an "
                        "independent flood map before drawing conclusions.")
            except Exception as e:
                self.show_error(f"HAND failed: {e}")
        self._start_validation_job(job, done, "HAND")

    # ---- TWI / SPI
    def calculate_twi_spi(self):
        try:
            self._with_flow("TWI: flow", self._twi_ready)
        except Exception as e:
            self.show_error(f"TWI/SPI failed: {e}")

    def _twi_ready(self, dem, profile, res):
        xres, yres = terrain.cell_size(profile['transform'], dem.shape, profile.get('crs'))

        def job(progress):
            progress(30, "TWI...")
            w = basins.twi(res['acc'], res['filled'], xres, yres)
            progress(70, "SPI...")
            p = basins.spi(res['acc'], res['filled'], xres, yres)
            return {'twi': w, 'spi': p}

        def done(r):
            self._val_progress.close()
            try:
                for key, name, cmap in (('twi', 'TWI', 'YlGnBu'), ('spi', 'SPI', 'magma')):
                    a = r[key]
                    fin = a[np.isfinite(a)]
                    if fin.size:
                        self._add_derived_layer(name, a, profile, cmap=cmap, vmin=float(np.percentile(fin, 2)),
                                                vmax=float(np.percentile(fin, 98)))
                self.display_raster_layer(self.layers["TWI"], "TWI")
            except Exception as e:
                self.show_error(f"TWI/SPI failed: {e}")
        self._start_validation_job(job, done, "TWI / SPI")

    # ---- watershed
    def delineate_watershed_tool(self):
        try:
            self._with_flow("Watershed: flow", self._watershed_ready)
        except Exception as e:
            self.show_error(f"Watershed failed: {e}")

    def _watershed_ready(self, dem, profile, res):
        self._pending_basin = (dem, profile, res)
        QMessageBox.information(
            self, "Pour point",
            "Display the DEM (or any layer derived from it, e.g. Flow_Accumulation), then click the outlet of the basin on the map.\n"
            "The click is snapped to the cell with the highest flow accumulation within 5 cells.\n"
            "Turn off the zoom/pan tool first.")
        self.canvas.setCursor(Qt.CrossCursor)
        self._basin_cid = self.canvas.mpl_connect('button_press_event', self._on_pour_point_click)

    def _on_pour_point_click(self, event):
        if event.inaxes is not self.ax or event.button != 1 or event.xdata is None:
            return
        if getattr(self, 'nav_toolbar', None) is not None and getattr(self.nav_toolbar, 'mode', ''):
            return                                          # zoom/pan active: not a pour-point click
        self.canvas.mpl_disconnect(self._basin_cid)
        self.canvas.unsetCursor()
        dem, profile, res = self._pending_basin
        try:
            shown = self.layers.get(getattr(self, 'current_layer', None)) or {}
            img = shown.get('image')
            if img is None or np.asarray(img).shape[-2:] != dem.shape:
                QMessageBox.warning(self, "Pour point",
                                    "The displayed layer does not have the DEM's grid. Display the DEM or one of its derived layers and try again.")
                return
            sp = shown.get('profile') or {}
            tr = sp.get('transform') or profile['transform']
            if georef.axis_spec(tr, sp.get('crs') or profile.get('crs'), *dem.shape).extent is None:
                row, col = int(round(event.ydata)), int(round(event.xdata))
            else:
                rc = georef.map_to_pixel(tr, event.xdata, event.ydata)
                if rc is None:
                    raise ValueError("Cannot convert the click to a pixel")
                row, col = rc
            if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
                QMessageBox.warning(self, "Pour point", "The click is outside the DEM.")
                return
            self._run_watershed(row, col, dem, profile, res)
        except Exception as e:
            self.show_error(f"Watershed failed: {e}")

    def _run_watershed(self, row, col, dem, profile, res):
        streams, params = self._stream_mask_from_dialog(res)
        if streams is None:
            return
        xres, yres = terrain.cell_size(profile['transform'], dem.shape, profile.get('crs'))
        fd, acc = res['flow_dir'], res['acc']

        def job(progress):
            progress(10, "Snapping the pour point to the stream...")
            r, c = basins.snap_to_stream(acc, row, col, radius=5)
            progress(30, "Delineating the watershed...")
            mask = basins.delineate_watershed(fd, r, c)
            progress(55, "Stream order...")
            order = terrain.stream_order(fd, streams, 'strahler')
            progress(75, "Morphometrics...")
            metrics = basins.basin_morphometrics(dem, mask, fd, streams, order, xres, yres, (r, c))
            try:
                ra, rh, hi = basins.hypsometric_curve(dem, mask)
            except ValueError:
                ra = rh = None
                hi = float('nan')
            return {'mask': mask, 'row': r, 'col': c, 'metrics': metrics, 'hi': hi, 'curve': (ra, rh)}

        self._start_validation_job(job, lambda r: self._on_watershed_done(r, profile, params), "Watershed")

    def _on_watershed_done(self, r, profile, params):
        self._val_progress.close()
        try:
            mask, metrics = r['mask'], r['metrics']
            k = 1
            while f"Watershed_{k}" in self.layers:
                k += 1
            name = f"Watershed_{k}"
            self._add_derived_layer(name, mask.astype(np.uint8), profile, cmap='Reds', vmin=0, vmax=1, nodata=0, dtype='uint8')
            tr = profile['transform']
            polys = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
            if polys:
                attrs = {  # short names: valid as shapefile fields (<= 10 characters)
                    'area_km2': metrics["Area (km2)"], 'perim_km': metrics["Perimeter (km)"],
                    'len_km': metrics["Basin length, longest flow path (km)"], 'relief_m': metrics["Relief (m)"],
                    'dd_kmkm2': metrics["Drainage density Dd (km/km2)"], 'fs': metrics["Stream frequency Fs (segments/km2)"],
                    'rb': metrics["Mean bifurcation ratio Rb"], 'rf': metrics["Form factor Rf (A / Lb^2)"],
                    're': metrics["Elongation ratio Re (2 sqrt(A/pi) / Lb)"], 'rc': metrics["Circularity ratio Rc (4 pi A / P^2)"],
                    'hi': r['hi'], 'max_order': metrics["Maximum stream order (Strahler)"],
                }
                geom = polys[0] if len(polys) == 1 else unary_union(polys)
                gdf = gpd.GeoDataFrame([attrs], geometry=[geom], crs=profile.get('crs'))
                lname = f"{name}_Polygon"
                layer_data = {'type': 'vector', 'gdf': gdf, 'file_name': lname, 'style': {'color': '#d62728', 'width': 2}}
                self.layers[lname] = layer_data
                self.layer_list.addItem(QListWidgetItem(lname))
            self.display_raster_layer(self.layers[name], name)
            self._show_basin_report(name, r, params)
        except Exception as e:
            self.show_error(f"Could not create the watershed layers: {e}")

    def _show_basin_report(self, name, r, params):
        metrics, hi = r['metrics'], r['hi']
        text = (f"{name}: outlet at row {r['row']}, column {r['col']}\n"
                f"Streams: {params['method']} threshold {params['threshold']}; Strahler order\n"
                f"{'-' * 60}\n{basins.format_report(metrics, hi)}\n{'-' * 60}\n"
                "Notes: lengths follow D8 paths (they over-estimate straight-line distances); the perimeter counts cell\n"
                "edges (staircase). Geographic DEMs use cell sizes converted to metres at the mid-latitude.")
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Basin morphometrics - {name}")
        dlg.setMinimumSize(760, 560)
        lay = QVBoxLayout(dlg)
        box = QTextEdit(); box.setReadOnly(True); box.setFontFamily("Courier New"); box.setPlainText(text)
        lay.addWidget(box)
        row = QHBoxLayout()
        b_csv = QPushButton("Save table (CSV)..."); b_ok = QPushButton("Close")
        row.addWidget(b_csv); row.addStretch(1); row.addWidget(b_ok)
        lay.addLayout(row)
        b_ok.clicked.connect(dlg.accept)

        def save_csv():
            import csv
            path, _ = QFileDialog.getSaveFileName(self, "Save morphometrics", f"{name}_morphometrics.csv", "CSV (*.csv)")
            if not path:
                return
            try:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    wr = csv.writer(fh)
                    wr.writerow(["parameter", "value"])
                    for kk, vv in metrics.items():
                        wr.writerow([kk, vv if not isinstance(vv, dict) else "; ".join(f"{a}:{b}" for a, b in vv.items())])
                    wr.writerow(["Hypsometric integral HI", hi])
                    ra, rh = r['curve']
                    if ra is not None:
                        wr.writerow([]); wr.writerow(["relative_area", "relative_height"])
                        for a, b_ in zip(ra, rh):
                            wr.writerow([f"{a:.5f}", f"{b_:.5f}"])
                QMessageBox.information(self, "Saved", f"Saved to {path}")
            except OSError as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
        b_csv.clicked.connect(save_csv)
        dlg.exec_()

######################################  drainage_channels #################



    # def display_layer(self, layer_name, cmap=None):
    #     """Display a raster layer with optional colormap"""
    #     layer = self.layers.get(layer_name)
    #     if not layer or 'image' not in layer:
    #         return
        
    #     # Use layer's colormap if specified, otherwise use parameter
    #     cmap = layer.get('colormap', cmap)
        
    #     # Your existing display implementation here
    #     # ... (use cmap parameter when calling imshow)





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ cliping raster $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
    def raster_clipping_process(self):
        """Raster Clipping window with improved input validation and options"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Raster Clipping")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout()

        # File selection
        layout.addWidget(QLabel("Select Layer to Clip:"))
        self.layer_list_calc = QListWidget()
        self.layer_list_calc.addItems(self.layers.keys())
        self.layer_list_calc.itemClicked.connect(self.select_layer_for_clipping)
        layout.addWidget(self.layer_list_calc)

        # Coordinate input with validation
        coord_group = QGroupBox("Clipping Extent")
        coord_layout = QFormLayout()
        
        self.xmin_input = QLineEdit()
        self.ymin_input = QLineEdit()
        self.xmax_input = QLineEdit()
        self.ymax_input = QLineEdit()
        
        # Set validators for numeric input
        for input_field in [self.xmin_input, self.ymin_input, self.xmax_input, self.ymax_input]:
            input_field.setValidator(QDoubleValidator())
        
        coord_layout.addRow("Minimum X:", self.xmin_input)
        coord_layout.addRow("Minimum Y:", self.ymin_input)
        coord_layout.addRow("Maximum X:", self.xmax_input)
        coord_layout.addRow("Maximum Y:", self.ymax_input)
        coord_group.setLayout(coord_layout)
        layout.addWidget(coord_group)

        # Options
        options_group = QGroupBox("Clipping Options")
        options_layout = QVBoxLayout()
        
        self.nodata_checkbox = QCheckBox("Preserve NoData values")
        self.nodata_checkbox.setChecked(True)
        options_layout.addWidget(self.nodata_checkbox)
        
        self.pixel_coord_radio = QRadioButton("Use pixel coordinates")
        self.geo_coord_radio = QRadioButton("Use geographic coordinates")
        self.geo_coord_radio.setChecked(True)
        
        coord_type_group = QButtonGroup()
        coord_type_group.addButton(self.pixel_coord_radio)
        coord_type_group.addButton(self.geo_coord_radio)
        options_layout.addWidget(QLabel("Coordinate Type:"))
        options_layout.addWidget(self.pixel_coord_radio)
        options_layout.addWidget(self.geo_coord_radio)
        
        # Add checkbox for legend removal
        self.remove_legend_checkbox = QCheckBox("Remove legend after clipping")
        self.remove_legend_checkbox.setChecked(True)
        self.remove_legend_checkbox.setToolTip("Recommended to prevent display issues with clipped rasters")
        options_layout.addWidget(self.remove_legend_checkbox)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Action buttons
        btn_box = QDialogButtonBox()
        btn_clip = btn_box.addButton("Clip Raster", QDialogButtonBox.ActionRole)
        btn_cancel = btn_box.addButton(QDialogButtonBox.Cancel)
        
        btn_clip.clicked.connect(lambda: self.execute_clipping(dialog))
        btn_cancel.clicked.connect(dialog.reject)
        layout.addWidget(btn_box)

        dialog.setLayout(layout)
        dialog.exec_()


    def select_layer_for_clipping(self, item):
        """Handle layer selection and populate coordinate hints"""
        self.selected_layer = item.text()
        layer_data = self.layers[self.selected_layer]
        
        # Get bounds from the selected layer
        try:
            profile = layer_data.get('profile') or {}
            transform = profile.get('transform')
            img = layer_data.get('image')
            if img is None or not georef.is_georeferenced(transform, profile.get('crs')):
                QMessageBox.warning(self, "Not georeferenced",
                                    f"Layer '{self.selected_layer}' has no geotransform, so map coordinates are unknown.\n"
                                    "Use 'pixel coordinates' for this layer, or select the original GeoTIFF layer.")
                return
            height, width = np.asarray(img).shape[-2:]
            xmin, ymax = georef.pixel_to_map(transform, 0, 0)
            xmax, ymin = georef.pixel_to_map(transform, height, width)
            geographic = georef.crs_description(profile.get('crs'))[0]
            fmt = "{:.6f}" if geographic else "{:.2f}"

            # Update input fields with suggested values
            self.xmin_input.setText(fmt.format(min(xmin, xmax)))
            self.ymin_input.setText(fmt.format(min(ymin, ymax)))
            self.xmax_input.setText(fmt.format(max(xmin, xmax)))
            self.ymax_input.setText(fmt.format(max(ymin, ymax)))

            QMessageBox.information(self, "Layer Selected",
                                    f"Layer '{self.selected_layer}' loaded.\n"
                                    f"Suggested bounds (full extent, {'degrees' if geographic else 'map units'}) have been populated.")
        except Exception as e:
            QMessageBox.warning(self, "Info",
                                f"Selected layer '{self.selected_layer}' but couldn't get bounds: {str(e)}")

    def execute_clipping(self, dialog):
        """Execute the clipping process with comprehensive error handling"""
        try:
            # Validate layer selection
            if not hasattr(self, 'selected_layer') or self.selected_layer not in self.layers:
                QMessageBox.warning(self, "Error", "Please select a valid layer first.")
                return

            # Get layer data
            layer_data = self.layers[self.selected_layer]
            image = layer_data['image']
            profile = layer_data.get('profile') or {}
            if profile.get('transform') is None:
                QMessageBox.warning(self, "Not georeferenced",
                                    f"Layer '{self.selected_layer}' has no geotransform and cannot be clipped by map coordinates.")
                return
            original_path = layer_data.get('file_name', 'memory_layer')

            # Get and validate coordinates
            try:
                coords = (
                    float(self.xmin_input.text()),
                    float(self.ymin_input.text()),
                    float(self.xmax_input.text()),
                    float(self.ymax_input.text())
                )
            
            except ValueError:
                QMessageBox.warning(self, "Error", "All coordinates must be numeric values.")
                return

            # Validate coordinate order
            if coords[0] >= coords[2] or coords[1] >= coords[3]:
                QMessageBox.warning(self, "Error", "Invalid bounds: xmin must be < xmax and ymin < ymax")
                return

            # Determine options
            use_pixel_coords = self.pixel_coord_radio.isChecked()
            preserve_nodata = self.nodata_checkbox.isChecked()
            remove_legend = self.remove_legend_checkbox.isChecked()

            # Perform clipping
            clipped_image, new_profile, output_path = self.image_processor.clip_raster(
                image=image,
                profile=profile,
                bbox=coords,
                file_name=original_path,
                use_pixel_coords=use_pixel_coords,
                preserve_nodata=preserve_nodata
            )

            # Validate results
            if clipped_image is None:
                QMessageBox.warning(self, "Error", "Clipping operation failed - no output generated")
                return

            # Add clipped layer to management
            clipped_name = f"Clipped_{self.selected_layer}"
            counter = 1
            while clipped_name in self.layers:
                clipped_name = f"Clipped_{self.selected_layer}_{counter}"
                counter += 1

            self.layers[clipped_name] = {
                'image': clipped_image,
                'profile': new_profile,
                'file_name': output_path,
                'no_legend': remove_legend  # Flag to indicate legend should be removed
            }

            # Update UI
            self.layer_list.addItem(clipped_name)
            
            # Display image without legend if requested
            if remove_legend:
                self.display_image_no_legend(clipped_image, clipped_name, new_profile)
            else:
                self.display_raster_layer(clipped_image, clipped_name)
            
            # Ask user if they want to save the clipped raster
            reply = QMessageBox.question(
                self,
                "Clipping Complete",
                "Raster clipped successfully!\n\nWould you like to save a copy to disk?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.save_raster_results(
                    clipped_image,
                    new_profile,
                    "clipped_raster",
                    clipped_name
                )
            
            dialog.accept()           

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clip raster:\n{str(e)}")

    def display_image_no_legend(self, image, layer_name, profile=None):
        """Display image using matplotlib with proper channel ordering (map extent kept, axes hidden)."""
        self.ax.clear()
        if profile is None:
            profile = (self.layers.get(layer_name) or {}).get('profile') or self._inherited_profile(image.shape)
        
        # Fix channel order if needed (for (channels, height, width) input)
        if len(image.shape) == 3 and image.shape[0] in [3, 4]:  # Channel-first format
            image = np.transpose(image, (1, 2, 0))  # Convert to (height, width, channels)
        
        # Display image
        if len(image.shape) == 2:  # Grayscale
            georef.draw_raster(self.ax, image, profile, decorate=False, cmap='gray')
        else:  # RGB/RGBA
            georef.draw_raster(self.ax, image, profile, decorate=False)
        
        self.ax.axis('off')
        self.canvas.draw()
        self.statusBar().showMessage(f"Displaying: {layer_name} (no legend)")


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
   

    def save_edges_as_shapefile(self):
        if hasattr(self, 'image') and self.image.shape[0] == 1:  # Check if edges are detected
            try:
                # Convert the edges to polygons
                edges = self.image[0]  # Get the edge-detected raster
                gdf = self.raster_to_polygons(edges)
                
                # Set the CRS from the profile
                if 'crs' in self.profile:
                    gdf.set_crs(self.profile['crs'], inplace=True)
                else:
                    raise ValueError("CRS information is missing in the profile.")
                
                # Ask the user for the output shapefile path
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Edges as Shapefile", "", "Shapefiles (*.shp)")
                if file_name:
                    # Save the GeoDataFrame as a shapefile
                    gdf.to_file(file_name, driver='ESRI Shapefile')
                    
                    QMessageBox.information(self, "Success", f"Edges saved as shapefile: {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save edges as shapefile: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No edge-detected image found. Please apply Canny edge detection first.")

    ################################## Setting Menue ########################################
    ####General Settings Dialog


    #####Display Settings Dialog

    ##### Processing Settings Dialog

######Advanced Settings Dialog


    ##### About Dialog


    def update_info_box(self, layer_data=None, file_path=None):
        """Update info box with enhanced vector file handling and nodata exclusion for rasters"""
        try:
            # Determine data source
            if layer_data is not None:
                data_source = 'layer'
            elif file_path is not None:
                data_source = 'file'
                layer_data = {'file_name': file_path}
            elif hasattr(self, 'image'):
                data_source = 'current'
                layer_data = {
                    'image': self.image,
                    'profile': getattr(self, 'profile', {}),
                    'file_name': getattr(self, 'image_path', 'Current Image')
                }
            else:
                self.info_box.setText("No data available")
                return

            metadata = {}
            is_raster = False

            # Handle vector data (unchanged)
            if 'gdf' in layer_data.keys() or (data_source == 'file' and 
                        any(ext in file_path for ext in ['.shp','.geojson','.gpkg'])):
                        try:
                            gdf = layer_data.get('gdf', None)
                            if gdf is None:
                                gdf = gpd.read_file(layer_data['file_name'])  # CORRECTED LINE
                            
                            if gdf.empty:
                                metadata = {"Data Type": "Vector (Empty)"}
                            else:
                                # Get unique geometry types
                                geom_types = gdf.geometry.type.unique()
                                geom_type_str = ', '.join(geom_types) if len(geom_types) > 0 else 'Unknown'
                                
                                # Get CRS details
                                crs = gdf.crs
                                crs_name = crs.name if crs else 'Unknown'
                                crs_units = crs.axis_info[0].unit_name if crs and hasattr(crs, 'axis_info') else 'Unknown'
                                crs_epsg = f"EPSG:{crs.to_epsg()}" if crs and crs.to_epsg() else 'Not EPSG'
                                
                                # Calculate area/length stats for polygons/lines
                                stats = {}
                                if 'Polygon' in geom_types or 'MultiPolygon' in geom_types:
                                    gdf['_area'] = gdf.geometry.area
                                    stats = {
                                        "Total Area": f"{gdf['_area'].sum():.2f} {crs_units}²",
                                        "Avg Area": f"{gdf['_area'].mean():.2f} {crs_units}²",
                                        "Min Area": f"{gdf['_area'].min():.2f} {crs_units}²",
                                        "Max Area": f"{gdf['_area'].max():.2f} {crs_units}²"
                                    }
                                elif 'LineString' in geom_types or 'MultiLineString' in geom_types:
                                    gdf['_length'] = gdf.geometry.length
                                    stats = {
                                        "Total Length": f"{gdf['_length'].sum():.2f} {crs_units}",
                                        "Avg Length": f"{gdf['_length'].mean():.2f} {crs_units}",
                                        "Min Length": f"{gdf['_length'].min():.2f} {crs_units}",
                                        "Max Length": f"{gdf['_length'].max():.2f} {crs_units}"
                                    }

                                # Count features by geometry type
                                type_counts = gdf.geometry.type.value_counts().to_dict()
                                type_counts_str = ', '.join([f"{k} ({v})" for k, v in type_counts.items()])
                                
                                metadata = {
                                    "File Name": os.path.basename(layer_data.get('file_name', 'Unknown')),
                                    "Data Type": "Vector",
                                    "Geometry Type": geom_type_str,
                                    "Features": f"{len(gdf)}",
                                    "Feature Types": type_counts_str,
                                    "Coordinate System": crs_name,
                                    "CRS Authority": crs_epsg,
                                    "CRS Units": crs_units,
                                    "Extent": f"X: {gdf.total_bounds[0]:.2f} - {gdf.total_bounds[2]:.2f}\n"
                                            f"Y: {gdf.total_bounds[1]:.2f} - {gdf.total_bounds[3]:.2f}",
                                    **stats,
                                    "Attributes": f"{len(gdf.columns)} fields",
                                    "Memory Usage": f"{gdf.memory_usage(deep=True).sum() / 1024:.1f} KB"
                                }
                                
                        except Exception as e:
                            self.info_box.setText(f"Vector data error: {str(e)}")
                            return  
                    
            # Handle raster data with nodata exclusion
            elif 'image' in layer_data or (data_source == 'file' and 
                                        any(file_path.endswith(ext) for ext in ['.tif', '.tiff'])):
                try:
                    if 'image' in layer_data:
                        img = layer_data['image']
                        profile = layer_data.get('profile', {})
                    else:
                        with rasterio.open(file_path) as src:
                            img = src.read()
                            profile = dict(src.profile)  # rasterio's Profile is a UserDict, not a dict
                    
                    crs_info = self.get_crs_info(profile.get('crs'))
                    is_raster = True
                    
                    # Get nodata value
                    nodata = profile.get('nodata', None)
                    valid_pixels = None
                    
                    # Calculate value range excluding nodata
                    if len(img.shape) == 3:  # Multi-band
                        if nodata is not None:
                            valid_mask = img != nodata
                            valid_pixels = np.count_nonzero(valid_mask)
                            min_val = np.min(img[valid_mask]) if valid_pixels > 0 else np.nan
                            max_val = np.max(img[valid_mask]) if valid_pixels > 0 else np.nan
                        else:
                            min_val = np.min(img)
                            max_val = np.max(img)
                            valid_pixels = img.size
                    else:  # Single band
                        if nodata is not None:
                            valid_mask = img != nodata
                            valid_pixels = np.count_nonzero(valid_mask)
                            min_val = np.min(img[valid_mask]) if valid_pixels > 0 else np.nan
                            max_val = np.max(img[valid_mask]) if valid_pixels > 0 else np.nan
                        else:
                            min_val = np.min(img)
                            max_val = np.max(img)
                            valid_pixels = img.size
                    
                    # Calculate nodata percentage
                    if nodata is not None:
                        total_pixels = img.size
                        nodata_pixels = total_pixels - valid_pixels
                        nodata_percentage = (nodata_pixels / total_pixels) * 100
                    else:
                        nodata_percentage = 0
                    
                    # Get cell size from transform
                    transform = profile.get('transform')
                    if transform:
                        cell_size_x = transform.a
                        cell_size_y = transform.e
                    else:
                        cell_size_x = cell_size_y = "Unknown"
                    
                    metadata = {
                        "File Name": os.path.basename(layer_data.get('file_name', 'Current Image')),
                        "Dimensions": f"{img.shape[-2]} × {img.shape[-1]} pixels",
                        "Bands": img.shape[0] if len(img.shape) == 3 else 1,
                        "Data Type": str(img.dtype),
                        "Coordinate System": f"{crs_info['name']} ({crs_info['epsg']})",
                        "CRS Type": crs_info['type'],
                        "Units": crs_info['units'],
                        "Pixel Value Range": f"{min_val:.2f} - {max_val:.2f}" if not np.isnan(min_val) else "N/A",
                        "Nodata Value": f"{nodata}" if nodata is not None else "Not defined",
                        "Nodata Pixels": f"{nodata_pixels:,} ({nodata_percentage:.1f}%)" if nodata is not None else "N/A",
                        "Valid Pixels": f"{valid_pixels:,}" if nodata is not None else f"{img.size:,}",
                        "Cell Size (X, Y)": f"{abs(cell_size_x):.2f}, {abs(cell_size_y):.2f}" 
                                        if transform else "Unknown"
                    }
                except Exception as e:
                    self.info_box.setText(f"Error processing raster data: {str(e)}")
                    return

            # Generate HTML content (unchanged)
            info_html = f"""<style>
                .info-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-family: Arial;
                    font-size: 10pt;
                }}
                .info-table td {{
                    padding: 4px;
                    vertical-align: top;
                    border-bottom: 1px solid #eee;
                }}
                h2 {{
                    color: #2c3e50;
                    margin-bottom: 5px;
                }}
                .section-title {{
                    background-color: #f5f5f5;
                    font-weight: bold;
                    padding: 4px;
                    margin-top: 10px;
                }}
            </style>
            <h2>File Information Box</h2>
            <table class="info-table">"""
            
            # Common metadata
            info_html += f"""<tr><td colspan="2" class="section-title">Basic Information</td></tr>
                <tr><td><b>File Name</b></td><td>{metadata.get('File Name', 'Unknown')}</td></tr>
                <tr><td><b>Data Type</b></td><td>{metadata.get('Data Type', 'Unknown')}</td></tr>"""
            
            # Vector-specific metadata (unchanged)
            if 'Geometry Type' in metadata:
                info_html += f"""
                    <tr><td colspan="2" class="section-title">Vector Properties</td></tr>
                    <tr><td><b>Geometry Type</b></td><td>{metadata['Geometry Type']}</td></tr>
                    <!-- Rest of vector HTML remains exactly the same -->
                    """
            # Raster-specific metadata with nodata info
            else:
                info_html += f"""<tr><td colspan="2" class="section-title">Raster Properties</td></tr>"""
                
                for key in ["Dimensions", "Bands", "Data Type", "Coordinate System", 
                        "CRS Type", "Units", "Pixel Value Range", "Cell Size (X, Y)"]:
                    if key in metadata:
                        info_html += f"""<tr>
                            <td><b>{key}</b></td>
                            <td>{metadata[key]}</td>
                        </tr>"""
                
                # Add nodata information section
                if metadata.get('Nodata Value') != "Not defined":
                    info_html += f"""<tr><td colspan="2" class="section-title">Nodata Information</td></tr>
                        <tr><td><b>Nodata Value</b></td><td>{metadata['Nodata Value']}</td></tr>
                        <tr><td><b>Nodata Pixels</b></td><td>{metadata['Nodata Pixels']}</td></tr>
                        <tr><td><b>Valid Pixels</b></td><td>{metadata['Valid Pixels']}</td></tr>"""
            
            info_html += "</table>"
            
            if is_raster:
                info_html += self.generate_suggestions(img, metadata["Bands"])
            
            self.info_box.setHtml(info_html)
            self.info_box.setContextMenuPolicy(Qt.CustomContextMenu)
            
        except Exception as e:
            self.info_box.setText(f"Error updating information: {str(e)}")
            traceback.print_exc()


    def get_crs_info(self, crs):
        """
        الحصول على معلومات وصفية عن نظام الإحداثيات
        
        Args:
            crs: كائن CRS من rasterio أو geopandas
            
        Returns:
            dict: معلومات عن نظام الإحداثيات
        """
        crs_info = {
            "name": "Unknown",
            "epsg": "Unknown",
            "type": "Unknown",
            "units": "Unknown",
            "details": "No CRS information available"
        }
        
        if crs is None:
            return crs_info
        
        try:
            # إذا كان CRS كائنًا من rasterio/geopandas
            if hasattr(crs, 'to_epsg'):
                epsg_code = crs.to_epsg()
                crs_info['epsg'] = f"EPSG:{epsg_code}" if epsg_code else "Unknown"
                
                # معلومات إضافية من pyproj
                from pyproj import CRS
                pyproj_crs = CRS(crs)
                crs_info['name'] = pyproj_crs.name
                crs_info['type'] = "Geographic" if pyproj_crs.is_geographic else "Projected"
                crs_info['units'] = pyproj_crs.axis_info[0].unit_name if pyproj_crs.axis_info else "Unknown"
                
                # تفاصيل إضافية حسب نوع النظام
                if "UTM zone" in pyproj_crs.name:
                    crs_info['details'] = f"UTM Projection - Zone {pyproj_crs.utm_zone}"
                elif "WGS 84" in pyproj_crs.name:
                    crs_info['details'] = "World Geodetic System 1984"
                elif "Web Mercator" in pyproj_crs.name:
                    crs_info['details'] = "Web Mercator (Google Maps compatible)"
                else:
                    crs_info['details'] = pyproj_crs.name
                
            elif isinstance(crs, dict):
                # إذا كان CRS في شكل قاموس
                crs_info.update({
                    'name': crs.get('name', 'Unknown'),
                    'epsg': f"EPSG:{crs.get('epsg', 'Unknown')}",
                    'details': crs.get('proj4', 'Unknown')
                })
                
        except Exception as e:
            logging.info(f"Error getting CRS info: {str(e)}")
        return crs_info

    def generate_suggestions(self, image, num_bands):
        suggestions = []
        suggestions.append("Right-click here or on the file name in the layer list for more options.")

        # if num_bands == 1:
        #     suggestions.append("Use edge detection tools (Canny)")
        #     suggestions.append("Try different color gradients from the menu")
        # elif num_bands == 3:
        #     suggestions.append("Calculate NDVI from spectral indices tools")
        #     suggestions.append("Adjust color balance from processing menu")

        # if image.max() > 1000:
        #     suggestions.append("You may need to normalize the values")

        suggestions_html = """
        <div class="suggestion-list">
            <h3 style="color:#3498db;">Smart Suggestions</h3>
            <ul style="list-style-type: square; padding-left: 20px;">
        """

        for suggestion in suggestions:
            suggestions_html += f"<li>{suggestion}</li>"

        suggestions_html += "</ul></div>"
        return suggestions_html   

    def show_context_menu(self, pos):
        """عرض القائمة المنبثقة عند النقر بزر الماوس الأيمن"""
        menu = QMenu()
        
        # Add Properties option if a layer is selected
        current_item = self.layer_list.currentItem()
        if current_item:
            properties_action = menu.addAction("Properties")
            properties_action.triggered.connect(self.show_layer_properties)
            menu.addSeparator()  # Add a separator line
        
        # Keep existing options
        if hasattr(self, 'image') or hasattr(self, 'image_path') or hasattr(self, 'layers'):
            stats_action = menu.addAction("Show Statistics")
            stats_action.triggered.connect(self.show_stats_dialog)
        
        copy_action = menu.addAction("Copy Information")
        copy_action.triggered.connect(self.copy_metadata)
        
        save_action = menu.addAction("Save as Text File")
        save_action.triggered.connect(self.save_metadata)
        
        menu.exec_(self.info_box.mapToGlobal(pos))

    def show_stats_dialog(self):
        """عرض نافذة الإحصائيات للبيانات الحالية"""
        current_item = self.layer_list.currentItem()  # الحصول على العنصر المحدد حالياً
        self.show_stats(current_item)  # تمرير العنصر المحدد أو None إذا لم يكن هناك عنصر




    def copy_metadata(self):
        self.info_box.selectAll()
        self.info_box.copy()

    def save_metadata(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Information", "", "Text Files (*.txt)")
        if file_name:
            with open(file_name, 'w') as f:
                f.write(self.info_box.toPlainText())

    
 

    def show_stats(self, selected_item=None):
        """
        Display comprehensive statistics for selected file or current image
        
        Args:
            selected_item: Can be:
                - QListWidgetItem (from layer_list)
                - None (show current image stats)
        """
        try:
            # Configuration
            WINDOW_WIDTH = 1000
            WINDOW_HEIGHT = 700
            BAND_COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
            BAND_NAMES = ['Red (Band 1)', 'Green (Band 2)', 'Blue (Band 3)'] + \
                        [f'Band {i}' for i in range(4, 7)]

            # 1. Determine data source
            layer_data = self._get_layer_data(selected_item)
            if layer_data is None:
                QMessageBox.warning(self, "Statistics", "No data available.")
                return

            # Get nodata value from profile if exists
            nodata = (layer_data.get('profile') or {}).get('nodata', None)

            # 2. Create dialog
            stats_dialog = QDialog(self)
            stats_dialog.setWindowTitle(f"Statistics {layer_data.get('title_suffix', '')}")
            stats_dialog.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
            main_layout = QVBoxLayout(stats_dialog)

            # Add bin selection controls
            control_layout = QHBoxLayout()
            bin_label = QLabel("Histogram Bins:")
            bin_spinbox = QSpinBox()
            bin_spinbox.setRange(10, 1000)
            bin_spinbox.setValue(256)
            bin_spinbox.setSingleStep(10)
            control_layout.addWidget(bin_label)
            control_layout.addWidget(bin_spinbox)
            
            # Add nodata toggle checkbox
            nodata_check = QCheckBox("Exclude Nodata Values")
            nodata_check.setChecked(True)  # Default to excluding nodata
            nodata_check.setEnabled(nodata is not None)  # Only enable if nodata exists
            control_layout.addWidget(nodata_check)
            
            control_layout.addStretch()
            main_layout.addLayout(control_layout)

            # Create a container for the stats display that we can refresh
            stats_container = QWidget()
            stats_layout = QVBoxLayout(stats_container)
            main_layout.addWidget(stats_container)

            def update_stats_display():
                # Clear previous content
                for i in reversed(range(stats_layout.count())): 
                    stats_layout.itemAt(i).widget().setParent(None)
                
                # Get current settings
                bins = bin_spinbox.value()
                exclude_nodata = nodata_check.isChecked() if nodata is not None else False
                
                # 3. Process data based on type
                if layer_data.get('type') == 'vector':
                    self._show_vector_stats(layer_data, stats_layout)
                else:
                    # Create valid pixel mask if excluding nodata
                    valid_mask = None
                    if exclude_nodata and nodata is not None:
                        img = layer_data['image']
                        if img.ndim == 3:  # Multi-band
                            valid_mask = np.all(img != nodata, axis=0)
                        else:  # Single band
                            valid_mask = img != nodata
                    
                    self._show_raster_stats(
                        layer_data=layer_data,
                        layout=stats_layout,
                        band_colors=BAND_COLORS,
                        band_names=BAND_NAMES,
                        bins=bins,
                        valid_mask=valid_mask
                    )

            # Initial display
            update_stats_display()
            
            # Connect controls to update function
            bin_spinbox.valueChanged.connect(update_stats_display)
            nodata_check.stateChanged.connect(update_stats_display)

            # Add Ok button
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(stats_dialog.accept)
            main_layout.addWidget(button_box)

            stats_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
            traceback.print_exc()
        finally:
            plt.close('all')

    def _show_raster_stats(self, layer_data, layout, band_colors, band_names, bins=256, valid_mask=None):
        """Display raster layer statistics with customizable bins, excluding nodata values"""
        try:
            img = layer_data['image']
            stats_text = QTextEdit()
            stats_text.setReadOnly(True)
            stats_text.setFont(QFont('Consolas', 10))
            
            # Get nodata value from profile if exists
            nodata = (layer_data.get('profile') or {}).get('nodata', None)
            
            # Calculate statistics based on valid_mask
            if valid_mask is not None:
                if img.ndim == 3:  # Multi-band image
                    # For multi-band, valid_mask should be 2D (height x width)
                    valid_count = np.count_nonzero(valid_mask)
                    valid_data = [band[valid_mask] for band in img]  # Apply same mask to all bands
                else:  # Single band image
                    valid_count = np.count_nonzero(valid_mask)
                    valid_data = img[valid_mask]
                
                nodata_count = img.size - valid_count
                nodata_percentage = (nodata_count / img.size) * 100 if img.size > 0 else 0
            else:
                valid_count = img.size
                nodata_count = 0
                nodata_percentage = 0
                valid_data = img  # Use all data if no mask

            stats_content = [
                f"=== RASTER STATISTICS ===",
                f"File: {os.path.basename(layer_data.get('file_name', 'Current Image'))}",
                f"Dimensions: {img.shape}",
                f"Data Type: {img.dtype}",
                f"Bands: {img.shape[0] if img.ndim == 3 else 1}",
                f"Total Pixels: {img.size:,}",
                f"Nodata Values: {nodata_count:,} ({nodata_percentage:.2f}%)" + 
                    (" - EXCLUDED FROM STATS" if valid_mask is not None else ""),
                f"Valid Pixels: {valid_count:,}",
                f"Histogram Bins: {bins}"
            ]
            
            prof = layer_data.get('profile') or {}
            if prof.get('crs'):
                stats_content.append(f"CRS: {prof['crs']}")
            if prof.get('transform') is not None:
                stats_content.append(f"Pixel size: {abs(prof['transform'].a):.6g} (CRS units)")

            # Create plot and add band statistics
            fig, ax = plt.subplots(figsize=(10, 6))
            if img.ndim == 2 or img.shape[0] == 1:
                self._process_single_band(img, stats_content, ax, bins, valid_mask)
            else:
                self._process_multi_band(img, stats_content, ax, band_colors, band_names, bins, valid_mask)

            ax.set_title(f"Pixel Value Distribution ({bins} bins)" + 
                        ("\n(Excluding Nodata Values)" if valid_mask is not None else ""))
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(alpha=0.3)
            plt.tight_layout()

            # Add widgets to layout
            self._add_plot_to_layout(fig, layout)
            stats_text.setPlainText("\n".join(stats_content))
            layout.addWidget(stats_text)

        except Exception as e:
            raise Exception(f"Raster processing error: {str(e)}")

    def _process_single_band(self, img, stats_content, ax, bins, valid_mask=None):
        """Process single band image statistics"""
        if valid_mask is not None:
            valid_data = img[valid_mask] if img.ndim == 2 else img[0][valid_mask]
        else:
            valid_data = img if img.ndim == 2 else img[0]
            
        # Calculate statistics
        if valid_data.size > 0:  # Only calculate if we have valid data
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            median_val = np.median(valid_data)
            
            # Update stats content
            stats_content.extend([
                "\n=== Band Statistics ===",
                f"Min: {min_val:.4f}",
                f"Max: {max_val:.4f}",
                f"Mean: {mean_val:.4f}",
                f"Std Dev: {std_val:.4f}",
                f"Median: {median_val:.4f}"
            ])
            
            # Plot histogram
            ax.hist(valid_data.ravel(), bins=bins, color='blue', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.2f}')
            ax.legend()
        else:
            stats_content.extend(["\n=== Band Statistics ===", "No valid pixels found"])

    def _process_multi_band(self, img, stats_content, ax, band_colors, band_names, bins, valid_mask=None):
        """Process multi-band image statistics"""
        num_bands = img.shape[0]
        stats_content.append("\n=== Band Statistics ===")
        
        for band_idx in range(num_bands):
            band_data = img[band_idx]
            
            if valid_mask is not None:
                valid_data = band_data[valid_mask]
            else:
                valid_data = band_data
                
            # Calculate statistics if we have valid data
            if valid_data.size > 0:
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                median_val = np.median(valid_data)
                
                # Update stats content
                band_name = band_names[band_idx] if band_idx < len(band_names) else f"Band {band_idx+1}"
                stats_content.extend([
                    f"\n{band_name}:",
                    f"  Min: {min_val:.4f}",
                    f"  Max: {max_val:.4f}",
                    f"  Mean: {mean_val:.4f}",
                    f"  Std Dev: {std_val:.4f}",
                    f"  Median: {median_val:.4f}"
                ])
                
                # Plot histogram
                color = band_colors[band_idx] if band_idx < len(band_colors) else 'gray'
                ax.hist(valid_data.ravel(), bins=bins, color=color, alpha=0.5, 
                    label=f'{band_name} (μ={mean_val:.2f})')
            else:
                band_name = band_names[band_idx] if band_idx < len(band_names) else f"Band {band_idx+1}"
                stats_content.extend([f"\n{band_name}:", "  No valid pixels found"])

        if num_bands > 0:
            ax.legend()
    

 
    def _get_layer_data(self, selected_item):
        """Extract layer data from selection"""
        if selected_item is None or not isinstance(selected_item, QListWidgetItem):
            # Current image case
            if not hasattr(self, 'image'):
                return None
                
            return {
                'image': self.image,
                'profile': getattr(self, 'profile', {}),
                'file_name': getattr(self, 'image_path', 'Current Image'),
                'title_suffix': " (Current Image)",
                'type': 'raster'
            }
        else:
            # Layer list case
            layer_name = selected_item.text()
            layer_data = self.layers.get(layer_name)
            if not layer_data:
                return None
                
            layer_data['title_suffix'] = f" ({layer_name})"
            layer_data['type'] = layer_data.get('type', 'raster')
            return layer_data

    def _show_vector_stats(self, layer_data, layout):
        """Display vector layer statistics"""
        try:
            gdf = layer_data['gdf']
            stats_text = QTextEdit()
            stats_text.setReadOnly(True)
            stats_text.setFont(QFont('Consolas', 10))

            # Generate statistics text
            stats_content = [
                f"=== VECTOR STATISTICS ===",
                f"File: {layer_data.get('file_name', 'Unknown')}",
                f"Type: {gdf.geom_type.iloc[0] if len(gdf) > 0 else 'Empty'}",
                f"Features: {len(gdf):,}",
                f"CRS: {gdf.crs}"
            ]

            if not gdf.empty:
                bounds = gdf.total_bounds
                stats_content.extend([
                    "\nEXTENT:",
                    f"X: {bounds[0]:.2f} → {bounds[2]:.2f} (Δ {bounds[2]-bounds[0]:.2f})",
                    f"Y: {bounds[1]:.2f} → {bounds[3]:.2f} (Δ {bounds[3]-bounds[1]:.2f})"
                ])

            # Attribute statistics
            stats_content.append("\nATTRIBUTES:")
            for col in gdf.columns:
                if col != 'geometry':
                    self._add_attribute_stats(gdf, col, stats_content)

            stats_text.setPlainText("\n".join(stats_content))

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            if not gdf.empty:
                gdf.plot(ax=ax)
                ax.set_title("Vector Preview")
            plt.tight_layout()

            # Add widgets to layout
            self._add_plot_to_layout(fig, layout)
            layout.addWidget(stats_text)

        except Exception as e:
            raise Exception(f"Vector processing error: {str(e)}")


    def _add_attribute_stats(self, gdf, col, stats_content):
        """Add statistics for a single attribute column"""
        dtype = gdf[col].dtype
        stats_content.append(f"\n{col} ({dtype}):")
        
        if np.issubdtype(dtype, np.number):
            stats_content.extend([
                f"  Min: {gdf[col].min():.2f}",
                f"  Max: {gdf[col].max():.2f}",
                f"  Mean: {gdf[col].mean():.2f}",
                f"  Std: {gdf[col].std():.2f}"
            ])
        else:
            stats_content.append(f"  Unique values: {gdf[col].nunique()}")

    def _add_plot_to_layout(self, fig, layout):
        """Add matplotlib plot to QLayout with toolbar"""
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)


    
    def get_cluster_color(self, cluster_id):
            # Define a color map for the clusters
            colors = [
                [255, 0, 0],   # Red
                [0, 255, 0],   # Green
                [0, 0, 255],   # Blue
                [255, 255, 0], # Yellow
                [255, 0, 255], # Magenta
                [0, 255, 255], # Cyan
                [128, 0, 0],   # Maroon
                [0, 128, 0],   # Green (dark)
                [0, 0, 128],   # Navy
                [128, 128, 0], # Olive
            ]
            return colors[cluster_id % len(colors)]
    
    def display_classified_image(self, classified_image, profile=None, names=None, title='Classified Image'):
        """Draw a classified raster in its real map coordinates with a legend.

        ``profile``  grid of the raster (transform + CRS); defaults to the grid of the layer it came from.
        ``names``    {class value: name}; a value 0 that is not named is treated as 'no data' and left blank.
        """
        arr = np.asarray(classified_image)
        if profile is None:
            profile = self._inherited_profile(arr.shape)
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        palette = [self.get_cluster_color(k) for k in range(10)]
        georef.draw_classified(self.ax, arr, profile, names=names, palette=palette, title=title)
        self.fig.tight_layout()
        self.canvas.draw()


    def classify_with_kmeans(self):
        """
        Classifies the image using K-Means clustering.
        """
        if not hasattr(self, 'image'):
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        try:
            # Ask the user for the number of clusters
            n_clusters, ok = QInputDialog.getInt(
                self, "Number of Clusters", 
                "Enter the number of clusters (2-10):", 
                4, 2, 10, 1
            )
            if not ok:
                QMessageBox.information(self, "Cancelled", "K-Means clustering cancelled by user.")
                return  # User canceled the input

            # Get the active layer
            if not self.layer_list.currentItem():
                raise ValueError("No active layer selected")

            active_layer = self.layer_list.currentItem().text()
            layer_data = self.layers.get(active_layer, {})
            if not isinstance(layer_data, dict) or 'image' not in layer_data:
                raise ValueError("Invalid layer data structure")

            image_data = layer_data['image']

            # Preprocess the image for classification
            if image_data.ndim == 3:  # every band takes part (3 for RGB, 5 for a Sentinel-2 stack, ...)
                flattened_image = image_data.transpose(1, 2, 0).reshape(-1, image_data.shape[0]).astype(np.float32)
            else:  # Grayscale image
                flattened_image = image_data.reshape(-1, 1).astype(np.float32)

            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            labels = kmeans.fit_predict(flattened_image)

            # Reshape the labels to the original image dimensions
            height, width = image_data.shape[-2], image_data.shape[-1]

            # Cluster ids 1..k (0 stays free for "no data", as in every other classified output of the program)
            classified_image = (labels.reshape(height, width) + 1).astype(np.uint8)

            # The result covers the same ground as the source image: same transform and CRS
            result_prof = georef.result_profile(layer_data.get('profile'), count=1, dtype='uint8',
                                                height=height, width=width)

            # Store the classified image
            self.classified_image = classified_image[np.newaxis, ...]
            self.classified_profile = result_prof

            # Display the clusters in map coordinates, with a legend
            self.display_classified_image(classified_image, profile=result_prof,
                                          names={k + 1: f"Cluster {k + 1}" for k in range(n_clusters)},
                                          title=f"K-Means clustering ({n_clusters} clusters)")

            # Save the segmentation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_layer = f"Result_K_Mean_{timestamp}"
            self.layers[new_layer] = {
                'image': self.classified_image,
                'profile': dict(result_prof),
                'file_name': f"K-Means Segmentation with {n_clusters} Clusters",
                'crs': layer_data.get('crs', None),  # Ensure CRS is included
                'cmap': 'viridis',  # only used by tools that need a continuous map
                'type': 'raster',
                'classified': True,
                'class_names': {k + 1: f"Cluster {k + 1}" for k in range(n_clusters)},
                'title': f"K-Means clustering ({n_clusters} clusters)",
            }
            self.current_layer = new_layer

            # Update UI
            self.layer_list.addItem(QListWidgetItem(new_layer))
            QMessageBox.information(
                self, "Success",
                f"K-Means clustering completed successfully.\n"
                f"Result saved to layer: {new_layer}"
            )

            # Ask user for confirmation to save the classified image
            save_reply = QMessageBox.question(
                self, "Save Classified Image", "Do you want to save the classified image?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if save_reply == QMessageBox.Yes:
                # Save the classified image with the grid (transform + CRS) of the source image
                self.save_classified_image(profile=result_prof)
            else:
                QMessageBox.information(self, "Cancelled", "Classified image saving cancelled by user.")

        except Exception as e:
            error_msg = (
                f"Error: {str(e)}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            QMessageBox.critical(self, "Classification Error", error_msg)
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444
    
    def save_classified_image(self, crs=None, profile=None):
        """Save the last classified/segmented image as a GeoTIFF that keeps the real grid of the source image.

        ``profile``  grid of the result (transform + CRS); defaults to the profile stored with the result.
        ``crs``      only used when the profile carries no CRS. Returns True when a file was written.
        The program never writes a made-up position: without georeferencing the file has none, and says so.
        """
        if not hasattr(self, 'classified_image'):
            QMessageBox.warning(self, "No Image", "No classified image to save.")
            return False

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Classified Image", "", "GeoTIFF (*.tif);;PNG (*.png);;All Files (*)"
            )
            if not file_path:
                return False  # User canceled the save dialog
            arr = np.asarray(self.classified_image)
            if arr.ndim == 3:
                arr = arr[0]
            if profile is None:
                profile = getattr(self, 'classified_profile', None)
            if profile is None or georef.profile_transform_crs(profile)[0] is None:
                profile = self._inherited_profile(arr.shape)
            profile = dict(profile or {})
            if profile.get('crs') is None and crs is not None:
                profile['crs'] = georef.as_crs(crs)

            if not os.path.splitext(file_path)[1]:
                file_path += '.tif'
            if file_path.lower().endswith(('.tif', '.tiff')):
                info = georef.write_geotiff(file_path, arr, profile)
                QMessageBox.information(self, "Saved", f"Classified image saved:\n{file_path}\n\n{georef.georef_note(info)}")
            else:
                from PIL import Image
                Image.fromarray(arr.astype(np.uint8)).save(file_path)
                QMessageBox.information(
                    self, "Saved",
                    f"Classified image saved as a plain picture:\n{file_path}\n\n"
                    "Pictures carry no coordinates. Use GeoTIFF (*.tif) to keep the real position.")
            return True

        except Exception as e:
            error_msg = (
                f"Error: {str(e)}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            QMessageBox.critical(self, "Save Error", error_msg)
            return False

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def mean_shift_segmentation(self):
        """
        Performs Mean-Shift segmentation on the image.
        """
        if not hasattr(self, 'image'):
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        try:
            # Ask the user for the bandwidth parameter
            bandwidth, ok = QInputDialog.getDouble(
                self, "Bandwidth", 
                "Enter bandwidth (e.g., 0.1-1.0):", 
                0.2, 0.01, 1.0, 2
            )
            if not ok:
                QMessageBox.information(self, "Cancelled", "Mean-Shift segmentation cancelled by user.")
                return  # User canceled the input

            # Get the active layer
            if not self.layer_list.currentItem():
                raise ValueError("No active layer selected")

            active_layer = self.layer_list.currentItem().text()
            layer_data = self.layers.get(active_layer, {})
            if not isinstance(layer_data, dict) or 'image' not in layer_data:
                raise ValueError("Invalid layer data structure")

            image_data = layer_data['image']

            
            # Perform Mean-Shift segmentation
            segmented_image, num_clusters = ImageProcessor.mean_shift_segmentation(image_data, bandwidth)  # Call static method

            # Convert the segmented image to uint8 (required for Shapefile saving)
            if int(segmented_image.max()) + 1 > 255:
                raise ValueError(f"Mean-Shift found {int(segmented_image.max()) + 1} segments; at most 255 can be stored. "
                                 "Increase the bandwidth.")
            segmented_image_uint8 = (segmented_image + 1).astype(np.uint8)  # segment ids 1..k, 0 = no data

            # Store the segmented image
            self.classified_image = segmented_image_uint8[np.newaxis, ...]

            # Same ground as the source image: same transform and CRS
            result_prof = georef.result_profile(layer_data.get('profile'), count=1, dtype='uint8',
                                                height=segmented_image_uint8.shape[0], width=segmented_image_uint8.shape[1])
            self.classified_profile = result_prof

            # Display the segments in map coordinates, with a legend
            self.display_classified_image(segmented_image_uint8, profile=result_prof,
                                          title=f"Mean-Shift segmentation ({num_clusters} segments)")

            # Save the segmentation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_layer = f"Result_Mean_Shift_{timestamp}"
            self.layers[new_layer] = {
                'image': self.classified_image,
                'profile': dict(result_prof),
                'file_name': f"Mean-Shift Segmentation with Bandwidth {bandwidth}",
                'type': 'raster',
                'classified': True,
                'title': f"Mean-Shift segmentation ({num_clusters} segments)",
            }
            self.current_layer = new_layer

            # Update UI
            self.layer_list.addItem(QListWidgetItem(new_layer))
            QMessageBox.information(
                self, "Success",
                f"Mean-Shift segmentation completed successfully.\n"
                f"Result saved to layer: {new_layer}"
            )

            # Ask user for confirmation to save the segmented image
            save_reply = QMessageBox.question(
                self, "Save Segmented Image", "Do you want to save the segmented image?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if save_reply == QMessageBox.Yes:
                # Save the segmented image with the grid (transform + CRS) of the source image
                self.save_classified_image(profile=result_prof)
            else:
                QMessageBox.information(self, "Cancelled", "Segmented image saving cancelled by user.")

        except Exception as e:
            error_msg = (
                f"Error: {str(e)}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            QMessageBox.critical(self, "Segmentation Error", error_msg)


    def analyze_land_use(self):
        try:
            if hasattr(self, 'analysis_data'):
                del self.analysis_data

            if not hasattr(self, 'classified_image') or self.classified_image is None:
                QMessageBox.warning(self, "Input Error", "Please classify an image first")
                return

            class_labels = self.load_class_labels()
            self.validate_class_labels(class_labels)

            classified_data = self.classified_image[0]
            unique_classes, counts = np.unique(classified_data, return_counts=True)
            total_pixels = classified_data.size
            percentages = (counts / total_pixels) * 100

            pixel_area = None
            if hasattr(self, 'profile') and self.profile:
                transform = self.profile['transform']
                pixel_area = abs(transform.a * transform.e)

            total_area = (total_pixels * pixel_area / 10000) if pixel_area else None

            self.analysis_data = {
                'classes': unique_classes,
                'counts': counts,
                'percentages': percentages,
                'class_labels': class_labels,
                'total_pixels': total_pixels,
                'total_area': total_area,
                'pixel_area': pixel_area
            }

            report = self.generate_analysis_report()
            self.show_results_dialog(report)

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {str(e)}")

    def generate_analysis_report(self):
        if not hasattr(self, 'analysis_data') or not self.analysis_data:
            return "Error: No land use analysis data available"

        data = self.analysis_data
        report = [
            "Land Use Classification Report",
            "==============================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total Pixels Analyzed: {data['total_pixels']:,}",
            f"Total Area: {data['total_area']:.2f} hectares" if data['total_area'] else "Spatial reference not available",
            "",
            "Class Distribution:",
            "-------------------"
        ]

        for cls, count, percent in zip(data['classes'], data['counts'], data['percentages']):
            class_info = data['class_labels'].get(str(cls), {'name': f'Class {cls}', 'color': '#CCCCCC'})
            report.append(
                f"{class_info['name']} ({class_info['color']}):\n"
                f"  - Percentage: {percent:.2f}%\n"
                f"  - Pixel Count: {count:,}\n"
                f"  - Estimated Area: {count * data['pixel_area'] / 10000:.2f} ha" if data['pixel_area'] else ""
            )

        return "\n".join(report)

    def show_results_dialog(self, report):
        if not report:
            QMessageBox.warning(self, "No Data", "No analysis results to display")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis Results")
        dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        layout = QVBoxLayout()

        text_edit = QTextEdit()
        text_edit.setPlainText(report)
        layout.addWidget(text_edit)

        fig = plt.Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        data = self.analysis_data
        labels = [data['class_labels'].get(str(c), {}).get('name', f'Class {c}') 
                for c in data['classes']]
        ax.pie(data['percentages'], labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title("Land Use Distribution")
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        btn_frame = QFrame()
        btn_layout = QHBoxLayout()

        btn_save = QPushButton("Save Report")
        btn_save.clicked.connect(lambda: self.save_analysis_report(report))

        btn_export = QPushButton("Export to Excel")
        btn_export.clicked.connect(self.export_analysis_data)

        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_export)
        btn_frame.setLayout(btn_layout)
        layout.addWidget(btn_frame)

        dialog.setLayout(layout)
        dialog.exec_()

    def save_analysis_report(self, report):
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Analysis Report",
                "",
                "Text Files (*.txt);;All Files (*)"
            )

            if file_name:
                if not file_name.lower().endswith('.txt'):
                    file_name += '.txt'

                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(report)

                QMessageBox.information(self, "Success", 
                                    "Report saved successfully", 
                                    QMessageBox.Ok)

        except Exception as e:
            QMessageBox.critical(self, "Save Error", 
                            f"Failed to save report: {str(e)}")

    def export_analysis_data(self):
        try:
            if not hasattr(self, 'analysis_data'):
                raise ValueError("No analysis data to export")

            data = self.analysis_data
            df = pd.DataFrame({
                'Class Code': data['classes'],
                'Class Name': [data['class_labels'].get(str(c), {}).get('name', 'Unknown') 
                            for c in data['classes']],
                'Class Color': [data['class_labels'].get(str(c), {}).get('color', '#CCCCCC') 
                            for c in data['classes']],
                'Pixel Count': data['counts'],
                'Percentage': data['percentages']
            })

            if data['pixel_area']:
                df['Area (m²)'] = df['Pixel Count'] * data['pixel_area']
                df['Area (ha)'] = df['Area (m²)'] / 10000

            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Export to Excel",
                "",
                "Excel Files (*.xlsx);;All Files (*)"
            )

            if file_name:
                if not file_name.lower().endswith('.xlsx'):
                    file_name += '.xlsx'

                df.to_excel(file_name, index=False, engine='openpyxl')
                QMessageBox.information(self, "Success", 
                                    "Data exported to Excel successfully", 
                                    QMessageBox.Ok)

        except ImportError:
            QMessageBox.critical(self, "Dependency Missing",
                            "Required packages not found:\n"
                            "Please install pandas and openpyxl")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                            f"Failed to export data: {str(e)}")

    def load_class_labels(self):
        try:
            default_path = "class_labels.json"
            if os.path.exists(default_path):
                with open(default_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                    self.validate_class_labels(labels)
                    return labels

            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Class Labels",
                "",
                "JSON Files (*.json);;All Files (*)"
            )

            if file_name:
                with open(file_name, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                    self.validate_class_labels(labels)
                    return labels

            return {}

        except json.JSONDecodeError:
            QMessageBox.critical(self, "Invalid File", 
                            "Selected file is not valid JSON")
            return {}
        except Exception as e:
            QMessageBox.warning(self, "Loading Error", 
                            f"Error loading class labels: {str(e)}")
            return {}

    def validate_class_labels(self, labels):
        seen_names = set()
        seen_colors = set()

        for key, value in labels.items():
            if not isinstance(value, dict):
                raise ValueError(f"Invalid format for class {key}")

            name = value.get('name', f'Class {key}')
            color = value.get('color', '#CCCCCC').upper()

            if name in seen_names:
                raise ValueError(f"Duplicate class name detected: {name} (Class ID: {key})")
            if color in seen_colors:
                raise ValueError(f"Duplicate color code detected: {color} (Class ID: {key})")
            if not re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color):
                raise ValueError(f"Invalid color code: {color}")

            seen_names.add(name)
            seen_colors.add(color)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ old 
    def ask_mask_processing_option(self):
        """Dialog to choose between raw export or classification."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Mask Processing Options")
        layout = QVBoxLayout(dialog)
        
        label = QLabel("How would you like to process the generated segments?")
        layout.addWidget(label)
        
        btn_group = QButtonGroup()
        
        # Option 1: Export raw masks
        export_btn = QRadioButton("Export raw masks (TIFF + vector formats)")
        export_btn.setChecked(True)
        btn_group.addButton(export_btn)
        layout.addWidget(export_btn)
        
        # Option 2: Classify masks
        classify_btn = QRadioButton("Classify masks for land use map")
        btn_group.addButton(classify_btn)
        layout.addWidget(classify_btn)
        
        # Dialog buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        if dialog.exec_() == QDialog.Accepted:
            return "export_raw" if export_btn.isChecked() else "classify"
        return None
    
    def export_to_autocad(self, filename, format_type='dxf'):
        """
        Export masks to AutoCAD format (DXF or DWG) with organized layers
        
        Parameters:
            filename (str): Output file path
            format_type (str): 'dxf' or 'dwg' (default: 'dxf')
        """
        try:
            if getattr(self, 'masks_gdf', None) is None:
                QMessageBox.warning(self, "No segments", "Export the segmentation masks first (nothing to write to DXF).")
                return
            import ezdxf
            from ezdxf.addons import geo
            from ezdxf.colors import RGB

            # Create a new DXF document
            doc = ezdxf.new('R2018')  # AutoCAD 2018 format
            msp = doc.modelspace()

            # Create organized layers with proper color assignments
            layers = {
                'polygons': doc.layers.add("MASK_POLYGONS"),
                'holes': doc.layers.add("MASK_HOLES"),
                'multipolygons': doc.layers.add("MASK_MULTIPOLYGONS"),
                'crs': doc.layers.add("CRS_INFO")
            }
            
            # Set layer properties using valid AutoCAD color indices (1-255)
            layers['polygons'].color = 1       # Red
            layers['holes'].color = 2         # Yellow
            layers['multipolygons'].color = 3  # Green
            layers['crs'].color = 5           # Blue
            
            # Set lineweights to valid values (-3 = Default)
            for layer in layers.values():
                layer.dxf.lineweight = -3

            # Add CRS/projection information if available
            if hasattr(self, 'profile') and 'crs' in self.profile:
                crs = self.profile['crs']
                try:
                    # Newer versions of ezdxf might use this approach
                    geo_proxy = geo.GeoProxy(crs.to_wkt())
                    msp.add_entity(geo_proxy)
                    # Set layer for the proxy
                    geo_proxy.dxf.layer = 'CRS_INFO'
                except AttributeError:
                    try:
                        # Older versions might work with this
                        proxy = geo.proxy(msp, crs.to_wkt())
                        proxy.dxf.layer = 'CRS_INFO'
                    except Exception as e:
                        logging.info(f"Could not add CRS information: {str(e)}")
            # Convert each geometry to DXF entities with proper layer assignment
            for geom in self.masks_gdf.geometry:
                if geom.geom_type == 'Polygon':
                    # Main polygon outline
                    points = list(geom.exterior.coords)
                    poly = msp.add_lwpolyline(points, format='xy')
                    poly.dxf.layer = 'MASK_POLYGONS'
                    
                    # Interior holes
                    for interior in geom.interiors:
                        points = list(interior.coords)
                        hole = msp.add_lwpolyline(points, format='xy')
                        hole.dxf.layer = 'MASK_HOLES'
                        
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        # Main multipolygon outline
                        points = list(poly.exterior.coords)
                        mpoly = msp.add_lwpolyline(points, format='xy')
                        mpoly.dxf.layer = 'MASK_MULTIPOLYGONS'
                        
                        # Interior holes
                        for interior in poly.interiors:
                            points = list(interior.coords)
                            hole = msp.add_lwpolyline(points, format='xy')
                            hole.dxf.layer = 'MASK_HOLES'

            # Save the file with appropriate format
            if format_type.lower() == 'dwg':
                try:
                    doc.saveas(filename, encoding='utf-8')
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "DWG Export Warning",
                        "DWG export requires additional dependencies. Falling back to DXF."
                    )
                    filename = filename.replace('.dwg', '.dxf')
                    doc.saveas(filename)
            else:
                if not filename.lower().endswith('.dxf'):
                    filename += '.dxf'
                doc.saveas(filename)

            return True

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Export Error", 
                f"Failed to export AutoCAD file:\n{str(e)}"
            )
            logging.info(f"Error: {traceback.format_exc()}")
            return False
    
    def export_raw_masks(self):
        """Export masks in various formats while preserving CRS from the original image"""
        try:
            # Verify an image is loaded with proper profile data
            if not hasattr(self, 'profile') or not hasattr(self, 'image'):
                QMessageBox.warning(self, "Warning", "No image has been loaded")
                return
            if not getattr(self, 'masks', None):
                QMessageBox.warning(self, "No segments", "Run Segment Anything first: there are no masks to export.")
                return

            # Create mask image with uint16 dtype
            mask_image = np.zeros(self.image.shape[1:], dtype=np.uint16)
            
            # Validate segment count
            if len(self.masks) > 65535:
                QMessageBox.warning(
                    self,
                    "Too Many Segments",
                    f"Cannot export {len(self.masks)} segments (maximum is 65535)"
                )
                return
                
            # Assign unique values to each mask
            for i, mask in enumerate(self.masks, 1):
                mask_image[mask['segmentation']] = i

            # Get CRS and transform from original image profile
            crs = self.profile.get('crs')
            transform = self.profile.get('transform')
            
            if crs is None:
                QMessageBox.warning(self, "Warning", "The original image has no CRS information")
                return

            # Save as GeoTIFF with original CRS and transform
            with rasterio.open(
                "raw_masks.tif",
                'w',
                driver='GTiff',
                height=mask_image.shape[0],
                width=mask_image.shape[1],
                count=1,
                dtype=mask_image.dtype,
                crs=crs,
                transform=transform,
                nodata=0
            ) as dst:
                dst.write(mask_image, 1)

            # Convert to vector format
            with rasterio.open("raw_masks.tif") as src:
                shapes = features.shapes(
                    src.read(1).astype(np.uint16),
                    transform=src.transform
                )
                geometries = []
                values = []
                for geom, val in shapes:
                    if val > 0:
                        geometries.append(shape(geom))
                        values.append(int(val))

            # Create GeoDataFrame with original CRS
            self.masks_gdf = gpd.GeoDataFrame(
                {
                    'geometry': geometries,
                    'value': values,
                    'area': [self.masks[int(v)-1]['area'] for v in values],
                    'bbox': [self.masks[int(v)-1]['bbox'] for v in values],
                    'predicted_iou': [self.masks[int(v)-1]['predicted_iou'] for v in values],
                    'stability_score': [self.masks[int(v)-1]['stability_score'] for v in values]
                },
                crs=crs
            )

            # Calculate geometric properties
            self.masks_gdf['geometry'] = self.masks_gdf.geometry.simplify(tolerance=0.01)
            self.masks_gdf['area_m2'] = self.masks_gdf.geometry.area.round(2)
            bounds = self.masks_gdf.bounds
            self.masks_gdf['width'] = (bounds['maxx'] - bounds['minx']).round(2)
            self.masks_gdf['height'] = (bounds['maxy'] - bounds['miny']).round(2)
            self.masks_gdf['perimeter'] = self.masks_gdf.geometry.length.round(2)
            self.masks_gdf['aspect_ratio'] = (self.masks_gdf['width'] / self.masks_gdf['height']).round(2)
            
            # Export to multiple formats
            self.masks_gdf.to_file("raw_masks.gpkg", driver="GPKG")
            self.masks_gdf.to_file("raw_masks.shp", driver="ESRI Shapefile")
            
            # Export to AutoCAD formats
            self.export_to_autocad("raw_masks.dxf")
            #self.export_to_autocad("raw_masks.dwg", format_type='dwg')

            # Export properties to CSV
            csv_columns = ['value', 'area', 'area_m2', 'width', 'height', 
                        'perimeter', 'aspect_ratio', 'predicted_iou', 'stability_score']
            self.masks_gdf.to_csv("raw_masks_info.csv", columns=csv_columns, index=False)

            # Success message
            QMessageBox.information(
                self,
                "Export Successful",
                f"Masks exported with CRS preservation ({crs}):\n"
                "- raw_masks.tif (GeoTIFF)\n"
                "- raw_masks.gpkg (GeoPackage)\n"
                "- raw_masks.shp (Shapefile)\n"
                "- raw_masks.dxf (AutoCAD DXF)\n"
                "- raw_masks_info.csv (properties)"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Export Error", 
                f"Failed to export masks:\n{str(e)}\n\n{traceback.format_exc()}"
            )
            logging.info(f"Error: {traceback.format_exc()}")
    def auto_classify_by_size(self):
        """Advanced automatic size-based classification using statistical analysis"""
        if not hasattr(self, 'masks') or not self.masks:
            QMessageBox.warning(self, "Warning", "No masks available for classification")
            return
        
        try:
            # Calculate area statistics
            areas = np.array([m['area'] for m in self.masks])
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            
            # Calculate dynamic thresholds
            small_thresh = mean_area - std_area
            large_thresh = mean_area + std_area
            
            # Advanced size classification with descriptive names
            size_classes = {
                0: "Very Small",
                1: "Small",
                2: "Medium",
                3: "Large",
                4: "Very Large"
            }
            
            for mask in self.masks:
                if mask['area'] < small_thresh/2:
                    class_name = size_classes[0]
                elif mask['area'] < small_thresh:
                    class_name = size_classes[1]
                elif mask['area'] < large_thresh:
                    class_name = size_classes[2]
                elif mask['area'] < large_thresh*1.5:
                    class_name = size_classes[3]
                else:
                    class_name = size_classes[4]
                
                # Add class if not exists
                if class_name not in self.classes:
                    color = self.generate_distinct_color(len(self.classes))
                    self._add_class_to_combobox(class_name, len(self.classes)+1, color)
                
                # Assign mask to class
                self.class_masks[class_name] = np.logical_or(
                    self.class_masks.get(class_name, np.zeros_like(mask['segmentation'])), 
                    mask['segmentation'])
            
            # Log classification statistics
            self.log_classification_stats(areas)
            self.update_mask_display(0)
            QMessageBox.information(self, "Success", "Size-based classification completed successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Size classification failed: {str(e)}")

    def auto_classify_by_position(self):
        """Advanced automatic position-based classification with region detection"""
        if not hasattr(self, 'masks') or not self.masks:
            QMessageBox.warning(self, "Warning", "No masks available for classification")
            return
        
        try:
            img_height, img_width = self.masks[0]['segmentation'].shape
            
            # Analyze mask distribution
            positions = np.array([[m['bbox'][0]+m['bbox'][2]/2, 
                                m['bbox'][1]+m['bbox'][3]/2] for m in self.masks])
            
            # Smart clustering using density estimation
            from sklearn.cluster import MeanShift
            clustering = MeanShift(bandwidth=min(img_width, img_height)/4).fit(positions)
            
            # Position-based classification scheme
            position_classes = {
                -1: "Isolated",
                0: "Central",
                1: "Peripheral",
                2: "Corner"
            }
            
            for i, mask in enumerate(self.masks):
                x, y = positions[i]
                label = clustering.labels_[i]
                
                if label == -1:  # Isolated masks
                    class_name = position_classes[-1]
                else:
                    # Precise region detection
                    if x > img_width*0.4 and x < img_width*0.6 and y > img_height*0.4 and y < img_height*0.6:
                        class_name = position_classes[0]  # Central
                    elif x < img_width*0.3 or x > img_width*0.7 or y < img_height*0.3 or y > img_height*0.7:
                        class_name = position_classes[2]  # Corner
                    else:
                        class_name = position_classes[1]  # Peripheral
                    
                    class_name = f"{class_name}_{label}"
                
                # Add class if not exists
                if class_name not in self.classes:
                    color = self.generate_distinct_color(len(self.classes))
                    self._add_class_to_combobox(class_name, len(self.classes)+1, color)
                
                # Assign mask to class
                self.class_masks[class_name] = np.logical_or(
                    self.class_masks.get(class_name, np.zeros_like(mask['segmentation'])), 
                    mask['segmentation'])
            
            # Log spatial distribution statistics
            self.log_position_stats(positions, clustering.labels_)
            self.update_mask_display(0)
            QMessageBox.information(self, "Success", "Position-based classification completed successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Position classification failed: {str(e)}")


    def generate_distinct_color(self, index):
        """Generate distinct colors for different classes"""
        hues = np.linspace(0, 1, 8, endpoint=False)  # 8 distinct hues
        saturation = 0.7 + (index % 3) * 0.1
        value = 0.8 - (index % 4) * 0.1
        hue = hues[index % len(hues)]
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (int(r*255), int(g*255), int(b*255))

    def log_classification_stats(self, areas):
        """Log classification statistics"""
        stats = {
            "Total masks": len(areas),
            "Mean area": np.mean(areas),
            "Min area": np.min(areas),
            "Max area": np.max(areas),
            "Standard deviation": np.std(areas)
        }
        logging.info("Classification statistics: %s", stats)
    def log_position_stats(self, positions, labels):
        """Log spatial distribution statistics"""
        unique, counts = np.unique(labels, return_counts=True)
        stats = {
            "Cluster count": len(unique) - (1 if -1 in unique else 0),
            "Isolated masks": counts[-1] if -1 in unique else 0,
            "Region distribution": dict(zip(unique, counts))
        }
        logging.info("Position statistics: %s", stats)
    def _add_class_to_combobox(self, name, value, color=None):
        """Helper to add a new class to the combo box and tracking structures."""
        if name not in self.classes:
            self.classes[name] = value
            self.class_combo.addItem(name)
            self.class_masks[name] = np.zeros_like(self.masks[0]['segmentation'], dtype=bool)
            if color is not None:
                if not hasattr(self, 'class_colors'):
                    self.class_colors = {}
                self.class_colors[name] = color

    def classify_automatic_masks(self):
        """Improved classification dialog with batch operations."""
        if not hasattr(self, 'masks') or not self.masks:
            QMessageBox.warning(self, "No Segments", "No segments available for classification.")
            return
        
        # Ensure classes are defined
        if not hasattr(self, 'classes') or not self.classes:
            if not self.ask_for_classes():
                return
        
        # Initialize classification storage
        self.class_masks = {
            name: np.zeros(self.masks[0]['segmentation'].shape, dtype=bool)
            for name in self.classes
        }
        self.current_mask_index = 0
        
        # Create classification dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Segment Classification")
        dialog.setMinimumSize(800, 600)
        layout = QVBoxLayout(dialog)
        
        # Visualization panel
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, dialog)
        
        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Class selection
        class_group = QGroupBox("Classification")
        class_layout = QVBoxLayout(class_group)
        
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.classes.keys())
        class_layout.addWidget(QLabel("Assign to class:"))
        class_layout.addWidget(self.class_combo)
        
        # Segment info
        self.segment_info = QLabel()
        class_layout.addWidget(self.segment_info)
        
        # Navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)
        
        nav_buttons = QHBoxLayout()
        self.prev_btn = QPushButton("◄ Previous")
        self.next_btn = QPushButton("Next ►")
        self.prev_btn.clicked.connect(lambda: self.update_mask_display(-1))
        self.next_btn.clicked.connect(lambda: self.update_mask_display(1))
        nav_buttons.addWidget(self.prev_btn)
        nav_buttons.addWidget(self.next_btn)
        
        # Assignment buttons
        assign_buttons = QHBoxLayout()
        self.assign_btn = QPushButton("Assign Current")
        self.assign_all_btn = QPushButton("Assign All")
        self.assign_btn.clicked.connect(self.assign_current_mask)
        self.assign_all_btn.clicked.connect(self.assign_all_masks)
        assign_buttons.addWidget(self.assign_btn)
        assign_buttons.addWidget(self.assign_all_btn)
        
        # Batch operations group
        batch_group = QGroupBox("Batch Operations")
        batch_layout = QHBoxLayout(batch_group)
        
        size_class_btn = QPushButton("Auto-Classify by Size")
        size_class_btn.clicked.connect(self.auto_classify_by_size)
        batch_layout.addWidget(size_class_btn)
        
        loc_class_btn = QPushButton("Auto-Classify by Position")
        loc_class_btn.clicked.connect(self.auto_classify_by_position)
        batch_layout.addWidget(loc_class_btn)
        
        # Assemble navigation group
        nav_layout.addLayout(nav_buttons)
        nav_layout.addLayout(assign_buttons)
        
        # Add to control panel
        control_layout.addWidget(class_group)
        control_layout.addWidget(nav_group)
        control_layout.addWidget(batch_group)
        
        # Dialog buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        
        # Final layout assembly
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(control_panel)
        layout.addWidget(btn_box)
        
        # Initial display
        self.update_mask_display(0)
        
        # Execute dialog
        if dialog.exec_() == QDialog.Accepted:
            self.generate_landuse_map()
        else:
            self.class_masks = None



    def _load_sam_model(self):
        """Return (model, device) or (None, None). Asks for the .pth file once and remembers it."""
        if not SAM_AVAILABLE:
            QMessageBox.warning(self, "SAM not installed",
                                "SAM needs PyTorch and segment-anything:\n\n"
                                "    pip install torch segment-anything\n\n"
                                "Weights: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            return None, None
        settings = self._snap_qsettings()
        model_type = settings.value("sam/model_type", "vit_h", type=str)
        ckpt = resources.find_sam_checkpoint(model_type, settings.value("sam/checkpoint", "", type=str) or None)
        if ckpt is None:
            ckpt, _ = QFileDialog.getOpenFileName(
                self, "Select SAM checkpoint (sam_vit_h / vit_l / vit_b .pth)", "", "PyTorch weights (*.pth)")
            if not ckpt:
                return None, None
        model_type = resources.model_type_from_filename(ckpt) or model_type
        settings.setValue("sam/checkpoint", ckpt)
        settings.setValue("sam/model_type", model_type)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to(device=device)
        return sam, device

    def sam_segmentation(self):
        """Perform SAM segmentation with options for classification or raw export."""
        if not hasattr(self, 'image'):
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        progress = None
        try:
            # Show loading dialog
            progress = QProgressDialog(
                "Initializing SAM...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            # Initialize SAM model (weights located via settings / SAM_CHECKPOINT_DIR / ./models)
            sam, device = self._load_sam_model()
            if sam is None:
                progress.close()
                return
            
            # Prepare image
            if len(self.image.shape) == 3 and self.image.shape[0] == 3:
                image_data = self.image.transpose(1, 2, 0)
            else:
                image_data = np.stack([self.image]*3, axis=-1)
            
            image_data = (image_data * 255).astype(np.uint8)
            
            # Generate masks
            progress.setLabelText("Generating segments...")
            QApplication.processEvents()
            
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=50,
                pred_iou_thresh=0.9,
                stability_score_thresh=0.96,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            
            self.masks = mask_generator.generate(image_data)
            progress.close()
            
            if not self.masks:
                QMessageBox.warning(self, "No Segments", "No segments were generated.")
                return
            
            # Ask user what they want to do with the masks
            choice = self.ask_mask_processing_option()
            
            if choice == "export_raw":
                self.export_raw_masks()
            elif choice == "classify":
                self.show_automatic_masks(image_data, self.masks)
                self.classify_automatic_masks()
            else:
                return  # User canceled

        except Exception as e:
            if progress is not None:
                progress.close()
            QMessageBox.critical(
                self, 
                "Segmentation Error", 
                f"Failed to perform SAM segmentation:\n{str(e)}"
            )
            logging.info(f"Error: {traceback.format_exc()}")
    def show_automatic_masks(self, image, masks):
        """Display the automatically generated masks with proper figure management."""
        # Close any existing figure
        if self.current_layer:
            plt.close(self.current_layer)
        
        # Create new figure
        self.current_layer = plt.figure(figsize=(10, 10))
        self.figure_initialized = True
        
        try:
            plt.imshow(image)
            
            if len(masks) == 0:
                plt.close(self.current_layer)
                self.current_layer = None
                return
            
            sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)
            
            for ann in sorted_anns:
                m = ann['segmentation']
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.35)))
            
            plt.axis('off')
            plt.title("Generated Segments - Close to Continue")
            
            # Store canvas reference
            self.current_canvas = self.current_layer.canvas
            
            # Use non-blocking show
            plt.show(block=False)
            plt.pause(0.1)
            
        except Exception as e:
            if self.current_layer:
                plt.close(self.current_layer)
                self.current_layer = None
            raise e

    def cleanup_figures(self):
        """Clean up matplotlib figures before layer switching."""
        if self.current_layer:
            try:
                plt.close(self.current_layer)
            except Exception:
                pass
            self.current_layer = None
        self.current_canvas = None
        self.figure_initialized = False

        # Your existing layer switching code here

    def closeEvent(self, event):
        """Handle application closing: stop background jobs first, a running QThread must never be destroyed."""
        for attr, th in list(vars(self).items()):
            if not isinstance(th, QThread):
                continue
            try:
                if hasattr(th, 'cancel'):
                    th.cancel()
                if th.isRunning():
                    th.quit()
                    if not th.wait(3000):
                        logging.warning("Background job %s did not stop in time", attr)
            except RuntimeError:      # underlying C++ object already deleted
                pass
        try:
            self.cleanup_figures()
        except Exception:
            logging.exception("cleanup_figures failed")
        event.accept()
        


    def assign_current_mask(self):
        """Assign current segment to selected class."""
        if not getattr(self, 'masks', None) or not isinstance(getattr(self, 'class_masks', None), dict) \
                or not hasattr(self, 'class_combo'):
            QMessageBox.warning(self, "No segments", "Run automatic segmentation and open the classification dialog first.")
            return
        class_name = self.class_combo.currentText()
        if not class_name:
            QMessageBox.warning(self, "No Class", "Please select a class first.")
            return
        
        mask = self.masks[self.current_mask_index]['segmentation']
        self.class_masks[class_name] = np.logical_or(
            self.class_masks[class_name], mask)
        
        # Auto-advance to next segment
        if self.current_mask_index < len(self.masks) - 1:
            self.update_mask_display(1)

    def assign_all_masks(self):
        """Assign all segments to selected class with proper message box usage."""
        if not getattr(self, 'masks', None) or not isinstance(getattr(self, 'class_masks', None), dict) \
                or not hasattr(self, 'class_combo'):
            QMessageBox.warning(self, "No segments", "Run automatic segmentation and open the classification dialog first.")
            return
        class_name = self.class_combo.currentText()
        if not class_name:
            QMessageBox.warning(self, "No Class", "Please select a class first.")
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Assignment",
            f"Assign all {len(self.masks)} segments to '{class_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Create progress dialog
            progress = QProgressDialog(
                "Assigning segments...", 
                "Cancel", 
                0, 
                len(self.masks), 
                self
            )
            progress.setWindowModality(Qt.WindowModal)
            
            combined_mask = np.zeros_like(self.masks[0]['segmentation'], dtype=bool)
            
            for i, mask in enumerate(self.masks):
                if progress.wasCanceled():
                    break
                    
                combined_mask = np.logical_or(combined_mask, mask['segmentation'])
                progress.setValue(i + 1)
                QApplication.processEvents()
            
            if not progress.wasCanceled():
                self.class_masks[class_name] = combined_mask
                QMessageBox.information(
                    self,
                    "Assignment Complete",
                    f"All segments assigned to '{class_name}'"
                )
            
            progress.close()

    def update_mask_display(self, delta=0):
        """Update the displayed segment with navigation."""
        if not getattr(self, 'masks', None):
            return
        new_index = self.current_mask_index + delta
        if 0 <= new_index < len(self.masks):
            self.current_mask_index = new_index
        
        mask = self.masks[self.current_mask_index]
        
        # Update info label
        self.segment_info.setText(
            f"Segment {self.current_mask_index + 1}/{len(self.masks)}\n"
            f"Size: {mask['area']} px\n"
            f"BBox: {mask['bbox']}"
        )
        
        # Update navigation buttons
        self.prev_btn.setEnabled(self.current_mask_index > 0)
        self.next_btn.setEnabled(self.current_mask_index < len(self.masks) - 1)
        
        # Update visualization
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if len(self.image.shape) == 3 and self.image.shape[0] == 3:
            img_data = self.image.transpose(1, 2, 0)
        else:
            img_data = np.stack([self.image]*3, axis=-1)
        
        img_data = (img_data * 255).astype(np.uint8)
        ax.imshow(img_data)
        ax.imshow(mask['segmentation'], alpha=0.5, cmap='viridis')
        ax.set_title(f"Segment {self.current_mask_index + 1}")
        ax.axis('off')
        self.canvas.draw()
    
    def ask_for_classes(self):
        """Dialog for defining land use classes with proper layout management."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Define Land Use Classes")
        dialog.setMinimumSize(500, 400)
        
        # Main layout
        main_layout = QVBoxLayout(dialog)
        
        # Form layout inside scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        self.form_layout = QFormLayout(form_widget)
        self.form_layout.setVerticalSpacing(10)
        self.form_layout.setHorizontalSpacing(15)
        self.form_layout.setContentsMargins(10, 10, 10, 10)
        
        # Initialize class entries
        self.class_entries = []
        
        # Add initial class row
        self._add_class_row()
        
        scroll.setWidget(form_widget)
        main_layout.addWidget(scroll)
        
        # Button row
        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add Class")
        done_btn = QPushButton("Done")
        
        add_btn.clicked.connect(lambda: self._add_class_row())
        done_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(add_btn)
        button_layout.addWidget(done_btn)
        main_layout.addLayout(button_layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # Store classes as dictionary
            self.classes = {
                entry[0].text().strip(): entry[1].value()
                for entry in self.class_entries
                if entry[0].text().strip()
            }
            
            # Initialize class masks storage if masks exist
            if hasattr(self, 'masks') and self.masks:
                mask_shape = self.masks[0]['segmentation'].shape
                self.class_masks = {
                    name: np.zeros(mask_shape, dtype=bool) 
                    for name in self.classes
                }
            return bool(self.classes)
        return False
    
    def _add_class_row(self):
        """Add a properly formatted class definition row to the form."""
        row_index = len(self.class_entries)
        
        # Create widgets
        name_edit = QLineEdit()
        name_edit.setMinimumWidth(200)
        
        value_spin = QSpinBox()
        value_spin.setRange(1, 255)
        value_spin.setValue(row_index + 1)
        value_spin.setMaximumWidth(80)
        
        # Set default class names for first few rows
        default_classes = ["Urban", "Vegetation", "Water", "Barren"]
        if row_index < len(default_classes):
            name_edit.setText(default_classes[row_index])
        
        # Create horizontal layout for inputs
        input_layout = QHBoxLayout()
        input_layout.addWidget(name_edit)
        input_layout.addWidget(value_spin)
        
        # Add remove button for non-first rows
        if row_index > 0:
            remove_btn = QPushButton("Remove")
            remove_btn.setMaximumWidth(80)
            remove_btn.clicked.connect(
                lambda: self._remove_class_row(row_index))
            input_layout.addWidget(remove_btn)
        
        # Add to form layout
        self.form_layout.addRow(QLabel(f"Class {row_index + 1}:"), input_layout)
        
        # Store references
        self.class_entries.append((name_edit, value_spin))

    def _remove_class_row(self, index):
        """Safely remove a class definition row."""
        if 0 < index < len(self.class_entries):
            # Remove from layout
            self.form_layout.removeRow(index)
            
            # Remove from tracking list
            del self.class_entries[index]
            
            # Renumber remaining rows
            for i, entry in enumerate(self.class_entries):
                # Update label
                label_item = self.form_layout.itemAt(i, QFormLayout.LabelRole)
                if label_item and label_item.widget():
                    label_item.widget().setText(f"Class {i + 1}:")
                
                # Update spinbox value
                entry[1].setValue(i + 1)
    def generate_landuse_map(self):
        """Generate landuse map with enhanced legend showing class names and colors."""
        try:
            # Validate inputs
            if not hasattr(self, 'class_masks') or not self.class_masks:
                QMessageBox.warning(self, "No Data", "No classifications available.")
                return
            if not hasattr(self, 'masks') or not self.masks:
                QMessageBox.warning(self, "No Segments", "No segmentation data available.")
                return

            # Create landuse map array
            landuse_map = np.zeros(self.masks[0]['segmentation'].shape, dtype=np.uint8)
            
            # Apply classifications
            for class_name, mask in self.class_masks.items():
                if np.any(mask):
                    landuse_map[mask] = self.classes[class_name]

            # Create visualization with proper legend
            self.display_landuse_map_with_legend(landuse_map)

            # Save outputs
            output_path = self.save_landuse_outputs_with_legend(landuse_map)
            
            if output_path:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Land use map saved to:\n{output_path}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Generation Error",
                f"Failed to generate landuse map:\n{str(e)}"
            )
            logging.info(f"Error: {traceback.format_exc()}")
    def display_landuse_map_with_legend(self, landuse_map):
        """Display map with legend showing class names and colors."""
        plt.figure(figsize=(14, 10))
        
        # Create colormap with consistent colors for classes
        unique_classes = sorted(list(set(self.classes.values())))
        class_names = {v:k for k,v in self.classes.items()}
        
        # Create colormap and legend elements
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(len(unique_classes)+1)-0.5, len(unique_classes))
        
        # Plot the map in its real coordinates
        img, _spec = georef.draw_raster(plt.gca(), landuse_map, self._inherited_profile(landuse_map.shape),
                                        cmap=cmap, norm=norm, interpolation='nearest')
        
        # Create legend patches
        patches = [mpatches.Patch(color=colors[i], 
                                label=f"{class_names[cls]} ({cls})") 
                for i, cls in enumerate(unique_classes)]
        
        # Add legend
        plt.legend(handles=patches, 
                bbox_to_anchor=(1.05, 1), 
                loc='upper left', 
                borderaxespad=0.,
                title="Land Use Classes")
        
        plt.title("Land Use Classification Map")
        plt.tight_layout()
        plt.show(block=False)

    def save_landuse_outputs_with_legend(self, landuse_map):
        """Save map with legend to multiple formats."""
        output_path = ""
        
        try:
            # Create figure for saving
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Same visualization as display function
            unique_classes = sorted(list(set(self.classes.values())))
            class_names = {v:k for k,v in self.classes.items()}
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))
            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(np.arange(len(unique_classes)+1)-0.5, len(unique_classes))
            
            lu_profile = self._inherited_profile(landuse_map.shape)
            img, _spec = georef.draw_raster(ax, landuse_map, lu_profile, cmap=cmap, norm=norm, interpolation='nearest')
            
            # Create and add legend
            patches = [mpatches.Patch(color=colors[i], 
                                    label=f"{class_names[cls]} ({cls})") 
                    for i, cls in enumerate(unique_classes)]
            ax.legend(handles=patches, 
                    bbox_to_anchor=(1.05, 1), 
                    loc='upper left', 
                    borderaxespad=0.,
                    title="Land Use Classes")
            
            ax.set_title("Land Use Classification Map")
            plt.tight_layout()
            
            # Save visual representation
            output_path = "landuse_classification.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # GeoTIFF with the real grid of the source image (only when the image is georeferenced)
            if georef.is_georeferenced(*georef.profile_transform_crs(lu_profile)):
                geotiff_path = "landuse_classification.tif"
                georef.write_geotiff(geotiff_path, landuse_map, lu_profile, nodata=0)
                output_path += f"\nand {geotiff_path}"

            return output_path

        except Exception as e:
            QMessageBox.warning(
                self,
                "Save Error",
                f"Partial save completed. Could not save all formats:\n{str(e)}"
            )
            return output_path if output_path else None
    
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$44

    def add_layer_to_list(self, name, image, profile, is_change_map=False):
        layer_data = {
            'name': name,
            'image': image,
            'profile': profile,
            'is_change_map': is_change_map
        }
        self.layers[name] = layer_data
        item = QListWidgetItem(name)
        self.layer_list.addItem(item)

    def show_help(self):
        help_message = """
        Space GeoVision Software Help (v1.0)
        ====================================

        1. Getting Started:
        - Open images: File > Open (supports GeoTIFF)
        - Save results: File > Save (GeoTIFF, PNG, JPEG, Shapefile)
        - Switch layers: Double-click items in layer list

        2. Core Features:
        a) Raster Processing:
        - Resample/Enhance/Clip via 'Raster Processing Tools' menu
        - Adjust contrast, smoothness, and sharpness

        b) Classification:
        - Supervised: Random Forest, SVM (set number of classes)
        - Unsupervised: K-Means, KNN
        - Save results: 'Classification Tools > Save Classified Image'

        c) Spectral Indices:
        - Calculate NDVI, NDWI, NDNI, AVI, NDMI
        - Load required bands when prompted
        - Save indices: Right-click results > Export

        d) Map Preparation:
        - Add titles, legends, grids via 'Prepare Map' menu
        - Insert dynamic scale bars and north arrows
        - Export layouts: Prepare Map > Export Map Layout (PDF/PNG)

        e) Change Detection:
        - Load two temporally different images
        - Ensure matching CRS using resample/clip tools
        - Analyze changes via 'Change Detection Tools'

        3. Shortcuts & Tips:
        - Undo/Redo: Ctrl+Z / Ctrl+Y
        - Progress bar shows during long operations
        - Right-click layers for export options
        - Check 'worker_errors.log' for technical issues

        4. Raster Calculator Formulas
        -----------------------------

        Basic Arithmetic:
        "Layer1" + "Layer2"        Add two rasters
        "DEM" - 100                Subtract constant
        ("B5" - "B4") / ("B5" + "B4")  Normalized Difference

        Mathematical Functions:
        sqrt("Slope")              Square root 
        log("Population" + 1)      Natural logarithm
        sin("Aspect_Radians")      Trigonometric functions

        Conditional Operations:
        where("Temp" > 30, 1, 0)   Threshold classification
        np.where("Class" == 2, 1, 0)  Mask specific values

        Boolean Operations:
        ("Forest" > 0.5) & ("Slope" < 15)  Combined conditions
        ~("Water_Mask")             Invert mask

        Common Indices:
        NDVI: ("NIR" - "Red") / ("NIR" + "Red" + 1e-9)
        NDWI: ("Green" - "NIR") / (Green + NIR + 1e-9)
        NDBI: (SWIR - NIR) / (SWIR + NIR + 1e-9)

        Advanced Formulas:
        ("Night_Lights" - min) / (max - min)  Normalization
        sqrt(X² + Y²)              Magnitude calculation
        (A * 0.3) + (B * 0.7)     Weighted overlay

        Formula Rules:
        1. Always quote layer names: "Layer1"
        2. Use parentheses for complex operations
        3. Automatic division protection (1e-9 added)
        4. Available functions: sqrt, log, sin, cos, where
        5. Operators: +, -, *, /, &, |, ~

        Support: support@spacegeovision.com
        """
        QMessageBox.information(self, "Help Guide", help_message)


    def canny_edge_detection(self):
        if hasattr(self, 'image'):
            try:
                # Convert to normalized grayscale with explicit checks
                if self.image.ndim == 3 and self.image.shape[0] == 3:
                    # RGB to grayscale using BT.709 luminance coefficients
                    gray = np.dot(
                        self.image.transpose(1, 2, 0).astype(np.float32),
                        [0.2126, 0.7152, 0.0722]
                    )
                else:
                    gray = self.image[0] if self.image.ndim == 3 else self.image
                    gray = gray.astype(np.float32)

                # Robust normalization using percentile-based scaling
                p_low, p_high = np.percentile(gray, (2, 98))
                gray_norm = np.clip((gray - p_low) / (p_high - p_low + 1e-8), 0, 1)

                # Adaptive sigma calculation using median absolute deviation
                median = np.median(gray_norm)
                mad = np.median(np.abs(gray_norm - median))
                sigma = max(0.1, 0.5 * mad)

                # Automatic threshold calculation using Otsu's method
                from skimage.filters import try_all_threshold, threshold_otsu
                thresh_otsu = threshold_otsu(gray_norm)
                low_thresh = thresh_otsu * 0.5
                high_thresh = thresh_otsu * 1.5

                # Apply Canny with validated parameters
                edges = canny(
                    gray_norm,
                    sigma=sigma,
                    low_threshold=low_thresh,
                    high_threshold=high_thresh,
                    use_quantiles=False  # Use absolute intensity values
                )

                # Convert to uint8 and verify edge content
                edges_uint8 = (edges * 255).astype(np.uint8)[np.newaxis, ...]
                
                if np.mean(edges_uint8) > 250:
                    QMessageBox.warning(
                        self, 
                        "Excessive Edges",
                        "Thresholds too low - try increasing sigma or thresholds"
                    )
                    return

                # Update display and metadata only if valid edges exist
                self.image = edges_uint8
                self.display_raster_layer(edges_uint8, "Canny Edges")
                self.profile.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'nodata': 0,
                    'height': edges_uint8.shape[1],
                    'width': edges_uint8.shape[2]
                })

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Edge Detection Error",
                    f"Failed to process image:\n{str(e)}\n"
                    f"Image Stats - Min: {gray.min():.2f}, Max: {gray.max():.2f}, MAD: {mad:.2f}"
                )
        else:
            QMessageBox.warning(self, "No Image", "Please load an image first")
    
    #Sobel
    def sobel_edge_detection(self):
        if hasattr(self, 'image'):
            try:
                # Convert to grayscale
                gray = self.image[0] if self.image.ndim == 3 else self.image
                gray = gray.astype(np.float32)

                # Apply Sobel filter
                from skimage.filters import sobel
                edges = sobel(gray)

                # Convert to uint8
                edges_uint8 = (edges * 255).astype(np.uint8)[np.newaxis, ...]

                # Update display and metadata
                self.image = edges_uint8
                self.display_raster_layer(edges_uint8, "Sobel Edges")
                self.profile.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'nodata': 0,
                    'height': edges_uint8.shape[1],
                    'width': edges_uint8.shape[2]
                })

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Edge Detection Error",
                    f"Failed to process image:\n{str(e)}"
                )
        else:
            QMessageBox.warning(self, "No Image", "Please load an image first")
    
    #Prewitt Edge Detection
    def prewitt_edge_detection(self):
        if hasattr(self, 'image'):
            try:
                # Convert to grayscale
                gray = self.image[0] if self.image.ndim == 3 else self.image
                gray = gray.astype(np.float32)

                # Apply Prewitt filter
                from skimage.filters import prewitt
                edges = prewitt(gray)

                # Convert to uint8
                edges_uint8 = (edges * 255).astype(np.uint8)[np.newaxis, ...]

                # Update display and metadata
                self.image = edges_uint8
                self.display_raster_layer(edges_uint8, "Prewitt Edges")
                self.profile.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'nodata': 0,
                    'height': edges_uint8.shape[1],
                    'width': edges_uint8.shape[2]
                })

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Edge Detection Error",
                    f"Failed to process image:\n{str(e)}"
                )
        else:
            QMessageBox.warning(self, "No Image", "Please load an image first")
   
   #Laplacian of Gaussian (LoG) Edge Detection
    def log_edge_detection(self):
        if hasattr(self, 'image'):
            try:
                # Convert to grayscale
                gray = self.image[0] if self.image.ndim == 3 else self.image
                gray = gray.astype(np.float32)

                # Apply Laplacian of Gaussian filter
                from skimage.filters import laplace
                from scipy.ndimage import gaussian_filter
                blurred = gaussian_filter(gray, sigma=1)
                edges = laplace(blurred)

                # Convert to uint8
                edges_uint8 = (edges * 255).astype(np.uint8)[np.newaxis, ...]

                # Update display and metadata
                self.image = edges_uint8
                self.display_raster_layer(edges_uint8, "LoG Edges")
                self.profile.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'nodata': 0,
                    'height': edges_uint8.shape[1],
                    'width': edges_uint8.shape[2]
                })

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Edge Detection Error",
                    f"Failed to process image:\n{str(e)}"
                )
        else:
            QMessageBox.warning(self, "No Image", "Please load an image first")
 
    def raster_to_polygons(self, edges):
    # Use the edges array directly (0 and 255)
        mask = edges == 255
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(edges.astype(np.uint8), mask=mask, transform=self.profile['transform'])
            )
        )
        geoms = list(results)
        gdf = gpd.GeoDataFrame.from_features(geoms)
        
        # Ask user for confirmation to save the polygons
        save_reply = QMessageBox.question(
            self, "Save Polygons", "Do you want to save the polygons?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if save_reply == QMessageBox.Yes:
            # Save the polygons
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Polygons", "", "GeoJSON Files (*.geojson);;Shapefiles (*.shp)")
            if file_name:
                gdf.to_file(file_name)
                QMessageBox.information(self, "Success", f"Polygons saved as '{file_name}'")
        else:
            QMessageBox.information(self, "Cancelled", "Polygon saving cancelled by user.")
        
        return gdf

    #Clips a portion of the loaded raster image based on user-defined bounding box.
    
    
 
       
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   
    

    
    #                  Band Combination
    # 1- Natural Color (B4, B3, B2) to display imagery the same way our eyes see the world
    # 2- Color Infrared (B8, B4, B3) it’s especially good at reflecting chlorophyll. This is 
    #       why in a color infrared image, denser vegetation is red. But urban areas are white.
    # 3-Short-Wave Infrared (B12, B8A, B4) shows vegetation in various shades of green. In general, 
    #      darker shades of green indicate denser vegetation. But brown is indicative of bare soil and built-up areas.
    # 4- Agriculture (B11, B8, B2) Both these bands are particularly good at highlighting dense vegetation that appears as dark green.
    # 5- Geology (B12, B11, B2) This includes faults, lithology, and geological formations
    # 6-Bathymetric (B4, B3, B1). Using the coastal aerosol band is good for estimating suspended sediment in the water.
    # 7-Vegetation Index (B8-B4)/(B8+B4). While high values suggest dense canopy, low or negative values indicate urban and water features.
    # 8-Moisture Index (B8A-B11)/(B8A+B11). In general, wetter vegetation has higher values. But lower moisture index values suggest plants 
    #    are under stress from insufficient moisture.
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
   
    def load_band(self, band_name):
        """
        Loads a specific band (NIR or Red) from an image file selected by the user.
        
        :param band_name: اسم النطاق المطلوب ("NIR Band (5 LandSat)" أو "Red Band").
        #      LandSat 8
        # Bands 	Wavelength               (micrometers)	Resolution (meters)
        # Band 1 - Coastal aerosol            	0.43-0.45	30
        # Band 2 - Blue                      	0.45-0.51	30
        # Band 3 - Green                    	0.53-0.59	30
        # Band 4 - Red                       	0.64-0.67	30
        # Band 5 - Near Infrared (NIR)      	0.85-0.88	30
        # Band 6 - Shortwave Infrared (SWIR) 1	1.57-1.65	30
        # Band 7 - Shortwave Infrared (SWIR) 2	2.11-2.29	30
        # Band 8 - Panchromatic             	0.50-0.68	15
        # Band 9 - Cirrus                   	1.36-1.38	30
        # Band 10 - Thermal Infrared (TIRS) 1	10.6-11.19	100 (resampled to 30)*
        # Band 11 - Thermal Infrared (TIRS) 2	11.50-12.51	100 (resampled to 30)*
        
        #                 Sentinel 2 Bands
        # B1	60 m	443 nm	Ultra Blue (Coastal and Aerosol)
        # B2	10 m	490 nm	Blue
        # B3	10 m	560 nm	Green
        # B4	10 m	665 nm	Red
        # B5	20 m	705 nm	Visible and Near Infrared (VNIR)
        # B6	20 m	740 nm	Visible and Near Infrared (VNIR)
        # B7	20 m	783 nm	Visible and Near Infrared (VNIR)
        # B8	10 m	842 nm	Visible and Near Infrared (VNIR)
        # B8a	20 m	865 nm	Visible and Near Infrared (VNIR)
        # B9	60 m	940 nm	Short Wave Infrared (SWIR)
        # B10	60 m	1375 nm	Short Wave Infrared (SWIR)
        # B11	20 m	1610 nm	Short Wave Infrared (SWIR)
        # B12	20 m	2190 nm	Short Wave Infrared (SWIR)
        """
        try:
            # إعلام المستخدم باسم النطاق المطلوب تحميله
            QMessageBox.information(self, "Info", f"Load the band: {band_name}")

            # فتح نافذة اختيار الملفات
            file_name, _ = QFileDialog.getOpenFileName(
                self,  # النافذة الرئيسية
                f"Select {band_name} File",  # عنوان النافذة
                "",  # المسار الافتراضي
                "GeoTIFF Files (*.tif *.tiff);;All Files (*)"  # أنواع الملفات
            )

            # إذا لم يتم اختيار ملف، يتم إرجاع None
            if not file_name:
                QMessageBox.warning(self, "Warning", f"No file selected for {band_name}.")
                return

            # تحميل الصورة
            with rasterio.open(file_name) as src:
                image = src.read()  # قراءة جميع النطاقات
                profile = dict(src.profile)  # الحصول على معلومات الملف

            # تخزين البيانات في self.layers
            layer_name = os.path.basename(file_name)
            self.layers[layer_name] = {
                'image': image,
                'profile': profile,
                'file_name': file_name,
                'band_name': band_name,  # إضافة اسم النطاق
                'type': 'raster'  # تحديد نوع البيانات كنقطية
            }

            # إضافة العنصر إلى قائمة الطبقات
            item = QListWidgetItem(layer_name)
            self.layer_list.addItem(item)

            # تعيين الصورة الحالية
            self.image = image
            self.profile = profile
            self.image_path = file_name

            # التبديل إلى الطبقة المحملة
            self.switch_layer(item)  # استخدام switch_layer لعرض الطبقة

            # إرجاع الصورة المحملة
            return image

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {band_name}: {str(e)}")
            return None
            
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  
    def save_index_result(self, index_image, profile, index_name):
        """
        Saves the calculated index (e.g., NDVI, NDWI, NDMI) to a GeoTIFF file.

        :param index_image: المصفوفة الناتجة من حساب المؤشر.
        :param profile: ملف التعريف (metadata) للصورة.
        :param index_name: اسم المؤشر (مثل "NDVI"، "NDWI").
        """
        try:
            # فتح نافذة اختيار الملفات للحفظ
            file_path, _ = QFileDialog.getSaveFileName(
                self,  # النافذة الرئيسية
                f"Save {index_name} Result",  # عنوان النافذة
                "",  # المسار الافتراضي
                "GeoTIFF Files (*.tif *.tiff);;All Files (*)"  # أنواع الملفات
            )

            # إذا لم يتم تحديد مسار للحفظ، يتم إلغاء العملية
            if not file_path:
                QMessageBox.warning(self, "Warning", f"No file path selected. {index_name} result not saved.")
                return

            # تقليل الأبعاد إذا كانت الصورة تحتوي على أبعاد إضافية
            if index_image.ndim == 4 and index_image.shape[0] == 1:  # إذا كانت الأبعاد (1, 1, H, W)
                index_image = index_image[0]  # تحويلها إلى (1, H, W)
            elif index_image.ndim == 3 and index_image.shape[0] == 1:  # إذا كانت الأبعاد (1, H, W)
                index_image = index_image[0]  # تحويلها إلى (H, W)
            elif index_image.ndim != 2:  # إذا كانت الأبعاد غير متوقعة
                raise ValueError(f"Unsupported {index_name} shape: {index_image.shape}. Expected (H, W) or (1, H, W).")

            # تحديث ملف التعريف لحفظ المؤشر
            profile.update({
                'count': 1,  # نطاق واحد
                'dtype': 'float32',  # نوع البيانات
                'nodata': -9999  # قيمة nodata
            })

            # حفظ النتيجة كملف GeoTIFF
            with rasterio.open(
                file_path,  # مسار الملف
                'w',  # وضع الكتابة
                **profile  # استخدام ملف التعريف المحدث
            ) as dst:
                dst.write(index_image.astype(np.float32), 1)  # كتابة النطاق الأول

            QMessageBox.information(self, "Success", f"{index_name} result saved successfully at:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save {index_name} result: {str(e)}")   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    NDVI   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    def load_and_calculate_ndvi(self):
        """
        Loads two images (NIR and Red bands) one by one, displays them, calculates NDVI, and adds the result to the file list.
        """
        try:
            # تحميل صورة NIR
            nir_band = self.load_band("NIR (Nr. 5 LandSat and 8 for Sentinel-2)")
            if nir_band is None:
                raise ValueError("Failed to load NIR band band (Nr. 5 LandSat and 8 for Sentinel-2).")

            # انتظار المستخدم لتحميل الصورة الثانية
            QMessageBox.information(self, "Info", "NIR band (Nr. 5 LandSat and 8 for Sentinel-2) loaded and displayed.")

            # تحميل صورة Red
            red_band = self.load_band("Red (4 LandSat)")
            if red_band is None:
                raise ValueError("Failed to load Red band (4 for LandSat and Sentinel-2).")

            # التحقق من أبعاد الصور
            if nir_band.shape != red_band.shape:
                raise ValueError("NIR and Red bands must have the same dimensions.")

            # حساب NDVI
            ndvi = self.image_classifier.calculate_ndvi(nir_band, red_band)

            # عرض نتيجة NDVI
            if ndvi is not None:
                # إضافة نتيجة NDVI إلى self.layers
                layer_name = "NDVI Result"
                self.layers[layer_name] = {
                    'image': ndvi,
                    'profile': self.profile,
                    'file_name': "NDVI Result",
                    'type': 'raster',
                    'cmap': 'viridis'  # Store the colormap
                    
                }

                # إضافة العنصر إلى قائمة الطبقات
                item = QListWidgetItem(layer_name)
                self.layer_list.addItem(item)

                # التبديل إلى طبقة NDVI
                self.switch_layer(item)

                # سؤال المستخدم إذا كان يريد حفظ النتيجة
                reply = QMessageBox.question(
                    self,  # النافذة الرئيسية
                    "Save NDVI Result",  # عنوان النافذة
                    "Do you want to save the NDVI result?",  # نص السؤال
                    QMessageBox.Yes | QMessageBox.No,  # الأزرار
                    QMessageBox.Yes  # الزر الافتراضي
                )

                # إذا وافق المستخدم على الحفظ
                if reply == QMessageBox.Yes:
                    self.save_ndvi_result(ndvi)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process images: {str(e)}")
  
    def save_ndvi_result(self, ndvi):
        """
        Saves the NDVI result using the general save function.
        """
        self.save_index_result(ndvi, self.profile, "NDVI")   
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    NDWI   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    def load_and_calculate_ndwi(self):
        """
        Loads Green and NIR bands, calculates NDWI, and displays/saves the result.
        """
        try:
            # Load Green band
            green_band = self.load_band("Green Band (Band 3)")
            if green_band is None:
                raise ValueError("Failed to load Green band (Band 3).")

            # Load NIR band
            nir_band = self.load_band("NIR Band (5 LandSat and 8 for Sentinel-2)")
            if nir_band is None:
                raise ValueError("Failed to load NIR band.")

            # Check if bands have the same dimensions
            if green_band.shape != nir_band.shape:
                #raise ValueError("Green and NIR bands must have the same dimensions.")
                raise ValueError(f"Dimension mismatch between Green Band ({green_band.shape}) and NIR Band ({nir_band.shape}).")

            # Calculate NDWI
            ndwi = self.image_classifier.calculate_ndwi(green_band, nir_band)

            # Display NDWI result
            if ndwi is not None:
                layer_name = "NDWI Result"
                self.layers[layer_name] = {
                    'image': ndwi,
                    'profile': self.profile,
                    'file_name': "NDWI Result",
                    'type': 'raster'
                }

                # Add to layer list
                item = QListWidgetItem(layer_name)
                self.layer_list.addItem(item)

                # Switch to NDWI layer
                self.switch_layer(item)

                # Ask user if they want to save the result
                reply = QMessageBox.question(
                    self, "Save NDWI Result", "Do you want to save the NDWI result?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self.save_ndwi_result(ndwi)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate NDWI: {str(e)}")

    
    def save_ndwi_result(self, ndwi):
        """
        Saves the NDWI result to a file.
        """
        self.save_index_result(ndwi, self.profile, "NDWI")   
     
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    def load_and_calculate_ndni(self):
        """
        Loads SWIR1 and SWIR2 bands, calculates NDNI, and displays/saves the result.
        """
        try:
            # Load SWIR1 band
            swir1_band = self.load_band("SWIR1 Band")
            if swir1_band is None:
                raise ValueError("Failed to load SWIR1 band.")

            # Load SWIR2 band
            swir2_band = self.load_band("SWIR2 Band")
            if swir2_band is None:
                raise ValueError("Failed to load SWIR2 band.")

            # Check if bands have the same dimensions
            if swir1_band.shape != swir2_band.shape:
                #raise ValueError("SWIR1 and SWIR2 bands must have the same dimensions.")
                raise ValueError(f"Dimension mismatch between SWIR1 Band ({swir1_band.shape}) and SWIR2 Band ({swir2_band.shape}).")

            # Calculate NDNI
            ndni = self.image_classifier.calculate_ndni(swir1_band, swir2_band)

            # Display NDNI result
            if ndni is not None:
                layer_name = "NDNI Result"
                self.layers[layer_name] = {
                    'image': ndni,
                    'profile': self.profile,
                    'file_name': "NDNI Result",
                    'type': 'raster'
                }

                # Add to layer list
                item = QListWidgetItem(layer_name)
                self.layer_list.addItem(item)

                # Switch to NDNI layer
                self.switch_layer(item)

                # Ask user if they want to save the result
                reply = QMessageBox.question(
                    self, "Save NDNI Result", "Do you want to save the NDNI result?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self.save_ndni_result(ndni)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate NDNI: {str(e)}")

    def save_ndni_result(self, ndni):
        """
        Saves the NDNI result to a file.
        """
        self.save_index_result(ndni, self.profile, "NDNI")  


##############################################   AVI   ############################
    def load_and_calculate_avi(self):
        """
        Loads Red and NIR bands, calculates AVI, and displays/saves the result.
        """
        try:
             # Load NIR band
            nir_band = self.load_band("NIR Band (5 LandSat and 8 for Sentinel-2)")
            if nir_band is None:
                raise ValueError("Failed to load NIR band 5 LandSat.")
            # Load Red band
            red_band = self.load_band("Red Band 4 Landsat")
            if red_band is None:
                raise ValueError("Failed to load Red band 4 Landsat.")       
            
            

            # Check if bands have the same dimensions
            if red_band.shape != nir_band.shape:
                #raise ValueError("Red and NIR bands must have the same dimensions.")
                raise ValueError(f"Dimension mismatch between Red Band ({red_band.shape}) and NIR Band ({nir_band.shape}).")

            # Calculate AVI
            
            avi = self.image_classifier.calculate_avi(nir_band, red_band)
            
            # Display AVI result
            if avi is not None:
                layer_name = "AVI Result"
                self.layers[layer_name] = {
                    'image': avi,
                    'profile': self.profile,
                    'file_name': "AVI Result",
                    'type': 'raster'
                }

                # Add to layer list
                item = QListWidgetItem(layer_name)
                self.layer_list.addItem(item)

                # Switch to AVI layer
                self.switch_layer(item)

                # Ask user if they want to save the result
                reply = QMessageBox.question(
                    self, "Save AVI Result", "Do you want to save the AVI result?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self.save_avi_result(avi)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate AVI: {str(e)}")

    def save_avi_result(self, avi):
        """
        Saves the AVI result to a file.
        """
        self.save_index_result(avi, self.profile, "AVI")   
      
    
############################################# NDMI ################################
    def load_and_calculate_ndmi(self):
        """
        Loads NIR and SWIR bands, calculates NDMI, and displays/saves the result. 
        
        Sentinel-2 NDMI = (B08 - B11) / (B08 + B11)
        Landsat 4-5 TM NDMI = (B04 - B05) / (B04 + B05)
        Landsat 7 ETM+ NDMI = (B04 - B05) / (B04 + B05)
        Landsat 8 NDMI = (B05 - B06) / (B05 + B06)
        MODIS NDMI = (B02 - B06) / (B02 + B06)
        """
        try:
            # Load NIR band
            nir_band = self.load_band("B05  LandSat-8 and B08 for Sentinel-2")
            if nir_band is None:
                raise ValueError("Failed to load NIR band.")

            # Load SWIR band
            swir_band = self.load_band("SWIR Band (B06  LandSat-8 and B11 for Sentinel-2")
            if swir_band is None:
                raise ValueError("Failed to load SWIR band.")

            # Check if bands have the same dimensions
            if nir_band.shape != swir_band.shape:
                #raise ValueError("NIR and SWIR bands must have the same dimensions.")
                raise ValueError(f"Dimension mismatch between NIR Band ({nir_band.shape}) and SWIR Band ({swir_band.shape}).")

            # Calculate NDMI
            ndmi = (nir_band - swir_band) / (nir_band + swir_band)

            # Display NDMI result
            if ndmi is not None:
                layer_name = "NDMI Result"
                self.layers[layer_name] = {
                    'image': ndmi,
                    'profile': self.profile,
                    'file_name': "NDMI Result",
                    'type': 'raster'
                }

                # Add to layer list
                item = QListWidgetItem(layer_name)
                self.layer_list.addItem(item)

                # Switch to NDMI layer
                self.switch_layer(item)

                # Ask user if they want to save the result
                reply = QMessageBox.question(
                    self, "Save NDMI Result", "Do you want to save the NDMI result?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self.save_ndmi_result(ndmi)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate NDMI: {str(e)}")

    def save_ndmi_result(self, ndmi):
        """
        Saves the NDMI result to a file.
        """
        self.save_index_result(ndmi, self.profile, "NDMI")   


                

    
    # دالة حساب NDWI


               

    # دالة حساب NDMI

    def save_current_index(self):
        """حفظ المؤشر الحالي"""
        if hasattr(self, 'image'):
            file_name, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Index", 
                "", 
                "GeoTIFF (*.tif);;PNG (*.png);;JPEG (*.jpg)"
            )
            if file_name:
                try:
                    if file_name.endswith('.tif'):
                        profile = self.profile.copy()
                        profile.update(dtype=rasterio.float32, count=1)
                        with rasterio.open(file_name, 'w', **profile) as dst:
                            dst.write(self.image.astype(rasterio.float32), 1)
                    else:
                        img = (self.image[0] * 255).astype(np.uint8)
                        Image.fromarray(img).save(file_name)
                        
                    QMessageBox.information(self, "Success", "Index saved successfully")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")



 
    





                
            


########################################### Print
    def export_map_dialog(self):
        """Print-quality export: DPI, page size, clipping to a polygon or a drawn rectangle, title, scale bar, ..."""
        try:
            import mapexport_ui
            if not any(isinstance(v, dict) and v.get('image') is not None for v in self.layers.values()):
                QMessageBox.information(self, "Export Map", "Load a georeferenced raster first.")
                return
            old = getattr(self, '_export_dlg', None)
            if old is not None:
                try:
                    old.reject()
                except Exception:
                    pass
            self._export_dlg = mapexport_ui.ExportMapDialog(self)
            self._export_dlg.setAttribute(Qt.WA_DeleteOnClose, False)
            self._export_dlg.show()
        except Exception as exc:
            logging.exception("export_map_dialog failed")
            QMessageBox.critical(self, "Export Map", f"Could not open the export dialog: {exc}")

    def stack_bands_dialog(self):
        """Merge single-band Sentinel-2 GeoTIFFs (B02,B03,B04,B08,B11) of one date into one multi-band file."""
        try:
            import bandstack
            folder = QFileDialog.getExistingDirectory(
                self, "Folder with the single-band files of ONE date (B02, B03, B04, B08, B11)")
            if not folder:
                return
            rep = bandstack.stack_bands(folder)
            QMessageBox.information(self, "Stack Bands", bandstack.format_report(rep))
        except Exception as exc:
            logging.exception("stack_bands_dialog failed")
            QMessageBox.critical(self, "Stack Bands", str(exc))

    def print_image(self):
        """Handle both printing to physical printer and PDF export"""
        if not hasattr(self, 'ax') or not self.ax:
            QMessageBox.warning(self, "Error", "No display content available")
            return

        if not self._printing_available():
            return
        # Create printer with high resolution
        printer = QPrinter(QPrinter.HighResolution)
        
        # Configure print dialog
        print_dialog = QPrintDialog(printer, self)
        print_dialog.setWindowTitle("Print Map")
        
        # Add PDF option if available
        if hasattr(print_dialog, 'setOption'):
            print_dialog.setOption(QPrintDialog.PrintToFile, True)
        
        # Show print dialog
        if print_dialog.exec_() != QDialog.Accepted:
            return

        try:
            # Create a painter for printing
            painter = QPainter()
            if not painter.begin(printer):
                QMessageBox.warning(self, "Error", "Could not start printing")
                return

            # Get page dimensions
            page_rect = printer.pageRect()
            margin = 20  # 20 pixel margin
            content_rect = page_rect.adjusted(margin, margin, -margin, -margin)

            # Draw white background
            painter.fillRect(page_rect, Qt.white)

            # Get current display as pixmap
            pixmap = QPixmap(self.canvas.size())
            self.canvas.render(pixmap)
            
            # Calculate scaling to fit content area
            pixmap_ratio = pixmap.width() / pixmap.height()
            content_ratio = content_rect.width() / content_rect.height()
            
            if pixmap_ratio > content_ratio:
                scale = content_rect.width() / pixmap.width()
            else:
                scale = content_rect.height() / pixmap.height()
            
            scaled_width = pixmap.width() * scale
            scaled_height = pixmap.height() * scale
            
            # Center the content
            x = content_rect.x() + (content_rect.width() - scaled_width) / 2
            y = content_rect.y() + (content_rect.height() - scaled_height) / 2
            
            # Corrected drawPixmap call - using QRectF consistently
            target_rect = QRectF(x, y, scaled_width, scaled_height)
            painter.drawPixmap(target_rect, pixmap, QRectF(pixmap.rect()))
            
            # Draw title and border
            painter.setPen(Qt.black)
            painter.drawRect(content_rect)
            painter.drawText(page_rect.x() + 10, page_rect.y() + 15, 
                            f"Map Export - {QDateTime.currentDateTime().toString()}")
            
            painter.end()

            if printer.outputFormat() == QPrinter.PdfFormat:
                QMessageBox.information(self, "PDF Export", 
                                    f"PDF saved to:\n{printer.outputFileName()}")
            else:
                QMessageBox.information(self, "Print", "Content sent to printer successfully")

        except Exception as e:
            if painter.isActive():
                painter.end()
            QMessageBox.critical(self, "Print Error", f"Failed to print: {str(e)}")


###################################################
 

           
    def _printing_available(self):
        """Qt's Windows print support can abort the whole process (native fatal error 0x80040155, which no Python
        try/except can catch) when no printer driver is installed. Check first and tell the user instead."""
        try:
            has = bool(QPrinterInfo.availablePrinters())
        except Exception:
            has = False
        if not has:
            QMessageBox.information(self, "Printing",
                                    "No printer is installed on this computer.\n"
                                    "Install a printer (e.g. 'Microsoft Print to PDF') to use Print / Page Setup, "
                                    "or use File > Save as image / PDF report.")
        return has

    def page_setup(self):
        """Configure page settings"""
        if not self._printing_available():
            return
        printer = QPrinter()
        dialog = QPageSetupDialog(printer, self)
        dialog.exec_()



    # ======================================================================
    # Supervised classification (RF / SVM) with real validation
    # ======================================================================
    def supervised_classification(self):
        """Random Forest / SVM using labelled vector samples and an honest accuracy assessment."""
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle("Supervised classification (Random Forest / SVM)")
            form = QFormLayout(dlg)

            def file_row(edit, title, filt, folder=False):
                btn = QPushButton("Browse...")
                row = QHBoxLayout(); row.addWidget(edit); row.addWidget(btn)
                def pick():
                    if folder:
                        p = QFileDialog.getExistingDirectory(dlg, title, edit.text())
                    else:
                        p, _ = QFileDialog.getOpenFileName(dlg, title, edit.text(), filt)
                    if p:
                        edit.setText(p)
                btn.clicked.connect(pick)
                return row

            raster_e = QLineEdit(getattr(self, 'input_file', '') or '')
            train_e, val_e = QLineEdit(), QLineEdit()
            field_e = QLineEdit("class")
            out_e = QLineEdit(os.path.join(tempfile.gettempdir(), "SpaceGeoVision_classification"))
            model_c = QComboBox()
            model_c.addItems(["Random Forest", "SVM (RBF, tuned)", "SVM (linear, tuned)"])
            trees = QSpinBox(); trees.setRange(10, 1000); trees.setValue(100)
            block = QSpinBox(); block.setRange(5, 5000); block.setValue(100); block.setSuffix(" px")

            form.addRow("Raster (multi-band GeoTIFF):", file_row(raster_e, "Raster", "GeoTIFF (*.tif *.tiff)"))
            form.addRow("Training samples (vector):", file_row(train_e, "Training samples", "Vector (*.shp *.gpkg *.geojson)"))
            form.addRow("Class field:", field_e)
            form.addRow("Validation samples (optional):", file_row(val_e, "Validation samples", "Vector (*.shp *.gpkg *.geojson)"))
            form.addRow("Model:", model_c)
            form.addRow("Trees (RF):", trees)
            form.addRow("Validation block size:", block)
            form.addRow("Output folder:", file_row(out_e, "Output folder", "", folder=True))
            form.addRow(QLabel("Without an independent validation layer, whole spatial blocks of the training\n"
                               "samples are held out. Accuracy is never computed on training pixels."))
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
            form.addRow(btns)
            if dlg.exec_() != QDialog.Accepted:
                return
            for label, e in (("raster", raster_e), ("training samples", train_e)):
                if not e.text().strip() or not os.path.isfile(e.text().strip()):
                    raise ValueError(f"Please choose a valid {label} file.")
            if val_e.text().strip() and not os.path.isfile(val_e.text().strip()):
                raise ValueError("The validation file does not exist.")
            model_key = {0: "rf", 1: "svm_rbf", 2: "svm_linear"}[model_c.currentIndex()]
            cfg = supervised.ClassifierConfig(model=model_key, n_trees=trees.value(), block_px=block.value())

            self._cls_progress = QProgressDialog("Starting...", None, 0, 100, self)
            self._cls_progress.setWindowTitle("Supervised classification")
            self._cls_progress.setWindowModality(Qt.WindowModal)
            self._cls_progress.setMinimumDuration(0)
            worker = FuncWorker(
                lambda progress: supervised.run_supervised(
                    raster_e.text().strip(), train_e.text().strip(), field_e.text().strip(), out_e.text().strip(),
                    cfg, val_e.text().strip() or None, progress=progress))
            worker.progress.connect(lambda v, m: (self._cls_progress.setValue(v), self._cls_progress.setLabelText(m)))
            worker.done.connect(self._on_supervised_done)
            worker.failed.connect(lambda m: (self._cls_progress.close(), self.show_error(f"Classification failed:\n{m}")))
            self._cls_worker = worker
            worker.start()
        except Exception as exc:
            self.show_error(f"Cannot start classification: {exc}")

    def _on_supervised_done(self, res):
        self._cls_progress.close()
        classified = res["classified"]
        self.classified_image = classified[np.newaxis, ...]
        self.profile = res["profile"]
        self.classified_profile = res["profile"]
        names = dict(res["evaluation"].names)
        # The result becomes a layer of its own (real grid), so the cursor read-out shows map coordinates and class
        layer_name = os.path.splitext(os.path.basename(res["classified_tif"]))[0]
        self.layers[layer_name] = {'image': self.classified_image, 'profile': dict(res["profile"]),
                                   'file_name': res["classified_tif"], 'type': 'raster', 'cmap': 'viridis',
                                   'class_names': names, 'classified': True, 'title': layer_name}
        if not any(self.layer_list.item(i).text() == layer_name for i in range(self.layer_list.count())):
            self.layer_list.addItem(QListWidgetItem(layer_name))
        self.current_layer = layer_name
        self.display_classified_image(classified, profile=res["profile"], names=names, title=layer_name)
        dlg = QDialog(self)
        dlg.setWindowTitle("Accuracy assessment (held-out samples)")
        dlg.setMinimumSize(760, 520)
        lay = QVBoxLayout(dlg)
        box = QTextEdit(); box.setReadOnly(True)
        box.setFontFamily("Courier New"); box.setPlainText(res["report"] +
            f"\n\nFiles:\n  {res['classified_tif']}\n  {res['confusion_csv']}\n  {res['report_txt']}\n  {res['provenance']}")
        lay.addWidget(box)
        ok = QDialogButtonBox(QDialogButtonBox.Ok); ok.accepted.connect(dlg.accept); lay.addWidget(ok)
        dlg.exec_()

    # ======================================================================
    # Bi-temporal change detection (CVA with radiometric normalisation)
    # ======================================================================
    def _two_image_dialog(self, title, with_options):
        """Guided dialog: EARLIER and LATER image. Returns (f1, f2, opts) or None if cancelled."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(560)
        outer = QVBoxLayout(dlg)
        guide = QLabel(
            "<b>Compare two images of the same area.</b><br>"
            "1. Choose the <b>EARLIER</b> image (before the event).<br>"
            "2. Choose the <b>LATER</b> image (after the event).<br>"
            "Both must have the same grid (size, resolution, CRS) and the same bands. "
            "Changes are reported as <i>later minus earlier</i>.")
        guide.setWordWrap(True)
        outer.addWidget(guide)
        fields = {}

        def add_row(key, label):
            row = QHBoxLayout()
            lbl = QLabel(label); lbl.setMinimumWidth(150)
            edit = QLineEdit(); edit.setPlaceholderText("Click Browse… and select a GeoTIFF")
            btn = QPushButton("Browse…")

            def browse():
                start = os.path.dirname(edit.text()) if edit.text() else ""
                p, _ = QFileDialog.getOpenFileName(dlg, label, start, "GeoTIFF (*.tif *.tiff)")
                if p:
                    edit.setText(p)
            btn.clicked.connect(browse)
            row.addWidget(lbl); row.addWidget(edit, 1); row.addWidget(btn)
            outer.addLayout(row)
            fields[key] = edit
        add_row('f1', "1) EARLIER (before):")
        add_row('f2', "2) LATER (after):")

        def swap_files():
            a_, b_ = fields['f1'].text(), fields['f2'].text()
            fields['f1'].setText(b_); fields['f2'].setText(a_)
        swap = QPushButton("Swap the two images")
        swap.clicked.connect(swap_files)
        outer.addWidget(swap, 0, Qt.AlignLeft)
        norm = meth = kspin = None
        if with_options:
            form = QFormLayout()
            outer.addLayout(form)
            norm = QCheckBox("Radiometric normalisation (recommended)"); norm.setChecked(True)
            meth = QComboBox(); meth.addItems(["Robust sigma (k · σ)", "Otsu"])
            kspin = QDoubleSpinBox(); kspin.setRange(1.0, 10.0); kspin.setValue(3.0); kspin.setSingleStep(0.5)
            form.addRow("", norm); form.addRow("Threshold method:", meth); form.addRow("k (sigma units):", kspin)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Run")
        outer.addWidget(btns)

        def check_and_accept():
            p1, p2 = fields['f1'].text().strip(), fields['f2'].text().strip()
            if not p1 or not p2:
                QMessageBox.warning(dlg, title, "Please select both the EARLIER and the LATER image.")
            elif not (os.path.isfile(p1) and os.path.isfile(p2)):
                QMessageBox.warning(dlg, title, "One of the selected files does not exist.")
            elif os.path.abspath(p1) == os.path.abspath(p2):
                QMessageBox.warning(dlg, title, "The earlier and later images are the same file.")
            else:
                dlg.accept()
        btns.accepted.connect(check_and_accept); btns.rejected.connect(dlg.reject)
        if dlg.exec_() != QDialog.Accepted:
            return None
        opts = None
        if with_options:
            opts = dict(normalise=norm.isChecked(), method="sigma" if meth.currentIndex() == 0 else "otsu", k=kspin.value())
        return fields['f1'].text().strip(), fields['f2'].text().strip(), opts

    def change_sensitivity(self):
        """How much does the reported change depend on the threshold k and on radiometric normalisation?"""
        try:
            picked = self._two_image_dialog("Change detection: sensitivity", with_options=False)
            if picked is None:
                return
            f1, f2, _o = picked

            def job(progress):
                progress(10, "Reading images")
                img1, p1 = ImageProcessor.load_image(f1)
                img2, p2 = ImageProcessor.load_image(f2)
                change_detection.check_alignment(p1, p2)
                progress(50, "Computing the sensitivity table")
                return change_detection.change_sensitivity(img1.astype('float32'), img2.astype('float32'), nodata=p1.get('nodata'))
            self._sens_progress = QProgressDialog("Starting...", None, 0, 100, self)
            self._sens_progress.setWindowTitle("Sensitivity analysis")
            self._sens_progress.setWindowModality(Qt.WindowModal)
            self._sens_progress.setMinimumDuration(0)
            worker = FuncWorker(job)
            worker.progress.connect(lambda v, m: (self._sens_progress.setValue(v), self._sens_progress.setLabelText(m)))
            worker.done.connect(self._on_change_sensitivity_done)
            worker.failed.connect(lambda m: (self._sens_progress.close(), self.show_error(f"Sensitivity analysis failed:\n{m}")))
            self._sens_worker = worker
            worker.start()
        except Exception as exc:
            self.show_error(f"Sensitivity analysis failed: {exc}")

    def _on_change_sensitivity_done(self, rows):
        self._sens_progress.close()
        dlg = QDialog(self)
        dlg.setWindowTitle("Change detection: sensitivity")
        dlg.setMinimumSize(560, 380)
        lay = QVBoxLayout(dlg)
        box = QTextEdit(); box.setReadOnly(True); box.setFontFamily("Courier New")
        box.setPlainText(change_detection.format_sensitivity(rows)); lay.addWidget(box)
        ok = QDialogButtonBox(QDialogButtonBox.Ok); ok.accepted.connect(dlg.accept); lay.addWidget(ok)
        dlg.exec_()

    def detect_changes(self, control=False):
        try:
            picked = self._two_image_dialog("Change detection", with_options=True)
            if picked is None:
                return
            f1, f2, opts = picked

            def job(progress):
                progress(10, "Reading images")
                img1, p1 = ImageProcessor.load_image(f1)
                img2, p2 = ImageProcessor.load_image(f2)
                change_detection.check_alignment(p1, p2)
                nodata = p1.get('nodata')
                t = p1['transform']
                area = abs(t.a * t.e) if (p1['crs'] is not None and not p1['crs'].is_geographic) else None
                progress(50, "Analysing changes")
                res = change_detection.detect_changes(img1.astype('float32'), img2.astype('float32'),
                                                      nodata=nodata, pixel_area_m2=area, **opts)
                return res, p1, f1, f2
            self._chg_progress = QProgressDialog("Starting...", None, 0, 100, self)
            self._chg_progress.setWindowTitle("Change detection")
            self._chg_progress.setWindowModality(Qt.WindowModal)
            self._chg_progress.setMinimumDuration(0)
            self._chg_control = control is True
            worker = FuncWorker(job)
            worker.progress.connect(lambda v, m: (self._chg_progress.setValue(v), self._chg_progress.setLabelText(m)))
            worker.done.connect(self._on_change_done)
            worker.failed.connect(lambda m: (self._chg_progress.close(), self.show_error(f"Change detection failed:\n{m}")))
            self._chg_worker = worker
            worker.start()
        except Exception as exc:
            self.show_error(f"Change detection failed: {exc}")

    def _on_change_done(self, payload):
        self._chg_progress.close()
        res, prof, f1, f2 = payload
        if getattr(self, '_chg_control', False):
            s = res.stats
            QMessageBox.information(
                self, "False-alarm test",
                f"{s['changed_percent']:.2f}% of the {s['valid_pixels']:,} valid pixels were flagged as change "
                f"(threshold {res.threshold:.1f} sigma, {res.method}).\n\n"
                "For two images that show NO real change, this is the false-alarm rate of the method with these settings.\n"
                "If it is not close to 0, raise k, check co-registration and radiometry, or use images from the same season.\n\n"
                + ("\n".join("WARNING: " + w for w in res.warnings) if res.warnings else ""))
            self._chg_control = False
            return
        self.change_result = res
        self.profile = prof
        self.display_change_map(np.clip(res.magnitude, 0, 2 * res.threshold))
        s = res.stats
        rows = "".join(f"<tr><td><b>{change_detection.CLASS_NAMES[c]}</b></td><td>{s['counts'][c]:,} px</td>"
                       f"<td>{s['percent'][c]:.2f}%</td></tr>" for c in (1, 2, 3))
        area = f"<tr><td><b>Changed area</b></td><td colspan='2'>{s['changed_area_ha']:.2f} ha</td></tr>" if s['changed_area_ha'] is not None else ""
        self.info_box.setHtml(
            '<h4 style="color:#3498db;">Change detection (CVA, normalised)</h4>'
            f"<table><tr><td><b>Threshold</b></td><td colspan='2'>{res.threshold:.2f} σ ({res.method})</td></tr>"
            f"<tr><td><b>Valid pixels</b></td><td colspan='2'>{s['valid_pixels']:,} / {s['total_pixels']:,}</td></tr>"
            f"{rows}{area}</table>"
            '<p style="color:#c0392b"><b>Candidate change, not validated.</b> Use the menu <i>Validation</i> '
            '(1. Create Sampling Points, 2. Compute Accuracy) and the <i>Sensitivity Analysis</i> / <i>False-Alarm Test</i> items.</p>')
        if res.warnings:
            QMessageBox.warning(self, "Check the result", "\n\n".join(res.warnings))
        self.add_layer_to_list("Change magnitude", res.magnitude, prof, is_change_map=True)
        self.ask_to_save_change_results()

    def ask_to_save_change_results(self):
        if QMessageBox.question(self, "Save results", "Save magnitude, class map and report?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            self.save_change_results()

    def save_change_results(self):
        try:
            import rasterio
            base, _ = QFileDialog.getSaveFileName(self, "Save change results (base name)", "", "GeoTIFF (*.tif)")
            if not base:
                return
            base = os.path.splitext(base)[0]
            res, prof = self.change_result, self.profile
            for suffix, arr, dtype, nodata in (("_magnitude", res.magnitude, "float32", -9999.0),
                                              ("_classes", change_detection.export_classes(res.classes), "uint8", 255)):
                p = {k: v for k, v in prof.items() if k in ("crs", "transform", "width", "height")}
                p.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="lzw")
                data = np.where(np.isfinite(arr), arr, nodata).astype(dtype) if dtype == "float32" else arr
                with rasterio.open(base + suffix + ".tif", "w", **p) as dst:
                    dst.write(data, 1)
            with open(base + "_report.txt", "w", encoding="utf-8") as f:
                f.write(change_detection.format_report(res))
            self.show_status(f"Saved {base}_magnitude.tif, _classes.tif, _report.txt", 8000)
            st = self._snap_qsettings()
            st.setValue("val/raster", base + "_classes.tif"); st.setValue("val/groups", "change:2,3;no_change:1")
            if QMessageBox.question(self, "Validation",
                                    "Create validation sampling points for this change map now?\n"
                                    "(Classes: 1 no change, 2 change, 3 strong change, 255 no data.\nYou can also do it later from the Validation menu.)",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                self.validation_sample_points()
        except Exception as exc:
            self.show_error(f"Could not save change results: {exc}")

    def display_change_map(self, change_map, profile=None):
        """
        عرض خريطة التغير مع colorbar (في الإحداثيات الحقيقية للصورة).
        """
        if profile is None:
            profile = getattr(self, 'profile', None) or self._inherited_profile(np.shape(change_map))
        # مسح المحتوى السابق
        self.ax.clear()
        
        # إزالة colorbar إذا كان موجودًا
        if hasattr(self, 'cbar'):
            self.cbar.remove()
            delattr(self, 'cbar')
        
        # عرض خريطة التغير
        cax, _spec = georef.draw_raster(self.ax, change_map, profile, cmap='viridis')
        
        # إضافة colorbar
        divider = make_axes_locatable(self.ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.fig.colorbar(cax, cax=cbar_ax, orientation='vertical', label='Change Intensity')
        
        # تحديث canvas
        self.canvas.draw()

        # إظهار label فقط مع خريطة التغير
        self.stats_label.setVisible(True)

    

    def raster_calculator(self):
        """Raster Calculator window with clickable layer list and save option"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Raster Calculator")
        layout = QHBoxLayout()
        
        # Layer list section
        layer_frame = QFrame()
        layer_layout = QVBoxLayout()
        layer_frame.setLayout(layer_layout)
        
        self.layer_list_calc = QListWidget()
        self.layer_list_calc.addItems(self.layers.keys())
        self.layer_list_calc.itemClicked.connect(self.add_layer_to_formula)
        
        layer_layout.addWidget(QLabel("Available Layers (Click to Add):"))
        layer_layout.addWidget(self.layer_list_calc)
        
        # Formula input section
        formula_frame = QFrame()
        formula_layout = QVBoxLayout()
        formula_frame.setLayout(formula_layout)
        
        self.formula_edit = QTextEdit()
        btn_calculate = QPushButton("Calculate")
        btn_calculate.clicked.connect(lambda: self.execute_calculation(dialog))
        
        formula_layout.addWidget(QLabel("Calculation Formula:"))
        formula_layout.addWidget(self.formula_edit)
        formula_layout.addWidget(btn_calculate)
        
        layout.addWidget(layer_frame, 30)
        layout.addWidget(formula_frame, 70)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def add_layer_to_formula(self, item):
        """Add clicked layer to formula with quotes"""
        current_text = self.formula_edit.toPlainText()
        layer_name = f'"{item.text()}"'  # Maintain quoted layer names
        self.formula_edit.setText(f"{current_text} {layer_name}")
    
    
    def execute_calculation(self, dialog):
        """Execute raster calculation and prompt to save results"""
        try:
            formula = self.formula_edit.toPlainText().strip()
            if not formula:
                QMessageBox.warning(self, "Empty Formula", "Please enter a formula")
                return

            names = raster_expr.layer_names(formula)
            missing = [n for n in names if n not in self.layers]
            if missing:
                raise ValueError(f"Missing layers: {', '.join(missing)}")
            arrays = {}
            for n in names:
                if 'image' not in self.layers[n]:
                    raise ValueError(f"Layer '{n}' is not a raster")
                arrays[n] = self.layers[n]['image']
            # Evaluate with the whitelisted expression engine (no eval of arbitrary code, 0-division -> NaN)
            result = raster_expr.evaluate(formula, arrays)
            if result.ndim == 2:
                result = result[np.newaxis, ...]
            layer_names = names

            # Create new layer
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_layer = f"Calculation_{timestamp}"
            profile = dict(self.layers[layer_names[0]]['profile'])
            profile.update(dtype='float32', nodata=float('nan'), count=result.shape[0])
            
            self.layers[new_layer] = {
                'image': result,
                'profile': profile,
                'file_name': f"Formula: {formula}",
                'formula': formula  # Store original formula
            }

            # Update UI
            self.layer_list.addItem(QListWidgetItem(new_layer))
            dialog.accept()
            self.display_raster_layer(result, new_layer)

            # Show statistics and ask about saving
            stats = (f"Calculation Statistics:\n"
                    f"Dimensions: {result.shape[1]}x{result.shape[2]}\n"
                    f"Data Range: {np.nanmin(result):.4g} to {np.nanmax(result):.4g} (NaN = undefined)\n"
                    f"Formula: {formula}")
            
            msg = QMessageBox()
            msg.setWindowTitle("Calculation Complete")
            msg.setText(stats)
            msg.setInformativeText("Would you like to save the results?")
            msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
            msg.setDefaultButton(QMessageBox.Save)
            
            if msg.exec_() == QMessageBox.Save:
                self.save_raster_calculation(
                    result,
                    profile,
                    "raster_calculation",
                    new_layer
                )

        except Exception as e:
            error_msg = (f"Calculation failed:\n{str(e)}\n\n"
                        f"Formula: {formula}\n"
                        f"Possible issues:\n"
                        f"1. Invalid syntax\n"
                        f"2. Missing layers\n"
                        f"3. Dimension mismatch")
            QMessageBox.critical(self, "Error", error_msg)

    def save_raster_calculation(self, data, profile, data_type, default_name):
        """Save raster calculation results to file"""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                f"Save {data_type.replace('_', ' ').title()}",
                f"{default_name}.tif",
                "GeoTIFF Files (*.tif);;All Files (*)",
                options=options
            )
            
            if not file_path:
                return
                
            # Ensure proper data structure
            if data.ndim == 2:
                data = data[np.newaxis, ...]
                
            # Prepare profile for saving
            save_profile = profile.copy()
            save_profile.update({
                'driver': 'GTiff',
                'dtype': data.dtype,
                'count': data.shape[0],
                'nodata': None
            })
            
            # Save the data
            with rasterio.open(file_path, 'w', **save_profile) as dst:
                dst.write(data)
                
            QMessageBox.information(self, "Success", 
                                f"Results saved successfully to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", 
                            f"Failed to save results: {str(e)}")
    
####################################### Raster Projection ####################################33
    

 

    def project_raster(self):
        """Project raster to target CRS (UTM or user-defined) with comprehensive CRS handling"""
        try:
            # 1. Validate layer selection
            if not self.layer_list.currentItem():
                QMessageBox.warning(self, "Error", "Please select a raster layer first")
                return

            layer_name = self.layer_list.currentItem().text()
            layer = self.layers.get(layer_name)
            if not layer:
                QMessageBox.warning(self, "Error", "Selected layer not found")
                return

            # 2. Validate raster data
            if 'image' not in layer or 'profile' not in layer:
                QMessageBox.warning(self, "Error", "Selected layer is missing required raster data")
                return

            # Get original nodata value (matches original function exactly)
            nodata = layer['profile'].get('nodata', None)

            # 3. Create temporary file if needed (matches original)
            temp_file_created = False
            if 'path' in layer and layer['path'] and os.path.exists(layer['path']):
                raster_path = layer['path']
            elif hasattr(layer['image'], 'name') and os.path.exists(layer['image'].name):
                raster_path = layer['image'].name
            else:
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
                        raster_path = temp_file.name
                    with rasterio.open(raster_path, 'w', **layer['profile']) as dst:
                        dst.write(layer['image'])
                    temp_file_created = True
                    layer['temp_path'] = raster_path
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not create temporary file: {str(e)}")
                    return

            # 4. Analyze source raster and prepare CRS options
            try:
                with rasterio.open(raster_path) as src:
                    src_crs = src.crs
                    if not src_crs or not src_crs.is_valid:
                        QMessageBox.critical(self, "Error", "Raster has invalid or missing coordinate reference system")
                        return

                    # Get source CRS information
                    src_crs_type = "projected" if src_crs.is_projected else "geographic"
                    src_units = "meters" if src_crs.is_projected else "degrees"
                    
                    # Calculate bounds in WGS84 for UTM zone determination
                    try:
                        transformer = pyproj.Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)
                        bounds = {
                            'min_lon': transformer.transform(src.bounds.left, src.bounds.bottom)[0],
                            'max_lon': transformer.transform(src.bounds.right, src.bounds.top)[0],
                            'min_lat': transformer.transform(src.bounds.left, src.bounds.bottom)[1],
                            'max_lat': transformer.transform(src.bounds.right, src.bounds.top)[1]
                        }

                        # Validate bounds
                        if not (-180 <= bounds['min_lon'] <= 180 and -180 <= bounds['max_lon'] <= 180 and
                                -90 <= bounds['min_lat'] <= 90 and -90 <= bounds['max_lat'] <= 90):
                            QMessageBox.critical(self, "Error", "Invalid geographic bounds")
                            return

                        # Calculate suggested UTM zone
                        center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
                        center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
                        suggested_zone = int((center_lon + 180) / 6) + 1
                        suggested_hemi = 'N' if center_lat >= 0 else 'S'
                        suggested_epsg = f"EPSG:{32600 + suggested_zone if suggested_hemi == 'N' else 32700 + suggested_zone}"
                    except Exception as e:
                        QMessageBox.warning(self, "Warning", f"Could not determine UTM zone: {str(e)}")
                        suggested_zone = None
                        suggested_hemi = None
                        suggested_epsg = None

                    # Store resolution info (matches original)
                    resolution = {
                        'x': src.transform.a,
                        'y': abs(src.transform.e),
                        'units': src_units
                    }

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to analyze raster: {str(e)}")
                if temp_file_created:
                    os.remove(raster_path)
                    del layer['temp_path']
                return

            # 5. Create projection dialog with enhanced CRS options
            dialog = QDialog(self)
            dialog.setWindowTitle("Raster Projection Options")
            dialog.setMinimumWidth(600)
            layout = QVBoxLayout()

            # Current CRS info
            current_box = QGroupBox("Current Coordinate System")
            current_layout = QVBoxLayout()
            current_layout.addWidget(QLabel(f"<b>CRS:</b> {src_crs.to_string() if src_crs else 'Unknown'}"))
            current_layout.addWidget(QLabel(f"<b>Type:</b> {src_crs_type.capitalize()}"))
            current_layout.addWidget(QLabel(f"<b>Resolution:</b> {resolution['x']:.6f} x {resolution['y']:.6f} {resolution['units']}"))
            current_layout.addWidget(QLabel(f"<b>Nodata:</b> {nodata if nodata is not None else 'None'}"))
            current_box.setLayout(current_layout)
            layout.addWidget(current_box)

            # Target CRS Selection
            crs_box = QGroupBox("Target Coordinate System")
            crs_layout = QVBoxLayout()
            
            # CRS selection method
            method_group = QButtonGroup()
            utm_radio = QRadioButton("UTM Zone (Auto-detected)")
            epsg_radio = QRadioButton("EPSG Code")
            proj_radio = QRadioButton("PROJ String")
            wkt_radio = QRadioButton("WKT String")
            
            method_group.addButton(utm_radio)
            method_group.addButton(epsg_radio)
            method_group.addButton(proj_radio)
            method_group.addButton(wkt_radio)
            
            crs_method_layout = QHBoxLayout()
            crs_method_layout.addWidget(utm_radio)
            crs_method_layout.addWidget(epsg_radio)
            crs_method_layout.addWidget(proj_radio)
            crs_method_layout.addWidget(wkt_radio)
            crs_layout.addLayout(crs_method_layout)
            
            # Default to UTM if available, otherwise EPSG
            if suggested_epsg:
                utm_radio.setChecked(True)
            else:
                epsg_radio.setChecked(True)
                QMessageBox.warning(self, "Notice", "Could not auto-detect UTM zone. Please specify target CRS manually.")

            # UTM Zone Selection (only shown when UTM selected)
            utm_widget = QWidget()
            utm_layout = QHBoxLayout()
            self.zone_combo = QComboBox()
            for zone in range(1, 61):
                self.zone_combo.addItem(f"Zone {zone}", userData=zone)
            if suggested_zone:
                self.zone_combo.setCurrentIndex(suggested_zone - 1)
            utm_layout.addWidget(self.zone_combo)

            self.hemi_combo = QComboBox()
            self.hemi_combo.addItem("Northern Hemisphere", 'N')
            self.hemi_combo.addItem("Southern Hemisphere", 'S')
            if suggested_hemi:
                self.hemi_combo.setCurrentIndex(0 if suggested_hemi == 'N' else 1)
            utm_layout.addWidget(self.hemi_combo)
            utm_widget.setLayout(utm_layout)
            crs_layout.addWidget(utm_widget)

            # EPSG input (only shown when EPSG selected)
            epsg_widget = QWidget()
            epsg_layout = QHBoxLayout()
            self.epsg_input = QLineEdit()
            self.epsg_input.setPlaceholderText("e.g., 4326 for WGS84")
            epsg_validate = QPushButton("Validate")
            self.epsg_status = QLabel()
            epsg_layout.addWidget(QLabel("EPSG:"))
            epsg_layout.addWidget(self.epsg_input)
            epsg_layout.addWidget(epsg_validate)
            epsg_layout.addWidget(self.epsg_status)
            epsg_widget.setLayout(epsg_layout)
            crs_layout.addWidget(epsg_widget)

            # PROJ string input
            proj_widget = QWidget()
            proj_layout = QHBoxLayout()
            self.proj_input = QLineEdit()
            self.proj_input.setPlaceholderText("e.g., +proj=utm +zone=32 +datum=WGS84")
            proj_validate = QPushButton("Validate")
            self.proj_status = QLabel()
            proj_layout.addWidget(QLabel("PROJ:"))
            proj_layout.addWidget(self.proj_input)
            proj_layout.addWidget(proj_validate)
            proj_layout.addWidget(self.proj_status)
            proj_widget.setLayout(proj_layout)
            crs_layout.addWidget(proj_widget)

            # WKT string input
            wkt_widget = QWidget()
            wkt_layout = QVBoxLayout()
            self.wkt_input = QTextEdit()
            self.wkt_input.setPlaceholderText("Paste WKT CRS definition here...")
            wkt_validate = QPushButton("Validate")
            self.wkt_status = QLabel()
            wkt_layout.addWidget(self.wkt_input)
            wkt_buttons = QHBoxLayout()
            wkt_buttons.addWidget(wkt_validate)
            wkt_buttons.addWidget(self.wkt_status)
            wkt_layout.addLayout(wkt_buttons)
            wkt_widget.setLayout(wkt_layout)
            crs_layout.addWidget(wkt_widget)

            # Connect signals to show/hide appropriate widgets
            def update_crs_inputs():
                utm_widget.setVisible(utm_radio.isChecked())
                epsg_widget.setVisible(epsg_radio.isChecked())
                proj_widget.setVisible(proj_radio.isChecked())
                wkt_widget.setVisible(wkt_radio.isChecked())
            
            utm_radio.toggled.connect(update_crs_inputs)
            epsg_radio.toggled.connect(update_crs_inputs)
            proj_radio.toggled.connect(update_crs_inputs)
            wkt_radio.toggled.connect(update_crs_inputs)
            update_crs_inputs()  # Initialize visibility

            # Validation functions
            def validate_epsg():
                try:
                    epsg_code = int(self.epsg_input.text())
                    crs = rasterio.crs.CRS.from_epsg(epsg_code)
                    if crs.is_valid:
                        self.epsg_status.setText("✓ Valid")
                        self.epsg_status.setStyleSheet("color: green")
                        return crs
                    else:
                        raise ValueError("Invalid EPSG code")
                except Exception as e:
                    self.epsg_status.setText("✗ Invalid")
                    self.epsg_status.setStyleSheet("color: red")
                    return None
            
            def validate_proj():
                try:
                    proj_str = self.proj_input.text()
                    crs = rasterio.crs.CRS.from_proj4(proj_str) if proj_str.startswith('+proj') else rasterio.crs.CRS.from_string(proj_str)
                    if crs.is_valid:
                        self.proj_status.setText("✓ Valid")
                        self.proj_status.setStyleSheet("color: green")
                        return crs
                    else:
                        raise ValueError("Invalid PROJ string")
                except Exception as e:
                    self.proj_status.setText("✗ Invalid")
                    self.proj_status.setStyleSheet("color: red")
                    return None
            
            def validate_wkt():
                try:
                    wkt_str = self.wkt_input.toPlainText()
                    crs = rasterio.crs.CRS.from_wkt(wkt_str)
                    if crs.is_valid:
                        self.wkt_status.setText("✓ Valid")
                        self.wkt_status.setStyleSheet("color: green")
                        return crs
                    else:
                        raise ValueError("Invalid WKT string")
                except Exception as e:
                    self.wkt_status.setText("✗ Invalid")
                    self.wkt_status.setStyleSheet("color: red")
                    return None

            epsg_validate.clicked.connect(validate_epsg)
            proj_validate.clicked.connect(validate_proj)
            wkt_validate.clicked.connect(validate_wkt)

            crs_box.setLayout(crs_layout)
            layout.addWidget(crs_box)

            # Resolution controls
            res_box = QGroupBox("Output Resolution")
            res_layout = QFormLayout()
            self.res_check = QCheckBox("Set custom resolution:")
            self.x_res = QDoubleSpinBox()
            self.x_res.setRange(0.0001, 100000)
            self.x_res.setValue(10 if resolution['units'] == 'degrees' else resolution['x'])
            self.x_res.setDecimals(6)
            self.y_res = QDoubleSpinBox()
            self.y_res.setRange(0.0001, 100000)
            self.y_res.setValue(10 if resolution['units'] == 'degrees' else resolution['y'])
            self.y_res.setDecimals(6)
            
            res_layout.addRow(self.res_check)
            res_layout.addRow("X resolution:", self.x_res)
            res_layout.addRow("Y resolution:", self.y_res)
            res_box.setLayout(res_layout)
            layout.addWidget(res_box)

            # Nodata handling (exactly matches original function)
            nodata_box = QGroupBox("Nodata Value")
            nodata_layout = QHBoxLayout()
            self.nodata_check = QCheckBox("Preserve nodata value")
            self.nodata_check.setChecked(nodata is not None)
            self.nodata_value = QLineEdit(str(nodata) if nodata is not None else "")
            self.nodata_value.setEnabled(nodata is not None)
            self.nodata_check.toggled.connect(lambda checked: self.nodata_value.setEnabled(checked and nodata is not None))
            nodata_layout.addWidget(self.nodata_check)
            nodata_layout.addWidget(self.nodata_value)
            nodata_box.setLayout(nodata_layout)
            layout.addWidget(nodata_box)

            # Resampling method
            resample_box = QGroupBox("Resampling Method")
            resample_layout = QVBoxLayout()
            self.resample_combo = QComboBox()
            self.resample_combo.addItems(["Nearest", "Bilinear", "Cubic", "Cubic Spline", "Lanczos", "Average", "Mode"])
            resample_layout.addWidget(self.resample_combo)
            resample_box.setLayout(resample_layout)
            layout.addWidget(resample_box)

            # Dialog buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            dialog.setLayout(layout)

            if not dialog.exec_():
                if temp_file_created:
                    os.remove(raster_path)
                    del layer['temp_path']
                return

            # Determine target CRS based on user selection
            target_crs = None
            if utm_radio.isChecked():
                zone = self.zone_combo.currentData()
                hemi = self.hemi_combo.currentData()
                target_epsg = f"EPSG:{32600 + zone if hemi == 'N' else 32700 + zone}"
                try:
                    target_crs = rasterio.crs.CRS.from_string(target_epsg)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Invalid UTM zone specification: {str(e)}")
                    return
            elif epsg_radio.isChecked():
                target_crs = validate_epsg()
                if not target_crs:
                    QMessageBox.critical(self, "Error", "Please enter a valid EPSG code")
                    return
            elif proj_radio.isChecked():
                target_crs = validate_proj()
                if not target_crs:
                    QMessageBox.critical(self, "Error", "Please enter a valid PROJ string")
                    return
            elif wkt_radio.isChecked():
                target_crs = validate_wkt()
                if not target_crs:
                    QMessageBox.critical(self, "Error", "Please enter a valid WKT string")
                    return

            if not target_crs or not target_crs.is_valid:
                QMessageBox.critical(self, "Error", "Invalid target coordinate system")
                return

            # Get other parameters
            resample_method = self.resample_combo.currentText().lower().replace(' ', '_')
            custom_res = (self.x_res.value(), self.y_res.value()) if self.res_check.isChecked() else None
            preserve_nodata = self.nodata_check.isChecked()
            try:
                target_nodata = float(self.nodata_value.text()) if preserve_nodata and self.nodata_value.text() else None
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid nodata value (must be a number)")
                return

            # Perform reprojection with original nodata handling
            try:
                with rasterio.open(raster_path) as src:
                    # Calculate transform
                    if custom_res:
                        transform, width, height = rasterio.warp.calculate_default_transform(
                            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=custom_res)
                    else:
                        transform, width, height = rasterio.warp.calculate_default_transform(
                            src.crs, target_crs, src.width, src.height, *src.bounds)

                    # Prepare output - exactly matches original function's nodata handling
                    dst_array = np.empty((src.count, height, width), dtype=src.dtypes[0])
                    
                    # Exactly match the original function's nodata behavior
                    if preserve_nodata and target_nodata is not None:
                        dst_array.fill(target_nodata)
                        src_nodata = src.nodata
                    else:
                        src_nodata = None
                        target_nodata = None

                    # Reproject with same nodata parameters as original
                    rasterio.warp.reproject(
                        source=src.read(),
                        destination=dst_array,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=getattr(rasterio.warp.Resampling, resample_method),
                        src_nodata=src_nodata,
                        dst_nodata=target_nodata)

                    # Update layer with same nodata handling as original
                    layer['image'] = dst_array
                    layer['profile'].update({
                        'crs': target_crs,
                        'transform': transform,
                        'width': width,
                        'height': height,
                        'res_x': transform.a,
                        'res_y': abs(transform.e)
                    })
                    
                    # Match original function's nodata storage exactly
                    if preserve_nodata and target_nodata is not None:
                        layer['profile']['nodata'] = target_nodata
                    else:
                        if 'nodata' in layer['profile']:
                            del layer['profile']['nodata']

                    # Ask user for save action (matches original)
                    save_dialog = QMessageBox(self)
                    save_dialog.setWindowTitle("Reprojection Complete")
                    save_dialog.setText("Projection completed successfully. What would you like to do?")
                    save_dialog.setIcon(QMessageBox.Question)
                    
                    save_to_file = save_dialog.addButton("Save to File", QMessageBox.ActionRole)
                    save_to_layer = save_dialog.addButton("Add to Layers", QMessageBox.ActionRole)
                    save_dialog.addButton(QMessageBox.Cancel)
                    
                    save_dialog.exec_()

                    clicked_button = save_dialog.clickedButton()

                    if clicked_button == save_to_file:
                        # Enhanced file saving with user feedback
                        crs_name = f"UTM{zone}{hemi}" if utm_radio.isChecked() else f"EPSG{target_crs.to_epsg()}" if target_crs.to_epsg() else "projected"
                        default_name = f"{layer_name}_{crs_name}.tif"
                        file_path, _ = QFileDialog.getSaveFileName(
                            self, 
                            "Save Projected Raster", 
                            default_name,
                            "GeoTIFF Files (*.tif *.tiff);;All Files (*)"
                        )

                        if file_path:
                            # Ensure .tif extension
                            if not file_path.lower().endswith(('.tif', '.tiff')):
                                file_path += '.tif'

                            try:
                                # Save the file with same profile handling as original
                                profile = layer['profile'].copy()
                                if not preserve_nodata and 'nodata' in profile:
                                    del profile['nodata']
                                    
                                with rasterio.open(file_path, 'w', **profile) as dst:
                                    dst.write(layer['image'])

                                # Get absolute path for user clarity
                                abs_path = os.path.abspath(file_path)
                                dir_name = os.path.dirname(abs_path)
                                file_name = os.path.basename(abs_path)

                                # Generate success message
                                res_x = layer['profile']['res_x']
                                res_y = layer['profile']['res_y']
                                message = (
                                    f"<b>✓ Raster saved successfully!</b><br><br>"
                                    f"<b>Folder:</b> {dir_name}<br>"
                                    f"<b>File:</b> {file_name}<br><br>"
                                    f"<b>Projection:</b> {target_crs.to_string()}<br>"
                                    f"<b>Resolution:</b> {res_x:.6f}m × {res_y:.6f}m<br>"
                                    f"<b>Nodata:</b> {layer['profile'].get('nodata', 'None')}"
                                )

                                # Create a rich-text message box
                                msg_box = QMessageBox(self)
                                msg_box.setWindowTitle("Success")
                                msg_box.setTextFormat(Qt.RichText)
                                msg_box.setText(message)
                                
                                # Add a "Copy Path" button
                                copy_button = msg_box.addButton("Copy Path", QMessageBox.ActionRole)
                                msg_box.addButton(QMessageBox.Ok)
                                msg_box.exec_()

                                # Copy path to clipboard if requested
                                if msg_box.clickedButton() == copy_button:
                                    clipboard = QApplication.clipboard()
                                    clipboard.setText(abs_path)
                                    QMessageBox.information(self, "Copied", "Path copied to clipboard!")

                            except Exception as e:
                                QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")

                    elif clicked_button == save_to_layer:
                        # Add to layers with automatic naming (matches original behavior)
                        crs_name = f"UTM{zone}{hemi}" if utm_radio.isChecked() else f"EPSG{target_crs.to_epsg()}" if target_crs.to_epsg() else "projected"
                        new_layer_name = f"{layer_name}_{crs_name}"
                        counter = 1
                        while new_layer_name in self.layers:
                            new_layer_name = f"{layer_name}_{crs_name}_{counter}"
                            counter += 1
                        
                        # Create new layer with same profile handling as original
                        new_profile = layer['profile'].copy()
                        if not preserve_nodata and 'nodata' in new_profile:
                            del new_profile['nodata']
                        
                        self.layers[new_layer_name] = {
                            'image': layer['image'],
                            'profile': new_profile,
                            'path': None
                        }
                        self.layer_list.addItem(new_layer_name)
                        
                        QMessageBox.information(
                            self, "Success", 
                            f"Added to layers as: {new_layer_name}\n"
                            f"Resolution: {layer['profile']['res_x']:.6f}m × {layer['profile']['res_y']:.6f}m"
                        )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Reprojection failed: {str(e)}")
            finally:
                if temp_file_created and os.path.exists(raster_path):
                    try:
                        os.remove(raster_path)
                        del layer['temp_path']
                    except Exception as e:
                        logging.info(f"Warning: Could not delete temp file: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")         


####################################################################################################3
        

    


    def show_overlay_dialog(self):
        """
        Open an enhanced dialog to select layers for overlay with individual transparency control.
        Includes layer ordering and preview functionality.
        """
        try:
            # Create a dialog for layer selection
            dialog = QDialog(self)
            dialog.setWindowTitle("Layer Overlay Control")
            dialog.setMinimumWidth(400)
            layout = QVBoxLayout()

            # Create a splitter for better layout organization
            splitter = QSplitter(Qt.Vertical)

            # Layer selection and ordering panel
            layer_panel = QWidget()
            layer_layout = QVBoxLayout()
            
            # Layer list with drag-and-drop reordering
            layer_list = QListWidget()
            layer_list.setDragDropMode(QListWidget.InternalMove)
            layer_list.setSelectionMode(QListWidget.MultiSelection)
            layer_list.addItems(self.layers.keys())
            
            # Add checkboxes for visibility toggle
            for i in range(layer_list.count()):
                item = layer_list.item(i)
                widget = QWidget()
                hbox = QHBoxLayout()
                hbox.setContentsMargins(0, 0, 0, 0)
                
                checkbox = QCheckBox()
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(lambda state, item=item: self.toggle_layer_visibility(item.text(), state))
                
                hbox.addWidget(checkbox)
                hbox.addWidget(QLabel(item.text()))
                widget.setLayout(hbox)
                
                item_widget = QListWidgetItem()
                item_widget.setSizeHint(widget.sizeHint())
                layer_list.setItemWidget(item_widget, widget)
                layer_list.addItem(item_widget)
            
            layer_layout.addWidget(QLabel("Drag to reorder layers (top is rendered last):"))
            layer_layout.addWidget(layer_list)
            layer_panel.setLayout(layer_layout)
            
            # Transparency control panel
            transparency_panel = QWidget()
            transparency_layout = QVBoxLayout()
            self.transparency_sliders = {}
            self.layer_opacity_labels = {}
            
            # Add a slider for each layer
            for layer_name in self.layers.keys():
                slider_container = QWidget()
                hbox = QHBoxLayout()
                
                label = QLabel(f"{layer_name}:")
                opacity_label = QLabel("50%")
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(100)
                slider.setValue(50)
                
                # Connect slider to update function
                slider.valueChanged.connect(lambda value, name=layer_name: self.update_layer_opacity(name, value))
                
                hbox.addWidget(label)
                hbox.addWidget(slider)
                hbox.addWidget(opacity_label)
                slider_container.setLayout(hbox)
                
                self.transparency_sliders[layer_name] = slider
                self.layer_opacity_labels[layer_name] = opacity_label
                transparency_layout.addWidget(slider_container)
            
            transparency_panel.setLayout(transparency_layout)
            
            # Add panels to splitter
            splitter.addWidget(layer_panel)
            splitter.addWidget(transparency_panel)
            
            # Add preview and apply buttons
            button_box = QDialogButtonBox()
            preview_button = QPushButton("Preview")
            preview_button.clicked.connect(lambda: self.update_overlay_preview())
            button_box.addButton(preview_button, QDialogButtonBox.ActionRole)
            
            apply_button = QPushButton("Apply")
            apply_button.clicked.connect(dialog.accept)
            button_box.addButton(apply_button, QDialogButtonBox.AcceptRole)
            
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(dialog.reject)
            button_box.addButton(cancel_button, QDialogButtonBox.RejectRole)
            
            # Main layout organization
            layout.addWidget(splitter)
            layout.addWidget(button_box)
            dialog.setLayout(layout)
            
            # Show the dialog
            if dialog.exec_() == QDialog.Accepted:
                self.apply_overlay_changes()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show overlay dialog: {str(e)}")
    def toggle_layer_visibility(self, layer_name, state):
        """Toggle layer visibility in the overlay."""
        visible = state == Qt.Checked
        if layer_name in self.layers:
            self.layers[layer_name]['visible'] = visible
            self.update_overlay_preview()

    def update_layer_opacity(self, layer_name, value):
        """Update layer opacity and display."""
        if layer_name in self.layer_opacity_labels:
            self.layer_opacity_labels[layer_name].setText(f"{value}%")
        if layer_name in self.layers:
            self.layers[layer_name]['opacity'] = value / 100.0
            self.update_overlay_preview()

    def update_overlay_preview(self):
        """Update the overlay preview without finalizing changes."""
        try:
            # Get all layers in current order with their opacity
            layer_order = []
            transparency_values = {}
            
            for i in range(self.layer_list.count()):
                layer_name = self.layer_list.item(i).text()
                if self.layers[layer_name].get('visible', True):
                    layer_order.append(layer_name)
                    transparency_values[layer_name] = self.layers[layer_name].get('opacity', 0.5)
            
            # Update the overlay
            self.overlay_layers(layer_order, transparency_values)
            
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not update preview: {str(e)}")

    def apply_overlay_changes(self):
        """Finalize the overlay changes."""
        self.update_overlay_preview()  # Essentially the same as preview but signals completion
        QMessageBox.information(self, "Success", "Overlay applied successfully.")

    # def overlay_layers(self, layer_order, transparency_values):
    #     """
    #     Efficiently overlay multiple layers with transparency control.
        
    #     :param layer_order: Ordered list of layer names (first is bottom, last is top)
    #     :param transparency_values: Dictionary of opacity values (0.0 to 1.0)
    #     """
    #     try:
    #         # Clear the current display
    #         self.ax.clear()
            
    #         # Create a composite image if there are multiple layers
    #         if len(layer_order) > 1:
    #             # Get the first layer as base
    #             base_layer = self.layers[layer_order[0]]['image']
    #             base_opacity = transparency_values.get(layer_order[0], 1.0)
                
    #             if base_layer.ndim == 3:  # RGB
    #                 composite = base_layer.transpose(1, 2, 0) * base_opacity
    #             else:  # Grayscale
    #                 composite = np.stack([base_layer[0]]*3, axis=-1) * base_opacity
                
    #             # Blend other layers
    #             for layer_name in layer_order[1:]:
    #                 layer_data = self.layers[layer_name]
    #                 if not layer_data.get('visible', True):
    #                     continue
                        
    #                 image = layer_data['image']
    #                 opacity = transparency_values.get(layer_name, 0.5)
                    
    #                 if image.ndim == 3:  # RGB
    #                     layer_img = image.transpose(1, 2, 0)
    #                 else:  # Grayscale
    #                     layer_img = np.stack([image[0]]*3, axis=-1)
                    
    #                 # Alpha blending
    #                 composite = composite * (1 - opacity) + layer_img * opacity
                
    #             # Display the composite image
    #             self.ax.imshow(composite)
    #         else:
    #             # Single layer case
    #             layer_name = layer_order[0]
    #             layer_data = self.layers[layer_name]
    #             if layer_data.get('visible', True):
    #                 image = layer_data['image']
    #                 opacity = transparency_values.get(layer_name, 1.0)
                    
    #                 if image.ndim == 3:  # RGB
    #                     self.ax.imshow(image.transpose(1, 2, 0), alpha=opacity)
    #                 else:  # Grayscale
    #                     self.ax.imshow(image[0], cmap='gray', alpha=opacity)
            
    #         # Update the canvas
    #         self.canvas.draw()
            
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Failed to overlay layers: {str(e)}")


    


    def overlay_layers(self, layer_names, transparency_values):
        """
        Overlay the selected layers on the canvas with individual transparency control.
        
        :param layer_names: List of layer names to overlay.
        :param transparency_values: Dictionary of transparency values for each layer (0.0 to 1.0).
        """
        try:
            # Clear the current display
            self.ax.clear()

            # Iterate through the selected layers and display them
            for layer_name in layer_names:
                if layer_name in self.layers:
                    layer_data = self.layers[layer_name]
                    image = layer_data['image']
                    transparency = transparency_values.get(layer_name, 0.5)  # Default transparency if not provided

                    lp = layer_data.get('profile') or {}
                    if image.ndim == 3 and image.shape[0] in (3, 4):  # RGB image
                        _, spec = georef.draw_raster(self.ax, image.transpose(1, 2, 0), lp, alpha=transparency)
                    elif image.ndim == 3:  # single band stored as (1, H, W) or a band stack: show the first band
                        _, spec = georef.draw_raster(self.ax, image[0], lp, cmap='gray', alpha=transparency)
                    else:  # Grayscale image
                        _, spec = georef.draw_raster(self.ax, image, lp, cmap='gray', alpha=transparency)

            # Update the canvas
            self.canvas.draw()
            QMessageBox.information(self, "Success", "Layers overlaid successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to overlay layers: {str(e)}")



    def choose_bands(self, num_bands):
        """
        Prompt the user to choose between displaying 1 band or 3 bands,
        then select the appropriate bands.
        """
        # عرض خيارات العرض: 1 Band أو 3 Bands
        display_option, ok = QInputDialog.getItem(
            self,
            "Display Option",
            "This image has {} bands. Choose display option:".format(num_bands),
            ["1 Band", "3 Bands"],  # الخيارات المتاحة
            0,  # الخيار الافتراضي (1 Band)
            False  # لا يسمح بالتحرير
        )
        
        if not ok:
            return None  # إذا ضغط المستخدم Cancel

        if display_option == "1 Band":
            # اختيار باند واحد
            band, ok = QInputDialog.getItem(
                self,
                "Select Band",
                "Choose 1 band to display:",
                [str(i) for i in range(1, num_bands + 1)],  # قائمة الباندات المتاحة
                0,  # الخيار الافتراضي
                False  # لا يسمح بالتحرير
            )
            if ok and band:
                return [int(band)]  # إرجاع الباند المختار كقائمة من عنصر واحد
            else:
                return None

        elif display_option == "3 Bands":
            # اختيار 3 باندات
            bands, ok = QInputDialog.getText(
                self,
                "Select Bands",
                "Enter 3 bands to display (comma-separated):"
            )
            if ok and bands:
                try:
                    # تحويل النص المدخل إلى قائمة من الأعداد الصحيحة
                    selected_bands = [int(band.strip()) for band in bands.split(',')]
                    if len(selected_bands) == 3:
                        # التأكد من أن الأبعاد المختارة صالحة
                        for band in selected_bands:
                            if band < 1 or band > num_bands:
                                QMessageBox.warning(self, "Warning", f"Invalid band: {band}. Please choose from 1 to {num_bands}.")
                                return None
                        return selected_bands
                    else:
                        QMessageBox.warning(self, "Warning", "Please select exactly 3 bands.")
                        return None
                except ValueError:
                    QMessageBox.warning(self, "Warning", "Invalid input. Please enter numbers separated by commas.")
                    return None
        return None
    


    
    ################################ Filters  ###############################3
    
    def _apply_filter(self, filter_func, *args):
        """Helper method to apply filter with statistics comparison"""
        if hasattr(self, 'image'):
            pushed = False
            try:
                # Capture pre-filter statistics
                pre_stats = self._get_image_stats(self.image)
                
                # Save current state for undo
                current_layer = self.layer_list.currentItem().text() if self.layer_list.currentItem() else "Filtered"
                self._push_undo()
                pushed = True
                
                # Apply the filter
                filtered_image = filter_func(self.image.copy(), *args)
                
                # Capture post-filter statistics
                post_stats = self._get_image_stats(filtered_image)
                
                # Show comparison
                self._show_stats_comparison(pre_stats, post_stats, filter_func.__name__)
                
                # Update image and display
                self.image = filtered_image
                filter_name = filter_func.__name__.replace("_", " ").title()
                new_layer_name = f"{current_layer}_{filter_name}"
                self.display_raster_layer(self.image, new_layer_name)
                
                # Update layers
                self.layers[new_layer_name] = {
                    'image': self.image,
                    'profile': self.profile,
                    'file_name': new_layer_name
                }
                self.layer_list.addItem(QListWidgetItem(new_layer_name))
                
                # Prompt to save
                self._prompt_save_filtered_image()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Filter failed: {str(e)}")
                if pushed and self.undo_stack:
                    self.undo_stack.pop()  # discard the snapshot; the image was never replaced

    def _get_image_stats(self, image):
        """Calculate comprehensive image statistics for all bands"""
        stats = {
            'dtype': str(image.dtype),
            'shape': image.shape,
            'nodata_pixels': np.sum(image == 0) if np.issubdtype(image.dtype, np.integer) else np.sum(np.isnan(image))
        }
        
        if image.ndim == 3:  # Multi-band
            stats['bands'] = []
            for i in range(image.shape[0]):  # Explicitly iterate through all bands
                band = image[i]
                valid_pixels = band[~np.isnan(band)] if np.issubdtype(band.dtype, np.floating) else band[band != 0]
                stats['bands'].append({
                    'min': np.min(valid_pixels) if valid_pixels.size > 0 else 0,
                    'max': np.max(valid_pixels) if valid_pixels.size > 0 else 0,
                    'mean': np.mean(valid_pixels) if valid_pixels.size > 0 else 0,
                    'std': np.std(valid_pixels) if valid_pixels.size > 0 else 0,
                    'valid_pixels': valid_pixels.size,
                    'values': valid_pixels
                })
        else:  # Single-band
            valid_pixels = image[~np.isnan(image)] if np.issubdtype(image.dtype, np.floating) else image[image != 0]
            stats.update({
                'min': np.min(valid_pixels) if valid_pixels.size > 0 else 0,
                'max': np.max(valid_pixels) if valid_pixels.size > 0 else 0,
                'mean': np.mean(valid_pixels) if valid_pixels.size > 0 else 0,
                'std': np.std(valid_pixels) if valid_pixels.size > 0 else 0,
                'valid_pixels': valid_pixels.size,
                'values': valid_pixels
            })
        
        return stats

    def _show_stats_comparison(self, pre_stats, post_stats, filter_name):
        """Display statistics comparison in a dialog with enhanced functionality"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Filter Results: {filter_name.replace('_', ' ').title()}")
        dialog.resize(800, 600)
        
        main_layout = QVBoxLayout()
        tab_widget = QTabWidget()
        
        # Statistics Tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        
        # Calculate number of rows needed
        num_bands = len(pre_stats['bands']) if 'bands' in pre_stats else 0
        table_rows = 2 + (num_bands * 4)  # Base rows + 4 metrics per band
        
        table = QTableWidget(table_rows, 4)
        table.setHorizontalHeaderLabels(["Metric", "Before", "After", "Change"])
        
        # Populate table
        row = 0
        table.setItem(row, 0, QTableWidgetItem("Data Type"))
        table.setItem(row, 1, QTableWidgetItem(pre_stats['dtype']))
        table.setItem(row, 2, QTableWidgetItem(post_stats['dtype']))
        table.setItem(row, 3, QTableWidgetItem(""))
        row += 1
        
        table.setItem(row, 0, QTableWidgetItem("NoData Pixels"))
        table.setItem(row, 1, QTableWidgetItem(str(pre_stats['nodata_pixels'])))
        table.setItem(row, 2, QTableWidgetItem(str(post_stats['nodata_pixels'])))
        table.setItem(row, 3, QTableWidgetItem(f"{post_stats['nodata_pixels']-pre_stats['nodata_pixels']:+d}"))
        row += 1
        
        if 'bands' in pre_stats:
            for i in range(num_bands):  # Ensure all bands are processed
                pre_band = pre_stats['bands'][i]
                post_band = post_stats['bands'][i]
                
                for metric in ['min', 'max', 'mean', 'std']:
                    table.setItem(row, 0, QTableWidgetItem(f"Band {i+1} - {metric.title()}"))
                    table.setItem(row, 1, QTableWidgetItem(f"{pre_band[metric]:.4f}"))
                    table.setItem(row, 2, QTableWidgetItem(f"{post_band[metric]:.4f}"))
                    table.setItem(row, 3, QTableWidgetItem(f"{post_band[metric]-pre_band[metric]:+.4f}"))
                    row += 1
        else:
            for metric in ['min', 'max', 'mean', 'std']:
                table.setItem(row, 0, QTableWidgetItem(metric.title()))
                table.setItem(row, 1, QTableWidgetItem(f"{pre_stats[metric]:.4f}"))
                table.setItem(row, 2, QTableWidgetItem(f"{post_stats[metric]:.4f}"))
                table.setItem(row, 3, QTableWidgetItem(f"{post_stats[metric]-pre_stats[metric]:+.4f}"))
                row += 1
        
        table.resizeColumnsToContents()
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        stats_layout.addWidget(table)
        stats_tab.setLayout(stats_layout)
        
        # Histogram Tab
        hist_tab = QWidget()
        hist_layout = QVBoxLayout()
        
        hist_fig = Figure(figsize=(8, 4))
        hist_canvas = FigureCanvas(hist_fig)
        self._plot_histogram_comparison(hist_fig, pre_stats, post_stats)
        
        # Histogram controls
        controls = QHBoxLayout()
        log_check = QCheckBox("Log Scale")
        bin_spin = QSpinBox()
        bin_spin.setRange(10, 500)
        bin_spin.setValue(50)
        
        def update_hist():
            hist_fig.clear()
            self._plot_histogram_comparison(
                hist_fig, pre_stats, post_stats,
                log_scale=log_check.isChecked(),
                bins=bin_spin.value()
            )
            hist_canvas.draw()
        
        log_check.stateChanged.connect(update_hist)
        bin_spin.valueChanged.connect(update_hist)
        
        controls.addWidget(QLabel("Bins:"))
        controls.addWidget(bin_spin)
        controls.addWidget(log_check)
        controls.addStretch()
        
        hist_layout.addLayout(controls)
        hist_layout.addWidget(hist_canvas)
        hist_tab.setLayout(hist_layout)
        
        tab_widget.addTab(stats_tab, "Statistics")
        tab_widget.addTab(hist_tab, "Histogram")
        
        # Buttons
        button_box = QDialogButtonBox()
        ok_button = button_box.addButton(QDialogButtonBox.Ok)
        ok_button.clicked.connect(dialog.accept)
        
        export_csv = QPushButton("Export to CSV")
        export_plot = QPushButton("Save Plot")
        export_report = QPushButton("Save Full Report")
        
        def save_csv():
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", f"{filter_name}_stats.csv", "CSV (*.csv)"
            )
            if path:
                try:
                    with open(path, 'w') as f:
                        f.write("Metric,Before,After,Change\n")
                        for row in range(table.rowCount()):
                            items = [table.item(row, col).text() for col in range(4)]
                            f.write(','.join(items) + '\n')
                    QMessageBox.information(self, "Success", "CSV exported successfully!")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
        
        def save_plot():
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Plot", filter_name, 
                "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)"
            )
            if path:
                try:
                    hist_fig.savefig(path, dpi=300, bbox_inches='tight')
                    QMessageBox.information(self, "Success", "Plot saved successfully!")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")
        
        def save_full_report():
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Full Report", f"{filter_name}_report.pdf", "PDF (*.pdf)"
            )
            if path:
                try:
                    from matplotlib.backends.backend_pdf import PdfPages
                    from matplotlib.pyplot import Figure
                    
                    with PdfPages(path) as pdf:
                        # Create a title page
                        fig_title = Figure(figsize=(8, 11))
                        ax_title = fig_title.add_subplot(111)
                        ax_title.axis('off')
                        ax_title.text(0.5, 0.5, 
                                    f"Filter Results Report\n\n{filter_name.replace('_', ' ').title()}\n\n"
                                    f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                    ha='center', va='center', fontsize=14)
                        pdf.savefig(fig_title, bbox_inches='tight')
                        
                        # Add statistics table page
                        fig_table = Figure(figsize=(11, 8))
                        ax_table = fig_table.add_subplot(111)
                        ax_table.axis('off')
                        
                        # Prepare table data
                        table_data = []
                        headers = ["Metric", "Before", "After", "Change"]
                        table_data.append(headers)
                        
                        for row in range(table.rowCount()):
                            table_data.append([table.item(row, col).text() for col in range(4)])
                        
                        # Create table
                        table_plot = ax_table.table(
                            cellText=table_data,
                            loc='center',
                            cellLoc='center',
                            colWidths=[0.3, 0.2, 0.2, 0.3]
                        )
                        table_plot.auto_set_font_size(False)
                        table_plot.set_fontsize(8)
                        table_plot.scale(1, 1.5)
                        
                        ax_table.set_title("Statistics Comparison", pad=20)
                        pdf.savefig(fig_table, bbox_inches='tight')
                        
                        # Add histogram page
                        fig_hist = Figure(figsize=(8, 6))
                        self._plot_histogram_comparison(
                            fig_hist, pre_stats, post_stats,
                            log_scale=log_check.isChecked(),
                            bins=bin_spin.value()
                        )
                        fig_hist.suptitle("Pixel Value Distribution Comparison", y=0.95)
                        pdf.savefig(fig_hist, bbox_inches='tight')
                    
                    QMessageBox.information(self, "Success", "Full report saved successfully!")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")
        
        export_csv.clicked.connect(save_csv)
        export_plot.clicked.connect(save_plot)
        export_report.clicked.connect(save_full_report)
        
        button_box.addButton(export_csv, QDialogButtonBox.ActionRole)
        button_box.addButton(export_plot, QDialogButtonBox.ActionRole)
        button_box.addButton(export_report, QDialogButtonBox.ActionRole)
        
        main_layout.addWidget(tab_widget)
        main_layout.addWidget(button_box)
        dialog.setLayout(main_layout)
        dialog.exec_()

    def _plot_histogram_comparison(self, fig, pre_stats, post_stats, log_scale=False, bins=50):
        """Plot histogram comparison for all bands"""
        ax = fig.add_subplot(111)
        
        if 'bands' in pre_stats:
            colors = _get_cmap('tab10', len(pre_stats['bands']))
            for i in range(len(pre_stats['bands'])):
                pre_band = pre_stats['bands'][i]
                post_band = post_stats['bands'][i]
                
                ax.hist(
                    pre_band['values'], bins=bins, alpha=0.5,
                    label=f"Band {i+1} Before", color=colors(i), density=True
                )
                ax.hist(
                    post_band['values'], bins=bins, alpha=0.5,
                    label=f"Band {i+1} After", color=colors(i),
                    histtype='step', linewidth=2, density=True
                )
        else:
            ax.hist(
                pre_stats['values'], bins=bins, alpha=0.5,
                label="Before", color='blue', density=True
            )
            ax.hist(
                post_stats['values'], bins=bins, alpha=0.5,
                label="After", color='orange',
                histtype='step', linewidth=2, density=True
            )
        
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Density")
        ax.legend()
        
        if log_scale:
            ax.set_yscale('log')
        
        fig.tight_layout()

    def _prompt_save_filtered_image(self):
        """Prompt user to save filtered image"""
        reply = QMessageBox.question(
            self, 'Save Result', 
            'Would you like to save the filtered image?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.save_image()

    # Filter application methods
    def apply_nodata_filter(self):
        self._apply_filter(self.image_processor.filter_nodata)

    def apply_outlier_filter(self):
        threshold, ok = QInputDialog.getDouble(
            self, 'Outlier Threshold', 
            'Enter standard deviation threshold:', 
            3, 1, 10, 1
        )
        if ok:
            self._apply_filter(self.image_processor.remove_outliers, threshold)

    def apply_median_filter(self):
        size, ok = QInputDialog.getInt(
            self, 'Window Size', 
            'Enter filter window size:', 
            3, 1, 15, 2
        )
        if ok:
            self._apply_filter(self.image_processor.apply_median_filter, size)

    def apply_gaussian_filter(self):
        sigma, ok = QInputDialog.getDouble(
            self, 'Sigma Value', 
            'Enter Gaussian sigma:', 
            1.0, 0.1, 5.0, 1
        )
        if ok:
            self._apply_filter(self.image_processor.apply_gaussian_filter, sigma)

    def apply_lee_filter(self):
        window_size, ok1 = QInputDialog.getInt(
            self, 'Lee Filter', 
            'Enter window size:', 
            5, 3, 15, 2
        )
        if ok1:
            k, ok2 = QInputDialog.getDouble(
                self, 'Noise Coefficient', 
                'Enter noise coefficient k:', 
                1.0, 0.1, 5.0, 1
            )
            if ok2:
                self._apply_filter(self.image_processor.apply_lee_filter, window_size, k)

    def apply_normalization(self):
        self._apply_filter(self.image_processor.normalize_image)

    def convert_to_8bit(self):
        self._apply_filter(self.image_processor.to_8bit)
######
    def save_image(self):
        """Save the current image with proper geospatial metadata"""
        if not hasattr(self, 'image'):
            QMessageBox.warning(self, "Warning", "No image to save")
            return

        # Get suggested filename
        default_name = "filtered_image.tif"
        if hasattr(self, 'current_file') and self.current_file:
            default_name = f"filtered_{os.path.basename(self.current_file)}"

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Filtered Image",
            default_name,
            "GeoTIFF (*.tif *.tiff);;All Files (*)"
        )

        if file_path:
            try:
                # Ensure we have the required profile data
                if not hasattr(self, 'profile'):
                    # Create minimal profile if none exists
                    self.profile = {
                        'driver': 'GTiff',
                        'dtype': self.image.dtype,
                        'count': self.image.shape[0] if self.image.ndim == 3 else 1,
                        'width': self.image.shape[-1],
                        'height': self.image.shape[-2]
                    }
                
                # Save with both image data and profile
                self.image_processor.save_image(
                    image=self.image,
                    output_path=file_path,
                    profile=self.profile  # Pass the required profile
                )
                QMessageBox.information(self, "Success", f"Image saved to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")


    def toggle_edit_buttons(self, enable):
        """Enable/disable Save/Stop buttons (called by polygon_processor)."""
        self.actionSave_Edits.setEnabled(enable)
        self.actionStop_Editing.setEnabled(enable)

    


############################################# Wet Area 

    # # Add to SatelliteImageApp class methods
######################################## New SAR ############################
    
    # ======================================================================
    # SAR water workflow - real Sentinel-1 preprocessing through ESA SNAP (gpt)
    # ======================================================================
    def _init_sar_workflow_actions(self):
        """Step-by-step SAR water workflow shown under the 'SAR Water Classification' menu."""
        self.workflow_steps = ['load', 'snap', 'analyze', 'classify']
        self.workflow_actions = [
            self.sar_menu.addAction('1. Select Sentinel-1 GRD Product'),
            self.sar_menu.addAction('2. Run SNAP Preprocessing (orbit, σ₀, speckle, terrain, dB)'),
            self.sar_menu.addAction('3. Choose Water Threshold (histogram)'),
            self.sar_menu.addAction('4. Classify Water'),
        ]
        self.workflow_actions[0].triggered.connect(self.start_sar_workflow)
        self.workflow_actions[1].triggered.connect(self.run_snap_preprocessing)
        self.workflow_actions[2].triggered.connect(self.analyze_histogram)
        self.workflow_actions[3].triggered.connect(self.classify_water)
        self.sar_menu.addSeparator()
        load_action = self.sar_menu.addAction('Load processed σ₀ dB GeoTIFF (skip SNAP)...')
        load_action.triggered.connect(self.load_processed_db)
        flood_action = self.sar_menu.addAction('5. Flood Mapping (before / after)...')
        flood_action.triggered.connect(self.flood_mapping)
        self.sar_menu.addSeparator()
        settings_action = self.sar_menu.addAction('SNAP Settings (gpt location)...')
        settings_action.triggered.connect(self.snap_settings)
        self.sar_product = None
        self.sar_footprint = None
        self.snap_worker = None
        self._reset_workflow()

    def _snap_dialog(self, info):
        """SNAP parameters + flexible study-area selection. Returns (SnapParams, out_dir) or None."""
        from matplotlib.widgets import RectangleSelector
        path = self.sar_product.path
        dlg = QDialog(self)
        dlg.setWindowTitle("SNAP Sentinel-1 preprocessing")
        dlg.setMinimumSize(1000, 640)
        outer = QHBoxLayout(dlg)
        left = QWidget(); form = QFormLayout(left)
        outer.addWidget(left, 5)

        pol = QComboBox(); pol.addItems(info.polarisations)
        orbit = QComboBox(); orbit.addItems(sar_snap.ORBIT_TYPES)
        thermal = QCheckBox("Thermal noise removal"); thermal.setChecked(True)
        speckle = QComboBox(); speckle.addItems(sar_snap.SPECKLE_FILTERS); speckle.setCurrentText("Lee")
        win = QSpinBox(); win.setRange(3, 15); win.setSingleStep(2); win.setValue(5)
        dem = QComboBox(); dem.addItems(sar_snap.DEM_NAMES)
        dem_file = QLineEdit(); dem_file.setEnabled(False)
        dem.currentTextChanged.connect(lambda t: dem_file.setEnabled(t == "External DEM"))
        spacing = QDoubleSpinBox(); spacing.setRange(1.0, 1000.0); spacing.setValue(10.0); spacing.setSuffix(" m")
        proj = QComboBox(); proj.addItems(["WGS84(DD)", "AUTO:42001"])      # EPSG:4326 made SNAP 10 fail (zero-size output)
        out_edit = QLineEdit(self._snap_qsettings().value("snap/out_dir", os.path.join(tempfile.gettempdir(), "SpaceGeoVision_SAR"), type=str))
        out_btn = QPushButton("Browse...")
        out_row = QHBoxLayout(); out_row.addWidget(out_edit); out_row.addWidget(out_btn)
        out_btn.clicked.connect(lambda: out_edit.setText(
            QFileDialog.getExistingDirectory(dlg, "Output folder", out_edit.text()) or out_edit.text()))

        # ---------------- study area -----------------
        fp = self.sar_footprint
        fx0, fy0, fx1, fy1 = sar_snap.footprint_bbox(fp) if fp else (-180.0, -90.0, 180.0, 90.0)

        def dspin(value, lo, hi, dec=4):
            w = QDoubleSpinBox(); w.setDecimals(dec); w.setRange(lo, hi); w.setValue(value); return w
        mode_c = QComboBox()
        mode_c.addItems(["Whole scene (automatic bounds)", "Bounding box (lon / lat)",
                         "Centre point + radius (km)", "Custom polygon (WKT)"])
        stack = QStackedWidget()
        p0 = QLabel(f"Bounds detected automatically from the raw product:\nlongitude {fx0:.3f} .. {fx1:.3f}\nlatitude {fy0:.3f} .. {fy1:.3f}\n"
                    "Processing the whole scene is slow and memory-hungry.")
        p0.setWordWrap(True)
        stack.addWidget(p0)
        pb = QWidget(); fb = QFormLayout(pb)
        b_x0, b_y0 = dspin(fx0, -180, 180), dspin(fy0, -90, 90)
        b_x1, b_y1 = dspin(fx1, -180, 180), dspin(fy1, -90, 90)
        fb.addRow("Longitude min:", b_x0); fb.addRow("Latitude min:", b_y0)
        fb.addRow("Longitude max:", b_x1); fb.addRow("Latitude max:", b_y1)
        fb.addRow(QLabel("Tip: drag a rectangle on the map to the right."))
        stack.addWidget(pb)
        pc = QWidget(); fc = QFormLayout(pc)
        c_lon, c_lat = dspin((fx0 + fx1) / 2, -180, 180), dspin((fy0 + fy1) / 2, -90, 90)
        c_rad = QDoubleSpinBox(); c_rad.setRange(1, 150); c_rad.setValue(20); c_rad.setSuffix(" km")
        fc.addRow("Centre longitude:", c_lon); fc.addRow("Centre latitude:", c_lat); fc.addRow("Radius (half-width):", c_rad)
        fc.addRow(QLabel("The scene centre is only a starting value and may be open sea:\nenter YOUR study area."))
        stack.addWidget(pc)
        pp = QWidget(); fpg = QVBoxLayout(pp)
        poly_edit = QTextEdit(); poly_edit.setPlaceholderText("POLYGON((lon lat, lon lat, lon lat, lon lat))   WGS84, longitude first")
        fpg.addWidget(poly_edit)
        stack.addWidget(pp)
        mode_c.currentIndexChanged.connect(stack.setCurrentIndex)
        size_lbl = QLabel(); size_lbl.setWordWrap(True)
        msg_lbl = QLabel(); msg_lbl.setWordWrap(True)

        form.addRow("Polarisation:", pol); form.addRow("Orbit file:", orbit); form.addRow("", thermal)
        form.addRow("Speckle filter:", speckle); form.addRow("Filter window (odd):", win)
        form.addRow("DEM:", dem); form.addRow("External DEM file:", dem_file)
        form.addRow("Terrain-corrected pixel size:", spacing); form.addRow("Map projection:", proj)
        form.addRow(QLabel("<b>Study area</b>")); form.addRow("Select by:", mode_c); form.addRow(stack)
        form.addRow(size_lbl); form.addRow(msg_lbl); form.addRow("Output folder:", out_row)
        form.addRow(QLabel("Use the SAME area and settings for the before and the after scene.\n"
                           "SNAP downloads orbit files and DEM tiles automatically: internet is required."))
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        form.addRow(btns)

        # ---------------- map preview with an interactive rectangle -----------------
        fig = Figure(figsize=(4.6, 4.6)); canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        outer.addWidget(canvas, 4)
        if fp:
            xs = [p[0] for p in fp] + [fp[0][0]]; ys = [p[1] for p in fp] + [fp[0][1]]
            ax.plot(xs, ys, color='#2c3e50', lw=1.5, label='scene footprint')
            ax.fill(xs, ys, color='#bdc3c7', alpha=0.4)
        mx, my = (fx1 - fx0) * 0.05, (fy1 - fy0) * 0.05
        ax.set_xlim(fx0 - mx, fx1 + mx); ax.set_ylim(fy0 - my, fy1 + my)
        ax.set_aspect(1.0 / max(0.1, float(np.cos(np.radians((fy0 + fy1) / 2)))))
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.grid(True, alpha=0.3)
        sub_line, = ax.plot([], [], color='red', lw=2, label='study area')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()

        def current_spec():
            i = mode_c.currentIndex()
            if i == 0:
                return sar_snap.SubsetSpec("full")
            if i == 1:
                return sar_snap.SubsetSpec("bbox", bbox=(b_x0.value(), b_y0.value(), b_x1.value(), b_y1.value()))
            if i == 2:
                return sar_snap.SubsetSpec("center", center=(c_lon.value(), c_lat.value()), radius_km=c_rad.value())
            return sar_snap.SubsetSpec("polygon", wkt=poly_edit.toPlainText())

        def refresh(*_):
            try:
                wkt = sar_snap.resolve_subset(current_spec(), path)
                msg_lbl.setText("")
            except sar_snap.SnapError as exc:
                msg_lbl.setText(f'<span style="color:#c0392b">{exc}</span>')
                size_lbl.setText(""); sub_line.set_data([], []); canvas.draw_idle()
                return
            shown = wkt if wkt else sar_snap.bbox_wkt(fx0, fy0, fx1, fy1)
            nums = re.findall(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", shown)
            sub_line.set_data([float(a) for a, _ in nums], [float(b) for _, b in nums])
            canvas.draw_idle()
            px = sar_snap.estimate_output_pixels(wkt, path, spacing.value())
            colour = '#c0392b' if px == 0 or px > 200e6 else ('#d35400' if px > 60e6 else '#27ae60')
            size_lbl.setText(f'<span style="color:{colour}">Output size: {sar_snap.size_advice(px)}</span>')

        for w in (b_x0, b_y0, b_x1, b_y1, c_lon, c_lat, c_rad, spacing):
            w.valueChanged.connect(refresh)
        mode_c.currentIndexChanged.connect(refresh); poly_edit.textChanged.connect(refresh)

        def on_rect(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return
            x0, x1 = sorted((eclick.xdata, erelease.xdata)); y0, y1 = sorted((eclick.ydata, erelease.ydata))
            if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
                return
            for w, v in ((b_x0, x0), (b_y0, y0), (b_x1, x1), (b_y1, y1)):
                w.blockSignals(True); w.setValue(float(v)); w.blockSignals(False)
            mode_c.setCurrentIndex(1)
            refresh()
        dlg._rect_selector = RectangleSelector(ax, on_rect, useblit=True, button=[1], interactive=False)
        refresh()

        result = {}

        def on_ok():
            try:
                spec = current_spec()
                wkt = sar_snap.resolve_subset(spec, path)
                if wkt:
                    frac = sar_snap.check_subset_in_footprint(wkt, path)        # raises if there is no overlap
                    if frac < 0.999 and QMessageBox.question(
                            dlg, "Partly outside the scene",
                            f"Only {frac * 100:.0f}% of the sample points of this area lie inside the scene footprint.\n"
                            "Part of it will have no data. Continue?") != QMessageBox.Yes:
                        return
                px = sar_snap.estimate_output_pixels(wkt, path, spacing.value())
                if px > 200e6 and QMessageBox.question(
                        dlg, "Very large area",
                        f"{sar_snap.size_advice(px)}.\nThis can take hours or exhaust the memory. Continue anyway?") != QMessageBox.Yes:
                    return
                params = sar_snap.SnapParams(
                    polarisation=pol.currentText(), orbit_type=orbit.currentText(),
                    remove_thermal_noise=thermal.isChecked(), speckle_filter=speckle.currentText(),
                    speckle_window=win.value(), dem_name=dem.currentText(),
                    external_dem=dem_file.text().strip() or None, pixel_spacing_m=spacing.value(),
                    map_projection=proj.currentText(), subset_wkt=wkt, subset_mode="operator",
                    nodata_at_sea=False, output_db=True)
                params.validate(info)
                sar_snap.check_dem_coverage(params, path)
                result['value'] = (params, out_edit.text().strip())
            except sar_snap.SnapError as exc:
                QMessageBox.warning(dlg, "Please check the settings", str(exc))
                return
            self._snap_qsettings().setValue("snap/out_dir", out_edit.text().strip())
            dlg.accept()
        btns.accepted.connect(on_ok); btns.rejected.connect(dlg.reject)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return result.get('value')

    def _set_db_layer(self, data, profile, file_name, polarisation="", date="", history=None):
        """Make a sigma0-dB array the current SAR layer (used after SNAP and when loading a processed file)."""
        valid = data[np.isfinite(data)]
        crs = profile.get('crs')
        pix = abs(profile['transform'].a)
        unit = '°' if (crs is not None and crs.is_geographic) else ' m'
        self.sar_steps = {
            'dB': {
                'data': data, 'profile': profile, 'stats': self._calculate_stats(data),
                'vis_params': {'cmap': 'gray',
                               'vmin': float(np.nanpercentile(valid, 2)) if valid.size else -25.0,
                               'vmax': float(np.nanpercentile(valid, 98)) if valid.size else 0.0},
                'metadata': {'file_name': file_name, 'acquisition_date': date, 'polarization': polarisation,
                             'dimensions': f"{data.shape[1]}×{data.shape[0]}", 'resolution': f"{pix:.6g}{unit}",
                             'data_type': 'float32 (σ₀, dB)', 'crs': crs.to_string() if crs is not None else 'Unknown',
                             'processing_history': history or []},
            }
        }
        self._display_sar_layer('dB')

    def load_processed_db(self):
        """Load a σ₀ dB GeoTIFF produced earlier by SNAP (no need to run SNAP again)."""
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open processed σ₀ dB GeoTIFF", "", "GeoTIFF (*.tif *.tiff)")
            if not path:
                return
            data, profile = sar_snap.read_geotiff(path)
            warnings = sar_snap.sanity_check_db(data)
            self._set_db_layer(data, profile, os.path.basename(path), history=["loaded from file"])
            self.layer_list.addItem(QListWidgetItem(f"📈 σ₀ dB ({os.path.basename(path)})"))
            self._reset_workflow()
            self._enable_next_step('snap')
            if warnings:
                QMessageBox.warning(self, "Check the data", "\n".join(warnings))
        except Exception as exc:
            self.show_error(f"Could not load the file: {exc}")

    def analyze_histogram(self):
        """Step 3: choose ONE water threshold (dB) with a histogram, a preview and a sensitivity table."""
        try:
            if 'dB' not in self.sar_steps:
                raise ValueError("Run SNAP preprocessing (step 2) or load a processed dB file first.")
            data = self.sar_steps['dB']['data']
            valid_mask = np.isfinite(data) & (data != 0.0)
            vals = data[valid_mask]
            if vals.size == 0:
                raise ValueError("The layer has no valid pixels.")

            dlg = QDialog(self)
            dlg.setWindowTitle("Water threshold (σ₀ dB)")
            dlg.setMinimumSize(900, 650)
            lay = QVBoxLayout(dlg)
            fig = Figure(figsize=(8, 4)); canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, dlg)); lay.addWidget(canvas)
            form = QFormLayout()
            method = QComboBox()
            method.addItems(["Tile-based bimodal (recommended)", "Fixed threshold", "Global Otsu (unreliable if water is a small part)"])
            thr_spin = QDoubleSpinBox(); thr_spin.setRange(-50.0, 10.0); thr_spin.setDecimals(2); thr_spin.setValue(-15.0)
            minpx = QSpinBox(); minpx.setRange(1, 100000); minpx.setValue(100)
            form.addRow("Method:", method); form.addRow("Threshold (dB):", thr_spin); form.addRow("Remove water objects smaller than (pixels):", minpx)
            lay.addLayout(form)
            info_box = QTextEdit(); info_box.setReadOnly(True); info_box.setFixedHeight(130)
            lay.addWidget(info_box)
            sample = vals if vals.size <= 3_000_000 else np.random.default_rng(0).choice(vals, 3_000_000, replace=False)

            def redraw():
                thr = thr_spin.value()
                fig.clf(); ax = fig.add_subplot(111)
                ax.hist(sample, bins=200, color='#7f8c8d')
                ax.axvline(thr, color='red', linestyle='--', label=f'threshold {thr:.2f} dB')
                ax.set_xlabel('σ₀ (dB)'); ax.set_ylabel('pixels'); ax.legend()
                canvas.draw()
                pct = lambda t: 100.0 * float(np.mean(sample < t))
                txt = [f"Pixels below {thr:.2f} dB: {pct(thr):.1f}% (before removing small objects)",
                       "Sensitivity (% of valid pixels below): " + "   ".join(f"{t:.0f} dB: {pct(t):.1f}%" for t in (-12, -14, -16, -18, -20))]
                if thr > -10 or pct(thr) > 40:
                    txt.append("WARNING: this threshold is above typical open-water backscatter (about -15 to -25 dB in VV) "
                               "or labels a very large share as water.")
                info_box.setPlainText("\n".join(txt))

            def compute():
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    if method.currentIndex() == 0:
                        thr, ti = water_mapping.tile_bimodal_threshold(data)
                        msg = f"Bimodal tiles used: {ti['tiles_used']} of {ti['tiles_tested']}"
                    elif method.currentIndex() == 2:
                        thr = water_mapping.otsu_water_threshold(data); msg = "Global Otsu"
                    else:
                        return
                    thr_spin.setValue(float(np.clip(thr, -50.0, 10.0)))
                    self.show_status(f"Threshold {thr:.2f} dB ({msg})", 8000)
                except ValueError as exc:
                    QMessageBox.warning(dlg, "Cannot compute", str(exc))
                finally:
                    QApplication.restoreOverrideCursor()
                redraw()

            btns = QDialogButtonBox()
            comp_btn = btns.addButton("Compute", QDialogButtonBox.ActionRole)
            ok_btn = btns.addButton("Use this threshold", QDialogButtonBox.AcceptRole)
            btns.addButton(QDialogButtonBox.Cancel)
            comp_btn.clicked.connect(compute); thr_spin.valueChanged.connect(lambda _: redraw())
            btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
            lay.addWidget(btns)
            redraw()
            if dlg.exec_() != QDialog.Accepted:
                return
            thr = thr_spin.value()
            self.sar_steps['dB']['metadata']['thresholds'] = {
                'lower': float(np.min(vals)), 'upper': float(thr), 'threshold_db': float(thr),
                'method': method.currentText(), 'min_pixels': int(minpx.value())}
            self._enable_next_step('analyze')
            self.show_status(f"Water threshold set to {thr:.2f} dB. Next: 4. Classify Water", 8000)
        except Exception as exc:
            self.show_error(f"Threshold selection failed: {exc}")

    def classify_water(self):
        """Step 4: water = backscatter below the chosen threshold (dB), small objects removed."""
        try:
            md = self.sar_steps.get('dB', {}).get('metadata', {})
            if 'thresholds' not in md:
                raise ValueError("Choose the water threshold first (step 3).")
            th = md['thresholds']
            data = self.sar_steps['dB']['data']
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                mask, info = water_mapping.map_water(data, "fixed", threshold=th['threshold_db'], min_pixels=th.get('min_pixels', 100))
            finally:
                QApplication.restoreOverrideCursor()
            valid = np.isfinite(data) & (data != 0.0)
            profile = self.sar_steps['dB']['profile']
            geographic = profile['crs'] is not None and profile['crs'].is_geographic
            lat = (profile['transform'].f + profile['transform'].f + profile['transform'].e * data.shape[0]) / 2
            area = flood.pixel_area_m2(profile['transform'], geographic, lat)
            ha = info['water_pixels'] * area / 1e4
            self.sar_steps['classification'] = {'mask': mask, 'valid': valid, 'thresholds': th, 'info': info}

            dlg = QDialog(self)
            dlg.setWindowTitle("Water classification")
            dlg.setMinimumSize(1100, 720)
            lay = QVBoxLayout(dlg)
            fig = Figure(figsize=(12, 6)); canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, dlg)); lay.addWidget(canvas)
            ax1 = fig.add_subplot(121)
            im, _s1 = georef.draw_raster(ax1, np.ma.masked_invalid(np.where(valid, data, np.nan)), profile, cmap='gray',
                                         vmin=info.get('db_p2', -25), vmax=info.get('db_p98', 0))
            ax1.set_title("σ₀ (dB)"); fig.colorbar(im, ax=ax1, fraction=0.04)
            ax2 = fig.add_subplot(122)
            show = np.where(valid, mask.astype(float), np.nan)
            georef.draw_raster(ax2, np.ma.masked_invalid(show), profile, cmap='Blues', vmin=0, vmax=1.4)
            ax2.set_title(f"Water (threshold {th['threshold_db']:.2f} dB)")
            canvas.draw()
            box = QTextEdit(); box.setReadOnly(True); box.setFixedHeight(120)
            lines = [f"Threshold: {th['threshold_db']:.2f} dB   ({th['method']})",
                     f"Water pixels: {info['water_pixels']:,} of {info['valid_pixels']:,} valid ({info['water_percent']:.2f}%)",
                     f"Approximate water area: {ha:,.1f} ha  ({ha / 100:,.2f} km²)  [pixel ~ {area:.1f} m²]",
                     "Not validated against a reference mask."]
            if info.get('warning'):
                lines.append("WARNING: " + info['warning'])
            box.setPlainText("\n".join(lines)); lay.addWidget(box)
            btns = QDialogButtonBox()
            save_btn = btns.addButton("Save water mask (GeoTIFF)...", QDialogButtonBox.ActionRole)
            btns.addButton("Close", QDialogButtonBox.RejectRole)
            save_btn.clicked.connect(lambda: self._save_water_mask(mask, valid))
            btns.rejected.connect(dlg.reject)
            lay.addWidget(btns)
            dlg.exec_()
        except Exception as exc:
            self.show_error(f"Classification failed: {exc}")

    def _save_water_mask(self, mask, valid):
        try:
            import rasterio
            path, _ = QFileDialog.getSaveFileName(self, "Save water mask", "", "GeoTIFF (*.tif)")
            if not path:
                return
            prof = self.sar_steps['dB']['profile'].copy()
            prof.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="lzw")
            with rasterio.open(path, "w", **prof) as dst:
                dst.write(np.where(valid, mask, 255).astype("uint8"), 1)
            self.show_status(f"Saved {path}  (0 = land, 1 = water, 255 = no data)", 8000)
            st = self._snap_qsettings()
            st.setValue("val/raster", path); st.setValue("val/groups", "water:1;land:0")
        except Exception as exc:
            self.show_error(f"Could not save the mask: {exc}")

    def flood_mapping(self):
        """Step 5: flood extent from a BEFORE and an AFTER σ₀ dB GeoTIFF of the same track."""
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle("Flood mapping (before / after)")
            form = QFormLayout(dlg)
            settings = self._snap_qsettings()
            before_e, after_e = QLineEdit(settings.value("flood/before", "", type=str)), QLineEdit(settings.value("flood/after", "", type=str))
            out_e = QLineEdit(settings.value("flood/out", os.path.join(tempfile.gettempdir(), "SpaceGeoVision_flood"), type=str))

            def row(edit, title, folder=False):
                btn = QPushButton("Browse..."); r = QHBoxLayout(); r.addWidget(edit); r.addWidget(btn)
                def pick():
                    p = QFileDialog.getExistingDirectory(dlg, title, edit.text()) if folder else \
                        QFileDialog.getOpenFileName(dlg, title, edit.text(), "GeoTIFF (*.tif *.tiff)")[0]
                    if p:
                        edit.setText(p)
                btn.clicked.connect(pick)
                return r
            auto = QCheckBox("Common threshold from both images (recommended)"); auto.setChecked(True)
            thr = QDoubleSpinBox(); thr.setRange(-50, 10); thr.setValue(-14.0); thr.setSuffix(" dB"); thr.setEnabled(False)
            auto.toggled.connect(lambda on: thr.setEnabled(not on))
            drop = QDoubleSpinBox(); drop.setRange(0, 15); drop.setValue(3.0); drop.setSuffix(" dB")
            minpx = QSpinBox(); minpx.setRange(1, 100000); minpx.setValue(100)
            form.addRow("BEFORE image (σ₀ dB GeoTIFF):", row(before_e, "Before image"))
            form.addRow("AFTER image (σ₀ dB GeoTIFF):", row(after_e, "After image"))
            form.addRow("Output folder:", row(out_e, "Output folder", True))
            form.addRow("", auto); form.addRow("Fixed threshold:", thr)
            form.addRow("Required backscatter drop (0 = off):", drop); form.addRow("Min. object size (pixels):", minpx)
            form.addRow(QLabel("Both images must come from the SAME relative orbit and be processed with the same settings."))
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject); form.addRow(btns)
            if dlg.exec_() != QDialog.Accepted:
                return
            for label, e in (("BEFORE", before_e), ("AFTER", after_e)):
                if not os.path.isfile(e.text().strip()):
                    raise ValueError(f"Please choose the {label} GeoTIFF.")
            settings.setValue("flood/before", before_e.text().strip()); settings.setValue("flood/after", after_e.text().strip())
            settings.setValue("flood/out", out_e.text().strip())
            args = (before_e.text().strip(), after_e.text().strip(), out_e.text().strip(),
                    None if auto.isChecked() else thr.value(), drop.value(), minpx.value())
            self._flood_progress = QProgressDialog("Starting...", None, 0, 100, self)
            self._flood_progress.setWindowTitle("Flood mapping")
            self._flood_progress.setWindowModality(Qt.WindowModal)
            self._flood_progress.setMinimumDuration(0)
            worker = FuncWorker(lambda progress: flood.run_flood_db(*args, progress=progress))
            worker.progress.connect(lambda v, m: (self._flood_progress.setValue(v), self._flood_progress.setLabelText(m)))
            worker.done.connect(self._on_flood_done)
            worker.failed.connect(lambda m: (self._flood_progress.close(), self.show_error(f"Flood mapping failed:\n{m}")))
            self._flood_worker = worker
            worker.start()
        except Exception as exc:
            self.show_error(f"Cannot start flood mapping: {exc}")

    def _on_flood_done(self, res):
        self._flood_progress.close()
        st = self._snap_qsettings()
        st.setValue("val/raster", res["paths"]["flood_classes"]); st.setValue("val/groups", "flood:2;not_flood:0,1,3,4")
        from matplotlib.colors import ListedColormap
        cls = res["classes"]
        dlg = QDialog(self)
        dlg.setWindowTitle("Flood extent")
        dlg.setMinimumSize(1100, 760)
        lay = QVBoxLayout(dlg)
        fig = Figure(figsize=(11, 6)); canvas = FigureCanvas(fig)
        lay.addWidget(NavigationToolbar(canvas, dlg)); lay.addWidget(canvas)
        ax = fig.add_subplot(111)
        cmap = ListedColormap([flood.CLASS_COLOURS[k] for k in (0, 1, 2, 3, 4)])
        show = np.where(cls == 255, np.nan, cls.astype(float))
        georef.draw_raster(ax, np.ma.masked_invalid(show), res.get("profile"), cmap=cmap, vmin=-0.5, vmax=4.5,
                           interpolation='nearest')
        ax.set_title("Flood extent: grey land | blue permanent water | red flooded | green receded | orange uncertain")
        canvas.draw()
        box = QTextEdit(); box.setReadOnly(True); box.setFontFamily("Courier New"); box.setFixedHeight(230)
        box.setPlainText(res["report"] + "\n\nFiles:\n  " + "\n  ".join(res["paths"].values()))
        lay.addWidget(box)
        ok = QDialogButtonBox(QDialogButtonBox.Ok); ok.accepted.connect(dlg.accept); lay.addWidget(ok)
        dlg.exec_()

    # ======================================================================
    # Validation menu: independent accuracy assessment of any classified map
    # ======================================================================
    def _browse_row(self, dlg, edit, title, filt="", folder=False):
        """Line edit + Browse button row used by the validation dialogs."""
        btn = QPushButton("Browse...")
        row = QHBoxLayout(); row.addWidget(edit); row.addWidget(btn)

        def pick():
            if folder:
                p = QFileDialog.getExistingDirectory(dlg, title, edit.text())
            else:
                p, _ = QFileDialog.getOpenFileName(dlg, title, edit.text(), filt)
            if p:
                edit.setText(p)
        btn.clicked.connect(pick)
        return row

    def validation_sample_points(self):
        """Step 1 of validation: random points in every class of a classified map, to be labelled BLIND."""
        try:
            st = self._snap_qsettings()
            dlg = QDialog(self)
            dlg.setWindowTitle("Validation 1: create sampling points")
            dlg.setMinimumWidth(640)
            form = QFormLayout(dlg)
            raster_e = QLineEdit(st.value("val/raster", "", type=str))
            out_e = QLineEdit(st.value("val/out", os.path.join(tempfile.gettempdir(), "SpaceGeoVision_validation"), type=str))
            counts_e = QLineEdit(); counts_e.setPlaceholderText('points per class, e.g. 0=100,1=50,2=100')
            seed = QSpinBox(); seed.setRange(0, 10_000_000); seed.setValue(42)
            mind = QSpinBox(); mind.setRange(1, 500); mind.setValue(5); mind.setSuffix(" px")
            info = QLabel(""); info.setWordWrap(True)
            detect = QPushButton("Detect classes in the map")
            groups_e = QLineEdit(st.value("val/groups", "flood:2;not_flood:0,1,3,4", type=str))
            groups_e.setToolTip("Reference categories you will label with, and which map classes each one merges. "
                                "Works for any map: water:1;land:0   change:2,3;no_change:1   urban:1;forest:2;crops:3;bare:4 ...")
            preset = QComboBox(); preset.addItems(list(validation.PRESETS) + ["One category per map class", "Custom"])
            def on_preset(i):
                names = list(validation.PRESETS)
                if i < len(names):
                    groups_e.setText(validation.PRESETS[names[i]])
                elif i == len(names):
                    try:
                        groups_e.setText(validation.groups_for_classes(list(validation.raster_classes(raster_e.text().strip()))))
                    except Exception as exc:
                        QMessageBox.warning(dlg, "Cannot read the map", str(exc))
            preset.currentIndexChanged.connect(on_preset)
            preset.setCurrentIndex(preset.count() - 1)

            def do_detect():
                try:
                    cls = validation.raster_classes(raster_e.text().strip())
                    counts_e.setText(",".join(f"{c}={100 if n > 5000 else 50}" for c, n in cls.items()))
                    try:
                        covered = {c for cs in validation.parse_groups(groups_e.text()).values() for c in cs}
                    except Exception:
                        covered = set()
                    if not set(cls) <= covered:   # the current categories do not describe this map: one category per class
                        groups_e.setText(validation.groups_for_classes(list(cls)))
                    info.setText("Classes found (value: pixels):  " + "   ".join(f"{c}: {n:,}" for c, n in cls.items()) +
                                 "\nAt least 50 points per class (100 for the most important) is advisable. Rare classes need their own points.")
                except Exception as exc:
                    QMessageBox.warning(dlg, "Cannot read the map", str(exc))
            detect.clicked.connect(do_detect)
            form.addRow("Classified map (GeoTIFF):", self._browse_row(dlg, raster_e, "Classified map", "GeoTIFF (*.tif *.tiff)"))
            form.addRow("", detect); form.addRow(info)
            form.addRow("Points per class:", counts_e); form.addRow("Random seed:", seed); form.addRow("Minimum distance:", mind)
            form.addRow("Map type (preset):", preset)
            form.addRow("Categories to label:", groups_e)
            form.addRow(QLabel("Categories name what the analyst will write in ref_label and merge map classes: name:class,class;name2:class . "
                               "They are saved with the points, so labelling and accuracy use the same names."))
            form.addRow("Output folder:", self._browse_row(dlg, out_e, "Output folder", folder=True))
            form.addRow(QLabel("Then open points_to_label.csv (or the GeoJSON) in QGIS and fill ref_label for every point from reference\n"
                               "imagery WITHOUT looking at the map. points_key.csv holds the map classes: do not open it while labelling."))
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject); form.addRow(btns)
            if dlg.exec_() != QDialog.Accepted:
                return
            if not os.path.isfile(raster_e.text().strip()):
                raise ValueError("Please choose the classified map.")
            counts = validation.parse_counts(counts_e.text())
            groups_spec = groups_e.text().strip()
            groups = validation.parse_groups(groups_spec)
            missing = sorted(set(counts) - {c for cs in groups.values() for c in cs})
            if missing:
                raise ValueError(f"Map classes {missing} get points but belong to no category. Add them to 'Categories to label' "
                                 f"or remove them from 'Points per class'.")
            st.setValue("val/raster", raster_e.text().strip()); st.setValue("val/out", out_e.text().strip())
            st.setValue("val/groups", groups_spec)
            args = (raster_e.text().strip(), out_e.text().strip(), counts, seed.value(), 255, mind.value())
            self._start_validation_job(lambda progress: validation.make_sampling_from_raster(*args, progress=progress, groups_spec=groups_spec),
                                       self._on_sampling_done, "Creating sampling points")
        except Exception as exc:
            self.show_error(f"Cannot create sampling points: {exc}")

    def validation_shuffle_points(self):
        """New blind copy of an existing sample: random order, new ids, empty labels, id_map.csv to compare passes."""
        try:
            st = self._snap_qsettings()
            start = st.value("val/out", "", type=str)
            lab, _ = QFileDialog.getOpenFileName(self, "Choose the points_to_label.csv to shuffle", start, "CSV (*.csv)")
            if not lab:
                return
            key = os.path.join(os.path.dirname(lab), "points_key.csv")
            if not os.path.isfile(key):
                key, _ = QFileDialog.getOpenFileName(self, "Choose the matching points_key.csv", os.path.dirname(lab), "CSV (*.csv)")
                if not key:
                    return
            out = os.path.join(os.path.dirname(lab), "shuffled_" + time.strftime("%Y%m%d_%H%M%S"))
            res = validation.shuffle_points(lab, key, out)
            st.setValue("val/out", out)
            QMessageBox.information(
                self, "Shuffled copy created",
                f"{res['n']} points in random order, new ids, empty labels.\n\nFolder:\n  {out}\n\n"
                "Label THIS copy (Validation > 1b, or the .kml / .csv inside it) and use its points_key.csv in step 2. "
                "id_map.csv links the new ids to the old ones, so two labelling passes can be compared.")
        except Exception as exc:
            self.show_error(f"Cannot shuffle the points: {exc}")

    def validation_label_points(self):
        """Label the sampling points inside the app: reference-image chips, links to Google Earth / Wayback, autosave."""
        try:
            import labeler_ui
            st = self._snap_qsettings()
            out = st.value("val/out", "", type=str)
            csv_path = os.path.join(out, "points_to_label.csv") if out else ""
            try:   # fallback only: the scheme saved beside the points (or the labels already in the file) takes precedence
                names = list(validation.parse_groups(st.value("val/groups", "", type=str)))
            except Exception:
                names = []
            self._label_dlg = labeler_ui.LabelPointsDialog(self, csv_path if os.path.isfile(csv_path) else "", ",".join(names))
            self._label_dlg.show()
        except Exception as exc:
            logging.exception("validation_label_points failed")
            self.show_error(f"Cannot open the labelling window: {exc}")

    def validation_accuracy(self):
        """Step 2 of validation: accuracy and area estimates from the labelled points."""
        try:
            st = self._snap_qsettings()
            dlg = QDialog(self)
            dlg.setWindowTitle("Validation 2: compute accuracy")
            dlg.setMinimumWidth(680)
            form = QFormLayout(dlg)
            raster_e = QLineEdit(st.value("val/raster", "", type=str))
            labels_e = QLineEdit(st.value("val/labels", "", type=str)); key_e = QLineEdit(st.value("val/key", "", type=str))
            out_e = QLineEdit(st.value("val/out", os.path.join(tempfile.gettempdir(), "SpaceGeoVision_validation"), type=str))
            sch0 = validation.read_scheme(labels_e.text().strip()) if labels_e.text().strip() else None
            groups_e = QLineEdit(sch0["groups"] if sch0 else st.value("val/groups", "flood:2;not_flood:0,1,3,4", type=str))
            def on_labels_changed(txt):   # the categories chosen when the points were created follow the points file
                s = validation.read_scheme(txt.strip()) if txt.strip() else None
                if s:
                    groups_e.setText(s["groups"])
            labels_e.textChanged.connect(on_labels_changed)
            form.addRow("Classified map (GeoTIFF):", self._browse_row(dlg, raster_e, "Classified map", "GeoTIFF (*.tif *.tiff)"))
            form.addRow("Labelled points (CSV):", self._browse_row(dlg, labels_e, "points_to_label.csv", "CSV (*.csv)"))
            form.addRow("Key file (points_key.csv):", self._browse_row(dlg, key_e, "points_key.csv", "CSV (*.csv)"))
            preset = QComboBox(); preset.addItems(list(validation.PRESETS) + ["One category per map class", "Custom"])
            def on_acc_preset(i):
                names = list(validation.PRESETS)
                if i < len(names):
                    groups_e.setText(validation.PRESETS[names[i]])
                elif i == len(names):
                    try:
                        groups_e.setText(validation.groups_for_classes(list(validation.raster_classes(raster_e.text().strip()))))
                    except Exception as exc:
                        QMessageBox.warning(dlg, "Cannot read the map", str(exc))
            preset.currentIndexChanged.connect(on_acc_preset)
            preset.setCurrentIndex(preset.count() - 1)
            form.addRow("Map type (preset):", preset)
            form.addRow("Categories:", groups_e)
            form.addRow(QLabel("Categories group map classes, e.g.  flood:2;not_flood:0,1,3,4  or  water:1;land:0  or  urban:1;forest:2;bare:3 .\n"
                               "The ref_label values in the CSV must be these names (case ignored); 'uncertain' or blank is excluded."))
            form.addRow("Output folder:", self._browse_row(dlg, out_e, "Output folder", folder=True))
            score_e = QLineEdit(st.value("val/score", "", type=str))
            target_e = QLineEdit(st.value("val/target", "", type=str)); target_e.setPlaceholderText("e.g. water  (category that is the target)")
            low_c = QCheckBox("Low values of the layer mean the target (e.g. water in dB)"); low_c.setChecked(st.value("val/low", True, type=bool))
            form.addRow(QLabel("Optional - ROC / precision-recall curves: a continuous layer on the SAME grid as the map (e.g. the sigma0-dB image, or the change magnitude)"))
            form.addRow("Layer (GeoTIFF):", self._browse_row(dlg, score_e, "Continuous layer", "GeoTIFF (*.tif *.tiff)"))
            form.addRow("Target category:", target_e); form.addRow("", low_c)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject); form.addRow(btns)
            if dlg.exec_() != QDialog.Accepted:
                return
            for label, e in (("classified map", raster_e), ("labelled points", labels_e), ("key file", key_e)):
                if not os.path.isfile(e.text().strip()):
                    raise ValueError(f"Please choose the {label} file.")
            groups = validation.parse_groups(groups_e.text())
            for k, e in (("val/raster", raster_e), ("val/labels", labels_e), ("val/key", key_e), ("val/out", out_e), ("val/groups", groups_e)):
                st.setValue(k, e.text().strip())
            args = (raster_e.text().strip(), labels_e.text().strip(), key_e.text().strip(), groups, out_e.text().strip())
            score, target = score_e.text().strip(), target_e.text().strip()
            for k, v in (("val/score", score), ("val/target", target)):
                st.setValue(k, v)
            st.setValue("val/low", low_c.isChecked())
            if score and not target:
                raise ValueError("Give the target category (e.g. water) to draw the curves, or clear the layer field.")
            low = low_c.isChecked()

            def job(progress):
                res = validation.validate_from_files(*args, progress=progress)
                if score:
                    progress(90, "ROC / precision-recall curves")
                    cur = validation.validate_curves_from_files(score, args[1], args[2], args[0], groups,
                                                                [s.strip() for s in target.split(",") if s.strip()], args[4], low_is_positive=low)
                    res["report"] += "\n\n" + cur["report"]; res["curves"] = cur["files"]
                return res
            self._start_validation_job(job, self._on_validation_done, "Computing accuracy")
        except Exception as exc:
            self.show_error(f"Cannot compute accuracy: {exc}")

    def _start_validation_job(self, fn, on_done, title):
        self._val_progress = QProgressDialog("Starting...", None, 0, 100, self)
        self._val_progress.setWindowTitle(title)
        self._val_progress.setWindowModality(Qt.WindowModal)
        self._val_progress.setMinimumDuration(0)
        worker = FuncWorker(fn)
        worker.progress.connect(lambda v, m: (self._val_progress.setValue(v), self._val_progress.setLabelText(m)))
        worker.done.connect(on_done)
        worker.failed.connect(lambda m: (self._val_progress.close(), self.show_error(f"{title} failed:\n{m}")))
        self._val_worker = worker
        worker.start()

    def _on_sampling_done(self, res):
        self._val_progress.close()
        QMessageBox.information(
            self, "Sampling points created",
            f"{res['n']} points {res['per_class']}\n\nFiles:\n  " + "\n  ".join(res['files'].values()) +
            "\n\nNext: label the points - either in the app (Validation > 1b. Label Points in the App) or in Google Earth Pro "
            "(open the .kml and use its Historical Imagery slider) and write ref_label in the CSV - without looking at the map. "
            "Then use Validation > 2. Compute Accuracy.")
        if res['files'].get('kml') and QMessageBox.question(
                self, "Google Earth", "Open the points in Google Earth now (if Google Earth Pro is installed)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            try:
                if hasattr(os, 'startfile'):
                    os.startfile(res['files']['kml'])
                else:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(res['files']['kml']))
            except Exception as exc:
                self.show_error(f"Could not open the KML: {exc}")

    def _on_validation_done(self, res):
        self._val_progress.close()
        dlg = QDialog(self)
        dlg.setWindowTitle("Accuracy assessment (area-adjusted, from labelled sample points)")
        dlg.setMinimumSize(900, 520)
        lay = QVBoxLayout(dlg)
        box = QTextEdit(); box.setReadOnly(True); box.setFontFamily("Courier New")
        box.setPlainText(res["report"] + "\n\nSaved: " + res["report_path"] + ("\n       " + "\n       ".join(res["curves"].values()) if res.get("curves") else ""))
        lay.addWidget(box)
        ok = QDialogButtonBox(QDialogButtonBox.Ok); ok.accepted.connect(dlg.accept); lay.addWidget(ok)
        dlg.exec_()

    def _snap_qsettings(self):
        return QSettings("SpaceGeoVision", "SpaceGeoVision")

    def _get_gpt(self, ask=True):
        """Return the gpt path (settings, $SNAP_GPT, PATH, default folders) or None."""
        saved = self._snap_qsettings().value("snap/gpt_path", "", type=str) or None
        try:
            return sar_snap.find_gpt(saved)
        except sar_snap.SnapError as exc:
            if ask:
                self.show_error(str(exc))
            return None

    def snap_settings(self):
        """Let the user point to SNAP's gpt executable and test it."""
        current = self._get_gpt(ask=False) or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SNAP 'gpt' executable (…/snap/bin/gpt)", current, "gpt (gpt gpt.exe);;All Files (*)")
        if not path:
            return
        try:
            sar_snap.check_gpt(path)
        except sar_snap.SnapError as exc:
            self.show_error(f"This file does not run as SNAP gpt:\n{exc}")
            return
        self._snap_qsettings().setValue("snap/gpt_path", path)
        self.show_status(f"SNAP gpt set to {path} (version: {sar_snap.snap_version(path)})", 8000)

    def start_sar_workflow(self):
        """Step 1: choose the ORIGINAL Sentinel-1 GRD product (.zip or .SAFE folder)."""
        try:
            box = QMessageBox(self)
            box.setWindowTitle("Sentinel-1 product")
            box.setText("Choose the original Sentinel-1 GRD product.\n"
                        "GeoTIFF exports cannot be orbit-corrected or calibrated to σ₀.")
            zip_btn = box.addButton("ZIP file...", QMessageBox.AcceptRole)
            dir_btn = box.addButton(".SAFE folder...", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            clicked = box.clickedButton()
            if clicked is zip_btn:
                path, _ = QFileDialog.getOpenFileName(self, "Open Sentinel-1 GRD ZIP", "", "Sentinel-1 ZIP (*.zip)")
            elif clicked is dir_btn:
                path = QFileDialog.getExistingDirectory(self, "Open Sentinel-1 .SAFE folder")
            else:
                return
            if not path:
                return
            info = sar_snap.parse_product_name(path)
            self.sar_product = info
            try:
                self.sar_footprint = sar_snap.read_footprint(path)
            except sar_snap.SnapError:
                self.sar_footprint = None
            self.sar_steps = {}
            self.current_sar_group = 'sentinel1'
            self._reset_workflow()
            self.info_box.setText(
                "🛰 Sentinel-1 product selected\n"
                f"• File: {os.path.basename(path)}\n"
                f"• Mission: {info.mission}   Mode: {info.mode}   Type: {info.product_type}\n"
                f"• Polarisations: {', '.join(info.polarisations)}\n"
                f"• Acquisition start: {info.start}\n\n"
                "Next: SAR menu → 2. Run SNAP Preprocessing")
            self.layer_list.addItem(QListWidgetItem(f"🛰 {os.path.basename(path)}"))
            self._enable_next_step('load')
        except sar_snap.SnapError as exc:
            self.show_error(str(exc))
            self._reset_workflow()
        except Exception as exc:
            self.show_error(f"Could not read the product: {exc}")
            self._reset_workflow()

    def run_snap_preprocessing(self):
        """Step 2: run the whole SNAP chain in a background thread."""
        try:
            if self.sar_product is None:
                raise ValueError("Select a Sentinel-1 product first (step 1).")
            if self.snap_worker is not None and self.snap_worker.isRunning():
                raise ValueError("A SNAP job is already running.")
            gpt = self._get_gpt()
            if not gpt:
                return
            chosen = self._snap_dialog(self.sar_product)
            if not chosen:
                return
            params, out_dir = chosen
            heap_gb = max(4, int(psutil.virtual_memory().total / 2 ** 30 * 0.6))

            self._snap_progress = QProgressDialog("Starting SNAP...", "Cancel", 0, 100, self)
            self._snap_progress.setWindowTitle("SNAP preprocessing")
            self._snap_progress.setWindowModality(Qt.WindowModal)
            self._snap_progress.setMinimumDuration(0)
            self._snap_progress.setValue(0)

            worker = SnapWorker(self.sar_product.path, out_dir, params, gpt, heap_gb)
            worker.progress.connect(lambda v, m: (self._snap_progress.setValue(v),
                                                  self._snap_progress.setLabelText(m)))
            worker.done.connect(self._on_snap_done)
            worker.failed.connect(self._on_snap_failed)
            self._snap_progress.canceled.connect(worker.cancel)
            self.snap_worker = worker
            worker.start()
        except sar_snap.SnapError as exc:
            self.show_error(str(exc))
        except Exception as exc:
            self.show_error(f"Cannot start SNAP preprocessing: {exc}")

    def _on_snap_failed(self, message):
        if getattr(self, '_snap_progress', None):
            self._snap_progress.close()
        if message == '__cancelled__':
            self.show_status("SNAP preprocessing cancelled", 5000)
        else:
            self.show_error(f"SNAP preprocessing failed:\n{message}")

    def _on_snap_done(self, payload):
        """Store the REAL SNAP output (sigma0 in dB, terrain-corrected) as the 'dB' step."""
        try:
            result, data, profile = payload
            if getattr(self, '_snap_progress', None):
                self._snap_progress.close()
            warnings = sar_snap.sanity_check_db(data)
            vv = data[np.isfinite(data) & (data != 0.0)]
            if vv.size and float(np.mean(vv < -15.0)) > 0.6:
                warnings.append('More than 60% of the processed area has very low backscatter (< -15 dB), typical of open water '
                                '(sea, large lake) or radar shadow. The selected study area may be mostly water: check the area in step 2.')
            crs = profile.get('crs')
            pix = abs(profile['transform'].a)
            unit = '°' if (crs is not None and crs.is_geographic) else ' m'
            stats = self._calculate_stats(data)
            valid = data[np.isfinite(data)]
            p = result.params
            self.sar_steps = {
                'dB': {
                    'data': data,
                    'profile': profile,
                    'stats': stats,
                    'vis_params': {'cmap': 'gray',
                                   'vmin': float(np.nanpercentile(valid, 2)) if valid.size else -25.0,
                                   'vmax': float(np.nanpercentile(valid, 98)) if valid.size else 0.0},
                    'metadata': {
                        'file_name': os.path.basename(result.geotiff),
                        'acquisition_date': self.sar_product.start,
                        'polarization': p.polarisation,
                        'dimensions': f"{data.shape[1]}×{data.shape[0]}",
                        'resolution': f"{pix:.6g}{unit}",
                        'data_type': 'float32 (σ₀, dB)',
                        'crs': crs.to_string() if crs is not None else 'Unknown',
                        'processing_history': [
                            f"SNAP {p.orbit_type}", "ThermalNoiseRemoval" if p.remove_thermal_noise else "no thermal noise removal",
                            "Calibration σ₀", f"Speckle-Filter {p.speckle_filter} {p.speckle_window}x{p.speckle_window}",
                            f"Terrain-Correction (Range-Doppler, {p.dem_name}, {p.pixel_spacing_m} m)", "LinearToFromdB"],
                    },
                }
            }
            self.sar_snap_result = result
            side = ""
            try:
                side = sar_snap.write_provenance(result, self._get_gpt(ask=False) or "", self.sar_product.path)
            except Exception as exc:  # provenance must never break the workflow
                logging.warning("Could not write provenance file: %s", exc)
            self._display_sar_layer('dB')
            self.layer_list.addItem(QListWidgetItem(f"📈 σ₀ dB ({p.polarisation}, SNAP)"))
            self._enable_next_step('snap')
            msg = f"SNAP preprocessing finished: {result.geotiff}"
            if side:
                msg += f"\nProvenance: {side}"
            if warnings:
                QMessageBox.warning(self, "Check the result", "\n".join(warnings) + "\n\n" + msg)
            else:
                self.show_status(msg.replace("\n", "  |  "), 10000)
        except Exception as exc:
            self.show_error(f"Could not load the SNAP result: {exc}\n{traceback.format_exc()}")


    


    def _display_sar_layer(self, layer_name):
        """Display SAR layer with full metadata and visualization"""
        try:
            # Clear previous figure
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            
            # Get layer data with validation
            layer_data = self.sar_steps.get(layer_name)
            if not layer_data:
                raise ValueError(f"Layer {layer_name} not found in workflow steps")
                
            data = layer_data.get('data')
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                raise ValueError("Invalid SAR data format - expected 2D array")

            # Get visualization parameters with defaults
            vis_params = layer_data.get('vis_params', {})
            cmap = vis_params.get('cmap', 'gray')
            vmin = vis_params.get('vmin', np.nanpercentile(data, 2))
            vmax = vis_params.get('vmax', np.nanpercentile(data, 98))

            # Create masked array for visualization
            masked_data = np.ma.masked_invalid(data)
            
            # Configure colormap
            cmap = copy(_get_cmap(cmap))
            cmap.set_bad('magenta', 1.0)  # Highlight invalid values
            
            # Display image
            prof = layer_data.get('profile') or {}
            spec = georef.axis_spec(prof.get('transform'), prof.get('crs'), data.shape[0], data.shape[1])
            img = self.ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax, extent=spec.extent)
            self._decorate_map_axes(spec)
            plt.colorbar(img, ax=self.ax, label='Intensity Value')
            self.ax.set_title(f"{layer_name} Layer")
            self.canvas.draw()

            # Get metadata with fallback values
            metadata = layer_data.get('metadata', {})
            stats = layer_data.get('stats', {})
            
            # Format numerical values safely
            def format_value(val, precision=2):
                try:
                    if np.isnan(val):
                        return 'N/A'
                    return f"{float(val):.{precision}f}"
                except (TypeError, ValueError):
                    return str(val)

            # Build information text
            info_lines = [
                f"📂 File: {metadata.get('file_name', 'N/A')}",
                f"📅 Acquisition: {metadata.get('acquisition_date', 'N/A')}",
                f"🔄 Polarization: {metadata.get('polarization', 'N/A')}",
                f"📏 Dimensions: {metadata.get('dimensions', 'N/A')}",
                f"🎯 Resolution: {metadata.get('resolution', 'N/A')}",
                f"🧮 Data Type: {metadata.get('data_type', 'N/A')}",
                f"🌐 CRS: {metadata.get('crs', 'Unknown')}"
            ]


            # Add processing parameters if available
            if 'parameters' in layer_data:
                params = layer_data['parameters']
                info_lines.extend([
                    "\n⚙ Processing Parameters:",
                    f"• Filter Type: {params.get('filter_type', 'N/A')}",
                    f"• Window Size: {params.get('window_size', 'N/A')}",
                    f"• SNR Estimate: {format_value(params.get('snr_estimate'))}",
                    f"• Processed At: {params.get('timestamp', 'N/A')}"
                ])

            # Add statistics section
            info_lines.extend([
                "\n📈 Statistics:",
                f"• Min: {format_value(stats.get('min'))}",
                f"• Max: {format_value(stats.get('max'))}",
                f"• Mean: {format_value(stats.get('mean'))}",
                f"• Std Dev: {format_value(stats.get('std'))}",
                f"• 2nd %ile: {format_value(stats.get('percentile_2'))}",
                f"• 98th %ile: {format_value(stats.get('percentile_98'))}"
            ])

            # Update info box
            self.info_box.setText("\n".join(info_lines))

        except ValueError as ve:
            self.show_error(f"Data validation error: {str(ve)}")
        except AttributeError as ae:
            self.show_error(f"Display configuration error: {str(ae)}")
        except Exception as e:
            self.show_error(f"Unexpected display error: {str(e)}\n{traceback.format_exc()}")
    

    
    
    
    
    
    


    


    


    

  
    


    

    

    
    


    

    def _calculate_stats(self, data):
        """Robust statistics calculation"""
        valid_data = data[np.isfinite(data)]
        if valid_data.size == 0:
            return {
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'percentile_2': np.nan,
                'percentile_98': np.nan
            }
        
        return {
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'mean': np.mean(valid_data),
            'std': np.std(valid_data),
            'percentile_2': np.percentile(valid_data, 2),
            'percentile_98': np.percentile(valid_data, 98)
        }
    


            

    

    # Add to menu bar initialization

    



    # In analyze_histogram method, update the Complete button connection:
    # Replace:
    # complete_btn.clicked.connect(analysis_dialog.accept)
    # With: complete_btn.clicked.connect(self._on_complete_step)

    # Remove the old _save_thresholds method and replace with:
    
   

    
    


    




    


 


 
    


    

    
    def _enable_next_step(self, completed_step_name):
        """Unified step enabling using step names"""
        try:
            current_idx = self.workflow_steps.index(completed_step_name)
            next_idx = current_idx + 1
            
            if next_idx < len(self.workflow_actions):
                self.workflow_actions[next_idx].setEnabled(True)
                self.statusBar().showMessage(
                    f"Next step available: {self.workflow_steps[next_idx].replace('_', ' ').title()}",
                    3000
                )
        except ValueError as e:
            self.show_error(f"Workflow error: {str(e)}")


    def show_error(self, message):
            """Standard error display method"""
            QMessageBox.critical(self, "Error", message)
            self.error_log.append(message)
            self.statusBar().showMessage(f"Error: {message.split(':')[0]}", 5000)

    def show_status(self, message, timeout=3000):
        """Status display method"""
        self.statusBar().showMessage(message, timeout)

    
    
    
    def _reset_workflow(self):
        """Reset workflow to initial state"""
        for action in self.workflow_actions:
            action.setEnabled(False)
        self.workflow_actions[0].setEnabled(True)
        self.current_step = 0


        
#############################################

    

    
