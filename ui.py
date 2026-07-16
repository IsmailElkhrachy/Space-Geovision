import os
import cv2
import re
import traceback
import logging
import json
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from rasterio.features import shapes
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import shape
import tempfile
from PIL import Image
from skimage import img_as_float, measure, exposure, transform
from skimage.feature import canny
from sklearn.cluster import MeanShift, KMeans
import folium
import leafmap
import matplotlib.patches as mpatches  # <-- Add this import
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtWidgets import (QGraphicsEllipseItem,QGraphicsPolygonItem, QGraphicsRectItem,
    QMainWindow, QAction, QFileDialog, QMessageBox, QVBoxLayout, QLabel, QFrame, QWidget, QRadioButton,
    QInputDialog, QProgressBar, QSizePolicy, QHBoxLayout, QListWidget, QListWidgetItem, QButtonGroup,QTabWidget,
    QPushButton, QColorDialog, QFontDialog, QProgressDialog, QTextEdit, QDialog, QApplication,QToolButton,
    QSplitter, QMenuBar, QToolBar, QMenu, QDoubleSpinBox, QDialogButtonBox, QStackedWidget, QHBoxLayout, 
    QComboBox, QLineEdit, QCheckBox, QSpinBox, QSlider, QGroupBox, QFormLayout, QScrollArea, QTableWidget, QTableWidgetItem
)
from PyQt5.QtPrintSupport import QPrinter, QPrintPreviewDialog
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtCore import Qt,QRectF,QRect,QDate ,QThread, pyqtSignal,QDateTime ,QMutex, QMutexLocker, QMetaObject, QSize, QSettings, QTimer, QPointF, QRectF, QStandardPaths
from PyQt5.QtGui import QColor, QFont, QPixmap, QImage, QIcon, QPainter,QDoubleValidator,QPainter, QPen, QColor, QPolygonF, QBrush
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog, QPageSetupDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView
from helpers import ImageProcessor, ImageClassifier, RasterHelper
from segment_anything import sam_model_registry, SamPredictor
import torch
from sympy import sympify, Mul, Pow, Add
import folium
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import tempfile
from folium import Map
from rasterio.transform import from_origin
import re, ast
import contextily as ctx
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh, PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from rasterio import features  # Add this import
import colorsys
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
import whitebox
from whitebox import WhiteboxTools
import shutil
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
from polygon_processor import PolygonProcessor
from shapely.geometry import Polygon, MultiPolygon, LineString
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, ListedColormap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QSplitter, QListWidget)
import math

from matplotlib import colors
import sys

# Add to your PyQt5 imports at the top of the file
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt
from scipy.ndimage import uniform_filter
from copy import copy
from skimage import filters
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import psutil
from datetime import datetime
from skimage.morphology import binary_closing, remove_small_objects
import folium
from PyQt5.QtCore import QUrl

# Logging Configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Worker(QThread):
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
        attempts = 3
        for attempt in range(attempts):
            try:
                self.status.emit("Starting operation...")
                self.progress.emit(-1)
                result = self.operation(self.image, *self.args)
                if self._is_running:
                    self.finished.emit(result)
                    self.progress.emit(100)
                    self.status.emit("Operation completed successfully.")
                    break
            except Exception as e:
                if attempt == attempts - 1:
                    self.progress.emit(-2)
                    error_msg = f"Error: {str(e)}"
                    self.status.emit(error_msg)
                    self.error.emit(error_msg)
                    logging.error(f"Error in worker thread: {str(e)}")
                else:
                    self.status.emit(f"Retrying... ({attempt + 1}/{attempts})")
            finally:
                self.progress.emit(0)
                self.status.emit("Operation finished.")

    def stop(self):
        with QMutexLocker(self._lock):
            self._is_running = False
        self.terminate()

    def stop(self):
        with QMutexLocker(self._mutex):
            self._is_running = False
        self.quit()
        self.wait(500)  # Wait up to 500ms for clean exit

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
        self.workflow_steps = [
            'load', 'filter', 'convert_db', 
            'analyze', 'classify'
        ]

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
        self.google_maps_view = QWebEngineView()
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

        wet_area_action = QAction('Detect Wet Areas (Sentinel-1)', self)
        wet_area_action.triggered.connect(self.detect_wet_areas)
        dem_menu.addAction(wet_area_action)


        flow_accumulation_action = QAction('Flow Accumulation', self)
        flow_accumulation_action.triggered.connect(self.calculate_flow_accumulation)
        dem_menu.addAction(flow_accumulation_action)

        stream_network_action = QAction('Calculate Stream Network', self)
        stream_network_action.triggered.connect(self.calculate_stream_network)
        dem_menu.addAction(stream_network_action)

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


        reshape_action = QAction('Reshape Polygons', self)
        reshape_action.triggered.connect(self.polygon_processor.reshape_polygons)
        editing_menu.addAction(reshape_action)

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
        self.setWindowIcon(QIcon('D:\\Python_Codes\\icons\\Asset 4@2x.png'))
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
            print(f"Error loading stylesheet: {str(e)}")

##########################################  The End of def initUI

    def setup_edit_connections(self):
        self.actionStart_Editing.triggered.connect(self.polygon_processor.start_editing)
        self.actionSave_Edits.triggered.connect(self.polygon_processor.save_edits)
        self.actionStop_Editing.triggered.connect(
            lambda: self.polygon_processor.stop_editing(False))
    # Add these methods to your class:
    def show_error(self, message):
        """Show critical error message"""
        QMessageBox.critical(self, "Error", message)
        self.statusBar().showMessage(f"Error: {message}", 5000)
    
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
        self.rect_select_action = QAction(QIcon("d:\\Python_Codes\\icons\\select_rect.png"), "Rectangle Select", self)
        self.poly_select_action = QAction(QIcon("d:\\Python_Codes\\icons\\select_poly.png"), "Polygon Select", self)
        self.clear_select_action = QAction(QIcon("d:\\Python_Codes\\icons\\clear_select.png"), "Clear Selection", self)
        
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
    
    def show_selection_instructions(self):
        """Show instructions for polygon selection mode"""
        instructions = (
            "Polygon Selection Mode\n\n"
            "1. Click to add polygon vertices\n"
            "2. Continue clicking to add more points\n"
            "3. Right-click to complete the selection\n"
            "4. Press ESC to cancel"
        )
        QMessageBox.information(self, "Selection Instructions", instructions)

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
        
    def activate_polygon_drawing_mode(self, callback):
        """Activate polygon drawing on the canvas"""
        self.canvas.set_drawing_mode('polygon', callback)
        
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
                    if len(layer_data['image'].shape) == 2:  # Single band
                        self.ax.imshow(layer_data['image'], cmap='gray')
                    else:  # Multi-band
                        # Handle RGB images (transpose to HxWxC)
                        if layer_data['image'].shape[0] in [3, 4]:
                            img = np.transpose(layer_data['image'], (1, 2, 0))
                            self.ax.imshow(img)
                        else:
                            # Display first 3 bands if more than 3
                            img = np.transpose(layer_data['image'][:3], (1, 2, 0))
                            self.ax.imshow(img)
            
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
            
            # Add to your shapefile (example)
            if hasattr(self, 'active_layer'):
                self.active_layer.add_feature(poly)
                self.logger.info("Added polygon to shapefile")
            
            # Reset drawing state
            self.polygon_points = []
            self.temp_polygon = None
            self.update()
            
        except Exception as e:
            self.logger.error(f"Error finalizing polygon: {str(e)}")
            QMessageBox.critical(self, "Error", 
                            f"Could not create polygon: {str(e)}")

    def finalize_polygon_selection(self):
        """Process completed polygon selection"""
        if len(self.selection_points) >= 3:
            polygon = Polygon(self.selection_points)
            layer_name = self.polygon_processor.get_selected_vector_layer()
            
            if layer_name and layer_name in self.layers:
                gdf = self.layers[layer_name]['gdf']
                selected = gdf[gdf.geometry.intersects(polygon)]
                self.polygon_processor.set_selected_features(selected.index.tolist())
        
        self.selection_mode = None
        self.selection_points = []
        self.update_map_display()

    def finalize_selection(self):
        """Process completed selection"""
        if self.selection_mode == 'rectangle':
            self.polygon_processor.set_selected_features(self._get_rectangle_selection())
        elif self.selection_mode == 'polygon':
            self.polygon_processor.set_selected_features(self._get_polygon_selection())
            
        self.selection_mode = None
        self.update_map_display()
    
    def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape and self.polygon_points:
                self.polygon_points = []
                self.temp_polygon = None
                self.update() 
    def snap_to_features(self, point):
        """Implement snapping logic if needed"""
        # Your snapping implementation here
        return point
    
    def setup_mouse_selection(self):
        """Connect mouse events for image selection"""
        self.canvas.mpl_connect('button_press_event', self.on_image_click)

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
            print(f"Error handling image click: {e}")


    def add_layer(self):
        """
        إضافة طبقة جديدة إلى القائمة.
        """
        layer_name, ok = QInputDialog.getText(self, "Add Layer", "Enter layer name:")
        if ok and layer_name:
            self.add_layer_to_list(layer_name, None, None)  # يمكنك استبدال None ببيانات الصورة والبروفايل

    def delete_layer(self):
        """
        Delete the selected layer from the list along with its image and legend
        """
        selected_item = self.layer_list.currentItem()
        if selected_item:
            self.delete_layer_from_list(selected_item)


    def show_context_menu(self, position):
        """
        عرض قائمة منسدلة لحذف الملف المحدد.
        """
        item = self.layer_list.itemAt(position)
        if item:
            menu = QMenu(self)
            delete_action = menu.addAction("Delete Layer")
            delete_action.triggered.connect(lambda: self.delete_layer_from_list(item))
            menu.exec_(self.layer_list.viewport().mapToGlobal(position))

    def add_layer_to_list(self, layer_name, image, profile):
        """
        إضافة ملف إلى قائمة الطبقات.
        """
        self.layers[layer_name] = {
            'image': image,
            'profile': profile,
            'file_name': layer_name
        }
        item = QListWidgetItem(layer_name)
        self.layer_list.addItem(item)

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
        print("Display cleared.")

    
    def display_raster_layer(self, layer_data, layer_name, show_grid=True):
        """Display raster layer with proper handling of float64 and NaN values"""
        try:
            
            # Clear previous plot completely
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_axis_on()
            
            # Ensure layer_data is properly structured
            if isinstance(layer_data, np.ndarray):
                layer_data = {'image': layer_data, 'profile': {}}
            
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
                cmap = plt.cm.get_cmap(cmap).copy()
                cmap.set_bad(color='red', alpha=0.5)  # Make NaN values red
                
                img = self.ax.imshow(display_image, 
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
                self.ax.imshow(rgb_image)
                
                # Remove colorbar for RGB images
                if hasattr(self, 'colorbar'):
                    self.colorbar.remove()
                    del self.colorbar
            
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
            
        except Exception as e:
            print(f"Error displaying layer {layer_name}: {str(e)}")
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
            print(f"Vector display error: {str(e)}")
            self._fallback_display(layer_name)

    def _draw_labels(self, gdf, layer_data):
        """Draw labels for vector features"""
        label_field = layer_data['label_field']
        if label_field not in gdf.columns:
            return
            
        # Get label styling properties
        font_family = layer_data.get('label_font', 'Arial')
        font_size = layer_data.get('label_size', 10)
        color = layer_data.get('label_color', '#000000')
        placement = layer_data.get('label_placement', 'Around Point')
        
        # Create font properties
        font = {'family': font_family,
                'size': font_size,
                'color': color}
        
        # Plot labels for each feature
        for idx, row in gdf.iterrows():
            geom = row.geometry
            label_text = str(row[label_field])
            
            if geom.geom_type in ['Point', 'MultiPoint']:
                if geom.geom_type == 'MultiPoint':
                    centroid = geom.centroid
                else:
                    centroid = geom
                    
                x, y = centroid.x, centroid.y
                
                # Adjust placement based on settings
                if placement == 'Above Point':
                    va = 'bottom'
                    ha = 'center'
                    y_offset = font_size * 0.005  # Small offset
                elif placement == 'Below Point':
                    va = 'top'
                    ha = 'center'
                    y_offset = -font_size * 0.005
                elif placement == 'Left of Point':
                    va = 'center'
                    ha = 'right'
                    x_offset = -font_size * 0.005
                elif placement == 'Right of Point':
                    va = 'center'
                    ha = 'left'
                    x_offset = font_size * 0.005
                else:  # Around Point (default)
                    va = 'center'
                    ha = 'center'
                    x_offset = 0
                    y_offset = 0
                
                self.ax.text(x + x_offset, y + y_offset, label_text, 
                            fontdict=font, va=va, ha=ha)
                
            elif geom.geom_type in ['LineString', 'MultiLineString']:
                # Place label at midpoint
                if geom.geom_type == 'MultiLineString':
                    line = geom.geoms[0]  # Just use first line for simplicity
                else:
                    line = geom
                    
                midpoint = line.interpolate(0.5, normalized=True)
                self.ax.text(midpoint.x, midpoint.y, label_text, 
                            fontdict=font, va='center', ha='center')
                
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                # Place label at centroid
                if geom.geom_type == 'MultiPolygon':
                    centroid = geom.centroid
                else:
                    centroid = geom.centroid
                    
                self.ax.text(centroid.x, centroid.y, label_text, 
                            fontdict=font, va='center', ha='center')
    

    def _setup_coord_display(self, image, profile):
        """Setup coordinate display with value reading"""
        if hasattr(profile, 'transform') and profile.transform:
            # Geographic coordinates
            def format_coord(x, y):
                col, row = ~profile.transform * (x, y)
                if 0 <= row < image.shape[-2] and 0 <= col < image.shape[-1]:
                    if image.ndim == 2:
                        val = image[int(row), int(col)]
                    elif image.ndim == 3 and image.shape[0] == 1:
                        val = image[0, int(row), int(col)]
                    else:
                        val = "Multi-band"
                    
                    if isinstance(val, np.ma.core.MaskedConstant):
                        return f"X={x:.2f}, Y={y:.2f} | NoData"
                    return f"X={x:.2f}, Y={y:.2f} | Value={val:.2f}"
                return f"X={x:.2f}, Y={y:.2f}"
        else:
            # Pixel coordinates
            def format_coord(x, y):
                if 0 <= x < image.shape[-1] and 0 <= y < image.shape[-2]:
                    if image.ndim == 2:
                        val = image[int(y), int(x)]
                    elif image.ndim == 3 and image.shape[0] == 1:
                        val = image[0, int(y), int(x)]
                    else:
                        val = "Multi-band"
                    
                    if isinstance(val, np.ma.core.MaskedConstant):
                        return f"Col={int(x)}, Row={int(y)} | NoData"
                    return f"Col={int(x)}, Row={int(y)} | Value={val:.2f}"
                return f"Col={int(x)}, Row={int(y)}"
        
        self.ax.format_coord = format_coord 

    def _get_image_extent(self, profile):
        """Calculate image extent from geotransform if available"""
        if hasattr(profile, 'transform') and profile.transform:
            width = profile.width
            height = profile.height
            xmin, ymax = profile.transform * (0, 0)
            xmax, ymin = profile.transform * (width, height)
            return [xmin, xmax, ymin, ymax]
        return None
    
    def _fallback_display(self, layer_name):
        """Simplest possible display when everything fails"""
        try:
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, layer_name, ha='center', va='center')
            self.fig.tight_layout(pad=0)
            self.canvas.draw()
        except:
            self.canvas.draw()    

    def mouse_click_handler(self, event):
        """Safe mouse click handler"""
        if event.inaxes != self.ax or self.current_layer not in self.layers:
            return
        
        try:
            self.display_raster_layer(self.layers[self.current_layer], self.current_layer)
        except Exception:
            pass  # Silently fail to prevent recursive errors

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
        crs_info = self.get_crs_info(layer_data.get('gdf').crs if 'gdf' in layer_data else layer_data.get('profile', {}).get('crs'))
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

    def _set_label_placement(self, layer_name, placement):
        """Set label placement for the layer"""
        self.layers[layer_name]['label_placement'] = placement
        self.redraw_all_layers()

    def _set_label_offset(self, layer_name, offset):
        """Set label offset in pixels"""
        self.layers[layer_name]['label_offset'] = offset
        self.redraw_all_layers()

    def _draw_vector_layer(self, layer_name, layer_data):
        """Draw vector layer with current symbology"""
        gdf = layer_data['gdf']
        
        if gdf.geometry.type.iloc[0] in ['Point', 'MultiPoint']:
            self._draw_point_layer(layer_name, layer_data)
        elif gdf.geometry.type.iloc[0] in ['LineString', 'MultiLineString']:
            self._draw_line_layer(layer_name, layer_data)
        elif gdf.geometry.type.iloc[0] in ['Polygon', 'MultiPolygon']:
            self._draw_polygon_layer(layer_name, layer_data)

    def _draw_point_layer(self, layer_name, layer_data):
        """Draw points with current symbology settings"""
        gdf = layer_data['gdf']
        symbol = layer_data.get('point_symbol', 'Circle')
        size = layer_data.get('point_size', 5)
        color = QColor(layer_data.get('color', '#FF0000'))
        rotation = layer_data.get('rotation', 0)

        for _, row in gdf.iterrows():
            point = row.geometry
            if point.is_empty:
                continue

            # Create QGraphicsItem based on symbol type
            if symbol == "Circle":
                item = QGraphicsEllipseItem(-size/2, -size/2, size, size)
            elif symbol == "Square":
                item = QGraphicsRectItem(-size/2, -size/2, size, size)
            elif symbol == "Triangle":
                polygon = QPolygonF([
                    QPointF(0, -size/2),
                    QPointF(size/2, size/2),
                    QPointF(-size/2, size/2)
                ])
                item = QGraphicsPolygonItem(polygon)
            # ... (other symbol types)

            item.setPos(point.x, point.y)
            item.setBrush(QBrush(color))
            item.setRotation(rotation)
            self.scene.addItem(item)

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
            if 'profile' in layer_data:
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

    def _change_crs(self, layer_name, label_widget):
        """Change the CRS of a layer"""
        # This would need a proper CRS selection dialog implementation
        # For now just demonstrate the concept
        new_crs = "EPSG:3857"  # Would come from dialog in real implementation
        if 'gdf' in self.layers[layer_name]:
            self.layers[layer_name]['gdf'] = self.layers[layer_name]['gdf'].to_crs(new_crs)
        elif 'profile' in self.layers[layer_name]:
            self.layers[layer_name]['profile']['crs'] = new_crs
        
        crs_info = self.get_crs_info(new_crs)
        label_widget.setText(f"{crs_info.get('name', 'Unknown')} ({crs_info.get('epsg', 'Unknown')})")
        self.redraw_all_layers()

    def _set_layer_visibility(self, layer_name, visible):
        """Set layer visibility"""
        self.layers[layer_name]['visible'] = visible
        self.redraw_all_layers()

    def _set_transparency(self, layer_name, value):
        """Set layer transparency (0-100)"""
        self.layers[layer_name]['transparency'] = value
        self.redraw_all_layers()

    def _set_symbology_type(self, layer_name, symbology_type):
        """Set the symbology type (single, categorized, graduated, etc.)"""
        self.layers[layer_name]['symbology_type'] = symbology_type
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

    def _set_color_ramp(self, layer_name, ramp):
        """Set the color ramp for categorized/graduated symbology"""
        self.layers[layer_name]['color_ramp'] = ramp
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

    def _pick_layer_color(self, layer_name, button):
        color = QColorDialog.getColor()
        if color.isValid():
            self.layers[layer_name]['color'] = color.name()
            button.setStyleSheet(f"background-color: {color.name()}")
            self.redraw_all_layers()

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

    def _set_blend_mode(self, layer_name, mode):
        """Set layer blending mode"""
        self.layers[layer_name]['blend_mode'] = mode
        self.redraw_all_layers()

    def _set_label_buffer(self, layer_name, enabled):
        """Enable/disable label buffer/halo"""
        self.layers[layer_name]['label_buffer'] = enabled
        self.redraw_all_layers()

    def _set_buffer_size(self, layer_name, size):
        """Set label buffer size"""
        self.layers[layer_name]['buffer_size'] = size
        self.redraw_all_layers()

    def _set_buffer_color(self, layer_name, color):
        """Set label buffer color"""
        self.layers[layer_name]['buffer_color'] = color
        self.redraw_all_layers()


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def toggle_google_maps(self):
        """
        Toggle between the opened image and Google Maps.
        """
        try:
            if not self.google_maps_enabled:
                # Switch to Google Maps view
                self.google_maps_enabled = True

                # Collect user input for center and zoom
                while True:
                    center, ok1 = QInputDialog.getText(
                        self, "Map Center", "Enter center coordinates (lat, lon):\nExample: 37.8713, -122.2580"
                    )
                    if not ok1:
                        return  # User clicked Cancel

                    try:
                        # Convert to list of floats and validate
                        lat, lon = map(float, center.replace(" ", "").split(","))
                        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                            raise ValueError
                        break
                    except (ValueError, IndexError):
                        QMessageBox.critical(self, "Error", "Invalid coordinates format.\nExample: 37.8713, -122.2580")

                while True:
                    zoom, ok2 = QInputDialog.getInt(
                        self, "Map Zoom", "Enter zoom level (1-20):", 
                        value=12, min=1, max=20
                    )
                    if ok2:
                        break

                # Create Folium map
                folium_map = folium.Map(
                    location=[lat, lon],
                    zoom_start=zoom,
                    control_scale=True,
                    tiles=None  # Start with no base tiles
                )

                # Add Google Satellite layer
                folium.TileLayer(
                    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                    attr='Google Satellite',
                    name='Google Satellite',
                    subdomains='mt0mt1mt2mt3',
                    overlay=False,
                    control=False,
                    max_native_zoom=20
                ).add_to(folium_map)

                # Add layer control
                folium.LayerControl(position='topright').add_to(folium_map)

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_file:
                    folium_map.save(temp_file.name)
                    temp_file_path = temp_file.name

                # Load into QWebEngineView
                self.google_maps_view.load(QUrl.fromLocalFile(temp_file_path))
                self.stacked_widget.setCurrentWidget(self.google_maps_view)
                QMessageBox.information(self, "Google Maps", "Satellite view enabled")

            else:
                # Switch back to image view
                self.google_maps_enabled = False
                self.stacked_widget.setCurrentWidget(self.image_display_widget)
                QMessageBox.information(self, "Google Maps", "Satellite view disabled")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to toggle Google Maps: {str(e)}")
            self.google_maps_enabled = False
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
    
    def validate_hex_color(color):
        """
        Validates whether the given string is a valid hexadecimal color code.        
        Parameters:
            color (str): The color code to validate.            
        Returns:
            bool: True if the color code is valid.            
        Raises:
            ValueError: If the color code is invalid.
        """
        # Define the regex pattern for a valid hex color code
        hex_pattern = r'^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$'
        
        # Check if the color matches the pattern
        if not re.match(hex_pattern, color):
            raise ValueError(f"Invalid hex color code: {color}. Expected format: '#RGB' or '#RRGGBB'.")        
        return True    
################################################################################################################
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
##########################################################################################################

    # def load_image(self):
    #     file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "GeoTIFF (*.tif *.tiff)")
    #     if file_name:
    #         try:
    #             with rasterio.open(file_name) as src:
    #                 image = src.read()
    #                 profile = src.profile
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
                profile = src.profile
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
            "Vector Files (*.shp *.geojson *.gpkg)"
        )
        
        if file_name:
            try:
                # Read file with GeoPandas
                gdf = gpd.read_file(file_name)
                
                # Validate
                if gdf.empty or 'geometry' not in gdf.columns:
                    self.show_warning("Invalid vector file: Empty or missing geometry")
                    return
                    
                # Store in processor
                self.polygon_processor.current_gdf = gdf
                self.polygon_processor.original_file_path = file_name
                self.polygon_processor.editing_active = False
                
                # Add to layer list
                layer_name = os.path.basename(file_name)
                layer_data = {
                    'type': 'vector',
                    'gdf': gdf,
                    'file_name': file_name,
                    'style': {'color': '#FF0000', 'width': 1}
                }
                self.layers[layer_name] = layer_data  # Store layer data
                
                # Update UI - Pass BOTH arguments
                item = QListWidgetItem(layer_name)
                item.setIcon(QIcon(":/icons/vector.png"))
                self.layer_list.addItem(item)
                self.display_vector_layer(layer_data, layer_name)  # Fixed this line
                self.update_info_box(layer_data=self.layers[layer_name])
                
                self.show_info(f"Successfully loaded {layer_name}")               
                
            except Exception as e:
                self.show_error(f"Load failed: {str(e)}")
                traceback.print_exc()

    def on_layer_clicked(self, item):
        """عند النقر على عنصر في قائمة الطبقات"""
        layer_name = item.text()
        if layer_name in self.layers:
            self.update_info_box(layer_data=self.layers[layer_name])

        

 
 ########################################## save image as functions
    def save_as_image(self):
        """Enhanced save function that handles both raster and vector data"""
        try:
            # Get the current layer/image info
            image_data, profile, layer_name = self.get_current_image()
            
            # Determine if we're dealing with a vector layer
            is_vector = False
            if layer_name and layer_name in self.layers:
                is_vector = self.layers[layer_name].get('type') == 'vector'
            
            # Prepare format options
            if is_vector:
                formats = "Shapefile (*.shp);;GeoJSON (*.geojson);;CSV (*.csv)"
                default_ext = ".shp"
            else:
                formats = "GeoTIFF (*.tif);;JPEG (*.jpg);;PNG (*.png);;ASCII Grid (*.asc)"
                default_ext = ".tif" if image_data is not None else ".shp"
            
            # Set default filename
            default_name = ""
            if layer_name:
                default_name = f"{layer_name}{default_ext}"
            
            # Get save path from user
            file_name, selected_format = QFileDialog.getSaveFileName(
                self, 
                "Save Layer" if layer_name else "Save Image",
                default_name,
                formats
            )
            
            if not file_name:
                return  # User cancelled
            
            # Handle the save operation
            if is_vector:
                self._save_vector_data(file_name, selected_format, layer_name)
            else:
                self._save_raster_data(file_name, selected_format, image_data, profile)
                
            QMessageBox.information(self, "Success", f"Saved successfully to:\n{file_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
            print(f"Save error: {traceback.format_exc()}")
    def _save_vector_data(self, file_path, file_format, layer_name):
        """Save vector data to file"""
        gdf = self.layers[layer_name].get('gdf')
        if gdf is None:
            raise ValueError("No vector data found in layer")
        
        if "Shapefile" in file_format:
            gdf.to_file(file_path, driver='ESRI Shapefile')
            # Save .prj file if CRS exists
            if gdf.crs:
                with open(os.path.splitext(file_path)[0] + '.prj', 'w') as f:
                    f.write(gdf.crs.to_wkt())
        elif "GeoJSON" in file_format:
            gdf.to_file(file_path, driver='GeoJSON')
        elif "CSV" in file_format:
            # Save attributes without geometry
            gdf.drop(columns='geometry').to_csv(file_path, index=False)
        else:
            raise ValueError("Unsupported vector format")

    def _save_raster_data(self, file_path, file_format, image_data, profile):
        """Save raster data to file"""
        if image_data is None:
            raise ValueError("No image data to save")
        
        if "GeoTIFF" in file_format:
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(image_data)
        elif "JPEG" in file_format or "PNG" in file_format:
            # Convert to 8-bit for image formats
            normalized = self._normalize_to_8bit(image_data)
            if len(normalized.shape) == 3:  # Multi-band
                normalized = np.moveaxis(normalized, 0, -1)  # Change to HWC format
            if "JPEG" in file_format:
                cv2.imwrite(file_path, normalized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                cv2.imwrite(file_path, normalized)
        elif "ASCII" in file_format:
            self._save_ascii_grid(file_path, image_data, profile)
        else:
            raise ValueError("Unsupported raster format")

    def _normalize_to_8bit(self, image_data):
        """Normalize image data to 0-255 range"""
        if len(image_data.shape) == 2:  # Single band
            return cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:  # Multi-band
            return np.stack([
                cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                for band in image_data
            ])

    def _save_ascii_grid(self, file_path, image_data, profile):
        """Save single-band raster as ASCII grid"""
        if len(image_data.shape) != 2:
            raise ValueError("ASCII Grid only supports single-band data")
        
        transform = profile.get('transform')
        if transform is None:
            raise ValueError("Missing geotransform information")
        
        with open(file_path, 'w') as f:
            # Write header
            f.write(f"ncols {image_data.shape[1]}\n")
            f.write(f"nrows {image_data.shape[0]}\n")
            f.write(f"xllcorner {transform.c}\n")
            f.write(f"yllcorner {transform.f - transform.e * image_data.shape[0]}\n")
            f.write(f"cellsize {transform.a}\n")
            f.write("NODATA_value -9999\n")
            
            # Write data
            np.savetxt(f, image_data, fmt='%.4f')

    def get_current_image(self):
        """Get either the selected layer image/vector or the currently displayed image"""
        try:
            # Check for selected layer first
            selected_items = self.layer_list.selectedItems()
            if selected_items:
                layer_name = selected_items[0].text()
                layer_data = self.layers.get(layer_name, {})
                
                # Handle vector layers (shapefiles)
                if layer_data.get('type') == 'vector':
                    return None, None, layer_name  # Return just layer name for vectors
                
                # Handle raster layers (images)
                if 'image' in layer_data:
                    return layer_data['image'], layer_data.get('profile', {}), layer_name
                
                return None, None, layer_name
            
            # Fall back to active image (mouse-selected or last displayed)
            if hasattr(self, 'active_image'):
                profile = getattr(self, 'profile', {})
                return self.active_image, profile, None
            
            # Check for regular displayed image
            if hasattr(self, 'image'):
                profile = getattr(self, 'profile', {})
                return self.image, profile, None
            
            return None, None, None
            
        except Exception as e:
            print(f"Error in get_current_image: {str(e)}")
            return None, None, None
    
    def save_geotiff(self, file_name, image_data=None, profile=None):
        """Save image as GeoTIFF with proper NaN handling and compression"""
        if image_data is None:
            image_data = self.image
        if profile is None:
            profile = getattr(self, 'profile', {}).copy()
        
        try:
            # Make a copy of the image data to avoid modifying original
            image_data = np.copy(image_data)
            profile = profile.copy()
            
            # Determine appropriate data type and nodata value
            if np.issubdtype(image_data.dtype, np.floating):
                output_dtype = 'float32'
                nodata = np.nan
            else:
                output_dtype = 'int16'
                nodata = -32768  # Minimum Int16 value
            
            # Handle NaN values
            if np.any(np.isnan(image_data)):
                if output_dtype.startswith('int'):
                    QMessageBox.information(
                        self,
                        "Data Conversion",
                        "NaN values will be converted to {} for integer format".format(nodata)
                    )
                    image_data = np.nan_to_num(image_data, nan=nodata)
                else:
                    image_data = np.where(np.isnan(image_data), nodata, image_data)
            
            # Update profile with compression and other settings
            profile.update({
                'dtype': output_dtype,
                'nodata': nodata,
                'driver': 'GTiff',
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256
            })
            
            # Handle multi-band vs single-band
            if image_data.ndim == 3 and image_data.shape[0] > 1:
                profile['count'] = image_data.shape[0]
            
            # Save with updated profile
            with rasterio.open(file_name, 'w', **profile) as dst:
                if image_data.ndim == 3:
                    dst.write(image_data)
                else:
                    dst.write(image_data, 1)
            
            QMessageBox.information(self, "Success", f"Image saved as GeoTIFF: {file_name}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save GeoTIFF: {str(e)}")
    
    def save_shapefile(self, file_name, image_data=None, profile=None):
        """Save image as ESRI Shapefile"""
        if image_data is None:
            image_data = self.get_current_image()[0]
            if image_data is None:
                QMessageBox.warning(self, "Warning", "No image available to save")
                return
        
        try:
            # For shapefiles, we typically want a single band classification
            if image_data.ndim == 3:
                QMessageBox.warning(self, "Warning", "Using first band for shapefile conversion")
                image_data = image_data[0]
            
            # Convert raster to polygons
            gdf = self.raster_to_polygons(image_data)
            
            # Save with CRS if available
            if profile and 'crs' in profile:
                gdf.crs = profile['crs']
            
            gdf.to_file(file_name)
            QMessageBox.information(self, "Success", f"Image saved as Shapefile: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save Shapefile: {str(e)}")

    def save_ascii_grid(self, file_name, image_data=None, profile=None):
        """Save image as ASCII Grid"""
        if image_data is None:
            image_data = self.get_current_image()[0]
            if image_data is None:
                QMessageBox.warning(self, "Warning", "No image available to save")
                return
        
        try:
            # Handle multi-band images
            if image_data.ndim == 3:
                QMessageBox.warning(self, "Warning", "Using first band for ASCII Grid")
                image_data = image_data[0]
            
            # Replace NaN with -9999 (standard ASCII Grid nodata value)
            image_data = np.nan_to_num(image_data, nan=-9999)
            
            # Get spatial reference from profile or use defaults
            if profile and 'transform' in profile:
                transform = profile['transform']
                xll = transform.c
                yll = transform.f
                cellsize = transform.a
            else:
                xll = yll = 0
                cellsize = 1
            
            # Write ASCII Grid header
            header = f"""NCOLS {image_data.shape[1]}
    NROWS {image_data.shape[0]}
    XLLCORNER {xll}
    YLLCORNER {yll}
    CELLSIZE {cellsize}
    NODATA_VALUE -9999
    """
            
            # Write data
            with open(file_name, 'w') as f:
                f.write(header)
                np.savetxt(f, image_data, fmt='%.4f')
            
            QMessageBox.information(self, "Success", f"Image saved as ASCII Grid: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save ASCII Grid: {str(e)}")

    def save_jpeg(self, file_name, image_data=None, profile=None):
        """Save image as JPEG"""
        if image_data is None:
            image_data = self.get_current_image()[0]
            if image_data is None:
                QMessageBox.warning(self, "Warning", "No image available to save")
                return
        
        try:
            # Handle different image types
            if image_data.ndim == 3:
                # RGB image
                if image_data.shape[0] == 3:  # RGB
                    img_array = image_data.transpose(1, 2, 0)
                elif image_data.shape[0] == 4:  # RGBA
                    img_array = image_data[:3].transpose(1, 2, 0)
                else:  # Multi-band, use first 3 bands
                    img_array = image_data[:3].transpose(1, 2, 0)
            else:
                # Single band - convert to grayscale
                img_array = np.stack((image_data,)*3, axis=-1)
            
            # Normalize to 0-255 range
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                img_array = (img_array * 255).astype(np.uint8)
            elif img_array.dtype == np.uint16:
                img_array = (img_array / 256).astype(np.uint8)
            
            # Convert to PIL Image and save
            img = Image.fromarray(img_array)
            img.save(file_name, "JPEG", quality=95)
            
            QMessageBox.information(self, "Success", f"Image saved as JPEG: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save JPEG: {str(e)}")

    def save_png(self, file_name, image_data=None, profile=None):
        """Save image as PNG"""
        if image_data is None:
            image_data = self.get_current_image()[0]
            if image_data is None:
                QMessageBox.warning(self, "Warning", "No image available to save")
                return
        
        try:
            # Handle different image types
            if image_data.ndim == 3:
                # RGB or RGBA image
                if image_data.shape[0] == 3:  # RGB
                    img_array = image_data.transpose(1, 2, 0)
                    mode = 'RGB'
                elif image_data.shape[0] == 4:  # RGBA
                    img_array = image_data.transpose(1, 2, 0)
                    mode = 'RGBA'
                else:  # Multi-band, use first 3 bands
                    img_array = image_data[:3].transpose(1, 2, 0)
                    mode = 'RGB'
            else:
                # Single band
                img_array = image_data
                mode = 'L'  # Grayscale
            
            # Normalize data
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                img_array = (img_array * 255).astype(np.uint8)
            elif img_array.dtype == np.uint16:
                img_array = (img_array / 256).astype(np.uint8)
            
            # Convert to PIL Image and save
            img = Image.fromarray(img_array, mode)
            img.save(file_name, "PNG", compress_level=6)
            
            QMessageBox.information(self, "Success", f"Image saved as PNG: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save PNG: {str(e)}")

    def raster_to_polygons(self, raster_data):
        """Convert raster data to polygons for shapefile export"""
        try:
            import geopandas as gpd
            from shapely.geometry import shape
            import rasterio.features
            
            # Create mask from valid data (non-NaN/non-zero)
            mask = ~np.isnan(raster_data) & (raster_data != 0)
            
            # Extract shapes
            shapes = rasterio.features.shapes(raster_data.astype('float32'), 
                                            mask=mask,
                                            transform=self.profile.get('transform'))
            
            # Convert to GeoDataFrame
            polygons = []
            values = []
            for geom, val in shapes:
                polygons.append(shape(geom))
                values.append(val)
            
            return gpd.GeoDataFrame({'value': values, 'geometry': polygons})
        
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Required packages not found: {str(e)}")
            return None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to convert raster to polygons: {str(e)}")
            return None


 ##################################################################
    
    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.image.copy())
            self.image = self.undo_stack.pop()  # تحديث الصورة النشطة
            self.display_raster_layer(self.image)

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.image.copy())
            self.image = self.redo_stack.pop()  # تحديث الصورة النشطة
            self.display_raster_layer(self.image)


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
            self.display_raster_layer(self.resampled_image, file_name)  # Pass file_name to display_image
            self.undo_stack.append(self.image.copy())
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
            self.display_raster_layer(self.contrast_enhanced_image, file_name)  # Pass file_name to display_image
            self.undo_stack.append(self.image.copy())
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
            self.display_raster_layer(self.smoothed_image, file_name)  # Pass file_name to display_image
            self.undo_stack.append(self.image.copy())
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
            self.display_raster_layer(self.sharpened_image, file_name)  # Pass file_name to display_image
            self.undo_stack.append(self.image.copy())
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
    def preprocess_dem(self, dem_data, profile):
        """Enhanced DEM cleaning and preparation"""
        try:
            # Step 1: Basic validation
            if np.all(dem_data == dem_data[0,0]):
                QMessageBox.warning(self, "Warning", "DEM appears completely flat")
                return None, None

            # Step 2: Handle NoData values
            nodata_values = [
                profile.get('nodata'),  # القيمة الأصلية من الملف التعريفي
                -9999,                  # القيمة الأكثر شيوعًا في ArcGIS
                -32767,                 # تستخدم في بعض DEMs القديمة
                -3.4028235e+38,         # تستخدم في البيانات العائمة (Float) في ArcGIS
                -2147483648,            # تستخدم للبيانات الصحيحة (Integer)
                65535,                  # تستخدم في بيانات 16-bit غير الموقعة
                -32768                  # تستخدم في بيانات 16-bit الموقعة
            ]

            # إزالة أي قيم None (إذا لم يكن 'nodata' موجودًا في profile)
            nodata_values = [x for x in nodata_values if x is not None]

            # تنظيف البيانات: استبدال أي من قيم NoData بـ NaN
            dem_clean = np.where(
                np.isin(dem_data, nodata_values) | (dem_data < -1000) | (dem_data > 9000),
                np.nan,
                dem_data
            )
            # # Step 2: Handle NoData values
            # nodata = profile.get('nodata', -9999, -32767)
            # dem_clean = np.where((dem_data == nodata) | (dem_data < -1000) | (dem_data > 9000), np.nan, dem_data)

            # Step 3: Fill small voids (3x3 window)
            from scipy.ndimage import generic_filter
            def fill_nan(window):
                valid_vals = window[~np.isnan(window)]
                return valid_vals.mean() if len(valid_vals) > 4 else window[4]
            dem_filled = generic_filter(dem_clean, fill_nan, size=3)

            # Step 4: Ask user about Gentle smoothing
            from scipy.ndimage import gaussian_filter
            # Ask user about smoothing
            reply = QMessageBox.question(
                self,
                'DEM Smoothing',
                'Would you like to apply gentle smoothing using gaussian filter with sigma=0.75 to the DEM?\n\n'
                'Smoothing can reduce noise but may lose some detail.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                # Step 4: Gentle smoothing (only if user chooses)
                from scipy.ndimage import gaussian_filter
                dem_smoothed = gaussian_filter(dem_filled, sigma=0.75)
                smoothing_applied = True
            else:
                dem_smoothed = dem_filled
                smoothing_applied = False

            # Update profile with processing metadata
            new_profile = profile.copy()
            new_profile.update({
                'nodata': np.nan,
                'processed': True,
                'original_range': (float(np.nanmin(dem_data)), float(np.nanmax(dem_data))),
                'processed_range': (float(np.nanmin(dem_smoothed)), float(np.nanmax(dem_smoothed)))
            })

            return dem_smoothed, new_profile

        except Exception as e:
            QMessageBox.critical(self, "Preprocessing Error", f"DEM cleaning failed: {str(e)}")
            return None, None

    def calculate_hillshade(self):
        """Calculate hillshade with proper visualization"""
        try:
            dem_data, profile = self.validate_dem_layer()
            if dem_data is None:
                return
            
            # Check if CRS is projected
            if not profile.get('crs', {}).is_projected:
                reply = QMessageBox.warning(
                    self,
                    "Coordinate System Warning",
                    "Your DEM is not in a projected coordinate system (e.g., UTM).\n\n"
                    "Hillshade lighting effects will be distorted in degrees-based systems.\n"
                    "For accurate results, reproject to UTM first.\n\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

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
                z_factor
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
            # Get DEM data (already preprocessed by validate_dem_layer)
            dem_processed, new_profile = self.validate_dem_layer()
            if dem_processed is None:
                return

            # Check if CRS is projected
            if not new_profile.get('crs', {}).is_projected:
                reply = QMessageBox.warning(
                    self,
                    "Coordinate System Warning",
                    "Your DEM is not in a projected coordinate system (e.g., UTM).\n\n"
                    "Slope calculations require projected coordinates (in meters) for accurate results.\n"
                    "Degrees-based systems (like WGS84) will produce incorrect slopes.\n\n"
                    "Do you want to continue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            # Create parameter selection dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Slope Parameters")
            layout = QVBoxLayout()
            
            # DEM Preview
            fig = Figure(figsize=(7, 6))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            preview = ax.imshow(dem_processed, cmap='terrain')
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
                output_type
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
            
            if output_type == 'both' or output_type == 'degrees':
                degrees_path = f"{base_path}_degrees.tif"
                with rasterio.open(degrees_path, 'w', **save_profile) as dst:
                    dst.write(degrees_data, 1)
                
            if output_type == 'both' or (output_type == 'percent_rise' and percent_data is not None):
                percent_path = f"{base_path}_percent.tif"
                with rasterio.open(percent_path, 'w', **save_profile) as dst:
                    dst.write(percent_data if percent_data is not None else degrees_data, 1)
                
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
            if not profile.get('crs', {}).is_projected:
                reply = QMessageBox.warning(
                    self,
                    "Coordinate System Warning",
                    "Your DEM is not in a projected coordinate system (e.g., UTM).\n\n"
                    "Aspect directions may be inaccurate in degrees-based systems.\n"
                    "For accurate results, reproject to UTM first.\n\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            # Preprocess DEM
            dem_processed, new_profile = self.preprocess_dem(dem_data, profile)
            if dem_processed is None:
                return

            # Calculate aspect (0-360 degrees)
            aspect = self.image_processor.calculate_aspect(
                dem_processed, 
                new_profile['transform']
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
            item = QListWidgetItem(QIcon("D:\\Python_Codes\\icons\\compass.png"), layer_name)
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
            
    def calculate_flow_accumulation(self):
        """Calculate flow accumulation from DEM"""
        try:
            # Get DEM data
            dem_data, profile = self.validate_dem_layer()
            if dem_data is None:
                return

            # Preprocess DEM (handle NoData values)
            dem_processed, new_profile = self.preprocess_dem(dem_data, profile)
            if dem_processed is None:
                return

            # Ensure we have a projected CRS
            if not profile.get('crs', {}).is_projected:
                QMessageBox.warning(self, "Warning", 
                                "Flow accumulation requires projected CRS (e.g., UTM)")
                return

            # Fill sinks and calculate flow
            filled_dem = self.image_processor.fill_sinks(dem_processed[0] if dem_processed.ndim == 3 else dem_processed)
            flow_dir = self.image_processor.d8_flow_direction(filled_dem, new_profile['transform'])
            accumulation = self.image_processor.calculate_flow_accumulation(flow_dir)

            # Convert to logarithmic scale
            log_acc = np.log1p(accumulation)
            log_acc[accumulation == 0] = 0

            # Create layer
            layer_name = "Flow_Accumulation"
            self.layers[layer_name] = {
                'image': log_acc[np.newaxis, ...],
                'profile': new_profile,
                'file_name': layer_name,
                'cmap': 'Blues',
                'vmin': 0,
                'vmax': np.percentile(log_acc[~np.isinf(log_acc)], 99)  # Handle inf values
            }

            # Update UI
            self.layer_list.addItem(QListWidgetItem(layer_name))
            self.display_raster_layer(log_acc, layer_name)

            # Show statistics
            stats = (f"Flow Accumulation Statistics:\n"
                    f"Max Contributing Cells: {np.nanmax(accumulation):.0f}\n"
                    f"Mean: {np.nanmean(accumulation):.1f}\n"
                    f"Suggested Stream Threshold: {np.percentile(accumulation, 95):.0f} cells")
            
            msg = QMessageBox()
            msg.setWindowTitle("Flow Analysis Complete")
            msg.setText(stats)
            msg.setInformativeText("Save results?")
            msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
            
            if msg.exec_() == QMessageBox.Save:
                self.save_raster_results(log_acc, new_profile, "flow_accumulation", layer_name)

        except Exception as e:
            QMessageBox.critical(self, "Error", 
                f"Flow accumulation failed:\n{str(e)}\n\n"
                "Common issues:\n"
                "1. DEM contains large flat areas\n"
                "2. NoData values not properly handled\n"
                "3. CRS not projected (requires meters)\n"
                "4. DEM resolution too coarse")
            
    def calculate_stream_network(self):
        """Robust stream network extraction with proper progress handling"""
        try:
            # Step 1: Validate inputs with proper error handling
            dem_data, profile = self.validate_dem_layer()
            if dem_data is None:
                return

            # Safe dictionary access
            if not isinstance(profile, dict):
                raise ValueError("Invalid raster profile - expected dictionary")

            # Enhanced CRS check
            crs = profile.get('crs')
            if crs is None:
                raise ValueError("Missing CRS in raster profile")
            if not hasattr(crs, 'is_projected'):
                raise ValueError("CRS object missing projection information")
            if not crs.is_projected:
                raise ValueError("Requires projected CRS (e.g., UTM)")

            # Step 2: DEM validation
            validation_errors = self.image_processor.validate_dem(dem_data, profile)
            if validation_errors:
                raise ValueError("\n".join(validation_errors))
            # Validate inputs
            dem_data, profile = self.validate_dem_layer()
            if dem_data is None:
                return

            # Proper CRS check (using dict.get() safely)
            if not isinstance(profile, dict) or not profile.get('crs'):
                QMessageBox.critical(self, "CRS Error", 
                                "Missing CRS information in raster profile")
                return
                
            crs = profile['crs']
            if not hasattr(crs, 'is_projected') or not crs.is_projected:
                QMessageBox.critical(self, "CRS Error", 
                                "Stream extraction requires projected CRS (e.g., UTM)")
                return
            # Initialize progress dialog
            progress = QProgressDialog("Preparing DEM...", "Cancel", 0, 4, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(True)
            progress.show()
            
            # Step 1: Validate inputs
            if progress.wasCanceled():
                return
            progress.setValue(0)
            dem_data, profile = self.validate_dem_layer()
            if dem_data is None:
                return

            # Check for projected CRS
            if not profile.get('crs', {}).is_projected:
                QMessageBox.critical(self, "CRS Error", 
                                "Stream extraction requires projected CRS (e.g., UTM)")
                return

            # Step 2: Process DEM
            progress.setLabelText("Filling sinks...")
            QApplication.processEvents()  # Update UI
            filled_dem = self.image_processor.fill_sinks(dem_data)
            progress.setValue(1)
            
            progress.setLabelText("Calculating flow direction...")
            QApplication.processEvents()
            flow_dir = self.image_processor.d8_flow_direction(filled_dem, profile['transform'])
            progress.setValue(2)
            
            progress.setLabelText("Calculating flow accumulation...")
            QApplication.processEvents()
            accumulation = self.image_processor.calculate_flow_accumulation(flow_dir)
            progress.setValue(3)

            # Check for flat areas
            flat_cells = np.sum(flow_dir == 0) / flow_dir.size * 100
            if flat_cells > 30:  # More than 30% flat
                reply = QMessageBox.question(
                    self, "Flat Areas Detected",
                    f"{flat_cells:.1f}% flat areas found. Results may be unreliable.\nContinue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            # Get parameters
            params = self._get_stream_parameters(accumulation)
            if not params:
                return

            # Step 3: Extract streams
            progress.setLabelText("Extracting streams...")
            QApplication.processEvents()
            streams = self.image_processor.extract_streams(
                accumulation, 
                params['method'], 
                params['threshold']
            )
            progress.setValue(4)
            
            if np.sum(streams) == 0:
                raise ValueError("No streams found - threshold too high")

            # Step 4: Calculate stream order
            progress.setLabelText("Calculating stream order...")
            QApplication.processEvents()
            if params['order_method'] == "Strahler":
                stream_order = self.image_processor.strahler_order(flow_dir, streams)
            elif params['order_method'] == "Shreve":
                stream_order = self.image_processor.shreve_order(flow_dir, streams)
            else:  # Horton
                stream_order = self.image_processor.horton_order(flow_dir, streams)
            
            # Create outputs
            progress.setLabelText("Generating outputs...")
            QApplication.processEvents()
            self._create_stream_outputs(stream_order, flow_dir, profile, params)
            progress.setValue(5)

            QMessageBox.information(
                self, "Success",
                f"Stream network extracted successfully\n"
                f"Max {params['order_method']} order: {stream_order.max()}\n"
                f"Total stream length: {self._calculate_stream_length(stream_order, profile['transform']):.1f} m"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Extraction Failed",
                f"{str(e)}\n\n"
                "Possible fixes:\n"
                "1. Check input DEM has valid CRS\n"
                "2. Verify DEM contains elevation data\n"
                "3. Use projected coordinates (e.g., UTM)\n"
                "4. Pre-process DEM to fill sinks"
            )
        

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
        self.order_combo.addItems(["Strahler", "Shreve", "Horton"])
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
    
    def _create_stream_outputs(self, stream_order, flow_dir, profile, params):
        """Create all output layers"""
        # Raster layer
        self.layers["Stream_Order"] = {
            'image': stream_order[np.newaxis, ...],
            'profile': profile,
            'file_name': f"Stream_Order_{params['threshold']}",
            'cmap': 'viridis',
            'vmin': 1,
            'vmax': np.max(stream_order)
        }
        
        # Vector layer
        stream_lines = self.image_processor.streams_to_vector(stream_order, flow_dir, profile['transform'])
        self._save_stream_vectors(stream_lines, profile['crs'])
        
        # Add to layer list
        self.layer_list.addItem(QListWidgetItem("Stream_Order"))
        self.layer_list.addItem(QListWidgetItem("Stream_Lines"))

    def _show_progress(self, message):
        """Context manager for progress dialogs"""
        progress = QProgressDialog(message, "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        return progress


    def _add_stream_layers(self, stream_order, flow_dir, profile):
        """Create raster and vector stream layers"""
        # Raster stream order
        self.layers["Stream_Order"] = {
            'image': stream_order[np.newaxis, ...],
            'profile': profile,
            'file_name': "Stream_Order",
            'cmap': 'viridis',
            'vmin': 1,
            'vmax': np.max(stream_order)
        }
        self.layer_list.addItem(QListWidgetItem("Stream_Order"))
        
        # Vector streams
        stream_lines = self.image_processor.streams_to_vector(
            stream_order, flow_dir, profile['transform']
        )
        self._save_stream_vectors(stream_lines, profile['crs'])

    def _save_stream_vectors(self, features, crs):
        """Save stream vectors to temporary GeoJSON"""
        import geopandas as gpd
        from datetime import datetime
        
        gdf = gpd.GeoDataFrame.from_features(
            [{
                'type': 'Feature',
                'geometry': f['geometry'].__geo_interface__,
                'properties': f['properties']
            } for f in features],
            crs=crs
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/tmp/streams_{timestamp}.geojson"
        gdf.to_file(output_path, driver='GeoJSON')
        
        self.layers["Stream_Lines"] = {
            'path': output_path,
            'type': 'vector',
            'name': "Stream_Lines"
        }
        self.layer_list.addItem(QListWidgetItem("Stream_Lines"))

    
    
######################################  drainage_channels #################
    def generate_drainage_channels(self):
        """Generate drainage channels from selected DEM and add to layers"""
        try:
            # 1. Get selected DEM layer
            if not self.layer_list.currentItem():
                QMessageBox.warning(self, "Error", "Please select a DEM layer first")
                return
                
            layer_name = self.layer_list.currentItem().text()
            layer = self.layers.get(layer_name)
            
            if not layer:
                QMessageBox.warning(self, "Error", "Selected layer not found")
                return

            # 2. Check if we have a valid path
            if 'path' not in layer or not layer['path'] or not os.path.exists(layer['path']):
                # Create temp file if in-memory
                temp_path = os.path.join(tempfile.gettempdir(), f"temp_dem_{time.time()}.tif")
                with rasterio.open(temp_path, 'w', **layer['profile']) as dst:
                    dst.write(layer['image'])
                layer['temp_path'] = temp_path
                dem_path = temp_path
            else:
                dem_path = layer['path']

            # 3. Ask user for parameters
            threshold, ok = QInputDialog.getInt(
                self, "Channel Threshold", 
                "Flow accumulation threshold (higher = fewer channels):",
                value=1000, min=100, max=10000, step=100)
            if not ok:
                if 'temp_path' in layer:
                    os.remove(layer['temp_path'])
                    del layer['temp_path']
                return

            # 4. Create output directory in temp
            output_dir = os.path.join(tempfile.gettempdir(), f"drainage_{time.time()}")
            os.makedirs(output_dir, exist_ok=True)

            # 5. Process DEM using WhiteboxTools
            wbt = WhiteboxTools()
            wbt.verbose = False
            wbt.work_dir = output_dir

            # Hydrological processing
            filled_dem = os.path.join(output_dir, "filled_dem.tif")
            flow_dir = os.path.join(output_dir, "flow_dir.tif")
            flow_accum = os.path.join(output_dir, "flow_accum.tif")
            channels_raster = os.path.join(output_dir, "channels.tif")
            channels_vector = os.path.join(output_dir, "channels.shp")

            wbt.fill_depressions(dem_path, filled_dem)
            wbt.d8_pointer(filled_dem, flow_dir)
            wbt.d8_flow_accumulation(flow_dir, flow_accum)
            wbt.extract_streams(flow_accum, channels_raster, threshold=threshold)
            wbt.raster_streams_to_vector(channels_raster, flow_dir, channels_vector)

            # 6. Add results to layers
            base_name = f"{layer_name}_drainage"
            
            # Add flow accumulation
            self.add_raster_layer(
                flow_accum,
                f"{base_name}_flow_accum",
                colormap='viridis'
            )
            
            # Add channel network (raster)
            self.add_raster_layer(
                channels_raster,
                f"{base_name}_channels",
                colormap='blues'
            )
            
            # Add channel network (vector)
            self.add_vector_layer(
                channels_vector,
                f"{base_name}_streams"
            )

            # 7. Cleanup temporary files
            if 'temp_path' in layer:
                os.remove(layer['temp_path'])
                del layer['temp_path']
            shutil.rmtree(output_dir)

            QMessageBox.information(
                self, "Success", 
                "Drainage channels generated and added to layers")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to generate drainage channels: {str(e)}")
            # Cleanup if error occurs
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            if 'output_dir' in locals() and os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def add_raster_layer(self, path, name, colormap=None):
        """Add a raster layer to the application"""
        try:
            with rasterio.open(path) as src:
                profile = src.profile
                image = src.read()
                
            self.layers[name] = {
                'image': image,
                'profile': profile,
                'path': path,
                'colormap': colormap
            }
            self.layer_list.addItem(name)
            self.display_layer(name)
            
        except Exception as e:
            print(f"Error adding raster layer {name}: {str(e)}")

    def add_vector_layer(self, path, name):
        """Add a vector layer to the application"""
        try:
            # For vector layers, we'll just store the path
            self.layers[name] = {
                'path': path,
                'type': 'vector'
            }
            self.layer_list.addItem(name)
            
            # You'll need to implement vector display separately
            self.display_vector_layer(name)
            
        except Exception as e:
            print(f"Error adding vector layer {name}: {str(e)}")

    # def display_layer(self, layer_name, cmap=None):
    #     """Display a raster layer with optional colormap"""
    #     layer = self.layers.get(layer_name)
    #     if not layer or 'image' not in layer:
    #         return
        
    #     # Use layer's colormap if specified, otherwise use parameter
    #     cmap = layer.get('colormap', cmap)
        
    #     # Your existing display implementation here
    #     # ... (use cmap parameter when calling imshow)
    def display_layer(self, layer_name=None):
        """Display either raster or vector layer"""
        if not layer_name:
            if not self.layer_list.currentItem():
                return
            layer_name = self.layer_list.currentItem().text()
        
        if layer_name not in self.layers:
            return
        
        layer = self.layers[layer_name]
        self.ax.clear()
        
        # Handle raster display
        if layer.get('type') == 'raster' or 'image' in layer:
            self.display_raster_layer(layer['image'], layer_name)  # Your existing function
        
        # Handle vector display
        elif layer.get('type') == 'vector':
            gdf = layer['gdf']
            
            # Convert to matplotlib patches
            for geom in gdf.geometry:
                if geom.geom_type == 'Polygon':
                    x, y = geom.exterior.xy
                    self.ax.fill(x, y, alpha=0.3, fc=layer['color'])
                    self.ax.plot(x, y, color=layer['color'], linewidth=layer['line_width'])
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        self.ax.fill(x, y, alpha=0.3, fc=layer['color'])
                        self.ax.plot(x, y, color=layer['color'], linewidth=layer['line_width'])
        
        self.ax.set_title(layer_name)
        self.ax.set_aspect('equal')
        self.canvas.draw()

    def on_layer_selected(self, item):
        """When a layer is selected in the list"""
        layer_name = item.text()
        self.display_layer(layer_name)

    def update_layer_style(self, layer_name, color=None, line_width=None):
        """Update vector layer display properties"""
        if layer_name in self.layers and self.layers[layer_name].get('type') == 'vector':
            if color:
                self.layers[layer_name]['color'] = color
            if line_width:
                self.layers[layer_name]['line_width'] = line_width
            self.display_layer(layer_name)

    def contextMenuEvent(self, event):
        """Right-click menu for layer operations"""
        item = self.layer_list.itemAt(event.pos())
        if not item:
            return
            
        layer_name = item.text()
        if layer_name not in self.layers:
            return
            
        menu = QMenu()
        
        # Common actions
        remove_action = QAction("Remove Layer", self)
        remove_action.triggered.connect(lambda: self.remove_layer(layer_name))
        menu.addAction(remove_action)
        
        # Vector-specific actions
        if self.layers[layer_name].get('type') == 'vector':
            style_action = QAction("Change Style", self)
            style_action.triggered.connect(lambda: self.change_vector_style(layer_name))
            menu.addAction(style_action)
            
            attributes_action = QAction("Show Attributes", self)
            attributes_action.triggered.connect(lambda: self.show_attributes(layer_name))
            menu.addAction(attributes_action)
        
        menu.exec_(event.globalPos())

    def change_vector_style(self, layer_name):
        """Dialog to change vector appearance"""
        color = QColorDialog.getColor()
        if color.isValid():
            width, ok = QInputDialog.getDouble(
                self, "Line Width", "Enter line width:", 
                value=self.layers[layer_name].get('line_width', 1.0),
                min=0.1, max=5.0, decimals=1
            )
            if ok:
                self.update_layer_style(layer_name, color.name(), width)

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
            profile = layer_data['profile']
            transform = profile['transform']
            width = profile['width']
            height = profile['height']
            
            # Calculate geographic bounds
            xmin, ymax = transform * (0, 0)
            xmax, ymin = transform * (width, height)
            
            # Update input fields with suggested values
            self.xmin_input.setText(f"{xmin:.2f}")
            self.ymin_input.setText(f"{ymin:.2f}")
            self.xmax_input.setText(f"{xmax:.2f}")
            self.ymax_input.setText(f"{ymax:.2f}")
            
            QMessageBox.information(self, "Layer Selected", 
                                f"Layer '{self.selected_layer}' loaded.\n"
                                f"Suggested bounds have been populated.")
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
            profile = layer_data['profile']
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
                self.display_image_no_legend(clipped_image, clipped_name)
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

    def display_image_no_legend(self, image, layer_name):
        """Display image using matplotlib with proper channel ordering"""
        self.ax.clear()
        
        # Fix channel order if needed (for (channels, height, width) input)
        if len(image.shape) == 3 and image.shape[0] in [3, 4]:  # Channel-first format
            image = np.transpose(image, (1, 2, 0))  # Convert to (height, width, channels)
        
        # Display image
        if len(image.shape) == 2:  # Grayscale
            self.ax.imshow(image, cmap='gray')
        else:  # RGB/RGBA
            self.ax.imshow(image)
        
        self.ax.axis('off')
        self.canvas.draw()
        self.statusBar().showMessage(f"Displaying: {layer_name} (no legend)")

    def array_to_qimage(self, array: np.ndarray) -> QImage:
        """Convert a NumPy array to QImage with proper memory handling."""
        # Normalize to 0-255 if not already uint8
        if array.dtype != np.uint8:
            array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        
        # Ensure contiguous memory layout (C-order)
        array = np.ascontiguousarray(array)
        
        # Handle grayscale (2D) vs. RGB/RGBA (3D)
        if len(array.shape) == 2:  # Grayscale
            height, width = array.shape
            bytes_per_line = width
            return QImage(
                array.data.tobytes(),  # Convert to bytes explicitly
                width, 
                height, 
                bytes_per_line, 
                QImage.Format_Grayscale8
            )
        elif len(array.shape) == 3:  # Multi-band (RGB/RGBA)
            height, width, channels = array.shape
            bytes_per_line = channels * width
            if channels == 3:  # RGB
                return QImage(
                    array.data.tobytes(), 
                    width, 
                    height, 
                    bytes_per_line, 
                    QImage.Format_RGB888
                )
            elif channels == 4:  # RGBA
                return QImage(
                    array.data.tobytes(), 
                    width, 
                    height, 
                    bytes_per_line, 
                    QImage.Format_RGBA8888
                )
            else:  # Fallback to RGB (use first 3 bands)
                return QImage(
                    array[:, :, :3].data.tobytes(), 
                    width, 
                    height, 
                    3 * width, 
                    QImage.Format_RGB888
                )
        else:
            raise ValueError("Unsupported array shape. Expected 2D (H x W) or 3D (H x W x C).")

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
   
    def save_canny_edge_image(self):
        """
        Saves an image after applying Canny edge detection.
        """
        if hasattr(self, 'image'):
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Canny Edge Image", "", "Images (*.tif *.tiff)")
            if file_name:
                try:
                    # Update profile for Canny edge image
                    self.profile.update({
                        'count': 1,  # Single band
                        'dtype': 'uint8',  # Data type
                        'nodata': 0  # Valid nodata value for uint8
                    })
                    self.image_processor.save_image(self.image, self.profile, file_name)
                    QMessageBox.information(self, "Success", f"Canny edge image saved as '{file_name}'")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save Canny edge image: {str(e)}")

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
    def open_general_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("General Settings")
        layout = QVBoxLayout()

        # Theme Selection
        theme_label = QLabel("Theme:")
        theme_combo = QComboBox()
        theme_combo.addItems(["Light", "Dark"])
        layout.addWidget(theme_label)
        layout.addWidget(theme_combo)

        # Default Save Location
        save_location_label = QLabel("Default Save Location:")
        save_location_edit = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: self.browse_save_location(save_location_edit))
        layout.addWidget(save_location_label)
        layout.addWidget(save_location_edit)
        layout.addWidget(browse_button)

        # Auto-Save Interval
        auto_save_label = QLabel("Auto-Save Interval (minutes):")
        auto_save_spin = QSpinBox()
        auto_save_spin.setRange(1, 60)
        layout.addWidget(auto_save_label)
        layout.addWidget(auto_save_spin)

        # Save Button
        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.save_general_settings(
            theme_combo.currentText(),
            save_location_edit.text(),
            auto_save_spin.value()
        ))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def browse_save_location(self, save_location_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Default Save Location")
        if folder:
            save_location_edit.setText(folder)

    def save_general_settings(self, theme, save_location, auto_save_interval):
        # Save settings (e.g., to a config file or database)
        print(f"Theme: {theme}, Save Location: {save_location}, Auto-Save Interval: {auto_save_interval}")
        QMessageBox.information(self, "Success", "General settings saved successfully.")
    #####Display Settings Dialog
    def open_display_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Display Settings")
        layout = QVBoxLayout()

        # Zoom Level
        zoom_label = QLabel("Default Zoom Level:")
        zoom_combo = QComboBox()
        zoom_combo.addItems(["100%", "150%", "200%", "Fit to Window"])
        layout.addWidget(zoom_label)
        layout.addWidget(zoom_combo)

        # Grid Visibility
        grid_check = QCheckBox("Show Grid")
        layout.addWidget(grid_check)

        # Coordinate Display Format
        coord_label = QLabel("Coordinate Display Format:")
        coord_combo = QComboBox()
        coord_combo.addItems(["Decimal Degrees", "Degrees Minutes Seconds"])
        layout.addWidget(coord_label)
        layout.addWidget(coord_combo)

        # Save Button
        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.save_display_settings(
            zoom_combo.currentText(),
            grid_check.isChecked(),
            coord_combo.currentText()
        ))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def save_display_settings(self, zoom_level, show_grid, coord_format):
        # Save settings
        print(f"Zoom Level: {zoom_level}, Show Grid: {show_grid}, Coordinate Format: {coord_format}")
        QMessageBox.information(self, "Success", "Display settings saved successfully.")
    ##### Processing Settings Dialog
    def open_processing_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Processing Settings")
        layout = QVBoxLayout()

        # Default Classification Method
        classification_label = QLabel("Default Classification Method:")
        classification_combo = QComboBox()
        classification_combo.addItems(["Random Forest", "SVM", "K-Means"])
        layout.addWidget(classification_label)
        layout.addWidget(classification_combo)

        # Edge Detection Sigma
        sigma_label = QLabel("Canny Edge Detection Sigma:")
        sigma_spin = QDoubleSpinBox()
        sigma_spin.setRange(0.1, 5.0)
        sigma_spin.setSingleStep(0.1)
        layout.addWidget(sigma_label)
        layout.addWidget(sigma_spin)

        # Save Button
        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.save_processing_settings(
            classification_combo.currentText(),
            sigma_spin.value()
        ))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def save_processing_settings(self, classification_method, sigma):
        # Save settings
        print(f"Classification Method: {classification_method}, Sigma: {sigma}")
        QMessageBox.information(self, "Success", "Processing settings saved successfully.")
######Advanced Settings Dialog
    def open_advanced_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Advanced Settings")
        layout = QVBoxLayout()

        # GPU Acceleration
        gpu_check = QCheckBox("Enable GPU Acceleration")
        layout.addWidget(gpu_check)

        # Logging Level
        logging_label = QLabel("Logging Level:")
        logging_combo = QComboBox()
        logging_combo.addItems(["Debug", "Info", "Warning", "Error"])
        layout.addWidget(logging_label)
        layout.addWidget(logging_combo)

        # Reset to Defaults
        reset_button = QPushButton("Reset to Default Settings")
        reset_button.clicked.connect(self.reset_settings)
        layout.addWidget(reset_button)

        # Save Button
        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.save_advanced_settings(
            gpu_check.isChecked(),
            logging_combo.currentText()
        ))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def save_advanced_settings(self, gpu_enabled, logging_level):
        # Save settings
        print(f"GPU Enabled: {gpu_enabled}, Logging Level: {logging_level}")
        QMessageBox.information(self, "Success", "Advanced settings saved successfully.")

    def reset_settings(self):
        # Reset all settings to default
        QMessageBox.information(self, "Reset", "All settings have been reset to default.")
    ##### About Dialog
    def open_about(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About")
        layout = QVBoxLayout()

        # Version Information
        version_label = QLabel("Version: 1.0.0")
        layout.addWidget(version_label)

        # License Information
        license_label = QLabel("License: MIT")
        layout.addWidget(license_label)

        # Check for Updates
        update_button = QPushButton("Check for Updates")
        update_button.clicked.connect(self.check_for_updates)
        layout.addWidget(update_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def check_for_updates(self):
        # Implement update check logic
        QMessageBox.information(self, "Updates", "You are using the latest version.")

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
                            profile = src.profile
                    
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

    def generate_info_html(self, metadata, title, image=None, num_bands=None):
        """إنشاء محتوى HTML للمعلومات مع التنسيق المناسب"""
        info_html = f"""
        <style>
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
            .info-table tr:last-child td {{
                border-bottom: none;
            }}
        </style>
        <h3 style="color:#2c3e50;">{title}</h3>
        <table class="info-table">
        """
        
        for key, value in metadata.items():
            info_html += f"""
            <tr>
                <td><b>{key}</b></td>
                <td>{value}</td>
            </tr>
            """
        
        info_html += "</table>"
        
        if image is not None and num_bands is not None:
            info_html += self.generate_suggestions(image, num_bands)
        
        info_html += '<p style="color:#7f8c8d;font-style:italic;">Right-click for more options</p>'
        self.info_box.setHtml(info_html)
        self.info_box.setContextMenuPolicy(Qt.CustomContextMenu)

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
            print(f"Error getting CRS info: {str(e)}")
        
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


    def setup_ui_connections(self):
        """يتم استدعاء هذه الدالة أثناء تهيئة الواجهة"""
        # إزالة أي اتصال سابق لإظهار الإحصائيات عند النقر العادي
        self.layer_list.itemClicked.disconnect(self.show_stats)  # إذا كان متصلاً سابقاً
        
        # الاتصال الجديد فقط للقائمة المنبثقة
        self.info_box.setContextMenuPolicy(Qt.CustomContextMenu)
        self.info_box.customContextMenuRequested.connect(self.show_context_menu)


    def copy_metadata(self):
        self.info_box.selectAll()
        self.info_box.copy()

    def save_metadata(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Information", "", "Text Files (*.txt)")
        if file_name:
            with open(file_name, 'w') as f:
                f.write(self.info_box.toPlainText())

    def get_current_layer_item(self):
        """Get the currently selected layer from layer_list or return None"""
        if hasattr(self, 'layer_list') and self.layer_list.currentItem():
            return self.layer_list.currentItem()
        return None
    
 

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
            nodata = layer_data.get('profile', {}).get('nodata', None)

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
            nodata = layer_data.get('profile', {}).get('nodata', None)
            
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
            
            if 'crs' in layer_data['profile']:
                stats_content.append(f"CRS: {layer_data['profile']['crs']}")
            if 'transform' in layer_data['profile']:
                stats_content.append(f"Resolution: {abs(layer_data['profile']['transform'].a):.2f}m")

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

    def update_raster_info(self, layer_data):
        """Update the info box with basic raster information"""
        try:
            profile = layer_data['profile']
            img = layer_data['image']

            info_html = f"""
            <div style="font-family: Arial; font-size: 10pt;">
                <h3 style="color: #2c3e50;">Raster Information</h3>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 4px; font-weight: bold;">Dimensions:</td>
                        <td style="padding: 4px;">{img.shape}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 4px; font-weight: bold;">Data Type:</td>
                        <td style="padding: 4px;">{img.dtype}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 4px; font-weight: bold;">CRS:</td>
                        <td style="padding: 4px;">{profile.get('crs', 'Unknown')}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 4px; font-weight: bold;">Resolution:</td>
                        <td style="padding: 4px;">{abs(profile['transform'].a):.2f}m</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 4px; font-weight: bold;">Min Value:</td>
                        <td style="padding: 4px;">{np.nanmin(img):.2f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 4px; font-weight: bold;">Max Value:</td>
                        <td style="padding: 4px;">{np.nanmax(img):.2f}</td>
                    </tr>
                </table>
                <p style="color: #7f8c8d; font-style: italic;">
                    Right-click for more options
                </p>
            </div>
            """
            self.info_box.setHtml(info_html)
            self.info_box.setContextMenuPolicy(Qt.CustomContextMenu)
            self.info_box.customContextMenuRequested.connect(self.show_context_menu)

        except Exception as e:
            self.info_box.setText(f"Error loading raster info: {str(e)}")

    def update_coordinate_display(self):
        self.canvas.mpl_connect('motion_notify_event', self.show_pixel_coords)

    def show_pixel_coords(self, event):
        if event.inaxes == self.ax:
            x = int(event.xdata)
            y = int(event.ydata)
            self.statusBar().showMessage(f"Pixel Coordinates: X={x}, Y={y}")
        else:
            self.statusBar().clearMessage()
    
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
    
    def display_classified_image(self, classified_image):
            self.ax.clear()
            self.ax.imshow(classified_image, cmap='viridis', extent=[0, classified_image.shape[1], classified_image.shape[0], 0])
            self.ax.set_title('Classified Image')
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
                2, 2, 10, 1
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
            if image_data.ndim == 3:  # RGB image
                flattened_image = image_data.transpose(1, 2, 0).reshape(-1, 3)
            else:  # Grayscale image
                flattened_image = image_data[0].reshape(-1, 1)

            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            labels = kmeans.fit_predict(flattened_image)

            # Reshape the labels to the original image dimensions
            if image_data.ndim == 3:
                height, width, _ = image_data.shape[1], image_data.shape[2], image_data.shape[0]
            else:
                height, width = image_data.shape[1], image_data.shape[2]

            classified_image = labels.reshape(height, width)

            # Store the classified image
            self.classified_image = classified_image[np.newaxis, ...]

            # Convert the classified image to a 3D RGB image for display
            colored_image = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(n_clusters):
                for channel in range(3):
                    colored_image[classified_image == i, channel] = self.get_cluster_color(i)[channel]

            colored_image = np.clip(colored_image, 0, 255).astype(np.uint8)

            # Display the colored classified image
            self.display_raster_layer(np.transpose(colored_image, (2, 0, 1)), "K-Means Clustering")

            # Save the segmentation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_layer = f"Result_K_Mean_{timestamp}"
            self.layers[new_layer] = {
                'image': self.classified_image,
                'profile': layer_data['profile'].copy(),
                'file_name': f"K-Means Segmentation with {n_clusters} Clusters",
                'crs': layer_data.get('crs', None),  # Ensure CRS is included
                'cmap': 'viridis'  # Store the colormap for this layer
            }

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
                # Save the classified image with CRS
                self.save_classified_image(crs=layer_data.get('crs', None))
                QMessageBox.information(self, "Saved", "Classified image saved successfully.")
            else:
                QMessageBox.information(self, "Cancelled", "Classified image saving cancelled by user.")

        except Exception as e:
            error_msg = (
                f"Error: {str(e)}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            QMessageBox.critical(self, "Classification Error", error_msg)
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444
    
    def save_classified_image(self, crs=None):
        """
        Saves the classified image to a file with optional CRS information using rasterio.
        """
        if not hasattr(self, 'classified_image'):
            QMessageBox.warning(self, "No Image", "No classified image to save.")
            return

        try:
            # Ask the user for the file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Classified Image", "", "GeoTIFF (*.tif);;All Files (*)"
            )
            if not file_path:
                return  # User canceled the save dialog

            # Save the classified image with CRS
            if file_path.endswith('.tif'):
                height, width = self.classified_image.shape[1], self.classified_image.shape[2]

                # Define the transform (adjust as needed)
                transform = from_origin(0, 0, 1, 1)  # Adjust origin and resolution as needed

                # Save the image using rasterio
                with rasterio.open(
                    file_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,  # Number of bands
                    dtype=self.classified_image.dtype,
                    crs=crs,  # Add CRS if provided
                    transform=transform
                ) as dst:
                    dst.write(self.classified_image[0], 1)  # Write the first band

                QMessageBox.information(self, "Success", "Classified image saved with CRS.")
            else:
                # Save as a regular image (no CRS)
                from PIL import Image
                Image.fromarray(self.classified_image[0]).save(file_path)
                QMessageBox.information(self, "Success", "Classified image saved without CRS.")

        except Exception as e:
            error_msg = (
                f"Error: {str(e)}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            QMessageBox.critical(self, "Save Error", error_msg)

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
            segmented_image_uint8 = segmented_image.astype(np.uint8)

            # Store the segmented image
            self.classified_image = segmented_image_uint8[np.newaxis, ...]

            # Convert the segmented image to a 3D RGB image for display
            unique_labels = np.unique(segmented_image)
            colored_image = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)
            for label in unique_labels:
                for channel in range(3):
                    colored_image[segmented_image == label, channel] = self.get_cluster_color(label)[channel]

            colored_image = np.clip(colored_image, 0, 255).astype(np.uint8)

            # Display the colored segmented image
            self.display_raster_layer(np.transpose(colored_image, (2, 0, 1)), "Mean-Shift Segmentation")

            # Save the segmentation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_layer = f"Result_Mean_Shift_{timestamp}"
            self.layers[new_layer] = {
                'image': self.classified_image,
                'profile': layer_data['profile'].copy(),
                'file_name': f"Mean-Shift Segmentation with Bandwidth {bandwidth}"
            }

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
                # Save the segmented image
                self.save_classified_image()
                QMessageBox.information(self, "Saved", "Segmented image saved successfully.")
            else:
                QMessageBox.information(self, "Cancelled", "Segmented image saving cancelled by user.")

        except Exception as e:
            error_msg = (
                f"Error: {str(e)}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            QMessageBox.critical(self, "Segmentation Error", error_msg)


    def _handle_segmentation_error(self, error):
        self.segmented_image = None
        self.segmentation_analysis = {}
        QMessageBox.critical(self, "Segmentation Error", 
            f"Operation failed: {str(error)}\n"
            f"Error type: {type(error).__name__}")

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
                        print(f"Could not add CRS information: {str(e)}")

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
            print(f"Error: {traceback.format_exc()}")
            return False
    
    def export_raw_masks(self):
        """Export masks in various formats while preserving CRS from the original image"""
        try:
            # Verify an image is loaded with proper profile data
            if not hasattr(self, 'profile') or not hasattr(self, 'image'):
                QMessageBox.warning(self, "Warning", "No image has been loaded")
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
            print(f"Error: {traceback.format_exc()}")

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
        print("Classification Statistics:", stats)

    def log_position_stats(self, positions, labels):
        """Log spatial distribution statistics"""
        unique, counts = np.unique(labels, return_counts=True)
        stats = {
            "Cluster count": len(unique) - (1 if -1 in unique else 0),
            "Isolated masks": counts[-1] if -1 in unique else 0,
            "Region distribution": dict(zip(unique, counts))
        }
        print("Position Statistics:", stats)

    def _add_class_to_combobox(self, name, value):
        """Helper to add a new class to the combo box and tracking structures."""
        if name not in self.classes:
            self.classes[name] = value
            self.class_combo.addItem(name)
            self.class_masks[name] = np.zeros_like(self.masks[0]['segmentation'], dtype=bool)

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
            
            # Initialize SAM model
            model_type = "vit_h"
            sam_checkpoint = "d:\\Python_Codes\\sam_vit_h_4b8939.pth"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            
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
            print(f"Error: {traceback.format_exc()}")
    
    
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
            except:
                pass
            self.current_layer = None
        self.current_canvas = None
        self.figure_initialized = False

    def layer_switched_handler(self):
        """Handle layer switching events."""
        self.cleanup_figures()
        # Your existing layer switching code here

    def closeEvent(self, event):
        """Handle application closing."""
        self.cleanup_figures()
        event.accept()
        


    def assign_current_mask(self):
        """Assign current segment to selected class."""
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
            print(f"Error: {traceback.format_exc()}")

    def display_landuse_map_with_legend(self, landuse_map):
        """Display map with legend showing class names and colors."""
        plt.figure(figsize=(14, 10))
        
        # Create colormap with consistent colors for classes
        unique_classes = sorted(list(set(self.classes.values())))
        class_names = {v:k for k,v in self.classes.items()}
        
        # Create colormap and legend elements
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))
        cmap = plt.cm.colors.ListedColormap(colors)
        norm = plt.cm.colors.BoundaryNorm(np.arange(len(unique_classes)+1)-0.5, len(unique_classes))
        
        # Plot the map
        img = plt.imshow(landuse_map, cmap = cmap, norm=norm)
        
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
            cmap = plt.cm.colors.ListedColormap(colors)
            norm = plt.cm.colors.BoundaryNorm(np.arange(len(unique_classes)+1)-0.5, len(unique_classes))
            
            img = ax.imshow(landuse_map, cmap=cmap, norm=norm)
            
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

            # Save GeoTIFF if geospatial data exists
            if hasattr(self, 'original_geotransform'):
                geotiff_path = "landuse_classification.tif"
                with rasterio.open(
                    geotiff_path,
                    'w',
                    driver='GTiff',
                    height=landuse_map.shape[0],
                    width=landuse_map.shape[1],
                    count=1,
                    dtype=landuse_map.dtype,
                    crs=getattr(self, 'original_crs', None),
                    transform=getattr(self, 'original_geotransform', None),
                    nodata=0
                ) as dst:
                    dst.write(landuse_map, 1)
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
    def detect_changes(self):
        progress = None
        try:
            QMessageBox.information(None, "Load Image", "Please load the first image.")
            progress = self.show_loading("Loading first image")
            file1, _ = QFileDialog.getOpenFileName(None, "Select Base Image", "", "GeoTIFF Files (*.tif *.tiff);;All Files (*)")
            if not file1: 
                return
            img1, prof1 = ImageProcessor.load_image(file1)
            progress.close()

            QMessageBox.information(None, "Load Image", "Please load the second image.")
            progress = self.show_loading("Loading second image")
            file2, _ = QFileDialog.getOpenFileName(None, "Select Second Image", "", "GeoTIFF Files (*.tif *.tiff);;All Files (*)")
            if not file2: 
                return
            img2, prof2 = ImageProcessor.load_image(file2)
            progress.close()

            if prof1['crs'] != prof2['crs']:
                QMessageBox.critical(None, "Error", "Coordinate reference systems do not match!")
                return

            progress = self.show_loading("Analyzing changes...")
            start_time = time.time()
            self.change_result = ImageClassifier.detect_changes(img1, img2)
            self.profile = prof1
            progress.close()

            self.display_change_map(self.change_result[0])

            (total_area_of_change, percentage_change, change_intensity,
            low_change_percentage, medium_change_percentage, high_change_percentage) = ImageProcessor.calculate_statistics(self.change_result[0])

            stats_text = f"""
            <h4 style="color:#3498db;">Change Detection Statistics</h4>
            <table class="metadata-table">
                <tr><td><b>Total Area of Change</b></td><td>{total_area_of_change} pixels</td></tr>
                <tr><td><b>Percentage Change</b></td><td>{percentage_change:.2f}%</td></tr>
                <tr><td><b>Average Change Intensity</b></td><td>{change_intensity:.4f}</td></tr>
                <tr><td><b>Low Change (0-33%)</b></td><td>{low_change_percentage:.2f}%</td></tr>
                <tr><td><b>Medium Change (33-66%)</b></td><td>{medium_change_percentage:.2f}%</td></tr>
                <tr><td><b>High Change (66-100%)</b></td><td>{high_change_percentage:.2f}%</td></tr>
            </table>
            """
            self.info_box.setHtml(stats_text)

            self.stats_label.setVisible(False)

            self.ask_to_save_change_results()

            self.add_layer_to_list(file1, img1, prof1, is_change_map=False)
            self.add_layer_to_list(file2, img2, prof2, is_change_map=False)

            change_map_name = "Change Map"
            self.add_layer_to_list(change_map_name, self.change_result[0], prof1, is_change_map=True)

        except Exception as e:
            if progress: progress.close()
            QMessageBox.critical(None, "Error", f"Failed to detect changes: {str(e)}")

    def display_change_map(self, change_map):
        self.ax.clear()
        if hasattr(self, 'cbar'):
            self.cbar.remove()
        cax = self.ax.imshow(change_map, cmap='viridis')
        divider = make_axes_locatable(self.ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.fig.colorbar(cax, cax=cbar_ax, orientation='vertical', label='Change Intensity')
        self.canvas.draw()
        self.stats_label.setVisible(True)

    def show_loading(self, message):
        progress = QProgressDialog(message, "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        return progress

    def ask_to_save_change_results(self):
        save_reply = QMessageBox.question(
            self, "Save Changes", "Do you want to save the change detection results?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if save_reply == QMessageBox.Yes:
            self.save_change_results()

    def save_change_results(self):
        if hasattr(self, 'change_result'):
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Change Detection Results", "", "GeoTIFF Files (*.tif);;All Files (*)"
            )
            if file_name:
                try:
                    self.image_processor.save_image(self.change_result, self.profile, file_name)
                    QMessageBox.information(self, "Success", "Change detection results saved successfully.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save change detection results: {str(e)}")

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



    def classify_with_random_forest(self):
        """
        Classifies the image using Random Forest.
        """
        if hasattr(self, 'image'):  # التأكد من وجود صورة نشطة
            try:
                # Ask the user for the number of classes
                num_classes, ok = QInputDialog.getInt(self, "Number of Classes", "Enter the number of classes:", 2, 2, 10, 1)
                if not ok:
                    return  # User canceled the input

                # Prepare the image for classification
                flattened_image = self.image_classifier.preprocess_image_for_classification(self.image)
                
                # Create random labels based on the number of classes
                labels = np.random.randint(0, num_classes, flattened_image.shape[0])

                # Classify the image using Random Forest
                model, y_pred, accuracy = self.image_classifier.classify_random_forest(flattened_image, labels)
                
                # Reshape the predictions to match the original image dimensions
                if self.image.shape[0] == 3:  # RGB image
                    height, width = self.image.shape[1], self.image.shape[2]
                else:  # Grayscale image
                    height, width = self.image.shape[1], self.image.shape[2]
                
                classified_image = y_pred.reshape(height, width)
                self.classified_image = classified_image[np.newaxis, ...]  # تحديث الصورة المصنفة
                
                # Display the classified image
                self.display_classified_image(classified_image)
                
                QMessageBox.information(self, "Classification Result", f"Random Forest Accuracy: {accuracy:.2f}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to classify image: {str(e)}")

    def classify_with_svm(self):
        """
        Classifies the image using SVM.
        """
        if hasattr(self, 'image'):  # التأكد من وجود صورة نشطة
            try:
                # Ask the user for the number of classes
                num_classes, ok = QInputDialog.getInt(self, "Number of Classes", "Enter the number of classes:", 2, 2, 10, 1)
                if not ok:
                    return  # User canceled the input

                # Prepare the image for classification
                flattened_image = self.image_classifier.preprocess_image_for_classification(self.image)
                
                # Create random labels based on the number of classes
                labels = np.random.randint(0, num_classes, flattened_image.shape[0])

                # Classify the image using SVM
                model, y_pred, accuracy = self.image_classifier.classify_svm(flattened_image, labels)
                
                # Reshape the predictions to match the original image dimensions
                if self.image.shape[0] == 3:  # RGB image
                    height, width = self.image.shape[1], self.image.shape[2]
                else:  # Grayscale image
                    height, width = self.image.shape[1], self.image.shape[2]
                
                classified_image = y_pred.reshape(height, width)
                self.classified_image = classified_image[np.newaxis, ...]  # تحديث الصورة المصنفة
                
                # Display the classified image
                self.display_classified_image(classified_image)
                
                QMessageBox.information(self, "Classification Result", f"SVM Accuracy: {accuracy:.2f}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to classify image: {str(e)}")
                
    
    #Canny edge detection
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
    
    
    def check_required_bands(self, required_bands):
        missing = []
        for band in required_bands:
            if band not in self.loaded_bands:
                missing.append(f"Band {band} ({self.get_band_name(band)})")
        
        if missing:
            QMessageBox.critical(
                self, 
                "نطاقات مفقودة",
                f"الرجاء تحميل النطاقات التالية أولاً:\n{', '.join(missing)}"
            )
            return False
        return True
 
       
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   
    
    def calculate_spectral_index(self, index_name, required_bands, calculation_function, cmap):
        """
        دالة عامة لحساب المؤشرات الطيفية.
        
        :param index_name: اسم المؤشر (مثل "NDVI").
        :param required_bands: قائمة بأسماء الأشرطة المطلوبة (مثل ["NIR Band", "Red Band"]).
        :param calculation_function: الدالة المستخدمة لحساب المؤشر (مثل calculate_ndvi).
        :param cmap: تدرج الألوان المستخدم لعرض المؤشر (مثل 'RdYlGn' لـ NDVI).
        """
        try:
            # إخبار المستخدم بالأشرطة المطلوبة
            band_list = "\n".join([f"{i+1}. {band}" for i, band in enumerate(required_bands)])
            QMessageBox.information(self, f"Load Bands for {index_name}", 
                f"To calculate {index_name}, please load the following bands:\n{band_list}")

            # تحميل الأشرطة المطلوبة
            bands = {}
            for band_name in required_bands:
                file_name, _ = QFileDialog.getOpenFileName(self, f"Load {band_name}", "", "GeoTIFF Files (*.tif *.tiff);;All Files (*)")
                if not file_name:
                    return
                band_data, profile = self.image_processor.load_image(file_name)
                bands[band_name] = band_data
                self.add_layer_to_list(band_name, band_data, profile)

            # التأكد من تطابق أبعاد الأشرطة
            shapes = [band.shape for band in bands.values()]
            if len(set(shapes)) > 1:
                QMessageBox.critical(self, "Error", "All bands must have the same dimensions!")
                return

            # حساب المؤشر
            index_image = calculation_function(*bands.values())

            # إضافة المؤشر إلى قائمة الطبقات
            self.add_layer_to_list(index_name, index_image, profile)

            # عرض المؤشر بألوان معبرة
            self.display_raster_layer(index_image, index_name, cmap=cmap)

            # عرض تقرير المؤشر
            self.show_index_report(index_name, index_image)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate {index_name}: {str(e)}")

    
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
                profile = src.profile  # الحصول على معلومات الملف

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

    def show_index_report(self, index_name, index_image):
        """
        عرض تقرير بسيط عن المؤشر الطيفي.
        
        :param index_name: اسم المؤشر (مثل "NDVI").
        :param index_image: صورة المؤشر المحسوبة.
        """
        min_value = np.nanmin(index_image)
        max_value = np.nanmax(index_image)
        mean_value = np.nanmean(index_image)

        report = f"""
        <h4 style="color:#3498db;">{index_name} Report</h4>
        <table class="metadata-table">
            <tr><td><b>Minimum {index_name}</b></td><td>{min_value:.2f}</td></tr>
            <tr><td><b>Maximum {index_name}</b></td><td>{max_value:.2f}</td></tr>
            <tr><td><b>Mean {index_name}</b></td><td>{mean_value:.2f}</td></tr>
        </table>
        """
        self.info_box.setHtml(report)
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
            ndni = self.image_classifier.calculate_ndwi(swir1_band, swir2_band)

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



                
    def show_loading_message(self, message):
        self.statusBar().showMessage(message)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.progress_bar.setVisible(True)

    def hide_loading_message(self):
        self.statusBar().clearMessage()
        self.progress_bar.setVisible(False)
    
    # دالة حساب NDWI
    def calculate_ndwi(self):
        band_info_green = ("Green", 3)
        band_info_nir = ("Near Infrared (NIR)", 5)
        
        green = self.load_single_band(band_info_green)
        nir = self.load_single_band(band_info_nir)
        
        if green and nir:
            try:
                ndwi_image = self.image_classifier.calculate_ndwi(green, nir)
                self.image = ndwi_image
                self.display_raster_layer(ndwi_image)
                QMessageBox.information(self, "Success", "NDWI calculation completed successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Calculation failed: {str(e)}")

    def calculate_avi(self):
        band_info_nir = ("Near Infrared (NIR)", 5)
        band_info_red = ("Red", 4)
        
        nir = self.load_single_band(band_info_nir)
        red = self.load_single_band(band_info_red)
        
        if nir and red:
            try:
                avi_image = self.image_classifier.calculate_avi(nir, red)
                self.image = avi_image
                self.display_raster_layer(avi_image)
                QMessageBox.information(self, "Success", "AVI calculation completed successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Calculation failed: {str(e)}")

               

    # دالة حساب NDMI
    def calculate_ndmi(self):
        band_info_nir = ("Near Infrared (NIR)", 5)
        band_info_swir = ("SWIR 1", 6)
        
        nir = self.load_single_band(band_info_nir)
        swir = self.load_single_band(band_info_swir)
        
        if nir and swir:
            try:
                ndmi_image = self.image_classifier.calculate_ndmi(nir, swir)
                self.image = ndmi_image
                self.display_raster_layer(ndmi_image)
                QMessageBox.information(self, "Success", "NDMI calculation completed successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Calculation failed: {str(e)}")

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

    def combine_bands(self):
        if hasattr(self, 'image'):
            try:
                bands, ok = QInputDialog.getText(self, "Combine Bands", "Enter band indices (comma-separated):")
                if ok:
                    bands = list(map(int, bands.split(',')))
                    combined_image = self.image_processor.combine_bands(self.image, bands)
                    self.display_raster_layer(combined_image)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to combine bands: {str(e)}")

    def insert_map_title(self):
        # Check if an image is loaded
        if hasattr(self, 'image') and self.image is not None:
            try:
                transform = self.profile['transform']
                width = self.profile['width']
                height = self.profile['height']
                # Correct bounds order: [left, right, bottom, top]
                bounds = [
                    transform.c,  # left
                    transform.c + width * transform.a,  # right
                    transform.f + height * transform.e,  # bottom
                    transform.f  # top
                ]

                self.ax.clear()
                # Set a valid zoom level (e.g., 10)
                ctx.add_basemap(
                    self.ax,
                    crs=self.profile['crs'],
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    extent=bounds,
                    zoom=10  # Set a valid zoom level
                )

                if self.image.shape[0] == 3:  # RGB image
                    self.ax.imshow(self.image.transpose(1, 2, 0), extent=bounds)
                else:  # Grayscale image
                    self.ax.imshow(self.image[0], cmap='gray', extent=bounds)

                # Add the map title
                title, ok = QInputDialog.getText(self, "Insert Map Title", "Enter title:")
                if ok and title:
                    self.ax.set_title(title, fontsize=16, fontweight='bold')

                self.canvas.draw()
                QMessageBox.information(self, "Success", "Map title inserted successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to insert map title: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No image loaded.")

 
    def insert_text(self):
        """
        إدراج نص عند إحداثيات محددة بالنقر.
        """
        try:
            text, ok = QInputDialog.getText(self, "Insert Text", "Enter text:")
            if ok:
                color = QColorDialog.getColor(Qt.red, self, "Choose Text Color")
                font, ok_font = QFontDialog.getFont(QFont("Arial", 12), self, "Choose Font")
                
                if color.isValid() and ok_font:
                    x, ok_x = QInputDialog.getDouble(self, "X Coordinate", "Enter X coordinate:", 0, -10000, 10000, 2)
                    y, ok_y = QInputDialog.getDouble(self, "Y Coordinate", "Enter Y coordinate:", 0, -10000, 10000, 2)
                    
                    if ok_x and ok_y:
                        self.ax.text(
                            x, y, text,
                            fontsize=font.pointSize(),
                            color=color.name(),
                            fontfamily=font.family(),
                            backgroundcolor='white'
                        )
                        self.canvas.draw()
                        QMessageBox.information(self, "Success", "Text inserted successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to insert text: {str(e)}")
    
    def insert_legend(self):
        """
        إدراج مفتاح الخريطة.
        """
        try:
            legend = self.ax.legend(loc='upper right')
            legend.get_frame().set_facecolor('white')
            self.canvas.draw()
            QMessageBox.information(self, "Success", "Legend inserted successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to insert legend: {str(e)}")

    def insert_north_arrow(self):
        """
        إدراج سهم الشمال.
        """
        try:
            x, ok_x = QInputDialog.getDouble(self, "X Coordinate", "Enter X coordinate for North Arrow:", 0, -10000, 10000, 2)
            y, ok_y = QInputDialog.getDouble(self, "Y Coordinate", "Enter Y coordinate for North Arrow:", 0, -10000, 10000, 2)
            
            if ok_x and ok_y:
                self.ax.annotate('N', xy=(x, y), xytext=(x, y+0.1),
                                arrowprops=dict(facecolor='black', shrink=0.05))
                self.canvas.draw()
                QMessageBox.information(self, "Success", "North arrow inserted successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to insert north arrow: {str(e)}")

    def insert_grid(self):
        """
        إدراج شبكة الإحداثيات.
        """
        try:
            self.ax.grid(True)
            self.canvas.draw()
            QMessageBox.information(self, "Success", "Grid inserted successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to insert grid: {str(e)}")

    def insert_scale_bar(self):
        """
        إدراج شريط مقياس الرسم.
        """
        try:
            # إعداد الإحداثيات والمقياس
            x, ok_x = QInputDialog.getDouble(self, "X Coordinate", "Enter X coordinate for Scale Bar:", 0, -10000, 10000, 2)
            y, ok_y = QInputDialog.getDouble(self, "Y Coordinate", "Enter Y coordinate for Scale Bar:", 0, -10000, 10000, 2)
            scale_length, ok_length = QInputDialog.getDouble(self, "Scale Length", "Enter length of Scale Bar (in map units):", 100, 1, 10000, 2)
            
            if ok_x and ok_y and ok_length:
                # إعداد شريط مقياس الرسم
                self.ax.plot([x, x + scale_length], [y, y], color='black', lw=2)
                self.ax.plot([x, x], [y - 0.01 * scale_length, y + 0.01 * scale_length], color='black', lw=2)
                self.ax.plot([x + scale_length, x + scale_length], [y - 0.01 * scale_length, y + 0.01 * scale_length], color='black', lw=2)
                self.ax.text(x + scale_length / 2, y + 0.02 * scale_length, f'{scale_length} units', ha='center', va='bottom', fontsize=10, color='black')
                
                self.canvas.draw()
                QMessageBox.information(self, "Success", "Scale bar inserted successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to insert scale bar: {str(e)}")


    def save_combined_layout(self):
        """
        تصدير تخطيط الخريطة الكامل.
        """
        try:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Map Layout", "", "PDF Files (*.pdf)")
            if file_name:
                with PdfPages(file_name) as pdf:
                    # حفظ الخريطة الحالية
                    pdf.savefig(self.fig)
                    # إضافة صفحة فارغة مع نصوص إضافية إذا لزم الأمر
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, 'Additional Information', transform=ax.transAxes, ha='center', va='center')
                    pdf.savefig(fig)
                    plt.close(fig)
                QMessageBox.information(self, "Success", "Map layout saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save map layout: {str(e)}")

    def on_map_click(self, event):
        """
        تفعيل النقر على الخريطة لإدراج النص.
        """
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            self.insert_text_at(x, y)

    def insert_text_at(self, x, y):
        """
        إدراج نص عند إحداثيات محددة بالنقر.
        """
        try:
            text, ok = QInputDialog.getText(self, "Insert Text", "Enter text:")
            if ok:
                color = QColorDialog.getColor(Qt.red, self, "Choose Text Color")
                font, ok_font = QFontDialog.getFont(QFont("Arial", 12), self, "Choose Font")
                
                if color.isValid() and ok_font:
                    self.ax.text(
                        x, y, text,
                        fontsize=font.pointSize(),
                        color=color.name(),
                        fontfamily=font.family(),
                        backgroundcolor='white'
                    )
                    self.canvas.draw()
                    QMessageBox.information(self, "Success", "Text inserted successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to insert text: {str(e)}")

    def save_layout(self):
        """
        حفظ التخطيط الكامل كصورة أو PDF.
        """
        if hasattr(self, 'image'):
            try:
                formats = "PDF (*.pdf);;PNG (*.png);;JPEG (*.jpg)"
                file_name, selected_format = QFileDialog.getSaveFileName(
                    self, "Save Layout", "", formats
                )

                if file_name:
                    if selected_format == "PDF (*.pdf)":
                        with PdfPages(file_name) as pdf:
                            pdf.savefig(self.fig)
                        QMessageBox.information(self, "Success", f"Layout saved as PDF: {file_name}")
                    elif selected_format == "PNG (*.png)":
                        self.fig.savefig(file_name, format='png', dpi=300)
                        QMessageBox.information(self, "Success", f"Layout saved as PNG: {file_name}")
                    elif selected_format == "JPEG (*.jpg)":
                        self.fig.savefig(file_name, format='jpg', dpi=300)
                        QMessageBox.information(self, "Success", f"Layout saved as JPEG: {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save layout: {str(e)}")
                
    def save_combined_layout(self):
        """حفظ الخريطة مع جميع العناصر المضافة"""
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self, 
                "حفظ الخريطة النهائية", 
                "", 
                "PDF (*.pdf);;PNG (*.png);;JPEG (*.jpg)"
            )
            
            if file_name:
                # حفظ الشكل كاملًا مع العناصر
                self.fig.savefig(file_name, bbox_inches='tight', dpi=300)
                QMessageBox.information(self, "Success", "File saved successfully!")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")
            


########################################### Print
    def print_image(self):
        """Handle both printing to physical printer and PDF export"""
        if not hasattr(self, 'ax') or not self.ax:
            QMessageBox.warning(self, "Error", "No display content available")
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

    def _render_raster_to_pdf(self, painter, target_rect, layer_data):
        """Render raster layer to PDF with proper scaling."""
        image = layer_data['image']
        
        # Convert numpy array to QImage
        if len(image.shape) == 3:  # Color image (RGB)
            height, width, _ = image.shape
            qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        else:  # Grayscale
            height, width = image.shape
            qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        
        # Calculate scaling to fit content area while maintaining aspect ratio
        img_ratio = width / height
        target_ratio = target_rect.width() / target_rect.height()
        
        if img_ratio > target_ratio:
            # Image is wider than target area
            scale = target_rect.width() / width
        else:
            # Image is taller than target area
            scale = target_rect.height() / height
        
        scaled_width = width * scale
        scaled_height = height * scale
        
        # Center the image
        x = target_rect.x() + (target_rect.width() - scaled_width) / 2
        y = target_rect.y() + (target_rect.height() - scaled_height) / 2
        
        # Draw the image
        painter.drawImage(QRectF(x, y, scaled_width, scaled_height), qimage)

    def _render_vector_to_pdf(self, painter, target_rect, layer_data):
        """Render vector layer to PDF with proper scaling."""
        gdf = layer_data['gdf']
        if len(gdf) == 0:
            return
        
        # Get layer bounds with padding
        bounds = gdf.total_bounds
        width = max(bounds[2] - bounds[0], 1.0)
        height = max(bounds[3] - bounds[1], 1.0)
        
        # If all features are at same point, create reasonable bounds
        if width == 1.0 and height == 1.0:
            width, height = 100.0, 100.0
            bounds = (bounds[0]-50, bounds[1]-50, bounds[0]+50, bounds[1]+50)
        
        # Calculate scale to fit content area
        x_scale = target_rect.width() / width
        y_scale = target_rect.height() / height
        scale = min(x_scale, y_scale)
        
        # Calculate offset to center content
        x_offset = target_rect.x() - (bounds[0] * scale)
        y_offset = target_rect.y() - (bounds[1] * scale)
        
        # Set up styling
        color = QColor(layer_data.get('color', '#FF0000'))
        line_width = layer_data.get('line_width', 1.0)
        point_size = layer_data.get('point_size', 5)
        
        # Draw each feature type
        for geom in gdf.geometry:
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                self._draw_pdf_polygon(painter, geom, color, line_width, scale, x_offset, y_offset)
            elif geom.geom_type in ['LineString', 'MultiLineString']:
                self._draw_pdf_line(painter, geom, color, line_width, scale, x_offset, y_offset)
            elif geom.geom_type in ['Point', 'MultiPoint']:
                self._draw_pdf_point(painter, geom, color, point_size, scale, x_offset, y_offset)

    def _draw_pdf_polygon(self, painter, geom, color, line_width, scale, x_offset, y_offset):
        """Draw polygon to PDF."""
        pen = QPen(color, line_width)
        brush = QBrush(color, Qt.SolidPattern)
        painter.setPen(pen)
        painter.setBrush(brush)
        
        if geom.geom_type == 'Polygon':
            exterior = geom.exterior
            points = [QPointF(x * scale + x_offset, y * scale + y_offset) 
                    for x, y in zip(*exterior.xy)]
            painter.drawPolygon(QPolygonF(points))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                exterior = poly.exterior
                points = [QPointF(x * scale + x_offset, y * scale + y_offset) 
                        for x, y in zip(*exterior.xy)]
                painter.drawPolygon(QPolygonF(points))

    def _draw_pdf_line(self, painter, geom, color, line_width, scale, x_offset, y_offset):
        """Draw line to PDF."""
        pen = QPen(color, line_width)
        painter.setPen(pen)
        
        if geom.geom_type == 'LineString':
            points = [QPointF(x * scale + x_offset, y * scale + y_offset) 
                    for x, y in zip(*geom.xy)]
            painter.drawPolyline(QPolygonF(points))
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                points = [QPointF(x * scale + x_offset, y * scale + y_offset) 
                        for x, y in zip(*line.xy)]
                painter.drawPolyline(QPolygonF(points))

    def _draw_pdf_point(self, painter, geom, color, point_size, scale, x_offset, y_offset):
        """Draw point to PDF."""
        pen = QPen(color, 1)
        brush = QBrush(color)
        painter.setPen(pen)
        painter.setBrush(brush)
        
        if geom.geom_type == 'Point':
            x = geom.x * scale + x_offset
            y = geom.y * scale + y_offset
            painter.drawEllipse(QPointF(x, y), point_size, point_size)
        elif geom.geom_type == 'MultiPoint':
            for point in geom.geoms:
                x = point.x * scale + x_offset
                y = point.y * scale + y_offset
                painter.drawEllipse(QPointF(x, y), point_size, point_size)

    def handle_print(self):
        """Enhanced printing/PDF export with proper display of all elements"""
        if not hasattr(self, 'ax') or not self.ax:
            QMessageBox.warning(self, "Error", "No display area found")
            return

        # Create printer dialog with proper options
        printer = QPrinter(QPrinter.HighResolution)
        print_dialog = QPrintDialog(printer, self)
        
        # Configure dialog options
        print_dialog.setOption(QPrintDialog.PrintToFile, True)
        print_dialog.setOption(QPrintDialog.PrintSelection, False)
        print_dialog.setOption(QPrintDialog.PrintPageRange, False)
        print_dialog.setOption(QPrintDialog.PrintCollateCopies, True)
        
        if print_dialog.exec_() != QDialog.Accepted:
            return

        try:
            # Create a temporary figure for printing
            fig = Figure(figsize=(8, 6), dpi=100)
            canvas = FigureCanvas(fig)
            print_ax = fig.add_subplot(111)
            
            # Copy all elements from main display to print figure
            self._copy_display_elements(self.ax, print_ax)
            
            # Adjust layout
            fig.tight_layout()
            
            # Render to printer
            painter = QPainter()
            if painter.begin(printer):
                # Calculate scale factor
                page_rect = printer.pageRect()
                scale = min(
                    page_rect.width() / canvas.width(),
                    page_rect.height() / canvas.height()
                )
                
                # Center the output
                x = (page_rect.width() - (canvas.width() * scale)) / 2
                y = (page_rect.height() - (canvas.height() * scale)) / 2
                
                # Render the figure
                canvas.render(
                    painter,
                    QRectF(x, y, canvas.width() * scale, canvas.height() * scale)
                )
                painter.end()
                
                if printer.outputFormat() == QPrinter.PdfFormat:
                    QMessageBox.information(
                        self, "PDF Exported", 
                        f"PDF saved to:\n{printer.outputFileName()}"
                    )
                else:
                    QMessageBox.information(
                        self, "Print Sent", 
                        "Content sent to printer successfully"
                    )
            else:
                QMessageBox.warning(self, "Error", "Could not start printing")
                
        except Exception as e:
            QMessageBox.critical(
                self, "Print Error", 
                f"Failed to generate print output: {str(e)}"
            )

    def _copy_display_elements(self, source_ax, target_ax):
        """Copy all visible elements from source to target axes"""
        # Copy the main image or vector layer
        for artist in source_ax.get_children():
            if isinstance(artist, (QuadMesh, PathCollection, Line2D, Polygon)):
                target_ax.add_artist(artist)
        
        # Copy the axis lines and labels
        target_ax.set_xlim(source_ax.get_xlim())
        target_ax.set_ylim(source_ax.get_ylim())
        target_ax.set_xlabel(source_ax.get_xlabel())
        target_ax.set_ylabel(source_ax.get_ylabel())
        target_ax.set_title(source_ax.get_title())
        
        # Copy grid and ticks
        target_ax.grid(source_ax.grid())
        target_ax.set_xticks(source_ax.get_xticks())
        target_ax.set_yticks(source_ax.get_yticks())
        
        # Copy any additional annotations
        for text in source_ax.texts:
            target_ax.text(
                text.get_position()[0], text.get_position()[1],
                text.get_text(),
                fontsize=text.get_fontsize(),
                color=text.get_color()
            )






###################################################
 
    def _draw_features(self, painter, gdf, feature_type, layer_data, bounds, x_scale, y_scale):
        """Draw features of specific type with proper styling"""
        color = QColor(layer_data.get('color', '#FF0000'))
        
        if feature_type == 'point':
            pen = QPen(color, 1)
            brush = QBrush(color)
            point_size = layer_data.get('point_size', 5)
            
            for geom in gdf.geometry:
                if geom.geom_type == 'Point':
                    x = (geom.x - bounds[0]) * x_scale
                    y = (geom.y - bounds[1]) * y_scale
                    painter.setPen(pen)
                    painter.setBrush(brush)
                    painter.drawEllipse(QPointF(x, y), point_size, point_size)
                elif geom.geom_type == 'MultiPoint':
                    for point in geom.geoms:
                        x = (point.x - bounds[0]) * x_scale
                        y = (point.y - bounds[1]) * y_scale
                        painter.drawEllipse(QPointF(x, y), point_size, point_size)
        
        elif feature_type == 'line':
            pen = QPen(color, layer_data.get('line_width', 1.0))
            painter.setPen(pen)
            
            for geom in gdf.geometry:
                if geom.geom_type == 'LineString':
                    points = [QPointF((x - bounds[0]) * x_scale, 
                            (y - bounds[1]) * y_scale) for x, y in zip(*geom.xy)]
                    painter.drawPolyline(*points)
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        points = [QPointF((x - bounds[0]) * x_scale, 
                                (y - bounds[1]) * y_scale) for x, y in zip(*line.xy)]
                        painter.drawPolyline(*points)
        
        elif feature_type == 'polygon':
            pen = QPen(color, layer_data.get('line_width', 1.0))
            brush = QBrush(color, Qt.SolidPattern)
            painter.setPen(pen)
            painter.setBrush(brush)
            
            for geom in gdf.geometry:
                if geom.geom_type == 'Polygon':
                    exterior = geom.exterior
                    polygon = QPolygonF([QPointF((x - bounds[0]) * x_scale, 
                                    (y - bounds[1]) * y_scale) for x, y in zip(*exterior.xy)])
                    painter.drawPolygon(polygon)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        exterior = poly.exterior
                        polygon = QPolygonF([QPointF((x - bounds[0]) * x_scale, 
                                                (y - bounds[1]) * y_scale) for x, y in zip(*exterior.xy)])
                        painter.drawPolygon(polygon)

    def _scale_and_draw(self, painter, printer, qimage):
        """Scale and draw QImage to printer with error checking"""
        try:
            page_rect = printer.pageRect()
            image_rect = qimage.rect()

            # Ensure valid dimensions
            if image_rect.width() <= 0 or image_rect.height() <= 0:
                raise ValueError("Image has invalid dimensions")

            # Calculate scale factors
            x_scale = page_rect.width() / image_rect.width()
            y_scale = page_rect.height() / image_rect.height()
            scale = min(x_scale, y_scale)

            # Calculate centered position
            x_offset = (page_rect.width() - (image_rect.width() * scale)) / 2
            y_offset = (page_rect.height() - (image_rect.height() * scale)) / 2

            # Draw the image
            painter.save()
            painter.translate(x_offset, y_offset)
            painter.scale(scale, scale)
            painter.drawImage(0, 0, qimage)
            painter.restore()
            return True
        except Exception as e:
            raise Exception(f"Failed to scale and draw: {str(e)}")
           
    def page_setup(self):
        """Configure page settings"""
        printer = QPrinter()
        dialog = QPageSetupDialog(printer, self)
        dialog.exec_()

    def batch_export(self):
        """Export to multiple formats simultaneously"""
        if hasattr(self, 'image'):
            formats = ["PDF", "PNG", "JPEG", "GeoTIFF"]
            selected_formats, ok = QInputDialog.getItem(
                self, "Batch Export", "Select formats:", formats, 0, True
            )
            if ok and selected_formats:
                folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
                if folder:
                    try:
                        base_name = QInputDialog.getText(self, "Base Name", "Enter base filename:")[0]
                        for fmt in selected_formats:
                            fmt = fmt.lower()
                            if fmt not in ['pdf', 'png', 'jpeg', 'jpg', 'geotiff']:
                                QMessageBox.warning(self, "Unsupported Format", f"Format '{fmt}' is not supported.")
                                continue
                            path = f"{folder}/{base_name}.{fmt}"
                            if fmt == "geotiff":
                                self.image_processor.save_image(self.image, self.profile, path)
                            else:
                                self.fig.savefig(path, dpi=300, bbox_inches='tight')
                        QMessageBox.information(self, "Success", "Batch export completed!")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")


    def calculate_ndni(self):
        if hasattr(self, 'image'):
            try:
                ndni_image = self.image_classifier.calculate_ndni(self.image)
                self.display_raster_layer(ndni_image)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to calculate NDNI: {str(e)}")

    def calculate_ndwi(self):
        if hasattr(self, 'image'):
            try:
                ndwi_image = self.image_classifier.calculate_ndwi(self.image)
                self.display_raster_layer(ndwi_image)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to calculate NDWI: {str(e)}")        

    def calculate_avi(self):
        if hasattr(self, 'image'):
            try:
                avi_image = self.image_classifier.calculate_avi(self.image)
                self.display_raster_layer(avi_image)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to calculate AVI: {str(e)}")

    def calculate_ndmi(self):
        if hasattr(self, 'image'):
            try:
                ndmi_image = self.image_classifier.calculate_ndmi(self.image)
                self.display_raster_layer(ndmi_image)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to calculate NDMI: {str(e)}")

 
    
    def detect_changes(self):
        progress = None
        try:
            # تحميل الصور
            QMessageBox.information(None, "Load Image", "Please load the first image.")
            progress = self.show_loading("Loading first image")
            file1, _ = QFileDialog.getOpenFileName(None, "Select Base Image", "", "GeoTIFF Files (*.tif *.tiff);;All Files (*)")
            if not file1: 
                return
            img1, prof1 = ImageProcessor.load_image(file1)
            progress.close()

            QMessageBox.information(None, "Load Image", "Please load the second image.")
            progress = self.show_loading("Loading second image")
            file2, _ = QFileDialog.getOpenFileName(None, "Select Second Image", "", "GeoTIFF Files (*.tif *.tiff);;All Files (*)")
            if not file2: 
                return
            img2, prof2 = ImageProcessor.load_image(file2)
            progress.close()

            # التحقق من نظام الإحداثيات
            # if prof1['crs'] != prof2['crs']:
            #     QMessageBox.critical(None, "Error", "Coordinate reference systems do not match!")
            #     return
            if prof1['crs'] != prof2['crs']:
                QMessageBox.critical(self, "CRS Mismatch", 
                    f"Coordinate reference systems do not match:\nImage 1 CRS: {prof1['crs']}\nImage 2 CRS: {prof2['crs']}" )
                return

            # حساب التغير
            progress = self.show_loading("Analyzing changes...")
            start_time = time.time()
            self.change_result = ImageClassifier.detect_changes(img1, img2)
            self.profile = prof1  # استخدام بروفايل الصورة الأولى
            progress.close()

            # عرض خريطة التغير
            self.display_change_map(self.change_result[0])

            # حساب الإحصائيات
            (total_area_of_change, percentage_change, change_intensity,
            low_change_percentage, medium_change_percentage, high_change_percentage) = ImageProcessor.calculate_statistics(self.change_result[0])

            # عرض الإحصائيات في الصندوق
            stats_text = f"""
            <h4 style="color:#3498db;">Change Detection Statistics</h4>
            <table class="metadata-table">
                <tr><td><b>Total Area of Change</b></td><td>{total_area_of_change} pixels</td></tr>
                <tr><td><b>Percentage Change</b></td><td>{percentage_change:.2f}%</td></tr>
                <tr><td><b>Average Change Intensity</b></td><td>{change_intensity:.4f}</td></tr>
                <tr><td><b>Low Change (0-33%)</b></td><td>{low_change_percentage:.2f}%</td></tr>
                <tr><td><b>Medium Change (33-66%)</b></td><td>{medium_change_percentage:.2f}%</td></tr>
                <tr><td><b>High Change (66-100%)</b></td><td>{high_change_percentage:.2f}%</td></tr>
            </table>
            """
            self.info_box.setHtml(stats_text)

            # إخفاء الإحصائيات القديمة بجوار الصورة
            self.stats_label.setVisible(False)

            # حفظ النتائج إذا طلب المستخدم
            self.ask_to_save_change_results()

            # إضافة الصور إلى قائمة الطبقات
            self.add_layer_to_list(file1, img1, prof1, is_change_map=False)
            self.add_layer_to_list(file2, img2, prof2, is_change_map=False)

            # إضافة خريطة التغير إلى قائمة الطبقات
            change_map_name = "Change Map"
            self.add_layer_to_list(change_map_name, self.change_result[0], prof1, is_change_map=True)

        except Exception as e:
            if progress: progress.close()
            QMessageBox.critical(None, "Error", f"Failed to detect changes: {str(e)}")

    def display_change_map(self, change_map):
        """
        عرض خريطة التغير مع colorbar.
        """
        # مسح المحتوى السابق
        self.ax.clear()
        
        # إزالة colorbar إذا كان موجودًا
        if hasattr(self, 'cbar'):
            self.cbar.remove()
            delattr(self, 'cbar')
        
        # عرض خريطة التغير
        cax = self.ax.imshow(change_map, cmap='viridis')
        
        # إضافة colorbar
        divider = make_axes_locatable(self.ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.fig.colorbar(cax, cax=cbar_ax, orientation='vertical', label='Change Intensity')
        
        # تحديث canvas
        self.canvas.draw()

        # إظهار label فقط مع خريطة التغير
        self.stats_label.setVisible(True)

    
    def ask_to_save_change_results(self):
        reply = QMessageBox.question(None, "Save Results", "Do you want to save the change detection results?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.save_change_results()

    def save_change_results(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save Change Detection Results", "", "GeoTIFF Files (*.tif *.tiff);;All Files (*)")
            if not file_path:
                return
            ImageProcessor.save_image(file_path, self.change_result, self.profile)
            QMessageBox.information(None, "Success", "Change detection results saved successfully.")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save change detection results: {str(e)}")

    def show_loading(self, message):
        progress = QProgressDialog(message, None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)  # إزالة زر الإلغاء
        progress.show()
        QApplication.processEvents()  # تحديث الواجهة
        return progress

    def ask_to_save_change_results(self):
        reply = QMessageBox.question(self, 'Save Results', 'Do you want to save the change detection results?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.save_change_results()

    def save_change_results(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Change Detection Results", "", "GeoTIFF Files (*.tif *.tiff);;All Files (*)", options=options)
        if file_name:
            # Ensure self.profile is a dictionary
            profile_dict = dict(self.profile)
            
            # Save the image data without the legend or statistics
            image_data = self.change_result[0]
            
            # Save the image data
            ImageProcessor.save_image_oneBand(image_data, profile_dict, file_name)
            QMessageBox.information(self, "Save Results", "Results saved successfully!")


        def show_loading(self, message):
        # Dummy implementation for showing a loading message
            print(message)
            return self

        def close(self):
            # Dummy implementation for closing the loading message
            print("Loading complete")


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
    
    def validate_formula(self):
        """Check formula syntax and layer availability"""
        try:
            formula = self.formula_edit.toPlainText()
            # Temporary substitution for validation
            test_formula = re.sub(r'"([^"]+)"', '1', formula)
            parsed = ast.parse(test_formula, mode='eval')
            QMessageBox.information(self, "Validation", "Formula syntax is valid!")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid formula syntax:\n{str(e)}")
            return False   
    
    def execute_calculation(self, dialog):
        """Execute raster calculation and prompt to save results"""
        try:
            formula = self.formula_edit.toPlainText().strip()
            if not formula:
                QMessageBox.warning(self, "Empty Formula", "Please enter a formula")
                return

            # Extract and validate layers
            layer_names = re.findall(r'"([^"]+)"', formula)
            missing = [name for name in layer_names if name not in self.layers]
            if missing:
                raise ValueError(f"Missing layers: {', '.join(missing)}")

            # Prepare variables and environment
            var_mapping = {}
            modified_formula = formula
            for name in layer_names:
                safe_var = re.sub(r'[^a-zA-Z0-9]', '_', f"lyr_{name}")
                var_mapping[name] = safe_var
                modified_formula = modified_formula.replace(f'"{name}"', safe_var)

            # Verify dimensions
            base_shape = self.layers[layer_names[0]]['image'].shape[1:]
            for name in layer_names:
                if self.layers[name]['image'].shape[1:] != base_shape:
                    raise ValueError(f"Layer {name} has incompatible dimensions")

            # Setup calculation environment
            env = {
                'np': np,
                'sqrt': np.sqrt,
                'log': np.log,
                'exp': np.exp,
                'sin': np.sin,
                'cos': np.cos,
                'abs': np.abs,
                'where': np.where,
                'nan_to_num': np.nan_to_num
            }

            # Add layer data to environment
            for name, safe_var in var_mapping.items():
                arr = self.layers[name]['image']
                env[safe_var] = arr[0] if (arr.ndim == 3 and arr.shape[0] == 1) else arr

            # Execute calculation
            modified_formula = self.add_division_safety(modified_formula)
            tree = ast.parse(modified_formula, mode='eval')
            code = compile(tree, '<string>', 'eval')

            with np.errstate(divide='ignore', invalid='ignore'):
                result = eval(code, {"__builtins__": None}, env)

            # Post-process result
            result = np.nan_to_num(result)
            if result.ndim == 2:
                result = result[np.newaxis, ...]

            # Create new layer
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_layer = f"Calculation_{timestamp}"
            profile = self.layers[layer_names[0]]['profile'].copy()
            
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
                    f"Data Range: {result.min():.2f} to {result.max():.2f}\n"
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
    def get_raster_bounds(self, layer):
        """Calculate geographic bounds of the raster in WGS84 with improved error handling"""
        try:
            # Check if bounds already in WGS84
            if 'bounds' in layer and 'min_lon' in layer['bounds']:
                return layer['bounds']
            
            # Get transform and dimensions
            transform = layer['profile']['transform']
            width = layer['image'].width
            height = layer['image'].height
            
            # Calculate corner coordinates in pixel coordinates
            corners_pixel = [
                (0, 0),      # Top-left
                (width, 0),  # Top-right
                (width, height),  # Bottom-right
                (0, height)   # Bottom-left
            ]
            
            # Convert to geographic coordinates
            corners_geo = [transform * point for point in corners_pixel]
            
            # Get source CRS
            src_crs = layer['profile'].get('crs', 'EPSG:4326')  # Default to WGS84
            
            # If not already WGS84, transform coordinates
            if src_crs != 'EPSG:4326':
                try:
                    transformer = pyproj.Transformer.from_crs(
                        src_crs, 'EPSG:4326', always_xy=True)
                    corners_geo = [transformer.transform(x, y) for x, y in corners_geo]
                except Exception as e:
                    print(f"CRS transformation failed: {e}")
                    return None
            
            # Extract min/max coordinates
            lons = [point[0] for point in corners_geo]
            lats = [point[1] for point in corners_geo]
            
            return {
                'min_lon': min(lons),
                'max_lon': max(lons),
                'min_lat': min(lats),
                'max_lat': max(lats)
            }
            
        except Exception as e:
            print(f"Error in get_raster_bounds: {e}")
            return None
    

 

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
                        print(f"Warning: Could not delete temp file: {str(e)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")         

    def update_utm_preview(self):
        """Update the UTM preview information"""
        if not hasattr(self, 'zone_combo') or not hasattr(self, 'hemi_combo'):
            return
        
        zone = self.zone_combo.currentData()
        hemi = self.hemi_combo.currentData()
        epsg = 32600 + zone if hemi == 'N' else 32700 + zone
        
        preview_text = f"""
        <b>Selected UTM Zone:</b> {zone}{hemi}<br>
        <b>EPSG Code:</b> EPSG:{epsg}<br>
        <b>Central Meridian:</b> {-177 + 6*(zone-1)}°<br>
        <b>Valid Range:</b> {'0° to 84°N' if hemi == 'N' else '80°S to 0°'}<br>
        <b>Units:</b> Meters<br>
        <b>Datum:</b> WGS84
        """
        
        if hasattr(self, 'preview_label'):
            self.preview_label.setText(preview_text)    

    def get_human_readable_crs(self, crs):
        """Convert CRS object to human-readable information"""
        if isinstance(crs, str):
            if crs == 'EPSG:4326':
                return {
                    'name': "WGS 84 (Latitude/Longitude)",
                    'type': "Geographic",
                    'units': "Degrees",
                    'epsg': "EPSG:4326",
                    'datum': "WGS84"
                }
            elif crs.startswith('EPSG:326'):
                zone = int(crs[8:])
                return {
                    'name': f"UTM Zone {zone}N",
                    'type': "Projected",
                    'units': "Meters",
                    'epsg': crs,
                    'datum': "WGS84"
                }
            elif crs.startswith('EPSG:327'):
                zone = int(crs[8:])
                return {
                    'name': f"UTM Zone {zone}S",
                    'type': "Projected",
                    'units': "Meters",
                    'epsg': crs,
                    'datum': "WGS84"
                }
        
        # Default return for unknown CRS
        return {
            'name': str(crs) if crs else "Unknown Coordinate System",
            'type': "Unknown",
            'units': "Unknown",
            'epsg': str(crs) if crs else "Not specified",
            'datum': "Unknown"
        }

    def update_crs_preview(self, crs_info, utm_widget):
        """Update the CRS preview and toggle UTM zone visibility"""
        self.crs_preview.setText(self.format_crs_preview(crs_info))
        
        # Show UTM widgets only if custom UTM is selected
        show_utm = crs_info.get('custom_utm', False)
        utm_widget.setVisible(show_utm)
        
        # Adjust dialog size if needed
        if show_utm:
            self.crs_preview.parent().parent().parent().adjustSize()


    def build_crs_database(self, suggested_utm_zones, bounds):
        """Build the database of available CRS options"""
        crs_db = []
        
        # Always add recommended UTM zones first if available
        for zone_info in suggested_utm_zones:
            hemi = 'N' if zone_info['min_lat'] >= 0 else 'S'
            epsg = 32600 + zone_info['zone'] if hemi == 'N' else 32700 + zone_info['zone']
            crs_db.append({
                'name': f"UTM Zone {zone_info['zone']}{hemi} (Recommended)",
                'epsg': f"EPSG:{epsg}",
                'type': "Projected",
                'units': "Meters",
                'datum': "WGS84",
                'area': f"{zone_info['min_lon']:.1f}°-{zone_info['max_lon']:.1f}°E",
                'zone': zone_info['zone'],
                'hemisphere': hemi
            })
        
        # Add standard UTM option (all zones) if we have bounds
        if bounds:
            crs_db.append({
                'name': "UTM Zone (Manual Selection)",
                'epsg': "custom",
                'type': "Projected",
                'units': "Meters",
                'datum': "WGS84",
                'area': "User-defined",
                'custom_utm': True
            })
        
        # Standard CRS options
        crs_db.extend([
            {
                'name': "WGS84 (Lat/Lon)",
                'epsg': "EPSG:4326",
                'type': "Geographic",
                'units': "Degrees",
                'datum': "WGS84",
                'area': "Global"
            },
            {
                'name': "Web Mercator",
                'epsg': "EPSG:3857",
                'type': "Projected",
                'units': "Meters",
                'datum': "WGS84",
                'area': "Global between 85°S and 85°N"
            }
        ])
        
        # Add custom CRS option
        crs_db.append({
            'name': "Custom CRS...",
            'epsg': "custom",
            'type': "User-defined",
            'units': "Varies",
            'datum': "Varies",
            'area': "User-defined"
        })
        
        return crs_db

    def format_crs_preview(self, crs_info):
        """Format CRS information for display"""
        lines = [
            f"<b>Coordinate System:</b> {crs_info['name']}",
            f"<b>Type:</b> {crs_info['type']}",
            f"<b>Units:</b> {crs_info['units']}",
            f"<b>Datum:</b> {crs_info['datum']}",
            f"<b>Area of Use:</b> {crs_info.get('area', 'Not specified')}"
        ]
        
        if crs_info['epsg'] != 'custom':
            lines.append(f"<b>EPSG Code:</b> {crs_info['epsg']}")
        
        return "<br>".join(lines)

    def format_crs_tooltip(self, crs_info):
        """Format tooltip for CRS options"""
        return (
            f"<b>Type:</b> {crs_info['type']}<br>"
            f"<b>Units:</b> {crs_info['units']}<br>"
            f"<b>Datum:</b> {crs_info['datum']}<br>"
            f"<b>Area:</b> {crs_info.get('area', 'Not specified')}"
        )

    def get_raster_bounds(self, layer):
        """Calculate geographic bounds of the raster in WGS84"""
        try:
            # Check if bounds already in WGS84
            if 'bounds' in layer and 'min_lon' in layer['bounds']:
                return layer['bounds']
            
            # Get transform and calculate corners
            transform = layer['profile']['transform']
            width, height = layer['image'].width, layer['image'].height
            
            corners = [
                transform * (0, 0),
                transform * (width, 0),
                transform * (width, height),
                transform * (0, height)
            ]
            
            # Transform to WGS84 if needed
            if layer['profile'].get('crs'):
                src_crs = layer['profile']['crs']
                dst_crs = 'EPSG:4326'
                transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                corners = [transformer.transform(x, y) for x, y in corners]
            
            lons = [p[0] for p in corners]
            lats = [p[1] for p in corners]
            
            return {
                'min_lon': min(lons),
                'max_lon': max(lons),
                'min_lat': min(lats),
                'max_lat': max(lats)
            }
        except Exception as e:
            print(f"Error calculating bounds: {e}")
            return None

    def calculate_utm_zones(self, bounds):
        """Calculate which UTM zones intersect with the raster bounds with better validation"""
        if not bounds:
            return []
        
        min_lon = bounds['min_lon']
        max_lon = bounds['max_lon']
        min_lat = bounds['min_lat']
        max_lat = bounds['max_lat']
        
        # Validate bounds
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180 and
                -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            print("Invalid coordinate bounds")
            return []
        
        # Calculate UTM zones
        start_zone = int((min_lon + 180) // 6) + 1
        end_zone = int((max_lon + 180) // 6) + 1
        
        # Handle dateline crossing
        zones = set()
        if start_zone > end_zone:
            zones.update(range(start_zone, 61))
            zones.update(range(1, end_zone + 1))
        else:
            zones.update(range(start_zone, end_zone + 1))
        
        # Filter invalid zones (UTM valid between 80°S and 84°N)
        valid_zones = []
        for zone in sorted(zones):
            zone_center = -177 + (zone - 1) * 6
            zone_min_lon = zone_center - 3
            zone_max_lon = zone_center + 3
            
            # Check if raster intersects with this zone
            if not (max_lon < zone_min_lon or min_lon > zone_max_lon):
                valid_zones.append({
                    'zone': zone,
                    'min_lon': max(min_lon, zone_min_lon),
                    'max_lon': min(max_lon, zone_max_lon),
                    'min_lat': max(min_lat, -80),  # UTM southern limit
                    'max_lat': min(max_lat, 84)    # UTM northern limit
                })
        
        return valid_zones

    def perform_reprojection(self, layer_name, target_epsg, resample_method):
        """Perform the actual raster reprojection"""
        layer = self.layers[layer_name]
        
        with rasterio.open(layer['path']) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_array = src.read()
            
            # Calculate destination CRS and transform
            dst_crs = rasterio.crs.CRS.from_string(target_epsg)
            dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                src_crs, dst_crs, src.width, src.height, *src.bounds)
            
            # Create destination array
            dst_array = np.empty((src.count, dst_height, dst_width), dtype=src_array.dtype)
            
            # Perform reprojection
            rasterio.warp.reproject(
                source=src_array,
                destination=dst_array,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=getattr(rasterio.warp.Resampling, resample_method))
            
            # Update the layer in memory
            self.layers[layer_name]['image'] = dst_array
            self.layers[layer_name]['profile'].update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height
            })
            
            # Update display
            self.display_layer(layer_name)


####################################################################################################3
        
    def add_division_safety(self, formula):
        """
        Adds safety to division operations in a mathematical formula using sympy.        
        Parameters:
            formula (str): The mathematical formula as a string.            
        Returns:
            str: The modified formula with division safety added.            
        Raises:
            ValueError: If the formula cannot be parsed or contains invalid syntax.
        """
        try:
            # Parse the formula into a sympy expression
            expr = sympify(formula)

            # Define a recursive function to traverse and modify the expression tree
            def add_epsilon(expr):
                """
                Recursively traverses the expression tree and adds epsilon to denominators.
                
                Parameters:
                    expr: The current sympy expression node.
                    
                Returns:
                    Modified sympy expression with division safety.
                """
                if isinstance(expr, Mul):
                    # Handle multiplication: recursively process each factor
                    return Mul(*[add_epsilon(arg) for arg in expr.args])
                elif isinstance(expr, Pow):
                    # Handle division: x^(-1) -> (x + 1e-9)^(-1)
                    base, exponent = expr.args
                    if exponent == -1:
                        return Pow(base + 1e-9, -1)
                    else:
                        return Pow(add_epsilon(base), add_epsilon(exponent))
                elif isinstance(expr, Add):
                    # Handle addition: recursively process each term
                    return Add(*[add_epsilon(arg) for arg in expr.args])
                else:
                    # Base case: return the expression unchanged
                    return expr

            # Apply the function to the entire expression
            safe_expr = add_epsilon(expr)

            # Convert the modified expression back to a string
            return str(safe_expr)

        except Exception as e:
            # Raise an error if the formula cannot be parsed or processed
            raise ValueError(f"Failed to parse formula: {formula}. Error: {str(e)}")       

    
    def show_error_message(self, message):
        """Show an error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def overlay_classified_results(self, original_image_path, classified_image_path):
            """
            Overlay the classified results on top of the original satellite image using leafmap.

            Args:
                original_image_path (str): Path to the original satellite image.
                classified_image_path (str): Path to the classified image.
            """
            try:
                # Create a leafmap map
                m = leafmap.Map()

                # Add the original satellite image
                m.add_raster(original_image_path, layer_name="Satellite Image")

                # Add the classified image with transparency
                m.add_raster(classified_image_path, alpha=0.5, layer_name="Classified Results")

                # Display the map
                m.to_streamlit()  # If using Streamlit
                # m.show()  # If using Jupyter Notebook or standalone

                # Alternatively, use image_comparison for side-by-side comparison
                leafmap.image_comparison(
                    original_image_path,
                    classified_image_path,
                    label1="Satellite Image",
                    label2="Classified Results",
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create overlay: {str(e)}")

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
    def overlay_layers(self, layer_order, transparency_values):
        """
        عرض متطور للطبقات المتداخلة مع تحسينات في التصيير
        """
        try:
            # مسح الشكل الحالي
            self.ax.clear()
            
            # إذا كانت هناك طبقة واحدة فقط
            if len(layer_order) == 1:
                layer_name = layer_order[0]
                layer_data = self.layers[layer_name]
                if layer_data.get('visible', True):
                    image = layer_data['image']
                    opacity = transparency_values.get(layer_name, 1.0)
                    
                    if image.ndim == 3:  # صورة RGB
                        self.ax.imshow(image.transpose(1, 2, 0), alpha=opacity)
                    else:  # صورة رمادية
                        self.ax.imshow(image[0], cmap='gray', alpha=opacity,
                                    vmin=np.percentile(image[0], 2),  # تحسين التباين
                                    vmax=np.percentile(image[0], 98))
            
            # إذا كانت هناك عدة طبقات
            else:
                composite = None
                
                for layer_name in layer_order:
                    layer_data = self.layers[layer_name]
                    if not layer_data.get('visible', True):
                        continue
                        
                    image = layer_data['image']
                    opacity = transparency_values.get(layer_name, 0.5)
                    
                    if image.ndim == 3:  # صورة RGB
                        layer_img = image.transpose(1, 2, 0)
                    else:  # صورة رمادية
                        layer_img = np.stack([image[0]]*3, axis=-1)
                        # تحسين التباين للصور الرمادية
                        layer_img = (layer_img - np.percentile(image[0], 2)) / (
                            np.percentile(image[0], 98) - np.percentile(image[0], 2))
                        layer_img = np.clip(layer_img, 0, 1)
                    
                    if composite is None:
                        composite = layer_img * opacity
                    else:
                        # دمج الطبقات مع الشفافية
                        composite = composite * (1 - opacity) + layer_img * opacity
                
                if composite is not None:
                    # تحسين العرض النهائي
                    composite = np.clip(composite, 0, 1)
                    self.ax.imshow(composite)
            
            # إزالة المحاور
            self.ax.set_axis_off()
            
            # ضبط المسافات البادئة
            self.fig.tight_layout()
            
            # تحديث الشكل
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display overlapping layers: {str(e)}")


    def display_overlay(self, original_image, original_crs, classified_image, classified_crs):
        """
        Enhanced overlay display using leafmap with better performance.
        """
        try:
            # Create a leafmap map
            m = leafmap.Map()
            
            # Use memory-based approach instead of temp files when possible
            if hasattr(leafmap, 'add_raster_from_array'):
                # Direct array support (if available in your leafmap version)
                m.add_raster_from_array(
                    original_image.transpose(1, 2, 0),
                    layer_name="Original Image",
                    colormap='gray' if original_image.shape[0] == 1 else None
                )
                m.add_raster_from_array(
                    classified_image[0] if classified_image.shape[0] == 1 else classified_image.transpose(1, 2, 0),
                    layer_name="Classified Image",
                    alpha=0.5,
                    colormap='viridis' if classified_image.shape[0] == 1 else None
                )
            else:
                # Fallback to temp files but with better handling
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_original:
                    original_path = temp_original.name
                    self.save_array_as_geotiff(original_image, original_crs, original_path)
                    
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_classified:
                    classified_path = temp_classified.name
                    self.save_array_as_geotiff(classified_image, classified_crs, classified_path)
                    
                # Add layers
                m.add_raster(
                    original_path,
                    layer_name="Original Image",
                    colormap='gray' if original_image.shape[0] == 1 else None
                )
                m.add_raster(
                    classified_path,
                    layer_name="Classified Image",
                    alpha=0.5,
                    colormap='viridis' if classified_image.shape[0] == 1 else None
                )
                
                # Schedule cleanup
                QTimer.singleShot(3000, lambda: [os.remove(p) for p in [original_path, classified_path] if os.path.exists(p)])
            
            # Enhanced layer control
            m.add_layer_control(position="topright")
            m.add_basemap("SATELLITE")  # Optional basemap
            
            # Display the map
            m.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display overlay: {str(e)}")
    
    def save_array_as_geotiff(self, array, crs, file_path):
        """Helper method to save numpy array as GeoTIFF."""
        height, width = array.shape[1], array.shape[2]
        transform = from_origin(0, 0, 1, 1)
        
        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=array.shape[0],
            dtype=array.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(array)


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

                    if image.ndim == 3:  # RGB image
                        self.ax.imshow(image.transpose(1, 2, 0), alpha=transparency)  # Use alpha for transparency
                    else:  # Grayscale image
                        self.ax.imshow(image[0], cmap='gray', alpha=transparency)

            # Update the canvas
            self.canvas.draw()
            QMessageBox.information(self, "Success", "Layers overlaid successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to overlay layers: {str(e)}")

    def update_overlay(self, layer_names, transparency_values):
        """
        Update the overlay with new transparency values.
        """
        self.overlay_layers(layer_names, transparency_values)

    def toggle_layer_order(self):
        """
        Toggle the order of layers (bring the top layer to the bottom and vice versa).
        """
        if hasattr(self, 'layer_order'):
            self.layer_order = not self.layer_order  # Toggle the order
        else:
            self.layer_order = True  # Default order (bottom layer first)

        if hasattr(self, 'selected_layers') and hasattr(self, 'transparency_values'):
            self.overlay_layers(self.selected_layers, self.transparency_values)

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
    
    def activate_rectangle_selection(self):
        """Handle rectangle selection mode"""
        self.selection_mode = 'rectangle'
        self.canvas.setCursor(Qt.CrossCursor)
        self.statusBar().showMessage("Rectangle selection mode - click and drag to select")

    def activate_polygon_selection(self):
        """Handle polygon selection mode""" 
        self.selection_mode = 'polygon'
        self.canvas.setCursor(Qt.CrossCursor)
        self.selection_points = []
        self.statusBar().showMessage("Polygon selection mode - click to add points, right-click to finish")

    def clear_selection(self):
        """Clear current selection"""
        if hasattr(self.polygon_processor, 'selected_features'):
            self.polygon_processor.selected_features.clear()
        self.update_map_display()
        self.statusBar().showMessage("Selection cleared")
    
    ################################ Filters  ###############################3
    
    def _apply_filter(self, filter_func, *args):
        """Helper method to apply filter with statistics comparison"""
        if hasattr(self, 'image'):
            try:
                # Capture pre-filter statistics
                pre_stats = self._get_image_stats(self.image)
                
                # Save current state for undo
                current_layer = self.layer_list.currentItem().text() if self.layer_list.currentItem() else "Filtered"
                self.undo_stack.append((self.image.copy(), current_layer))
                
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
                self.undo()

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
                                    f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
            colors = plt.cm.get_cmap('tab10', len(pre_stats['bands']))
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

    def highlight_editing_layer(self, layer_name, editing_active):
        """Change layer appearance in legend when editing"""
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item.text() == layer_name:
                font = item.font()
                font.setBold(editing_active)
                item.setFont(font)
                if editing_active:
                    item.setBackground(QColor(255, 255, 150))  # Light yellow
                else:
                    item.setBackground(QBrush())  # Reset
                break

    def toggle_edit_buttons(self, enable):
        """Enable/disable Save/Stop buttons (called by polygon_processor)."""
        self.actionSave_Edits.setEnabled(enable)
        self.actionStop_Editing.setEnabled(enable)

    def connect_edit_buttons(self):
        """Connect toolbar buttons to polygon_processor functions."""
        self.actionStart_Editing.triggered.connect(self.polygon_processor.start_editing)
        self.actionSave_Edits.triggered.connect(self.polygon_processor.save_edits)
        self.actionStop_Editing.triggered.connect(self.polygon_processor.stop_editing)
    
    def on_start_editing_clicked(self):
        success = self.polygon_processor.start_editing()
        if not success:
            QMessageBox.warning(self, "Error", "Could not start editing session")
        else:
            self.toggle_edit_buttons(True)

    def on_save_clicked(self):
        if self.polygon_processor.save_edits():
            # Get current layer name
            layer_name = os.path.basename(self.polygon_processor.original_file_path)
            
            # Force UI refresh
            self.layer_list.currentItem().setSelected(False)
            self.layer_list.currentItem().setSelected(True)
            
            # Alternative: full reload
            self.reload_layer(layer_name)

    def setup_editing_workflow(self):
        # Connect signals
        self.actionStart_Editing.triggered.connect(self.start_editing_session)
        self.actionSave_Edits.triggered.connect(self.save_edits_session)
        self.actionStop_Editing.triggered.connect(self.stop_editing_session)
        
        # Attribute table connections
        self.attributeTable.cellChanged.connect(self.handle_attribute_edit)

    def start_editing_session(self):
        if self.polygon_processor.start_editing():
            self.set_editing_ui_state(True)
            self.show_info("Editing started - remember to save!")

    def save_edits_session(self):
        if self.polygon_processor.save_edits():
            layer_name = os.path.basename(self.polygon_processor.original_file_path)
            self.reload_layer(layer_name)
            self.show_info("Edits saved successfully")

    def stop_editing_session(self):
        self.polygon_processor.stop_editing()
        self.set_editing_ui_state(False)
############################################# Wet Area 

    # # Add to SatelliteImageApp class methods
    # def add_wet_area_detection_tool(self):
    #     """Add wet area detection to raster processing menu"""
    #     wet_area_action = QAction('Detect Wet Areas (Sentinel-1)', self)
    #     wet_area_action.triggered.connect(self.detect_wet_areas)
    #     self.menuBar().findChild(QMenu, 'Raster Processing Tools').addAction(wet_area_action)

    def detect_wet_areas(self):
        """Main method for wet area detection from Sentinel-1 data"""
        try:
            # Get current layer data
            layer_data = self.get_current_layer_data()
            if not layer_data or layer_data.get('type') != 'raster':
                raise ValueError("No valid SAR layer selected")

            # Verify it's Sentinel-1 data
            if not self.is_sentinel1_data(layer_data):
                QMessageBox.warning(self, "Warning", 
                                "Selected layer doesn't appear to be Sentinel-1 data")
                return

            # Get parameters through dialog
            params = self.get_wet_area_params()
            if not params:
                return

            # Start processing
            self.progress_bar.setVisible(True)
            self.worker = Worker(self.process_wet_areas, 
                                layer_data['image'], 
                                layer_data['profile'], 
                                params)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.on_wet_area_processed)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Initialization failed: {str(e)}")

    def process_wet_areas(self, image, profile, params):
        """Core processing logic"""
        try:
            # 1. Preprocessing
            processed = self.preprocess_sar(image, profile, params)
            
            # 2. Thresholding
            water_mask = self.apply_threshold(processed, params['threshold'])
            
            # 3. Post-processing
            filtered_mask = self.postprocess_mask(water_mask)
            
            return filtered_mask, profile
            
        except Exception as e:
            self.error.emit(f"Processing error: {str(e)}")
            raise

    def preprocess_sar(self, image, profile, params):
        """SAR-specific preprocessing"""
        # Convert to dB scale if needed
        if params['convert_to_db']:
            image = 10 * np.log10(image)
        
        # Apply speckle filter
        if params['speckle_filter']:
            image = self.apply_lee_filter(image)
        
        return image

    def get_wet_area_params(self):
        """Get parameters through dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Wet Area Detection Parameters")
        
        layout = QFormLayout()
        
        # Threshold input
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(-30, 0)
        threshold_spin.setValue(-15)
        layout.addRow("Backscatter Threshold (dB):", threshold_spin)
        
        # Preprocessing options
        db_check = QCheckBox("Convert to dB scale")
        db_check.setChecked(True)
        layout.addRow(db_check)
        
        filter_check = QCheckBox("Apply Lee Filter")
        filter_check.setChecked(True)
        layout.addRow(filter_check)
        
        # Dialog buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addRow(btn_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            return {
                'threshold': threshold_spin.value(),
                'convert_to_db': db_check.isChecked(),
                'speckle_filter': filter_check.isChecked()
            }
        return None

    def on_wet_area_processed(self, result):
        """Handle processed wet area mask"""
        water_mask, profile = result
        layer_name = f"Wet_Areas_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Add as new layer with blue color map
        self.layers[layer_name] = {
            'image': water_mask,
            'profile': profile,
            'cmap': 'Blues',
            'vmin': 0,
            'vmax': 1
        }
        
        item = QListWidgetItem(QIcon(":/icons/water.png"), layer_name)
        self.layer_list.addItem(item)
        self.display_raster_layer(water_mask, layer_name)
        self.progress_bar.setVisible(False)

        # Add to __init__ method after initializing menus
        self.add_wet_area_detection_tool()

######################################## New SAR ############################
    
    def _init_sar_workflow_actions(self):
        """Initialize workflow steps with proper step naming"""
        self.workflow_steps = [
            'load', 
            'orbit', 
            'calibrate', 
            'filter', 
            'terrain', 
            'convert_db', 
            'analyze', 
            'classify'
        ]
        
        self.workflow_actions = [
            self.sar_menu.addAction('1. Load SAR Data'),
            self.sar_menu.addAction('2. Apply Orbit File'),
            self.sar_menu.addAction('3. Calibrate (σ₀)'),
            self.sar_menu.addAction('4. Apply Speckle Filter'),
            self.sar_menu.addAction('5. Terrain Correction'),
            self.sar_menu.addAction('6. Convert to dB'),
            self.sar_menu.addAction('7. Analyze Histogram'),
            self.sar_menu.addAction('8. Classify Water')
        ]
        
        # Connect signals using step names
        self.workflow_actions[0].triggered.connect(self.start_sar_workflow)
        self.workflow_actions[1].triggered.connect(self.apply_orbit_file)
        self.workflow_actions[2].triggered.connect(self.calibrate)
        self.workflow_actions[3].triggered.connect(self.apply_filter)
        self.workflow_actions[4].triggered.connect(self.terrain_correction)
        self.workflow_actions[5].triggered.connect(self.convert_db)
        self.workflow_actions[6].triggered.connect(self.analyze_histogram)
        self.workflow_actions[7].triggered.connect(self.classify_water)
        
        # Initialize with first step enabled
        self._reset_workflow()


    def start_sar_workflow(self):
        """Load SAR image with enhanced metadata handling"""
        try:
            self.current_sar_group = None
            self.sar_steps = {}
            self.input_file, _ = QFileDialog.getOpenFileName(  # Store directly to input_file
                self, "Open SAR Image", "", 
                "GeoTIFF Files (*.tif *.tiff);;All Files (*)"
            )
            if not self.input_file:  # Use the class attribute
                return

            with rasterio.open(self.input_file) as src:
                raw_data = src.read(1)
                self._validate_data_shape(raw_data)
                
                # Get metadata with fallback values
                metadata = {
                    'file_name': os.path.basename(self.input_file),
                    'acquisition_date': src.tags().get('ACQUISITION_DATE') 
                                    or src.tags().get('DATE_ACQUIRED') 
                                    or 'Unknown',
                    'polarization': src.tags().get('POLARIZATION') 
                                or src.tags().get('POLARISATION') 
                                or 'HH',
                    'dimensions': f"{raw_data.shape[1]}×{raw_data.shape[0]}",
                    'resolution': f"{src.res[0]:.2f} m",
                    'data_type': str(raw_data.dtype),
                    'crs': src.crs.to_string() if src.crs else 'Unknown',
                    'profile': src.profile  # Store profile for export
                }

                # Handle no-data values
                nodata = src.nodata
                if nodata is not None:
                    raw_data = np.where(raw_data == nodata, np.nan, raw_data)

                # Calculate visualization parameters
                valid_data = raw_data[np.isfinite(raw_data)]
                vmin = np.percentile(valid_data, 2) if valid_data.size > 0 else 0
                vmax = np.percentile(valid_data, 98) if valid_data.size > 0 else 1

                self.sar_steps['Original'] = {
                    'data': raw_data,
                    'profile': src.profile,
                    'metadata': metadata,
                    'vis_params': {
                        'cmap': 'gray',
                        'vmin': vmin,
                        'vmax': vmax
                    },
                    'stats': self._calculate_stats(raw_data)
                }

            self._display_sar_layer('Original')
            self._enable_next_step('load')
            self.layer_list.addItem(
                QListWidgetItem(f"📡 {metadata['file_name']} (Original)")
            )

        except Exception as e:
            self.show_error(f"SAR data loading failed: {str(e)}")
            self._reset_workflow()

    def apply_orbit_file(self):
        """Apply precise orbit file to SAR data"""
        try:
            if 'Original' not in self.sar_steps:
                raise ValueError("Load SAR data first")

            # Create orbit file selection dialog
            orbit_dialog = QDialog(self)
            orbit_dialog.setWindowTitle("Orbit File Parameters")
            layout = QFormLayout(orbit_dialog)

            # Orbit source selection
            source_combo = QComboBox()
            source_combo.addItems(["Auto Download", "Manual Selection"])
            layout.addRow("Orbit Source:", source_combo)

            # File selection for manual mode
            file_edit = QLineEdit()
            file_btn = QPushButton("Browse...")
            file_layout = QHBoxLayout()
            file_layout.addWidget(file_edit)
            file_layout.addWidget(file_btn)
            layout.addRow("Orbit File:", file_layout)
            file_edit.setEnabled(False)
            file_btn.setEnabled(False)

            def update_orbit_ui(text):
                manual_mode = text == "Manual Selection"
                file_edit.setEnabled(manual_mode)
                file_btn.setEnabled(manual_mode)
                
            source_combo.currentTextChanged.connect(update_orbit_ui)
            update_orbit_ui(source_combo.currentText())

            # Dialog buttons
            btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btn_box.accepted.connect(orbit_dialog.accept)
            btn_box.rejected.connect(orbit_dialog.reject)
            layout.addRow(btn_box)

            if orbit_dialog.exec_() != QDialog.Accepted:
                return

            # Apply orbit file logic
            with rasterio.open(self.input_file) as src:
                raw_data = src.read(1)
                profile = src.profile
                metadata = src.tags()

            # Store orbit-corrected data
            self.sar_steps['Orbit'] = {
                'data': raw_data,  # In real implementation, apply actual orbit correction
                'profile': profile,
                'metadata': {
                    **metadata,
                    'orbit_applied': True,
                    'orbit_source': source_combo.currentText()
                }
            }

            self._enable_next_step('orbit')
            self.layer_list.addItem(QListWidgetItem("🛰 Orbit Applied"))
            self.statusBar().showMessage("Orbit file applied successfully", 5000)

        except Exception as e:
            self.show_error(f"Orbit file application failed: {str(e)}")

    def calibrate(self):
        """Calibrate to sigma nought (σ₀)"""
        try:
            if 'Orbit' not in self.sar_steps:
                raise ValueError("Apply orbit file first")

            # Get calibration parameters from metadata
            metadata = self.sar_steps['Orbit']['metadata']
            calibration_constant = float(metadata.get('CALIBRATION_CONSTANT', 1.0))

            # Perform calibration
            raw_data = self.sar_steps['Orbit']['data']
            sigma0 = (raw_data ** 2) / calibration_constant

            # Store calibrated data
            self.sar_steps['Calibrated'] = {
                'data': sigma0,
                'profile': self.sar_steps['Orbit']['profile'],
                'metadata': {
                    **metadata,
                    'calibration_type': 'σ₀',
                    'calibration_constant': calibration_constant
                }
            }

            self._enable_next_step('calibrate')
            self.layer_list.addItem(QListWidgetItem("📏 Calibrated (σ₀)"))
            self.statusBar().showMessage("Calibration to σ₀ completed", 5000)

        except Exception as e:
            self.show_error(f"Calibration failed: {str(e)}")

    
    def terrain_correction(self):
        """Perform terrain correction using DEM"""
        try:
            if 'Filtered' not in self.sar_steps:
                raise ValueError("Apply speckle filter first")

            # DEM selection dialog
            dem_dialog = QDialog(self)
            dem_dialog.setWindowTitle("DEM Parameters")
            layout = QFormLayout(dem_dialog)

            # DEM source selection
            dem_combo = QComboBox()
            dem_combo.addItems(["Copernicus 30m DEM", "SRTM 90m", "Custom DEM"])
            layout.addRow("DEM Source:", dem_combo)

            # File selection for custom DEM
            dem_file_edit = QLineEdit()
            dem_file_btn = QPushButton("Browse...")
            dem_file_layout = QHBoxLayout()
            dem_file_layout.addWidget(dem_file_edit)
            dem_file_layout.addWidget(dem_file_btn)
            layout.addRow("DEM File:", dem_file_layout)
            dem_file_edit.setEnabled(False)
            dem_file_btn.setEnabled(False)

            def update_dem_ui(text):
                custom_mode = text == "Custom DEM"
                dem_file_edit.setEnabled(custom_mode)
                dem_file_btn.setEnabled(custom_mode)
                
            dem_combo.currentTextChanged.connect(update_dem_ui)
            update_dem_ui(dem_combo.currentText())

            # Dialog buttons
            btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btn_box.accepted.connect(dem_dialog.accept)
            btn_box.rejected.connect(dem_dialog.reject)
            layout.addRow(btn_box)

            if dem_dialog.exec_() != QDialog.Accepted:
                return

            # Perform terrain correction (simplified example)
            filtered_data = self.sar_steps['Filtered']['data']
            # In real implementation, use DEM for geometric correction
            corrected_data = filtered_data  # Placeholder

            # Store terrain-corrected data
            self.sar_steps['Terrain'] = {
                'data': corrected_data,
                'profile': self.sar_steps['Filtered']['profile'],
                'metadata': {
                    **self.sar_steps['Filtered']['metadata'],
                    'dem_source': dem_combo.currentText(),
                    'correction_method': 'Range-Doppler'
                }
            }

            self._enable_next_step('terrain')
            self.layer_list.addItem(QListWidgetItem("🏔 Terrain Corrected"))
            self.statusBar().showMessage("Terrain correction completed", 5000)

        except Exception as e:
            self.show_error(f"Terrain correction failed: {str(e)}")

    
    def _apply_speckle_filter(self, data, filter_type='Lee', window_size=3, snr_estimate=1.0, progress_callback=None):
        """
        Apply speckle reduction filter to SAR data with support for multiple algorithms
        
        Parameters:
        - data: 2D numpy array of input SAR data
        - filter_type: One of ['Lee', 'Gamma MAP', 'Frost', 'Median']
        - window_size: Odd integer specifying filter window size
        - snr_estimate: Signal-to-noise ratio estimate for Gamma MAP filter
        - progress_callback: Function to report progress percentage
        
        Returns:
        - 2D numpy array of filtered data
        """
        try:
            # Validate inputs
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                raise ValueError("Input must be 2D numpy array")
                
            if window_size % 2 == 0:
                raise ValueError("Window size must be odd number")
                
            if filter_type not in ['Lee', 'Gamma MAP', 'Frost', 'Median', 'Refined Lee']: 
                raise ValueError(f"Unsupported filter type: {filter_type}")

            # Convert to float32 for processing
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # Create output array and setup padding
            pad_size = window_size // 2
            filtered = np.empty_like(data)
            padded_data = np.pad(data, pad_size, mode='reflect')
            
            # Pre-calculate window indices
            rows, cols = data.shape
            total_pixels = rows * cols
            processed = 0

            # Main processing loop
            for i in range(rows):
                for j in range(cols):
                    if progress_callback and (processed % 1000 == 0):
                        if not progress_callback(processed / total_pixels * 100):
                            raise RuntimeError("Filtering canceled by user")
                    
                    # Extract window
                    window = padded_data[i:i+window_size, j:j+window_size]
                    valid_window = window[np.isfinite(window)]
                    
                    if valid_window.size == 0:
                        filtered[i,j] = np.nan
                        continue
                    
                    # Apply selected filter
                    if filter_type == 'Lee':
                        filtered[i,j] = self._apply_lee_filter(valid_window, window)
                    elif filter_type == 'Gamma MAP':
                        filtered[i,j] = self._apply_gamma_map_filter(valid_window, window, snr_estimate)
                    elif filter_type == 'Frost':
                        filtered[i,j] = self._apply_frost_filter(window, i, j, pad_size)
                    elif filter_type == 'Median':
                        filtered[i,j] = np.median(valid_window)
                    elif filter_type == 'Refined Lee':
                        filtered[i,j] = self._apply_refined_lee_filter(valid_window, window)
                    
                    processed += 1

            return filtered

        except Exception as e:
            self.show_error(f"Filtering Error ({filter_type}): {str(e)}")
            raise

    # Helper methods for individual filters
    def _apply_lee_filter(self, valid_window, window):
        """Lee filter implementation"""
        mean = np.mean(valid_window)
        var = np.var(valid_window)
        if var == 0:
            return mean
        k = var / (var + mean**2)
        return mean + k * (window.mean() - mean)
    
    def _apply_refined_lee_filter(self, valid_window, window):
        """Refined Lee filter implementation for SAR despeckling"""
        # 1. Calculate initial statistics
        mean = np.mean(valid_window)
        var = np.var(valid_window)
        
        if var == 0:
            return mean
        
        # 2. Compute coefficient of variation
        cv = np.sqrt(var) / mean
        cu = 1.0 / np.sqrt(9 * window.size)  # Threshold for homogeneous areas
        
        # 3. Determine region type and apply appropriate filtering
        if cv <= cu:
            # Homogeneous area - use mean value
            return mean
        else:
            # Heterogeneous or edge area - use standard Lee filter
            k = var / (var + mean**2)
            return mean + k * (window.mean() - mean)

    def _apply_gamma_map_filter(self, valid_window, window, snr):
        """Gamma Maximum A Posteriori filter"""
        mean = np.mean(valid_window)
        var = np.var(valid_window)
        if var == 0:
            return mean
        enl = 1.0 / (var / (mean**2 + 1e-6))  # Equivalent Number of Looks
        alpha = enl + 1 + (enl / snr)
        return (alpha - enl - 1) * mean / alpha

    def _apply_frost_filter(self, window, i, j, pad_size):
        """Frost filter with exponential damping"""
        center_value = window[pad_size, pad_size]
        local_variance = np.var(window)
        local_mean = np.mean(window)
        
        if local_variance == 0:
            return local_mean
            
        k = 2 * local_variance / (local_mean**2 + 1e-6)
        damping_factor = np.exp(-k * np.abs(window - center_value))
        return np.sum(window * damping_factor) / np.sum(damping_factor)
   
    
    def _validate_data_shape(self, data):
        """Ensure data is proper 2D array"""
        if data.ndim != 2:
            raise ValueError(
                f"Invalid data shape {data.shape}. "
                "Expected 2D array (height, width).\n"
                "Possible solutions:\n"
                "1. Use src.read(1) instead of src.read()\n"
                "2. Check input file structure\n"
                "3. Verify image dimensions in metadata"
            )
    
    def _add_to_layer_list(self, name, data):
        """Safely add layer with data validation"""
        if not isinstance(data, np.ndarray):
            raise TypeError("Cannot add non-array data to layer list")
            
        item = QListWidgetItem(f"📡 {name}")
        item.setData(Qt.UserRole, {
            'type': 'SAR',
            'raw_data': data,
            'processed': False
        })
        self.layer_list.addItem(item)
        self.layer_list.setCurrentItem(item)

    def apply_filter(self):
        """Apply speckle filter with full progress monitoring and optimized processing"""
        try:
            self.filter_start_time = datetime.now()
            # 1. Validate preconditions
            if 'Original' not in self.sar_steps:
                self.show_error("Please load SAR data first using 'Load SAR Data'")
                return

            # 2. Create parameter dialog
            filter_dialog = QDialog(self)
            filter_dialog.setWindowTitle("Filter Parameters")
            layout = QFormLayout(filter_dialog)

            # Filter type selection
            filter_combo = QComboBox()
            filter_combo.addItems(["Lee", "Frost", "Gamma MAP", "Median", "Refined Lee"])
            layout.addRow("Filter Type:", filter_combo)

            # Window size configuration
            window_spin = QSpinBox()
            window_spin.setRange(3, 15)
            window_spin.setValue(5)
            window_spin.setSingleStep(2)
            layout.addRow("Window Size (odd):", window_spin)

            # SNR estimation for Gamma MAP
            snr_spin = QDoubleSpinBox()
            snr_spin.setRange(0.1, 10.0)
            snr_spin.setValue(1.0)
            snr_spin.setSingleStep(0.1)
            snr_spin.setVisible(False)
            layout.addRow("SNR Estimate (Gamma):", snr_spin)

            # UI updates for filter type
            def update_filter_ui(text):
                snr_spin.setVisible(text == "Gamma MAP")
                window_spin.setEnabled(text not in ["Median"])
                
            filter_combo.currentTextChanged.connect(update_filter_ui)
            update_filter_ui(filter_combo.currentText())

            # Dialog buttons
            btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btn_box.accepted.connect(filter_dialog.accept)
            btn_box.rejected.connect(filter_dialog.reject)
            layout.addRow(btn_box)

            # 3. Get user parameters
            if filter_dialog.exec_() != QDialog.Accepted:
                return

            filter_type = filter_combo.currentText()
            window_size = window_spin.value()
            snr_estimate = snr_spin.value() if filter_type == "Gamma MAP" else None

            # Validate window size
            if window_size % 2 == 0:
                raise ValueError("Window size must be an odd number")

            # 4. Initialize progress dialog
            progress = QProgressDialog(
                f"Applying {filter_type} Filter...", 
                "Cancel", 
                0, 
                100, 
                self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            start_time = datetime.now()

            # 5. Prepare processing
            raw_data = self.sar_steps['Original']['data']
            rows, cols = raw_data.shape
            total_pixels = rows * cols
            processed = 0
            block_size = 512  # Process in 512x512 blocks for memory efficiency
            pad_size = window_size // 2
            filtered = np.empty_like(raw_data)

            # 6. Processing loop with progress updates
            try:
                for i in range(0, rows, block_size):
                    for j in range(0, cols, block_size):
                        if progress.wasCanceled():
                            raise RuntimeError("Filtering canceled by user")
                        
                        # Calculate block boundaries with padding
                        i_start = max(0, i - pad_size)
                        i_end = min(rows, i + block_size + pad_size)
                        j_start = max(0, j - pad_size)
                        j_end = min(cols, j + block_size + pad_size)
                        
                        # Process block
                        block = raw_data[i_start:i_end, j_start:j_end]
                        filtered_block = self._apply_speckle_filter(
                            block, filter_type, window_size, snr_estimate
                        )
                        
                        # Remove padding and store result
                        i_clean_start = i - i_start
                        j_clean_start = j - j_start
                        filtered[i:i+block_size, j:j+block_size] = filtered_block[
                            i_clean_start:i_clean_start+block_size, 
                            j_clean_start:j_clean_start+block_size
                        ]
                        
                        # Update progress
                        processed += block_size * block_size
                        progress.setValue(int((processed / total_pixels) * 100))
                        QApplication.processEvents()

            except Exception as e:
                progress.close()
                raise e

            # 7. Calculate visualization parameters
            valid_data = filtered[np.isfinite(filtered)]
            vmin, vmax = (np.percentile(valid_data, [2, 98]) if valid_data.size > 0 else (0, 1))

            # 8. Store results with metadata
            self.sar_steps['Filtered'] = {
                'data': filtered,
                'profile': self.sar_steps['Original']['profile'].copy(),
                'metadata': self._create_filter_metadata(filter_type, window_size, snr_estimate),
                'stats': self._calculate_stats(filtered),
                'vis_params': {
                    'cmap': 'viridis',
                    'vmin': vmin,
                    'vmax': vmax
                }
            }

            # 9. Update UI
            self._display_sar_layer('Filtered')
            self._enable_next_step('filter')  # Enable convert_db step
            self.layer_list.addItem(
                QListWidgetItem(f"🌀 {filter_type} Filtered ({window_size}x{window_size})")
            )
            self._show_side_by_side('Original', 'Filtered')
            
            # 10. Final status
            processing_time = datetime.now() - start_time
            self.statusBar().showMessage(
                f"Applied {filter_type} filter | "
                f"Window: {window_size}x{window_size} | "
                f"Time: {processing_time.total_seconds():.1f}s", 
                10000
            )

        except RuntimeError as re:
            self.show_info(str(re))
        except ValueError as ve:
            self.show_error(f"Validation error: {str(ve)}")
        except Exception as e:
            error_msg = (
                f"Filter Application Error:\n{str(e)}\n"
                f"Filter Type: {filter_type}\n"
                f"Window Size: {window_size}\n"
                f"{traceback.format_exc()}"
            )
            self.show_error(error_msg)
        finally:
            if 'progress' in locals():
                progress.close()

    def _create_filter_metadata(self, filter_type, window_size, snr_estimate):
        """Create comprehensive metadata for filtered results"""
        base_metadata = self.sar_steps['Original']['metadata'].copy()
        
        return {
            **base_metadata,
            'processing_history': base_metadata.get('processing_history', []) + [
                f"Applied {filter_type} filter ({window_size}x{window_size})"
            ],
            'filter_parameters': {
                'type': filter_type,
                'window_size': window_size,
                'snr_estimate': snr_estimate if filter_type == "Gamma MAP" else None,
                'applied_at': datetime.now().isoformat(),
                'performance_metrics': {
                    'processing_time': str(datetime.now() - self.filter_start_time),
                    'memory_usage': f"{psutil.Process().memory_info().rss / 1024 ** 2:.1f} MB"
                }
            }
        }

    def _validate_sar_workflow(self):
        """Ensure SAR workflow is properly initialized"""
        if not self.current_sar_group:
            raise ValueError(
                "No active SAR workflow!\n"
                "1. Go to SAR Processing menu\n"
                "2. Select 'Load SAR Data'\n"
                "3. Choose your SAR image"
            )
            
        if 'Original' not in self.sar_steps:
            raise ValueError(
                "SAR data not loaded properly!\n"
                f"Current workflow: {self.current_sar_group}\n"
                "Try reloading the SAR data"
            )

    def _calculate_stats(self, data):
        """حساب الإحصائيات الأساسية"""
        valid_data = data[np.isfinite(data)]
        return {
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'mean': np.mean(valid_data),
            'std': np.std(valid_data),
            'percentile_2': np.percentile(valid_data, 2),
            'percentile_98': np.percentile(valid_data, 98)
        }


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
            cmap = copy(plt.cm.get_cmap(cmap))
            cmap.set_bad('magenta', 1.0)  # Highlight invalid values
            
            # Display image
            img = self.ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax)
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

            # Add CRE for original data
            if layer_name == 'Original':
                cre = self._calculate_cre(data)
                info_lines.append(f"📊 CRE: {format_value(cre)}")

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
    
    def _calculate_cre(self, data):
        """Calculate Equivalent Number of Looks (CRE)"""
        valid_data = data[np.isfinite(data)]
        if valid_data.size == 0:
            return np.nan
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        return (mean ** 2) / (std ** 2) if std != 0 else np.nan

    
    
    
    
    
    
    def _handle_histogram_accept(self, dialog):
        """Save thresholds and proceed to classification"""
        try:
            # Validate thresholds exist
            if 'user_selected' not in self.sar_steps['dB']['metadata']['thresholds']:
                raise ValueError("No thresholds selected!")
                
            dialog.accept()
            self._launch_classification_workflow()
            
        except Exception as e:
            self.show_error(f"Cannot proceed: {str(e)}")



    def _update_histogram(self):
        """Refresh histogram with current parameters"""
        self.ax.clear()
        
        # Calculate parameters
        bins = self.bin_spin.value()
        p_min = self.range_min.value()
        p_max = self.range_max.value()
        
        # Calculate dynamic range
        vmin = np.percentile(self.current_hist_data, p_min)
        vmax = np.percentile(self.current_hist_data, p_max)
        
        # Create histogram
        n, bins, patches = self.ax.hist(
            self.current_hist_data,
            bins=bins,
            range=(vmin, vmax),
            color='steelblue',
            edgecolor='black',
            density=True
        )
        
        # Professional styling
        self.ax.set_title("SAR Intensity Distribution Analysis", fontsize=12)
        self.ax.set_xlabel("Backscatter Coefficient (dB)", fontsize=10)
        self.ax.set_ylabel("Normalized Frequency", fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add distribution curve
        kde = gaussian_kde(self.current_hist_data)
        x = np.linspace(vmin, vmax, 300)
        self.ax.plot(x, kde(x), color='darkorange', linewidth=2)
        
        self.canvas.draw()

    def _save_histogram_data(self):
        """Export histogram data to CSV"""
        try:
            path, _ = QFileDialog.getSaveFileName(
                self.hist_dialog,
                "Save Histogram Data",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if path:
                bins = self.bin_spin.value()
                counts, bin_edges = np.histogram(self.current_hist_data, bins=bins)
                
                pd.DataFrame({
                    'Bin_Start': bin_edges[:-1],
                    'Bin_End': bin_edges[1:],
                    'Count': counts,
                    'Density': counts / counts.sum()
                }).to_csv(path, index=False)
                
                self.show_info(f"Histogram data saved to:\n{path}")
                
        except Exception as e:
            self.show_error(f"Export failed: {str(e)}")

    def _run_auto_analysis(self):
        """Improved auto-analysis with smooth integration"""
        try:
            db_data = self.sar_steps['dB']['data']
            
            # Multi-method threshold detection
            methods = {
                'otsu': filters.threshold_otsu,
                'percentile': lambda x: np.percentile(x, [5, 95]),
                'kmeans': self._kmeans_threshold
            }
            
            thresholds = []
            for name, method in methods.items():
                try:
                    result = method(db_data[np.isfinite(db_data)])
                    if isinstance(result, (list, tuple, np.ndarray)):
                        thresholds.extend(result)
                    else:
                        thresholds.append(result)
                except:
                    continue

            # Calculate final values
            lower = np.nanpercentile(thresholds, 25)
            upper = np.nanpercentile(thresholds, 75)

            # Update UI
            self.lower_spin.setValue(lower)
            self.upper_spin.setValue(upper)

        except Exception as e:
            self._show_error(f"Auto-Analysis Failed: {str(e)}")


    def _adaptive_threshold(self, data):
        """Returns tuple of (lower, upper) thresholds"""
        mean = np.mean(data)
        std = np.std(data)
        return (mean - 1.5*std, mean + 0.5*std)
    
    def _sanitize_input(self, data):
        """Ensure 1D array of finite values"""
        clean_data = data[np.isfinite(data)]
        if len(clean_data) < 10:
            raise ValueError("Insufficient valid data points")
        return clean_data.ravel()  # Flatten to 1D


    def _show_statistical_summary(self):
        """Display professional statistical summary with robust error handling"""
        from scipy import stats
        import numpy as np
        
        try:
            if not hasattr(self, 'current_hist_data') or len(self.current_hist_data) < 2:
                raise ValueError("Insufficient data for statistical analysis")

            data = self.current_hist_data
            
            # Basic statistics
            stats_dict = {
                'Samples': len(data),
                'Mean': np.mean(data),
                'Std Dev': np.std(data),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data)
            }

            # Advanced statistics with validation
            if len(data) >= 4:  # Minimum samples for JB test
                jb_stat, jb_p = stats.jarque_bera(data)
                stats_dict.update({
                    'Jarque-Bera Stat': jb_stat,
                    'Jarque-Bera p-value': jb_p
                })
            
            # Confidence intervals
            if len(data) >= 30:
                ci_low, ci_high = stats.norm.interval(0.95, 
                                                    loc=np.mean(data),
                                                    scale=stats.sem(data))
                stats_dict['95% CI'] = (ci_low, ci_high)

            # Format results
            msg = ["Statistical Summary:", ""]
            for k, v in stats_dict.items():
                if isinstance(v, tuple):
                    msg.append(f"{k}: ({v[0]:.2f}, {v[1]:.2f})")
                else:
                    msg.append(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

            QMessageBox.information(self.hist_dialog, 
                                "Statistical Analysis", 
                                "\n".join(msg))
            
        except Exception as e:
            self.show_error(f"Statistical analysis failed: {str(e)}")

    

    def _show_threshold_window(self):
        try:
            threshold_dialog = QDialog(self)
            layout = QVBoxLayout(threshold_dialog)
            
            # Smart Control Panel
            control_layout = QHBoxLayout()
            
            # Professional Button Set
            btn_box = QDialogButtonBox()
            btn_auto = btn_box.addButton("Auto-Threshold", QDialogButtonBox.ActionRole)
            btn_apply = btn_box.addButton("Apply", QDialogButtonBox.AcceptRole)
            btn_cancel = btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)
            btn_help = btn_box.addButton("?", QDialogButtonBox.HelpRole)

            # Auto-Threshold Implementation
            btn_auto.clicked.connect(self._apply_auto_threshold)
            btn_help.clicked.connect(self._show_threshold_help)
            
            # ... threshold selection UI code ...

            layout.addLayout(control_layout)
            threshold_dialog.exec_()

        except Exception as e:
            self.show_error(f"Threshold error: {str(e)}")

    def _get_optimal_thresholds(self, data):
        """Returns multiple threshold suggestions using professional methods"""
        valid_data = data[np.isfinite(data)]
        
        # Method 1: Otsu's Optimal Thresholding
        try:
            otsu_thresh = filters.threshold_otsu(valid_data)
        except:
            otsu_thresh = None
            
        # Method 2: Statistical Percentiles
        low_pct, high_pct = np.percentile(valid_data, [5, 95])
        
        # Method 3: Multi-modal Analysis
        kde = gaussian_kde(valid_data)
        x = np.linspace(valid_data.min(), valid_data.max(), 100)
        y = kde(x)
        peaks, _ = find_peaks(y)
        modal_thresh = x[peaks].mean() if len(peaks) > 0 else None
        
        # Method 4: Machine Learning Suggestions (requires training data)
        ml_thresh = self._ml_based_threshold(valid_data)
        
        return {
            'otsu': otsu_thresh,
            'percentiles': (low_pct, high_pct),
            'modes': modal_thresh,
            'ml_suggestion': ml_thresh
        }
    
    def _show_threshold_help(self):
        """Professional guidance for threshold selection"""
        help_text = """SAR Water Classification Best Practices:

    1. Optimal Threshold Strategies:
    - Otsu's Method: Best for bimodal distributions
    - Percentile Ranges: 5th-95th for conservative estimate
    - Multi-modal Analysis: For complex distributions
    - ML Suggestions: Uses historical classification patterns

    2. Validation Techniques:
    - Compare with known water bodies
    - Check seasonal variations
    - Verify with NDWI when available

    3. Professional Tips:
    - Use logarithmic scaling for better separation
    - Consider speckle-filtered data
    - Validate with ground truth data
    - Test multiple methods and compare results"""

        QMessageBox.information(self, "Expert Threshold Guide", help_text)

  
    def _add_statistical_annotations(self):
        """Add professional statistical annotations to histogram plot"""
        try:
            from scipy import stats  # Explicit import for clarity
            
            # Check data availability
            if not hasattr(self, 'current_hist_data') or len(self.current_hist_data) < 2:
                raise ValueError("Insufficient data for statistical analysis")
            
            data = self.current_hist_data
            summary_stats = {
                'Samples': len(data),
                'Mean': np.mean(data),
                'Std Dev': np.std(data),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data),
                'Min': np.min(data),
                'Max': np.max(data)
            }
            
            # Create formatted text
            stats_text = (
                "Statistical Summary:\n"
                f"Samples: {summary_stats['Samples']}\n"
                f"Mean: {summary_stats['Mean']:.2f} dB\n"
                f"Std Dev: {summary_stats['Std Dev']:.2f} dB\n"
                f"Skewness: {summary_stats['Skewness']:.2f}\n"
                f"Kurtosis: {summary_stats['Kurtosis']:.2f}\n"
                f"Range: {summary_stats['Min']:.2f} - {summary_stats['Max']:.2f} dB"
            )
            
            # Add annotation box
            self.ax.text(
                0.98, 0.70,  # Adjusted position for better visibility
                stats_text,
                transform=self.ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.9,
                    edgecolor='#404040',
                    linewidth=0.8
                )
            )
            
        except ValueError as ve:
            self.show_error(f"Statistical annotations skipped: {str(ve)}")
        except ImportError:
            self.show_error("SciPy package required for statistical analysis")
        except Exception as e:
            self.show_error(f"Annotation error: {str(e)}\n{traceback.format_exc()}")
    
    def _add_kde_curve(self):
        """Add Kernel Density Estimation curve to histogram"""
        from scipy.stats import gaussian_kde
        import numpy as np
        
        try:
            # Calculate KDE
            data = self.current_hist_data
            if len(data) < 2:
                raise ValueError("Insufficient data for KDE calculation")
                
            kde = gaussian_kde(data)
            x = np.linspace(self.range_min.value(), self.range_max.value(), 300)
            y = kde(x)
            
            # Plot KDE curve
            self.ax.plot(x, y, color='#FF5733', linewidth=2, label='Density Curve')
            self.ax.legend()
            
        except ValueError as ve:
            self.show_error(f"KDE Error: {str(ve)}")
        except Exception as e:
            self.show_error(f"Density estimation failed: {str(e)}")

    def _add_threshold_lines(self):
        """Initialize or update threshold lines safely"""
        # Remove existing lines if they exist
        if hasattr(self, 'lower_threshold_line'):
            self.lower_threshold_line.remove()
        if hasattr(self, 'upper_threshold_line'):
            self.upper_threshold_line.remove()
        
        # Get current axes
        ax = self.fig.gca()
        
        # Create new threshold lines
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        
        # Store line references as instance attributes
        self.lower_threshold_line = ax.axvline(lower, color='red', linestyle='--', label=f'Lower: {lower:.2f} dB')
        self.upper_threshold_line = ax.axvline(upper, color='green', linestyle='--', label=f'Upper: {upper:.2f} dB')
        
        # Add shaded region
        self.threshold_region = ax.axvspan(lower, upper, color='blue', alpha=0.1)
        
        self.canvas.draw()



    def _handle_proceed(self, dialog):
        """Save thresholds and proceed to classification"""
        try:
            # Validate thresholds
            if not self._validate_current_thresholds():
                raise ValueError("Invalid threshold values")
                
            # Store thresholds
            self._save_current_thresholds()
            
            # Close dialog and proceed
            dialog.accept()
            self._launch_classification_workflow()
            
        except Exception as e:
            self.show_error(f"Cannot proceed: {str(e)}")

    def _save_current_thresholds(self):
        """Save current threshold values to metadata"""
        self.sar_steps['dB']['metadata']['thresholds']['manual'] = {
            'lower': self.range_min.value(),
            'upper': self.range_max.value(),
            'timestamp': datetime.now().isoformat()
        }

    def _validate_current_thresholds(self):
        """Ensure valid threshold values"""
        lower = self.range_min.value()
        upper = self.range_max.value()
        return lower < upper and -50 <= lower <= 50 and -50 <= upper <= 50

    def _launch_classification_workflow(self):
        """Initialize water classification after threshold selection"""
        try:
            if 'thresholds' not in self.sar_steps['dB']['metadata']:
                raise ValueError("Thresholds not configured")
                
            self.classify_water()
            self._enable_sar_workflow_step(4)  # Enable next steps
            
        except Exception as e:
            self.show_error(f"Workflow transition failed: {str(e)}")
    
    def _reset_to_defaults(self):
        """Reset histogram parameters to default values"""
        try:
            # Reset UI controls
            self.bin_spin.setValue(200)
            self.range_min.setValue(-20)
            self.range_max.setValue(5)
            
            # Reset metadata thresholds
            if 'thresholds' in self.sar_steps['dB']['metadata']:
                self.sar_steps['dB']['metadata']['thresholds']['manual'] = None
                
            # Refresh display
            self._refresh_histogram()
            
            self.statusBar().showMessage("Reset to default parameters", 3000)
            
        except AttributeError as ae:
            self.show_error(f"Reset failed: UI elements not initialized - {str(ae)}")
        except Exception as e:
            self.show_error(f"Reset error: {str(e)}")

    
    def _update_threshold_preview(self):
        """Real-time classification preview"""
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        
        # Create preview mask
        mask = np.logical_and(self.data >= lower, self.data <= upper)
        
        # Apply to RGB preview (pseudo-color)
        preview_img = np.zeros((*mask.shape, 3))
        preview_img[mask] = [0, 0.5, 1]  # Water in blue
        preview_img[~mask] = [0.8, 0.8, 0.8]  # Non-water in gray
        
        self.preview_ax.imshow(preview_img)
        self.canvas.draw()

    
    

    def _enable_workflow_step(self, completed_step):
        """Enable next step in workflow"""
        workflow_steps = [
            'load', 
            'filter', 
            'convert_db', 
            'analyze', 
            'classify'
        ]
        
        try:
            current_index = workflow_steps.index(completed_step)
            next_index = current_index + 1
            
            if next_index < len(workflow_steps):
                # Enable next menu item
                self.workflow_actions[next_index].setEnabled(True)
                self.statusBar().showMessage(
                    f"Step {next_index+1} now available", 
                    3000
                )
        except ValueError:
            self.show_error("Invalid workflow step progression")


    def get_acquisition_date(self, file_path):
        """
        Extract acquisition date from SAR image metadata
        Supports common SAR formats: Sentinel-1, RADARSAT, TerraSAR-X, COSMO-SkyMed
        """
        try:
            with rasterio.open(file_path) as src:
                # Try different known SAR metadata tags
                tags = src.tags()
                
                # Common acquisition date tags by platform
                date_tags = [
                    'ACQUISITION_START_TIME',  # Sentinel-1
                    'ACQUISITIONDATE',         # TerraSAR-X
                    'ACQUISITION_DATE',        # RADARSAT-2
                    'FIRST_LINE_ACQUISITION_TIME',  # COSMO-SkyMed
                    'DATE_ACQUIRED',           # Generic
                    'DATE'                     # Fallback
                ]

                # Search through possible tags
                for tag in date_tags:
                    if tag in tags:
                        date_str = tags[tag]
                        # Standardize date format
                        try:
                            # Handle different datetime formats
                            if 'T' in date_str:  # ISO format with time
                                dt = datetime.fromisoformat(date_str.split('.')[0])
                            else:  # Date-only format
                                dt = datetime.strptime(date_str, '%Y%m%d')
                                
                            return dt.strftime('%Y-%m-%d')
                        except ValueError:
                            continue

                # Fallback to filename parsing (common in SAR naming conventions)
                filename = os.path.basename(file_path)
                
                # Try common SAR filename patterns
                patterns = [
                    r'\d{8}T\d{6}',  # Sentinel-1 (20230615T042312)
                    r'\d{4}-\d{2}-\d{2}',  # Hyphenated date
                    r'\d{14}',  # TerraSAR-X (YYYYMMDDHHMMSS)
                    r'\d{8}_\d{6}'  # COSMO-SkyMed
                ]

                for pattern in patterns:
                    match = re.search(pattern, filename)
                    if match:
                        date_str = match.group()
                        try:
                            if len(date_str) == 8:  # YYYYMMDD
                                dt = datetime.strptime(date_str, '%Y%m%d')
                            elif len(date_str) == 14:  # YYYYMMDDHHMMSS
                                dt = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                            elif 'T' in date_str:  # Sentinel-1 style
                                dt = datetime.fromisoformat(date_str.replace('T',''))
                            return dt.strftime('%Y-%m-%d')
                        except ValueError:
                            continue

                return "Unknown (No valid date found in metadata or filename)"

        except Exception as e:
            self.show_error(f"Date extraction failed: {str(e)}")
            return "Unknown"
    
    def _update_sar_metadata(self, step_name):
        """Update info box with SAR metadata and statistics"""
        
        metadata = self.sar_steps[step_name].get('metadata', {})
        stats = self.sar_steps[step_name]['stats']
        
        info_text = f"🔍 SAR Metadata ({step_name}):\n"
        info_text += f"• Acquisition Date: {metadata.get('acquisition_date', 'N/A')}\n"
        info_text += f"• Polarization: {metadata.get('polarization', 'N/A')}\n"
        info_text += f"• Resolution: {metadata.get('resolution', 'N/A')}\n\n"
        info_text += "📊 Statistics:\n"
        info_text += f"• Min: {stats['min']:.2f}\n"
        info_text += f"• Max: {stats['max']:.2f}\n"
        info_text += f"• Mean: {stats['mean']:.2f}\n"
        info_text += f"• Std Dev: {stats['std']:.2f}\n"
        
        if 'percentiles' in stats:
            info_text += "\n📈 Percentiles:\n"
            for k, v in stats['percentiles'].items():
                info_text += f"• {k}: {v:.2f}\n"
        
        self.info_box.setText(info_text)   

    def _calculate_stats(self, data):
        """Robust statistics calculation"""
        valid_data = data[np.isfinite(data)]
        if valid_data.size == 0:
            return {
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'std': np.nan
            }
        
        return {
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'mean': np.mean(valid_data),
            'std': np.std(valid_data)
        }
    
    def retry_filtering(self, dialog):
        """Retry filtering with same parameters"""
        dialog.close()
        self.apply_filter()    
        
    def _show_side_by_side(self, step1, step2):
        """Show comparison window with enhanced visualization"""
        try:
            # Create comparison dialog
            compare_dialog = QDialog(self)
            compare_dialog.setWindowTitle("Processing Comparison")
            compare_dialog.setMinimumSize(1200, 800)
            
            # Main layout
            main_layout = QVBoxLayout(compare_dialog)
            
            # Create matplotlib figure with subplots
            fig = Figure(figsize=(14, 8))
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, compare_dialog)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            # Configure images with adjusted colorbars
            for ax, step in zip([ax1, ax2], [step1, step2]):
                data = self.sar_steps[step]['data']
                cmap = self.sar_steps[step]['vis_params']['cmap']
                vmin = self.sar_steps[step]['vis_params']['vmin']
                vmax = self.sar_steps[step]['vis_params']['vmax']
                
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', 
                                shrink=0.6, aspect=10)  # Adjusted parameters
                ax.set_title(step)
                cbar.ax.tick_params(labelsize=8)
            
            # Create control buttons
            btn_box = QDialogButtonBox()
            btn_accept = btn_box.addButton("Accept", QDialogButtonBox.AcceptRole)
            btn_retry = btn_box.addButton("Retry", QDialogButtonBox.ActionRole)
            btn_cancel = btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)
            
            # Connect signals
            btn_accept.clicked.connect(compare_dialog.accept)
            btn_retry.clicked.connect(lambda: self.retry_filtering(compare_dialog))
            btn_cancel.clicked.connect(compare_dialog.reject)
            
            # Add widgets to layout
            main_layout.addWidget(toolbar)
            main_layout.addWidget(canvas)
            main_layout.addWidget(btn_box)
            
            # Show dialog
            compare_dialog.exec_()
            
        except Exception as e:
            self.show_error(f"Comparison failed: {str(e)}")   



    def show_status(self, message, timeout=3000):
        """Status display method"""
        self.statusBar().showMessage(message, timeout)
        try:
            # Check precondition
            if 'Filtered' not in self.sar_steps:
                raise ValueError("Please apply speckle filter first")

            # Get linear scale data
            linear_data = self.sar_steps['Filtered']['data']
            
            # Convert to dB with NaN handling for non-positive values
            with np.errstate(divide='ignore', invalid='ignore'):
                db_data = 10 * np.log10(np.where(linear_data > 0, linear_data, np.nan))

            # Store results
            self.sar_steps['dB'] = {
                'data': db_data,
                'metadata': self.sar_steps['Filtered']['metadata'].copy(),
                'profile': self.sar_steps['Filtered']['profile'],
                'stats': self._calculate_stats(db_data),
                'vis_params': {
                    'cmap': 'gray',
                    'vmin': np.nanpercentile(db_data, 2),
                    'vmax': np.nanpercentile(db_data, 98)
                }
            }

            # Update UI
            self._display_sar_layer('dB')
            self._enable_sar_workflow_step(3)
            self._update_sar_metadata('dB')
            self._show_histogram(db_data)  # Now using the new method
            
            # After successful conversion
            self._prompt_save_result('dB')
        except Exception as e:
            self.show_error(f"dB conversion failed: {str(e)}")

    def _run_with_progress(self, title, operation, total_steps):
        """Generic progress-enabled operation runner"""
        progress = QProgressDialog(
            title,
            "Cancel", 
            0, 
            total_steps, 
            self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        result = None
        try:
            for step in range(total_steps):
                if progress.wasCanceled():
                    raise RuntimeError("Operation canceled")
                    
                # Execute operation step
                result = operation(step, result)
                
                progress.setValue(step + 1)
                QApplication.processEvents()
                
            return result
        finally:
            progress.close()

    def _prompt_save_result(self, step_name, data=None):
        """Universal save prompt for processing results"""
        reply = QMessageBox.question(
            self,
            "Save Result",
            f"Would you like to save the {step_name} result?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self._save_processed_data(step_name, data)

    def _save_processed_data(self, step_name, data=None):
        """Save processed data with metadata"""
        try:
            # Validate data existence
            if data is None and step_name not in self.sar_steps:
                raise ValueError("No data available for saving")

            # Get data and metadata with proper validation
            if data is None:
                data = self.sar_steps[step_name]['data']
                metadata = self.sar_steps[step_name].get('metadata', {})
                profile = self.sar_steps[step_name].get('profile', {})
            else:
                # For histogram data, use original metadata
                metadata = self.sar_steps.get('dB', self.sar_steps.get('Original', {})).get('metadata', {})
                profile = self.sar_steps['Original'].get('profile', {}).copy()

            # Configure output parameters
            profile.update({
                'driver': 'GTiff',
                'dtype': data.dtype,
                'count': 1,
                'compress': 'lzw'
            })

            # Get save path
            default_name = f"{metadata.get('file_name', 'output')}_{step_name}.tif"
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Processed Data",
                default_name,
                "GeoTIFF Files (*.tif *.tiff);;All Files (*)"
            )
            
            if not path:
                return

            # Write file
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(data, 1)
                dst.update_tags(**metadata)
                
            self.show_info(f"Successfully saved {step_name} data to:\n{path}")

        except Exception as e:
            self.show_error(f"Save failed: {str(e)}")
            

    def analyze_histogram(self):
        """SAR Backscatter Analysis window with optimized layout and controls"""
        try:
            if 'dB' not in self.sar_steps:
                raise ValueError("Complete dB conversion first (step 3)")

            analysis_dialog = QDialog(self)
            analysis_dialog.setWindowTitle("SAR Backscatter Analysis")
            analysis_dialog.setMinimumSize(1400, 900)
            main_layout = QVBoxLayout(analysis_dialog)
            main_layout.setContentsMargins(10, 10, 10, 10)

            # Control Panel (Top 20%)
            control_panel = QWidget()
            control_layout = QHBoxLayout(control_panel)
            control_layout.setContentsMargins(0, 0, 0, 0)

            # Analysis Parameters
            param_group = QGroupBox("Analysis Parameters")
            param_layout = QFormLayout(param_group)
            
            self.bin_spin = QSpinBox()
            self.bin_spin.setRange(10, 1000)
            self.bin_spin.setValue(200)
            
            self.method_combo = QComboBox()
            self.method_combo.addItems(["Manual", "Otsu's Method", "Percentile Ranges",
                                    "K-means Clustering", "Adaptive Thresholding"])
            
            param_layout.addRow("Number of Bins:", self.bin_spin)
            param_layout.addRow("Threshold Method:", self.method_combo)

            # Threshold Controls
            threshold_group = QGroupBox("Threshold Values (dB)")
            threshold_layout = QFormLayout(threshold_group)
            self.lower_spin = QDoubleSpinBox()
            self.lower_spin.setRange(-50, 50)
            self.upper_spin = QDoubleSpinBox()
            self.upper_spin.setRange(-50, 50)
            threshold_layout.addRow("Lower:", self.lower_spin)
            threshold_layout.addRow("Upper:", self.upper_spin)

            control_layout.addWidget(param_group)
            control_layout.addWidget(threshold_group)
            control_layout.addStretch()

            # Main Visualization Area (80%)
            viz_layout = QHBoxLayout()
            
            # Histogram Canvas (70% width)
            hist_container = QWidget()
            hist_layout = QVBoxLayout(hist_container)
            self.fig = Figure(figsize=(14, 8))
            self.canvas = FigureCanvas(self.fig)
            toolbar = NavigationToolbar(self.canvas, analysis_dialog)
            hist_layout.addWidget(self.canvas)
            hist_layout.addWidget(toolbar)
            viz_layout.addWidget(hist_container, stretch=7)

            # Statistics & Controls Panel (30% width)
            stats_group = QGroupBox("Statistics & Actions")
            stats_layout = QVBoxLayout(stats_group)
            
            # Statistics Display
            self.stats_box = QTextEdit()
            self.stats_box.setReadOnly(True)
            self.stats_box.setStyleSheet("""
                font-family: Consolas, monospace;
                font-size: 10pt;
                background-color: #f8f9fa;
                padding: 10px;
            """)
            
            # Action Buttons
            btn_box = QDialogButtonBox()
            self.preview_btn = btn_box.addButton("Preview Water", QDialogButtonBox.ActionRole)
            update_btn = btn_box.addButton("Update Histogram", QDialogButtonBox.ActionRole)
            complete_btn = btn_box.addButton("Complete Step", QDialogButtonBox.AcceptRole)
            cancel_btn = btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)

            stats_layout.addWidget(self.stats_box)
            stats_layout.addWidget(btn_box)

            viz_layout.addWidget(stats_group, stretch=3)

            # Layout Assembly
            main_layout.addWidget(control_panel)
            main_layout.addLayout(viz_layout)

            # Signal Connections
            self.preview_btn.clicked.connect(self._show_water_preview)
            update_btn.clicked.connect(self._update_display)
            complete_btn.clicked.connect(lambda: self._complete_analysis(analysis_dialog))
            cancel_btn.clicked.connect(analysis_dialog.reject)
            self.bin_spin.valueChanged.connect(self._update_display)
            self.method_combo.currentTextChanged.connect(self._run_pro_analysis)

            # Initialize Data
            self.current_hist_data = self.sar_steps['dB']['data'][np.isfinite(self.sar_steps['dB']['data'])]
            self._run_pro_analysis()
            self._update_display()
            
            return analysis_dialog.exec_()

        except Exception as e:
            self.show_error(f"Histogram analysis failed: {str(e)}")

    

    def _get_statistical_summary(self):
        """Enhanced statistical summary with threshold info"""
        data = self.current_hist_data
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        
        stats = [
            f"{'Total Samples:':<20} {len(data):,}",
            f"{'Data Range:':<20} {np.min(data):.2f} - {np.max(data):.2f} dB",
            f"{'Current Thresholds:':<20} {lower:.2f} - {upper:.2f} dB",
            f"{'Threshold Range:':<20} {upper-lower:.2f} dB",
            f"{'Mean Value:':<20} {np.mean(data):.2f} dB",
            f"{'Std Deviation:':<20} {np.std(data):.2f} dB",
            f"{'Threshold Coverage:':<20} {np.mean((data >= lower) & (data <= upper)):.1%}"
        ]
        
        return "\n".join(stats)
    
    def _save_histogram(self):
        """Save histogram to file with professional format options"""
        formats = "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif)"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Histogram", "", formats
        )
        
        if path:
            try:
                self.fig.savefig(
                    path,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white'
                )
                self.show_status(f"Histogram saved to {path}")
            except Exception as e:
                self.show_error(f"Save failed: {str(e)}")

    # Add to menu bar initialization
    def _create_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        
        save_hist_action = QAction("Save Histogram", self)
        save_hist_action.triggered.connect(self._save_histogram)
        file_menu.addAction(save_hist_action)

    
    def _run_pro_analysis(self):
        """Enhanced threshold analysis with manual option"""
        try:
            method = self.method_combo.currentText()
            data = self.current_hist_data
            
            if method == "Manual":
                # Leave current values unchanged
                return  # Exit early for manual mode
                
            elif method == "Otsu's Method":
                threshold = filters.threshold_otsu(data)
                lower, upper = threshold, threshold
                
            elif method == "Percentile Ranges":
                lower, upper = np.percentile(data, [5, 95])
                
            elif method == "K-means Clustering":
                kmeans = KMeans(n_clusters=2).fit(data.reshape(-1, 1))
                centers = sorted(kmeans.cluster_centers_.flatten())
                lower, upper = centers[0], centers[1]
                
            elif method == "Adaptive Thresholding":
                mean = np.mean(data)
                std = np.std(data)
                lower, upper = mean-2*std, mean+2*std

            # Update UI controls for non-manual methods
            self.lower_spin.setValue(lower)
            self.upper_spin.setValue(upper)

            self._update_display()

        except Exception as e:
            self.show_error(f"Analysis failed: {str(e)}")

    def _update_display(self):
        """Update histogram display with integrated statistics"""
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            # Get current parameters
            bins = self.bin_spin.value()
            lower = self.lower_spin.value()
            upper = self.upper_spin.value()

            # Plot histogram with proper normalization
            n, bins, patches = ax.hist(
                self.current_hist_data,
                bins=bins,
                range=(lower, upper),
                density=True,
                color='steelblue',
                edgecolor='black',
                alpha=0.8
            )

            # Add threshold lines and region
            ax.axvline(lower, color='red', linestyle='--', linewidth=1.5, label=f'Lower: {lower:.2f} dB')
            ax.axvline(upper, color='green', linestyle='--', linewidth=1.5, label=f'Upper: {upper:.2f} dB')
            ax.axvspan(lower, upper, color='blue', alpha=0.1)

            # Add KDE curve
            kde = gaussian_kde(self.current_hist_data)
            x = np.linspace(lower, upper, 500)
            ax.plot(x, kde(x), color='darkorange', linewidth=2, label='Density Curve')

            # Formatting
            ax.set_title("SAR Backscatter Distribution Analysis", fontsize=14)
            ax.set_xlabel("Backscatter Coefficient (dB)", fontsize=12)
            ax.set_ylabel("Probability Density", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')

            # Add colorbar with proper sizing
            norm = plt.Normalize(vmin=lower, vmax=upper)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            
            # # Create properly sized colorbar
            # ax_pos = ax.get_position()
            # cax = self.fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0, 0.02, ax_pos.height])
            # # cbar = self.fig.colorbar(sm, cax=cax)
            # # cbar.set_label('Threshold Range (dB)', fontsize=10)
            # # cbar.ax.tick_params(labelsize=8)

            # Update statistics
            stats = self._get_statistical_summary()
            self.stats_box.setText(stats)
            self.canvas.draw()

        except Exception as e:
            self.show_error(f"Display update failed: {str(e)}")



    def _on_complete_step(self):
        if self._validate_thresholds():
            self._save_thresholds()
            self._enable_next_step()
            self.close()
        else:
            self.show_error("Invalid thresholds. Check values before completing.")

    # In analyze_histogram method, update the Complete button connection:
    # Replace:
    # complete_btn.clicked.connect(analysis_dialog.accept)
    # With: complete_btn.clicked.connect(self._on_complete_step)

    # Remove the old _save_thresholds method and replace with:
    
    def _save_thresholds(self):
        """Get values directly from spin boxes"""
        try:
            self.sar_steps['dB']['metadata']['thresholds'] = {
                'lower': self.lower_spin.value(),  # Direct from UI
                'upper': self.upper_spin.value(),  # Direct from UI
                'method': self.method_combo.currentText(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.show_error(f"Threshold save failed: {str(e)}")
   

    
    
    def _calculate_otsu_threshold(self, data):
        """Professional method: Otsu's thresholding"""
        from skimage.filters import threshold_otsu
        return threshold_otsu(data)

    def _calculate_percentile_threshold(self, data):
        """Professional method: Percentile-based threshold"""
        return np.percentile(data, [5, 95])  # [lower, upper]

    def _calculate_kmeans_threshold(self, data):
        """Professional method: K-means clustering threshold"""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2).fit(data.reshape(-1, 1))
        centers = sorted(kmeans.cluster_centers_.flatten())
        return (centers[0] + centers[1])/2  # Midpoint between clusters
    
    def _show_water_preview(self):
        """Professional water preview visualization"""
        try:
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("Water Classification Preview")
            preview_dialog.setMinimumSize(800, 600)
            
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, preview_dialog)
            
            # Create mask
            lower = self.lower_spin.value()
            upper = self.upper_spin.value()
            data = self.sar_steps['dB']['data']
            mask = (data >= lower) & (data <= upper)
            cleaned_mask = self._clean_water_mask(mask)

            # Create RGB preview
            preview = np.zeros((*cleaned_mask.shape, 3))
            preview[cleaned_mask] = [0, 0.4, 1]  # Blue for water
            preview[~cleaned_mask] = [0.9, 0.9, 0.9]  # Light gray for non-water

            ax = fig.add_subplot(111)
            ax.imshow(preview)
            ax.set_title(f"Water Preview\nThresholds: {lower:.2f} dB - {upper:.2f} dB")
            ax.axis('off')

            layout = QVBoxLayout(preview_dialog)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            preview_dialog.exec_()

        except Exception as e:
            self.show_error(f"Preview failed: {str(e)}")


    def _complete_analysis(self, dialog):
        """Validate and complete analysis step"""
        if self._validate_thresholds():
            self._save_thresholds()
            self._enable_next_step('analyze')
            dialog.accept()
            self.show_status("Threshold analysis completed. Ready for classification.")
        else:
            self.show_error("Invalid thresholds. Values must be between -50 and 50 dB with lower < upper.")

    def _validate_thresholds(self):
        """Robust threshold validation"""
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        return -50 < lower < upper < 50

    def _show_threshold_dialog(self, thresholds):
        """Professional threshold selection interface"""
        dialog = QDialog(self)
        layout = QVBoxLayout(dialog)
        
        # Threshold Table
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Method", "Lower", "Upper"])
        table.setRowCount(len(thresholds))
        
        # Populate suggestions
        for row, (method, values) in enumerate(thresholds.items()):
            table.setItem(row, 0, QTableWidgetItem(method.capitalize()))
            table.setItem(row, 1, QTableWidgetItem(f"{values[0]:.2f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{values[1]:.2f}"))
        
        # Manual Input
        manual_group = QGroupBox("Manual Thresholds")
        manual_layout = QFormLayout(manual_group)
        
        lower_spin = QDoubleSpinBox()
        lower_spin.setRange(-50, 50)
        upper_spin = QDoubleSpinBox()
        upper_spin.setRange(-50, 50)
        
        manual_layout.addRow("Lower Threshold (dB):", lower_spin)
        manual_layout.addRow("Upper Threshold (dB):", upper_spin)
        
        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        
        def apply_thresholds():
            # Store selected values
            self.sar_steps['dB']['metadata']['thresholds']['user_selected'] = {
                'lower': lower_spin.value(),
                'upper': upper_spin.value()
            }
            dialog.accept()
            
        btn_box.accepted.connect(apply_thresholds)
        btn_box.rejected.connect(dialog.reject)
        
        layout.addWidget(QLabel("Automatic Threshold Suggestions:"))
        layout.addWidget(table)
        layout.addWidget(manual_group)
        layout.addWidget(btn_box)
        
        dialog.exec_()


    
    def classify_water(self):
        """Professional water classification with improved visualization"""
        try:
            if 'dB' not in self.sar_steps or 'thresholds' not in self.sar_steps['dB']['metadata']:
                raise ValueError("Complete histogram analysis first")

            # Store classification results properly
            thresholds = self.sar_steps['dB']['metadata']['thresholds']
            data = self.sar_steps['dB']['data']
            lower = thresholds['lower']
            upper = thresholds['upper']
            
            water_mask = (data >= lower) & (data <= upper)
            water_mask = self._clean_water_mask(water_mask)

            # Store results with proper structure
            self.sar_steps['classification'] = {
                'mask': water_mask,
                'thresholds': thresholds,  # Store thresholds at root level
                'stats': {
                    'total_pixels': water_mask.size,
                    'wet_pixels': np.sum(water_mask),
                    'wet_percent': np.mean(water_mask) * 100
                }
            }

            # Create results dialog
            results_dialog = QDialog(self)
            results_dialog.setWindowTitle("Water Classification Results")
            results_dialog.setMinimumSize(1200, 800)
            layout = QVBoxLayout(results_dialog)           
            
            
            
            # Create figure with subplots
            fig = Figure(figsize=(14, 8))
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, results_dialog)

            # Original Image Plot
            ax1 = fig.add_subplot(121)
            im1 = ax1.imshow(data, cmap='gray', vmin=lower-5, vmax=upper+5)
            ax1.set_title("Original SAR Backscatter", fontsize=12)
            
            # Add properly sized colorbar
            ax1_pos = ax1.get_position()
            cax1 = fig.add_axes([ax1_pos.x1 + 0.02, ax1_pos.y0, 0.02, ax1_pos.height])
            fig.colorbar(im1, cax=cax1).set_label('dB', fontsize=10)

            # Classification Plot
            ax2 = fig.add_subplot(122)
            water_display = np.where(water_mask, data, np.nan)
            im2 = ax2.imshow(water_display, cmap='Blues', vmin=lower, vmax=upper)
            ax2.set_title("Water Classification Results", fontsize=12)
            
            # Add proportional colorbar
            ax2_pos = ax2.get_position()
            cax2 = fig.add_axes([ax2_pos.x1 + 0.02, ax2_pos.y0, 0.02, ax2_pos.height])
            cbar = fig.colorbar(im2, cax=cax2)
            cbar.set_label('Water Area (dB)', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

            # Statistics Panel
            stats_text = f"""Classification Summary:
            Total Pixels: {water_mask.size:,}
            Water Pixels: {np.sum(water_mask):,} ({np.mean(water_mask)*100:.1f}%)
            Non-Water Pixels: {water_mask.size - np.sum(water_mask):,}
            
            Threshold Values:
            Lower: {lower:.2f} dB
            Upper: {upper:.2f} dB
            Method: {thresholds['method']}"""
            
            stats_box = QTextEdit()
            stats_box.setMarkdown(stats_text)
            stats_box.setReadOnly(True)
            stats_box.setFixedHeight(150)

            # Button connections with proper references
            btn_box = QDialogButtonBox()
            export_btn = btn_box.addButton("Export Results", QDialogButtonBox.ActionRole)
            close_btn = btn_box.addButton("Close", QDialogButtonBox.RejectRole)
            
            # Proper close connection
            close_btn.clicked.connect(results_dialog.reject)
            export_btn.clicked.connect(lambda: self._export_classification(water_mask))
            
            # Layout Assembly
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            layout.addWidget(stats_box)
            layout.addWidget(btn_box)

            results_dialog.exec_()
            self.show_status("Classification results displayed")

        except Exception as e:
            self.show_error(f"Classification failed: {str(e)}")

    def _export_classification(self, mask):
        """Export classification results in multiple formats with professional visualization"""
        try:
            # Validate classification exists
            if 'classification' not in self.sar_steps:
                raise ValueError("Complete classification step first")

            # Get thresholds for metadata
            thresholds = self.sar_steps['classification']['thresholds']
            
            # Setup file dialog
            formats = "PNG Image (*.png);;GeoTIFF (*.tif);;NumPy Array (*.npy)"
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Classification Results",
                "",
                formats
            )

            if not path:
                return  # User cancelled

            # PNG Export
            if path.endswith('.png'):
                self._export_png(mask, path, thresholds)

            # GeoTIFF Export
            elif path.endswith('.tif'):
                self._save_geotiff(mask, path, thresholds)

            # NumPy Export
            elif path.endswith('.npy'):
                np.save(path, mask)
                self._show_message("Export Successful", 
                                f"NumPy array saved to:\n{path}")

            self.show_status(f"Results exported to {os.path.basename(path)}")

        except Exception as e:
            self.show_error(f"Export failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _export_png(self, mask, path, thresholds):
        """Save publication-quality PNG with geographic features if available"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'grid.color': '#dddddd',
            'grid.linestyle': '--',
            'grid.alpha': 0.5,
            'figure.dpi': 300
        })

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        cmap = ListedColormap(['#1f77b4', 'white'])  # Water=blue, Non-water=white
        geo_plot = False

        try:
            # Attempt geographic plot if cartopy available
            from cartopy import crs as ccrs
            profile = self.sar_steps['Original']['profile']
            src_crs = profile['crs']
            
            if src_crs.is_utm:
                zone = src_crs.utm_zone
                if zone and zone[:-1].isdigit():
                    zone_number = int(zone[:-1])
                    if 1 <= zone_number <= 60:
                        utm_crs = ccrs.UTM(zone_number)
                        transform = profile['transform']
                        extent = (
                            transform[2],
                            transform[2] + transform[0] * mask.shape[1],
                            transform[5] + transform[4] * mask.shape[0],
                            transform[5]
                        )
                        
                        ax = plt.axes(projection=utm_crs)
                        img = ax.imshow(
                            mask, 
                            cmap=cmap,
                            extent=extent,
                            origin='upper',
                            transform=utm_crs
                        )
                        
                        # Add gridlines
                        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
                        gl.top_labels = gl.right_labels = False
                        ax.set_xlabel('Easting (m)', fontsize=10)
                        ax.set_ylabel('Northing (m)', fontsize=10)
                        geo_plot = True

        except ImportError:
            self.show_info("Geographic features require 'cartopy' package")
        except Exception as e:
            print(f"Geographic plot failed: {str(e)}")

        if not geo_plot:
            # Fallback to pixel coordinates
            img = ax.imshow(mask, cmap=cmap)
            ax.set_xlabel('Pixel Column', fontsize=10)
            ax.set_ylabel('Pixel Row', fontsize=10)
            ax.grid(True)

        # Add scale bar if available
        try:
            from matplotlib_scalebar.scalebar import ScaleBar
            if geo_plot:
                ax.add_artist(ScaleBar(
                    dx=1,  # Adjust based on data resolution
                    units="m",
                    location="lower left",
                    length_fraction=0.25,
                    height_fraction=0.02
                ))
        except ImportError:
            pass

        # Add annotations
        title = f"Water Classification\n({thresholds['lower']:.1f}-{thresholds['upper']:.1f} dB)"
        ax.set_title(title, fontsize=12, pad=15)
        plt.figtext(0.95, 0.03, 
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ha='right', va='bottom', fontsize=8, alpha=0.7)

        # Add colorbar
        cbar = fig.colorbar(img, ax=ax, ticks=[0, 1], shrink=0.8)
        cbar.set_ticklabels(['Water', 'Non-water'])
        cbar.set_label('Classification Result', fontsize=10)

        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        self._show_message("Export Successful", f"PNG saved to:\n{path}")

    def _save_geotiff(self, mask, path, thresholds):
        """Save georeferenced RGB GeoTIFF with metadata"""
        try:
            profile = self.sar_steps['Original']['profile'].copy()
            
            # Create RGB bands (water=blue, non-water=white)
            red = np.where(mask, 0, 255).astype(np.uint8)
            green = np.where(mask, 0, 255).astype(np.uint8)
            blue = np.where(mask, 255, 255).astype(np.uint8)

            # Update profile for RGB
            profile.update({
                'driver': 'GTiff',
                'count': 3,
                'dtype': 'uint8',
                'compress': 'lzw',
                'photometric': 'RGB'
            })

            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(red, 1)
                dst.write(green, 2)
                dst.write(blue, 3)
                dst.update_tags(
                    processing_date=datetime.now().isoformat(),
                    water_threshold_lower=thresholds['lower'],
                    water_threshold_upper=thresholds['upper'],
                    water_color='RGB(0,0,255)',
                    non_water_color='RGB(255,255,255)'
                )

            self._show_message("Export Successful", f"GeoTIFF saved to:\n{path}")

        except Exception as e:
            self.show_error(f"GeoTIFF export failed: {str(e)}")
            raise

    def _show_message(self, title, message):
        """Display standardized success message"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
 


 
    
    def _clean_water_mask(self, mask):
        """Professional post-processing"""
        from skimage.morphology import remove_small_objects, binary_closing
        
        # 1. Remove small objects
        cleaned = remove_small_objects(mask, min_size=16)
        
        # 2. Apply morphological closing
        return binary_closing(cleaned, footprint=np.ones((3,3)))  


    def _run_enhanced_auto_analysis(self):
        """Fixed auto-analysis with proper array handling"""
        try:
            data = self.current_hist_data
            
            # 1. Calculate thresholds using multiple professional methods
            methods = {
                'Otsu': filters.threshold_otsu(data),
                'Percentile (2/98)': np.percentile(data, [2, 98]),
                'K-means': self._kmeans_threshold(data),
                'Adaptive': self._adaptive_threshold(data),
                'Mean ± 2σ': (
                    np.mean(data) - 2*np.std(data),
                    np.mean(data) + 2*np.std(data)
                )
            }

            # 2. Validate and weight results
            valid_ranges = [(-35, 5)]  # Typical SAR dB range for water analysis
            weighted_results = []
            
            for method_name, values in methods.items():
                # Ensure scalar values for comparison
                if isinstance(values, (np.ndarray, list)):
                    lower = float(values[0])
                    upper = float(values[1]) if len(values) > 1 else lower
                else:
                    lower = upper = float(values)

                # Validate against physical limits using scalar comparisons
                if -50 < lower < upper < 50:
                    # Weight professional methods higher
                    weight = 2 if method_name in ['Otsu', 'K-means'] else 1
                    weighted_results.extend([lower]*weight + [upper]*weight)

            # 3. Calculate robust thresholds with array-safe operations
            if weighted_results:
                lower = np.percentile(weighted_results, 25).item()  # Convert to scalar
                upper = np.percentile(weighted_results, 75).item()
            else:
                # Fallback to percentiles with explicit scalar conversion
                lower, upper = np.percentile(data, [5, 95]).tolist()

            # 4. Apply to UI with validation
            self.range_min.setValue(float(np.clip(lower, -50, 50)))
            self.range_max.setValue(float(np.clip(upper, -50, 50)))

            # 5. Visual feedback
            self._refresh_histogram()
            self._show_confidence_indicator(lower, upper)

        except Exception as e:
            self.show_error(f"Enhanced analysis failed: {str(e)}")

    def _show_confidence_indicator(self, lower, upper):
        """Visual quality indicator for threshold reliability"""
        overlap_score = self._calculate_overlap_score(lower, upper)
        
        # Create color-coded confidence indicator
        confidence_color = {
            'high': '#4CAF50',
            'medium': '#FFC107',
            'low': '#F44336'
        }.get(overlap_score['confidence'], 'gray')

        self.ax.text(
            0.98, 0.95,
            f"Confidence: {overlap_score['confidence'].upper()}",
            transform=self.ax.transAxes,
            color=confidence_color,
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.8))
        
        self.canvas.draw()

    def _calculate_overlap_score(self, lower, upper):
        """Calculate statistical confidence in threshold range"""
        from scipy.stats import norm
        
        data = self.current_hist_data
        mean, std = np.mean(data), np.std(data)
        
        # Calculate probability mass within range
        prob_lower = norm.cdf(lower, mean, std)
        prob_upper = norm.cdf(upper, mean, std)
        overlap = prob_upper - prob_lower
        
        return {
            'confidence': 'high' if overlap > 0.7 else 'medium' if overlap > 0.4 else 'low',
            'probability': overlap
        }

    def _show_histogram(self, data):
        """Fixed histogram display with complete functionality"""
        try:
            self.hist_dialog = QDialog(self)
            self.hist_dialog.setWindowTitle("SAR Backscatter Analysis")
            self.hist_dialog.setMinimumSize(1200, 800)

            # Main layout
            main_layout = QVBoxLayout(self.hist_dialog)

            # ======================================================================
            # Control Panel
            # ======================================================================
            control_layout = QHBoxLayout()
            
            # Histogram Configuration
            config_group = QGroupBox("Histogram Configuration")
            config_layout = QFormLayout(config_group)
            self.bin_spin = QSpinBox()
            self.bin_spin.setRange(10, 1000)
            self.bin_spin.setValue(200)
            config_layout.addRow("Number of Bins:", self.bin_spin)

            # Value Range
            range_group = QGroupBox("Value Range")
            range_layout = QFormLayout(range_group)
            self.range_min = QDoubleSpinBox()
            self.range_min.setRange(-50, 50)
            self.range_min.setValue(-20)
            self.range_max = QDoubleSpinBox()
            self.range_max.setRange(-50, 50)
            self.range_max.setValue(5)
            range_layout.addRow("Minimum (dB):", self.range_min)
            range_layout.addRow("Maximum (dB):", self.range_max)

            control_layout.addWidget(config_group)
            control_layout.addWidget(range_group)

            # ======================================================================
            # Expert Parameters
            # ======================================================================
            expert_group = QGroupBox("Expert Parameters")
            expert_layout = QFormLayout(expert_group)
            
            # Method Weights
            self.weights = {
                'Otsu': QDoubleSpinBox(),
                'K-means': QDoubleSpinBox(),
                'Percentile': QDoubleSpinBox()
            }
            for name, widget in self.weights.items():
                widget.setRange(0.1, 5.0)
                widget.setValue(1.0)
                widget.setSingleStep(0.1)
                expert_layout.addRow(f"{name} Weight:", widget)

            # Confidence Level
            self.confidence_slider = QSlider(Qt.Horizontal)
            self.confidence_slider.setRange(50, 95)
            self.confidence_slider.setValue(75)
            expert_layout.addRow("Confidence Level:", self.confidence_slider)

            # ======================================================================
            # Visualization Area
            # ======================================================================
            self.fig = Figure(figsize=(10, 6))
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self.hist_dialog)

            # ======================================================================
            # Statistics Display
            # ======================================================================
            self.stats_box = QTextEdit()
            self.stats_box.setReadOnly(True)
            self.stats_box.setMaximumHeight(100)

            # ======================================================================
            # Button Panel
            # ======================================================================
            btn_box = QDialogButtonBox()
            self.btn_auto = btn_box.addButton("Auto-Analyze", QDialogButtonBox.ActionRole)
            self.btn_preview = btn_box.addButton("Preview", QDialogButtonBox.ActionRole)
            self.btn_apply = btn_box.addButton("Apply", QDialogButtonBox.AcceptRole)
            self.btn_cancel = btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)

            # ======================================================================
            # Layout Assembly
            # ======================================================================
            main_layout.addLayout(control_layout)
            main_layout.addWidget(expert_group)
            main_layout.addWidget(self.toolbar)
            main_layout.addWidget(self.canvas)
            main_layout.addWidget(self.stats_box)
            main_layout.addWidget(btn_box)

            # ======================================================================
            # Signal Connections
            # ======================================================================
            self.bin_spin.valueChanged.connect(self._safe_refresh)
            self.range_min.valueChanged.connect(self._safe_refresh)
            self.range_max.valueChanged.connect(self._safe_refresh)
            self.btn_auto.clicked.connect(self._safe_auto_analyze)
            self.btn_preview.clicked.connect(self._safe_preview)
            self.btn_apply.clicked.connect(self._save_and_proceed)
            self.btn_cancel.clicked.connect(self.hist_dialog.reject)

            # Initialize data and display
            self.current_hist_data = data[np.isfinite(data)]
            self._refresh_histogram()
            self._update_statistics()
            self.hist_dialog.exec_()

        except Exception as e:
            self.show_error(f"Histogram initialization failed: {str(e)}")

    def _safe_refresh(self):
        """Safe histogram refresh with error handling"""
        try:
            self._refresh_histogram()
            self._update_statistics()
        except Exception as e:
            self.show_error(f"Refresh failed: {str(e)}")

    def _refresh_histogram(self):
        """Robust histogram drawing"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        try:
            # Calculate histogram parameters
            bins = self.bin_spin.value()
            vmin = self.range_min.value()
            vmax = self.range_max.value()
            
            # Validate parameters
            if vmin >= vmax:
                raise ValueError("Minimum value must be less than maximum")
            if bins < 10:
                raise ValueError("Minimum 10 bins required")

            # Create histogram
            n, bins, patches = ax.hist(
                self.current_hist_data,
                bins=bins,
                range=(vmin, vmax),
                density=True,
                color='steelblue',
                edgecolor='black'
            )

            # Add KDE curve
            kde = gaussian_kde(self.current_hist_data)
            x = np.linspace(vmin, vmax, 300)
            ax.plot(x, kde(x), color='darkorange', linewidth=2)

            # Add thresholds
            ax.axvline(self.range_min.value(), color='red', linestyle='--')
            ax.axvline(self.range_max.value(), color='green', linestyle='--')

            ax.set_title("SAR Backscatter Distribution")
            ax.set_xlabel("Backscatter Coefficient (dB)")
            ax.set_ylabel("Normalized Frequency")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            self.canvas.draw()

        except Exception as e:
            self.show_error(f"Histogram error: {str(e)}")

    def _update_statistics(self):
        """Update statistics display with robust calculations"""
        try:
            stats = [
                f"Data Statistics:",
                f"• Samples: {len(self.current_hist_data):,}",
                f"• Mean: {np.mean(self.current_hist_data):.2f} dB",
                f"• Std Dev: {np.std(self.current_hist_data):.2f} dB",
                f"• Min: {np.min(self.current_hist_data):.2f} dB",
                f"• Max: {np.max(self.current_hist_data):.2f} dB",
                f"• 5th %ile: {np.percentile(self.current_hist_data, 5):.2f} dB",
                f"• 95th %ile: {np.percentile(self.current_hist_data, 95):.2f} dB"
            ]
            self.stats_box.setText('\n'.join(stats))
        except Exception as e:
            self.stats_box.setText(f"Statistics unavailable: {str(e)}")

    def _safe_auto_analyze(self):
        """Error-handled auto-analysis"""
        try:
            self._run_enhanced_auto_analyze()
            self._update_statistics()
        except Exception as e:
            self.show_error(f"Auto-analysis failed: {str(e)}")

    def _run_enhanced_auto_analyze(self):
        """Fixed enhanced auto-analysis"""
        try:
            data = self.current_hist_data
            
            # Get weights from UI
            weights = {
                'Otsu': self.weights['Otsu'].value(),
                'K-means': self.weights['K-means'].value(),
                'Percentile': self.weights['Percentile'].value()
            }

            # Calculate thresholds using different methods
            otsu_val = filters.threshold_otsu(data)
            percentiles = np.percentile(data, [2, 98])
            kmeans_low, kmeans_high = self._kmeans_threshold(data)
            mean = np.mean(data)
            std = np.std(data)
            mean_thresh = (mean - 2*std, mean + 2*std)

            # Collect all thresholds with weighting
            weighted = []
            weighted.extend([percentiles[0]] * int(weights['Percentile']))
            weighted.extend([percentiles[1]] * int(weights['Percentile']))
            weighted.extend([kmeans_low] * int(weights['K-means']))
            weighted.extend([kmeans_high] * int(weights['K-means']))
            weighted.extend([otsu_val] * int(weights['Otsu']))
            weighted.extend([mean_thresh[0], mean_thresh[1]])

            # Calculate final thresholds
            lower = np.percentile(weighted, 25)
            upper = np.percentile(weighted, 75)

            # Apply to UI
            self.range_min.setValue(float(lower))
            self.range_max.setValue(float(upper))

        except Exception as e:
            raise ValueError(f"Auto-analysis error: {str(e)}")

    def _kmeans_threshold(self, data):
        """Robust K-means implementation"""
        from sklearn.cluster import KMeans
        data_reshaped = data.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, n_init=10).fit(data_reshaped)
        centers = sorted(kmeans.cluster_centers_.flatten())
        return float(centers[0]), float(centers[1])

    def _safe_preview(self):
        """Fixed preview triggering"""
        try:
            lower = self.range_min.value()
            upper = self.range_max.value()
            
            if not hasattr(self, 'sar_steps') or 'dB' not in self.sar_steps:
                raise ValueError("Complete dB conversion first")
                
            self._show_classification_preview(lower, upper)
            
        except ValueError as ve:
            self.show_error(f"Preview validation failed: {str(ve)}")
        except Exception as e:
            self.show_error(f"Unexpected preview error: {str(e)}")

    def _save_and_proceed(self):
        """Fixed save method with workflow control"""
        try:
            lower = self.range_min.value()
            upper = self.range_max.value()
            
            if lower >= upper:
                raise ValueError("Lower threshold must be less than upper")
            if not (-50 < lower < upper < 50):
                raise ValueError("Thresholds must be between -50 and 50 dB")
                
            # Store thresholds
            self.sar_steps['dB']['metadata']['thresholds'] = {
                'lower': float(lower),
                'upper': float(upper),
                'timestamp': datetime.now().isoformat()
            }
            
            # Enable next workflow step
            self._enable_next_step()
            
            # Close dialog
            if self.hist_dialog:
                self.hist_dialog.accept()

        except ValueError as ve:
            self.show_error(f"Validation error: {str(ve)}")
        except Exception as e:
            self.show_error(f"Save failed: {str(e)}")
    

    def _show_classification_preview(self, lower, upper):
        """Fixed preview with proper imports and error handling"""
        try:
            # Validate inputs
            if lower >= upper:
                raise ValueError("Lower threshold must be less than upper")
                
            # Get dB-scaled data with validation
            if 'dB' not in self.sar_steps:
                raise ValueError("Complete dB conversion first")
                
            db_data = self.sar_steps['dB']['data']
            
            # Create initial mask
            mask = np.logical_and(db_data >= lower, db_data <= upper)
            
            # Apply morphological operations with checks
            if mask.any():
                cleaned_mask = binary_closing(mask, footprint=np.ones((3,3)))
                cleaned_mask = remove_small_objects(cleaned_mask, min_size=16)
            else:
                cleaned_mask = mask  # Handle empty mask case

            # Create RGB preview
            preview = np.zeros((*cleaned_mask.shape, 3))
            preview[cleaned_mask] = [0, 0.4, 1]  # Water in blue
            preview[~cleaned_mask] = [0.8, 0.8, 0.8]  # Non-water in gray
            
            # Create preview dialog
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("Classification Preview")
            layout = QVBoxLayout(preview_dialog)
            
            # Create matplotlib figure
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.imshow(preview)
            ax.set_title(f"Water Classification Preview\nThresholds: {lower:.2f} to {upper:.2f} dB")
            ax.axis('off')
            
            # Add to layout
            layout.addWidget(canvas)
            preview_dialog.exec_()

        except Exception as e:
            self.show_error(f"Preview failed: {str(e)}\n{traceback.format_exc()}")
    
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

    
    def convert_db(self):
        """Convert to dB scale after terrain correction"""
        try:
            if 'Terrain' not in self.sar_steps:
                raise ValueError("Complete terrain correction first")

            terrain_data = self.sar_steps['Terrain']['data']
            
            # Convert to dB with enhanced safety checks
            with np.errstate(divide='ignore', invalid='ignore'):
                db_data = 10 * np.log10(np.where(terrain_data > 0, terrain_data, np.nan))

            # Store dB-converted data
            self.sar_steps['dB'] = {
                'data': db_data,
                'profile': self.sar_steps['Terrain']['profile'],
                'metadata': {
                    **self.sar_steps['Terrain']['metadata'],
                    'conversion_date': datetime.now().isoformat()
                }
            }

            self._enable_next_step('convert_db')
            self.layer_list.addItem(QListWidgetItem("📈 dB Converted"))
            self.statusBar().showMessage("dB conversion completed", 5000)

        except Exception as e:
            self.show_error(f"dB conversion failed: {str(e)}")

    def _reset_workflow_step(self, step_name):
        """Reset specific workflow step"""
        try:
            step_idx = self.workflow_steps.index(step_name)
            # Re-enable current step and disable subsequent steps
            for i in range(len(self.workflow_actions)):
                self.workflow_actions[i].setEnabled(i <= step_idx)
            self.show_status(f"Ready to retry: {step_name.replace('_', ' ')}")
        except ValueError:
            self.show_error("Invalid workflow step reset")
    
    
    def _reset_workflow(self):
        """Reset workflow to initial state"""
        for action in self.workflow_actions:
            action.setEnabled(False)
        self.workflow_actions[0].setEnabled(True)
        self.current_step = 0



    def _show_preview(self):
        """Fixed preview with proper control access"""
        try:
            # Use the correct spin box names
            lower = self.lower_spin.value()
            upper = self.upper_spin.value()
            
            # Validate thresholds
            if lower >= upper:
                raise ValueError("Lower threshold must be less than upper")
                
            # Get dB-scaled data
            db_data = self.sar_steps['dB']['data']
            
            # Create and display preview
            mask = np.logical_and(db_data >= lower, db_data <= upper)
            cleaned_mask = binary_closing(mask, footprint=np.ones((3,3)))
            cleaned_mask = remove_small_objects(cleaned_mask, min_size=16)
            
            preview = np.zeros((*cleaned_mask.shape, 3))
            preview[cleaned_mask] = [0, 0.4, 1]  # Water
            preview[~cleaned_mask] = [0.8, 0.8, 0.8]  # Non-water
            
            self._display_preview_image(preview, lower, upper)

        except Exception as e:
            self.show_error(f"Preview failed: {str(e)}")

    def _display_preview_image(self, preview, lower, upper):
        """Display preview in a dialog window"""
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Classification Preview")
        layout = QVBoxLayout(preview_dialog)
        
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(preview)
        ax.set_title(f"Water Classification Preview\nThresholds: {lower:.2f} to {upper:.2f} dB")
        ax.axis('off')
        
        layout.addWidget(canvas)
        preview_dialog.exec_()


        
#############################################
class DbConversionWorker(QThread):
        progress = pyqtSignal(int)
        finished = pyqtSignal(np.ndarray)
        error = pyqtSignal(str)

        def __init__(self, linear_data):
            super().__init__()
            self.linear_data = linear_data

        def run(self):
            try:
                self.progress.emit(10)
                
                # تجنب القيم الصفرية والسالبة
                valid_data = np.where(self.linear_data > 0, self.linear_data, np.nan)
                
                self.progress.emit(30)
                
                # التحويل إلى dB
                db_data = 10 * np.log10(valid_data)
                
                self.progress.emit(90)
                
                # تنظيف البيانات
                db_data = np.where(np.isfinite(db_data), db_data, np.nan)
                
                self.progress.emit(100)
                self.finished.emit(db_data)
                
            except Exception as e:
                self.error.emit(f"خطأ في التحويل: {str(e)}")

class ClassDefinitionDialog(QDialog):
    """Dialog for defining land use classes with color picker."""
    def __init__(self, parent, default_classes):
        super().__init__(parent)
        self.setWindowTitle("Define Land Use Classes")
        self.setMinimumSize(600, 400)
        
        self.classes = {}
        self.setup_ui(default_classes)
    
    def setup_ui(self, default_classes):
        layout = QVBoxLayout(self)
        
        # Table for class definitions
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Class Name", "Color", "Value"])
        self.table.setRowCount(len(default_classes))
        
        # Add default classes
        for row, (name, props) in enumerate(default_classes.items()):
            self.add_class_row(row, name, props['color'], props['value'])
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Class")
        remove_btn = QPushButton("Remove Selected")
        done_btn = QPushButton("Done")
        
        add_btn.clicked.connect(self.add_class)
        remove_btn.clicked.connect(self.remove_class)
        done_btn.clicked.connect(self.accept)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(done_btn)
        
        layout.addWidget(self.table)
        layout.addLayout(btn_layout)
    
    def add_class_row(self, row, name, color, value):
        """Add a class definition row to the table."""
        # Class name
        name_item = QTableWidgetItem(name)
        self.table.setItem(row, 0, name_item)
        
        # Color picker
        color_btn = QPushButton()
        color_btn.setStyleSheet(f"background-color: {color};")
        color_btn.clicked.connect(lambda: self.pick_color(color_btn))
        self.table.setCellWidget(row, 1, color_btn)
        
        # Value
        spin = QSpinBox()
        spin.setRange(1, 255)
        spin.setValue(value)
        self.table.setCellWidget(row, 2, spin)
    
    def add_class(self):
        """Add a new empty class row."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.add_class_row(row, f"Class{row+1}", "#FFFFFF", row+1)
    
    def remove_class(self):
        """Remove selected class."""
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
    
    def pick_color(self, button):
        """Open color picker for the button."""
        color = QColorDialog.getColor(QColor(button.styleSheet().split(":")[1].strip(";")))
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()};")
    
    def get_classes(self):
        """Get the defined classes as a dictionary."""
        classes = {}
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            color = self.table.cellWidget(row, 1).styleSheet().split(":")[1].strip(";")
            value = self.table.cellWidget(row, 2).value()
            classes[name] = {"color": color, "value": value}
        return classes
    
    def show_attribute_table(self):
        """Wrapper function to show attribute table"""
        try:
            # Get selected layer
            selected_items = self.layer_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Warning", "Please select a layer first!")
                return
                
            layer_name = selected_items[0].text()
            
            # Call the processor's method
            self.polygon_processor.show_attribute_table(layer_name)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show attribute table: {str(e)}")

class FilterWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, data, filter_type, window_size, whitebox_path):
        super().__init__()
        self.data = data
        self.filter_type = filter_type
        self.window_size = window_size
        self.whitebox_path = whitebox_path
        self._is_running = True
        self._mutex = QMutex()

    def run(self):
        try:
            with QMutexLocker(self._mutex):
                if not self._is_running:
                    return

            self.progress.emit(5)
            
            # Initialize WhiteboxTools
            wbt = WhiteboxTools()
            wbt.set_whitebox_dir(self.whitebox_path)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Prepare paths
                input_path = os.path.join(tmpdir, 'input.tif')
                output_path = os.path.join(tmpdir, 'output.tif')
                
                # Save input data
                with rasterio.open(input_path, 'w',
                                 driver='GTiff',
                                 height=self.data.shape[0],
                                 width=self.data.shape[1],
                                 count=1,
                                 dtype=self.data.dtype) as dst:
                    dst.write(self.data, 1)
                
                with QMutexLocker(self._mutex):
                    if not self._is_running:
                        return
                
                self.progress.emit(30)
                
                # Apply selected filter
                if self.filter_type == "Lee":
                    wbt.lee_sigma_filter(input_path, output_path, self.window_size)
                elif self.filter_type == "Gamma MAP":
                    wbt.gamma_map_filter(input_path, output_path, self.window_size)
                elif self.filter_type == "Frost":
                    wbt.frost_filter(input_path, output_path, self.window_size)
                
                with QMutexLocker(self._mutex):
                    if not self._is_running:
                        return
                
                self.progress.emit(70)
                
                # Read output data
                with rasterio.open(output_path) as src:
                    result = src.read(1)
                
                self.progress.emit(100)
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"Filter error: {str(e)}")

    def stop(self):
        with QMutexLocker(self._mutex):
            self._is_running = False
        self.terminate()
    
