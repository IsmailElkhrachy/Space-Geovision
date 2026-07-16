import os
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from shapely.ops import split, unary_union
from scipy.spatial import KDTree, Voronoi
from PyQt5.QtWidgets import (QMessageBox, QInputDialog, QDialog, QListWidgetItem,QGroupBox,QWidget,QSpinBox,
                            QVBoxLayout, QTextEdit, QTableWidget, QComboBox, QLineEdit, QFrame,QToolBar,QListWidget,
                            QDialogButtonBox, QLabel, QTableWidgetItem, QScrollArea, QSizePolicy,
                            QPushButton, QFileDialog, QAction, QMenu, QHBoxLayout,QStackedWidget,
                            QMenuBar, QCheckBox)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import  QCursor, QIcon
import logging
from PyQt5.QtWidgets import QComboBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import tempfile
import shutil

import traceback




class PolygonProcessor:
    def __init__(self, ui_reference):
        self.ui = ui_reference
        self.current_gdf = None  # Will store GeoDataFrame
        self.editing_active = False
        self.original_file_path = None
        self.edit_buffer = None  # To store temporary edits

    def start_editing(self):
        """Begin editing session with proper user notification"""
        if self.current_gdf is None:
            self.ui.show_warning("No layer loaded! Load a vector layer first.")
            return False
            
        self.editing_active = True
        self.edit_buffer = self.current_gdf.copy()  # Create working copy
        self.ui.toggle_edit_buttons(True)
        
        # Get layer name for message
        layer_name = os.path.basename(self.original_file_path) if hasattr(self, 'original_file_path') else "current layer"
        
        # Use show_info for consistent notification
        self.ui.show_info(f"Editing session started for: {layer_name}\n"
                        "• Changes are temporary until saved\n"
                        "• Use Save Edits to commit changes\n"
                        "• Use Stop Editing to cancel")
        
        return True

    def save_edits(self):
        """Save edits to vector file with comprehensive validation and feedback"""
        try:
            if not self.editing_active:
                self.ui.show_warning("Cannot save - no active editing session")
                return False

            if self.current_gdf is None:
                self.ui.show_warning("Cannot save - no layer data available")
                return False

            # Validate geometries before saving
            invalid_geoms = ~self.current_gdf.geometry.is_valid
            if any(invalid_geoms):
                invalid_count = sum(invalid_geoms)
                self.ui.show_warning(
                    f"Cannot save - {invalid_count} invalid geometry(s) found\n"
                    "Please fix or remove invalid features before saving."
                )
                return False

            # Handle different file types
            if hasattr(self, 'original_file_path'):
                if self.original_file_path.endswith('.shp'):
                    success = self._save_shapefile()
                else:
                    success = self._save_geojson_gpkg()
            else:
                self.ui.show_warning("Cannot determine original file path")
                return False

            if success:
                self.ui.show_info("All edits saved successfully")
                return True
            else:
                self.ui.show_error("Failed to save edits")
                return False

        except Exception as e:
            self.ui.show_error(f"Unexpected error during save: {str(e)}")
            return False

    def stop_editing(self, save_changes=True):
        """Stop editing session with proper user confirmation and feedback"""
        try:
            if not self.editing_active:
                return True
                
            if save_changes:
                reply = QMessageBox.question(
                    self.ui, 
                    "Confirm Save Edits",
                    "Would you like to save your edits before stopping?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    if not self.save_edits():
                        self.ui.show_warning("Edits were not saved")
                        return False
                elif reply == QMessageBox.Cancel:
                    self.ui.show_info("Editing session continues - cancel was selected")
                    return False
                    
            self.editing_active = False
            self.current_editing_layer = None
            self.ui.toggle_edit_buttons(False)
            self.ui.show_info("Editing session ended successfully")
            return True
            
        except Exception as e:
            self.ui.show_error(f"Failed to stop editing: {str(e)}")
            return False


    
    def _save_shapefile(self):
        """Special handling for ESRI Shapefile components"""
        try:
            # 1. Create temp directory
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "temp.shp")
            
            # 2. Save to temp location
            self.current_gdf.to_file(temp_path, driver='ESRI Shapefile')
            
            # 3. Verify save
            test_gdf = gpd.read_file(temp_path)
            if len(test_gdf) != len(self.current_gdf):
                raise ValueError("Feature count mismatch after save")
            
            # 4. Replace original files
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                src = temp_path.replace('.shp', ext)
                if os.path.exists(src):
                    shutil.move(src, self.original_file_path.replace('.shp', ext))
            
            return True
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _save_geojson_gpkg(self):
        """Save GeoJSON or GeoPackage"""
        driver = 'GeoJSON' if self.original_file_path.endswith('.geojson') else 'GPKG'
        
        # Create backup first
        if os.path.exists(self.original_file_path):
            backup_path = self.original_file_path + '.bak'
            shutil.copy2(self.original_file_path, backup_path)
        
        # Save to original path
        self.current_gdf.to_file(self.original_file_path, driver=driver)
        
        # Verify
        test_gdf = gpd.read_file(self.original_file_path)
        if len(test_gdf) != len(self.current_gdf):
            if os.path.exists(backup_path):
                shutil.move(backup_path, self.original_file_path)  # Restore backup
            raise ValueError("Feature count mismatch after save")
        
        return True
    
    def reload_layer(self, layer_name):
        """Properly reload a vector layer with validation"""
        if layer_name not in self.layers:
            return False
        
        file_path = self.layers[layer_name]['file_name']
        
        try:
            # Load with geometry validation
            gdf = gpd.read_file(file_path)
            gdf.geometry = gdf.geometry.make_valid()  # Fix invalid geometries
            
            # Preserve original CRS
            if 'crs' in self.layers[layer_name]['gdf']:
                gdf.crs = self.layers[layer_name]['gdf'].crs
                
            # Update storage
            self.layers[layer_name]['gdf'] = gdf
            self.polygon_processor.current_gdf = gdf
            
            # Refresh display
            self.display_vector_layer(self.layers[layer_name], layer_name)
            return True
            
        except Exception as e:
            self.show_error(f"Reload failed: {str(e)}")
            return False

    def _get_driver(self):
        ext = os.path.splitext(self.original_file_path)[1].lower()
        return {
            '.shp': 'ESRI Shapefile',
            '.geojson': 'GeoJSON',
            '.gpkg': 'GPKG'
        }.get(ext, 'ESRI Shapefile')

    def _fix_shapefile_components(self, temp_path):
        base = os.path.splitext(temp_path)[0]
        for ext in ['.shx', '.dbf', '.prj', '.cpg']:
            if os.path.exists(base + ext):
                os.replace(base + ext, 
                        os.path.splitext(self.original_file_path)[0] + ext)



    def save_as(self):
        """Save edits to a new location"""
        try:
            if not self.editing_active:
                QMessageBox.warning(self.ui, "Warning", "No active editing session")
                return False
                
            layer_name = self.current_editing_layer
            gdf = self.ui.layers[layer_name]['gdf']
            
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self.ui, "Save Layer As", 
                layer_name,
                "Shapefiles (*.shp);;GeoJSON (*.geojson);;GeoPackage (*.gpkg)")
                
            if not file_path:
                return False  # User canceled
                
            # Determine driver from selected filter
            if selected_filter == "Shapefiles (*.shp)":
                driver = 'ESRI Shapefile'
                if not file_path.lower().endswith('.shp'):
                    file_path += '.shp'
            elif selected_filter == "GeoJSON (*.geojson)":
                driver = 'GeoJSON'
                if not file_path.lower().endswith('.geojson'):
                    file_path += '.geojson'
            else:  # GeoPackage
                driver = 'GPKG'
                if not file_path.lower().endswith('.gpkg'):
                    file_path += '.gpkg'
            
            # Save the file
            gdf.to_file(file_path, driver=driver)
            
            # For shapefiles, save the projection file if CRS exists
            if driver == 'ESRI Shapefile' and gdf.crs:
                prj_path = os.path.splitext(file_path)[0] + '.prj'
                with open(prj_path, 'w') as f:
                    f.write(gdf.crs.to_wkt())
            
            # Update the layer's path
            self.ui.layers[layer_name]['path'] = file_path
            self.original_file_path = file_path
            
            QMessageBox.information(self.ui, "Success", 
                                  f"Layer saved to:\n{file_path}")
            self.editing_active = False
            return True
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to save layer: {str(e)}")
            return False

 
    # Helper Methods
    def _get_driver(self, file_path):
        """Get appropriate driver based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.shp':
            return 'ESRI Shapefile'
        elif ext == '.geojson':
            return 'GeoJSON'
        elif ext == '.gpkg':
            return 'GPKG'
        else:
            return None

    def _save_layer_as(self, layer_name):
        """Save layer to a new file location"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self.ui, 
                "Save Layer As",
                layer_name,
                "Shapefiles (*.shp);;GeoJSON (*.geojson);;GPKG (*.gpkg)"
            )
            
            if file_path:
                driver = self._get_driver(file_path)
                if not driver:
                    QMessageBox.critical(self.ui, "Error", 
                        "Unsupported file format")
                    return
                    
                gdf = self.ui.layers[layer_name]['gdf']
                gdf.to_file(file_path, driver=driver)
                
                # Update layer metadata
                self.ui.layers[layer_name]['path'] = file_path
                self.ui.layers[layer_name]['is_new'] = False
                
                QMessageBox.information(self.ui, "Success", 
                    f"Layer saved to:\n{file_path}")
                    
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Save failed: {str(e)}")
    
    def _save_layer_to_file(self, layer_name, file_path=None):
        """Save layer to specified file or its original path"""
        try:
            gdf = self.ui.layers[layer_name]['gdf']
            if file_path is None:
                if 'path' not in self.ui.layers[layer_name]:
                    return False
                file_path = self.ui.layers[layer_name]['path']
            
            driver = self._get_driver(file_path)
            if not driver:
                return False
                
            gdf.to_file(file_path, driver=driver)
            return True
        except Exception as e:
            logging.error(f"Error saving layer: {str(e)}")
            return False

    def get_selected_vector_layer(self, prompt="Select a polygon layer"):
        """Get the currently selected vector layer from the UI"""
        try:
            items = self.ui.layer_list.selectedItems()
            if not items:
                QMessageBox.warning(self.ui, "Warning", prompt)
                return None
                
            layer_name = items[0].text()
            
            if layer_name not in self.ui.layers:
                QMessageBox.warning(self.ui, "Warning", f"Layer '{layer_name}' not found!")
                return None
                
            layer_data = self.ui.layers[layer_name]
            
            if 'gdf' not in layer_data:
                QMessageBox.warning(self.ui, "Warning", "Selected layer has no GeoDataFrame!")
                return None
                
            return layer_name
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Error getting selected layer: {str(e)}")
            return None

    def add_vector_layer(self, gdf, layer_name, file_path=None):
        """Add a new vector layer to the UI with unique naming"""
        try:
            if layer_name in self.ui.layers:
                i = 1
                while f"{layer_name}_{i}" in self.ui.layers:
                    i += 1
                layer_name = f"{layer_name}_{i}"
            
            # Reset index to convert FID from index to column
            gdf = gdf.reset_index().rename(columns={'index': 'FID'})
            
            # Store both the GeoDataFrame and the original file path
            self.ui.layers[layer_name] = {
                'gdf': gdf,
                'type': 'vector',
                'path': file_path,  # Can be None for new layers
                'is_new': file_path is None  # Flag for newly created layers
            }
            self.ui.layer_list.addItem(QListWidgetItem(layer_name))
            return layer_name
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to add layer: {str(e)}")
            return None

    # Attribute Table Methods
    def show_attribute_table(self, layer_name=None):
        """Display read-only attribute table"""
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            self._create_attribute_dialog(gdf, layer_name, editable=False)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to show attribute table: {str(e)}")

    def edit_attributes(self, layer_name=None):
        """Display editable attribute table"""
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            self._create_attribute_dialog(gdf, layer_name, editable=True)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to edit attributes: {str(e)}")

    # def _create_attribute_dialog(self, gdf, layer_name, editable=False):
    #     """Create attribute table dialog with enhanced tools (calculator, sorting, graphs)"""
    #     try:
    #         dialog = QDialog(self.ui)
    #         dialog.setWindowTitle(f"{'Edit' if editable else 'View'} Attributes - {layer_name}")
    #         dialog.resize(1200, 700)  # Slightly larger for better tool visibility
    #         layout = QVBoxLayout(dialog)
            
    #         # Create menu bar with enhanced tools
    #         menubar = QMenuBar(dialog)
    #         file_menu = menubar.addMenu("File")
    #         tools_menu = menubar.addMenu("Tools")
    #         sort_menu = menubar.addMenu("Sort")  # New sort menu
            
    #         if editable:
    #             edit_menu = menubar.addMenu("Edit")
            
    #         # Create table widget
    #         self.current_table = self._create_table_widget(gdf, editable)
            
    #         # File menu actions
    #         save_action = QAction(QIcon(":/icons/save.png"), "Save Changes", dialog)
    #         save_action.setShortcut("Ctrl+S")
    #         save_action.triggered.connect(lambda: self._save_changes(gdf, layer_name, dialog))
    #         file_menu.addAction(save_action)
            
    #         export_action = QAction(QIcon(":/icons/export.png"), "Export to CSV", dialog)
    #         export_action.setShortcut("Ctrl+E")
    #         export_action.triggered.connect(lambda: self.export_attributes(layer_name))
    #         file_menu.addAction(export_action)
            
    #         # Edit menu actions (if editable)
    #         if editable:
    #             add_field_action = QAction(QIcon("D:\\Python_Codes\\icons\\add_column.png"), "Add Field", dialog)
    #             add_field_action.setShortcut("Ctrl+N")
    #             add_field_action.triggered.connect(lambda: self.add_attribute_field(layer_name))
    #             edit_menu.addAction(add_field_action)
                
    #             del_field_action = QAction(QIcon("D:\\Python_Codes\\icons\\column-delete.png"), "Delete Field", dialog)
    #             del_field_action.setShortcut("Ctrl+D")
    #             del_field_action.triggered.connect(lambda: self.delete_attribute_field(layer_name))
    #             edit_menu.addAction(del_field_action)
            
    #         # Tools menu actions
    #         calc_action = QAction(QIcon("D:\\Python_Codes\\icons\\field_calculate.png"), "Field Calculator", dialog)
    #         calc_action.setShortcut("Ctrl+F")
    #         calc_action.triggered.connect(lambda: self._show_enhanced_calculator(gdf, layer_name))
    #         tools_menu.addAction(calc_action)
            
    #         stats_action = QAction(QIcon("D:\\Python_Codes\\icons\\Field_Statistics.png"), "Field Statistics", dialog)
    #         stats_action.triggered.connect(lambda: self.show_field_statistics(layer_name))
    #         tools_menu.addAction(stats_action)
            
    #         graph_action = QAction(QIcon("D:\\Python_Codes\\icons\\create_graph.png"), "Create Graph", dialog)
    #         graph_action.triggered.connect(lambda: self.create_graph(layer_name))
    #         tools_menu.addAction(graph_action)
            
    #         # Sort menu actions
    #         sort_asc_action = QAction(QIcon("D:\\Python_Codes\\icons\\Sort_ascending.png"), "Sort Ascending", dialog)
    #         sort_asc_action.triggered.connect(lambda: self._sort_table(True))
    #         sort_menu.addAction(sort_asc_action)
            
    #         sort_desc_action = QAction(QIcon("D:\\Python_Codes\\icons\\Sort_dscending.png"), "Sort Descending", dialog)
    #         sort_desc_action.triggered.connect(lambda: self._sort_table(False))
    #         sort_menu.addAction(sort_desc_action)
            
    #         advanced_sort_action = QAction(QIcon("D:\\Python_Codes\\icons\\Advanced_Sort.png"), "Advanced Sort...", dialog)
    #         advanced_sort_action.triggered.connect(lambda: self._show_advanced_sort(gdf, layer_name))
    #         sort_menu.addAction(advanced_sort_action)
            
    #         # Add toolbar for quick access
    #         toolbar = QToolBar("Tools", dialog)
    #         if editable:
    #             toolbar.addAction(add_field_action)
    #             toolbar.addAction(del_field_action)
    #         toolbar.addAction(calc_action)
    #         toolbar.addAction(stats_action)
    #         toolbar.addAction(graph_action)
    #         toolbar.addSeparator()
    #         toolbar.addAction(sort_asc_action)
    #         toolbar.addAction(sort_desc_action)
    #         toolbar.addAction(advanced_sort_action)
            
    #         # Setup layout
    #         scroll = QScrollArea()
    #         scroll.setWidget(self.current_table)
    #         scroll.setWidgetResizable(True)
            
    #         layout.setMenuBar(menubar)
    #         layout.addWidget(toolbar)
    #         layout.addWidget(scroll)
            
    #         # Enable sorting on columns
    #         self.current_table.setSortingEnabled(True)
    #         self.current_table.horizontalHeader().setSectionsClickable(True)
            
    #         dialog.exec_()
            
    #     except Exception as e:
    #         QMessageBox.critical(self.ui, "Error", f"Failed to create dialog: {str(e)}")
    #         traceback.print_exc()

    def _create_attribute_dialog(self, gdf, layer_name, editable=False):
        """Create enhanced attribute table dialog with editing tools, sorting, and analysis features"""
        try:
            dialog = QDialog(self.ui)
            dialog.setWindowTitle(f"{'Edit' if editable else 'View'} Attributes - {layer_name}")
            dialog.resize(1200, 700)
            layout = QVBoxLayout(dialog)
            
            # Create menu bar with all tools
            menubar = QMenuBar(dialog)
            file_menu = menubar.addMenu("File")
            tools_menu = menubar.addMenu("Tools")
            sort_menu = menubar.addMenu("Sort")
            
            if editable:
                edit_menu = menubar.addMenu("Edit")
            
            # Create table widget
            self.current_table = self._create_table_widget(gdf, editable)
            
            # File menu actions
            save_action = QAction(QIcon(":/icons/save.png"), "Save Changes", dialog)
            save_action.setShortcut("Ctrl+S")
            save_action.triggered.connect(lambda: self._save_changes(gdf, layer_name, dialog))
            file_menu.addAction(save_action)
            
            export_action = QAction(QIcon(":/icons/export.png"), "Export to CSV", dialog)
            export_action.setShortcut("Ctrl+E")
            export_action.triggered.connect(lambda: self.export_attributes(layer_name))
            file_menu.addAction(export_action)
            
            # Edit menu actions (if editable)
            if editable:
                # Row operations
                delete_action = QAction(QIcon("D:\\Python_Codes\\icons\\Delete_Selected_Rows.png"), "Delete Selected Rows", dialog)
                delete_action.setShortcut("Del")
                delete_action.triggered.connect(lambda: self._delete_selected_rows(gdf, layer_name))
                edit_menu.addAction(delete_action)
                
                # Field operations
                add_field_action = QAction(QIcon("D:\\Python_Codes\\icons\\add_column.png"), "Add Field", dialog)
                add_field_action.setShortcut("Ctrl+N")
                add_field_action.triggered.connect(lambda: self.add_attribute_field(layer_name))
                edit_menu.addAction(add_field_action)
                
                del_field_action = QAction(QIcon("D:\\Python_Codes\\icons\\column-delete.png"), "Delete Field", dialog)
                del_field_action.setShortcut("Ctrl+D")
                del_field_action.triggered.connect(lambda: self.delete_attribute_field(layer_name))
                edit_menu.addAction(del_field_action)
            
            # Tools menu actions
            calc_action = QAction(QIcon("D:\\Python_Codes\\icons\\field_calculate.png"), "Field Calculator", dialog)
            calc_action.setShortcut("Ctrl+F")
            calc_action.triggered.connect(lambda: self._show_enhanced_calculator(gdf, layer_name))
            tools_menu.addAction(calc_action)
            
            stats_action = QAction(QIcon("D:\\Python_Codes\\icons\\Field_Statistics.png"), "Field Statistics", dialog)
            stats_action.triggered.connect(lambda: self.show_field_statistics(layer_name))
            tools_menu.addAction(stats_action)
            
            graph_action = QAction(QIcon("D:\\Python_Codes\\icons\\create_graph.png"), "Create Graph", dialog)
            graph_action.triggered.connect(lambda: self.create_graph(layer_name))
            tools_menu.addAction(graph_action)
            
            # Sort menu actions
            sort_asc_action = QAction(QIcon("D:\\Python_Codes\\icons\\Sort_ascending.png"), "Sort Ascending", dialog)
            sort_asc_action.triggered.connect(lambda: self._sort_table(True))
            sort_menu.addAction(sort_asc_action)
            
            sort_desc_action = QAction(QIcon("D:\\Python_Codes\\icons\\Sort_dscending.png"), "Sort Descending", dialog)
            sort_desc_action.triggered.connect(lambda: self._sort_table(False))
            sort_menu.addAction(sort_desc_action)
            
            advanced_sort_action = QAction(QIcon("D:\\Python_Codes\\icons\\Advanced_Sort.png"), "Advanced Sort...", dialog)
            advanced_sort_action.triggered.connect(lambda: self._show_advanced_sort(gdf, layer_name))
            sort_menu.addAction(advanced_sort_action)
            
            # Create toolbar for quick access
            toolbar = QToolBar("Tools", dialog)
            if editable:
                toolbar.addAction(add_field_action)
                toolbar.addAction(del_field_action)
                toolbar.addAction(delete_action)
                toolbar.addSeparator()
            
            toolbar.addAction(calc_action)
            toolbar.addAction(stats_action)
            toolbar.addAction(graph_action)
            toolbar.addSeparator()
            toolbar.addAction(sort_asc_action)
            toolbar.addAction(sort_desc_action)
            toolbar.addAction(advanced_sort_action)
            
            # Setup layout
            scroll = QScrollArea()
            scroll.setWidget(self.current_table)
            scroll.setWidgetResizable(True)
            
            layout.setMenuBar(menubar)
            layout.addWidget(toolbar)
            layout.addWidget(scroll)
            
            # Enable table features
            self.current_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.current_table.setSelectionMode(QTableWidget.ExtendedSelection)
            self.current_table.setSortingEnabled(True)
            self.current_table.horizontalHeader().setSectionsClickable(True)
            
            dialog.exec_()
            
        except Exception as e:
            self.ui.show_error(f"Failed to create dialog: {str(e)}")
            traceback.print_exc()

    def _delete_selected_rows(self, gdf, layer_name):
        """Delete selected rows from the GeoDataFrame and properly update the table"""
        try:
            if not self.editing_active:
                self.ui.show_warning("You must be in editing mode to delete features")
                return
                
            selected_rows = sorted({index.row() for index in 
                                self.current_table.selectedIndexes()}, reverse=True)
            
            if not selected_rows:
                self.ui.show_warning("No rows selected for deletion")
                return
                
            # Confirm deletion
            reply = QMessageBox.question(
                self.ui,
                "Confirm Deletion",
                f"Delete {len(selected_rows)} selected feature(s)?\nThis action cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Create a list of actual indices in the GeoDataFrame
                gdf_indices = [gdf.index[row] for row in selected_rows]
                
                # Delete from GeoDataFrame (working copy)
                gdf.drop(gdf_indices, inplace=True)
                
                # Reset the index if needed (optional)
                gdf.reset_index(drop=True, inplace=True)
                
                # Completely refresh the table with the updated data
                self._refresh_table(gdf)
                
                self.ui.show_info(f"Deleted {len(selected_rows)} feature(s)")
                
        except Exception as e:
            self.ui.show_error(f"Failed to delete features: {str(e)}")
            traceback.print_exc()

    def _refresh_table(self, gdf):
        """Completely refresh the table widget with new data"""
        try:
            # Disable sorting during update to prevent issues
            self.current_table.setSortingEnabled(False)
            self.current_table.blockSignals(True)
            
            # Clear and repopulate the entire table
            self.current_table.clear()
            self.current_table.setRowCount(len(gdf))
            self.current_table.setColumnCount(len(gdf.columns))
            self.current_table.setHorizontalHeaderLabels(gdf.columns)
            
            # Populate the table with data
            for i, row in gdf.iterrows():
                for j, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    if not self.editing_active:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.current_table.setItem(i, j, item)
            
            # Adjust column widths
            self.current_table.resizeColumnsToContents()
            
        finally:
            # Re-enable sorting after update
            self.current_table.setSortingEnabled(True)
            self.current_table.blockSignals(False)
 

 
    def _show_enhanced_calculator(self, gdf, layer_name):
        """Show the enhanced field calculator dialog"""
        dialog = QDialog(self.ui)
        dialog.setWindowTitle(f"Field Calculator - {layer_name}")
        dialog.setMinimumWidth(600)
        layout = QVBoxLayout(dialog)
        
        # Field selection
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Target Field:"))
        
        field_combo = QComboBox()
        field_combo.addItems([col for col in gdf.columns if col != 'geometry'])
        field_layout.addWidget(field_combo)
        
        new_field_check = QCheckBox("Create new field")
        field_layout.addWidget(new_field_check)
        
        new_field_edit = QLineEdit()
        new_field_edit.setPlaceholderText("New field name")
        new_field_edit.setEnabled(False)
        field_layout.addWidget(new_field_edit)
        
        new_field_check.stateChanged.connect(lambda state: new_field_edit.setEnabled(state))
        layout.addLayout(field_layout)
        
        # Expression input
        expr_group = QGroupBox("Expression")
        expr_layout = QVBoxLayout()
        
        expr_edit = QTextEdit()
        expr_edit.setPlaceholderText("Enter calculation (e.g., 'Area * 0.0001' or 'IF(Population>1000,\"Large\",\"Small\")')")
        expr_layout.addWidget(expr_edit)
        
        # Function reference button
        func_btn = QPushButton("Show Functions Reference")
        func_btn.clicked.connect(self._show_function_reference)
        expr_layout.addWidget(func_btn)
        expr_group.setLayout(expr_layout)
        layout.addWidget(expr_group)
        
        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(lambda: self._apply_calculation(
            gdf, layer_name, 
            field_combo.currentText(),
            new_field_check.isChecked(),
            new_field_edit.text(),
            expr_edit.toPlainText(),
            dialog
        ))
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def _show_advanced_sort(self, gdf, layer_name):
        """Show advanced multi-column sort dialog"""
        dialog = QDialog(self.ui)
        dialog.setWindowTitle(f"Advanced Sort - {layer_name}")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        
        # Sort criteria list
        sort_list = QListWidget()
        layout.addWidget(sort_list)
        
        # Add sort criterion controls
        add_group = QGroupBox("Add Sort Criterion")
        add_layout = QHBoxLayout()
        
        field_combo = QComboBox()
        field_combo.addItems([col for col in gdf.columns if col != 'geometry'])
        add_layout.addWidget(field_combo)
        
        order_combo = QComboBox()
        order_combo.addItems(["Ascending", "Descending"])
        add_layout.addWidget(order_combo)
        
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(lambda: self._add_sort_criterion(
            field_combo.currentText(),
            order_combo.currentText(),
            sort_list
        ))
        add_layout.addWidget(add_btn)
        add_group.setLayout(add_layout)
        layout.addWidget(add_group)
        
        # Action buttons
        btn_box = QDialogButtonBox()
        apply_btn = btn_box.addButton("Apply Sort", QDialogButtonBox.ApplyRole)
        apply_btn.clicked.connect(lambda: self._apply_advanced_sort(
            gdf, layer_name, sort_list, dialog
        ))
        btn_box.addButton(QDialogButtonBox.Cancel).clicked.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def _apply_calculation(self, gdf, layer_name, field_name, is_new_field, new_name, expr, dialog):
        """Apply field calculation and refresh table"""
        try:
            if not expr.strip():
                QMessageBox.warning(self.ui, "Error", "Please enter an expression")
                return
                
            if is_new_field:
                if not new_name.strip():
                    QMessageBox.warning(self.ui, "Error", "Please enter a new field name")
                    return
                if new_name in gdf.columns:
                    QMessageBox.warning(self.ui, "Error", "Field name already exists")
                    return
                field_name = new_name
            
            # Calculate using pandas eval
            result = gdf.eval(expr, engine='python')
            gdf[field_name] = result
            
            # Refresh table
            self._refresh_table(gdf)
            
            QMessageBox.information(self.ui, "Success", 
                                f"Field '{field_name}' calculated successfully")
            dialog.accept()
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Calculation failed: {str(e)}")

    def _add_sort_criterion(self, field, order, sort_list):
        """Add sort criterion to the list"""
        item_text = f"{field} ({order})"
        
        # Remove existing sort for this field if present
        for i in range(sort_list.count()):
            if field in sort_list.item(i).text():
                sort_list.takeItem(i)
                break
                
        sort_list.addItem(item_text)

    def _apply_advanced_sort(self, gdf, layer_name, sort_list, dialog):
        """Apply the multi-column sort"""
        if sort_list.count() == 0:
            QMessageBox.warning(self.ui, "Error", "No sort criteria specified")
            return
            
        try:
            by = []
            ascending = []
            
            for i in range(sort_list.count()):
                item_text = sort_list.item(i).text()
                field = item_text.split(" (")[0]
                order = item_text.split("(")[1][:-1]
                
                by.append(field)
                ascending.append(order == "Ascending")
            
            # Perform the sort
            gdf.sort_values(by=by, ascending=ascending, inplace=True)
            
            # Refresh table
            self._refresh_table(gdf)
            
            QMessageBox.information(self.ui, "Success", "Data sorted successfully")
            dialog.accept()
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Sorting failed: {str(e)}")

 
    def _show_function_reference(self):
        """Show field calculator function reference"""
        func_ref = """
        <b>Field Calculator Functions Reference</b><br><br>
        <b>Basic Math:</b> +, -, *, /, ^ (power)<br>
        <b>Logical:</b> AND, OR, NOT, >, <, >=, <=, =, !=<br>
        <b>Functions:</b><br>
        - ABS(x): Absolute value<br>
        - SQRT(x): Square root<br>
        - LOG(x): Natural logarithm<br>
        - LOG10(x): Base-10 logarithm<br>
        - SIN(x), COS(x), TAN(x): Trigonometric functions<br>
        - ROUND(x,n): Round to n decimals<br>
        - CONCAT(str1,str2): Concatenate strings<br>
        - UPPER(str), LOWER(str): Case conversion<br>
        - LEN(str): String length<br>
        - REPLACE(str,old,new): String replacement<br>
        - IF(cond,val_if_true,val_if_false): Conditional<br><br>
        <b>Examples:</b><br>
        - Area conversion: "Shape_Area * 0.0001"<br>
        - Density calculation: "Population / Area"<br>
        - Conditional: "IF(Population > 1000, 'Large', 'Small')"
        """
        QMessageBox.information(self.ui, "Functions Reference", func_ref)

    def _sort_table(self, ascending=True):
        """Sort table by selected column"""
        if hasattr(self, 'current_table'):
            col = self.current_table.currentColumn()
            if col >= 0:
                self.current_table.sortItems(col, Qt.AscendingOrder if ascending else Qt.DescendingOrder)

    def _create_table_widget(self, gdf, editable):
        """Create and populate QTableWidget from GeoDataFrame"""
        
        # Ensure FID exists
        if 'FID' not in gdf.columns:
            gdf = gdf.reset_index().rename(columns={'index': 'FID'})

        table = QTableWidget()
        table.setRowCount(len(gdf))
        table.setColumnCount(len(gdf.columns))
        table.setHorizontalHeaderLabels(gdf.columns.tolist())
        
        table.setEditTriggers(
            QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed if editable 
            else QTableWidget.NoEditTriggers
        )
        
        for i, row in gdf.iterrows():
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val) if not pd.isna(val) else "")
                if gdf.columns[j] == 'geometry':
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(i, j, item)
        
        return table
    
    def _save_changes(self, gdf, layer_name, dialog):
        """Save changes from table back to GeoDataFrame and file"""
        try:
            # Update in-memory GeoDataFrame
            for i in range(self.current_table.rowCount()):
                for j in range(self.current_table.columnCount()):
                    col = gdf.columns[j]
                    if col == 'geometry': 
                        continue
                    item = self.current_table.item(i, j)
                    if item:
                        gdf.iloc[i, j] = item.text()
            
            # Update in-memory storage
            self.ui.layers[layer_name]['gdf'] = gdf
            
            # For new layers without a path, automatically trigger Save As
            if self.ui.layers[layer_name].get('is_new', True) or 'path' not in self.ui.layers[layer_name]:
                reply = QMessageBox.question(
                    dialog,
                    "Save New Layer",
                    "This layer doesn't have a saved file yet.\nWould you like to save it now?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self._save_layer_as(layer_name)
                return
            
            # Save to original file
            file_path = self.ui.layers[layer_name]['path']
            driver = self._get_driver(file_path)
            
            if not driver:
                QMessageBox.warning(dialog, "Cannot Save", 
                    f"Unsupported file format: {file_path}")
                return
                
            try:
                gdf.to_file(file_path, driver=driver)
                self.ui.layers[layer_name]['is_new'] = False  # No longer new
                QMessageBox.information(dialog, "Success", 
                    f"Changes saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(dialog, "Save Failed", 
                    f"Could not save to file:\n{str(e)}")
            
        except Exception as e:
            QMessageBox.critical(dialog, "Error", f"Save failed: {str(e)}")

    # Field Operations
    def add_attribute_field(self, layer_name=None):
        """Add new field with ArcMap-style dialog for field properties"""
        if not self.editing_active:
            QMessageBox.warning(self.ui, "Warning", 
                              "Please start an editing session first")
            return
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            
            # Create the dialog
            dialog = QDialog(self.ui)
            dialog.setWindowTitle("Add Field")
            dialog.resize(400, 300)
            layout = QVBoxLayout(dialog)
            
            # Field Name
            name_layout = QHBoxLayout()
            name_layout.addWidget(QLabel("Field Name:"))
            self.field_name_edit = QLineEdit()
            name_layout.addWidget(self.field_name_edit)
            layout.addLayout(name_layout)
            
            # Field Type
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("Field Type:"))
            self.field_type_combo = QComboBox()
            self.field_type_combo.addItems([
                "Text",
                "Short Integer (16-bit)",
                "Long Integer (32-bit)",
                "Float (32-bit)",
                "Double (64-bit)",
                "Date"
            ])
            type_layout.addWidget(self.field_type_combo)
            layout.addLayout(type_layout)
            
            # Field Properties (dynamic based on type)
            self.props_container = QStackedWidget()
            
            # Text properties
            text_widget = QWidget()
            text_layout = QVBoxLayout(text_widget)
            text_length = QSpinBox()
            text_length.setRange(1, 255)
            text_length.setValue(50)
            text_layout.addWidget(QLabel("Length:"))
            text_layout.addWidget(text_length)
            self.props_container.addWidget(text_widget)
            
            # Numeric properties
            numeric_widget = QWidget()
            numeric_layout = QVBoxLayout(numeric_widget)
            
            precision_layout = QHBoxLayout()
            precision_layout.addWidget(QLabel("Precision:"))
            self.precision_spin = QSpinBox()
            self.precision_spin.setRange(1, 18)
            self.precision_spin.setValue(6)
            precision_layout.addWidget(self.precision_spin)
            numeric_layout.addLayout(precision_layout)
            
            scale_layout = QHBoxLayout()
            scale_layout.addWidget(QLabel("Scale:"))
            self.scale_spin = QSpinBox()
            self.scale_spin.setRange(0, 10)
            self.scale_spin.setValue(2)
            scale_layout.addWidget(self.scale_spin)
            numeric_layout.addLayout(scale_layout)
            
            self.props_container.addWidget(numeric_widget)
            
            # Date properties (no additional settings)
            date_widget = QWidget()
            date_layout = QVBoxLayout(date_widget)
            date_layout.addWidget(QLabel("Date format: YYYY-MM-DD"))
            self.props_container.addWidget(date_widget)
            
            layout.addWidget(self.props_container)
            
            # Connect type change to update properties
            self.field_type_combo.currentTextChanged.connect(self._update_field_properties)
            self._update_field_properties()
            
            # Default value (optional)
            default_layout = QHBoxLayout()
            default_layout.addWidget(QLabel("Default Value:"))
            self.default_value_edit = QLineEdit()
            default_layout.addWidget(self.default_value_edit)
            layout.addLayout(default_layout)
            
            # Button box
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            # Show dialog
            if dialog.exec_() == QDialog.Accepted:
                field_name = self.field_name_edit.text().strip()
                if not field_name:
                    QMessageBox.warning(self.ui, "Error", "Field name cannot be empty!")
                    return
                    
                if field_name in gdf.columns:
                    QMessageBox.warning(self.ui, "Error", "Field already exists!")
                    return
                    
                # Determine field type and create appropriate column
                field_type = self.field_type_combo.currentText()
                if field_type == "Text":
                    length = text_length.value()
                    gdf[field_name] = pd.Series(dtype=f'str')
                elif "Integer" in field_type:
                    dtype = 'int32' if "Long" in field_type else 'int16'
                    gdf[field_name] = pd.Series(dtype=dtype)
                elif "Float" in field_type or "Double" in field_type:
                    dtype = 'float64' if "Double" in field_type else 'float32'
                    gdf[field_name] = pd.Series(dtype=dtype)
                elif field_type == "Date":
                    gdf[field_name] = pd.Series(dtype='datetime64[ns]')
                
                # Set default value if provided
                default_value = self.default_value_edit.text().strip()
                if default_value:
                    try:
                        if field_type == "Date":
                            gdf[field_name] = pd.to_datetime(default_value)
                        elif "Integer" in field_type:
                            gdf[field_name] = int(default_value)
                        elif "Float" in field_type or "Double" in field_type:
                            gdf[field_name] = float(default_value)
                        else:
                            gdf[field_name] = default_value
                    except ValueError:
                        QMessageBox.warning(self.ui, "Error", "Invalid default value for selected type!")
                        return
                
                self.ui.layers[layer_name]['gdf'] = gdf
                self.edit_attributes(layer_name)  # Refresh table
                
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to add field: {str(e)}")

    def _update_field_properties(self):
        """Update the properties panel based on selected field type"""
        field_type = self.field_type_combo.currentText()
        if field_type == "Text":
            self.props_container.setCurrentIndex(0)
        elif "Integer" in field_type or "Float" in field_type or "Double" in field_type:
            self.props_container.setCurrentIndex(1)
        elif field_type == "Date":
            self.props_container.setCurrentIndex(2)

    def delete_attribute_field(self, layer_name=None):
        """Delete field from attribute table"""
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            fields = [col for col in gdf.columns if col != 'geometry']
            
            if not fields:
                QMessageBox.information(self.ui, "Info", "No attributes available to delete")
                return
                
            field, ok = QInputDialog.getItem(
                self.ui, "Delete Field", 
                "Select field to delete:", fields, 0, False)
                
            if ok and field:
                gdf.drop(columns=[field], inplace=True)
                self.ui.layers[layer_name]['gdf'] = gdf
                self.edit_attributes(layer_name)  # Refresh table
                
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to delete field: {str(e)}")

    def calculate_field(self, layer_name=None):
        """Calculate field values using expression"""
        try:
            # Import required modules if not already imported
            from PyQt5.QtWidgets import QApplication
            
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            fields = [col for col in gdf.columns if col != 'geometry']
            
            # Create dialog
            dialog = QDialog(self.ui)
            dialog.setWindowTitle("Field Calculator")
            layout = QVBoxLayout(dialog)
            
            # Field selection
            field_combo = QComboBox()
            field_combo.addItems(fields)
            
            # Expression input
            expr_edit = QLineEdit()
            expr_edit.setPlaceholderText("e.g., 'area*10000' or 'population/area'")
            
            # Buttons
            btn_box = QDialogButtonBox()
            btn_box.addButton("Calculate", QDialogButtonBox.AcceptRole)
            btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)
            btn_box.accepted.connect(dialog.accept)
            btn_box.rejected.connect(dialog.reject)
            
            # Layout
            layout.addWidget(QLabel("Target Field:"))
            layout.addWidget(field_combo)
            layout.addWidget(QLabel("Expression:"))
            layout.addWidget(expr_edit)
            layout.addWidget(btn_box)
            
            if dialog.exec_():
                try:
                    field = field_combo.currentText()
                    expr = expr_edit.text()
                    
                    # Calculate the field
                    gdf[field] = gdf.eval(expr)
                    self.ui.layers[layer_name]['gdf'] = gdf
                    
                    # Force UI update
                    QApplication.processEvents()
                    
                    # If attribute table is open, refresh it
                    if hasattr(self, 'attribute_table') and hasattr(self.attribute_table, 'refresh_view'):
                        self.attribute_table.refresh_view()
                    
                    QMessageBox.information(self.ui, "Success", "Field calculated successfully!")
                    
                except Exception as e:
                    QMessageBox.critical(self.ui, "Error", f"Calculation failed: {str(e)}")
                    
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to calculate field: {str(e)}")

    # Statistics & Export
    def show_field_statistics(self, layer_name=None):
        """Display statistics for numeric fields with improved type detection"""
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            
            # Convert potential numeric columns that might be stored as strings
            numeric_cols = []
            for col in gdf.columns:
                if col == 'geometry':
                    continue
                try:
                    # Attempt to convert to numeric
                    converted = pd.to_numeric(gdf[col], errors='raise')
                    gdf[col] = converted  # Update the column in the DataFrame
                    numeric_cols.append(col)
                except:
                    continue
            
            if not numeric_cols:
                QMessageBox.information(self.ui, "Info", 
                    "No numeric fields found.\n\n"
                    "Tip: Even if fields contain numbers, they might be stored as text.\n"
                    "Try using 'Calculate Field' to create a numeric version.")
                return
                
            # Create statistics dialog
            dialog = QDialog(self.ui)
            dialog.setWindowTitle(f"Field Statistics - {layer_name}")
            layout = QVBoxLayout(dialog)
            
            table = QTableWidget()
            table.setColumnCount(6)
            table.setHorizontalHeaderLabels([
                "Field", "Min", "Max", "Mean", "Std Dev", "Sum"
            ])
            table.setRowCount(len(numeric_cols))
            
            for i, col in enumerate(numeric_cols):
                table.setItem(i, 0, QTableWidgetItem(col))
                table.setItem(i, 1, QTableWidgetItem(f"{gdf[col].min():.2f}"))
                table.setItem(i, 2, QTableWidgetItem(f"{gdf[col].max():.2f}"))
                table.setItem(i, 3, QTableWidgetItem(f"{gdf[col].mean():.2f}"))
                table.setItem(i, 4, QTableWidgetItem(f"{gdf[col].std():.2f}"))
                table.setItem(i, 5, QTableWidgetItem(f"{gdf[col].sum():.2f}"))
            
            layout.addWidget(table)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to show statistics: {str(e)}")

    def export_attributes(self, layer_name=None):
        """Export attributes to CSV"""
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            file_path, _ = QFileDialog.getSaveFileName(
                self.ui, "Export Attributes", 
                f"{layer_name}_attributes.csv", 
                "CSV Files (*.csv)"
            )
            
            if file_path:
                gdf = self.ui.layers[layer_name]['gdf']
                gdf.drop(columns='geometry').to_csv(file_path, index=False)
                QMessageBox.information(self.ui, "Success", f"Attributes exported to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to export attributes: {str(e)}")

    # Basic Operations
    def merge_polygons(self):
        """Merge selected polygons into single feature"""
        try:
            layer_name = self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            
            if not hasattr(gdf, '_selected_features') or not gdf._selected_features:
                QMessageBox.information(self.ui, "Info", "Please select polygons to merge first")
                return
                
            merged_geom = gdf[gdf._selected_features].unary_union
            new_row = gdf.iloc[0].copy()
            new_row.geometry = merged_geom
            new_row['FID'] = f"Merged_{gdf['FID'].min()}-{gdf['FID'].max()}"  # Custom FID
            self.add_vector_layer(gpd.GeoDataFrame([new_row], crs=gdf.crs), f"Merged_{layer_name}")
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to merge polygons: {str(e)}")

    def dissolve_polygons(self):
        """Dissolve polygons by attribute"""
        try:
            layer_name = self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            fields = [col for col in gdf.columns if col != 'geometry']
            
            if not fields:
                QMessageBox.information(self.ui, "Info", "No attributes available for dissolve")
                return
                
            field, ok = QInputDialog.getItem(
                self.ui, "Dissolve Field", 
                "Select attribute to dissolve by:", 
                fields, 0, False)
                
            if ok:
                dissolved = gdf.dissolve(by=field)
                self.add_vector_layer(dissolved, f"Dissolved_{layer_name}")
                
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to dissolve polygons: {str(e)}")

    # ... [Include all other polygon processing methods here] ...

    def setup_ui_connections(self):
        """Connect editing actions to UI"""
        if hasattr(self.ui, 'actionStart_Editing'):
            self.ui.actionStart_Editing.triggered.connect(self.start_editing)
        if hasattr(self.ui, 'actionSave_Edits'):
            self.ui.actionSave_Edits.triggered.connect(self.save_edits)
        if hasattr(self.ui, 'actionSave_As'):
            self.ui.actionSave_As.triggered.connect(self.save_as)
        if hasattr(self.ui, 'actionStop_Editing'):
            self.ui.actionStop_Editing.triggered.connect(lambda: self.stop_editing(True))

    def _show_layer_context_menu(self, position):
        """Show context menu for layer list with graph option"""
        if not self.ui.layer_list.selectedItems():
            return
            
        menu = QMenu()
        show_attr_action = menu.addAction("Show Attribute Table")
        graph_action = menu.addAction("Create Graph")
        export_action = menu.addAction("Export to Shapefile")
        
        action = menu.exec_(self.ui.layer_list.viewport().mapToGlobal(position))
        layer_name = self.ui.layer_list.selectedItems()[0].text()
        
        if action == show_attr_action:
            self.show_attribute_table(layer_name)
        elif action == graph_action:
            self.create_graph(layer_name)
        elif action == export_action:
            self.export_to_shapefile(layer_name)

    def export_to_shapefile(self, layer_name=None):
        """Export layer to shapefile"""
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            file_path, _ = QFileDialog.getSaveFileName(
                self.ui, "Save Shapefile", 
                f"{layer_name}.shp", 
                "Shapefiles (*.shp)")
                
            if file_path:
                gdf.to_file(file_path, driver='ESRI Shapefile')
                if gdf.crs:
                    from pyproj import CRS
                    crs = CRS(gdf.crs)
                    with open(file_path.replace('.shp', '.prj'), 'w') as f:
                        f.write(crs.to_wkt())
                
                # Update the stored path so future saves go here
                self.ui.layers[layer_name]['path'] = file_path
                QMessageBox.information(self.ui, "Success", f"Exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Export failed: {str(e)}")

    def clip_polygons(self):
        """Clip polygons using another polygon layer"""
        try:
            # Get input and clip layers
            input_layer = self.get_selected_vector_layer("Select layer to clip")
            if not input_layer:
                return
                
            clip_layer = self.get_selected_vector_layer("Select clipping layer")
            if not clip_layer:
                return
                
            input_gdf = self.ui.layers[input_layer]['gdf']
            clip_gdf = self.ui.layers[clip_layer]['gdf']
            
            # Perform clip
            clipped = gpd.clip(input_gdf, clip_gdf)
            
            # Add to layers
            layer_name = f"Clipped_{input_layer}"
            self.add_vector_layer(clipped, layer_name)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to clip polygons: {str(e)}")

    def intersect_polygons(self):
        """Intersect two polygon layers"""
        try:
            # Get input layers
            layer1 = self.get_selected_vector_layer("Select first layer")
            if not layer1:
                return
                
            layer2 = self.get_selected_vector_layer("Select second layer")
            if not layer2:
                return
                
            gdf1 = self.ui.layers[layer1]['gdf']
            gdf2 = self.ui.layers[layer2]['gdf']
            
            # Perform intersection
            intersected = gpd.overlay(gdf1, gdf2, how='intersection')
            
            # Add to layers
            layer_name = f"Intersection_{layer1}_{layer2}"
            self.add_vector_layer(intersected, layer_name)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to intersect polygons: {str(e)}")

    def union_polygons(self):
        """Union two polygon layers"""
        try:
            # Get input layers
            layer1 = self.get_selected_vector_layer("Select first layer")
            if not layer1:
                return
                
            layer2 = self.get_selected_vector_layer("Select second layer")
            if not layer2:
                return
                
            gdf1 = self.ui.layers[layer1]['gdf']
            gdf2 = self.ui.layers[layer2]['gdf']
            
            # Perform union
            unioned = gpd.overlay(gdf1, gdf2, how='union')
            
            # Add to layers
            layer_name = f"Union_{layer1}_{layer2}"
            self.add_vector_layer(unioned, layer_name)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to union polygons: {str(e)}")

    def create_buffers(self):
        """Create buffers around polygon features"""
        try:
            selected_layer = self.get_selected_vector_layer()
            if not selected_layer:
                return
                
            gdf = self.ui.layers[selected_layer]['gdf']
            
            # Get buffer distance
            distance, ok = QInputDialog.getDouble(
                self.ui, "Buffer Distance", 
                "Enter buffer distance in map units:", 
                100, 0, 10000, 2)
                
            if not ok:
                return
                
            # Create buffers
            buffered = gdf.copy()
            buffered.geometry = buffered.geometry.buffer(distance)
            
            # Add to layers
            layer_name = f"Buffers_{distance}_{selected_layer}"
            self.add_vector_layer(buffered, layer_name)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to create buffers: {str(e)}")

    def simplify_polygons(self):
        """Simplify polygon geometries"""
        try:
            selected_layer = self.get_selected_vector_layer()
            if not selected_layer:
                return
                
            gdf = self.ui.layers[selected_layer]['gdf']
            
            # Get tolerance
            tolerance, ok = QInputDialog.getDouble(
                self.ui, "Simplification Tolerance", 
                "Enter tolerance value:", 
                1, 0, 100, 2)
                
            if not ok:
                return
                
            # Simplify
            simplified = gdf.copy()
            simplified.geometry = simplified.geometry.simplify(tolerance)
            
            # Add to layers
            layer_name = f"Simplified_{selected_layer}"
            self.add_vector_layer(simplified, layer_name)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to simplify polygons: {str(e)}")

    def create_convex_hull(self):
        """Create convex hull around selected polygons"""
        try:
            selected_layer = self.get_selected_vector_layer()
            if not selected_layer:
                return
                
            gdf = self.ui.layers[selected_layer]['gdf']
            
            # Check if features are selected
            if hasattr(gdf, '_selected_features') and gdf._selected_features:
                selected = gdf[gdf._selected_features]
                hull = selected.unary_union.convex_hull
            else:
                hull = gdf.unary_union.convex_hull
                
            # Create new GeoDataFrame
            new_gdf = gpd.GeoDataFrame(geometry=[hull], crs=gdf.crs)
            
            # Add to layers
            layer_name = f"ConvexHull_{selected_layer}"
            self.add_vector_layer(new_gdf, layer_name)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to create convex hull: {str(e)}")

    def create_voronoi_polygons(self):
        """Create Voronoi polygons from point layer"""
        try:
            selected_layer = self.get_selected_vector_layer()
            if not selected_layer:
                return
                
            gdf = self.ui.layers[selected_layer]['gdf']
            
            # Check if layer contains points
            if not all(gdf.geometry.type == 'Point'):
                QMessageBox.warning(self.ui, "Error", "Voronoi diagrams require point features")
                return
                
            # Get bounding box for diagram extent
            xmin, ymin, xmax, ymax = gdf.total_bounds
            boundary = gpd.GeoSeries([box(xmin, ymin, xmax, ymax)], crs=gdf.crs)
            
            # Create Voronoi polygons
            points = np.array([[geom.x, geom.y] for geom in gdf.geometry])
            vor = Voronoi(points)
            
            # Convert to polygons
            polygons = []
            for region in vor.regions:
                if not region or -1 in region:  # Skip infinite regions
                    continue
                polygon = Polygon(vor.vertices[region])
                if polygon.is_valid:
                    polygons.append(polygon)
                    
            voronoi_gdf = gpd.GeoDataFrame(geometry=polygons, crs=gdf.crs)
            
            # Clip to bounding box
            voronoi_gdf = gpd.clip(voronoi_gdf, boundary)
            
            # Add to layers
            layer_name = f"Voronoi_{selected_layer}"
            self.add_vector_layer(voronoi_gdf, layer_name)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to create Voronoi polygons: {str(e)}")

    def calculate_centroids(self):
        """Calculate centroids of polygons"""
        try:
            selected_layer = self.get_selected_vector_layer()
            if not selected_layer:
                return
                
            gdf = self.ui.layers[selected_layer]['gdf']
            
            # Create centroids
            centroids = gdf.copy()
            centroids.geometry = centroids.geometry.centroid
            
            # Add to layers
            layer_name = f"Centroids_{selected_layer}"
            self.add_vector_layer(centroids, layer_name)
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to calculate centroids: {str(e)}")

    def calculate_areas(self):
        """Calculate areas of polygon features"""
        try:
            selected_layer = self.get_selected_vector_layer()
            if not selected_layer:
                return
                
            gdf = self.ui.layers[selected_layer]['gdf']
            
            # Calculate areas in square map units
            gdf['area_sq_units'] = gdf.geometry.area
            
            # If CRS is geographic, warn about area accuracy
            if gdf.crs and gdf.crs.is_geographic:
                QMessageBox.warning(self.ui, "Warning", 
                    "Layer has geographic CRS - areas are in decimal degrees squared\n"
                    "Consider projecting to a projected coordinate system for accurate area measurements")
                    
            # Update layer
            self.ui.layers[selected_layer]['gdf'] = gdf
            QMessageBox.information(self.ui, "Success", "Area calculations added to attribute table")
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to calculate areas: {str(e)}")

    def calculate_distances(self):
        """Calculate distances between polygon centroids"""
        try:
            selected_layer = self.get_selected_vector_layer()
            if not selected_layer:
                return
                
            gdf = self.ui.layers[selected_layer]['gdf']
        
            # Create distance matrix
            centroids = gdf.geometry.centroid
            n = len(centroids)
            distance_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    distance_matrix[i,j] = centroids.iloc[i].distance(centroids.iloc[j])
                    
            # Create distance table
            distance_df = pd.DataFrame(distance_matrix, 
                                     index=gdf.index, 
                                     columns=gdf.index)
            
            # Show results in a dialog
            dialog = QDialog(self.ui)
            dialog.setWindowTitle("Distance Matrix")
            dialog.resize(600, 400)
            layout = QVBoxLayout(dialog)
            
            scroll = QScrollArea()
            table = QTableWidget()
            table.setRowCount(len(distance_df))
            table.setColumnCount(len(distance_df.columns))
            table.setHorizontalHeaderLabels([str(i) for i in distance_df.columns])
            table.setVerticalHeaderLabels([str(i) for i in distance_df.index])
            
            for i in range(len(distance_df)):
                for j in range(len(distance_df.columns)):
                    table.setItem(i, j, QTableWidgetItem(f"{distance_df.iloc[i,j]:.2f}"))
            
            scroll.setWidget(table)
            layout.addWidget(scroll)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to calculate distances: {str(e)}")

    def nearest_neighbor_analysis(self):
        """Perform nearest neighbor analysis on polygon centroids"""
        try:
            selected_layer = self.get_selected_vector_layer()
            if not selected_layer:
                return
                
            gdf = self.ui.layers[selected_layer]['gdf']
            centroids = gdf.geometry.centroid
            
            # Calculate nearest neighbor distances
            points = np.array([[geom.x, geom.y] for geom in centroids])
            tree = KDTree(points)
            distances, indices = tree.query(points, k=2)  # k=2 to get self and nearest
            
            # Calculate nearest neighbor ratio
            observed_mean = distances[:,1].mean()
            area = gdf.unary_union.convex_hull.area
            expected_mean = 0.5 / np.sqrt(len(points)/area)
            nn_ratio = observed_mean / expected_mean
            
            # Create results dialog
            dialog = QDialog(self.ui)
            dialog.setWindowTitle("Nearest Neighbor Analysis")
            layout = QVBoxLayout(dialog)
            
            text = QTextEdit()
            text.setPlainText(
                f"Nearest Neighbor Analysis Results:\n"
                f"Observed mean distance: {observed_mean:.2f}\n"
                f"Expected mean distance: {expected_mean:.2f}\n"
                f"Nearest neighbor ratio: {nn_ratio:.2f}\n\n"
                f"Interpretation:\n"
                f"Ratio < 1: Clustered pattern\n"
                f"Ratio ≈ 1: Random pattern\n"
                f"Ratio > 1: Dispersed pattern"
            )
            text.setReadOnly(True)
            
            layout.addWidget(text)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to perform nearest neighbor analysis: {str(e)}")
    
    def split_polygons(self):
        """Split selected polygons using a user-drawn line"""
        try:
            layer_name = self.get_selected_vector_layer()
            if not layer_name:
                return
                
            if not self.selected_features:
                QMessageBox.information(
                    self.ui, 
                    "No Selection",
                    "Please select polygons to split first.\n\n"
                    "1. Use the selection tool to select polygons\n"
                    "2. Click 'Split Polygons'\n"
                    "3. Draw a splitting line on the map"
                )
                return
                
            # Get selected polygons
            selected_gdf = self.get_selected_gdf(layer_name)
            if selected_gdf.empty:
                return
                
            # Prompt user to draw splitting line
            reply = QMessageBox.question(
                self.ui,
                "Draw Splitting Line",
                "Ready to draw splitting line?\n\n"
                "Left-click to add points\n"
                "Right-click to finish\n"
                "ESC to cancel",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
                
            # Activate line drawing mode
            self.ui.activate_line_drawing_mode(
                callback=lambda line: self._perform_split(layer_name, line))
                
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Split failed: {str(e)}")

    def _perform_split(self, layer_name, split_line):
        """Perform the actual split operation"""
        try:
            gdf = self.ui.layers[layer_name]['gdf']
            selected_gdf = self.get_selected_gdf(layer_name)
            new_features = []
            
            for idx, row in selected_gdf.iterrows():
                try:
                    split_result = split(row.geometry, split_line)
                    if split_result.is_empty:
                        continue
                        
                    for part in split_result.geoms:
                        new_row = row.copy()
                        new_row.geometry = part
                        new_features.append(new_row)
                except Exception as e:
                    print(f"Could not split feature {idx}: {str(e)}")
                    continue
                    
            if new_features:
                # Create new GeoDataFrame with split features
                new_gdf = gpd.GeoDataFrame(
                    new_features,
                    crs=gdf.crs
                )
                
                # Remove original features and add new ones
                updated_gdf = gdf[~gdf.index.isin(self.selected_features)]
                updated_gdf = pd.concat([updated_gdf, new_gdf], ignore_index=True)
                
                # Update layer
                self.ui.layers[layer_name]['gdf'] = updated_gdf
                self.clear_selection()
                self.ui.update_map_display()
                
                QMessageBox.information(
                    self.ui,
                    "Success",
                    f"Split {len(self.selected_features)} polygons into {len(new_features)} new features"
                )
            else:
                QMessageBox.warning(
                    self.ui,
                    "No Splits",
                    "The splitting line didn't divide any selected polygons"
                )
                
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Split operation failed: {str(e)}")
    
    def reshape_polygons(self):
        """Activate polygon reshaping mode"""
        try:
            if not hasattr(self.parent, 'show_selection_instructions'):
                raise AttributeError("Parent app is missing required selection methods")
                
            self.parent.show_selection_instructions()
            self.parent.selection_mode = 'reshape'
            self.parent.canvas.setCursor(Qt.CrossCursor)
            
            # Clear previous selection points
            if hasattr(self.parent, 'selection_points'):
                self.parent.selection_points = []
                
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to activate reshape mode: {str(e)}")
            logging.error(f"Error in reshape_polygons: {str(e)}")

    def _perform_reshape(self, new_geometry):
        """Apply the reshape to selected features"""
        layer_name = self.get_selected_vector_layer()
        if not layer_name:
            return
            
        gdf = self.ui.layers[layer_name]['gdf']
        for idx in self.selected_features:
            if idx in gdf.index:
                gdf.at[idx, 'geometry'] = new_geometry
                
        self.ui.update_map_display()
        QMessageBox.information(self.ui, "Success", "Polygons reshaped successfully")
    
    def delete_polygon_parts(self):
        """Delete holes/interior rings from selected polygons"""
        try:
            layer_name = self.get_selected_vector_layer()
            if not layer_name:
                return
                
            if not self.selected_features:
                QMessageBox.information(
                    self.ui,
                    "No Selection",
                    "Please select polygons first.\n\n"
                    "1. Use the selection tool to select polygons\n"
                    "2. Click 'Delete Parts' to remove interior rings"
                )
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            selected_gdf = self.get_selected_gdf(layer_name)
            modified_count = 0
            
            for idx, row in selected_gdf.iterrows():
                geom = row.geometry
                
                if geom.geom_type == 'Polygon':
                    # Keep only the exterior ring
                    new_geom = Polygon(geom.exterior)
                    gdf.at[idx, 'geometry'] = new_geom
                    modified_count += 1
                    
                elif geom.geom_type == 'MultiPolygon':
                    # Process each part
                    new_parts = []
                    for part in geom.geoms:
                        new_parts.append(Polygon(part.exterior))
                    gdf.at[idx, 'geometry'] = MultiPolygon(new_parts)
                    modified_count += 1
                    
            if modified_count > 0:
                self.ui.layers[layer_name]['gdf'] = gdf
                self.clear_selection()
                self.ui.update_map_display()
                
                QMessageBox.information(
                    self.ui,
                    "Success",
                    f"Removed interior rings from {modified_count} polygons"
                )
            else:
                QMessageBox.warning(
                    self.ui,
                    "No Changes",
                    "No polygon parts were deleted\n"
                    "(Selected polygons had no interior rings to remove)"
                )
                
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Delete parts failed: {str(e)}")
 
    def _show_selection_instructions(self, operation):
        """Show detailed selection instructions"""
        msg = QMessageBox(self.ui)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(f"Select Polygons to {operation.title()}")
        msg.setText(f"Please select polygons to {operation} first")
        
        text = QTextEdit()
        text.setHtml(f"""
        <h3>How to select polygons:</h3>
        <ol>
            <li>Click the <b>Rectangle Select</b> or <b>Polygon Select</b> tool</li>
            <li>Draw a selection area on the map</li>
            <li>Selected polygons will highlight in <span style='color:yellow'>yellow</span></li>
            <li>Then click the '{operation.title()}' button again</li>
        </ol>
        """)
        text.setReadOnly(True)
        
        layout = msg.layout()
        layout.addWidget(text, 1, 0, 1, layout.columnCount())
        msg.exec_()

    def add_geometry_fields(self, layer_name=None):
        """إضافة حقول المساحة والمحيط والخصائص الهندسية"""
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            
            # إضافة الحقول إذا لم تكن موجودة
            if 'area' not in gdf.columns:
                gdf['area'] = 0.0
            if 'perimeter' not in gdf.columns:
                gdf['perimeter'] = 0.0
            if 'centroid_x' not in gdf.columns:
                gdf['centroid_x'] = 0.0
            if 'centroid_y' not in gdf.columns:
                gdf['centroid_y'] = 0.0
                
            self.ui.layers[layer_name]['gdf'] = gdf
            QMessageBox.information(self.ui, "Done", "Geometric fields added successfully")
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to add fields: {str(e)}")
            
    def calculate_geometry_properties(self, layer_name=None):
        """حساب الخصائص الهندسية للمضلعات"""
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return
                
            gdf = self.ui.layers[layer_name]['gdf']
            
            # حساب المساحة (بالوحدات المربعة لنظام الإحداثيات)
            gdf['Area'] = gdf.geometry.area.round(4)
            
            # حساب المحيط
            gdf['Perimeter'] = gdf.geometry.length.round(4)
            
            # حساب مركز الثقل (Centroid)
            gdf['centroid_x'] = gdf.geometry.centroid.x.round(4)
            gdf['centroid_y'] = gdf.geometry.centroid.y.round(4)

            gdf['Min_x'] = gdf.geometry.bounds.minx.round(4)
            gdf['Max_x'] = gdf.geometry.bounds.maxx.round(4)
            gdf['Min_y'] = gdf.geometry.bounds.miny.round(4)
            gdf['Max_y'] = gdf.geometry.bounds.maxy.round(4)
            
            # إضافة المزيد من الخصائص حسب الحاجة
            # مثل: عدد الأجزاء، عدد الثقوب، etc.
            
            self.ui.layers[layer_name]['gdf'] = gdf
            QMessageBox.information(self.ui, "Success", "Geometric properties calculated successfully")
            
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Calculation failed: {str(e)}")

    # def calculate_geometry_properties(self, layer_name=None):
    #     """حساب الخصائص الهندسية للمضللات مع تقريب النتائج"""
    #     try:
    #         layer_name = layer_name or self.get_selected_vector_layer()
    #         if not layer_name:
    #             return

    #         gdf = self.layers[layer_name]['gdf']  # Changed from self.ui.layers to self.layers

    #         # حساب المساحة مع التقريب
    #         gdf['area'] = gdf.geometry.area.round(4)
            
    #         # حساب المحيط مع التقريب
    #         gdf['perimeter'] = gdf.geometry.length.round(4)
            
    #         # حساب مركز الثقل مع التقريب
    #         gdf['centroid_x'] = gdf.geometry.centroid.x.round(4)
    #         gdf['centroid_y'] = gdf.geometry.centroid.y.round(4)
            
    #         # إضافة خصائص إضافية مع التقريب
    #         # حساب نسبة التضاغط (Compactness Ratio)
    #         gdf['compactness'] = (4 * np.pi * gdf['area'] / (gdf['perimeter'] ** 2)).round(4)
            
    #         # حساب إحداثيات Bounding Box
    #         gdf['minx'] = gdf.geometry.bounds.minx.round(4)
    #         gdf['maxx'] = gdf.geometry.bounds.maxx.round(4)
    #         gdf['miny'] = gdf.geometry.bounds.miny.round(4)
    #         gdf['maxy'] = gdf.geometry.bounds.maxy.round(4)

    #         self.layers[layer_name]['gdf'] = gdf  # Changed from self.ui.layers to self.layers
    #         QMessageBox.information(self, "Success", 
    #             "تم حساب الخصائص الهندسية بنجاح مع التقريب لأقرب 4 منازل عشرية")

    #     except Exception as e:
    #         QMessageBox.critical(self.ui, "Error", f"Calculation failed: {str(e)}")


    
    def set_selected_features(self, feature_ids):
        """Update the selected features and visualize them"""
        self.selected_features = set(feature_ids)
        self._update_selection_visuals()
        
    def _update_selection_visuals(self):
        """Update the display to highlight selected features"""
        layer_name = self.get_selected_vector_layer()
        if not layer_name:
            return
            
        gdf = self.ui.layers[layer_name]['gdf']
        
        # Clear existing selection visuals
        for artist in self.ui.ax.collections:
            artist.remove()
            
        # Plot selected features with highlight
        if self.selected_features:
            selected = gdf[gdf.index.isin(self.selected_features)]
            if not selected.empty:
                selected.plot(
                    ax=self.ui.ax,
                    color='yellow',
                    edgecolor='red',
                    alpha=0.5,
                    linewidth=2
                )
        
        self.ui.canvas.draw()
        self.ui.statusBar().showMessage(f"{len(self.selected_features)} features selected")

    def clear_selection(self):
        """Clear current selection"""
        self.selected_features.clear()
        self._update_selection_visuals()
        self.ui.statusBar().showMessage("Selection cleared")

    def get_selected_gdf(self, layer_name=None):
        """Get GeoDataFrame with only selected features"""
        layer_name = layer_name or self.get_selected_vector_layer()
        if not layer_name:
            return None
            
        gdf = self.ui.layers[layer_name]['gdf']
        return gdf[gdf.index.isin(self.selected_features)]
    
    def update_map_display(self):
        """Show selected features with highlights"""
        try:
            # Clear and redraw base layers
            self.ax.clear()
            
            # Draw unselected features
            layer_name = self.polygon_processor.get_selected_vector_layer()
            if layer_name and layer_name in self.layers:
                gdf = self.layers[layer_name]['gdf']
                
                if hasattr(self.polygon_processor, 'selected_features'):
                    unselected = gdf[~gdf.index.isin(self.polygon_processor.selected_features)]
                    unselected.plot(ax=self.ax, color='blue', alpha=0.5)
                    
                    # Draw selected features highlighted
                    selected = gdf[gdf.index.isin(self.polygon_processor.selected_features)]
                    if not selected.empty:
                        selected.plot(ax=self.ax, color='yellow', edgecolor='red', linewidth=2)
            
            # Draw temporary selection shapes
            if hasattr(self, 'temp_rect'):
                rect = self.temp_rect
                self.ax.plot(
                    [rect.left(), rect.right(), rect.right(), rect.left(), rect.left()],
                    [rect.top(), rect.top(), rect.bottom(), rect.bottom(), rect.top()],
                    'g--'
                )
                
            if hasattr(self, 'selection_points') and self.selection_points:
                xs, ys = zip(*self.selection_points)
                self.ax.plot(xs, ys, 'go-')
                
            self.canvas.draw()
            
        except Exception as e:
            print(f"Display error: {str(e)}")

    def mousePressEvent(self, event):
        """Handle mouse events for selection"""
        if self.selection_mode == 'rectangle' and event.button() == Qt.LeftButton:
            self.selection_start = self.canvas.mapToScene(event.pos())
            self.temp_shape = None
        
        elif self.selection_mode == 'polygon' and event.button() == Qt.LeftButton:
            if not hasattr(self, 'selection_points'):
                self.selection_points = []
            point = self.canvas.mapToScene(event.pos())
            self.selection_points.append((point.x(), point.y()))
            
            if len(self.selection_points) > 1:
                self.temp_shape = LineString(self.selection_points)
            self.update_map_display()
        
        elif event.button() == Qt.RightButton and self.selection_mode:
            if self.selection_mode == 'rectangle':
                self.finish_rectangle_selection()
            elif self.selection_mode == 'polygon':
                self.finish_polygon_selection()

    def finish_rectangle_selection(self):
        """Finalize rectangle selection"""
        end_point = self.canvas.mapToScene(self.mapFromGlobal(QCursor.pos()))
        rect = QRectF(self.selection_start, end_point).normalized()
        
        # Find features within rectangle
        layer_name = self.polygon_processor.get_selected_vector_layer()
        if layer_name:
            gdf = self.layers[layer_name]['gdf']
            selected = gdf[gdf.geometry.intersects(self.map_to_geo(rect))]
            self.polygon_processor.set_selected_features(selected.index.tolist())
        
        self.selection_mode = None
        self.update_map_display()

    def finish_polygon_selection(self):
        """Finalize polygon selection"""
        if hasattr(self, 'selection_points') and len(self.selection_points) >= 3:
            poly = Polygon(self.selection_points)
            
            # Find features within polygon
            layer_name = self.polygon_processor.get_selected_vector_layer()
            if layer_name:
                gdf = self.layers[layer_name]['gdf']
                selected = gdf[gdf.geometry.intersects(poly)]
                self.polygon_processor.set_selected_features(selected.index.tolist())
        
        self.selection_mode = None
        if hasattr(self, 'selection_points'):
            del self.selection_points
        self.update_map_display()

    def create_graph(self, layer_name=None):
        """Create an advanced graph without save/close buttons"""
        try:
            layer_name = layer_name or self.get_selected_vector_layer()
            if not layer_name:
                return

            # Initialize containers and trackers
            self.y_combos = []
            self.y_color_combos = []
            
            # Get layer data
            self.current_layer_name = layer_name
            self.current_gdf = self.ui.layers[layer_name]['gdf'].reset_index()  # Reset index to make FID a column
            
            # Get all fields (including FID) except geometry
            all_fields = [col for col in self.current_gdf.columns if col != 'geometry']
            numeric_fields = [col for col in all_fields if pd.api.types.is_numeric_dtype(self.current_gdf[col])]
            # # Example:
            # self.x_combo.addItems(all_fields)        # X can be any field
            # self.y_combo.addItems(numeric_fields)    # Y must be numeric
            if not all_fields:
                QMessageBox.warning(self.ui, "No Data", "No attribute fields found to plot")
                return

            # Create dialog
            dialog = QDialog(self.ui)
            dialog.setWindowTitle(f"Graph - {layer_name}")
            dialog.resize(1100, 900)
            main_layout = QVBoxLayout(dialog)

            # ================= TOP CONTROLS SECTION =================
            controls_layout = QVBoxLayout()
            
            # Field selection row
            field_layout = QHBoxLayout()
            
            # X-axis selection
            x_layout = QVBoxLayout()
            x_layout.addWidget(QLabel("X-axis Field:"))
            self.x_combo = QComboBox()
            self.x_combo.addItems(all_fields)
            x_layout.addWidget(self.x_combo)
            field_layout.addLayout(x_layout)
            field_layout.addSpacing(20)

            # Y-axis container
            y_container = QGroupBox("Y-axis Fields")
            self.extra_y_container = QVBoxLayout()
            y_container.setLayout(self.extra_y_container)
            
            # Add initial Y field
            self._add_y_field(all_fields, dialog, initial=True)
            
            # Add Y field button
            add_y_btn = QPushButton("➕ Add Y Field")
            add_y_btn.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    padding: 5px;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                }
            """)
            add_y_btn.clicked.connect(lambda: self._add_y_field(all_fields, dialog))
            
            # Y-axis layout organization
            y_layout = QVBoxLayout()
            y_layout.addWidget(y_container)
            y_layout.addWidget(add_y_btn)
            field_layout.addLayout(y_layout)
            controls_layout.addLayout(field_layout)

            # ================= CONFIGURATION ROW =================
            config_row = QHBoxLayout()
            
            # Graph type
            type_box = QGroupBox("Graph Type")
            type_layout = QVBoxLayout()
            self.type_combo = QComboBox()
            graph_types = ["Scatter", "Line", "Bar", "Histogram"]
            if numeric_fields:
                graph_types.extend(["Box Plot", "Area"])
            self.type_combo.addItems(graph_types)
            type_layout.addWidget(self.type_combo)
            type_box.setLayout(type_layout)
            config_row.addWidget(type_box)

            # X-axis formatting
            x_format_box = QGroupBox("X-axis Formatting")
            x_format_layout = QVBoxLayout()
            self.rotate_x_labels = QComboBox()
            self.rotate_x_labels.addItems(["0° (Horizontal)", "45° Diagonal", "90° Vertical"])
            x_format_layout.addWidget(self.rotate_x_labels)
            self.wrap_labels = QCheckBox("Wrap long labels")
            x_format_layout.addWidget(self.wrap_labels)
            x_format_box.setLayout(x_format_layout)
            config_row.addWidget(x_format_box)

            # Legend options
            legend_box = QGroupBox("Legend Options")
            legend_layout = QVBoxLayout()
            self.show_legend = QCheckBox("Show legend")
            self.show_legend.setChecked(True)
            legend_layout.addWidget(self.show_legend)
            self.legend_position = QComboBox()
            self.legend_position.addItems(["Best", "Upper Right", "Upper Left", 
                                        "Lower Left", "Lower Right", "Center"])
            legend_layout.addWidget(self.legend_position)
            legend_box.setLayout(legend_layout)
            config_row.addWidget(legend_box)

            controls_layout.addLayout(config_row)

            # ================= UPDATE BUTTON ONLY =================
            button_row = QHBoxLayout()
            button_row.addStretch()
            
            self.plot_btn = QPushButton("🔄 Update Graph")
            self.plot_btn.setFixedHeight(30)
            self.plot_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.plot_btn.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    padding: 5px;
                    min-width: 120px;
                    background-color: #e6f3ff;
                }
            """)
            
            button_row.addWidget(self.plot_btn)
            controls_layout.addLayout(button_row)
            main_layout.addLayout(controls_layout)

            # ================= GRAPH DISPLAY SECTION =================
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            main_layout.addWidget(separator)

            # Graph display with toolbar
            graph_container = QVBoxLayout()
            self.fig = Figure(figsize=(12, 8))
            self.canvas = FigureCanvas(self.fig)
            
            # Add navigation toolbar
            self.toolbar = NavigationToolbar(self.canvas, dialog)
            graph_container.addWidget(self.toolbar)
            graph_container.addWidget(self.canvas)
            
            main_layout.addLayout(graph_container)

            # ================= SIGNAL CONNECTIONS =================
            controls = [
                self.x_combo, self.type_combo, self.rotate_x_labels,
                self.wrap_labels, self.show_legend, self.legend_position,
                *self.y_combos, *self.y_color_combos
            ]
            
            for control in controls:
                if isinstance(control, QComboBox):
                    control.currentTextChanged.connect(lambda: self._plot_graph(dialog))
                else:
                    control.stateChanged.connect(lambda: self._plot_graph(dialog))
            
            self.plot_btn.clicked.connect(lambda: self._plot_graph(dialog))
            
            # Initial plot
            self._plot_graph(dialog)
            dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to create graph: {str(e)}")

 

    def _add_y_field(self, fields, dialog, initial=False):
        """Add Y-axis field row with proper variable capturing"""
        y_row = QHBoxLayout()
        
        # Field selection
        y_combo = QComboBox()
        y_combo.addItems(fields)
        
        # Color selection
        color_combo = QComboBox()
        color_combo.addItems(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"])
        
        # Remove button
        remove_btn = QPushButton("❌")
        remove_btn.setFixedSize(25, 25)
        
        # Add to layout
        y_row.addWidget(QLabel("Y Field:"))
        y_row.addWidget(y_combo)
        y_row.addWidget(QLabel("Color:"))
        y_row.addWidget(color_combo)
        if not initial:
            y_row.addWidget(remove_btn)
        
        # Add to container
        self.extra_y_container.addLayout(y_row)
        
        # Track components
        self.y_combos.append(y_combo)
        self.y_color_combos.append(color_combo)
        
        # Connect signals with proper variable capture
        y_combo.currentTextChanged.connect(
            lambda _, yc=y_combo, cc=color_combo: self._handle_y_field_change(yc, cc, dialog)
        )
        
        if not initial:
            remove_btn.clicked.connect(
                lambda _, yr=y_row, yc=y_combo, cc=color_combo: 
                self._remove_y_field(yr, yc, cc, dialog)
            )

    def _handle_y_field_change(self, y_combo, color_combo, dialog):
        """Handle Y field changes with captured parameters"""
        try:
            y_field = y_combo.currentText()
            # Rest of your field change logic
            self._plot_graph(dialog)
        except Exception as e:
            print(f"Error handling Y field change: {str(e)}")

    def _remove_y_field(self, y_row, y_combo, color_combo, dialog):
        """Remove Y field with captured parameters"""
        try:
            # Clear layout widgets
            while y_row.count():
                item = y_row.takeAt(0)
                if widget := item.widget():
                    widget.deleteLater()
            
            # Remove from tracking lists
            if y_combo in self.y_combos:
                self.y_combos.remove(y_combo)
            if color_combo in self.y_color_combos:
                self.y_color_combos.remove(color_combo)
            
            self._plot_graph(dialog)
        except Exception as e:
            print(f"Error removing Y field: {str(e)}")

    

    def _plot_graph(self, dialog):
        """Handle the actual graph plotting"""
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            # Get selected values
            x_field = self.x_combo.currentText()
            graph_type = self.type_combo.currentText()
            x_data = self.current_gdf[x_field]
            
            # Handle X-axis formatting
            rotation = 0
            ha = 'center'
            if "45" in self.rotate_x_labels.currentText():
                rotation = 45
                ha = 'right'
            elif "90" in self.rotate_x_labels.currentText():
                rotation = 90
                ha = 'right'
            
            # Handle different plot types
            if graph_type == "Histogram":
                self._plot_histogram(ax, x_data)
            else:
                self._plot_xy_graph(ax, x_field, x_data, graph_type, rotation, ha)
            
            # Final formatting
            ax.set_title(f"{graph_type} Plot: {x_field}", pad=20)
            if self.show_legend.isChecked() and len(self.y_combos) > 0 and graph_type != "Histogram":
                ax.legend(loc=self.legend_position.currentText().lower().replace(" ", ""))
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
            self.canvas.draw()

    def _plot_histogram(self, ax, x_data):
        """Handle histogram plotting"""
        if pd.api.types.is_numeric_dtype(x_data):
            ax.hist(x_data.dropna(), bins=20, color=self.y_color_combos[0].currentText() if self.y_color_combos else '#1f77b4')
        else:
            ax.hist(x_data.astype(str), bins=min(20, len(x_data.unique())), 
                color=self.y_color_combos[0].currentText() if self.y_color_combos else '#1f77b4')
        ax.set_xlabel(self.x_combo.currentText())
        ax.set_ylabel("Frequency")



    def _plot_xy_graph(self, ax, x_field, x_data, graph_type, rotation, ha):
        """Handle XY plot types with enhanced validation and error recovery"""
        ax.clear()
        error_messages = []
        valid_plots = False
        plotted_fields = []

        try:
            # Debug: Show data types
            #print(f"[Debug] Data types:\n{self.current_gdf.dtypes}")
            
            # Convert datetime X-data
            if pd.api.types.is_datetime64_any_dtype(x_data):
                x_data = x_data.astype('datetime64[ns]')
                #print(f"[Debug] Converted X-axis '{x_field}' to datetime")

            # Plot each Y field with validation
            for idx, (y_combo, color_combo) in enumerate(zip(self.y_combos, self.y_color_combos)):
                y_field = None
                try:
                    # Safely get field references
                    y_field = y_combo.currentText() if y_combo else f"unknown_field_{idx}"
                    color = color_combo.currentText() if color_combo else "#FF0000"
                    
                    # Get raw data and attempt conversion
                    y_raw = self.current_gdf[y_field]
                    #print(f"[Debug] Plotting '{y_field}' (dtype: {y_raw.dtype})")
                    
                    # Enhanced numeric conversion with cleaning
                    try:
                        y_data = pd.to_numeric(
                            y_raw.astype(str).str.replace(r'[^\d.]', '', regex=True),
                            errors='coerce'
                        )
                        if y_data.isnull().all():
                            raise ValueError("All values converted to NaN")
                    except Exception as e:
                        error_messages.append(
                            f"Failed to convert '{y_field}' to numeric\n"
                            f"First 3 values: {y_raw.head(3).tolist()}\n"
                            f"Error: {str(e)}"
                        )
                        continue

                    # Handle different plot types
                    if graph_type == "Scatter":
                        ax.scatter(x_data, y_data, color=color, label=y_field,
                                alpha=0.7, s=50, edgecolors='w')
                        valid_plots = True
                        
                    elif graph_type == "Line":
                        ax.plot(x_data, y_data, color=color, label=y_field,
                            linewidth=2, marker='o', markersize=5)
                        valid_plots = True
                        
                    elif graph_type == "Bar":
                        width = 0.8 / len(self.y_combos)
                        offset = width * idx - (width * len(self.y_combos) / 2) + width/2
                        
                        if pd.api.types.is_numeric_dtype(x_data):
                            ax.bar(x_data + offset, y_data, width=width,
                                color=color, label=y_field, edgecolor='black')
                        else:
                            pos = np.arange(len(x_data))
                            ax.bar(pos + offset, y_data, width=width,
                                color=color, label=y_field, edgecolor='black')
                            ax.set_xticks(pos)
                            ax.set_xticklabels(x_data, rotation=rotation, ha=ha)
                        valid_plots = True
                        
                    elif graph_type == "Area":
                        if idx == 0:
                            ax.fill_between(x_data, y_data, color=color,
                                        label=y_field, alpha=0.4, edgecolor=color)
                        else:
                            prev_y = pd.to_numeric(
                                self.current_gdf[self.y_combos[idx-1].currentText()],
                                errors='coerce'
                            )
                            ax.fill_between(x_data, prev_y, y_data, color=color,
                                        label=y_field, alpha=0.4, edgecolor=color)
                        valid_plots = True
                        
                    elif graph_type == "Box Plot":
                        if idx == 0:  # Plot only once
                            box_data = []
                            labels = []
                            for val in x_data.unique():
                                subset = y_data[x_data == val]
                                if not subset.empty:
                                    box_data.append(subset.dropna())
                                    labels.append(str(val))
                            
                            if box_data:
                                ax.boxplot(box_data, labels=labels, patch_artist=True,
                                        boxprops=dict(facecolor=color))
                                ax.set_xticklabels(labels, rotation=rotation, ha=ha)
                                valid_plots = True

                    plotted_fields.append(y_field)

                except Exception as e:
                    error_msg = f"Error plotting '{y_field if y_field else 'unknown'}': {str(e)}"
                    error_messages.append(error_msg)
                    print(f"[Error] {error_msg}")
                    traceback.print_exc()

            # Error handling and feedback
            if not valid_plots:
                error_text = "No valid plots created:\n" + "\n".join(error_messages)
                ax.text(0.5, 0.5, error_text, ha='center', va='center', 
                    color='red', fontsize=10, wrap=True,
                    bbox=dict(facecolor='white', edgecolor='red', pad=10))
                ax.set_xticks([])
                ax.set_yticks([])
            elif error_messages:
                error_text = "Partial errors:\n" + "\n".join(error_messages)
                ax.text(0.05, 0.95, error_text, ha='left', va='top', 
                    transform=ax.transAxes, color='orange', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='orange', pad=5))

            # Final formatting for valid plots
            if valid_plots:
                ax.set_xlabel(x_field, fontweight='bold', fontsize=10)
                y_label = "Value" if len(plotted_fields) > 1 else plotted_fields[0]
                ax.set_ylabel(y_label, fontweight='bold', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_title(f"{graph_type} Plot: {x_field}", 
                            fontweight='bold', pad=20, fontsize=12)
                
                if self.show_legend.isChecked() and graph_type != "Box Plot":
                    ax.legend(loc=self.legend_position.currentText().lower().replace(" ", ""),
                            framealpha=0.7, fontsize=8)

                # Handle categorical x-axis
                if not pd.api.types.is_numeric_dtype(x_data) and graph_type != "Box Plot":
                    ax.set_xticks(np.arange(len(x_data)))
                    ax.set_xticklabels(x_data, rotation=rotation, ha=ha, fontsize=8)

        except Exception as main_error:
            error_msg = f"Critical plotting error: {str(main_error)}"
            print(f"[Critical] {error_msg}")
            traceback.print_exc()
            ax.text(0.5, 0.5, "Critical plotting error", ha='center', va='center',
                color='red', fontsize=12, 
                bbox=dict(facecolor='white', edgecolor='red', pad=10))
            
        finally:
            ax.figure.canvas.draw_idle()

    def _save_graph(self, dialog):
        """Handle graph saving with high quality output"""
        path, _ = QFileDialog.getSaveFileName(
            dialog, "Save Graph", 
            f"{self.current_layer_name}_graph", 
            "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tiff)")
        
        if path:
            try:
                self.fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                QMessageBox.information(dialog, "Success", f"Graph saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(dialog, "Error", f"Failed to save graph:\n{str(e)}")

    def toggle_edit_buttons(self, enable):
            self.actionSave_Edits.setEnabled(enable)
            self.actionStop_Editing.setEnabled(enable)
            print(f"تم تغيير حالة الأزرار إلى: {enable}")  # للتتبع

    # Add to PolygonProcessor class
    def add_feature(self, geometry):
        """Add new feature to edit buffer"""
        # Get next FID
        next_fid = self.edit_buffer['FID'].max() + 1 if not self.edit_buffer.empty else 0
        
        new_row = {'FID': next_fid, 'geometry': geometry}
        if not self.editing_active:
            return False
            
        new_row = {'geometry': geometry}
        # Add attributes from existing columns
        for col in self.edit_buffer.columns:
            if col != 'geometry':
                new_row[col] = None  # Or appropriate default
                
        self.edit_buffer = self.edit_buffer.append(new_row, ignore_index=True)
        return True

    def delete_feature(self, index):
        """Delete feature by index"""
        if self.editing_active and 0 <= index < len(self.edit_buffer):
            self.edit_buffer = self.edit_buffer.drop(index).reset_index(drop=True)
            return True
        return False