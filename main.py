#!/usr/bin/env python3
"""
Space GeoVision - Main Application Entry Point
"""
import logging
import sys
import os
import time

# Configure matplotlib logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer  # <-- أضف QTimer هنا

from ui import SatelliteImageApp

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # 1. التحقق من وجود الملف
    image_path = r"D:\Python_Codes\icons\Pr_icon.jpeg"
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        sys.exit(1)
    
    # 2. تحميل الصورة بطريقة صحيحة
    try:
        image = QImage(image_path)
        if image.isNull():
            raise ValueError("Failed to load image")
            
        pixmap = QPixmap.fromImage(image)
        
        if pixmap.width() == 0 or pixmap.height() == 0:
            raise ValueError("Invalid image dimensions")
            
    except Exception as e:
        print(f"Image loading error: {str(e)}")
        sys.exit(1)
    
    # 3. تغيير طريقة التحجيم
    splash_pix = pixmap.scaled(
        800, 
        1000, 
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    
    # 4. إعدادات الشاشة
    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.setWindowFlags(
        Qt.WindowType.FramelessWindowHint | 
        Qt.WindowType.WindowStaysOnTopHint
    )
    splash.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    
    splash.show()
    app.processEvents()
    
    # 5. استخدام QTimer بشكل صحيح
    timer = QTimer()
    timer.singleShot(9000, lambda: None)  # 2000 ميلي ثانية = 2 ثانية
    
    window = SatelliteImageApp()
    splash.finish(window)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()