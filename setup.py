#!/usr/bin/env python3
"""
Space GeoVision - Setup Script
Author: Ismail Elkhrachy
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# قائمة الحزم المطلوبة للتشغيل (تم استبعاد snappy لمنع مشاكل التثبيت)
requirements = [
    "PyQt5>=5.15.0",
    "PyQtWebEngine>=5.15.0",
    "rasterio>=1.3.0",
    "geopandas>=0.14.0",
    "scikit-learn>=1.2.0",
    "scikit-image>=0.21.0",
    "matplotlib>=3.7.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "opencv-python>=4.8.0",
    "torch>=2.0.0",
    "folium>=0.14.0",
    "leafmap>=0.25.0",
    "whitebox>=2.3.0",
    "shapely>=2.0.0",
    "pyproj>=3.5.0",
    "tqdm>=4.65.0",
    "joblib>=1.2.0",
    "openpyxl>=3.1.0",
    "ezdxf>=0.18.0",
]

setup(
    name="space-geovision",
    version="1.0.0",
    author="Ismail Elkhrachy",
    author_email="iaelkhrachy@nu.edu.sa",
    description="Integrated open-source platform for multi-modal remote sensing analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ismaelelkhrachy/space-geovision",
    project_urls={
        "Bug Tracker": "https://github.com/ismaelelkhrachy/space-geovision/issues",
        "Documentation": "https://space-geovision.readthedocs.io",
        "Source Code": "https://github.com/ismaelelkhrachy/space-geovision",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "space-geovision=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)