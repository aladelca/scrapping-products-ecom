#!/usr/bin/env python3
"""Setup script for the e-commerce scraper and price predictor package."""

from setuptools import find_packages, setup

setup(
    name="ecommerce-predictor",
    version="0.1.0",
    packages=find_packages(),
    package_data={"src": ["*.py"]},
    include_package_data=True,
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.4.2",
        "python-multipart>=0.0.6",
        "catboost>=1.2.0",
        "pandas>=2.0.0",
        "joblib>=1.3.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "requests>=2.28.0",
        "selenium>=4.0.0",
    ],
    python_requires=">=3.8",
)
