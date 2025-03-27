#!/usr/bin/env python3
"""Wrapper para ejecutar la API de predicción de precios."""

import uvicorn

from src.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
