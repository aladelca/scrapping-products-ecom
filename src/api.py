#!/usr/bin/env python3
"""REST API for price prediction using FastAPI."""

import logging
import os
from typing import Dict, List, Optional, Union

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.models.price_predictor import CatBoostPricePredictor
from src.predict import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Price Prediction API",
    description="API for predicting product prices from descriptions",
    version="1.0.0",
)

# Default paths for model and vectorizer
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "trained_models/price_model.pkl")
DEFAULT_VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "trained_models/vectorizer.pkl")

# Load the model at startup
model = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    try:
        logger.info(f"Loading model from {DEFAULT_MODEL_PATH}")
        model = load_model(DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # We'll initialize later when needed
        model = None


# Data models for requests and responses
class TextPredictionRequest(BaseModel):
    """Request model for text-based prediction."""

    text: str = Field(..., description="Product description text")


class FilePredictionRequest(BaseModel):
    """Request model for file-based prediction."""

    file_path: str = Field(
        ..., description="Path to the file containing product descriptions"
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    product_description: str = Field(..., description="Product description")
    predicted_price: float = Field(..., description="Predicted price")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    summary: Dict[str, float] = Field(..., description="Summary statistics")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Price Prediction API is running"}


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_from_text(request: TextPredictionRequest):
    """
    Predict price from text description.

    Args:
        request: Text prediction request with product description

    Returns:
        Prediction response with predicted price
    """
    global model

    if model is None:
        try:
            model = load_model(DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    try:
        logger.info(f"Making prediction for text: {request.text}")

        # Create DataFrame with the text
        data = pd.DataFrame({"description": [request.text]})

        # Make prediction
        prediction = model.predict(data, step="inference")[0]

        logger.info(f"Prediction: {prediction:.2f}")

        return PredictionResponse(
            product_description=request.text, predicted_price=float(prediction)
        )

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to make prediction: {str(e)}"
        )


@app.post("/predict/file", response_model=BatchPredictionResponse)
async def predict_from_file(request: FilePredictionRequest):
    """
    Predict prices from a file containing product descriptions.

    Args:
        request: File prediction request with path to the file

    Returns:
        Batch prediction response with predictions and summary statistics
    """
    global model

    if model is None:
        try:
            model = load_model(DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    try:
        file_path = request.file_path

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        logger.info(f"Making predictions for file: {file_path}")

        # Load data based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".csv":
            data = pd.read_csv(file_path)
        elif file_ext == ".parquet":
            data = pd.read_parquet(file_path)
        elif file_ext == ".json":
            data = pd.read_json(file_path)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported formats: .csv, .parquet, .json",
            )

        # Ensure 'description' column exists
        if (
            "description" not in data.columns
            and "description_processed" not in data.columns
        ):
            raise HTTPException(
                status_code=400,
                detail="File must contain a 'description' or 'description_processed' column",
            )

        # Make predictions
        if "description" in data.columns:
            predictions = model.predict(data, step="inference")
            description_col = "description"
        else:
            predictions = model.predict(data, step="evaluate")
            description_col = "description_processed"

        # Create response
        prediction_items = []
        for i, pred in enumerate(predictions):
            prediction_items.append(
                PredictionResponse(
                    product_description=data[description_col].iloc[i],
                    predicted_price=float(pred),
                )
            )

        # Summary statistics
        summary = {
            "mean": float(predictions.mean()),
            "min": float(predictions.min()),
            "max": float(predictions.max()),
            "count": len(predictions),
        }

        logger.info(f"Made {len(predictions)} predictions. Mean: {summary['mean']:.2f}")

        return BatchPredictionResponse(predictions=prediction_items, summary=summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to make predictions: {str(e)}"
        )


@app.post("/predict/upload", response_model=BatchPredictionResponse)
async def predict_from_upload(file: UploadFile = File(...)):
    """
    Predict prices from an uploaded file containing product descriptions.

    Args:
        file: Uploaded file with product descriptions

    Returns:
        Batch prediction response with predictions and summary statistics
    """
    global model

    if model is None:
        try:
            model = load_model(DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        logger.info(f"File uploaded and saved as {temp_file_path}")

        # Process the file
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext == ".csv":
            data = pd.read_csv(temp_file_path)
        elif file_ext == ".parquet":
            data = pd.read_parquet(temp_file_path)
        elif file_ext == ".json":
            data = pd.read_json(temp_file_path)
        else:
            os.remove(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported formats: .csv, .parquet, .json",
            )

        # Remove the temporary file
        os.remove(temp_file_path)

        # Ensure 'description' column exists
        if (
            "description" not in data.columns
            and "description_processed" not in data.columns
        ):
            raise HTTPException(
                status_code=400,
                detail="File must contain a 'description' or 'description_processed' column",
            )

        # Make predictions
        if "description" in data.columns:
            predictions = model.predict(data, step="inference")
            description_col = "description"
        else:
            predictions = model.predict(data, step="evaluate")
            description_col = "description_processed"

        # Create response
        prediction_items = []
        for i, pred in enumerate(predictions):
            prediction_items.append(
                PredictionResponse(
                    product_description=data[description_col].iloc[i],
                    predicted_price=float(pred),
                )
            )

        # Summary statistics
        summary = {
            "mean": float(predictions.mean()),
            "min": float(predictions.min()),
            "max": float(predictions.max()),
            "count": len(predictions),
        }

        logger.info(f"Made {len(predictions)} predictions. Mean: {summary['mean']:.2f}")

        return BatchPredictionResponse(predictions=prediction_items, summary=summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to make predictions: {str(e)}"
        )


@app.post(
    "/predict/json", response_model=Union[PredictionResponse, BatchPredictionResponse]
)
async def predict_from_json(
    request: Dict = Body(...),
):
    """
    Predict prices from JSON data.

    The JSON can have two formats:
    1. Single prediction: {"text": "product description"}
    2. Batch prediction: {"file_path": "path/to/file.csv"} or {"texts": ["description1", "description2"]}

    Args:
        request: JSON request with text, texts array, or file_path

    Returns:
        Prediction response or batch prediction response
    """
    global model

    if model is None:
        try:
            model = load_model(DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    try:
        # Case 1: Single text prediction
        if "text" in request:
            text = request["text"]
            logger.info(f"Making prediction for text: {text}")

            # Create DataFrame with the text
            data = pd.DataFrame({"description": [text]})

            # Make prediction
            prediction = model.predict(data, step="inference")[0]

            logger.info(f"Prediction: {prediction:.2f}")

            return PredictionResponse(
                product_description=text, predicted_price=float(prediction)
            )

        # Case 2: File path prediction
        elif "file_path" in request:
            file_prediction_request = FilePredictionRequest(
                file_path=request["file_path"]
            )
            return await predict_from_file(file_prediction_request)

        # Case 3: Multiple texts prediction
        elif "texts" in request:
            texts = request["texts"]
            logger.info(f"Making batch predictions for {len(texts)} texts")

            # Create DataFrame with the texts
            data = pd.DataFrame({"description": texts})

            # Make predictions
            predictions = model.predict(data, step="inference")

            # Create response
            prediction_items = []
            for i, pred in enumerate(predictions):
                prediction_items.append(
                    PredictionResponse(
                        product_description=texts[i], predicted_price=float(pred)
                    )
                )

            # Summary statistics
            summary = {
                "mean": float(predictions.mean()),
                "min": float(predictions.min()),
                "max": float(predictions.max()),
                "count": len(predictions),
            }

            logger.info(
                f"Made {len(predictions)} predictions. Mean: {summary['mean']:.2f}"
            )

            return BatchPredictionResponse(
                predictions=prediction_items, summary=summary
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request format. Must include 'text', 'texts', or 'file_path'.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to make predictions: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
