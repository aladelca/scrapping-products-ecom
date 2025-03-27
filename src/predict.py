#!/usr/bin/env python3
"""Script for making price predictions using the trained model."""

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd

from src.models.price_predictor import CatBoostPricePredictor


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make price predictions using the trained model."
    )

    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--data-path",
        type=str,
        help="Path to a CSV file containing the data to predict",
    )
    input_group.add_argument(
        "--text", type=str, help="Product description text for prediction"
    )

    # Model options
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-path",
        type=str,
        default="models/price_model.pkl",
        help="Path to the trained model file",
    )
    model_group.add_argument(
        "--vectorizer-path",
        type=str,
        default="models/vectorizer.pkl",
        help="Path to the trained vectorizer file",
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-path", type=str, help="Path to save prediction results (CSV format)"
    )

    return parser.parse_args()


def load_model(model_path, vectorizer_path):
    """
    Load the trained model and vectorizer.

    Args:
        model_path (str): Path to the trained model file
        vectorizer_path (str): Path to the trained vectorizer file

    Returns:
        CatBoostPricePredictor: Loaded model
    """
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    # Load the complete model object
    model = joblib.load(model_path)

    # Ensure it's a CatBoostPricePredictor instance
    if not isinstance(model, CatBoostPricePredictor):
        raise TypeError(f"Expected CatBoostPricePredictor, got {type(model).__name__}")

    # For backward compatibility with older models
    if not hasattr(model, "is_trained") or not model.is_trained:
        model.is_trained = True

    # Load vectorizer for backward compatibility
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        model.set_vectorizer(vectorizer)

    return model


def predict_single_text(model, text):
    """
    Make a prediction for a single text description.

    Args:
        model (CatBoostPricePredictor): Trained model
        text (str): Product description text

    Returns:
        float: Predicted price
    """
    # Create a dataframe with the text
    data = pd.DataFrame({"description": [text]})

    # Make prediction

    prediction = model.predict(data, step="inference")

    return prediction[0]


def predict_from_csv(model, data_path, output_path=None):
    """
    Make predictions for all rows in a CSV file.

    Args:
        model (CatBoostPricePredictor): Trained model
        data_path (str): Path to the CSV file
        output_path (str, optional): Path to save prediction results

    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Load data
    data = pd.read_csv(data_path)

    # Check if required column exists
    if "description_processed" not in data.columns:
        raise ValueError("CSV file must contain a 'description_processed' column")

    # Make predictions
    predictions = model.predict(data, step="inference")

    # Add predictions to the dataframe
    result = data.copy()
    result["predicted_price"] = predictions

    # Save results if output path is provided
    if output_path:
        result.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

    return result


def main():
    """Execute the main prediction workflow."""
    # Set up logging
    setup_logging()

    # Parse command line arguments
    args = parse_args()

    # Load model
    logging.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.vectorizer_path)
    # Make predictions
    if args.text:
        # Predict for a single text
        logging.info("Making prediction for text input")
        prediction = predict_single_text(model, args.text)
        logging.info(f"Predicted price: {prediction:.2f}")

    elif args.data_path:
        # Predict for all rows in the CSV file
        logging.info(f"Making predictions for data in {args.data_path}")
        results = predict_from_csv(model, args.data_path, args.output_path)

        # Display summary of predictions
        logging.info(
            f"Average predicted price: {results['predicted_price'].mean():.2f}"
        )
        logging.info(f"Min predicted price: {results['predicted_price'].min():.2f}")
        logging.info(f"Max predicted price: {results['predicted_price'].max():.2f}")

    else:
        logging.error("Either --text or --data-path must be specified")


if __name__ == "__main__":
    main()
