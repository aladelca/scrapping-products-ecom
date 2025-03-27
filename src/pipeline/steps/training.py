"""Training step module."""

import logging
import os
import time
from typing import Any, Dict, Optional

import pandas as pd

from src.models.price_predictor import CatBoostPricePredictor
from src.pipeline.base import PipelineStep


class TrainingStep(PipelineStep):
    """Training step for model fitting and evaluation."""

    def __init__(
        self,
        model_path: str = "trained_models",
        model_filename: str = "price_model.pkl",
        vectorizer_filename: str = "vectorizer.pkl",
        target_column: str = "offer_price",
        text_column: str = "description_processed",
        learning_rate: float = 0.1,
        iterations: int = 1000,
        depth: int = 6,
    ):
        """
        Initialize the training step.

        Args:
            model_path (str): Directory path to save the model
            model_filename (str): Filename for the saved model
            vectorizer_filename (str): Filename for the saved vectorizer
            target_column (str): Column name for the target variable
            text_column (str): Column name for the processed text data
            learning_rate (float): Learning rate for CatBoost
            iterations (int): Number of iterations for CatBoost
            depth (int): Tree depth for CatBoost
        """
        self.model_path = model_path
        self.model_filename = model_filename
        self.vectorizer_filename = vectorizer_filename
        self.target_column = target_column
        self.text_column = text_column
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.depth = depth

        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

        # Initialize the model
        self.model = CatBoostPricePredictor(
            learning_rate=learning_rate, iterations=iterations, depth=depth
        )

        logging.info(f"Initialized TrainingStep with parameters:")
        logging.info(f"  - model_path: {model_path}")
        logging.info(f"  - model_filename: {model_filename}")
        logging.info(f"  - vectorizer_filename: {vectorizer_filename}")
        logging.info(f"  - target_column: {target_column}")
        logging.info(f"  - text_column: {text_column}")
        logging.info(f"  - learning_rate: {learning_rate}")
        logging.info(f"  - iterations: {iterations}")
        logging.info(f"  - depth: {depth}")

    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the training step.

        Args:
            data (pd.DataFrame): Input data containing features and target column

        Returns:
            Dict[str, Any]: Dictionary with evaluation metrics and model paths
        """
        logging.info("Starting model training step...")
        start_time = time.time()

        logging.info(f"Input data shape: {data.shape}")
        logging.info(f"Columns: {list(data.columns)}")

        # Check if required columns exist
        if self.target_column not in data.columns:
            error_msg = f"Target column '{self.target_column}' not found in data"
            logging.error(error_msg)
            raise ValueError(error_msg)

        if self.text_column not in data.columns:
            error_msg = f"Text column '{self.text_column}' not found in data"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Log data statistics
        logging.info(f"Training data summary:")

        # Target column stats
        target_min = data[self.target_column].min()
        target_max = data[self.target_column].max()
        target_mean = data[self.target_column].mean()
        target_median = data[self.target_column].median()
        target_null = data[self.target_column].isnull().sum()

        logging.info(f"Target column '{self.target_column}':")
        logging.info(f"  - Range: {target_min} to {target_max}")
        logging.info(f"  - Mean: {target_mean}")
        logging.info(f"  - Median: {target_median}")
        logging.info(f"  - Null values: {target_null}")

        # Text column stats
        text_null = data[self.text_column].isnull().sum()
        text_empty = (data[self.text_column] == "").sum()

        logging.info(f"Text column '{self.text_column}':")
        logging.info(f"  - Null values: {text_null}")
        logging.info(f"  - Empty strings: {text_empty}")

        # Evaluate model performance
        logging.info("Evaluating model performance...")
        eval_start_time = time.time()
        metrics = self.model.evaluate(data, target_column=self.target_column)
        eval_elapsed_time = time.time() - eval_start_time
        logging.info(f"Model evaluation completed in {eval_elapsed_time:.2f} seconds")

        # Log evaluation metrics
        logging.info("Model evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"  - {metric_name}: {metric_value}")

        # Train the model# Train the model
        logging.info("Training CatBoost model...")
        train_start_time = time.time()
        model = self.model.train(data, target_column=self.target_column)
        train_elapsed_time = time.time() - train_start_time
        logging.info(f"Model training completed in {train_elapsed_time:.2f} seconds")
        # Save the model
        model_full_path = os.path.join(self.model_path, self.model_filename)
        logging.info(f"Saving model to {model_full_path}...")
        save_start_time = time.time()
        self.model.save_model(model_full_path)

        # Save the vectorizer separately for backward compatibility
        vectorizer_full_path = os.path.join(self.model_path, self.vectorizer_filename)
        if hasattr(self.model, "vectorizer") and self.model.vectorizer is not None:
            logging.info(
                f"Saving vectorizer separately to {vectorizer_full_path} for backward compatibility..."
            )
            import joblib

            joblib.dump(self.model.vectorizer, vectorizer_full_path)
            logging.info("Vectorizer saved successfully")
        else:
            logging.warning("No vectorizer found in model. Skipping vectorizer save.")

        save_elapsed_time = time.time() - save_start_time
        logging.info(f"Model and vectorizer saved in {save_elapsed_time:.2f} seconds")
        logging.info(
            f"Note: The model is now saved as a complete object and includes the vectorizer."
        )

        # Return results
        results = {
            "metrics": metrics,
            "model_path": model_full_path,
            "vectorizer_path": vectorizer_full_path
            if hasattr(self.model, "vectorizer")
            else None,
            "is_trained": self.model.is_trained,
        }

        total_elapsed_time = time.time() - start_time
        logging.info(f"Training step completed in {total_elapsed_time:.2f} seconds")

        return results
