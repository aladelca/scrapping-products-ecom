"""Price prediction models."""
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_processing.text_processor import FeatureExtractor, TextProcessor
from src.pipeline.steps.text_processing import TextProcessingStep


class PricePredictor:
    """Base class for price prediction models."""

    def __init__(self, model: BaseEstimator):
        """Initialize the price predictor."""
        self.model = model
        self.feature_columns = None
        self.target_column = "price"

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model."""
        # This method should be implemented by specific predictor classes
        raise NotImplementedError

    def train(
        self, data: pd.DataFrame, target_column: str = "price"
    ) -> Dict[str, float]:
        """Train the model on the provided data."""
        self.target_column = target_column
        X = self.prepare_features(data)
        y = data[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
        }

        return metrics

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""

        return self.model.predict(data)

    def set_vectorizer(self, vectorizer: Any):
        """Set the text vectorizer for feature preparation."""
        self.vectorizer = vectorizer

    def save_model(self, path: str):
        """Save the trained model."""
        joblib.dump(self, path)

    def load_model(self, path: str):
        """Load a trained model."""
        loaded = joblib.load(path)
        self.model = loaded.model
        if hasattr(loaded, "vectorizer"):
            self.vectorizer = loaded.vectorizer
        if hasattr(loaded, "is_trained"):
            self.is_trained = loaded.is_trained
        return self


class CatBoostPricePredictor(PricePredictor):
    """Price predictor using CatBoost model."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        iterations: int = 1000,
        depth: int = 6,
        model: Optional[CatBoostRegressor] = None,
    ):
        """Initialize CatBoost price predictor with custom parameters."""
        model = CatBoostRegressor(
            learning_rate=learning_rate,
            iterations=iterations,
            depth=depth,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
        super().__init__(model=model)
        self.vectorizer = FeatureExtractor()
        if model is not None:
            self.is_trained = True
        else:
            self.is_trained = False

    def set_vectorizer(self, vectorizer: Any):
        """Set the text vectorizer for feature preparation."""
        self.vectorizer = vectorizer

    def train(
        self, data: pd.DataFrame, target_column: str = "price"
    ) -> Dict[str, float]:
        """Train the CatBoost model with early stopping on evaluation set."""
        self.target_column = target_column

        X = data.drop(columns=[target_column])

        X = self.vectorizer.fit_transform(X["description_processed"])

        y = data[target_column]

        # Train with early stopping
        self.model.fit(
            X,
            y,
            early_stopping_rounds=50,
            verbose=False,
        )
        self.is_trained = True
        return self.model

    def evaluate(
        self,
        data: pd.DataFrame,
        target_column: str = "price",
        eval_fraction: float = 0.2,
        already_preprocessed: bool = False,
    ) -> Dict[str, float]:
        """Evaluate the model on the provided data."""

        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Split the data
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=eval_fraction, random_state=123
        )

        X_train = self.vectorizer.fit_transform(X_train["description_processed"])
        X_eval = self.vectorizer.transform(X_eval["description_processed"])

        # Get predictions
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        eval_pred = self.model.predict(X_eval)

        # Calculate metrics
        metrics = {
            "train_mse": mean_squared_error(y_train, train_pred),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "eval_mse": mean_squared_error(y_eval, eval_pred),
            "eval_mae": mean_absolute_error(y_eval, eval_pred),
            "train_r2": r2_score(y_train, train_pred),
            "eval_r2": r2_score(y_eval, eval_pred),
        }

        return metrics

    def predict(self, data: pd.DataFrame, step="evaluate") -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        if step == "evaluate":
            if isinstance(data, str):
                data = pd.DataFrame({"description_processed": [data]})

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data, columns=["description_processed"])

        elif step == "inference":
            if isinstance(data, str):
                data = pd.DataFrame({"description": [data]})

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data, columns=["description"])
            data = TextProcessingStep().execute(data)

        data = self.vectorizer.transform(data["description_processed"])

        return super().predict(data)
