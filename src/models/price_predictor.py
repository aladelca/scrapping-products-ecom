"""Price prediction models."""
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
        X = self.prepare_features(data)
        return self.model.predict(X)

    def save_model(self, path: str):
        """Save the trained model."""
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load a trained model."""
        self.model = joblib.load(path)


class CatBoostPricePredictor(PricePredictor):
    """Price predictor using CatBoost model."""

    def __init__(
        self, learning_rate: float = 0.1, iterations: int = 1000, depth: int = 6
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
        self.vectorizer = None

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for the CatBoost model using text vectorization."""
        if not self.vectorizer:
            raise ValueError(
                "Vectorizer not set. Please set the vectorizer before preparing features."
            )

        # Ensure data is properly formatted
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        return data

    def set_vectorizer(self, vectorizer: Any):
        """Set the text vectorizer for feature preparation."""
        self.vectorizer = vectorizer

    def train(
        self,
        data: pd.DataFrame,
        target_column: str = "price",
        eval_fraction: float = 0.2,
    ) -> Dict[str, float]:
        """Train the CatBoost model with early stopping on evaluation set."""
        self.target_column = target_column
        X = self.prepare_features(data)
        y = data[target_column]

        # Split the data
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=eval_fraction, random_state=42
        )

        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_eval, y_eval),
            early_stopping_rounds=50,
            verbose=False,
        )

        # Get predictions
        train_pred = self.model.predict(X_train)
        eval_pred = self.model.predict(X_eval)

        # Calculate metrics
        metrics = {
            "train_mse": mean_squared_error(y_train, train_pred),
            "eval_mse": mean_squared_error(y_eval, eval_pred),
            "train_r2": r2_score(y_train, train_pred),
            "eval_r2": r2_score(y_eval, eval_pred),
            "best_iteration": self.model.get_best_iteration(),
        }

        return metrics
