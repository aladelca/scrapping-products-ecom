"""Price prediction models."""
import joblib
import numpy as np
import pandas as pd
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

    def train(self, data: pd.DataFrame, target_column: str = "price"):
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
