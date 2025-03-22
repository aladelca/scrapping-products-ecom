"""Utility class for data processing and feature engineering."""
from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessor:
    """Utility class for data processing and feature engineering."""

    def __init__(self):
        """Initialize the data processor."""
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data."""
        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df = df.fillna(
            {"price": df["price"].median(), "description": "", "category": "unknown"}
        )

        # Remove outliers in price (using IQR method)
        Q1 = df["price"].quantile(0.25)
        Q3 = df["price"].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df["price"] < (Q1 - 1.5 * IQR)) | (df["price"] > (Q3 + 1.5 * IQR)))]

        return df

    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables."""
        df_encoded = df.copy()

        for column in columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df_encoded[column] = self.label_encoders[column].fit_transform(
                df_encoded[column]
            )

        return df_encoded

    def scale_numerical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()
        df_scaled[columns] = self.scaler.fit_transform(df_scaled[columns])
        return df_scaled

    def extract_text_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract features from text data."""
        df_features = df.copy()

        # Add basic text features
        df_features["text_length"] = df_features[text_column].str.len()
        df_features["word_count"] = df_features[text_column].str.split().str.len()

        return df_features

    def prepare_features(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        numerical_columns: List[str],
        text_column: str = "description",
    ) -> pd.DataFrame:
        """Prepare all features for modeling."""
        # Clean the data
        df_clean = self.clean_data(df)

        # Extract text features
        df_features = self.extract_text_features(df_clean, text_column)

        # Encode categorical variables
        df_features = self.encode_categorical(df_features, categorical_columns)

        # Scale numerical features
        df_features = self.scale_numerical(df_features, numerical_columns)

        return df_features
