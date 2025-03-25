"""Utility class for data processing and feature engineering."""
from typing import List

import nltk
import pandas as pd
import spacy
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

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and converting to lowercase.

        Args:
            text (str): Text to clean

        Returns:
            str: Cleaned text in lowercase
        """
        if not isinstance(text, str) or pd.isna(text):
            return None

        # Remove special characters and convert to lowercase
        chars_to_remove = ["|", "\\", "/", ",", "."]
        for char in chars_to_remove:
            text = text.replace(char, "")

        return text.lower().strip()

    def remove_stopwords(self, text):
        """
        Remove Spanish stopwords from text.

        Args:
            text (str): Text to process (should be a string with words separated by spaces)

        Returns:
            str: Text with stopwords removed
        """
        # Descargar stopwords si no están descargadas

        stopwords = nltk.download("stopwords")

        # Obtener stopwords en español
        stop_words = nltk.corpus.stopwords.words("spanish")

        # Filtrar stopwords y unir las palabras
        filtered_words = [word for word in text if word not in stop_words]
        return " ".join(filtered_words)

    def lemmatize_text(self, text):
        """
        Lemmatize Spanish text using spaCy.

        Args:
            text (str): Text to lemmatize

        Returns:
            str: Lemmatized text
        """
        # Cargar el modelo en español de spaCy
        nlp = spacy.load("es_core_news_sm")

        # Procesar el texto
        doc = nlp(text.lower())

        # Obtener los lemas, ignorando stopwords y puntuación
        lemmas = [
            token.lemma_ for token in doc if not token.is_stop and not token.is_punct
        ]

        # Unir los lemas en un string
        return " ".join(lemmas)

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


class DataCleaner(DataProcessor):
    """Clean the data."""

    def __init__(self):
        """Initialize the data cleaner."""
        super().__init__()

    def filter_null(self, df: pd.DataFrame, field: str) -> pd.DataFrame:
        """Filter the data to remove null values."""
        return df[df[field].notnull()]

    def clean_price(self, price: str) -> float:
        """
        Clean the price string to get only the numeric value.
        Handles cases like 'S/279' or 'S/248,40' converting them to 279.0 and 248.40 respectively.
        """
        if not isinstance(price, str) or pd.isna(price):
            return None

        # Eliminar el símbolo de moneda y espacios en blanco
        price = price.replace("S/", "").strip()

        # Si hay una coma, tratarla como separador decimal
        if "," in price:
            parts = price.split(",")
            price = f"{parts[0].strip()}.{parts[1].strip()}"

        # Eliminar cualquier carácter que no sea número o punto decimal
        price = "".join(char for char in price if char.isdigit() or char == ".")

        try:
            return float(price)
        except (ValueError, TypeError):
            return None
