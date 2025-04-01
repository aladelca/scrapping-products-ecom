"""Text processing module."""

import os
from typing import List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


class TextProcessor:
    """Class for text processing operations."""

    def __init__(self):
        """Initialize the TextProcessor."""
        # Download required NLTK data
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk_data_dir = "/tmp/nltk_data"
            os.makedirs(nltk_data_dir, exist_ok=True)

            nltk.data.path.append(nltk_data_dir)

            nltk.download("stopwords", download_dir=nltk_data_dir)
        self.stemmer = SnowballStemmer("spanish")
        self.stop_words = set(stopwords.words("spanish"))

    def remove_stopwords(self, text: str) -> Optional[str]:
        """
        Remove Spanish stopwords from text.

        Args:
            text (str): Text to process

        Returns:
            Optional[str]: Text with stopwords removed
        """
        if not isinstance(text, str) or not text.strip():
            return None

        words = text.lower().split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return " ".join(filtered_words)

    def lemmatize(self, text: str) -> Optional[str]:
        """
        Lemmatize Spanish text using Spacy.

        Args:
            text (str): Text to lemmatize

        Returns:
            Optional[str]: Lemmatized text
        """
        if not isinstance(text, str) or not text.strip():
            return None

        nlp = spacy.load("es_core_news_sm")
        doc = nlp(text.lower())

        # Get lemmas, ignoring stopwords and punctuation
        lemmas = [
            token.lemma_ for token in doc if not token.is_stop and not token.is_punct
        ]

        return " ".join(lemmas)

    def process_text(self, text: str) -> Optional[str]:
        """
        Apply full text processing pipeline.

        Args:
            text (str): Text to process

        Returns:
            Optional[str]: Processed text
        """
        if not isinstance(text, str) or not text.strip():
            return None

        # Remove stopwords first, then lemmatize
        text_no_stopwords = self.remove_stopwords(text)
        if text_no_stopwords:
            return self.lemmatize(text_no_stopwords)
        return None


class FeatureExtractor(TextProcessor):
    """Class for feature extraction operations."""

    def __init__(self):
        """Initialize the FeatureExtractor."""
        super().__init__()
        self.vectorizer = CountVectorizer()
        self.is_fitted = False

    def fit_transform(
        self, texts: pd.Series, already_processed: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the vectorizer and transform the texts.
        Use this method for training data.

        Args:
            texts (pd.Series): Series of texts to vectorize

        Returns:
            Tuple[np.ndarray, List[str]]: Vectorized texts and feature names
        """
        if already_processed:
            self.is_fitted = True
            return self.vectorizer.fit_transform(texts).toarray()
        else:
            # Process texts first
            processed_texts = texts.apply(self.process_text).fillna("")
            # Fit and transform
            vectors = self.vectorizer.fit_transform(processed_texts)
            self.is_fitted = True

            return vectors.toarray()

    def transform(self, texts: pd.Series, already_processed: bool = True) -> np.ndarray:
        """
        Transform texts using the fitted vectorizer.
        Use this method for validation/test data.

        Args:
            texts (pd.Series): Series of texts to vectorize

        Returns:
            np.ndarray: Vectorized texts
        """
        if not self.is_fitted:
            raise ValueError(
                "Vectorizer must be fitted before calling transform. "
                "Call fit_transform() first."
            )
        if already_processed:
            return self.vectorizer.transform(texts).toarray()
        else:
            # Process texts first
            processed_texts = texts.apply(self.process_text).fillna("")

            # Transform only
            return self.vectorizer.transform(processed_texts).toarray()

    def get_feature_names(self) -> List[str]:
        """
        Get feature names from the vectorizer.

        Returns:
            List[str]: List of feature names
        """
        if not self.is_fitted:
            raise ValueError(
                "Vectorizer must be fitted before getting feature names. "
                "Call fit_transform() first."
            )
        return self.vectorizer.get_feature_names_out()
