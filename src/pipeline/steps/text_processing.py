"""Text processing step module."""

import logging
import time

import pandas as pd

from src.data_processing.text_processor import TextProcessor
from src.pipeline.base import PipelineStep


class TextProcessingStep(PipelineStep):
    """Step for processing text data."""

    def __init__(self, text_column: str = "description"):
        """Initialize the text processing step."""
        self.text_column = text_column
        self.text_processor = TextProcessor()

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the text processing step.

        Args:
            data (pd.DataFrame): Data containing text to process

        Returns:
            pd.DataFrame: Data with processed text
        """
        logging.info("Starting text processing step...")
        start_time = time.time()

        if data is None or len(data) == 0:
            logging.warning("Empty data provided to TextProcessingStep")
            return pd.DataFrame()

        logging.info(f"Input data shape: {data.shape}")

        # Check if the text column exists
        if self.text_column not in data.columns:
            logging.warning(
                f"Text column '{self.text_column}' not found in data. Columns available: {list(data.columns)}"
            )
            logging.info("Skipping text processing step")
            return data

        # Make a copy to avoid modifying the original data
        processed_data = data.copy()

        # Count non-null values in the text column
        non_null_count = processed_data[self.text_column].notna().sum()
        null_count = len(processed_data) - non_null_count
        logging.info(
            f"Processing {non_null_count} non-null text values in column '{self.text_column}' ({null_count} null values)"
        )

        # Process text data
        logging.info(
            "Applying text processing pipeline (remove stopwords, lemmatization)..."
        )

        # Create a new column for processed text
        processed_column = f"{self.text_column}_processed"

        # Apply text processing to non-null values
        processed_texts = []
        processed_count = 0
        error_count = 0

        for text in processed_data[self.text_column]:
            try:
                processed_text = self.text_processor.process_text(text)
                processed_texts.append(processed_text)
                if processed_text is not None:
                    processed_count += 1
            except Exception as e:
                logging.error(f"Error processing text: {str(e)}")
                processed_texts.append(None)
                error_count += 1

        # Add processed text column
        processed_data[processed_column] = processed_texts

        # Log results
        logging.info(f"Successfully processed {processed_count} texts")
        if error_count > 0:
            logging.warning(f"Encountered {error_count} errors during text processing")

        null_after = processed_data[processed_column].isnull().sum()
        logging.info(
            f"Column '{processed_column}': {null_after} null values after processing"
        )

        # Calculate average text length reduction
        if processed_count > 0:
            original_lengths = processed_data[self.text_column].fillna("").apply(len)
            processed_lengths = processed_data[processed_column].fillna("").apply(len)
            avg_original = original_lengths.mean()
            avg_processed = processed_lengths.mean()
            reduction_pct = (
                ((avg_original - avg_processed) / avg_original) * 100
                if avg_original > 0
                else 0
            )

            logging.info(
                f"Average text length: original={avg_original:.2f} chars, processed={avg_processed:.2f} chars"
            )
            logging.info(f"Text length reduction: {reduction_pct:.2f}%")

        elapsed_time = time.time() - start_time
        logging.info(f"Text processing completed in {elapsed_time:.2f} seconds")
        logging.info(f"Output data shape: {processed_data.shape}")

        return processed_data
