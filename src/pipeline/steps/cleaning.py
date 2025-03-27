"""Cleaning step module."""

import logging
import time

import pandas as pd

from src.data_processing.cleaner import DataCleaner
from src.pipeline.base import PipelineStep


class CleaningStep(PipelineStep):
    """Step for cleaning raw data."""

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the cleaning step.

        Args:
            data (pd.DataFrame): Raw data to clean

        Returns:
            pd.DataFrame: Cleaned data
        """
        logging.info("Starting data cleaning process...")
        start_time = time.time()

        if data is None or len(data) == 0:
            logging.warning("Empty data provided to CleaningStep")
            return pd.DataFrame()

        logging.info(f"Input data shape: {data.shape}")

        # Make a copy to avoid modifying the original data
        cleaned_data = data.copy()

        # Log initial null values
        for col in cleaned_data.columns:
            null_count = cleaned_data[col].isnull().sum()
            null_percent = (null_count / len(cleaned_data)) * 100
            logging.info(
                f"Column '{col}': {null_count} null values ({null_percent:.2f}%)"
            )

        # Drop duplicates if any
        original_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        duplicate_rows = original_rows - len(cleaned_data)
        logging.info(
            f"Removed {duplicate_rows} duplicate rows ({(duplicate_rows/original_rows)*100:.2f}% of data)"
        )

        # Fill missing values
        for col in cleaned_data.columns:
            if (
                cleaned_data[col].dtype == "object"
                or cleaned_data[col].dtype == "string"
            ):
                # Fill string columns with empty string
                null_count = cleaned_data[col].isnull().sum()
                if null_count > 0:
                    logging.info(
                        f"Filling {null_count} null values in column '{col}' with empty string"
                    )
                    cleaned_data[col] = cleaned_data[col].fillna("")
            else:
                # Fill numeric columns with 0
                null_count = cleaned_data[col].isnull().sum()
                if null_count > 0:
                    logging.info(
                        f"Filling {null_count} null values in column '{col}' with 0"
                    )
                    cleaned_data[col] = cleaned_data[col].fillna(0)

        # Log results
        elapsed_time = time.time() - start_time
        logging.info(f"Data cleaning completed in {elapsed_time:.2f} seconds")
        logging.info(f"Output data shape: {cleaned_data.shape}")

        return cleaned_data
