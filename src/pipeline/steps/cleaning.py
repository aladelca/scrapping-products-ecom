"""Cleaning step module."""

import pandas as pd

from data_processing.cleaner import DataCleaner

from ..base import PipelineStep


class CleaningStep(PipelineStep):
    """Step for cleaning scraped data."""

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the cleaning step.

        Args:
            data (pd.DataFrame): Data to clean

        Returns:
            pd.DataFrame: Cleaned data
        """
        cleaner = DataCleaner(data)
        return cleaner.clean_prices().clean_descriptions().get_cleaned_data()
