"""Text processing step module."""

import pandas as pd

from data_processing.text_processor import TextProcessor

from ..base import PipelineStep


class TextProcessingStep(PipelineStep):
    """Step for processing text data."""

    def __init__(self):
        """Initialize the text processing step."""
        self.processor = TextProcessor()

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the text processing step.

        Args:
            data (pd.DataFrame): Data to process

        Returns:
            pd.DataFrame: Processed data
        """
        if "description" in data.columns:
            data["description_processed"] = data["description"].apply(
                self.processor.process_text
            )
        return data
