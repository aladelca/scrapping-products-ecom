"""Feature extraction step module."""

import pandas as pd

from data_processing.text_processor import FeatureExtractor

from ..base import PipelineStep


class FeatureExtractionStep(PipelineStep):
    """Step for feature extraction."""

    def __init__(self, already_processed: bool = True):
        """Initialize the feature extraction step."""
        self.processor = FeatureExtractor()
        self.already_processed = already_processed

    def execute(self, data: pd.DataFrame, column: str, step: str) -> pd.DataFrame:
        """
        Execute the feature extraction step.

        Args:
            data (pd.DataFrame): Data to process
            step (str): Step to execute

        Returns:
            pd.DataFrame: Processed data
        """
        if step == "train":
            return self.processor.fit_transform(data[column], self.already_processed)
        elif step == "inference":
            return self.processor.transform(data[column], self.already_processed)
        else:
            raise ValueError(f"Invalid step: {step}")
