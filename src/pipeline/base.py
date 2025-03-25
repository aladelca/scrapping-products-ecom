"""Base pipeline module."""

from abc import ABC, abstractmethod
from typing import Any


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    @abstractmethod
    def execute(self, data: Any) -> Any:
        """
        Execute the pipeline step.

        Args:
            data: Input data for the step

        Returns:
            Processed data
        """
        pass


class Pipeline:
    """Class for managing the data processing pipeline."""

    def __init__(self):
        """Initialize the pipeline."""
        self.steps = []

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """
        Add a step to the pipeline.

        Args:
            step (PipelineStep): Step to add

        Returns:
            Pipeline: Self for method chaining
        """
        self.steps.append(step)
        return self

    def execute(self, initial_data: Any = None) -> Any:
        """
        Execute all steps in the pipeline.

        Args:
            initial_data: Initial data to process

        Returns:
            Processed data
        """
        data = initial_data
        for step in self.steps:
            data = step.execute(data)
        return data
