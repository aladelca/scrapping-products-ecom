"""Base pipeline module."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, List


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
        logging.info("Created new Pipeline instance")

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """
        Add a step to the pipeline.

        Args:
            step (PipelineStep): Step to add

        Returns:
            Pipeline: Self for method chaining
        """
        step_name = step.__class__.__name__
        logging.info(f"Adding step to pipeline: {step_name}")
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
        if not self.steps:
            logging.warning("Executing pipeline with no steps defined")
            return initial_data

        total_steps = len(self.steps)
        logging.info(f"Executing pipeline with {total_steps} steps")
        pipeline_start_time = time.time()

        data = initial_data

        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__
            step_number = i + 1

            logging.info(f"Running step {step_number}/{total_steps}: {step_name}")
            step_start_time = time.time()

            try:
                data = step.execute(data)
                step_elapsed_time = time.time() - step_start_time
                logging.info(
                    f"Step {step_number}/{total_steps}: {step_name} completed in {step_elapsed_time:.2f} seconds"
                )
            except Exception as e:
                logging.error(
                    f"Error in step {step_number}/{total_steps}: {step_name} - {str(e)}"
                )
                raise

        pipeline_elapsed_time = time.time() - pipeline_start_time
        logging.info(
            f"Pipeline execution completed in {pipeline_elapsed_time:.2f} seconds"
        )

        return data
