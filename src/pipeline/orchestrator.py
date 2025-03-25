"""Pipeline orchestrator module."""

import pandas as pd

from .base import Pipeline
from .steps.cleaning import CleaningStep
from .steps.scraping import ScrapingStep
from .steps.text_processing import TextProcessingStep


def create_pipeline(url: str, headless: bool = True) -> Pipeline:
    """
    Create a pipeline for scraping and processing product data.

    Args:
        url (str): URL to scrape
        headless (bool): Whether to run browser in headless mode

    Returns:
        Pipeline: Configured pipeline
    """
    pipeline = Pipeline()

    # Add pipeline steps in order
    pipeline.add_step(ScrapingStep(url=url, headless=headless)).add_step(
        CleaningStep()
    ).add_step(TextProcessingStep())

    return pipeline


def process_range(
    root_page: str,
    initial_page: int,
    final_page: int,
    headless: bool = True,
) -> pd.DataFrame:
    """
    Process a range of pages.

    Args:
        root_page (str): Base URL
        initial_page (int): Starting page number
        final_page (int): Ending page number
        headless (bool): Whether to run browser in headless mode

    Returns:
        pd.DataFrame: Processed data from all pages
    """
    all_data = []

    for page in range(initial_page, final_page + 1):
        url = f"{root_page}/{page}"
        pipeline = create_pipeline(url=url, headless=headless)
        page_data = pipeline.execute()
        all_data.append(page_data)

    return pd.concat(all_data, ignore_index=True)
