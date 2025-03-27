"""Scraping step module."""

import pandas as pd

from src.pipeline.base import PipelineStep
from src.scraper.mercado_libre import MercadoLibreScraper


class ScrapingStep(PipelineStep):
    """Step for scraping product data."""

    def __init__(self, url: str, headless: bool = True):
        """
        Initialize the scraping step.

        Args:
            url (str): URL to scrape
            headless (bool): Whether to run browser in headless mode
        """
        self.url = url
        self.headless = headless

    def execute(self, data=None) -> pd.DataFrame:
        """
        Execute the scraping step.

        Args:
            data: Not used in this step

        Returns:
            pd.DataFrame: Scraped data
        """
        scraper = MercadoLibreScraper(headless=self.headless)
        scraped_data = scraper.scrape_product_data(self.url)
        return pd.DataFrame(scraped_data)
