"""Base scraper module."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class BaseScraper(ABC):
    """Abstract base class for web scrapers."""

    def __init__(self, headless: bool = False):
        """
        Initialize the scraper.

        Args:
            headless (bool): Whether to run the browser in headless mode.
        """
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.driver = None

    @abstractmethod
    def setup_driver(self) -> None:
        """Set up the WebDriver with appropriate options."""
        pass

    @abstractmethod
    def scrape_product_data(
        self, webpage: str, num_pages: int = 1
    ) -> Dict[str, List[Optional[str]]]:
        """
        Scrape product data from the webpage.

        Args:
            webpage (str): URL to scrape
            num_pages (int): Number of pages to scrape

        Returns:
            Dict[str, List[Optional[str]]]: Scraped product data
        """
        pass

    def __del__(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
