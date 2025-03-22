"""Base class for all e-commerce scrapers."""

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


class BaseScraper(ABC):
    """Base class for all e-commerce scrapers."""

    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

    @abstractmethod
    def scrape_product(self, url: str) -> Dict[str, Any]:
        """Scrape a single product page."""
        pass

    @abstractmethod
    def scrape_search_results(
        self, query: str, num_pages: int = 1
    ) -> List[Dict[str, Any]]:
        """Scrape search results for a given query."""
        pass

    def _get_page(self, url: str) -> BeautifulSoup:
        """Get and parse a webpage with rate limiting."""
        time.sleep(random.uniform(1, 3))  # Basic rate limiting
        response = self.session.get(url, headers=self.headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _clean_text(self, text: str) -> str:
        """Clean scraped text data."""
        if not text:
            return ""
        return " ".join(text.strip().split())
