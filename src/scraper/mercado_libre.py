"""MercadoLibre scraper implementation."""

import time
from typing import Dict, List, Optional

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..utils.constants import XPATH_MAPPINGS
from .base import BaseScraper


class MercadoLibreScraper(BaseScraper):
    """Scraper for MercadoLibre website."""

    def setup_driver(self) -> None:
        """Set up the Chrome WebDriver with appropriate options."""
        try:
            self.driver = webdriver.Chrome(options=self.options)
            self.driver.maximize_window()
        except WebDriverException as e:
            raise RuntimeError(f"Failed to initialize WebDriver: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WebDriver: {str(e)}")

    def wait_for_element(
        self, xpath: str, timeout: int = 10
    ) -> Optional[webdriver.remote.webelement.WebElement]:
        """
        Wait for an element to be present on the page.

        Args:
            xpath (str): The XPath of the element to wait for
            timeout (int): Maximum time to wait in seconds

        Returns:
            Optional[WebElement]: The found element or None if not found
        """
        try:
            wait = WebDriverWait(self.driver, timeout)
            element_locator = (By.XPATH, xpath)
            return wait.until(EC.presence_of_element_located(element_locator))
        except TimeoutException:
            return None

    def get_text_safe(
        self, element: webdriver.remote.webelement.WebElement
    ) -> Optional[str]:
        """
        Safely get text from a WebElement.

        Args:
            element (WebElement): The WebElement to get text from

        Returns:
            Optional[str]: The text content or None if not available
        """
        try:
            return element.text.strip()
        except (AttributeError, WebDriverException):
            return None

    def scrape_product_data(
        self,
        webpage: str,
        num_pages: int = 1,
    ) -> Dict[str, List[Optional[str]]]:
        """
        Scrape product data from the given webpage.

        Args:
            webpage (str): The URL of the webpage to scrape
            num_pages (int): Number of pages to scrape

        Returns:
            Dict[str, List[Optional[str]]]: Dictionary containing lists of
            scraped data

        Raises:
            RuntimeError: If WebDriver initialization fails or webpage access
            fails
        """
        data = {
            "description": [],
            "brand": [],
            "original_price": [],
            "offer_price": [],
        }

        try:
            self.setup_driver()
            self.driver.get(webpage)
            time.sleep(5)  # Initial wait for dynamic content

            # Wait for product list to load
            if not self.wait_for_element(XPATH_MAPPINGS["product_list"]):
                return data

            for i in range(1, 52):
                # Scrape description
                try:
                    description = self.driver.find_element(
                        By.XPATH,
                        XPATH_MAPPINGS["description"].format(i),
                    )
                    data["description"].append(self.get_text_safe(description))
                except NoSuchElementException:
                    data["description"].append(None)

                # Scrape brand
                try:
                    brand = None
                    for xpath in XPATH_MAPPINGS["brand_options"]:
                        try:
                            element = self.driver.find_element(
                                By.XPATH,
                                xpath.format(i),
                            )
                            if "poly-component__brand" in element.get_attribute(
                                "class"
                            ):
                                brand = element
                                break
                        except NoSuchElementException:
                            continue

                    data["brand"].append(self.get_text_safe(brand) if brand else None)
                except NoSuchElementException:
                    data["brand"].append(None)

                # Scrape original price
                try:
                    original_price = self.driver.find_element(
                        By.XPATH,
                        XPATH_MAPPINGS["original_price"].format(i),
                    )
                    if "previous" in original_price.get_attribute("class"):
                        data["original_price"].append(
                            self.get_text_safe(original_price)
                        )
                    else:
                        data["original_price"].append(None)
                except NoSuchElementException:
                    data["original_price"].append(None)

                # Scrape offer price
                try:
                    offer_price = None
                    for xpath in XPATH_MAPPINGS["offer_price_options"]:
                        try:
                            element = self.driver.find_element(
                                By.XPATH,
                                xpath.format(i),
                            )
                            if "andes-money-amount" in element.get_attribute("class"):
                                offer_price = element
                                break
                        except NoSuchElementException:
                            continue

                    data["offer_price"].append(
                        self.get_text_safe(offer_price) if offer_price else None
                    )
                except NoSuchElementException:
                    data["offer_price"].append(None)

        except WebDriverException as e:
            raise RuntimeError(f"Failed to scrape webpage: {str(e)}")
        finally:
            if self.driver:
                self.driver.quit()

        return data
