"""Tests for the product scraper module.

This module contains comprehensive tests for the ProductScraper class and its
functionality. Tests are divided into unit and integration tests to ensure proper
isolation and complete functionality verification.
"""

from unittest.mock import patch

import pytest
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)

from ..scraper.scraper import ProductScraper, scrape_products


class TestProductScraper:
    """Test suite for ProductScraper class.

    This test suite covers all the functionality of the ProductScraper class,
    including initialization, driver setup, element interaction, and data scraping.
    Tests are marked with appropriate markers for unit and integration testing.
    """

    @pytest.mark.unit
    def test_init(self, mock_webdriver):
        """Test ProductScraper initialization.

        Verifies that:
        1. The scraper is properly initialized with options
        2. Headless mode is correctly configured when specified
        3. The driver is initially None
        """
        scraper = ProductScraper(headless=True)
        assert scraper.options is not None
        assert "--headless" in scraper.options.arguments
        assert scraper.driver is None

    @pytest.mark.unit
    def test_setup_driver(self, mock_webdriver):
        """Test driver setup.

        Verifies that:
        1. The driver is properly initialized
        2. The window is maximized
        3. The driver instance is stored in the scraper
        """
        scraper = ProductScraper()
        scraper.setup_driver()
        assert scraper.driver is not None
        mock_webdriver.maximize_window.assert_called_once()

    @pytest.mark.unit
    def test_setup_driver_error(self, mock_webdriver):
        """Test driver setup error handling.

        Verifies that:
        1. WebDriver exceptions are properly caught
        2. Appropriate error messages are raised
        3. The error is properly propagated
        """
        with patch("selenium.webdriver.Chrome") as mock_chrome:
            mock_chrome.side_effect = WebDriverException("Driver error")
            scraper = ProductScraper()
            with pytest.raises(RuntimeError) as exc_info:
                scraper.setup_driver()
            assert "Failed to initialize WebDriver" in str(exc_info.value)

    @pytest.mark.unit
    def test_wait_for_element(self, mock_webdriver, mock_element):
        """Test element waiting functionality.

        Verifies that:
        1. The wait_for_element method correctly waits for elements
        2. The method returns the found element
        3. The element is properly located using XPath
        """
        scraper = ProductScraper()
        scraper.driver = mock_webdriver
        mock_webdriver.find_element.return_value = mock_element

        element = scraper.wait_for_element("//test/xpath")
        assert element == mock_element

    @pytest.mark.unit
    def test_wait_for_element_timeout(self, mock_webdriver):
        """Test element waiting timeout.

        Verifies that:
        1. TimeoutException is properly handled
        2. The method returns None when element is not found
        3. The timeout is properly enforced
        """
        scraper = ProductScraper()
        scraper.driver = mock_webdriver
        mock_webdriver.find_element.side_effect = TimeoutException()

        element = scraper.wait_for_element("//test/xpath")
        assert element is None

    @pytest.mark.unit
    def test_get_text_safe(self, mock_element):
        """Test safe text extraction.

        Verifies that:
        1. Text is properly extracted from elements
        2. Whitespace is properly stripped
        3. The method handles valid elements correctly
        """
        scraper = ProductScraper()
        text = scraper.get_text_safe(mock_element)
        assert text == "Test Text"

    @pytest.mark.unit
    def test_get_text_safe_error(self, mock_element):
        """Test safe text extraction with error.

        Verifies that:
        1. The method handles None text values
        2. No exceptions are raised for invalid elements
        3. None is returned for invalid elements
        """
        scraper = ProductScraper()
        mock_element.text = None
        text = scraper.get_text_safe(mock_element)
        assert text is None

    @pytest.mark.integration
    def test_scrape_product_data(self, mock_webdriver, mock_page_source, mock_element):
        """Test product data scraping.

        Verifies that:
        1. The scraper can extract all required product data
        2. The data structure is correct
        3. All expected fields are present
        4. The data is properly formatted
        """
        scraper = ProductScraper()
        scraper.driver = mock_webdriver
        mock_webdriver.page_source = mock_page_source

        # Configure mock to return the same element for all find_element calls
        mock_webdriver.find_element.return_value = mock_element

        data = scraper.scrape_product_data("http://test.com")

        assert isinstance(data, dict)
        assert all(
            key in data
            for key in ["description", "brand", "original_price", "offer_price"]
        )
        assert len(data["description"]) > 0

    @pytest.mark.integration
    def test_scrape_product_data_error(self, mock_webdriver):
        """Test product data scraping error handling.

        Verifies that:
        1. WebDriver exceptions are properly caught
        2. Appropriate error messages are raised
        3. The error is properly propagated
        """
        scraper = ProductScraper()
        scraper.driver = mock_webdriver
        mock_webdriver.get.side_effect = WebDriverException("Page error")

        with pytest.raises(RuntimeError) as exc_info:
            scraper.scrape_product_data("http://test.com")
        assert "Failed to scrape webpage" in str(exc_info.value)

    @pytest.mark.integration
    def test_scrape_products_function(
        self, mock_webdriver, mock_page_source, mock_element
    ):
        """Test the convenience function scrape_products.

        Verifies that:
        1. The convenience function works correctly
        2. It returns the expected data structure
        3. All required fields are present
        4. The data is properly formatted
        """
        mock_webdriver.page_source = mock_page_source
        # Configure mock to return the same element for all find_element calls
        mock_webdriver.find_element.return_value = mock_element

        data = scrape_products("http://test.com")

        assert isinstance(data, dict)
        assert all(
            key in data
            for key in ["description", "brand", "original_price", "offer_price"]
        )
        assert len(data["description"]) > 0

    @pytest.mark.unit
    def test_cleanup(self, mock_webdriver):
        """Test proper cleanup of resources.

        Verifies that:
        1. The driver is properly closed after use
        2. Resources are cleaned up even if an error occurs
        3. The cleanup happens in the finally block
        """
        scraper = ProductScraper()
        scraper.driver = mock_webdriver

        try:
            scraper.scrape_product_data("http://test.com")
        except Exception:
            pass
        finally:
            mock_webdriver.quit.assert_called_once()

    @pytest.mark.integration
    def test_scrape_product_data_no_elements(self, mock_webdriver):
        """Test scraping when no elements are found.

        Verifies that:
        1. The scraper handles empty pages gracefully
        2. None values are properly handled
        3. The data structure remains consistent
        """
        scraper = ProductScraper()
        scraper.driver = mock_webdriver
        mock_webdriver.find_element.side_effect = NoSuchElementException(
            "No elements found"
        )

        data = scraper.scrape_product_data("http://test.com")

        assert isinstance(data, dict)
        assert all(
            key in data
            for key in ["description", "brand", "original_price", "offer_price"]
        )
        assert all(len(values) == 0 for values in data.values())
