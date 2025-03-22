"""Test fixtures and configuration for the scraper module."""

import pytest
from selenium.webdriver.remote.webelement import WebElement


@pytest.fixture
def mock_webdriver(mocker):
    """Create a mock WebDriver instance."""
    mock_driver = mocker.Mock()
    mock_driver.find_element.return_value = mocker.Mock()
    mock_driver.page_source = "<html><body>Test page</body></html>"
    return mock_driver


@pytest.fixture
def mock_element():
    """Create a mock WebElement instance."""
    mock_element = WebElement(None, "test-id")
    mock_element.text = "Test Text"
    mock_element.get_attribute.return_value = "test-class"
    return mock_element


@pytest.fixture
def sample_product_data():
    """Create sample product data for testing."""
    return {
        "description": ["Test Product 1", "Test Product 2"],
        "brand": ["Brand 1", "Brand 2"],
        "original_price": ["$100", "$200"],
        "offer_price": ["$90", "$180"],
    }


@pytest.fixture
def mock_page_source():
    """Create a mock HTML page source for testing."""
    return """
    <html>
        <body>
            <section>
                <ol>
                    <li>
                        <div>
                            <div>
                                <div>
                                    <h3>Test Product</h3>
                                    <span class="poly-component__brand">
                                        Test Brand
                                    </span>
                                    <div>
                                        <s class="previous">$100</s>
                                        <div>
                                            <span class="andes-money-amount">
                                                $90
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </li>
                </ol>
            </section>
        </body>
    </html>
    """
