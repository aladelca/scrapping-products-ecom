# E-commerce Product Scraper and Price Predictor

This project is part of a Python course at Cibertec Peru, where we explore web scraping and machine learning techniques to analyze e-commerce product data. The project demonstrates practical applications of Python in data collection, processing, and predictive modeling.

## Course Context

This project is developed as part of the Python programming course at Cibertec Peru, where students learn about:
- Web scraping with Selenium
- Data processing and analysis
- Machine learning fundamentals
- Best practices in Python development
- Testing and documentation

## Project Structure

```
.
├── src/
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── base_scraper.py
│   │   └── scraper.py
│   ├── models/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       └── test_scraper.py
├── notebooks/
│   └── 01_data_exploration.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── .pre-commit-config.yaml
├── conftest.py
└── README.md
```

## Features

### 1. Web Scraping Module (`src/scraper/`)
- `base_scraper.py`: Abstract base class defining the scraping interface
- `scraper.py`: Implementation of the product scraper using Selenium
- Robust error handling and retry mechanisms
- Support for headless browser mode
- Configurable wait times and timeouts

### 2. Testing (`src/tests/`)
- Comprehensive test suite for the scraper module
- Unit tests for individual components
- Integration tests for end-to-end functionality
- Mock objects and fixtures for reliable testing
- Test coverage reporting

### 3. Data Processing (`src/utils/`)
- Data cleaning and preprocessing utilities
- Feature extraction functions
- Data validation and quality checks

### 4. Machine Learning Models (`src/models/`)
- Price prediction models
- Feature engineering
- Model evaluation and metrics

### 5. Jupyter Notebooks (`notebooks/`)
- Data exploration and analysis
- Model development and experimentation
- Visualization of results

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecommerce-scraper.git
cd ecommerce-scraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up pre-commit hooks:
```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run against all files (optional)
pre-commit run --all-files
```

The project uses the following pre-commit hooks:
- **Black**: Code formatting
- **Flake8**: Code linting and style checking
- **isort**: Import sorting
- **mypy**: Static type checking

4. Run tests:
```bash
pytest src/tests/
```

5. Run with coverage:
```bash
pytest --cov=src src/tests/
```

## Usage

```python
from src.scraper.scraper import scrape_products

# Scrape products from a webpage
data = scrape_products("https://example.com/products", headless=True)

# Access the scraped data
descriptions = data["description"]
brands = data["brand"]
original_prices = data["original_price"]
offer_prices = data["offer_price"]
```

## Course Instructor

This project is developed and maintained by Adrian Larcon, a Python instructor at Cibertec Peru. The course focuses on practical applications of Python in data science and web automation, with emphasis on:
- Clean code practices
- Test-driven development
- Documentation and maintainability
- Real-world problem solving

## Contributing

Students are encouraged to:
1. Fork the repository
2. Create a feature branch
3. Commit their changes
4. Push to the branch
5. Create a Pull Request

## License

This project is part of the Cibertec Peru Python course curriculum and is licensed under the MIT License.
