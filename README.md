# E-commerce Product Scraper and Price Predictor

This project is part of a Python course at Cibertec Peru, where we explore web scraping techniques and machine learning to analyze e-commerce product data. The project demonstrates practical Python applications in data collection, processing, and predictive modeling.

## Course Context

This project is developed as part of the Python programming course at Cibertec Peru, where students learnt about:
- Web scraping with Selenium
- Data processing and analysis
- Machine learning fundamentals
- Python development best practices
- Testing and documentation

## Project Structure

```
.
├── app.py                  # REST API launcher (FastAPI)
├── train.py                # Model training launcher
├── setup.py                # Installable package configuration
├── pyproject.toml          # Python project configuration
├── src/
│   ├── scraper/            # Web scraping module
│   ├── data_processing/    # Data and text processing
│   ├── pipeline/           # Processing and training pipeline
│   ├── models/             # ML model implementations
│   ├── utils/              # Utility functions and constants
│   ├── tests/              # Test suite
│   ├── api.py              # REST API with FastAPI
│   ├── predict.py          # Script for making predictions
│   └── train_step.py       # Script for training models
├── notebooks/              # Jupyter notebooks for exploration
├── data/                   # Data storage
├── models/                 # Saved models and vectorizers
└── trained_models/         # Alternative directory for models
```

## Features

### 1. Web Scraping Module (`src/scraper/`)
- Selenium-based scraper for collecting product data from e-commerce sites
- Robust error handling and retry mechanisms
- Support for browser headless mode
- Configurable wait times and timeouts

### 2. Data Processing Pipeline (`src/pipeline/`)
- Modular architecture for data processing
- Step-based processing with clear interfaces
- Configurable pipeline components
- Text processing and feature extraction

### 3. Machine Learning Models (`src/models/`)
- Price prediction models based on CatBoost
- Feature engineering and text vectorization
- Model evaluation and metric tracking
- Model serialization and loading for inference
- Improved model state persistence
- Support for different prediction steps (evaluation and inference)

### 4. Training System (`src/train_step.py`, `train.py`)
- Command-line interface for model training
- Option to use existing data or scrape new data
- Support for raw data processing through the complete pipeline
- Configurable model hyperparameters
- Metric tracking and model persistence

### 5. Prediction System (`src/predict.py`)
- Command-line interface for making predictions
- Support for individual text and batch predictions
- Detailed prediction statistics and analysis
- Export capabilities for prediction results

### 6. REST API (`src/api.py`, `app.py`)
- FastAPI-based REST API for price predictions
- Multiple endpoints for different input formats (text, file, JSON)
- Support for batch processing and file uploads
- Interactive API documentation with Swagger UI and ReDoc
- Comprehensive error handling and logging

## Getting Started

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecommerce-scraper.git
cd ecommerce-scraper
```

### 2. Install dependencies:
```bash
# Install the package in development mode (recommended)
pip install -e .

# Or install only the basic dependencies
pip install -r requirements.txt
```

### 3. Set up pre-commit hooks (optional):
```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run against all files (optional)
pre-commit run --all-files
```

### 4. Run tests:
```bash
pytest src/tests/
```

## System Usage

### Web Scraping

The project allows extracting product data from e-commerce sites using Selenium:

```python
from src.pipeline.orchestrator import process_range

# Extract products from a range of pages
data = process_range(
    root_page="https://example.com/products",
    initial_page=1,
    final_page=5,
    headless=True
)
```

### Training the Price Prediction Model

There are several ways to train a model:

#### Option 1: Using the train.py wrapper

```bash
# Using existing processed data
python train.py --data-path data/processed/product_data.parquet

# Using raw data (will apply cleaning and preprocessing)
python train.py --data-path data/raw/raw_product_data.csv --raw-data --data-format csv

# Saving processed data after preprocessing
python train.py --data-path data/raw/raw_product_data.csv --raw-data --save-processed-data --processed-data-path data/processed/my_processed_data.parquet

# Extracting new data and training
python train.py --scrape --url https://example.com/products --start-page 1 --end-page 10

# With custom model parameters
python train.py --data-path data/processed/product_data.parquet --learning-rate 0.05 --iterations 2000 --depth 8
```

#### Option 2: Using the train_step.py script directly

```bash
# Using existing processed data
python src/train_step.py --data-path data/processed/product_data.parquet

# With other parameters (same options as with train.py)
python src/train_step.py --data-path data/raw/raw_product_data.csv --raw-data --data-format csv
```

#### Option 3: Using the Python API

```python
# For processed data
from src.pipeline.orchestrator import train_model
import pandas as pd

# Load processed data
data = pd.read_parquet("data/processed/product_data.parquet")

# Train model
results = train_model(
    data=data,
    model_path="models",
    model_filename="price_model.pkl",
    vectorizer_filename="vectorizer.pkl",
    learning_rate=0.1,
    iterations=1000,
    depth=6
)

# Access training metrics
print(results["metrics"])
```

### Complete Data Processing Pipeline

For a complete pipeline from raw data to trained model:

```python
from src.pipeline.base import Pipeline
from src.pipeline.steps.cleaning import CleaningStep
from src.pipeline.steps.text_processing import TextProcessingStep
from src.pipeline.steps.training import TrainingStep
import pandas as pd

# Load raw data
raw_data = pd.read_csv("data/raw/raw_product_data.csv")

# Create end-to-end pipeline
pipeline = Pipeline()
pipeline.add_step(CleaningStep()) \
        .add_step(TextProcessingStep()) \
        .add_step(TrainingStep(
            model_path="models",
            model_filename="price_model.pkl",
            vectorizer_filename="vectorizer.pkl",
            learning_rate=0.1,
            iterations=1000,
            depth=6
        ))

# Run the pipeline
results = pipeline.execute(raw_data)

# Access results
print(results["metrics"])
print(f"Model saved at: {results['model_path']}")
```

### Making Price Predictions

There are three main ways to make predictions:

#### Option 1: Using the Command Line (CLI)

```bash
# Prediction from a single text description
python src/predict.py --text "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone"

# Batch prediction from a file
python src/predict.py --data-path data/test/test_products.csv --output-path predictions.csv

# Using a specific model
python src/predict.py --text "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone" --model-path models/custom_model.pkl --vectorizer-path models/custom_vectorizer.pkl
```

#### Option 2: Using the Python API

```python
from src.predict import load_model
import pandas as pd

# Load model (the vectorizer is included in the model)
model = load_model("models/price_model.pkl", "models/vectorizer.pkl")

# Prediction from raw text (uses inference step with text processing)
text = "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone"
data = pd.DataFrame({"description": [text]})
prediction = model.predict(data, step="inference")[0]
print(f"Predicted price: {prediction:.2f}")

# Prediction from already processed text (uses evaluation step)
processed_data = pd.DataFrame({"description_processed": ["samsung galaxy a54 128gb ram black smartphone"]})
prediction = model.predict(processed_data, step="evaluate")[0]
print(f"Predicted price: {prediction:.2f}")
```

#### Option 3: Using the REST API

##### API Setup:

```bash
# Start the API server (Option 1: using src.api directly)
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Start the API server (Option 2: using the app.py wrapper)
python app.py
```

##### Making predictions with the API (using Python):

```python
import requests

# Prediction from text
response = requests.post(
    "http://localhost:8000/predict/text",
    json={"text": "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone"}
)
print(f"Predicted price: {response.json()['predicted_price']}")

# Prediction from multiple texts
response = requests.post(
    "http://localhost:8000/predict/json",
    json={
        "texts": [
            "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone",
            "HP 15.6\" Laptop Intel Core i5 8GB RAM 512GB SSD Windows 11"
        ]
    }
)
print(f"Average predicted price: {response.json()['summary']['mean']}")

# Prediction from a file
response = requests.post(
    "http://localhost:8000/predict/file",
    json={"file_path": "examples/sample_products.csv"}
)
print(f"Prediction summary: {response.json()['summary']}")
```

##### Making predictions with the API (using curl):

```bash
# Prediction from text
curl -X POST "http://localhost:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone"}'

# Prediction from file
curl -X POST "http://localhost:8000/predict/file" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "examples/sample_products.csv"}'

# Prediction from JSON with multiple texts
curl -X POST "http://localhost:8000/predict/json" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Samsung Galaxy A54", "HP 15.6 Intel Core i5 Laptop"]}'
```

## REST API for Price Prediction

The REST API provides endpoints for making price predictions from product descriptions.

### API Endpoints

#### 1. Health check
```
GET /health
```
Returns the current status of the API and the model.

#### 2. Prediction from text
```
POST /predict/text
```
Example body (JSON):
```json
{
  "text": "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone"
}
```

#### 3. Prediction from file
```
POST /predict/file
```
Example body (JSON):
```json
{
  "file_path": "data/products.csv"
}
```

#### 4. Prediction from uploaded file
```
POST /predict/upload
```
Send as `multipart/form-data` with the key `file` and the file to process.

#### 5. Flexible prediction from JSON
```
POST /predict/json
```
Supports three formats:

1. Single text:
```json
{
  "text": "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone"
}
```

2. File path:
```json
{
  "file_path": "data/products.csv"
}
```

3. Multiple texts:
```json
{
  "texts": [
    "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone",
    "HP 15.6\" Laptop Intel Core i5 8GB RAM 512GB SSD Windows 11"
  ]
}
```

### API Documentation

Access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Configuration

The API can be configured using environment variables:

- `MODEL_PATH`: Path to the model file (default: "trained_models/price_model.pkl")
- `VECTORIZER_PATH`: Path to the vectorizer file (default: "trained_models/vectorizer.pkl")

### API Response Formats

#### Individual prediction
```json
{
  "product_description": "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone",
  "predicted_price": 349.99
}
```

#### Batch prediction
```json
{
  "predictions": [
    {
      "product_description": "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone",
      "predicted_price": 349.99
    },
    {
      "product_description": "HP 15.6\" Laptop Intel Core i5 8GB RAM 512GB SSD Windows 11",
      "predicted_price": 699.99
    }
  ],
  "summary": {
    "mean": 524.99,
    "min": 349.99,
    "max": 699.99,
    "count": 2
  }
}
```

## Improved Model Persistence

The project includes improvements in model persistence functionality:

1. **Complete Model State Preservation**: The complete model object is serialized, including:
   - The CatBoost model
   - The text vectorizer
   - Training state information
   - Model parameters

2. **Backward Compatibility**: The system maintains compatibility with older models:
   - Saving vectorizers separately
   - Checking the model state during loading
   - Automatically setting missing attributes

3. **Improved Prediction Pipeline**:
   - The `step` parameter in `predict()` enables different workflows:
     - `step="inference"`: For raw text data requiring preprocessing
     - `step="evaluate"`: For already processed text data

This ensures that the trained model can be reliably loaded and used for predictions in different environments and data formats.

## Troubleshooting Common Issues

### Import Errors

If you encounter errors like `ModuleNotFoundError: No module named 'src'`, make sure to:

1. Have installed the package in development mode:
   ```bash
   pip install -e .
   ```

2. Or use the wrapper scripts from the root of the project:
   ```bash
   python train.py  # Instead of python src/train_step.py
   python app.py    # Instead of uvicorn src.api:app
   ```

### Model Loading Errors

If you encounter errors like `No module named 'models'` when loading serialized models:

1. Make sure the API is properly configured with module aliases:
   ```python
   import src.models
   import src.data_processing
   import src.pipeline
   sys.modules['models'] = src.models
   sys.modules['data_processing'] = src.data_processing
   sys.modules['pipeline'] = src.pipeline
   ```

2. Or consider retraining the model after installing the package in development mode.

## Course Instructor

This project is developed and maintained by Adrian Larcon, Python instructor at Cibertec Peru. The course focuses on practical Python applications in data science and web automation, with emphasis on:
- Clean code practices
- Test-driven development
- Documentation and maintainability
- Solving real-world problems

## Contributing

Students are encouraged to:
1. Fork the repository
2. Create a feature branch
3. Commit their changes
4. Push to the branch
5. Create a Pull Request

## License

This project is part of the Python course curriculum at Cibertec Peru and is licensed under the MIT license.

## AWS Lambda Deployment

You can deploy the price prediction model as an AWS Lambda function that accepts a JSON payload with a product description and returns the predicted price.

### Deployment Steps

1. **Prepare your deployment package**:
   ```bash
   # Make the script executable
   chmod +x prepare_lambda_package.sh

   # Run the packaging script
   ./prepare_lambda_package.sh
   ```

2. **Upload your model files to S3**:
   ```bash
   # Create an S3 bucket (if you don't have one already)
   aws s3 mb s3://your-model-bucket-name

   # Upload your model files
   aws s3 cp trained_models/price_model.pkl s3://your-model-bucket-name/models/price_model.pkl
   aws s3 cp trained_models/vectorizer.pkl s3://your-model-bucket-name/models/vectorizer.pkl
   ```

3. **Create an IAM role** for your Lambda function with the following permissions:
   - `AWSLambdaBasicExecutionRole` (for CloudWatch logs)
   - S3 read access to your model bucket

4. **Create the Lambda function**:
   - In the AWS Console, go to Lambda → Create function
   - Choose "Author from scratch"
   - Name: `price-predictor-lambda`
   - Runtime: Python 3.9 (or higher)
   - Architecture: x86_64
   - Permissions: Use the IAM role created above
   - Upload the `lambda_deployment_package.zip` file
   - Set the handler to: `lambda_function.lambda_handler`

5. **Configure environment variables** in the Lambda function:
   - `MODEL_BUCKET`: Your S3 bucket name (e.g., `your-model-bucket-name`)
   - `MODEL_KEY`: Path to your model in the bucket (e.g., `models/price_model.pkl`)
   - `VECTORIZER_KEY`: Path to your vectorizer in the bucket (e.g., `models/vectorizer.pkl`)

6. **Increase timeout and memory**:
   - Timeout: 30 seconds (the model may take time to load)
   - Memory: 512 MB or higher (depending on your model size)

7. **Create an API Gateway** to expose your Lambda function:
   - In the AWS Console, go to API Gateway → Create API
   - Choose "REST API" → Build
   - Name: `price-predictor-api`
   - Create a resource `/predict` and add a POST method
   - Integration type: Lambda Function
   - Lambda Function: `price-predictor-lambda`
   - Deploy the API to a stage (e.g., "prod")

### Using the Lambda API

Once deployed, you can make predictions using your API:

```bash
# Using curl
curl -X POST \
  https://your-api-id.execute-api.your-region.amazonaws.com/prod/predict \
  -H 'Content-Type: application/json' \
  -d '{"product_description": "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone"}'
```

```python
# Using Python requests
import requests
import json

url = "https://your-api-id.execute-api.your-region.amazonaws.com/prod/predict"
payload = {"product_description": "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)
result = response.json()
print(f"Predicted price: {result['predicted_price']}")
```

The response will be in this format:

```json
{
  "product_description": "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone",
  "predicted_price": 349.99
}
```

### Monitoring and Troubleshooting

- Check CloudWatch Logs for detailed Lambda execution logs
- Monitor Lambda metrics (invocations, errors, duration)
- Use AWS X-Ray for in-depth tracing (optional)

### Cost Optimization

- The Lambda function is configured to cache the model in memory between invocations
- Cold starts may be slower due to model loading from S3
- Consider using Provisioned Concurrency for consistent performance
- Use AWS Lambda Power Tuning to find the optimal memory configuration
