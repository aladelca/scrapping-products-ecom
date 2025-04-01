#!/usr/bin/env python3
"""Test script for the Lambda function locally.

This script mocks the AWS Lambda environment by setting up necessary
paths and simulating the Lambda event structure. It helps test the
Lambda function locally before deployment.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add the current directory to the path so lambda_function can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Determine paths to the model files
MODELS_DIR = Path("trained_models")
if not MODELS_DIR.exists():
    MODELS_DIR = Path("models")

MODEL_PATH = MODELS_DIR / "price_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

if not MODEL_PATH.exists():
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

if not VECTORIZER_PATH.exists():
    print(
        f"Warning: Vectorizer file not found at {VECTORIZER_PATH}. Continuing without it."
    )

# Set environment variables to use local files instead of S3
os.environ["MODEL_BUCKET"] = ""
os.environ["MODEL_KEY"] = str(MODEL_PATH)
os.environ["VECTORIZER_KEY"] = str(VECTORIZER_PATH)


# Mock the S3 download by creating a function that just copies the files
def mock_download(lambda_module):
    original_download = lambda_module.download_from_s3

    def mocked_download(bucket, key, local_path):
        print(f"Mocking S3 download: copying {key} to {local_path}")
        # Just copy the file locally
        import shutil

        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy2(key, local_path)
            return True
        except Exception as e:
            print(f"Error mocking download: {e}")
            return False

    lambda_module.download_from_s3 = mocked_download


import lambda_function

# Now import and patch the lambda function
from lambda_function import lambda_handler

mock_download(lambda_function)

# Test examples
test_examples = [
    "Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone",
    'HP 15.6" Laptop Intel Core i5 8GB RAM 512GB SSD Windows 11',
    "Apple iPad Pro 11-inch M2 chip 256GB WiFi",
]

# Create test events for direct invocation and API Gateway format
for i, example in enumerate(test_examples):
    print(f"\n{'='*50}")
    print(f"Test {i+1}: {example}")

    # Test direct invocation
    direct_event = {"product_description": example}

    # Simulate API Gateway event
    api_gateway_event = {
        "body": json.dumps(direct_event),
        "headers": {"Content-Type": "application/json"},
        "requestContext": {"http": {"method": "POST", "path": "/predict"}},
        "isBase64Encoded": False,
    }

    # Time the invocation
    start_time = time.time()

    # Test direct invocation first
    print("\nTesting direct invocation:")
    result = lambda_handler(direct_event, {})
    print(f"Predicted price: {result['predicted_price']}")

    # Then test API Gateway format
    print("\nTesting API Gateway format:")
    api_result = lambda_handler(api_gateway_event, {})
    print(f"Status code: {api_result['statusCode']}")
    print(f"Response body: {api_result['body']}")

    elapsed_time = time.time() - start_time
    print(f"\nInvocation took: {elapsed_time:.2f} seconds")

print(f"\n{'='*50}")
print("All tests completed!")
print(f"{'='*50}")
