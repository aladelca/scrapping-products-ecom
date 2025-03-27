#!/usr/bin/env python3
"""Examples for using the Price Prediction API."""

import json
import os

import pandas as pd
import requests

# API base URL - change this to match your deployment
API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Health check response: {response.status_code}")
    print(response.json())
    print("\n")


def test_single_text_prediction():
    """Test predicting price for a single product description."""

    print("Testing single text prediction...")

    # Example 1: Using the /predict/text endpoint with a TextPredictionRequest
    text = "Smartphone Samsung Galaxy A54 128GB 8GB RAM 5G Negro"
    response = requests.post(f"{API_BASE_URL}/predict/text", json={"text": text})

    print(f"Response status: {response.status_code}")
    print(f"Prediction result:")
    print(json.dumps(response.json(), indent=2))
    print("\n")

    # Example 2: Using the /predict/json endpoint with a text field
    response = requests.post(f"{API_BASE_URL}/predict/json", json={"text": text})

    print(f"JSON endpoint response status: {response.status_code}")
    print(f"Prediction result:")
    print(json.dumps(response.json(), indent=2))
    print("\n")


def test_batch_text_prediction():
    """Test predicting prices for multiple product descriptions."""

    print("Testing batch text prediction...")

    # Multiple texts
    texts = [
        "Smartphone Samsung Galaxy A54 128GB 8GB RAM 5G Negro",
        'Laptop HP 15.6" Intel Core i5 8GB RAM 512GB SSD Windows 11',
        "Audífonos inalámbricos Sony WH-1000XM4 con cancelación de ruido",
        'Televisor LG Smart TV 55" 4K UHD 55UN7310 WebOS',
    ]

    # Use the /predict/json endpoint with a texts array
    response = requests.post(f"{API_BASE_URL}/predict/json", json={"texts": texts})

    print(f"Response status: {response.status_code}")
    print(f"Batch prediction summary:")
    print(json.dumps(response.json()["summary"], indent=2))
    print(f"First prediction:")
    print(json.dumps(response.json()["predictions"][0], indent=2))
    print("\n")


def create_sample_csv():
    """Create a sample CSV file for batch prediction."""

    data = {
        "description": [
            "Smartphone Samsung Galaxy A54 128GB 8GB RAM 5G Negro",
            'Laptop HP 15.6" Intel Core i5 8GB RAM 512GB SSD Windows 11',
            "Audífonos inalámbricos Sony WH-1000XM4 con cancelación de ruido",
            'Televisor LG Smart TV 55" 4K UHD 55UN7310 WebOS',
        ]
    }

    df = pd.DataFrame(data)

    # Create directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)

    # Save to CSV
    csv_path = "examples/sample_products.csv"
    df.to_csv(csv_path, index=False)

    print(f"Sample CSV created at: {csv_path}")
    return csv_path


def test_file_prediction():
    """Test predicting prices from a file."""

    print("Testing file prediction...")

    # Create a sample CSV file
    csv_path = create_sample_csv()

    # Use the /predict/file endpoint
    response = requests.post(
        f"{API_BASE_URL}/predict/file", json={"file_path": csv_path}
    )

    print(f"Response status: {response.status_code}")
    if response.status_code == 200:
        print(f"Batch prediction summary:")
        print(json.dumps(response.json()["summary"], indent=2))
        print(f"Number of predictions: {len(response.json()['predictions'])}")
    else:
        print(f"Error: {response.json()}")
    print("\n")

    # Alternative: Use the /predict/json endpoint with a file_path
    response = requests.post(
        f"{API_BASE_URL}/predict/json", json={"file_path": csv_path}
    )

    print(f"JSON endpoint with file path - Response status: {response.status_code}")
    if response.status_code == 200:
        print(f"Batch prediction summary:")
        print(json.dumps(response.json()["summary"], indent=2))
    else:
        print(f"Error: {response.json()}")
    print("\n")


def test_file_upload():
    """Test uploading a file for prediction."""

    print("Testing file upload prediction...")

    # Create a sample CSV file
    csv_path = create_sample_csv()

    # Open the file and send it as multipart/form-data
    with open(csv_path, "rb") as f:
        response = requests.post(
            f"{API_BASE_URL}/predict/upload",
            files={"file": ("sample_products.csv", f, "text/csv")},
        )

    print(f"Response status: {response.status_code}")
    if response.status_code == 200:
        print(f"Batch prediction summary:")
        print(json.dumps(response.json()["summary"], indent=2))
        print(f"Number of predictions: {len(response.json()['predictions'])}")
    else:
        print(f"Error: {response.json()}")
    print("\n")


if __name__ == "__main__":
    # Make sure the API is running before executing these examples
    print("=" * 50)
    print("Price Prediction API Examples")
    print("=" * 50)
    print("Make sure the API is running with: uvicorn src.api:app --reload")
    print("=" * 50)

    try:
        test_health_check()
        test_single_text_prediction()
        test_batch_text_prediction()
        test_file_prediction()
        test_file_upload()

        print("All tests completed!")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API.")
        print("Make sure the API is running with: uvicorn src.api:app --reload")
