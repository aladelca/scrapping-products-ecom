"""AWS Lambda function for price prediction.

This module provides an AWS Lambda handler that accepts a product description
and returns a predicted price using a trained CatBoost model.
"""

import json
import logging
import os
import sys
import traceback

import boto3
import joblib
import pandas as pd
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add model path to system path if needed
sys.path.append(os.getcwd())

# Initialize global variables for NLP tools
nltk_available = False
spacy_available = False
nlp = None

# Try to import and setup NLTK
try:
    import nltk

    # Download NLTK resources to /tmp which is writable in Lambda
    nltk_data_dir = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    # Check if NLTK data is available
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
        nltk_available = True
        logger.info("NLTK data is available")
    except LookupError:
        # Attempt to download only if not already available
        try:
            logger.info("Downloading NLTK resources...")
            nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
            nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)
            nltk.download("wordnet", download_dir=nltk_data_dir, quiet=True)
            nltk_available = True
            logger.info("NLTK resources downloaded successfully")
        except Exception as e:
            logger.warning(f"Error downloading NLTK resources: {str(e)}")
            logger.warning(traceback.format_exc())
except ImportError:
    logger.warning("NLTK is not available, continuing without it")

# Try to import and setup spaCy with proper error handling
try:
    import spacy

    # Try to load spaCy model with better error handling
    try:
        logger.info("Loading spaCy model...")
        nlp = spacy.load("es_core_news_sm")
        spacy_available = True
        logger.info(f"spaCy model loaded successfully: {nlp.meta['name']}")
    except Exception as e:
        logger.warning(f"Error loading spaCy model: {str(e)}")
        logger.warning("Will continue without spaCy model")
except ImportError as e:
    logger.warning(f"spaCy is not available: {str(e)}, continuing without it")

# Import price predictor model
try:
    logger.info("Importing CatBoostPricePredictor...")
    from src.models.price_predictor import CatBoostPricePredictor

    logger.info("Successfully imported CatBoostPricePredictor")
except ImportError as e:
    logger.error(f"Failed to import CatBoostPricePredictor: {e}")
    logger.error(traceback.format_exc())
    raise

# Initialize S3 client
s3 = boto3.client("s3")

# Constants
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "your-model-bucket-name")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/price_model.pkl")
VECTORIZER_KEY = os.environ.get("VECTORIZER_KEY", "models/vectorizer.pkl")
LOCAL_MODEL_PATH = "/tmp/price_model.pkl"
LOCAL_VECTORIZER_PATH = "/tmp/vectorizer.pkl"

# Global variable to store the model once loaded
model = None


def download_from_s3(bucket, key, local_path):
    """Download a file from S3.

    Args:
        bucket (str): S3 bucket name in AWS
        key (str): S3 object key
        local_path (str): Local path to save the file

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading {key} from {bucket} to {local_path}")
        s3.download_file(bucket, key, local_path)
        return True
    except ClientError as e:
        logger.error(f"Failed to download {key}: {e}")
        return False


def load_model():
    """Load the price prediction model.

    Returns:
        CatBoostPricePredictor: Loaded model or None if loading failed
    """
    global model

    # Return existing model if already loaded
    if model is not None:
        return model

    try:
        # Download model and vectorizer from S3
        if not download_from_s3(MODEL_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH):
            return None

        if not download_from_s3(MODEL_BUCKET, VECTORIZER_KEY, LOCAL_VECTORIZER_PATH):
            logger.warning("Vectorizer not found. Continuing with model only.")

        # Load the model
        logger.info(f"Loading model from {LOCAL_MODEL_PATH}")
        model = joblib.load(LOCAL_MODEL_PATH)

        # Ensure it's a CatBoostPricePredictor instance
        if not isinstance(model, CatBoostPricePredictor):
            logger.error(f"Expected CatBoostPricePredictor, got {type(model).__name__}")
            return None

        # For backward compatibility with older models
        if not hasattr(model, "is_trained") or not model.is_trained:
            model.is_trained = True

        # Load vectorizer for backward compatibility
        if os.path.exists(LOCAL_VECTORIZER_PATH):
            vectorizer = joblib.load(LOCAL_VECTORIZER_PATH)
            model.set_vectorizer(vectorizer)

        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def predict_price(text):
    """Predict price for a product description.

    Args:
        text (str): Product description

    Returns:
        float: Predicted price or None if prediction failed
    """
    try:
        # Create a DataFrame with the text
        data = pd.DataFrame({"description": [text]})

        # Make prediction
        prediction = model.predict(data, step="inference")
        return float(prediction[0])

    except Exception as e:
        logger.error(f"Error predicting price: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def lambda_handler(event, context):
    """AWS Lambda handler function.

    Args:
        event (dict): AWS Lambda event containing the request data
        context (object): AWS Lambda context

    Returns:
        dict: Response containing the predicted price or error message
    """
    logger.info(f"Received event: {json.dumps(event)}")

    # Process API Gateway event
    if "body" in event:
        try:
            body = json.loads(event["body"])
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid JSON in request body"}),
            }
    else:
        body = event

    # Validate input
    if "product_description" not in body:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {"error": "Missing required field: product_description"}
            ),
        }

    product_description = body["product_description"]

    # Load model if not already loaded
    global model
    if model is None:
        model = load_model()
        if model is None:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Failed to load model"}),
            }

    # Make prediction
    predicted_price = predict_price(product_description)
    if predicted_price is None:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Failed to make prediction"}),
        }

    # Return response
    response = {
        "product_description": product_description,
        "predicted_price": predicted_price,
    }

    # Format response for API Gateway
    if "body" in event:
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response),
        }

    # Direct invocation response
    return response
