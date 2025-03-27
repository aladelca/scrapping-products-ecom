#!/usr/bin/env python3
"""Script for training the price prediction model."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Añadir el directorio raíz al path para poder importar desde src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.base import Pipeline
from src.pipeline.orchestrator import scrape_and_train, train_model
from src.pipeline.steps.cleaning import CleaningStep
from src.pipeline.steps.text_processing import TextProcessingStep
from src.pipeline.steps.training import TrainingStep


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a price prediction model.")

    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--data-path", type=str, help="Path to a CSV file containing the training data"
    )
    input_group.add_argument(
        "--raw-data",
        action="store_true",
        help="Indicate that the input data is raw and needs preprocessing",
    )
    input_group.add_argument(
        "--data-format",
        type=str,
        choices=["csv", "parquet"],
        default="parquet",
        help="Format of the input data (csv or parquet)",
    )
    input_group.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape data instead of loading from a file",
    )
    input_group.add_argument(
        "--url", type=str, help="Base URL for scraping (required if --scrape is used)"
    )
    input_group.add_argument(
        "--start-page", type=int, default=1, help="Starting page number for scraping"
    )
    input_group.add_argument(
        "--end-page", type=int, default=5, help="Ending page number for scraping"
    )
    input_group.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode for scraping",
    )

    # Model options
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate for CatBoost"
    )
    model_group.add_argument(
        "--iterations", type=int, default=1000, help="Number of iterations for CatBoost"
    )
    model_group.add_argument(
        "--depth", type=int, default=6, help="Tree depth for CatBoost"
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--model-path",
        type=str,
        default="trained_models",
        help="Directory path to save the model",
    )
    output_group.add_argument(
        "--model-filename",
        type=str,
        default="price_model.pkl",
        help="Filename for the saved model",
    )
    output_group.add_argument(
        "--vectorizer-filename",
        type=str,
        default="vectorizer.pkl",
        help="Filename for the saved vectorizer",
    )
    output_group.add_argument(
        "--save-processed-data",
        action="store_true",
        help="Save processed data after preprocessing",
    )
    output_group.add_argument(
        "--processed-data-path",
        type=str,
        default="data/processed/processed_data.parquet",
        help="Path to save processed data",
    )

    return parser.parse_args()


def save_metrics(metrics, path, filename="metrics.json"):
    """
    Save training metrics to a JSON file.

    Args:
        metrics (dict): Dictionary containing metrics
        path (str): Directory path to save the metrics
        filename (str): Filename for the metrics file
    """
    logging.info("Saving training metrics to JSON file...")
    os.makedirs(path, exist_ok=True)
    metrics_file = os.path.join(path, filename)

    # Add timestamp
    metrics["timestamp"] = datetime.now().isoformat()

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Metrics saved to {metrics_file}")


def process_raw_data(data):
    """
    Process raw data through cleaning and text processing steps.

    Args:
        data (pd.DataFrame): Raw input data

    Returns:
        pd.DataFrame: Processed data ready for training
    """
    logging.info("Starting raw data processing pipeline...")
    start_time = time.time()

    # Log initial data stats
    logging.info(f"Initial data shape: {data.shape}")
    logging.info(f"Columns: {list(data.columns)}")
    for col in data.columns:
        null_count = data[col].isnull().sum()
        logging.info(
            f"Column '{col}': {null_count} null values ({null_count/len(data)*100:.2f}%)"
        )

    # Create a pipeline for data preprocessing
    logging.info(
        "Creating preprocessing pipeline with CleaningStep and TextProcessingStep..."
    )
    preprocessing_pipeline = Pipeline()

    # Add cleaning step
    logging.info("Adding CleaningStep to pipeline...")
    cleaning_step = CleaningStep()
    preprocessing_pipeline.add_step(cleaning_step)

    # Add text processing step
    logging.info("Adding TextProcessingStep to pipeline...")
    text_processing_step = TextProcessingStep()
    preprocessing_pipeline.add_step(text_processing_step)

    # Execute the preprocessing pipeline
    logging.info("Executing preprocessing pipeline...")
    processed_data = preprocessing_pipeline.execute(data)

    # Log processed data stats
    logging.info(f"Processed data shape: {processed_data.shape}")
    logging.info(f"Columns after processing: {list(processed_data.columns)}")
    for col in processed_data.columns:
        null_count = processed_data[col].isnull().sum()
        logging.info(
            f"Column '{col}' after processing: {null_count} null values ({null_count/len(processed_data)*100:.2f}%)"
        )

    elapsed_time = time.time() - start_time
    logging.info(f"Raw data processing completed in {elapsed_time:.2f} seconds")

    return processed_data


def main():
    """Execute the main training workflow."""
    # Set up logging
    setup_logging()

    logging.info("=" * 80)
    logging.info("STARTING PRICE PREDICTION MODEL TRAINING")
    logging.info("=" * 80)

    # Start timing
    start_time = time.time()

    # Parse command line arguments
    args = parse_args()
    logging.info(f"Command line arguments: {vars(args)}")

    # Get training data
    if args.scrape:
        if not args.url:
            logging.error("URL is required when using --scrape")
            sys.exit(1)

        logging.info("=" * 50)
        logging.info(
            f"SCRAPING DATA from {args.url} (pages {args.start_page}-{args.end_page})"
        )
        logging.info("=" * 50)

        scrape_start_time = time.time()
        results = scrape_and_train(
            root_page=args.url,
            initial_page=args.start_page,
            final_page=args.end_page,
            model_path=args.model_path,
            model_filename=args.model_filename,
            vectorizer_filename=args.vectorizer_filename,
            headless=args.headless,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            depth=args.depth,
        )
        scrape_elapsed_time = time.time() - scrape_start_time

        logging.info(
            f"Scraping and training completed in {scrape_elapsed_time:.2f} seconds"
        )
        logging.info(f"Model saved to {results['model_path']}")

    elif args.data_path:
        # Load data from file
        data_path = Path(args.data_path)
        if not data_path.exists():
            logging.error(f"Data file not found: {data_path}")
            sys.exit(1)

        logging.info("=" * 50)
        logging.info(f"LOADING DATA from {data_path}")
        logging.info("=" * 50)

        load_start_time = time.time()
        # Load data based on the format
        if args.data_format == "csv":
            logging.info(f"Reading CSV file: {data_path}")
            data = pd.read_csv(data_path)
        else:  # parquet
            logging.info(f"Reading Parquet file: {data_path}")
            data = pd.read_parquet(data_path)

        load_elapsed_time = time.time() - load_start_time
        logging.info(f"Data loaded in {load_elapsed_time:.2f} seconds")
        logging.info(f"Data shape: {data.shape}")

        # Process raw data if needed
        if args.raw_data:
            logging.info("=" * 50)
            logging.info("PREPROCESSING RAW DATA")
            logging.info("=" * 50)

            data = process_raw_data(data)

            # Save processed data if requested
            if args.save_processed_data:
                processed_path = Path(args.processed_data_path)
                os.makedirs(processed_path.parent, exist_ok=True)

                logging.info(f"Saving processed data to {processed_path}")
                save_start_time = time.time()
                data.to_parquet(processed_path)
                save_elapsed_time = time.time() - save_start_time
                logging.info(f"Processed data saved in {save_elapsed_time:.2f} seconds")

        # Train model with processed data
        logging.info("=" * 50)
        logging.info("TRAINING MODEL")
        logging.info("=" * 50)

        logging.info("Model parameters:")
        logging.info(f"  - Learning rate: {args.learning_rate}")
        logging.info(f"  - Iterations: {args.iterations}")
        logging.info(f"  - Depth: {args.depth}")

        train_start_time = time.time()
        results = train_model(
            data=data,
            model_path=args.model_path,
            model_filename=args.model_filename,
            vectorizer_filename=args.vectorizer_filename,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            depth=args.depth,
        )
        train_elapsed_time = time.time() - train_start_time

        logging.info(f"Model training completed in {train_elapsed_time:.2f} seconds")
        logging.info(
            f"Model saved to {os.path.join(args.model_path, args.model_filename)}"
        )
        logging.info(
            f"Vectorizer saved to {os.path.join(args.model_path, args.vectorizer_filename)}"
        )

    else:
        logging.error("Either --data-path or --scrape must be specified")
        sys.exit(1)

    # Save metrics
    if "metrics" in results:
        logging.info("=" * 50)
        logging.info("TRAINING METRICS")
        logging.info("=" * 50)

        for metric_name, metric_value in results["metrics"].items():
            logging.info(f"{metric_name}: {metric_value}")

        save_metrics(results["metrics"], args.model_path)

    # Log total execution time
    total_elapsed_time = time.time() - start_time
    logging.info("=" * 80)
    logging.info(f"TOTAL EXECUTION TIME: {total_elapsed_time:.2f} seconds")
    logging.info("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
