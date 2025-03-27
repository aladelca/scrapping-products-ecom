"""Pipeline orchestrator module."""

import os

import pandas as pd

from src.pipeline.base import Pipeline
from src.pipeline.steps.cleaning import CleaningStep
from src.pipeline.steps.scraping import ScrapingStep
from src.pipeline.steps.text_processing import TextProcessingStep
from src.pipeline.steps.training import TrainingStep


def create_pipeline(url: str, headless: bool = True) -> Pipeline:
    """
    Create a pipeline for scraping and processing product data.

    Args:
        url (str): URL to scrape
        headless (bool): Whether to run browser in headless mode

    Returns:
        Pipeline: Configured pipeline
    """
    pipeline = Pipeline()

    # Add pipeline steps in order
    pipeline.add_step(ScrapingStep(url=url, headless=headless)).add_step(
        CleaningStep()
    ).add_step(TextProcessingStep())

    return pipeline


def create_training_pipeline(
    model_path: str = "trained_models",
    model_filename: str = "price_model.pkl",
    vectorizer_filename: str = "vectorizer.pkl",
    target_column: str = "offer_price",
    text_column: str = "description_processed",
    learning_rate: float = 0.1,
    iterations: int = 1000,
    depth: int = 6,
) -> Pipeline:
    """
    Create a pipeline for training the price prediction model.

    Args:
        model_path (str): Directory path to save the model
        model_filename (str): Filename for the saved model
        vectorizer_filename (str): Filename for the saved vectorizer
        target_column (str): Column name for the target variable
        text_column (str): Column name for the processed text data
        learning_rate (float): Learning rate for CatBoost
        iterations (int): Number of iterations for CatBoost
        depth (int): Tree depth for CatBoost

    Returns:
        Pipeline: Training pipeline
    """
    pipeline = Pipeline()

    pipeline.add_step(
        TrainingStep(
            model_path=model_path,
            model_filename=model_filename,
            vectorizer_filename=vectorizer_filename,
            target_column=target_column,
            text_column=text_column,
            learning_rate=learning_rate,
            iterations=iterations,
            depth=depth,
        )
    )

    return pipeline


def process_range(
    root_page: str,
    initial_page: int,
    final_page: int,
    headless: bool = True,
) -> pd.DataFrame:
    """
    Process a range of pages.

    Args:
        root_page (str): Base URL
        initial_page (int): Starting page number
        final_page (int): Ending page number
        headless (bool): Whether to run browser in headless mode

    Returns:
        pd.DataFrame: Processed data from all pages
    """
    all_data = []

    for page in range(initial_page, final_page + 1):
        url = f"{root_page}/{page}"
        pipeline = create_pipeline(url=url, headless=headless)
        page_data = pipeline.execute()
        all_data.append(page_data)

    return pd.concat(all_data, ignore_index=True)


def train_model(
    data: pd.DataFrame,
    model_path: str = "trained_models",
    model_filename: str = "price_model.pkl",
    vectorizer_filename: str = "vectorizer.pkl",
    target_column: str = "offer_price",
    text_column: str = "description_processed",
    learning_rate: float = 0.1,
    iterations: int = 1000,
    depth: int = 6,
) -> dict:
    """
    Train a price prediction model on the provided data.

    Args:
        data (pd.DataFrame): DataFrame containing the training data
        model_path (str): Directory path to save the model
        model_filename (str): Filename for the saved model
        vectorizer_filename (str): Filename for the saved vectorizer
        target_column (str): Column name for the target variable
        text_column (str): Column name for the processed text data
        learning_rate (float): Learning rate for CatBoost
        iterations (int): Number of iterations for CatBoost
        depth (int): Tree depth for CatBoost

    Returns:
        dict: Training results including metrics and model paths
    """
    # Create the training pipeline
    pipeline = create_training_pipeline(
        model_path=model_path,
        model_filename=model_filename,
        vectorizer_filename=vectorizer_filename,
        target_column=target_column,
        text_column=text_column,
        learning_rate=learning_rate,
        iterations=iterations,
        depth=depth,
    )

    # Execute the training pipeline
    results = pipeline.execute(data)

    return results


def scrape_and_train(
    root_page: str,
    initial_page: int,
    final_page: int,
    model_path: str = "models",
    model_filename: str = "price_model.pkl",
    vectorizer_filename: str = "vectorizer.pkl",
    headless: bool = True,
    target_column: str = "price",
    text_column: str = "description_processed",
    learning_rate: float = 0.1,
    iterations: int = 1000,
    depth: int = 6,
) -> dict:
    """
    Complete pipeline to scrape data and train a price prediction model.

    Args:
        root_page (str): Base URL
        initial_page (int): Starting page number
        final_page (int): Ending page number
        model_path (str): Directory path to save the model
        model_filename (str): Filename for the saved model
        vectorizer_filename (str): Filename for the saved vectorizer
        headless (bool): Whether to run browser in headless mode
        target_column (str): Column name for the target variable
        text_column (str): Column name for the processed text data
        learning_rate (float): Learning rate for CatBoost
        iterations (int): Number of iterations for CatBoost
        depth (int): Tree depth for CatBoost

    Returns:
        dict: Training results including metrics and model paths
    """
    # Scrape and process data
    processed_data = process_range(
        root_page=root_page,
        initial_page=initial_page,
        final_page=final_page,
        headless=headless,
    )

    # Train model on the processed data
    results = train_model(
        data=processed_data,
        model_path=model_path,
        model_filename=model_filename,
        vectorizer_filename=vectorizer_filename,
        target_column=target_column,
        text_column=text_column,
        learning_rate=learning_rate,
        iterations=iterations,
        depth=depth,
    )

    return results
