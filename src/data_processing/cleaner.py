"""Data cleaning module."""

from typing import Optional

import pandas as pd


class DataCleaner:
    """Class for cleaning scraped data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner.

        Args:
            df (pd.DataFrame): DataFrame to clean
        """
        self.df = df.copy()

    def _filter_null(self, column: str) -> pd.Series:
        """
        Filter null values from a column.
        """
        print(self.df[self.df[column].notna()].shape)
        return self.df[self.df[column].notna()]

    def clean_price(self, price: str) -> Optional[float]:
        """
        Clean price string to get numerical value.

        Args:
            price (str): Price string to clean

        Returns:
            Optional[float]: Cleaned price value
        """
        if not isinstance(price, str) or pd.isna(price):
            return None

        # Remove currency symbol and whitespace
        price = price.replace("S/", "").strip()

        # Split by comma if exists and join with decimal point
        if "," in price:
            parts = price.split(",")
            price = f"{parts[0].strip()}.{parts[1].strip()}"

        # Remove any remaining non-numeric characters except decimal point
        price = "".join(char for char in price if char.isdigit() or char == ".")

        try:
            return float(price)
        except (ValueError, TypeError):
            return None

    def clean_text(self, text: str) -> Optional[str]:
        """
        Clean text by removing special characters and converting to lowercase.

        Args:
            text (str): Text to clean

        Returns:
            Optional[str]: Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return None

        # Remove special characters and convert to lowercase
        chars_to_remove = ["|", "\\", "/", ",", "."]
        for char in chars_to_remove:
            text = text.replace(char, " ")

        return text.lower().strip()

    def clean_prices(self) -> "DataCleaner":
        """Clean price columns."""
        price_columns = ["offer_price", "original_price"]
        for column in price_columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].apply(self.clean_price)
        print(self.df.head())
        self.df = self._filter_null("offer_price")

        return self

    def clean_descriptions(self) -> "DataCleaner":
        """Clean description column."""
        if "description" in self.df.columns:
            self.df["description"] = self.df["description"].apply(self.clean_text)
            return self
        self.df = self._filter_null("description")
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        return self.df
