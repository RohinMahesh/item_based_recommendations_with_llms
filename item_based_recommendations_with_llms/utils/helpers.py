import re
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
import pandas as pd
from item_based_recommendations_with_llms.utils.constants import (
    HTML_PATTERN,
    TEXT_COLUMN,
)
from item_based_recommendations_with_llms.utils.file_paths import INDEX_PATH


@dataclass
class CleanData:
    """
    Basic cleaning of product description data

    :param df: dataframe containing text for feature extraction
    :param html_pattern: regular expression pattern
    """

    df: pd.DataFrame
    html_pattern: re.Pattern = HTML_PATTERN

    @staticmethod
    def remove_product_name(data) -> str:
        """
        Removes product name from text

        :param data: input sequence
        :returns processed: processed sequence
        """
        # Split in hyphen
        processed = data.split(" - ")

        # Remove product name
        processed.pop(0)

        # Join sequence
        processed = " ".join(processed)
        return processed

    def clean(self) -> List[str]:
        """
        Performs data cleaning

        :returns: list of cleaned data
        """
        # Remove html content
        self.df[TEXT_COLUMN] = list(
            map(lambda x: self.html_pattern.sub("", x), self.df[TEXT_COLUMN])
        )
        # Remove product name
        self.df[TEXT_COLUMN] = list(
            map(lambda x: __class__.remove_product_name(x), self.df[TEXT_COLUMN])
        )
        # Remove extra spaces
        self.df[TEXT_COLUMN] = list(
            map(lambda x: " ".join(x.split()), self.df[TEXT_COLUMN])
        )
        return self.df[TEXT_COLUMN].tolist()


def create_index(feature_space: np.ndarray) -> None:
    """
    Creates FAISS index

    :param feature_space: numpy array containing text for feature extraction
    :returns None
    """
    # Build the index using embedding dimension of OpenAI model
    index = faiss.IndexFlatL2(feature_space.shape[1])

    # Add vectors to index
    index.add(feature_space)

    # Save index
    faiss.write_index(index, INDEX_PATH)
