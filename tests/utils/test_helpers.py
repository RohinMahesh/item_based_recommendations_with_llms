import re
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from item_based_recommendations_with_llms.utils.helpers import CleanData

sample_df = pd.DataFrame(
    {
        "description": [
            "ProductName - Some product description with <html> tags",
            "AnotherProduct - Another description with <html> tags",
        ]
    }
)

sample_cleaned_df = pd.DataFrame(
    {
        "description": [
            "Some product description with tags",
            "Another description with tags",
        ]
    }
)

sample_feature_space = np.random.random((2, 128))


@pytest.mark.parametrize(
    "input_df, expected_output",
    [
        (sample_df.copy(), sample_cleaned_df["description"].tolist()),
    ],
)
def test_clean(input_df, expected_output):
    cleaner = CleanData(df=input_df, html_pattern=re.compile(r"<.*?>"))
    cleaned_data = cleaner.clean()
    assert cleaned_data == expected_output


@pytest.mark.parametrize(
    "feature_space",
    [
        sample_feature_space,
    ],
)
def test_create_index(feature_space):
    with TemporaryDirectory() as temp_dir:
        with patch(
            "item_based_recommendations_with_llms.utils.helpers.faiss.write_index"
        ) as mock_write_index:
            with patch(
                "item_based_recommendations_with_llms.utils.file_paths.INDEX_PATH",
                temp_dir + "/index.faiss",
            ):
                create_index(feature_space)

                mock_write_index.assert_called_once()


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("ProductName - Some product description", "Some product description"),
        ("AnotherProduct - Another description", "Another description"),
        ("ProductName -", ""),
        ("ProductName Only", ""),
    ],
)
def test_remove_product_name(input_text, expected_output):
    processed_text = CleanData.remove_product_name(input_text)
    assert processed_text == expected_output
