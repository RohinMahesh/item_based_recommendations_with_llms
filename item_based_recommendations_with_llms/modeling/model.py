from dataclasses import dataclass
from typing import List

import faiss

from item_based_recommendations_with_llms.data_preparation.hfembeddings import (
    HFEmbeddings,
)
from item_based_recommendations_with_llms.utils.constants import HIDDEN_SIZE
from item_based_recommendations_with_llms.utils.file_paths import INDEX_PATH


@dataclass
class ItemBasedRecs:
    """
    Performs item-based recommendations using product embeddings

    :param index_path: path to FAISS index file
    :param reference_id: id number for item-based recommendations
    """

    index_path: str = INDEX_PATH
    reference_id: int

    @staticmethod
    def _search_index(
        input_data: List[str],
        index_file: faiss.IndexFlatL2,
        k: int = 1,
        hidden_size: int = HIDDEN_SIZE,
    ) -> List[int]:
        """
        Searches FAISS index

        :param input_data: list containing text for feature extraction
        :param index_file: FAISS index file
        :param k: optional number of similar vectors for search,
            defaults to 1
        :param hidden_size: optional hidden size for tokenizer,
            defaults to HIDDEN_SIZE
        :returns output: list of similar product ids
        """
        # Get embeddings
        embedding = HFEmbeddings(data=input_data).create_embeddings()

        # Reshape input to match hidden state of upstream tokenizer
        embedding = embedding.reshape(1, hidden_size)

        # Search index
        distances, indices = index_file.search(embedding, k=k)

        # Map indices to product number
        output = [x + 1 for x in indices.tolist()[0]]

        return output

    def recommend(self) -> List[int]:
        """
        Make recommendations based on reference product id

        :returns recommendations: product ids for most similar products
        """
        # Load FAISS file
        index = faiss.load_index(self.index_path)

        # Get recommendations
        recommendations = self._search_index(
            input_data=[self.reference_id], index_file=index
        )

        return recommendations
