# Item-Based Recommendations with LLMs

![Build Status](https://github.com/RohinMahesh/item_based_recommendations_with_llms/actions/workflows/ci.yml/badge.svg)

# Background

With the introduction of BERT in 2017 along with the release of ChatGPT in 2022, the interest for Large Language Models (LLM) based solutions is growing aggressively across various domains. While the Question and Answering (Q&A) based offerings from companies like OpenAI are very powerful, another area of opportunity would be to leverage the embeddings from LLMs to serve as a representative function in enhancing downstream recommender systems.

Prior to LLMs, generating domain specific embeddings can be a difficult task as it requires a considerable amount of high-quality labeled data. In practice, many organizations have invested in creating abstractions to bring their text to a level that best represents their domain. While these methods are vital irrespective of the representative function, the challenge has always been around generating a rich feature space for downstream applications.

Traditionally, organizations have leveraged methods such as Bag of Words (BOW) of Term Frequency Inverse Document Frequency (TF-IDF) to build their feature space. Other more mature organizations have opted to represent their feature space by leveraging open-source embeddings like GloVe, Fasttext or Word2Vec, but have often faced problems of poor performance on domain specific data and inadequate sample size for fine-tuning.

With the release of BERT, along with the release of HuggingFace, organizations have access to readily available pre-trained models trained on a corpus of data much larger than what they may have access to. This allows them to leverage the embeddings from these models and fine-tune them to create a much richer feature space than the traditional methods used in the past.

While these embeddings are powerful, they are often quite large and methods like brute-force searches for item-based recommender systems will often require a considerable compute and may not be scalable in practice. 

In this project, we create a item-based recommender system by leveraging embeddings from BERT based models from HuggingFace to serve as our representative function.

Given the issue with brute-force searches on embeddings, we leverage FAISS to provide scalable searches and recommendations.

# Notes

1. Another area of rapid development is various open-source tools and frameworks for working with Large Language Models like LangChain. LangChain abstracts various components away, making the development of LLM applications easier. In this repository, LangChain specifically abstracts away the dependencies around FAISS (create_index() and search_index()) and HuggingFace embeddings (model.py). An example of this can be found in my repository "retrieval_augmented_generation_with_langchain".

2. It is recommended to create a logic to map FAISS index back to your upstream data assets unique identifier. In this implementation, this logic was left out. 

   - In the product description data in the /data folder, the "id" column is an increasing integer representing a unique identifier for that product, starting at 1 and ending at 500 while the FAISS index starts at index 0.
   - When receiving a item-based recommendation by calling /faissearch/searchindex.search_index, you will get a list of indices and distances based on your selection of k.
   - Ideally, you should be able to map these recommendations back to a unique category in your upstream data to serve your recommendations.
   - In the product description data in the /data folder, this mapping is simply the index + 1.
     
3. Currently the default model being used is a BERT based model defined in /utils/constants as "MODEL_CHECKPOINT". The hidden_size for this is defined as 768 in "HIDDEN_SIZE".
   
   - Depending on your compute, having downstream embeddings may not be feasible with a large sample size. 
   - In the scenario that you would like to change the MODEL_CHECKPOINT, it is imperative that the "HIDDEN_SIZE" is also updated to reflect this change. Failing to do so may result in a failure when calling /faisssearch/searchindex.search_index as the numpy dimensions may not match.
   - In the scenario that you would like to use other models from OpenAI for embeddings, it is imperative that the "EMBED_DIM" is modified to reflect this. Failing to do so may result in a failure when calling /faisssearch/searchindex.search_index as the numpy dimensions may not match.

3. Depending on your sample size for your vector database, processing of the vectors may be required to provide item-based recommendations with an acceptable latency.

   - FAISS offers vector processing and clustering techniques to reduce the latency for searches with minimal performance reduction that can be found in their documentation.
