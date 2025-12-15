# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import time
import torch
import pickle
import logging
import numpy as np
import torch.nn.functional as F
from rag.config import Config as RAGConfig
from rag.utils import check_censored_word_presence, pretty_log
from rag.models.embedding_models.embedding_models import EmbeddingModel

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self,
                 config: RAGConfig,
                 embedding_model: str | EmbeddingModel,
                 rag_db: str | dict,
                 ):

        self.top_k = config.top_k
        self.best_k = config.best_k
        self.reranking = config.reranking

        logger.debug(f"Loading embeddding model")
        if isinstance(embedding_model, str):
            self.embedding_model = EmbeddingModel(name=embedding_model)
        else:
            self.embedding_model = embedding_model

        db_loader = DatabaseLoader(source=rag_db)
        database = db_loader.load()

        rag_db_info = {}
        embedding_model_version = None

        # Check if the database contains the "embedding_model" key
        if "embedding_model" in database:
            embedding_model_version = database.pop("embedding_model")
            # Verify if the stored embedding model matches the current one
            if embedding_model_version != self.embedding_model.embedding_model_version:
                warning_text = (
                    f"Mismatch in embedding model versions:\n"
                    f"   • Database version = {embedding_model_version},\n"
                    f"   • Inference version = {self.embedding_model.embedding_model_version}\n"
                    f"You can either change the chosen embedding model in config.py or re-generate your database.")
                [logger.warning(line) for line in warning_text.split('\n')]
                raise UserWarning(warning_text)
        else:
            logger.warning("Unable to verify if the same embedding model was used during database generation.")

        if "database_description" in database:
            rag_db_info["Description"] = database.pop("database_description")
        if embedding_model_version:
            rag_db_info["Embedding model used for generation"] = embedding_model_version
        if "database_generator_files" in database:
            rag_db_info["Chunk files used for generation"] = database.pop("database_generator_files")
        if rag_db_info:
            pretty_log(name="RAG database information", result_dictionary=rag_db_info)

        self.chunk_list, self.embedding_list, self.reranking_embedding_list, self.metadata_list = self._split_database(database)

    @staticmethod
    def _split_database(database: dict) -> tuple[list, torch.Tensor, torch.Tensor, list]:
        """
        Split the input dictionary into three aligned components.
        :param database: Input dictionary containing the chunks (text) and their related embeddings and metadata.
        :return: List of chunks, tensor of embeddings, and list of metadata. The elements are aligned.
        """

        embedding_list = []
        reranking_embedding_list = []
        chunk_list = []
        metadata_list = []

        for key, value in database.items():
            num_elements = value['embeddings'].shape[0]
            embeddings = value.pop('embeddings')
            embedding_list.append(embeddings)
            reranking_embedding = value.pop('reranking_embedding')
            reranking_embedding_list.append(reranking_embedding.repeat(num_elements, 1))
            # Store chunks (keeping them as a list since they are likely text)
            chunk_list.extend(value.pop('chunks'))
            # Store metadata (keeping it as a list of dictionaries to avoid tensor conversion issues)
            metadata_list.extend([value.copy() for _ in range(num_elements)])  # Copy to avoid modifying original dict

        return chunk_list, torch.cat(embedding_list), torch.cat(reranking_embedding_list), metadata_list

    @staticmethod
    def _similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x: first input
        :param y: second input
        :return: similarity between the two inputs
        """

        return F.cosine_similarity(x, y, dim=-1)

    @staticmethod
    def _top_k(array: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the k highest value(s) and their indexes in the input array.
        :param array: input array in which we look for the highest value(s)
        :param k: number of element to keep
        :return: the index(es) of the highest value(s) in the input array and the corresponding element(s) in the array
        """

        values, indices = torch.topk(array, k, dim=-1)  # Returns values and indices
        return values, indices

    def _find_top_k(self,
                    query_embedding: torch.Tensor,
                    cluster_label: str | list[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity between query embedding and the database embeddings and find the top_k.
        :param query_embedding: embedding of the query
        :return: selected indices in the database and the related similarities
        """

        # compute similarity between database embeddings and the query
        sim = self._similarity(self.embedding_list, query_embedding)

        # Select top-k similarities
        top_similarity_list, top_index_list = self._top_k(array=sim.flatten(), k=self.top_k)

        return top_similarity_list, top_index_list

    def _rerank(self,
                top_index_list: torch.Tensor,
                top_similarity_list: torch.Tensor,
                query_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rerank the order of the relevant chunks by averaging the similarity with the similarity of the question
        part of the chunk (if it exists).
        :param top_index_list: tensor of indices of the most similar chunks
        :param top_similarity_list: tensor of similarities of the most similar chunks
        :param query_embedding: user query embedding
        :return: reranked indexes in the database and the related updated similarities
        """

        # Initialize questions_embeddings tensor
        questions_embeddings = torch.zeros((len(top_index_list), query_embedding.shape[-1]))

        # Extract reranking embeddings from metadata_list
        for i, index in enumerate(top_index_list):
            questions_embeddings[i] = self.reranking_embedding_list[index]

        # Compute similarity using PyTorch's cosine similarity
        new_similarity_list = self._similarity(questions_embeddings, query_embedding)

        # Compute final similarity by averaging
        new_similarity_list = (top_similarity_list + new_similarity_list) / 2

        # Get the reranked indices and similarities
        reranked_similarity_list, new_index_order_list = self._top_k(array=new_similarity_list, k=self.best_k)

        # Reorder top indices accordingly
        # reranked_index_list = [top_index_list[i].item() for i in new_index_order_list]
        reranked_index_list = top_index_list[new_index_order_list]

        return reranked_index_list, reranked_similarity_list

    def __call__(self, query: str, cluster_label: str | list[str] = None) -> tuple[list, list, list]:
        """
        Retrieve the most relevant chunks (and their related metadata) in the database for the input query.
        :param query: user query
        :return: most relevant chunks and related metadata
        """

        start_time = time.time()

        # text query is transformed in an embedding
        query_embedding = self.embedding_model.encode(query)

        # get top_k retrieved embeddings from data and their similarity
        top_similarity_list, top_index_list = self._find_top_k(query_embedding=query_embedding,
                                                               cluster_label=cluster_label)

        if self.best_k < self.top_k:
            # reranking
            if self.reranking:
                best_index_list, best_similarity_list = self._rerank(top_index_list, top_similarity_list,
                                                                     query_embedding)
            else:
                best_index_list, best_similarity_list = top_index_list[:self.best_k], top_similarity_list[
                                                                                      :self.best_k]

        elif self.best_k == self.top_k:
            best_index_list, best_similarity_list = top_index_list, top_similarity_list

        else:
            error_text = "In config.py, best_k value must be inferior or equal to top_k value."
            logger.error(error_text)
            raise ValueError(error_text)

        # get chunks (texts) and related metadata corresponding to the retrieved embeddings
        best_chunk_list = [self.chunk_list[i] for i in best_index_list]
        best_metadata_list = [self.metadata_list[i] for i in best_index_list]
        best_similarity_list = best_similarity_list.tolist()

        pretty_log(name="Retriever", result_dictionary={
            "Query": query,
            "Latency": f"{(time.time() - start_time):0.5f}s",
            "Chunks": best_chunk_list,
            "Similarities": best_similarity_list,
            "Metadata": best_metadata_list,

        })

        return best_chunk_list, best_similarity_list, best_metadata_list


class QueryClassifier:
    def __init__(self,
                 config: RAGConfig,
                 retriever: Retriever,
                 similarity_threshold: float = 0.65
                 ):

        self.rag_config = config
        self.retriever = retriever
        self.similarity_threshold = similarity_threshold

    def __call__(self, query: str, cluster_label: str | list[str] = None):
        chunk_list, similarity_list, metadata_list = self.retriever(query=query, cluster_label=cluster_label)

        out_of_domain_qty = sum(metadata.get("source", "unknown") in self.rag_config.out_of_domain_source_list
                                for metadata in metadata_list)  # Garbage model or censored queries detection

        if check_censored_word_presence(query):  # Check for censored words in query
            query_category = "CENSORED"

        elif "intent" in metadata_list[0]:  # Assumes intent when first chunk metadata contains intent attribute
            query_category = "INTENT"

        elif out_of_domain_qty > len(chunk_list) // 2:  # Out of domain if the majority of sources are out of domain
            query_category = "REJECTED"

        elif np.mean(similarity_list) < self.similarity_threshold:  # Ambiguous if the similarity average is under threshold
            query_category = "AMBIGUOUS"

        else:  # Default case when no specific conditions are met
            query_category = "ACCEPTED"

        logger.debug(f"QueryClassifier: The query category is: {query_category}")
        return query_category, chunk_list, similarity_list, metadata_list


class DatabaseLoader:
    """
    Safely loads and validates a RAG database (either from a dict or a pickle file).

    Supports:
    - Loading directly from an in-memory dict.
    - Loading from a path to a pickle (.pkl) file.

    Ensures the database structure follows the expected schema before returning it.
    """

    _ALLOWED_OPTIONAL_KEYS = {
        "database_description",
        "database_generator_files",
        "embedding_model",
    }

    def __init__(self, source: str | dict):
        """
        :param source (str | dict): Either a path to a pickle file or an in-memory database dict.
        :return: serialized database with the correct format
        """

        self.source = source

    class SafeUnpickler(pickle.Unpickler):

        def load(self):
            pkl_database = super().load()

            if not isinstance(pkl_database, dict):
                error_text = "Invalid pickle format: Expected a dictionary"
                logger.error(error_text)
                raise ValueError(error_text)

            return pkl_database

    def load(self) -> dict:
        """Load and validate the database from the provided source."""

        if isinstance(self.source, dict):
            logger.debug("Validating in-memory database dictionary.")
            return self._validate(self.source)

        elif isinstance(self.source, str):
            if not os.path.isfile(self.source):
                raise FileNotFoundError(f"Database file not found: {self.source}")
            logger.debug(f"Loading and validating database from file: {self.source}")
            with open(self.source, 'rb') as file:
                return self._validate(self.SafeUnpickler(file).load())

        else:
            raise TypeError(f"Unsupported source type: {type(self.source).__name__}")

    def _validate(self, database: dict) -> dict:
        """Ensure the database structure is valid."""

        if not isinstance(database, dict):
            raise ValueError("Invalid database format: expected a dictionary at top level.")

        for key, value in database.items():
            if key in self._ALLOWED_OPTIONAL_KEYS:
                continue  # Skip optional metadata keys

            if not isinstance(value, dict):
                raise ValueError(f"Invalid entry '{key}': expected dict, got {type(value).__name__}.")

            chunks = value.get("chunks")
            if chunks is None:
                raise ValueError(f"Invalid entry '{key}': missing required key 'chunks'.")

            if not isinstance(chunks, list) or not all(isinstance(item, str) for item in chunks):
                raise ValueError(f"Invalid entry '{key}': 'chunks' must be a list of strings.")

        logger.debug("Database format validated successfully.")
        return database
