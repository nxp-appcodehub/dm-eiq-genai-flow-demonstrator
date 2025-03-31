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
import sys
import torch
import torch.nn.functional as F
import errno
import time
from colorama import Fore
from rag.utils import load_pkl, check_censored_word_presence, pretty_print
from rag.models.embedding_models.embedding_models import EmbeddingModel


class Retriever:
    def __init__(self,
                 top_k: int,
                 reranking: bool,
                 best_k: int,
                 rag_db_path: str,
                 verbose: bool = False,
                 ):

        src_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.verbose = verbose
        self.top_k = top_k
        self.best_k = best_k
        self.reranking = reranking

        if hasattr(sys, '_MEIPASS'):
            # Update path for Pyinstaller package
            src_dir_path = src_dir_path.replace("_internal", "rag/src")

        self.embedding_model = EmbeddingModel(name="all-MiniLM-L6-v2")

        if not os.path.isfile(rag_db_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), rag_db_path)

        database = load_pkl(rag_db_path)

        rag_db_info = {}
        embedding_model_version = None

        # Check if the database contains the "embedding_model" key
        if "embedding_model" in database:
            embedding_model_version = database.pop("embedding_model")
            # Verify if the stored embedding model matches the current one
            if embedding_model_version != self.embedding_model.embedding_model_version:
                raise UserWarning(
                    f"Mismatch in embedding model versions:\n"
                    f" - Database version = {embedding_model_version},\n"
                    f" - Inference version = {self.embedding_model.embedding_model_version}\n"
                    "You can either change the chosen embedding model in config.py or re-generate your database."
                )
        else:
            print(Fore.RED,
                  "Warning: Unable to verify if the same embedding model was used during database generation.",
                  Fore.RESET)

        if "database_description" in database:
            rag_db_info["Description"] = database.pop("database_description")
        if embedding_model_version:
            rag_db_info["Embedding model used for generation"] = embedding_model_version
        if "database_generator_files" in database:
            rag_db_info["Chunk files used for generation"] = database.pop("database_generator_files")
        if self.verbose:
            pretty_print(name="RAG database information", result_dictionary=rag_db_info)
        else:
            if "Description" in rag_db_info:
                print(Fore.LIGHTGREEN_EX, f"\rDatabase used: {rag_db_info['Description']}", Fore.RESET)

        self.chunk_list, self.embedding_list, self.metadata_list = self._split_database(database)

    @staticmethod
    def _split_database(database: dict) -> tuple[list, torch.Tensor, list]:
        """
        Split the input dictionary into three aligned components.
        :param database: Input dictionary containing the chunks (text) and their related embeddings and metadata.
        :return: List of chunks, tensor of embeddings, and list of metadata. The elements are aligned.
        """

        embedding_list = torch.empty((0, database[0]['embeddings'].shape[1]))
        chunk_list = []
        metadata_list = []

        for key, value in database.items():
            num_elements = value['embeddings'].shape[0]
            embeddings = value.pop('embeddings')
            embedding_list = torch.cat((embedding_list, embeddings), dim=0)
            # Store chunks (keeping them as a list since they are likely text)
            chunk_list.extend(value.pop('chunks'))
            # Store metadata (keeping it as a list of dictionaries to avoid tensor conversion issues)
            metadata_list.extend([value.copy() for _ in range(num_elements)])  # Copy to avoid modifying original dict

        return chunk_list, embedding_list, metadata_list

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

    def _find_top_k(self, query_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
            questions_embeddings[i] = self.metadata_list[index]["reranking_embedding"]

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

    def __call__(self, query: str) -> tuple[list, list, list]:
        """
        Retrieve the most relevant chunks (and their related metadata) in the database for the input query.
        :param query: user query
        :return: most relevant chunks and related metadata
        """

        start_time = time.time()
        # Check presence of censored words
        is_query_censored = check_censored_word_presence(query)
        if is_query_censored:
            best_chunk_list = ["" for _ in range(self.best_k)]
            best_similarity_list = [0.0 for _ in range(self.best_k)]
            best_metadata_list = [{"source": "censored_queries"} for _ in range(self.best_k)]
        else:
            # text query is transformed in an embedding
            query_embedding = self.embedding_model.encode(query)

            # get top_k retrieved embeddings from data and their similarity
            top_similarity_list, top_index_list = self._find_top_k(query_embedding=query_embedding)

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
                raise ValueError("best_k value must be inferior or equal to top_k value.")

            # get chunks (texts) and related metadata corresponding to the retrieved embeddings
            best_chunk_list = [self.chunk_list[i] for i in best_index_list]
            best_metadata_list = [self.metadata_list[i] for i in best_index_list]
            best_similarity_list = best_similarity_list.tolist()

        if self.verbose:
            pretty_print(name="RAG", result_dictionary={
                "Latency": f"{(time.time() - start_time):0.2f}s",
                "Chunks": best_chunk_list,
                "Similarities": best_similarity_list,
                "Metadata": best_metadata_list,

            })

        return best_chunk_list, best_similarity_list, best_metadata_list
