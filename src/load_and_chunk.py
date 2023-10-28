import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydash
import streamlit as st
from langchain.embeddings import VertexAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from networkx.algorithms import community
from scipy.spatial.distance import cosine

from config import config
from utilities.custom_logger import CustomLogger

logger = CustomLogger()


@dataclass
class ProcessingPipeline:
    embeddings: Embeddings
    num_of_tokens: Optional[int] = None

    def process_document(self, document: str) -> List[str]:
        """process a long document into list of shorter chunks, where each chunk has a unique topic

        Args:
            document (str): long text

        Returns:
            List[str]: list of chunks
        """
        paragraphs = self.split_document(document)
        embedding_dict = self.get_embeddings(paragraphs)
        chunks = self.cluster_similar_chunks(embedding_dict)
        return chunks

    @staticmethod
    def get_num_of_tokens(text: str) -> int:
        """get No. of tokens in the text"""
        return len(config.TOKENIZER.tokenize(text))

    def is_paragraph(self, txt):
        """filter the paragraph with index"""
        if (re.match(r"^[0-9]+ ", txt) is None) and (self.get_num_of_tokens(txt) < 20):
            return False
        else:
            return True

    def split_document(self, document: str) -> List[str]:
        """split the document into chunks with shorter length, this is document-specific.

        Args:
            document (str): document, a long text

        Returns:
            List[str]: a list of chunks
        """
        # remove sub-headers
        document = "".join([p if self.is_paragraph(p) else "\n\n" for p in document.split("\n\n")])

        self.num_of_tokens = self.get_num_of_tokens(document)

        # split documents by newlines
        chunks = [ch for ch in document.split("\n\n") if len(ch) > 0]

        # ensure No. of tokens in each chunk < max context window
        chunks_require_split = list()
        for i, chunk in enumerate(chunks):
            if self.get_num_of_tokens(chunk) > config.CHUNK_SIZE:
                chunks_require_split.append(i)

        text_splitter = CharacterTextSplitter().\
            from_huggingface_tokenizer(
            config.TOKENIZER,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=50,
            separator="\n",
            keep_separator=True
        )

        # fine the chunks which need further split
        if len(chunks_require_split) > 0:
            for i in chunks_require_split:
                chunks[i] = text_splitter.split_text(chunks[i])

        # till this step, the chunks' size already less than window size
        chunks = pydash.flatten_deep(chunks)

        length_max = max([self.get_num_of_tokens(ch) for ch in chunks])
        length_min = min([self.get_num_of_tokens(ch) for ch in chunks])

        logger.info(f"After splitting by paragrah:\ntotal No. of chunks: {len(chunks)}, max length: {length_max}, min length: {length_min}")

        # apply recursive character split each chunk into paragraphs
        text_splitter = RecursiveCharacterTextSplitter(
            keep_separator=False,
            chunk_size=(config.CHUNK_SIZE / 2),
            chunk_overlap=50,
            length_function=self.get_num_of_tokens,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". "],
        )

        paragraphs = [s.page_content for s in text_splitter.create_documents(chunks)]

        # get the statistics of setences
        length_max = max([self.get_num_of_tokens(s) for s in paragraphs])
        length_min = min([self.get_num_of_tokens(s) for s in paragraphs])

        logger.info(f"After splitting by sentence:\ntotal No. of paragraphs: {len(paragraphs)}, max length: {length_max}, min length: {length_min}")

        return paragraphs

    def get_embeddings(self, paragraphs: List[str]) -> Dict[str, Dict]:
        """embeddings for each paragraph.
        The API accepts a maximum of 3,072 input tokens and outputs 768-dimensional vector embeddings.
        Use the following parameters for the text embeddings model textembedding-gecko(it belongs to PaLM Model)

        Args:
            paragraphs (List[str]): texts

        Returns:
            Dict[str, Dict]: embeddings
        """

        embedding_dict = dict()
        for idx, para in enumerate(paragraphs):
            sen_embedding = self.embeddings.embed_query(para)

            embedding_dict[str(idx)] = {
                "text": para,
                "embedding": sen_embedding
                }

        with open(config.OUT_PATH / "embedding_paragraph.json", "w") as f:
            json.dump(embedding_dict, f, indent=2)

        # # load embeddings and get the similarity matrix for assessment
        # with open(config.OUT_PATH / "embedding_paragraph.json", "r") as f:
        #     embedding_dict = json.load(f)

        return embedding_dict

    def cluster_similar_chunks(self, embedding_dict: Dict[str, Dict]) -> List:
        """
        cluster chunks into 1 if they share similar semantic meaning
        Args:
            embedding_dict (Dict[str, Dict]): embeddings

        Returns:
            List: list of chunks
        """
        # Get similarity matrix between the embeddings of the sentences' embeddings
        summary_similarity_matrix = np.zeros((len(embedding_dict), len(embedding_dict)))
        summary_similarity_matrix[:] = np.nan

        for row in range(len(embedding_dict)):
            for col in range(row, len(embedding_dict)):
                # Calculate cosine similarity between the two vectors
                similarity = 1 - cosine(embedding_dict[str(row)]["embedding"], embedding_dict[str(col)]["embedding"])
                summary_similarity_matrix[row, col] = similarity
                summary_similarity_matrix[col, row] = similarity

        plt.figure()
        plt.imshow(summary_similarity_matrix, cmap='Blues')
        plt.savefig(config.OUT_PATH / "similarity_matrix_paragraph.jpg")

        num_topics = self.num_of_tokens // config.COMMUNITY_SIZE
        topics_out = self.get_topics(
            summary_similarity_matrix,
            num_topics=num_topics,
            bonus_constant=0.2,
            min_size=10)
        chunk_topics = topics_out['chunk_topics']
        topics = topics_out['topics']

        # Plot a heatmap of this array
        plt.figure(figsize=(10, 4))
        plt.imshow(np.array(chunk_topics).reshape(1, -1), cmap='tab20')
        # Draw vertical black lines for every 1 of the x-axis
        for i in range(1, len(chunk_topics)):
            plt.axvline(x=i - 0.5, color='black', linewidth=0.5)

        plt.savefig(config.OUT_PATH / "clustering_paragraph.jpg")

        chunks = list()
        for chu_ids in topics:
            chunk = "\n".join([embedding_dict[str(i)]["text"] for i in chu_ids])
            chunks.append(chunk)

        # with open(config.OUT_PATH / "chunks", "wb") as fp:
        #     pickle.dump(chunks, fp)

        return chunks

    def get_topics(self,
                   title_similarity: np.ndarray,
                   num_topics: int=8,
                   bonus_constant: float=0.25,
                   min_size: int=3) -> Dict[str, List]:
        """calculate if chunks belong to same cluster based on louvain community detection algorithm

        Args:
            title_similarity (np.ndarray): cosine similarity between chunks
            num_topics (int, optional): number of chunks in the end. Defaults to 8.
            bonus_constant (float, optional): coefficient. Defaults to 0.25.
            min_size (int, optional): minimum size of a chunk. Defaults to 3.

        Returns:
            _type_: _description_
        """
        proximity_bonus_arr = np.zeros_like(title_similarity)
        for row in range(proximity_bonus_arr.shape[0]):
            for col in range(proximity_bonus_arr.shape[1]):
                if row == col:
                    proximity_bonus_arr[row, col] = 0
                else:
                    proximity_bonus_arr[row, col] = 1/(abs(row-col)) * bonus_constant

        title_similarity += proximity_bonus_arr

        title_nx_graph = nx.from_numpy_array(title_similarity)

        desired_num_topics = num_topics
        # Store the accepted partitionings
        topics_title_accepted = []

        resolution = 0.85
        resolution_step = 0.01
        iterations = 40

        # Find the resolution that gives the desired number of topics
        topics_title = []
        while len(topics_title) not in [desired_num_topics, desired_num_topics + 1, desired_num_topics + 2]:
            topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution, seed=1)
            resolution += resolution_step
        topic_sizes = [len(c) for c in topics_title]
        sizes_sd = np.std(topic_sizes)

        lowest_sd_iteration = 0
        # Set lowest sd to inf
        lowest_sd = float('inf')

        for i in range(iterations):
            topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution, seed=1)

            # Check SD
            topic_sizes = [len(c) for c in topics_title]
            sizes_sd = np.std(topic_sizes)

            topics_title_accepted.append(topics_title)

            if sizes_sd < lowest_sd and min(topic_sizes) < min_size:
                lowest_sd_iteration = i
                lowest_sd = sizes_sd

        # Set the chosen partitioning to be the one with highest modularity
        topics_title = topics_title_accepted[lowest_sd_iteration]
        logger.info(f'Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}')

        topic_id_means = [sum(e)/len(e) for e in topics_title]
        # Arrange title_topics in order of topic_id_means
        topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title), key = lambda pair: pair[0])]
        # Create an array denoting which topic each chunk belongs to
        chunk_topics = [None] * title_similarity.shape[0]
        for i, c in enumerate(topics_title):
            for j in c:
                chunk_topics[j] = i

        return {'chunk_topics': chunk_topics,
                'topics': topics_title}

# gcloud init:
# https://cloud.google.com/sdk/docs/initializing
# https://cloud.google.com/sdk/gcloud/reference/auth/activate-service-account#ACCOUNT
