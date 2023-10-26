import json
import os
import pickle
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydash
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from networkx.algorithms import community
from scipy.spatial.distance import cosine

from src.config import config
from utilities.custom_logger import CustomLogger

logger = CustomLogger()

# load SLR data
with open(config.SLR_PATH / 'judiciary_32k_test.jsonl', 'r') as json_file:
    json_list = list(json_file)

slr = dict()
for idx, json_str in enumerate(json_list):
    slr[idx] = json.loads(json_str)

# take 1 example for analysis
article = slr[0]["judgment"]
logger.info(article)


def get_num_of_tokens(text):
    return len(config.TOKENIZER.tokenize(text))


# remove chunck's title
def is_paragraph(txt):
    """filter the paragraph with index"""
    if (re.match(r"^[0-9]+ ", txt) is None) and (get_num_of_tokens(txt) < 20):
        return False
    else:
        return True


# 4 x newlines to separate chunks
article = "".join([p if is_paragraph(p) else "\n\n" for p in article.split("\n\n")])

# separate articles by newlines
chunks = [ch for ch in article.split("\n\n") if len(ch) > 0]

# ensure No. of tokens in each chunk < max context window
chunks_require_split = list()
for i, chunk in enumerate(chunks):
    if get_num_of_tokens(chunk) > config.CHUNK_SIZE:
        chunks_require_split.append(i)

text_splitter = CharacterTextSplitter().\
    from_huggingface_tokenizer(
    config.TOKENIZER,
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=50,
    separator="\n",
    keep_separator=True
)

if len(chunks_require_split) > 0:
    for i in chunks_require_split:
        chunks[i] = text_splitter.split_text(chunks[i])

chunks = pydash.flatten_deep(chunks)

for i, chunk in enumerate(chunks):
    logger.info(f"chunk size - {i}: {get_num_of_tokens(chunk)}")


# apply recursive character split each chunk into paragraphs
text_splitter = RecursiveCharacterTextSplitter(
    keep_separator=False,
    chunk_size=(config.CHUNK_SIZE / 4),
    chunk_overlap=0,
    length_function=get_num_of_tokens,
    is_separator_regex=False,
    separators=["\n\n", "\n", ". "],
)

paragraph = [s.page_content for s in text_splitter.create_documents(chunks)]

# get the statistics of setences
length_max = max([get_num_of_tokens(s) for s in paragraph])
length_min = min([get_num_of_tokens(s) for s in paragraph])


# embedding by model
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GCP_CRED_PATH

# # alternative way to add PATH
# from google.oauth2 import service_account
# credentials = service_account.Credentials.from_service_account_file('/home/kewen_yang/gptx2/iconic-vine-398108-54c67e9dcc36.json')

# # test if credentials created correctly
# import google.auth
# credentials, project_id = google.auth.default()

"""The API accepts a maximum of 3,072 input tokens and outputs 768-dimensional vector embeddings.
Use the following parameters for the text embeddings model textembedding-gecko(it belongs to PaLM Model)
"""
embeddings = VertexAIEmbeddings()

embedding_dict = dict()
long_para = list()

for idx, para in enumerate(paragraph):
    sen_embedding = embeddings.embed_query(para)

    embedding_dict[idx] = {
        "text": para,
        "embedding": sen_embedding
        }

with open(config.OUT_PATH / "embedding_paragraph.json", "w") as f:
    json.dump(embedding_dict, f, indent=2)


# load embeddings and get the similarity matrix for assessment
with open(config.OUT_PATH / "embedding_paragraph.json", "r") as f:
    embedding_dict = json.load(f)

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


# Run the community detection algorithm
def get_topics(title_similarity, num_topics = 8, bonus_constant = 0.25, min_size = 3):

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
  modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)

  lowest_sd_iteration = 0
  # Set lowest sd to inf
  lowest_sd = float('inf')

  for i in range(iterations):
    topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution, seed=1)
    modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)

    # Check SD
    topic_sizes = [len(c) for c in topics_title]
    sizes_sd = np.std(topic_sizes)

    topics_title_accepted.append(topics_title)

    if sizes_sd < lowest_sd and min(topic_sizes) < min_size:
      lowest_sd_iteration = i
      lowest_sd = sizes_sd

  # Set the chosen partitioning to be the one with highest modularity
  topics_title = topics_title_accepted[lowest_sd_iteration]
  print(f'Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}')

  topic_id_means = [sum(e)/len(e) for e in topics_title]
  # Arrange title_topics in order of topic_id_means
  topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title), key = lambda pair: pair[0])]
  # Create an array denoting which topic each chunk belongs to
  chunk_topics = [None] * title_similarity.shape[0]
  for i, c in enumerate(topics_title):
    for j in c:
      chunk_topics[j] = i

  return {
    'chunk_topics': chunk_topics,
    'topics': topics_title
    }


num_topics = get_num_of_tokens(article) // config.COMMUNITY_SIZE
topics_out = get_topics(summary_similarity_matrix,
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


with open(config.OUT_PATH / "chunks", "wb") as fp:
    pickle.dump(chunks, fp)

# gcloud init:
# https://cloud.google.com/sdk/docs/initializing
# https://cloud.google.com/sdk/gcloud/reference/auth/activate-service-account#ACCOUNT