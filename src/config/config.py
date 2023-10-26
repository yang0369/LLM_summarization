import os
from pathlib import Path

import google.auth
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from transformers import LlamaTokenizerFast

from utilities.custom_logger import CustomLogger
from langchain.docstore.document import Document

logger = CustomLogger()

# set paths
ROOT = Path(__file__).parents[2]
SRC = ROOT / "src"
DATA_PATH = ROOT / "data"
SLR_PATH = DATA_PATH / "SLR"
OUT_PATH = DATA_PATH / "artifacts"
MODEL_PATH = ROOT / "model"
CREDENTIAL_PATH = ROOT / "credential"
GCP_CRED_PATH = (CREDENTIAL_PATH / "gcp_credential.json").as_posix()
TOKENIZER_PATH = (MODEL_PATH / "Llama-2-13b-chat-hf").as_posix()

# load tokenizer
TOKENIZER = LlamaTokenizerFast.from_pretrained(TOKENIZER_PATH)

# add credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CRED_PATH

# verify if credential loaded successfully
credentials, project_id = google.auth.default()
if project_id == "iconic-vine-398108":
    logger.info("successfully loaded the GCP credentials")

# enable in-memory caching
set_llm_cache(InMemoryCache())

CHUNK_SIZE = 1000
COMMUNITY_SIZE = CHUNK_SIZE / 3

# test samples
DOCS = list()
samples = [
    "A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations (computation) automatically. Modern digital electronic computers can perform generic sets of operations known as programs. These programs enable computers to perform a wide range of tasks.",
    "A computer system is a nominally complete computer that includes the hardware, operating system (main software), and peripheral equipment needed and used for full operation. This term may also refer to a group of computers that are linked and function together, such as a computer network or computer cluster.",
    "A broad range of industrial and consumer products use computers as control systems. Simple special-purpose devices like microwave ovens and remote controls are included, as are factory devices like industrial robots and computer-aided design, as well as general-purpose devices like personal computers and mobile devices like smartphones.",
    "Computers power the Internet, which links billions of other computers and users.",
    ]
for s in samples:
    doc = Document(page_content=s, metadata={"source": "local"})
    DOCS.append(doc)