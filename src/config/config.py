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
SUMMARY_SIZE = 100

# test samples
DOCS = list()
samples = [
    "A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations (computation) automatically.",
    "Modern digital electronic computers can perform generic sets of operations known as programs. These programs enable computers to perform a wide range of tasks. A computer system is a nominally complete computer that includes the hardware, operating system (main software), and peripheral equipment needed and used for full operation. This term may also refer to a group of computers that are linked and function together, such as a computer network or computer cluster. A broad range of industrial and consumer products use computers as control systems. Simple special-purpose devices like microwave ovens and remote controls are included, as are factory devices like industrial robots and computer-aided design, as well as general-purpose devices like personal computers and mobile devices like smartphones. Computers power the Internet, which links billions of other computers and users. According to the Oxford English Dictionary, the first known use of computer was in a 1613 book called The Yong Mans Gleanings by the English writer Richard Brathwait: I haue [sic] read the truest computer of Times, and the best Arithmetician that euer [sic] breathed, and he reduceth thy dayes into a short number. This usage of the term referred to a human computer, a person who carried out calculations or computations. The word continued with the same meaning until the middle of the 20th century. During the latter part of this period women were often hired as computers because they could be paid less than their male counterparts.[1] By 1943, most human computers were women. The Online Etymology Dictionary gives the first attested use of computer in the 1640s, meaning 'one who calculates'; this is an 'agent noun from compute (v.)'. The Online Etymology Dictionary states that the use of the term to mean 'calculating machine' (of any type) is from 1897. The Online Etymology Dictionary indicates that the 'modern use' of the term, to mean 'programmable digital electronic computer' dates from 1945 under this name; [in a] theoretical [sense] from 1937, as Turing machine." +
    "The machine was about a century ahead of its time. All the parts for his machine had to be made by hand – this was a major problem for a device with thousands of parts. Eventually, the project was dissolved with the decision of the British Government to cease funding. Babbage's failure to complete the analytical engine can be chiefly attributed to political and financial difficulties as well as his desire to develop an increasingly sophisticated computer and to move ahead faster than anyone else could follow. Nevertheless, his son, Henry Babbage, completed a simplified version of the analytical engine's computing unit (the mill) in 1888. He gave a successful demonstration of its use in computing tables in 1906.",
    ]
for s in samples:
    doc = Document(page_content=s, metadata={"source": "local"})
    DOCS.append(doc)