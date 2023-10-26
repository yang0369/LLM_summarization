import pickle

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatVertexAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from config import config
from utilities.utility import get_num_of_tokens
from utilities.custom_logger import CustomLogger

logger = CustomLogger()


####### MAP-REDUCE SUMMARIZATION
# load the chunks
with open(config.OUT_PATH / "chunks", "rb") as fp:
    chunks = pickle.load(fp)

for ch in chunks:
    logger.info(f"chunck size: {get_num_of_tokens(ch)}")

split_docs = [Document(page_content=text, metadata={"source": "local"}) for text in chunks]

llm = ChatVertexAI()

# map
map_template = """The following is a set of documents
{input_documents}
Based on this list of docs, please identify the main themes with less than 250 words
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce
reduce_template = """The following is set of summaries:
{input_documents}
Take these and distill it into a final, consolidated summary.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=1000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="input_documents",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)


logger.info(map_reduce_chain({"input_documents": split_docs}))
