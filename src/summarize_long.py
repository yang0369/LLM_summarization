import pickle
from typing import List

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatVertexAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from config import config
from utilities.custom_logger import CustomLogger
from utilities.utility import get_num_of_tokens

logger = CustomLogger()


# OPTION_1: BY LANGCHAIN MAP REDUCE
# llm chain
def summarize_long_text_by_langchain(docs: List[Document]) -> None:
    document_variable_name = "text"
    map_prompt_template = """Wite a concise summary with less than 100 words for the following text:
    {text}
    CONCISE SUMMARY:"""
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    llm = ChatVertexAI()
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # reduce chain
    # We now define how to combine these summaries
    reduce_template = 'Write a ' + str(config.SUMMARY_SIZE) + """-word summary of the following, removing irrelevant information. Finish your answer:
    {text}
    """ + str(config.SUMMARY_SIZE) + """-WORD SUMMARY:"""
    reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["text"])
    reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain,
        document_variable_name=document_variable_name
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        token_max=1024,
    )

    chain = MapReduceDocumentsChain(
        # map
        llm_chain=map_chain,
        # reduce
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name=document_variable_name,
        # Return the results of the map steps in the output
        return_intermediate_steps=True,
    )

    summary = chain({"input_documents": docs})  # input_documents is required keyword for combined chain

    logger.info("\n\ninput text:\n")
    for doc in docs:
        logger.info(f"\n{doc.page_content}\n")
    logger.info(f"\nsummarization: \n{summary['output_text'].strip()}\n")


# OPTION_2: BY CUSTOM WAY
def summarize_long_text_by_custom(docs: List[Document]) -> None:
    def get_short_sum_chain(template: str) -> StuffDocumentsChain:
        """ prepare a summarization chain for single text
        """
        prompt = PromptTemplate(template=template, input_variables=["text"])

        # Define LLM chain
        llm = ChatVertexAI()
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        return StuffDocumentsChain(llm_chain=llm_chain)

    # map summarization for each chunk
    map_template = "Write a concise summary with only 1 sentence for the following text: {text}. CONCISE SUMMARY:"
    map_chain = get_short_sum_chain(map_template)

    summaries = {}
    for idx, doc in enumerate(docs):
        summ = map_chain({"input_documents": [doc]})
        summaries[idx] = summ["output_text"]

    # reduce all summarizations into one single summary
    reduce_template = "Write a summary with key points for the following text: {text}. CONCISE SUMMARY:"
    reduce_chain = get_short_sum_chain(reduce_template)

    combined = Document(page_content=" ".join([s.strip() for s in summaries.values()]), metadata={"source": "local"})
    summary = reduce_chain({"input_documents": [combined]})

    logger.info("\n\ninput text:\n")
    for doc in docs:
        logger.info(f"\n{doc.page_content}\n")
    logger.info(f"\nsummarization: \n{summary['output_text'].strip()}\n")


if __name__ == '__main__':
    # load the text chunks
    with open(config.OUT_PATH / "chunks", "rb") as fp:
        chunks = pickle.load(fp)

    chunks = chunks[:5]  # try with first 5 chunks only

    for ch in chunks:
        logger.info(f"chunck size: {get_num_of_tokens(ch)}")

    split_docs = [Document(page_content=text, metadata={"source": "local"}) for text in chunks]
    # summarize_long_text_by_custom(split_docs)
    summarize_long_text_by_langchain(split_docs)
