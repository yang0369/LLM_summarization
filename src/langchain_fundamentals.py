from langchain.chains import (LLMChain, MapReduceDocumentsChain,
                              ReduceDocumentsChain)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatVertexAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

from config import config  # noqa

llm = ChatVertexAI()

####################################################################################################
# # QUESTION ANSWER CHAIN
# prompt_template = """
# What is a good name for a company that makes {entity}?
# """

# llm_chain = LLMChain(
#     llm=llm,
#     prompt=PromptTemplate.from_template(prompt_template)
# )

# entity = 'colorful socks'
# print(f"\n\nquestions: {prompt_template.format(entity=entity)}\n\nanswer: {llm_chain.predict(entity=entity)}")


####################################################################################################
# # STUFF SUMMARIZATION
# """
# Stuff the documents into context
# >> Input: A list of documents
# >> Output: single summarized text

# 1. combine a list of docs into a single string with document_prompt and document_separator
# 2. feed the single string as document_variable_name to llm
# 3. llm_chain summarizes the document_variable_name
# """
# document_prompt = PromptTemplate(
#     input_variables=["page_content"],  # what we want to call the input string
#     template="{page_content}"
# )

# prompt_template = """Write a concise summary of the following:
# "{combined_string}"
# CONCISE SUMMARY:"""
# prompt = PromptTemplate.from_template(prompt_template)

# # Define LLM chain
# llm = ChatVertexAI()
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# # Define StuffDocumentsChain
# stuff_chain = StuffDocumentsChain(
#     llm_chain=llm_chain,
#     document_prompt=document_prompt,
#     document_separator="\n\n",
#     document_variable_name="combined_string"
# )

# out = stuff_chain({"input_documents": config.DOCS})  # "input_documents" is a fixed keyword as required by combine base class
# print(f'\n\nthe summary: {out["output_text"]}\n\n')

####################################################################################################
# MAP REDUCE
# llm chain
document_prompt = PromptTemplate(
    input_variables=["page_content"],
        template="{page_content}"
)
document_variable_name = "context"
# The prompt here should take as an input variable the
# `document_variable_name`
map_template = """Wite a 75-100 word summary of the following text:
{context}
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=prompt)

# reduce chain
# We now define how to combine these summaries
reduce_template = 'Write a ' + str(500) + """-word summary of the following, removing irrelevant information. Finish your answer:
{context}
""" + str(500) + """-WORD SUMMARY:"""

reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name
)

# collapse chain
# If we wanted to, we could also pass in collapse_documents_chain
# which is specifically aimed at collapsing documents BEFORE
# the final call.
prompt = PromptTemplate.from_template(
    "Collapse this content: {context}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
collapse_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name
)
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    # collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
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

import pickle
from langchain.docstore.document import Document

with open(config.OUT_PATH / "chunks", "rb") as fp:
    chunks = pickle.load(fp)

split_docs = [Document(page_content=text, metadata={"source": "local"}) for text in chunks]

out = chain({"input_documents": split_docs})

# print("\n\ninput text:")
# for doc in config.DOCS:
#     print(f"\n{doc.page_content}\n")

print(f"\nsummarization: {out['output_text']}\n")



import pickle
from langchain.docstore.document import Document

with open(config.OUT_PATH / "chunks", "rb") as fp:
    chunks = pickle.load(fp)

split_docs = [Document(page_content=text, metadata={"source": "local"}) for text in chunks]

map_prompt_template = """Wite a 75-100 word summary of the following text:
{text}
CONCISE SUMMARY:"""

combine_prompt_template = 'Write a ' + str(500) + """-word summary of the following, removing irrelevant information. Finish your answer:
{text}
""" + str(500) + """-WORD SUMMARY:"""


map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

chain = load_summarize_chain(chain_type="map_reduce",
                             map_prompt = map_prompt,
                             combine_prompt = combine_prompt,
                             return_intermediate_steps = True,
                             llm = llm,
                             reduce_llm = llm)

out = chain({"input_documents": split_docs})
print()