from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatVertexAI
from langchain.prompts import PromptTemplate

from config import config

# STUFF SUMMARIZATION
"""
Stuff the documents into context
>> Input: A list of documents
>> Output: summarized text

1. combine a list of docs into a single string with document_prompt and document_separator
2. feed the single string as document_variable_name to llm
3. llm_chain summarizes the document_variable_name
"""
document_prompt = PromptTemplate(
    input_variables=["page_content"],  # what we want to call the input string
    template="{page_content}"
)

prompt_template = """Write a concise summary of the following:
"{combined_string}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm = ChatVertexAI()
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_separator="\n\n",
    document_variable_name="combined_string"
)

out = stuff_chain({"input_documents": config.DOCS})  # "input_documents" is a fixed keyword as required by combine base class
print(f'\n\nthe summary: {out["output_text"]}\n\n')