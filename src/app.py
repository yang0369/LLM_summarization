from io import StringIO

import streamlit as st
from langchain.docstore.document import Document
from langchain.embeddings import VertexAIEmbeddings
from streamlit.components.v1 import html

from load_and_chunk import ProcessingPipeline
from summarize_long import summarize_long_text_by_custom

st.title('Long Text Summarization')

input_text = st.text_area('Please paste the text you want to summarise below')
uploaded_file = st.file_uploader("Or choose a file (supported type: .txt)")
if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode('unicode_escape'))

    # To read file as string:
    input_text = stringio.read()

with st.container():
    st.header('Input Statistics:', divider='rainbow')
    html(input_text, height=100, scrolling=True)
    st.write(f'Number Of Words: {len(input_text.split(" "))}')

if_summarise = st.button("Summarize", type="primary")

if if_summarise and len(input_text) > 0:
    with st.container():
        if len(input_text.split(" ")) > 500:
            pro = ProcessingPipeline(VertexAIEmbeddings())
            chunks = pro.process_document(input_text)

            # temporarily summarize for first 10 chunks due to small context window of VertexAI model
            chunks = chunks[:10]
            split_docs = [Document(page_content=text, metadata={"source": "local"}) for text in chunks]
        else:
            split_docs = [Document(page_content=input_text, metadata={"source": "local"})]

        summary = summarize_long_text_by_custom(split_docs)

        st.divider()
        st.header('Summary:', divider='rainbow')
        st.write(f'{summary}')
        st.write(f'Number Of Words: {len(summary.split(" "))}')
