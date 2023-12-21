from typing import List

import streamlit as st
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatVertexAI
from langchain.docstore.document import Document
from langchain.embeddings import VertexAIEmbeddings
from langchain.prompts import PromptTemplate

from config import config
from load_and_chunk import ProcessingPipeline
from project.src.search.util.util_eval import post_process
from utilities.custom_logger import CustomLogger

logger = CustomLogger()


# OPTION_1: BY LANGCHAIN MAP REDUCE
# llm chain
# @st.cache_data
def summarize_long_text_by_langchain(_docs: List[Document]) -> str:
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

    summary = chain({"input_documents": _docs})  # input_documents is required keyword for combined chain

    logger.info("\n\ninput text:\n")
    for doc in _docs:
        logger.info(f"\n{doc.page_content}\n")
    logger.info(f"\nsummarization: \n{summary['output_text'].strip()}\n")
    return summary['output_text'].strip()


# OPTION_2: BY CUSTOM WAY
# @st.cache_data
from transformers import LlamaTokenizerFast, LlamaForCausalLM
import torch

def summarize_long_text_by_custom(_docs: List[Document], max_tokens) -> str:

    MODEL_PATH = "/raid2/domain_ft/models/llama2/Llama-2-13b-chat-hf-slr-qlora-merged-2"
    data_type = torch.float16
    # Define LLM chain
    llm = LlamaForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=data_type,
                return_dict=True,
                load_in_8bit=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

    tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_PATH)
    facts = []
    holdings = []

    logger.info("############# summarize for facts #############")
    for id, seg in enumerate(_docs):
        torch.cuda.empty_cache()
        seg_len = tokenizer.encode(seg, return_tensors='pt').cuda().shape[1]
        logger.info(f"The original size of the {id}th chunk: {seg_len}")
        # head = [
        #     "<s>[INST] <<SYS>>\n",
        #     "You are an experienced and helpful lawyer. ",
        #     "You are given a text chunk (delimited by triple backticks) taken from a long legal judgment document.\n",
        #     "<</SYS>>\n\n",
        #     "Write a concise, well-structured and professional extractive summary that highlights the facts of the case whenever possible. ",
        #     "Ensure that the summary can inform a legal professional of the case's facts without requiring reference to the full judgment. ",
        #     "The summary needs to be accurate and based on the text. Double check against the original document.\n\n",
        #     "Text:\n",
        #     "```\n"
        #     ]
        head = [
            "<s>[INST] <<SYS>>\n",
            "You are an experienced and helpful lawyer. ",
            "You are given a text chunk (delimited by triple backticks) taken from a legal judgment document.\n",
            "<</SYS>>\n\n",
            "Extract the legal facts from the text chunk. ",
            "Ensure that:\n",
            "1. the extracted facts include all the key legal facts.\n",
            "2. the extracted facts align accurately with the original text.\n",
            "3. the extracted facts are as concise as possible.\n\n",
            "Text:\n",
            "```\n"
            ]
        tail = "\n```\n\n[/INST]\n\nThe extracted facts:\n"
        prompt = "".join(head) + seg + tail
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        prompt_len = prompt_ids.shape[1]

        penalty_factor = (seg_len // 100) ** 1.2 * 0.2
        adjust_ratio = 3 + penalty_factor
        new_token_len = int(seg_len / adjust_ratio)
        # word_max is indicated in config
        max_new_tokens = min(new_token_len, 4000 - prompt_len)
        min_new_tokens = (max_new_tokens + 1) // 2
        temperature = max(0.0001, 0.5)

        with torch.no_grad():
            pred = llm.generate(
                prompt_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_p=1.0,
                top_k=5,
                temperature=temperature,
                repetition_penalty=1.0,
                bad_words_ids=None,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                remove_invalid_values=True,
            )

        pred_tokens = [output[prompt_len:] for output in pred]
        summary = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)[0]
        summary = post_process(summary)
        logger.info(f"The size after summarization: {len(tokenizer.encode(summary))}")
        if tokenizer.encode(summary, return_tensors='pt').shape[0] > 10:
            facts.append(summary)

        del seg, seg_len, prompt_ids, pred, pred_tokens, summary
        torch.cuda.empty_cache()

    logger.info("############# summarize for holdings #############")
    for id, seg in enumerate(_docs):
        torch.cuda.empty_cache()
        seg_len = tokenizer.encode(seg, return_tensors='pt').cuda().shape[1]
        logger.info(f"The original size of the {id}th chunk: {seg_len}")
        head = [
            "<s>[INST] <<SYS>>\n",
            "You are an experienced and helpful lawyer. ",
            "You are given a text chunk (delimited by triple backticks) taken from a legal judgment document.\n",
            "<</SYS>>\n\n",
            "Extract the judge's decisions from the text chunk. ",
            "Ensure that:\n",
            "1. the extracted judge's decisions include all the key points.\n",
            "2. the extracted judge's decisions align accurately with the original text.\n",
            "3. the extracted judge's decisions are as concise as possible.\n\n",
            "Text:\n",
            "```\n"
        ]
        tail = "\n```\n\n[/INST]\n\nThe extracted judge's decisions:\n"
        prompt = "".join(head) + seg + tail
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        prompt_len = prompt_ids.shape[1]

        penalty_factor = (seg_len // 100) ** 1.2 * 0.2
        adjust_ratio = 2 + penalty_factor
        new_token_len = int(seg_len / adjust_ratio)
        # word_max is indicated in config
        max_new_tokens = min(new_token_len, 4000 - prompt_len)
        min_new_tokens = (max_new_tokens + 1) // 2
        temperature = max(0.0001, 0.5)

        with torch.no_grad():
            pred = llm.generate(
                prompt_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_p=1.0,
                top_k=5,
                temperature=temperature,
                repetition_penalty=1.0,
                bad_words_ids=None,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                remove_invalid_values=True,
            )

        pred_tokens = [output[prompt_len:] for output in pred]
        summary = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)[0]
        summary = post_process(summary)
        logger.info(f"The size after summarization: {len(tokenizer.encode(summary))}")
        if tokenizer.encode(summary, return_tensors='pt').shape[0] > 10:
            facts.append(summary)
        holdings.append(summary)

        del seg, seg_len, prompt_ids, pred, pred_tokens, summary
        torch.cuda.empty_cache()

    facts = "\n".join(facts)
    holdings = "\n".join(holdings)

    # join the segments summaries by default
    sum_gen_d = {"segments": None}
    sum_gen_d["summary"] = facts + "\n" + holdings

    return sum_gen_d


# OPTION_3: BY finetuned LLAMA2
# @st.cache_data
from transformers import LlamaTokenizerFast, LlamaForCausalLM
import torch


def summarize_long_text_by_llama2(_docs: List[Document], max_tokens) -> str:

    MODEL_PATH = "/raid2/domain_ft/models/llama2/Llama-2-13b-chat-hf-slr-qlora-merged-2"
    data_type = torch.float16
    # Define LLM chain
    llm = LlamaForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=data_type,
                return_dict=True,
                load_in_8bit=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

    tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_PATH)
    facts = []
    holdings = []

    logger.info("############# extract facts #############")
    for id, seg in enumerate(_docs):
        torch.cuda.empty_cache()
        seg_len = tokenizer.encode(seg, return_tensors='pt').cuda().shape[1]
        logger.info(f"The original size of the {id}th chunk: {seg_len}")
        head = [
            "<s>[INST] <<SYS>>\n",
            "You are an experienced and honest lawyer. ",
            "You are given a text chunk (delimited by triple backticks) taken from a legal judgment document.\n",
            "<</SYS>>\n\n",
            "Extract the legal facts from the text chunk. ",
            "Ensure that:\n",
            "1. the extracted facts include all the key legal facts.\n",
            "2. the extracted facts align accurately with the original text.\n",
            "3. the extracted facts are as concise as possible.\n\n",
            "Text:\n",
            "```\n"
            ]
        tail = "\n```\n\n[/INST]\n\nThe extracted facts:\n"
        prompt = "".join(head) + seg + tail
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        prompt_len = prompt_ids.shape[1]

        token_limit = min(seg_len, 4000)
        penalty_factor = (token_limit - 12000) / 12000 * 0.66
        adjust_ratio = 11 * (1 + penalty_factor)
        new_token_len = int(token_limit / adjust_ratio)
        # word_max is indicated in config
        max_new_tokens = min(new_token_len, 4000 - prompt_len)
        min_new_tokens = (max_new_tokens + 1) // 2

        with torch.no_grad():
            pred = llm.generate(
                prompt_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_p=1.0,
                top_k=config.TOP_K,
                temperature=config.TEM,
                repetition_penalty=1.0,
                bad_words_ids=None,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                remove_invalid_values=True,
            )

        pred_tokens = [output[prompt_len:] for output in pred]
        summary = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)[0]
        summary = post_process(summary)
        size = tokenizer.encode(summary, return_tensors='pt').shape[1]
        logger.info(f"The size after summarization: {size}")
        if size > 10:
            summary = config.remove_index(summary, r"[\n]+", "(\\d+\\.? +)|(\\d+\\.?)|(\\(\\w\\) +)", "\n")
            facts.append(summary)
            logger.info(summary)

        del seg, seg_len, prompt_ids, pred, pred_tokens, summary, size
        torch.cuda.empty_cache()

    text_splitter = RecursiveCharacterTextSplitter(
        keep_separator=False,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=0,
        length_function=config.get_num_of_tokens,
        is_separator_regex=False,
        separators=["\n\n"],
    )

    logger.info("############# final summarize for facts #############")
    # apply recursive character split each chunk into paragraphs
    facts = "\n\n".join(facts)
    segments = text_splitter.split_text(facts)
    facts = list()
    for seg in segments:
        seg_len = tokenizer.encode(seg, return_tensors='pt').cuda().shape[1]
        head = [
            "<s>[INST] <<SYS>>\n",
            "You are an experienced and helpful lawyer. ",
            "You are given a text chunk (delimited by triple backticks) taken from a legal judgment document.\n",
            "<</SYS>>\n\n",
            "Summarise the key facts from the text chunk. ",
            "Ensure that:\n",
            "1. the summary includes all the key legal facts.\n",
            "2. the summary aligns accurately with the original text chunk.\n",
            "3. the summary is as concise as possible.\n\n",
            "4. the summary has no duplicated information.",
            "Text:\n",
            "```\n"
            ]
        tail = "\n```\n\n[/INST]\n\nThe extracted facts:\n"
        prompt = "".join(head) + seg + tail
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        prompt_len = prompt_ids.shape[1]

        max_new_tokens = min(seg_len, 4000 - prompt_len)
        min_new_tokens = (max_new_tokens + 1) // 2

        with torch.no_grad():
            pred = llm.generate(
                prompt_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_p=1.0,
                top_k=config.TOP_K,
                temperature=config.TEM,
                repetition_penalty=1.0,
                bad_words_ids=None,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                remove_invalid_values=True,
            )

        pred_tokens = [output[prompt_len:] for output in pred]
        summary = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)[0]
        facts.append(post_process(summary))
        size = tokenizer.encode(summary, return_tensors='pt').shape[1]
        logger.info(f"The facts' size after summarization: {size}")

        del seg_len, prompt_ids, pred, pred_tokens, summary, size
        torch.cuda.empty_cache()
    facts = "\n\n".join(facts)

    logger.info("############# extract holdings #############")
    for id, seg in enumerate(_docs):
        torch.cuda.empty_cache()
        seg_len = tokenizer.encode(seg, return_tensors='pt').cuda().shape[1]
        logger.info(f"The original size of the {id}th chunk: {seg_len}")
        head = [
            "<s>[INST] <<SYS>>\n",
            "You are an experienced and honest lawyer. ",
            "You are given a text chunk (delimited by triple backticks) taken from a legal judgment document.\n",
            "<</SYS>>\n\n",
            "Extract the judge's decisions from the text chunk. ",
            "Ensure that:\n",
            "1. the extracted judge's decisions include all the key points.\n",
            "2. the extracted judge's decisions align accurately with the original text.\n",
            "3. the extracted judge's decisions are as concise as possible.\n\n",
            "Text:\n",
            "```\n"
        ]
        tail = "\n```\n\n[/INST]\n\nThe extracted judge's decisions:\n"
        prompt = "".join(head) + seg + tail
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        prompt_len = prompt_ids.shape[1]

        token_limit = min(seg_len, 4000)
        penalty_factor = (token_limit - 12000) / 12000 * 0.66
        adjust_ratio = 7 * (1 + penalty_factor)
        new_token_len = int(token_limit / adjust_ratio)
        # word_max is indicated in config
        max_new_tokens = min(new_token_len, 4000 - prompt_len)
        min_new_tokens = (max_new_tokens + 1) // 2

        with torch.no_grad():
            pred = llm.generate(
                prompt_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_p=1.0,
                top_k=config.TOP_K,
                temperature=config.TEM,
                repetition_penalty=1.0,
                bad_words_ids=None,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                remove_invalid_values=True,
            )

        pred_tokens = [output[prompt_len:] for output in pred]
        summary = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)[0]
        summary = post_process(summary)
        size = tokenizer.encode(summary, return_tensors='pt').shape[1]
        logger.info(f"The size after summarization: {size}")
        if size > 10:
            summary = config.remove_index(summary, r"[\n]+", "(\\d+\\.? +)|(\\d+\\.?)|(\\(\\w\\) +)", "\n")
            holdings.append(summary)
            logger.info(summary)

        del seg, seg_len, prompt_ids, pred, pred_tokens, summary, size
        torch.cuda.empty_cache()

    logger.info("############# final summarize for holdings #############")
    holdings = "\n\n".join(holdings)
    segments = text_splitter.split_text(holdings)
    holdings = list()
    for seg in segments:
        seg_len = tokenizer.encode(seg, return_tensors='pt').cuda().shape[1]
        head = [
            "<s>[INST] <<SYS>>\n",
            "You are an experienced and helpful lawyer. ",
            "You are given a text chunk (delimited by triple backticks) taken from a legal judgment document.\n",
            "<</SYS>>\n\n",
            "Summarise the key judge's decisions from the text chunk. ",
            "Ensure that:\n",
            "1. the summary includes all the key points.\n",
            "2. the summary aligns accurately with the original text chunk.\n",
            "3. the summary is as concise as possible.\n\n",
            "4. the summary has no duplicated information.",
            "Text:\n",
            "```\n"
        ]
        tail = "\n```\n\n[/INST]\n\nThe extracted judge's decisions:\n"
        prompt = "".join(head) + seg + tail
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        prompt_len = prompt_ids.shape[1]

        max_new_tokens = min(seg_len, 4000 - prompt_len)
        min_new_tokens = (max_new_tokens + 1) // 2

        with torch.no_grad():
            pred = llm.generate(
                prompt_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_p=1.0,
                top_k=config.TOP_K,
                temperature=config.TEM,
                repetition_penalty=1.0,
                bad_words_ids=None,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                remove_invalid_values=True,
            )

        pred_tokens = [output[prompt_len:] for output in pred]
        summary = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)[0]
        holdings.append(post_process(summary))
        size = tokenizer.encode(summary, return_tensors='pt').shape[1]
        logger.info(f"The holdings' size after summarization: {size}")

        del seg_len, prompt_ids, pred, pred_tokens, summary
        torch.cuda.empty_cache()
    holdings = "\n\n".join(holdings)

    # join the segments summaries by default
    sum_gen_d = {"segments": None}
    sum_gen_d["summary"] = "Facts\n\n" + facts + "\n\nHoldings\n\n" + holdings

    return sum_gen_d


if __name__ == '__main__':
    # # load test data
    # import json
    # import pickle
    # with open(config.SLR_PATH / 'judiciary_32k_test.jsonl', 'r') as json_file:
    #     json_list = list(json_file)

    # slr = dict()
    # for idx, json_str in enumerate(json_list):
    #     slr[idx] = json.loads(json_str)

    # # take 1 example for analysis
    # document = slr[0]["judgment"]
    # logger.info(document)

    with open(config.SLR_PATH / 'test.text', 'rb') as f:
        document = f.read()
        document = document.decode('unicode_escape')

    pro = ProcessingPipeline(VertexAIEmbeddings())
    chunks = pro.process_document(document)

    # # load the text chunks
    # with open(config.OUT_PATH / "chunks", "rb") as fp:
    #     chunks = pickle.load(fp)

    # chunks = chunks[:5]  # try with first 5 chunks only

    # for ch in chunks:
    #     logger.info(f"chunck size: {get_num_of_tokens(ch)}")

    split_docs = [Document(page_content=text, metadata={"source": "local"}) for text in chunks]
    summarize_long_text_by_custom(split_docs)
    # summarize_long_text_by_langchain(split_docs)

if __name__ == '__main__':
    # # load test data
    # import json
    # import pickle
    # with open(config.SLR_PATH / 'judiciary_32k_test.jsonl', 'r') as json_file:
    #     json_list = list(json_file)

    # slr = dict()
    # for idx, json_str in enumerate(json_list):
    #     slr[idx] = json.loads(json_str)

    # # take 1 example for analysis
    # document = slr[0]["judgment"]
    # logger.info(document)

    with open(config.SLR_PATH / 'test.text', 'rb') as f:
        document = f.read()
        document = document.decode('unicode_escape')

    pro = ProcessingPipeline(VertexAIEmbeddings())
    chunks = pro.process_document(document)

    # # load the text chunks
    # with open(config.OUT_PATH / "chunks", "rb") as fp:
    #     chunks = pickle.load(fp)

    # chunks = chunks[:5]  # try with first 5 chunks only

    # for ch in chunks:
    #     logger.info(f"chunck size: {get_num_of_tokens(ch)}")

    split_docs = [Document(page_content=text, metadata={"source": "local"}) for text in chunks]
    summarize_long_text_by_custom(split_docs)
    # summarize_long_text_by_langchain(split_docs)
