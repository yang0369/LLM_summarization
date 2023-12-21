import time
from load_and_chunk import ProcessingPipeline

from summarize_long import summarize_long_text_by_llama2
from langchain.embeddings import VertexAIEmbeddings
from rouge_score import rouge_scorer
import tensorflow as tf
import sys
import jsonlines


def load_json_SLR_dataset(args):
    jsonl_file = args.i
    testset = []
    input_headers = args.header_input.split('/')
    truth_headers = args.header_truth.split('/')
    assert (2 >= len(input_headers) >= 1)
    assert (2 >= len(truth_headers) >= 1)

    with jsonlines.open(jsonl_file, 'r') as reader:
        for sample in reader:
            d = {}
            d["citation_number"] = sample["citation_number"]
            d["neutral_citation"] = sample["neutral_citation"]

            d["input"] = sample[input_headers[0]][input_headers[1]] if len(input_headers) == 2 else \
                         sample[input_headers[0]]
            if 'facts' in truth_headers or 'holdings' in truth_headers:
                d["truth"] = sample[truth_headers[0]][truth_headers[1]] if len(truth_headers) == 2 else \
                         sample[truth_headers[0]]
            elif 'headnotes' == truth_headers[0]:
                d["truth"] = 'Facts\n\n' + sample['headnotes']['facts'] + '\n\nHoldings\n\n' + sample['headnotes']['holdings']
            else:
                raise Exception("Invalid header names name for SLR data")
            testset.append(d)

    return testset


def load_testset(args):
    """ Load testset file according to its format: json/tsv or folder containing json files to be summarised
        :return: A list of dict: containing "input" and "truth"
    """

    # Use rsplit to split the file path and get the file type
    file_type = args.i.rsplit('.', 1)[-1].lower()

    if "jsonl" == file_type:
        if 'SLR_' in args.i:
            testset = load_json_SLR_dataset(args)

    return testset


def rouge(target, prediction, score_keys=None):
    """Computes rouge score.

    Args:
      target: string
      prediction: string
      score_keys: list of strings with the keys to compute.
    Returns:
      dict with score_key: rouge score of target and prediction
    """

    if score_keys is None:
        score_keys = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(score_keys)

    def _prepare_summary(summary):
        # Make sure the summary is not bytes-type
        # Add newlines between sentences so that rougeLsum is computed correctly.
        summary = summary.replace(" . ", " .\n")
        return summary

    target = _prepare_summary(target)
    prediction = _prepare_summary(prediction)
    return scorer.score(target=target, prediction=prediction)


def compute_metrics(reference_text, hypothesis_text):
    """ Compute ROUGE and Semantic-Similarity scores for a pair of text strings
        :return: a dict containg scores: Rouge-1,2,L and semantic-similarity
    """
    scores = {
        "rouge_1_r": 0.0,
        "rouge_1_p": 0.0,
        "rouge_1_f": 0.0,
        "rouge_2_r": 0.0,
        "rouge_L_r": 0.0,
        "semantic_similarity": 0.0,
    }

    if hypothesis_text.split():
        # Compute ROUGE scores
        rouge_scores = rouge(reference_text, hypothesis_text)
        scores['rouge_1_r'] = float(f"{rouge_scores['rouge1'].recall:.3f}")
        scores['rouge_1_p'] = float(f"{rouge_scores['rouge1'].precision:.3f}")
        scores['rouge_1_f'] = float(f"{rouge_scores['rouge1'].fmeasure:.3f}")
        scores['rouge_2_r'] = float(f"{rouge_scores['rouge2'].recall:.3f}")
        scores['rouge_L_r'] = float(f"{rouge_scores['rougeLsum'].recall:.3f}")

        # Compute Semantic Similarity scores
        # use_score = use_scores([reference_text], [hypothesis_text])
        # scores['semantic_similarity'] = float(f"{use_score[0][0]:.3f}")

    return scores


def compute_average_scores(all_scores: list):
    """ Compute average ROUGE and Semantic-Similarity scores for the whole testset.
        :return: a dict containing average scores: Rouge-1,2,L and semantic-similarity
    """
    # Buffers for keeping scores
    scores_rouge_1_r = []
    scores_rouge_1_p = []
    scores_rouge_1_f = []
    scores_rouge_2_r = []
    scores_rouge_L_r = []

    # scores_semantic = []
    r_ratio = []

    # Place individual scores to according buffers
    for scores in all_scores:
        scores_rouge_1_r.append(scores["rouge_1_r"])
        scores_rouge_1_p.append(scores["rouge_1_p"])
        scores_rouge_1_f.append(scores["rouge_1_f"])
        scores_rouge_2_r.append(scores["rouge_2_r"])
        scores_rouge_L_r.append(scores["rouge_L_r"])

        # scores_semantic.append(scores["semantic_similarity"])
        if 'ratio' in scores.keys():
            r_ratio.append(1/scores['ratio'])

    # Compute average scores
    average_scores = {}
    average_scores['rouge_1_r'] = float(f"{average_score(scores_rouge_1_r):.3f}")
    average_scores['rouge_1_p'] = float(f"{average_score(scores_rouge_1_p):.3f}")
    average_scores['rouge_1_f'] = float(f"{average_score(scores_rouge_1_f):.3f}")
    average_scores['rouge_2_r'] = float(f"{average_score(scores_rouge_2_r):.3f}")
    average_scores['rouge_L_r'] = float(f"{average_score(scores_rouge_L_r):.3f}")

    # average_scores['semantic_similarity'] = float(f"{average_score(scores_semantic):.3f}")
    if len(r_ratio) > 0:
        average_scores['ratio'] = float(f"{1/average_score(r_ratio):.1f}")

    return average_scores


def extract_text_from_json(jsondict: list) -> str:
    """ Extract Judgment text from SAL JSON dict data.
        Output:
            text string for judgment
    """
    output_buf = []

    for data in jsondict:

        # Extract header text
        output_buf.append(data['header']['text'])

        # Extract paragraph text
        for parag_data in data['paragraphs']:
            if parag_data['paragraph_number']:
                output_buf.append(parag_data['paragraph_number'] + ' ' + parag_data['text'])
            else:
                output_buf.append(parag_data['text'])

        # Extract table text
        for table_data in data['tables']:
            rows = [row.replace('\t', ' | ') for row in table_data]
            output_buf.append('\n'.join(rows))

    text = '\n\n'.join(output_buf)
    return text


def summarize_long_by_clustering(
    text_or_jsondict,
    max_tokens):

    pro = ProcessingPipeline(VertexAIEmbeddings())
    chunks = pro.process_document(text_or_jsondict)
    sum_gen_d = summarize_long_text_by_llama2(chunks, max_tokens)

    return sum_gen_d


def summarize_general(text_or_jsondict,
                      prompt_prefix: str,
                      prompt_postfix: str,
                      config: dict,
                      model: str,
                      max_tokens: int,
                      merge: str,
                      task: str,
                      specific_section: str = None):
    """
    @param text_or_jsondict: plain text or json filepath
    @param prompt_prefix: prompt prefix for summarisation
    @param prompt_postfix: prompt postfix for summarisation
    @param config: model config parameters dict
    @param model: model name in config file
    @param max_tokens: max tokens in 1 segment includes input AND output.
    @param merge: segment merging method
    @param task: qa or sm
    @param specific_section: only for json; specify a specific section to summarise
    @return: output dictionary
    """
    prompt_postfix = ''.join(prompt_postfix) if type(prompt_postfix) is list else prompt_postfix
    prompt_postfix = ''.join(prompt_postfix) if type(prompt_postfix) is list else prompt_postfix

    if isinstance(text_or_jsondict, list):
        text_or_jsondict = extract_text_from_json(text_or_jsondict)

    sum_output = summarize_long_by_clustering(text_or_jsondict, max_tokens)

    return sum_output


# TODO: todel use_model
def benchmark(testset: list, config: dict, task: str, gptx_model: str, max_tokens=2048, merge='stage',
              o_tsv=False, peft_app=None):
    """ Run benchmarking of summarization
        @param testset: test set to benchmark on (see usage at top of file)
        @param config: model config parameters dict
        @param task: qa or sm
        @param gptx_model: model name in config json file
        @param max_tokens: max tokens in one segmentation. NOTE: for stage method, includes input AND output.
        @param merge: merge method used. join, summary or stage
        @param o_tsv: need output TSV file for the result list
        @peft_app: use peft tuned model (qlora or prompt_tune)
        @return: average scores, output list with model outputs and individual scores
    """
    # Buffers to store results
    output_lst = []
    all_scores = []

    prompt = config[task]["prompt"]["openai" if "openai" in gptx_model.lower() else "gptx"]

    prompt_head = ''.join(prompt["head"]) if isinstance(prompt["head"], list) else prompt["head"]
    prompt_tail = ''.join(prompt["tail"]) if isinstance(prompt["tail"], list) else prompt["tail"]

    if max_tokens == 0:
        max_tokens = config["context_length"][gptx_model] if gptx_model in config["context_length"] \
            else 4000

    # Benchmark loop
    for n, data in enumerate(testset):
        generated_text = summarize_general(
            data["input"], prompt_head, prompt_tail,
            config, gptx_model, max_tokens, merge, task, None
        )["summary"]

        print(f"{80 * '-'}\n#{n + 1}: {generated_text}", flush=True)

        if data["truth"]: # ignore the test item that has empty truth
            # Compute evaluation scores
            with tf.device('/CPU:0'):
                # to calculate rouge score of large models, we have to raise the recursion limit
                sys.setrecursionlimit(5000 * 5000)
                scores = compute_metrics(data["truth"], generated_text)
                if task == "sm":
                    scores['ratio'] = float(f'{(len(data["input"].split()) / len(generated_text.split())):.1f}')

                sys.setrecursionlimit(1000)
            all_scores.append(scores)

            # Save info for output
            d = {}
            d["sn"] = n + 1
            if task != 'sm':
                d["input"] = data["input"]
            d["truth"] = data["truth"]
            d["output"] = generated_text
            d["scores"] = scores
            for key in data.keys():
                if key not in ['input', 'truth']:
                    d[key] = data[key]

            output_lst.append(d)

            print(f"scores: {scores}", flush=True)

        del generated_text

    # Compute average scores for the whole testset
    average_scores = compute_average_scores(all_scores)
    return average_scores, output_lst


if __name__ == "__main__":

    # Load testset file
    testset = load_testset("/Users/kewenyang/Documents/GitHub/LLM_summarization/data/SLR/SLR_Short.jsonl")

    model_path_type = args.model[5:]  # md format: gptx:model_path:[model_type]
    data_type = torch.float16 if args.fp16 else torch.float32

    if args.peft_app == "qlora":
        data_type = torch.bfloat16
    # util_call_model_api.call_model_api_general.gptx_generator = GPTXGenerator(
    #     model_path_type, data_type,
    #     args.ds_inference, load_in_8bit=True,
    # )
    util_call_model_api.call_model_api_general.gptx_generator = None

    # with tf.device('/CPU:0'):
    #     # Load evaluation model
    #     use_model = hub.load(config["use_model"])

    # Run benchmark
    start = time.time()
    average_scores, output_lst = benchmark(
        testset, config_cp, args.task, args.model,
        args.max_tokens, args.merge, args.o_tsv, args.peft_app
    )
    end = time.time()
    average_time = float(f"{(end - start) / len(testset):.1f}")
    print(f"\nBenchmark average time: {average_time} seconds")
    print_json(average_scores)

    # Prepare output data
    output = {}
    output['task'] = args.task
    output['testset'] = args.i
    output['model'] = args.model
    if 'ratio' in config[args.task].keys():
        output['ratio'] = config[args.task]['ratio']

    output['config'] = config[args.task]['openai'] if 'openai' in args.model.lower() else config[args.task]['gptx']
    if args.task == "qa":
        output['prompt'] = config[args.task]["prompt"]["Model only"]
    else:
        output['prompt'] = config[args.task]["prompt"]["openai" if "openai" in args.model.lower() else "gptx"]

    output['average_scores'] = average_scores
    output['average_time'] = average_time
    output['results'] = output_lst

    # Save output to json fille
    with open(args.o, 'w') as file:
        json.dump(output, file, indent=4)
        print('\nOutput file saved: %s' % args.o)

    if args.o_tsv:
        tsv_file = args.o.rsplit('.', 1)[0] + '.tsv'
        write_dict_list_to_tsv(output_lst, tsv_file)
        print('\nOutput TSV file saved: %s' % tsv_file)
