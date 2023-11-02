from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from src.config import config
# import torch
# from huggingface_hub import login

# login("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")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token="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")

# data_type = torch.float16
model = AutoModelForCausalLM.from_pretrained(
    # config.LLAMA2,
    "meta-llama/Llama-2-7b-hf"
    # torch_dtype=data_type,
    # trust_remote_code=True,
    # load_in_8bit=False,
    # device_map="auto",
    # low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(config.LLAMA2)
text = "A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations (computation) automatically."
prompt = "Write 1 sentence concise summary for the following text: {text}. CONCISE SUMMARY:"

temperature = 0.5
max_length = 3000
input_ids = tokenizer.encode(prompt, return_tensors="pt")
generate_ids = model.generate(input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]