from transformers import LlamaForCausalLM
from transformers import AutoTokenizer
from src.config import config
import time

start = time.time()
model = LlamaForCausalLM.from_pretrained(config.LLAMA2)

tokenizer = AutoTokenizer.from_pretrained(config.LLAMA2)
text = "A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations (computation) automatically."
prompt = "Write 1 sentence concise summary for the following text: {text}. CONCISE SUMMARY:"

temperature = 0.5
max_length = 3000  # the max number of tokens
input_ids = tokenizer.encode(prompt, return_tensors="pt")
generate_ids = model.generate(input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
print(f"total time taken: {time.time() - start}")