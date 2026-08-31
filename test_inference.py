"""
Quick sanity check: loads Phi-3-mini in 4-bit and confirms generation works
before you spend time on the training run.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

print("Loading tokenizer and model in 4-bit on GPU...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

prompt = "<|user|>\nWhat is 2+2?<|end|>\n<|assistant|>\n"

# tokenizer(...) returns a dict-like BatchEncoding, not a tensor.
# .to(model.device) moves every tensor inside it to the right device.
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating a response...")
# The fix: unpack with ** instead of passing the BatchEncoding positionally.
out = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(out[0], skip_special_tokens=True))
