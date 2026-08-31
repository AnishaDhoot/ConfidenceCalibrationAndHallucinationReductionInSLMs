from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer and model in 4-bit on GPU...")
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

messages = [{"role": "user", "content": "Say hello in one sentence."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

print("Generating a response...")
out = model.generate(inputs, max_new_tokens=50)
response = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)

print("\n--- Model response ---")
print(response)

print("\n--- GPU memory used ---")
print(f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
