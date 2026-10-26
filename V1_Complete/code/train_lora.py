"""
QLoRA fine-tuning of Phi-3-mini-4k-instruct on the VIT Q&A dataset.

Sized for a 4GB laptop GPU (RTX 3050 Ti): 4-bit base model + small-rank LoRA
adapters + batch size 1 with gradient accumulation + gradient checkpointing.
Only the LoRA adapters are trained, so this fits in a few GB.

BEFORE RUNNING:
    pip install bitsandbytes peft trl datasets transformers accelerate
    (you already have these installed per your log)

DATASET FORMAT EXPECTED:
    A .jsonl file where each line is a JSON object with at least:
        {"question": "...", "answer": "...", ...other fields ignored...}

    IMPORTANT: this script automatically SKIPS any row whose answer starts
    with "NOT FOUND" (the placeholder used in the VIT dataset for facts that
    weren't sourced). Training on those would teach the model to output a
    canned refusal phrase regardless of the actual question, which is exactly
    the kind of thing that wrecks calibration (a rote non-answer isn't real
    uncertainty). Fill those in or drop them before training if you haven't
    already filtered them.
"""

import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ---------- CLI (optional -- falls back to the defaults below if omitted) ----------

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=None, help="Path to training JSONL (overrides DATA_PATH below)")
parser.add_argument("--output-dir", type=str, default=None, help="Adapter output folder (overrides OUTPUT_DIR below)")
cli_args, _ = parser.parse_known_args()

# ---------- CONFIG ----------

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH = cli_args.data or "combined_training_data.jsonl"
OUTPUT_DIR = cli_args.output_dir or "phi3-vit-lora-expc"
EVAL_SPLIT_PATH = "eval_set.jsonl"         # held-out set, exported below for eval_compare.py
MAX_SEQ_LENGTH = 512                        # raise if your Q&A pairs run long, lower if you OOM

# ---------- 1. LOAD BASE MODEL IN 4-BIT ----------

print("Loading tokenizer and base model in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Training data : {DATA_PATH}")
print(f"Output dir    : {OUTPUT_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Phi-3 has no pad token by default

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # required alongside gradient checkpointing

# ---------- 2. ATTACH LORA ADAPTERS ----------

lora_config = LoraConfig(
    r=8,                    # lower (e.g. 4) if you hit OOM; raise (16) if you have headroom
    lora_alpha=16,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],  # Phi-3 module names
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # sanity check: should be a small % of total params

# ---------- 3. LOAD AND FORMAT THE DATASET ----------

def load_records(path):
    records = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d.get("question", "").strip(), d.get("answer", "").strip()
            if not q or not a:
                continue
            if a.upper().startswith("NOT FOUND"):
                skipped += 1
                continue
            records.append({
                "text": f"<|user|>\n{q}<|end|>\n<|assistant|>\n{a}<|end|>",
                "question": q,   # kept alongside "text" so the held-out split
                "answer": a,     # below can be reused directly by eval_compare.py
            })
    print(f"Loaded {len(records)} training examples, skipped {skipped} unanswered rows.")
    return records

records = load_records(DATA_PATH)

# Truncate any overly long examples ourselves, rather than relying on a
# trainer-level max-length argument (field name for this varies across
# trl versions and changed again in some releases -- doing it manually
# here avoids depending on that).
def truncate_text(text, max_tokens=MAX_SEQ_LENGTH):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids)

before = len(records)
for r in records:
    r["text"] = truncate_text(r["text"])
print(f"Truncated any examples longer than {MAX_SEQ_LENGTH} tokens ({before} total examples).")

dataset = Dataset.from_list(records)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Export the held-out split so eval_compare.py evaluates on data the model
# never trained on. Without this, "comparison" would really be measuring
# memorization, not generalization.
dataset["test"].to_json(EVAL_SPLIT_PATH)
print(f"Held-out eval set ({len(dataset['test'])} examples) saved to {EVAL_SPLIT_PATH}")

# ---------- 4. TRAIN ----------

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch size 8
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=False,
    bf16=True,                       # RTX 3050 Ti (Ampere) supports bf16; avoids the
                                      # fp16 GradScaler crashing on bf16 tensors
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",        # memory-efficient optimizer, needs bitsandbytes
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

print("Starting training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. LoRA adapter saved to ./{OUTPUT_DIR}")
print("This is just the adapter, not a merged model; see the note below for how to load it.")

# ---------- 5. HOW TO LOAD THE ADAPTER LATER FOR INFERENCE/EVAL ----------
# from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
# model = PeftModel.from_pretrained(base, OUTPUT_DIR)
# (then generate exactly as in test_inference.py)
