import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def render_prompt(question):
    return f"<|user|>\n{question}<|end|>\n<|assistant|>\n"


def render_completion(answer):
    return f"{answer}<|end|>"


def encode_example(tokenizer, question, answer, max_len):
    prompt = render_prompt(question)
    full = prompt + render_completion(answer)
    full_ids = tokenizer(full)["input_ids"]
    prompt_ids = tokenizer(prompt)["input_ids"]
    n_prompt = len(prompt_ids)
    if len(full_ids) > max_len or full_ids[:n_prompt] != prompt_ids:
        return None
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def load_split(path, tokenizer, max_len):
    rows, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d.get("question", "").strip(), d.get("answer", "").strip()
            if not q or not a or a.upper().startswith("NOT FOUND"):
                continue
            enc = encode_example(tokenizer, q, a, max_len)
            if enc is None:
                dropped += 1
                continue
            rows.append(enc)
    print(f"{path}: {len(rows)} examples, dropped {dropped}")
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-len", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = load_split(args.train, tokenizer, args.max_len)
    valid_ds = load_split(args.valid, tokenizer, args.max_len)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"precision: {'bf16' if USE_BF16 else 'fp16'}")
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
