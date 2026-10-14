"""QLoRA fine-tuning for a Colab T4 (16 GB, fp16, no bf16).

Differences from v2:
  * prompts come from the model's own chat template + the shared system prompt
  * the best epoch is picked by validation ACCURACY (greedy answers, refusal-aware key-fact match), not
    eval loss -- loss bottoms out early while fact recall keeps improving
  * stops early if validation accuracy has not improved for --patience epochs
  * LoRA on all linear layers, rank 64

    python train_lora_v3.py --train data_v3/train.jsonl --valid data_v3/valid.jsonl --output-dir adapters/qwen3b
The best adapter is saved to <output-dir>/best.
"""

import argparse
import inspect
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments)

from v3_common import correctness, read_jsonl
from v3_llm import DEFAULT_MODEL, generate_greedy, load_model, load_tokenizer, render_full


def encode(tokenizer, model_name, question, answer, max_len):
    prompt, full = render_full(tokenizer, model_name, question, answer)
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if len(full_ids) > max_len:
        return None
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def build_dataset(path, tokenizer, model_name, max_len):
    rows, dropped = [], 0
    for r in read_jsonl(path):
        enc = encode(tokenizer, model_name, r["question"], r["answer"], max_len)
        if enc is None:
            dropped += 1
        else:
            rows.append(enc)
    print(f"{path}: {len(rows)} examples, dropped {dropped}")
    return Dataset.from_list(rows)


class ValAccuracyCallback(TrainerCallback):
    """After each epoch: greedy-decode a validation subset, keep the adapter if accuracy improved."""

    def __init__(self, tokenizer, model_name, valid_rows, out_dir, patience, eval_batch):
        self.tok, self.model_name, self.rows = tokenizer, model_name, valid_rows
        self.out_dir, self.patience, self.eval_batch = Path(out_dir), patience, eval_batch
        self.best, self.best_epoch, self.curve = -1.0, 0, []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        use_cache = model.config.use_cache
        model.config.use_cache = True
        self.tok.padding_side = "left"  # batched generation needs left padding; training uses right
        preds, _, _ = generate_greedy(model, self.tok, self.model_name, [r["question"] for r in self.rows],
                                      max_new_tokens=64, batch_size=self.eval_batch)
        self.tok.padding_side = "right"
        model.config.use_cache = use_cache
        model.train()
        scored = [correctness(p, r["answer"], r["is_idk"]) for p, r in zip(preds, self.rows)]
        strict = sum(s[0] for s in scored) / len(scored)
        keyfact = sum(s[1] for s in scored) / len(scored)
        self.curve.append({"epoch": epoch, "val_acc_keyfact": keyfact, "val_acc_f1": strict})
        print(f"[epoch {epoch}] val accuracy key-fact {keyfact:.3f}  F1 {strict:.3f}  (best {max(self.best, keyfact):.3f})")
        if keyfact > self.best:
            self.best, self.best_epoch = keyfact, epoch
            model.save_pretrained(self.out_dir / "best")
            self.tok.save_pretrained(self.out_dir / "best")
        elif epoch - self.best_epoch >= self.patience:
            print(f"no improvement for {self.patience} epochs, stopping")
            control.should_training_stop = True
        (self.out_dir / "val_curve.json").write_text(json.dumps(self.curve, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data_v3/train.jsonl")
    p.add_argument("--valid", default="data_v3/valid.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--eval-n", type=int, default=200, help="validation questions decoded after each epoch")
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    tokenizer = load_tokenizer(a.model, "right")
    model = load_model(a.model, four_bit=not a.no_4bit)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, target_modules="all-linear",
                                             lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_ds = build_dataset(a.train, tokenizer, a.model, a.max_len)
    valid_rows = read_jsonl(a.valid)
    random.Random(a.seed).shuffle(valid_rows)
    valid_rows = valid_rows[:a.eval_n]

    fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    steps_per_epoch = -(-len(train_ds) // (a.batch_size * a.grad_accum))
    warmup_steps = max(1, int(0.03 * steps_per_epoch * a.epochs))  # warmup_ratio was removed in newer transformers
    train_kwargs = dict(
        output_dir=a.output_dir, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=warmup_steps, logging_steps=20, eval_strategy="no",
        save_strategy="no", bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), fp16=fp16,
        seed=a.seed, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", remove_unused_columns=False, report_to="none", group_by_length=True)
    # transformers keeps renaming/removing arguments; drop any this version does not know instead of crashing
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    skipped = sorted(k for k in train_kwargs if k not in supported)
    if skipped:
        print(f"note: this transformers version does not accept {skipped}; ignoring them")
    targs = TrainingArguments(**{k: v for k, v in train_kwargs.items() if k in supported})

    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    cb = ValAccuracyCallback(tokenizer, a.model, valid_rows, a.output_dir, a.patience, a.eval_batch)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, callbacks=[cb],
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    trainer.train()
    meta = {"base_model": a.model, "best_epoch": cb.best_epoch, "best_val_acc_keyfact": cb.best, "args": vars(a)}
    (Path(a.output_dir) / "best" / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"best epoch {cb.best_epoch}, val accuracy {cb.best:.3f}; adapter at {a.output_dir}/best")


if __name__ == "__main__":
    main()
