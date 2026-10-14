"""Generate answers + confidence signals for the valid and test splits (GPU).

Per question it records the greedy answer, its mean token probability, and N sampled answers
(for self-consistency). Nothing is scored or calibrated here; calibrate_v3.py does that on CPU.

    python eval_v3.py --adapter adapters/qwen3b/best --splits valid test
Writes results_v3/rows_<split>.jsonl.
"""

import argparse
import json
from pathlib import Path

from peft import PeftModel
from tqdm import tqdm

from v3_common import read_jsonl
from v3_llm import (DEFAULT_MODEL, generate_greedy, generate_samples, load_model, load_tokenizer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", help="LoRA adapter dir; omit to evaluate the base model")
    p.add_argument("--model", default=None, help="base model; defaults to the one recorded with the adapter")
    p.add_argument("--data-dir", default="data_v3")
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out-dir", default="results_v3")
    p.add_argument("--tag", default="", help="suffix for output files, e.g. the run name")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    a = p.parse_args()

    model_name = a.model
    if model_name is None:
        meta = Path(a.adapter or ".") / "train_meta.json"
        model_name = json.loads(meta.read_text())["base_model"] if a.adapter and meta.exists() else DEFAULT_MODEL
    print(f"base model: {model_name}   adapter: {a.adapter}")

    tok = load_tokenizer(model_name, "left")
    model = load_model(model_name, four_bit=not a.no_4bit)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    for split in a.splits:
        rows = read_jsonl(Path(a.data_dir) / f"{split}.jsonl")
        if a.limit:
            rows = rows[:a.limit]
        qs = [r["question"] for r in rows]
        print(f"{split}: {len(rows)} questions -- greedy pass")
        greedy, conf, minp = generate_greedy(model, tok, model_name, qs)
        print(f"{split}: sampling {a.n_samples} answers each")
        samples = []
        for i in tqdm(range(0, len(qs), 64)):
            samples.extend(generate_samples(model, tok, model_name, qs[i:i + 64], a.n_samples, a.temperature))
        path = out / f"rows_{split}{a.tag}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r, g, c, m, s in zip(rows, greedy, conf, minp, samples):
                f.write(json.dumps({"question": r["question"], "reference": r["answer"], "is_idk": r["is_idk"],
                                    "greedy": g, "token_conf": c, "min_prob": m, "samples": s},
                                   ensure_ascii=False) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
