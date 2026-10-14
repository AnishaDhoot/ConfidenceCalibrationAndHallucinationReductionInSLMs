"""Paraphrase the TRAIN questions with a local instruct model -- no API key, no rate limits.

Runs on a Colab T4: Qwen2.5-3B-Instruct in fp16 by default (about 7 GB). Output is resumable, so
if Colab disconnects, run the same command again and it picks up where it stopped.

    python augment_v3.py --train data_v3/train.jsonl --out data_v3/paraphrases.jsonl

Only the question wording changes. A rewrite is dropped unless it keeps every course code and number
from the original, so facts cannot drift. Answers are never touched.
"""

import argparse
import json
import re
from pathlib import Path

from v3_common import COURSE_CODE, normalize, read_jsonl

INSTRUCTION = (
    "Rewrite the student's question below in {n} different ways. Each rewrite must ask for exactly the same "
    "information. Keep every course code, programme name, number and proper noun exactly as written. "
    "Change the sentence structure and wording; mix formal and casual phrasing. Do not answer the question. "
    'Reply with only a JSON list of {n} strings.\n\nQuestion: {q}'
)


def valid_paraphrase(original, candidate, accepted):
    candidate = candidate.strip()
    if not candidate or candidate.endswith(":"):
        return False
    if not set(COURSE_CODE.findall(original)) <= set(COURSE_CODE.findall(candidate)):
        return False
    if not set(re.findall(r"\d+", original)) <= set(re.findall(r"\d+", candidate)):
        return False
    n_orig, n_cand = len(original.split()), len(candidate.split())
    if not (0.5 * n_orig <= n_cand <= 2.5 * n_orig + 6):
        return False
    norm = normalize(candidate)
    return norm != normalize(original) and norm not in {normalize(a) for a in accepted}


def parse_list(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(0))
            return [s for s in items if isinstance(s, str)]
        except json.JSONDecodeError:
            pass
    lines = [re.sub(r'^\s*(?:\d+[.)]|[-*])\s*', "", ln).strip().strip('"') for ln in text.splitlines()]
    return [ln for ln in lines if len(ln.split()) >= 3]


class MockBackend:
    TEMPLATES = ["Could you tell me: {q}", "I'd like to know the following. {q}", "{q} Please explain briefly.",
                 "Quick question: {q}"]

    def paraphrase(self, questions, n):
        return [[t.format(q=q) for t in self.TEMPLATES[:n]] for q in questions]


class LocalBackend:
    def __init__(self, model_name, four_bit, batch_size):
        import torch
        from v3_llm import load_model, load_tokenizer
        self.torch = torch
        self.tok = load_tokenizer(model_name, "left")
        self.model = load_model(model_name, four_bit=four_bit)
        self.model.eval()
        self.batch_size = batch_size

    def paraphrase(self, questions, n):
        torch = self.torch
        prompts = [self.tok.apply_chat_template([{"role": "user", "content": INSTRUCTION.format(n=n, q=q)}],
                                                tokenize=False, add_generation_prompt=True) for q in questions]
        out = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tok(prompts[i:i + self.batch_size], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=60 * n + 40, do_sample=True, temperature=0.8,
                                          top_p=0.95, pad_token_id=self.tok.pad_token_id)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.extend(parse_list(t) for t in texts)
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data_v3/train.jsonl")
    p.add_argument("--out", default="data_v3/paraphrases.jsonl")
    p.add_argument("--backend", choices=["local", "mock"], default="local")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--four-bit", action="store_true", help="use for 7B models on a T4")
    p.add_argument("--n", type=int, default=3, help="paraphrases to ask for per question")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk", type=int, default=64, help="questions per checkpoint write")
    a = p.parse_args()

    rows = [r for r in read_jsonl(a.train) if r["source"] == "original"]
    out_path = Path(a.out)
    done = {json.loads(l)["question"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    todo = [r for r in rows if r["question"] not in done]
    print(f"{len(rows)} train questions, {len(done)} already done, {len(todo)} to go")

    backend = MockBackend() if a.backend == "mock" else LocalBackend(a.model, a.four_bit, a.batch_size)
    kept = total = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), a.chunk):
            chunk = todo[i:i + a.chunk]
            results = backend.paraphrase([r["question"] for r in chunk], a.n)
            for r, cands in zip(chunk, results):
                accepted = []
                for c in cands:
                    if valid_paraphrase(r["question"], c, accepted):
                        accepted.append(c.strip())
                kept += len(accepted)
                total += len(cands)
                f.write(json.dumps({"question": r["question"], "paraphrases": accepted}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(i + a.chunk, len(todo))}/{len(todo)}  accepted {kept}/{total} candidates")


if __name__ == "__main__":
    main()
