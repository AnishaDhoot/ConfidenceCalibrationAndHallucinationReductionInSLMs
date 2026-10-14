import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

from rag_context import COURSE_CODE

SYSTEM_PROMPT = (
    "You rewrite student questions about a university for a fine-tuning dataset. "
    "For each numbered question, write the requested number of rewrites that ask for exactly the same information. "
    "Keep every course code, programme name, number and proper noun exactly as written. "
    "Vary the sentence structure and wording, and mix formal and casual phrasing. "
    "Do not add or remove any requirement, and do not answer the question. "
    'Reply with JSON only, in the form {"0": ["rewrite", "rewrite"], "1": [...]}, using the given numbers as keys.'
)


def normalize(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def valid_paraphrase(original, candidate, accepted):
    candidate = candidate.strip()
    if not candidate:
        return False
    if not set(COURSE_CODE.findall(original)) <= set(COURSE_CODE.findall(candidate)):
        return False
    if not set(re.findall(r"\d+", original)) <= set(re.findall(r"\d+", candidate)):
        return False
    n_orig, n_cand = len(original.split()), len(candidate.split())
    if not (0.5 * n_orig <= n_cand <= 2.5 * n_orig + 6):
        return False
    norm = normalize(candidate)
    if norm == normalize(original) or norm in {normalize(a) for a in accepted}:
        return False
    return True


class MockBackend:
    TEMPLATES = ["Could you tell me: {q}", "I'd like to know the following. {q}", "{q} Please explain briefly.", "Quick question: {q}"]

    def paraphrase(self, questions, n):
        return {str(i): [t.format(q=q) for t in self.TEMPLATES[:n]] for i, q in enumerate(questions)}


class AnthropicBackend:
    def __init__(self, model):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def paraphrase(self, questions, n):
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))
        user = f"Write {n} rewrites for each question.\n\n{numbered}"
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        return json.loads(text)


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cache(path):
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    cache[d["q"]] = d["paraphrases"]
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache", default="paraphrase_cache.jsonl")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--backend", choices=["anthropic", "mock"], default="anthropic")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retries", type=int, default=3)
    a = p.parse_args()

    rows = load_rows(a.input)
    if any(r.get("raw_question", r["question"]) != r["question"] for r in rows):
        sys.exit("This file has retrieval context in the question field. Augment a --rag-k 0 file instead.")
    if "valid" in a.input or "test" in a.input:
        sys.exit("Only augment the train split.")
    base = rows[: a.limit] if a.limit else rows

    backend = MockBackend() if a.backend == "mock" else AnthropicBackend(a.model)
    cache_path = Path(a.cache)
    cache = load_cache(cache_path)
    todo = [r["question"] for r in base if r["question"] not in cache]
    print(f"{len(base)} train rows, {len(base) - len(todo)} cached, {len(todo)} to paraphrase")

    with open(cache_path, "a", encoding="utf-8") as cf:
        for start in range(0, len(todo), a.batch):
            chunk = todo[start:start + a.batch]
            result = None
            for attempt in range(a.retries):
                try:
                    result = backend.paraphrase(chunk, a.n)
                    break
                except Exception as e:
                    print(f"batch {start // a.batch} attempt {attempt + 1} failed: {e}")
                    time.sleep(2 * (attempt + 1))
            if result is None:
                continue
            for i, q in enumerate(chunk):
                accepted = []
                for cand in result.get(str(i), []):
                    if isinstance(cand, str) and valid_paraphrase(q, cand, accepted):
                        accepted.append(cand.strip())
                cache[q] = accepted
                cf.write(json.dumps({"q": q, "paraphrases": accepted}, ensure_ascii=False) + "\n")
            cf.flush()
            print(f"  {min(start + a.batch, len(todo))}/{len(todo)}")

    added = 0
    with open(a.output, "w", encoding="utf-8") as f:
        for r in base:
            f.write(json.dumps({**r, "aug": False}, ensure_ascii=False) + "\n")
        for r in base:
            for para in cache.get(r["question"], []):
                f.write(json.dumps(
                    {"question": para, "answer": r["answer"], "raw_question": para,
                     "is_idk": r["is_idk"], "aug": True, "source_question": r["question"]},
                    ensure_ascii=False) + "\n")
                added += 1
    print(f"wrote {a.output}: {len(base)} original + {added} paraphrased rows")
    if a.backend == "mock":
        print("mock backend used: for pipeline testing only, do not train on this file")


if __name__ == "__main__":
    main()
