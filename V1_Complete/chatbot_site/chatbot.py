"""
Answer backends for the VIT Q&A demo site.

Two modes, picked with the CHATBOT_BACKEND environment variable:

  retrieval (default)  Finds the closest question in final_qa_dataset.csv and
                       returns its answer. Needs no GPU and no trained adapter,
                       so the site always runs. Matching blends sentence
                       embeddings (handles paraphrases) with TF-IDF (handles
                       exact terms and course codes). It abstains with the same
                       canonical "I don't know" string the fine-tuned models
                       were trained on when nothing is close enough, or when
                       the question names something (a company, a course code)
                       that appears nowhere in the documents.

  lora                 Loads Phi-3-mini-4k-instruct in 4-bit plus one of your
                       trained LoRA adapters (ADAPTER_PATH) and answers
                       generatively. Confidence is the mean per-token
                       probability of the generated answer -- the same
                       definition eval_compare.py uses, so the number shown in
                       the UI matches what the paper reports.

Every response carries the backend name, so the UI never presents a retrieval
lookup as the fine-tuned model.
"""

import csv
import os
import re
import threading
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_DIR / "data" / "final_qa_dataset.csv"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

IDK_ANSWER = "I don't have enough information in my knowledge base to answer this question accurately."

# Same pattern as prepare_expc_data.py / eval_compare.py / compare_idk_stats.py.
IDK_PATTERN = re.compile(
    r"don't know|I don't|cannot verify|cannot check|cannot submit|"
    r"cannot see|real-time|moment to moment|I cannot|not in my knowledge",
    re.IGNORECASE,
)

COURSE_CODE = re.compile(r"\b[A-Z]{2,6}\d{3,4}[A-Z]?\b")

# Some rows in the source CSV have UTF-8 dashes / rupee signs that were decoded as
# Latin-1 or cp1252 (they show up as "a-circumflex + control characters"). Repair them
# for display; anything that does not round-trip cleanly is left alone.
_MOJIBAKE = re.compile(r"[Â-ô][^\x00-\x7f]{1,3}")


def repair_text(text):
    def fix(match):
        chunk = match.group(0)
        for codec in ("latin-1", "cp1252"):
            try:
                return chunk.encode(codec).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return chunk
    return _MOJIBAKE.sub(fix, text)


def _load_pairs(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = []
    for r in rows:
        q = repair_text((r.get("question") or "").strip())
        a = repair_text((r.get("answer") or "").strip())
        if q and a and not IDK_PATTERN.search(a):
            pairs.append((q, a))
    return pairs


def _tokens(text):
    """Split into alphanumeric tokens, keeping case. 'Ph.D' -> 'PhD', "VIT's" -> 'VIT'."""
    text = re.sub(r"(?<=\w)\.(?=\w)", "", text)
    text = re.sub(r"['’]s\b", "", text)
    return re.findall(r"[A-Za-z0-9]+", text)


class RetrievalBot:
    name = "retrieval"
    label = "Dataset lookup (demo)"

    LEXICAL_WEIGHT = 0.35   # share of the score coming from TF-IDF vs embeddings
    CODE_BONUS = 0.25       # a course code shared by question and row is a strong signal

    def __init__(self, csv_path=DATA_CSV, threshold=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.pairs = _load_pairs(csv_path)
        questions = [q for q, _ in self.pairs]

        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.word_mat = self.word_vec.fit_transform(questions)
        self.char_mat = self.char_vec.fit_transform(questions)
        self.codes = [set(COURSE_CODE.findall(q + " " + a)) for q, a in self.pairs]

        self.vocab = set()
        for q, a in self.pairs:
            self.vocab.update(t.lower() for t in _tokens(q + " " + a))

        self.encoder, self.embeddings = self._load_embeddings(csv_path, questions)
        if self.embeddings is None:
            self.label = "Dataset lookup, keyword only (demo)"
            self.lexical_weight = 1.0
            default_threshold = 0.42
        else:
            self.lexical_weight = self.LEXICAL_WEIGHT
            default_threshold = 0.43
        self.threshold = threshold if threshold is not None else float(
            os.environ.get("RETRIEVAL_THRESHOLD", default_threshold))

    def _load_embeddings(self, csv_path, questions):
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            encoder = SentenceTransformer(model_name)
        except Exception as err:  # not installed, or offline with no cached model
            print(f"[chatbot] sentence embeddings unavailable ({type(err).__name__}); "
                  f"falling back to keyword matching only.")
            return None, None

        CACHE_DIR.mkdir(exist_ok=True)
        stamp = int(Path(csv_path).stat().st_mtime)
        cache_file = CACHE_DIR / f"embeddings_{stamp}_{len(questions)}.npy"
        if cache_file.exists():
            return encoder, np.load(cache_file)
        print("[chatbot] encoding the dataset once (cached afterwards)...")
        vectors = encoder.encode(questions, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        np.save(cache_file, vectors)
        return encoder, vectors

    def unknown_entities(self, question):
        """Named things (companies, course codes, years) that appear nowhere in the documents."""
        found = []
        for i, tok in enumerate(_tokens(question)):
            if len(tok) < 3:
                continue
            looks_like_entity = (i > 0 and tok[0].isupper()) or any(c.isdigit() for c in tok) \
                or (tok.isupper() and len(tok) >= 3)
            if looks_like_entity and tok.lower() not in self.vocab:
                found.append(tok)
        return found

    def _scores(self, question):
        lexical = 0.6 * (self.word_mat @ self.word_vec.transform([question]).T).toarray().ravel() \
            + 0.4 * (self.char_mat @ self.char_vec.transform([question]).T).toarray().ravel()
        if self.embeddings is None:
            return lexical
        dense = self.embeddings @ self.encoder.encode([question], normalize_embeddings=True)[0]
        return (1 - self.lexical_weight) * dense + self.lexical_weight * lexical

    def answer(self, question):
        scores = self._scores(question)

        query_codes = set(COURSE_CODE.findall(question))
        if query_codes:
            for i, codes in enumerate(self.codes):
                if query_codes & codes:
                    scores[i] += self.CODE_BONUS

        best = int(scores.argmax())
        score = float(min(max(scores[best], 0.0), 1.0))

        unknown = self.unknown_entities(question)
        if unknown:
            named = ", ".join(unknown[:3])
            return {"answer": f"{IDK_ANSWER} (I could not find “{named}” in the VIT documents I know.)",
                    "confidence": round(min(score, 0.2), 3), "abstained": True,
                    "backend": self.name, "matched_question": None}
        if score < self.threshold:
            return {"answer": IDK_ANSWER, "confidence": round(score, 3), "abstained": True,
                    "backend": self.name, "matched_question": None}
        q, a = self.pairs[best]
        return {"answer": a, "confidence": round(score, 3), "abstained": False,
                "backend": self.name, "matched_question": q}


class LoraBot:
    name = "lora"
    label = "Phi-3 mini + LoRA"

    def __init__(self, adapter_path, model_name="microsoft/Phi-3-mini-4k-instruct", max_new_tokens=96):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.lock = threading.Lock()  # one generation at a time on a small GPU

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map={"": 0})
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

    def answer(self, question):
        torch = self.torch
        prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        with self.lock, torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_tokens = out.sequences[0][inputs["input_ids"].shape[1]:]
        text = repair_text(self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

        logprobs = []
        for step_scores, tok in zip(out.scores, gen_tokens):
            logprobs.append(torch.log_softmax(step_scores[0].float(), dim=-1)[tok].item())
        confidence = float(torch.tensor(logprobs).mean().exp()) if logprobs else 0.0

        return {"answer": text, "confidence": round(confidence, 3),
                "abstained": bool(IDK_PATTERN.search(text)),
                "backend": self.name, "matched_question": None}


def get_bot():
    choice = os.environ.get("CHATBOT_BACKEND", "retrieval").lower()
    if choice == "lora":
        adapter = os.environ.get("ADAPTER_PATH")
        if not adapter:
            raise SystemExit("CHATBOT_BACKEND=lora needs ADAPTER_PATH pointing at a LoRA adapter folder.")
        print(f"Loading Phi-3 + adapter from {adapter} ...")
        return LoraBot(adapter)
    print("Using retrieval backend (set CHATBOT_BACKEND=lora + ADAPTER_PATH for the fine-tuned model).")
    return RetrievalBot()
