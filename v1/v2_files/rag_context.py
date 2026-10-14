import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

COURSE_CODE = re.compile(r"\b[A-Z]{2,6}\d{3,4}[A-Z]?\b")

IDK_ANSWER = "I don't have enough information in my knowledge base to answer this question accurately."

INSTRUCTION = (
    "Answer the question using only the reference notes below. "
    "If the notes do not contain the answer, say you do not have enough information."
)

MAX_NOTE_WORDS = 60


def clip(text, max_words=MAX_NOTE_WORDS):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


class Retriever:
    WORD_WEIGHT = 0.5
    CHAR_WEIGHT = 0.5
    CODE_BONUS = 0.25

    def __init__(self, pairs):
        self.pairs = list(pairs)
        questions = [q for q, _ in self.pairs]
        full_text = [q + " " + a for q, a in self.pairs]
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.word_mat = self.word_vec.fit_transform(full_text)
        self.char_mat = self.char_vec.fit_transform(questions)
        self.codes = [set(COURSE_CODE.findall(t)) for t in full_text]

    def scores(self, question):
        w = (self.word_mat @ self.word_vec.transform([question]).T).toarray().ravel()
        c = (self.char_mat @ self.char_vec.transform([question]).T).toarray().ravel()
        s = self.WORD_WEIGHT * w + self.CHAR_WEIGHT * c
        q_codes = set(COURSE_CODE.findall(question))
        if q_codes:
            bonus = np.array([self.CODE_BONUS if q_codes & cs else 0.0 for cs in self.codes])
            s = s + bonus
        return s

    def search(self, question, k=3, exclude=None):
        s = self.scores(question)
        if exclude is not None:
            s[exclude] = -1.0
        order = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in order]


def build_prompt(question, hits, pairs):
    if not hits:
        return question
    notes = []
    for rank, (idx, _) in enumerate(hits, start=1):
        q, a = pairs[idx]
        notes.append(f"{rank}. Q: {clip(q, 40)}\n   A: {clip(a)}")
    return f"{INSTRUCTION}\n\nReference notes:\n" + "\n".join(notes) + f"\n\nQuestion: {question}"
