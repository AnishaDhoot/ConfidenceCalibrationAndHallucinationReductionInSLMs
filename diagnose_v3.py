"""Where does the model get things wrong? Runs on CPU from results_v3/rows_test.jsonl.

    python diagnose_v3.py --rows results_v3/rows_test.jsonl --extra extra_course_qa.jsonl

Prints accuracy by question type, and -- if --extra is given -- whether questions about courses whose facts
were in the extra training rows are answered better than questions about courses that were not (a direct test
of "did the model memorise what it was trained on?").
"""

import argparse
import re
from collections import defaultdict

from v3_common import COURSE_CODE, correctness, read_jsonl, refused


def kind(q):
    ql = q.lower()
    has_code = bool(COURSE_CODE.findall(q))
    if re.search(r"credit", ql):
        k = "credits"
    elif re.search(r"l-t-p|l, t|lecture|tutorial|practical|contact hour", ql):
        k = "L-T-P"
    elif re.search(r"course code|code of|which course|title of|name of", ql):
        k = "title/code"
    elif re.search(r"prerequisite|pre-requisite|semester|textbook|syllabus|objective|outcome|module|assessment|"
                   r"evaluation|elective|category|type|\blab\b", ql):
        k = "other course detail"
    else:
        k = "general (rules, programme, campus)"
    return k + (" [has course code]" if has_code else "")


def pct(x, n):
    return f"{x / n:.3f}" if n else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results_v3/rows_test.jsonl")
    ap.add_argument("--extra", help="extra_course_qa.jsonl used for training, to test memorisation")
    ap.add_argument("--show", type=int, default=6, help="wrong examples to print for memorised-course credit questions")
    a = ap.parse_args()

    rows = read_jsonl(a.rows)
    ans = [r for r in rows if not r["is_idk"]]
    by = defaultdict(list)
    for r in ans:
        s_f1, s_key = correctness(r["greedy"], r["reference"], False)
        by[kind(r["question"])].append((s_key, s_f1, refused(r["greedy"])))
    print(f"answerable test questions: {len(ans)}")
    print(f"{'question type':<46}{'n':>5}{'key-fact':>10}{'F1>=.5':>9}{'refused':>9}")
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        n = len(v)
        print(f"{k:<46}{n:>5}{pct(sum(x[0] for x in v), n):>10}{pct(sum(x[1] for x in v), n):>9}"
              f"{pct(sum(x[2] for x in v), n):>9}")

    idk = [r for r in rows if r["is_idk"]]
    print(f"\nunanswerable: {len(idk)}  correctly refused {pct(sum(refused(r['greedy']) for r in idk), len(idk))}")

    if a.extra:
        trained = {}  # code -> set of kinds that were trained on
        for r in read_jsonl(a.extra):
            code = COURSE_CODE.findall(r["answer"])[0]
            trained.setdefault(code, set()).add(r["kind"])
        seen, unseen, wrong = [], [], []
        for r in ans:
            cs = COURSE_CODE.findall(r["question"])
            if len(cs) != 1 or not re.search(r"credit|l-t-p|title|code", r["question"].lower()):
                continue
            _, ok = correctness(r["greedy"], r["reference"], False)
            (seen if cs[0] in trained else unseen).append(ok)
            if cs[0] in trained and not ok:
                wrong.append(r)
        print("\nMemorisation check (single-course credit / L-T-P / title / code questions):")
        print(f"  course facts WERE in the extra training rows : n={len(seen):>3}  key-fact acc {pct(sum(seen), len(seen))}")
        print(f"  course facts were NOT in the extra rows      : n={len(unseen):>3}  key-fact acc {pct(sum(unseen), len(unseen))}")
        print("  -> if the first number is not clearly higher, the model did not memorise the table facts.")
        for r in wrong[:a.show]:
            print(f"\n  Q:    {r['question'][:130]}\n  REF:  {r['reference'][:110]}\n  PRED: {r['greedy'][:110]}")


if __name__ == "__main__":
    main()
