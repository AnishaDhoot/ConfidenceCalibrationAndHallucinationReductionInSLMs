"""Extract course tables (code, title, L-T-P-(J)-C, credits) from the VIT curriculum PDFs.

Rule-based, so every fact comes straight from a table cell. It handles the layouts found in the
curriculum PDFs (one cell per line; "CODE Title" with hyphen-wrapped titles and "L T P C" on one line;
code on its own line). PDFs with no text layer (scans) are reported and skipped, not guessed at.

    python extract_courses.py "C:/path/to/Questions" --out course_tables.jsonl
Needs: pip install pymupdf
"""

import argparse
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

CODE = r"[A-Z]{3,6}\d{3,4}[A-Z]?"
TYPE_WORDS = r"(?:Theory Only|Lab Only|Embedded|Theory and Lab|Project|Soft Skill|Online Course|Theory)"
ROW = re.compile(
    rf"(?<![A-Za-z0-9])({CODE})\s+"                 # course code
    rf"(.{{3,130}}?)\s+"                              # title (lazy)
    rf"((?:{TYPE_WORDS}\s+)*(?:\d\.\d\s+)?)"          # optional course type / version (layout A)
    rf"(\d{{1,2}})\s+(\d{{1,2}})\s+(\d{{1,2}})\s+"    # L T P
    rf"(?:(\d{{1,2}})\s+)?"                            # optional J
    rf"(\d{{1,2}}(?:\.\d)?)(?![\d.])"                 # credits
)
CODE_RE = re.compile(rf"(?<![A-Za-z0-9])({CODE})(?![A-Za-z0-9])")
PROGRAM_START = re.compile(
    r"^(Bachelor of|Master of|Integrated|Doctor of|M\.Sc|B\.Sc|M\.Tech|B\.Tech|M\.Des|M\.Arch|MCA|BCA|B\.Des)", re.I)


def page_texts(path):
    import pymupdf
    doc = pymupdf.open(path)
    return [p.get_text() for p in doc]


def normalise(pages):
    text = "\n".join(pages)
    text = re.sub(r"-\n(?=[a-z])", "", text)           # "Meth-\nods" -> "Methods"
    text = text.replace("|", " ").replace("\u00a0", " ")
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text)


def program_name(pages, fname):
    lines = [l.strip() for p in pages[:8] for l in p.splitlines() if l.strip()]
    name = None
    for i, l in enumerate(lines):
        if PROGRAM_START.match(l) and len(l) < 120 and "of Technology" not in l[:0]:
            name = l
            # specialisation / second line of the title
            if i + 1 < len(lines) and re.match(r"^(Specialisation|Specialization|with|\()", lines[i + 1], re.I):
                name += " " + lines[i + 1]
            break
    year = re.search(r"(20\d\d)[-_](\d{2,4})", fname)
    if not name:
        name = re.sub(r"[_\-]+", " ", Path(fname).stem)
    return re.sub(r"\s+", " ", name).strip(), (f"{year.group(1)}-{year.group(2)[-2:]}" if year else None)


def clean_title(t):
    t = re.sub(r"\s+", " ", t).strip(" -*")
    return t


skipped_rule = [0]  # rows dropped by the credit-rule sanity check


def extract_pdf(path):
    pages = page_texts(path)
    n_chars = sum(len(p) for p in pages)
    if n_chars < 500 * max(1, len(pages) // 20):
        return None, pages, "no text layer"
    blob = normalise(pages)
    rows, seen = [], set()
    for m in ROW.finditer(blob):
        code, title, lead, l, t, p, extra, c = m.groups()
        title = re.sub(r"\s+L T P (?:J )?C\s*$", "", clean_title(title)).strip()
        if CODE_RE.search(title) or len(title) < 3 or not re.search(r"[A-Za-z]{3}", title):
            continue
        l, t, p = int(l), int(t), int(p)
        rule = l + t + p / 2

        def fits(x):  # VIT credit rule; online/project courses with L=T=P=0 can carry any credits
            return (l + t + p == 0) or abs(float(x) - rule) <= 0.51

        if extra is None:
            credits, j = float(c), None
        elif lead.strip():                  # layout A: ... L T P J C
            credits, j = float(c), int(extra)
        elif fits(extra):                   # table with one more column after credits: L T P C <other>
            credits, j = float(extra), None
        else:
            credits, j = float(c), int(extra)
        if credits > 30 or l > 9 or t > 9 or p > 40 or not fits(credits):
            skipped_rule[0] += 1
            continue
        key = (code, title.lower(), l, t, p, j, credits)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"code": code, "title": title, "L": l, "T": t, "P": p, "J": j, "credits": credits})
    return rows, pages, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="course_tables.jsonl")
    a = ap.parse_args()

    out, report = [], []
    for pdf in sorted(Path(a.folder).glob("*.pdf")):
        try:
            rows, pages, why = extract_pdf(pdf)
        except Exception as e:  # unreadable PDF
            report.append((pdf.name, "-", 0, f"error: {e}"))
            continue
        if rows is None:
            report.append((pdf.name, "-", 0, why))
            continue
        prog, year = program_name(pages, pdf.name)
        for r in rows:
            out.append({"source": pdf.name, "program": prog, "year": year, **r})
        report.append((pdf.name, prog, len(rows), ""))

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, prog, n, note in report:
        print(f"{name[:52]:<54}{n:>5}  {prog[:58]}  {note}")
    print(f"\n{len(out)} course rows from {sum(1 for r in report if r[2])} PDFs "
          f"({sum(1 for r in report if not r[2])} PDFs gave none); wrote {a.out}")


if __name__ == "__main__":
    main()
