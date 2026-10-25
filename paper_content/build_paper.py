"""Assemble the paper (.docx) from the markdown section drafts using the Springer LNCS Word template.

    python build_paper.py [--template template/splnproc2510.docm] [--out paper.docx]

The template's styles, numbering, headers and page setup are reused; only the body is replaced. Macros are dropped
(the output is a plain .docx), so heading numbers come from the template's own multilevel list.
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

HERE = Path(__file__).parent
SECTION_FILES = [
    "paper_abstract_intro_draft.md",
    "paper_related_work_draft.md",
    "paper_problem_formulation_draft.md",
    "paper_methodology_draft.md",
    "paper_experimental_setup_draft.md",
    "paper_results_draft.md",
    "paper_discussion_draft.md",
    "paper_conclusion_references_draft.md",
]

TITLE = "Confidence Calibration and Hallucination Reduction in Small Language Models through Controlled Abstention Fine-Tuning"
AUTHORS = ["Anisha Dhoot", "Yash Hirani"]
AFFILIATION = "School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India"

TEXT_WIDTH_EMU = int(4.8 * 914400)


# ---------------------------------------------------------------- OMML helpers
def _mr(t, plain=False):
    rpr = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return (f'<m:r>{rpr}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'
            f'<m:t xml:space="preserve">{escape(t)}</m:t></m:r>')


def R(t):  # italic variable
    return _mr(t)


def P(t):  # upright text
    return _mr(t, True)


def sub(b, s):
    return f"<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>"


def sup(b, s):
    return f"<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>"


def frac(n, d):
    return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"


def nary(ch, lo, hi, body):
    pr = f'<m:chr m:val="{ch}"/><m:limLoc m:val="undOvr"/>'
    if not hi:
        pr += '<m:supHide m:val="1"/>'
    return (f"<m:nary><m:naryPr>{pr}</m:naryPr><m:sub>{lo}</m:sub><m:sup>{hi}</m:sup><m:e>{body}</m:e></m:nary>")


def delim(body, beg="(", end=")"):
    return (f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr><m:e>{body}</m:e></m:d>')


def cases(*rows):
    e = "".join(f"<m:e>{r}</m:e>" for r in rows)
    return ('<m:d><m:dPr><m:begChr m:val="{"/><m:endChr m:val=""/></m:dPr><m:e>'
            f"<m:eqArr>{e}</m:eqArr></m:e></m:d>")


def cat(*parts):
    return "".join(parts)


Q = "\U0001d4ac"   # script Q
D = "\U0001d49f"   # script D
L = "\U0001d4c1"   # script l (unused)
CAL_L = "\U0001d4c1"
SCRIPT_L = "ℒ"
IND = "\U0001d7d9"  # double-struck 1

EQUATIONS = [
    # 1 partition of queries
    cat(R(Q), P(" = "), sub(R(Q), R("A")), P(" ∪ "), sub(R(Q), R("U")), P(",   "), sub(R(Q), R("A")), P(" ∩ "),
        sub(R(Q), R("U")), P(" = ∅")),
    # 2 hallucination
    cat(P("hallucination"), delim(cat(R("q"), P(", "), R("r"))), P(" = "), R("z"), delim(R("q")),
        delim(cat(P("1 − "), R("a"), delim(R("r"))))),
    # 3 HR
    cat(P("HR = "), frac(cat(nary("∑", R("q"), "", cat(R("z"), delim(R("q")), delim(cat(P("1 − "), R("a"), delim(R("r"))))))),
                         nary("∑", R("q"), "", cat(R("z"), delim(R("q")))))),
    # 4 refusal-aware correctness
    cat(R("y"), delim(cat(R("q"), P(", "), R("r"))), P(" = "),
        cases(cat(R("a"), delim(R("r")), P(",   if "), R("q"), P(" ∈ "), sub(R(Q), R("U"))),
              cat(delim(cat(P("1 − "), R("a"), delim(R("r")))), P(" · "), P(IND),
                  delim(cat(R("F1"), delim(cat(R("r"), P(", "), sup(R("r"), P("*")))), P(" ≥ "), R("τ")), "[", "]"),
                  P(",   if "), R("q"), P(" ∈ "), sub(R(Q), R("A"))))),
    # 5 number of abstention examples
    cat(sub(R("n"), P("idk")), P(" = "),
        delim(cat(sub(R("N"), P("reg")), P(" · "), frac(R("ρ"), cat(P("1 − "), R("ρ")))), "⌊", "⌋")),
    # 6 confidence
    cat(R("c"), delim(R("q")), P(" = exp"),
        delim(cat(frac(P("1"), R("T")), nary("∑", cat(R("t"), P("=1")), R("T"),
                                              cat(P("log "), sub(R("p"), R("θ")),
                                                  delim(cat(sub(R("r"), R("t")), P(" | "),
                                                            sub(R("r"), cat(P("<"), R("t"))), P(", "), R("q")))))))),
    # 7 calibration
    cat(P("Pr"), delim(cat(R("y"), delim(cat(R("q"), P(", "), R("r"))), P(" = 1  |  "), R("c"), delim(R("q")),
                           P(" = "), R("p"))), P(" = "), R("p"), P(",   ∀ "), R("p"), P(" ∈ [0, 1]")),
    # 8 discrimination
    cat(P("Pr"), delim(cat(R("c"), delim(sub(R("q"), R("i"))), P(" > "), R("c"), delim(sub(R("q"), R("j"))),
                           P("  |  "), sub(R("y"), R("i")), P(" = 1, "), sub(R("y"), R("j")), P(" = 0"))),
        P(" → 1")),
    # 9 loss
    cat(R(SCRIPT_L), delim(R("θ")), P(" = − "), nary("∑", cat(R("x"), P(" ∈ "), sub(R(D), P("train"))), "",
                                                      nary("∑", R("t"), "",
                                                           cat(P("log "), sub(R("p"), R("θ")),
                                                               delim(cat(sub(R("x"), R("t")), P(" | "),
                                                                         sub(R("x"), cat(P("<"), R("t")))))))) ),
    # 10 HR from counts
    cat(P("HR = "), frac(P("FN"), P("TP + FN"))),
    # 11 precision / recall / F1 (three lines)
    "|".join([
        cat(sub(R("P"), P("IDK")), P(" = "), frac(P("TP"), P("TP + FP"))),
        cat(sub(R("R"), P("IDK")), P(" = "), frac(P("TP"), P("TP + FN"))),
        cat(sub(R("F1"), P("IDK")), P(" = "), frac(cat(P("2 "), sub(R("P"), P("IDK")), sub(R("R"), P("IDK"))),
                                                    cat(sub(R("P"), P("IDK")), P(" + "), sub(R("R"), P("IDK"))))),
    ]),
    # 12 ECE
    cat(P("ECE = "), nary("∑", cat(R("m"), P("=1")), R("M"),
                          cat(frac(cat(P("|"), sub(R("B"), R("m")), P("|")), R("N")), P(" · "),
                              P("|"), P("acc"), delim(sub(R("B"), R("m"))), P(" − conf"), delim(sub(R("B"), R("m"))),
                              P("|")))),
    # 13 Brier
    cat(P("Brier = "), frac(P("1"), R("N")), nary("∑", cat(R("i"), P("=1")), R("N"),
                                                  cat(sup(delim(cat(sub(R("c"), R("i")), P(" − "), sub(R("y"), R("i")))),
                                                          P("2"))))),
]


# ---------------------------------------------------------------- inline markup
SUBS = {  # plain-text spellings used in the drafts -> (base, subscript)
    "𝒬_A": ("𝒬", "A"), "𝒬_U": ("𝒬", "U"), "𝒟_train": ("𝒟", "train"), "N_reg": ("N", "reg"),
    "f_θ": ("f", "θ"), "r_∅": ("r", "∅"), "p_θ": ("p", "θ"), "c_i": ("c", "i"), "y_i": ("y", "i"),
    "r_t": ("r", "t"), "r_{<t}": ("r", "<t"), "n_idk": ("n", "idk"),
}
SUB_RE = re.compile("(" + "|".join(re.escape(k) for k in sorted(SUBS, key=len, reverse=True)) + ")")
INLINE_RE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`)")


def run(text, b=False, i=False, code=False, sz=None, sup_=False, sub_=False):
    pr = ""
    if code:
        pr += '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'
    if b:
        pr += "<w:b/><w:bCs/>"
    if i:
        pr += "<w:i/><w:iCs/>"
    if sz:
        pr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
    elif code:
        pr += '<w:sz w:val="18"/><w:szCs w:val="18"/>'
    if sup_:
        pr += '<w:vertAlign w:val="superscript"/>'
    if sub_:
        pr += '<w:vertAlign w:val="subscript"/>'
    pr = f"<w:rPr>{pr}</w:rPr>" if pr else ""
    return f'<w:r>{pr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def plain_runs(text, b=False, i=False, sz=None):
    """Text with math-ish subscripts expanded."""
    out = []
    for part in SUB_RE.split(text):
        if part in SUBS:
            base, s = SUBS[part]
            out.append(run(base, b, True, sz=sz))
            out.append(run(s, b, False, sz=sz, sub_=True))
        elif part:
            part = re.sub(r"\br\*", "r∗", part)
            out.append(run(part, b, i, sz=sz))
    return "".join(out)


def inline(text, sz=None, base_b=False):
    text = re.sub(r"\bFigure (\d)", r"Fig. \1", text)
    out = []
    for part in INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append(plain_runs(part[2:-2], b=True, sz=sz))
        elif part.startswith("`") and part.endswith("`"):
            out.append(run(part[1:-1], code=True, sz=sz and sz - 1))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append(plain_runs(part[1:-1], i=True, b=base_b, sz=sz))
        else:
            out.append(plain_runs(part, b=base_b, sz=sz))
    return "".join(out)


def para(style, content, extra_ppr=""):
    st = f'<w:pStyle w:val="{style}"/>' if style else ""
    ppr = f"<w:pPr>{st}{extra_ppr}</w:pPr>" if (st or extra_ppr) else ""
    return f"<w:p>{ppr}{content}</w:p>"


# ---------------------------------------------------------------- blocks
class Builder:
    def __init__(self):
        self.body = []
        self.images = []  # (rId, filename)
        self.eq_no = 0
        self.after_float = True  # first paragraph after heading/table/figure/equation is unindented
        self.fig_no = 0

    def add(self, xml):
        self.body.append(xml)

    def heading(self, level, text):
        self.add(para(f"heading{level}", run(text)))
        self.after_float = True

    def paragraph(self, text):
        style = "p1a" if self.after_float else None
        self.add(para(style, inline(text)))
        self.after_float = False

    def bullet(self, text):
        self.add(para("bulletitem", inline(text)))
        self.after_float = False

    def numbered(self, n, text):
        self.add(para(None, run(f"{n}.\t") + inline(text),
                      '<w:tabs><w:tab w:val="left" w:pos="340"/></w:tabs><w:spacing w:before="40" w:after="40"/>'
                      '<w:ind w:left="340" w:hanging="340"/>'))
        self.after_float = False

    def equation(self):
        if self.eq_no >= len(EQUATIONS):
            raise SystemExit("more equations in the drafts than defined in EQUATIONS")
        omml = EQUATIONS[self.eq_no]
        self.eq_no += 1
        zones = omml.split("|") if self.eq_no == 11 else [omml]
        num = (f'<w:r><w:tab/></w:r><w:r><w:t>({self.eq_no})</w:t></w:r>')
        if len(zones) == 1:
            content = f"<w:r><w:tab/></w:r><m:oMath>{zones[0]}</m:oMath>{num}"
            self.add(para("equation", content))
        else:
            rows = []
            for k, z in enumerate(zones):
                tail = num if k == len(zones) - 1 else ""
                rows.append(f"<w:r><w:tab/></w:r><m:oMath>{z}</m:oMath>{tail}")
            self.add(para("equation", '<w:r><w:br/></w:r>'.join(rows)))
        self.after_float = True

    def table(self, rows, caption=None, small=False):
        ncols = len(rows[0])
        total = 6900
        sz = 15 if ncols >= 8 else 17
        if ncols >= 8:
            first = 850
            widths = [first] + [(total - first) // (ncols - 1)] * (ncols - 1)
        elif ncols == 2:
            widths = [total // 2] * 2
        else:
            first = 1800 if ncols == 3 else total // ncols
            widths = [first] + [(total - first) // (ncols - 1)] * (ncols - 1)
        if caption:
            m = re.match(r"\*\*(Table \d+\.)\*\*\s*(.*)", caption, re.S)
            cap = run(m.group(1), b=True) + run(" ") + inline(m.group(2))
            self.add(para("tablecaption", cap, '<w:jc w:val="both"/>' if len(caption) > 110 else ""))
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        trs = []
        for ri, row in enumerate(rows):
            tcs = []
            last = ri == len(rows) - 1
            for ci, cell in enumerate(row):
                borders = ""
                if ri == 0:
                    borders = ('<w:tcBorders><w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
                               '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>')
                elif last:
                    borders = '<w:tcBorders><w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/></w:tcBorders>'
                jc = "left" if ci == 0 else "center"
                txt = cell.strip()
                p = para(None, inline(txt, sz=sz, base_b=(ri == 0)),
                         f'<w:spacing w:before="30" w:after="30"/><w:ind w:firstLine="0"/><w:jc w:val="{jc}"/>')
                tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{borders}'
                           f'<w:vAlign w:val="center"/></w:tcPr>{p}</w:tc>')
            trs.append(f'<w:tr><w:trPr><w:cantSplit/><w:jc w:val="center"/></w:trPr>{"".join(tcs)}</w:tr>')
        tbl = ('<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
               '<w:tblCellMar><w:left w:w="50" w:type="dxa"/><w:right w:w="50" w:type="dxa"/></w:tblCellMar>'
               '<w:tblLook w:val="0000"/></w:tblPr><w:tblGrid>%s</w:tblGrid>%s</w:tbl>' % (sum(widths), grid, "".join(trs)))
        self.add(tbl)
        self.add(para(None, "", '<w:spacing w:before="0" w:after="60"/><w:ind w:firstLine="0"/>'))
        self.after_float = True

    def figure(self, filename, caption_md):
        path = HERE / "figures" / filename
        w, h = Image.open(path).size
        cx = int(TEXT_WIDTH_EMU * (0.80 if h / w > 0.8 else 0.92))
        cy = int(cx * h / w)
        rid = f"rIdImg{len(self.images) + 1}"
        self.images.append((rid, filename))
        n = len(self.images)
        drawing = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{100 + n}" name="Figure {n}" descr="{escape(filename)}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{escape(filename)}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            "</wp:inline></w:drawing></w:r>")
        self.add(para("image", drawing, '<w:keepNext/>'))
        m = re.match(r"\*\*Figure (\d+)\.\*\*\s*(.*)", caption_md, re.S)
        cap = run(f"Fig. {m.group(1)}.", b=False) + run(" ") + inline(m.group(2))
        self.add(para("figurecaption", cap, '<w:jc w:val="both"/>'))
        self.after_float = True

    def code(self, lines):
        for ln in lines:
            self.add(para("programcode", run(ln, code=True), '<w:spacing w:before="0" w:after="0"/>'))
        self.after_float = True

    def quote(self, text):
        self.add(para(None, inline(text), '<w:spacing w:before="80" w:after="120"/><w:ind w:left="567" w:right="567" w:firstLine="0"/>'))
        self.after_float = True


# ---------------------------------------------------------------- markdown parsing
def split_row(line):
    return [c for c in line.strip().strip("|").split("|")]


def parse_section(text, b, refs):
    lines = text.splitlines()
    i = 0
    pending_caption = None
    para_buf = []

    def flush():
        nonlocal para_buf
        if para_buf:
            b.paragraph(" ".join(s.strip() for s in para_buf))
            para_buf = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s or s == "---":
            flush()
            i += 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", s)
        if m:
            flush()
            lvl, title = len(m.group(1)), m.group(2)
            if title.startswith("LaTeX note"):
                break
            title = re.sub(r"^\d+(\.\d+)*\s+", "", title)
            if lvl == 1:
                b.heading(1, title)
            elif lvl == 2:
                b.heading(2, title)
            i += 1
            continue
        if s.startswith("$$"):
            flush()
            i += 1
            while not lines[i].strip().startswith("$$"):
                i += 1
            i += 1
            b.equation()
            continue
        if s.startswith("```"):
            flush()
            i += 1
            code = []
            while not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            b.code(code)
            continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not re.match(r"^\|[\s:|-]+\|$", lines[i].strip()):
                    rows.append(split_row(lines[i]))
                i += 1
            b.table(rows, pending_caption)
            pending_caption = None
            continue
        if s.startswith(">"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            # figures: "**Figure N.** caption *(file.png)*"
            joined = [x for x in block if x]
            if joined and re.match(r"\*\*Figure \d+\.\*\*", joined[0]):
                for entry in joined:
                    fm = re.match(r"(\*\*Figure \d+\.\*\*.*?)\s*\*\(([\w.]+\.png)\)\*\s*$", entry)
                    b.figure(fm.group(2), fm.group(1))
            elif joined and joined[0].startswith("*[Fill in"):
                pass
            else:
                b.quote(" ".join(joined))
            continue
        if re.match(r"^\*\*Table \d+\.\*\*", s):
            flush()
            pending_caption = s
            i += 1
            continue
        if re.match(r"^- ", s):
            flush()
            b.bullet(s[2:])
            i += 1
            continue
        m = re.match(r"^(\d+)\.\s+(.*)", s)
        if m and refs is not None and refs == "collect":
            i += 1
            continue
        if m:
            flush()
            b.numbered(m.group(1), m.group(2))
            i += 1
            continue
        para_buf.append(s)
        i += 1
    flush()


def read(name):
    return (HERE / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- package assembly
def build(template, out):
    b = Builder()

    # ---- front matter
    abstract_md = read(SECTION_FILES[0])
    abs_m = re.search(r"## Abstract\s+(.*?)\n## Keywords", abstract_md, re.S)
    kw_m = re.search(r"## Keywords\s+(.*?)\n# ", abstract_md, re.S)
    abstract = " ".join(abs_m.group(1).split())
    keywords = kw_m.group(1).strip()
    n_words = len(abstract.split())
    print(f"abstract: {n_words} words")

    b.add(para("papertitle", run(TITLE)))
    auth = ""
    for k, a in enumerate(AUTHORS):
        auth += run(a) + run("1", sup_=True)
        if k < len(AUTHORS) - 1:
            auth += run(" and " if k == len(AUTHORS) - 2 else ", ")
    b.add(para("author", auth))
    b.add(para("address", run("1", sup_=True) + run(" " + AFFILIATION)))
    b.add(para("abstract", run("Abstract. ", b=True) + inline(abstract),
               '<w:spacing w:after="0"/><w:ind w:firstLine="0"/>'))
    b.add(para("keywords", run("Keywords:", b=True) + run(" " + keywords.replace(", ", ", ") + ".")))

    # ---- body
    body_md = abstract_md[abstract_md.index("# 1 Introduction"):]
    parse_section(body_md, b, None)
    refs_md = read(SECTION_FILES[-1])
    ref_start = refs_md.index("## Acknowledgments")
    for name in SECTION_FILES[1:-1]:
        parse_section(read(name), b, None)
    parse_section(refs_md[:ref_start], b, None)

    # ---- acknowledgments, disclosure, references
    ack = ("The authors thank their project guide and the School of Computer Science and Engineering, Vellore Institute "
           "of Technology, for guidance and for access to the curriculum documents used in this study.")
    b.add(para("acknowlegments", run("Acknowledgments.", b=True) + run(" " + ack)))
    b.add(para("acknowlegments", run("Disclosure of Interests.", b=True) +
               run(" The authors have no competing interests to declare that are relevant to the content of this article."),
               '<w:spacing w:before="0"/>'))
    b.add(para("heading1", run("References"),
               '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr><w:ind w:left="567" w:hanging="567"/>'))
    refs_block = refs_md[refs_md.index("## References"):refs_md.index("### LaTeX note")]
    for ln in refs_block.splitlines():
        m = re.match(r"^(\d+)\.\s+(.*)", ln.strip())
        if m:
            b.add(para("referenceitem", inline(m.group(2))))

    if b.eq_no != len(EQUATIONS):
        raise SystemExit(f"equation count mismatch: drafts have {b.eq_no}, EQUATIONS has {len(EQUATIONS)}")

    # ---- write package
    zin = zipfile.ZipFile(template)
    doc = zin.read("word/document.xml").decode("utf-8-sig")
    head = doc[: doc.index("<w:body>") + len("<w:body>")]
    sect = re.search(r"<w:sectPr.*</w:sectPr>", doc, re.S).group(0)
    new_doc = head + "".join(b.body) + sect + "</w:body></w:document>"

    drop = ("word/vbaProject.bin", "word/vbaData.xml", "word/_rels/vbaProject.bin.rels", "word/charts/",
            "customUI/")
    rels = zin.read("word/_rels/document.xml.rels").decode("utf-8-sig")
    rels = re.sub(r'<Relationship [^>]*(vbaProject|relationships/chart)[^>]*/>', "", rels)
    img_rels = "".join(
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="media/{fn}"/>' for rid, fn in b.images)
    rels = rels.replace("</Relationships>", img_rels + "</Relationships>")

    ct = zin.read("[Content_Types].xml").decode("utf-8-sig")
    ct = ct.replace("application/vnd.ms-word.document.macroEnabled.main+xml",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml")
    ct = re.sub(r'<Override PartName="/word/(vbaData|charts/chart1)\.xml"[^>]*/>', "", ct)
    ct = re.sub(r'<Default Extension="(bin|ICO)"[^>]*/>', "", ct)
    ct = re.sub(r'<Default Extension="png"[^>]*/>', '<Default Extension="png" ContentType="image/png"/>', ct)

    root_rels = zin.read("_rels/.rels").decode("utf-8-sig")
    root_rels = re.sub(r'<Relationship [^>]*customUI[^>]*/>', "", root_rels)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name.endswith("/") or any(name.startswith(d) or name == d for d in drop):
                continue
            if name == "word/document.xml":
                data = new_doc.encode("utf-8")
            elif name == "word/_rels/document.xml.rels":
                data = rels.encode("utf-8")
            elif name == "[Content_Types].xml":
                data = ct.encode("utf-8")
            elif name == "_rels/.rels":
                data = root_rels.encode("utf-8")
            else:
                data = zin.read(name)
            zout.writestr(name, data)
        for rid, fn in b.images:
            zout.write(HERE / "figures" / fn, f"word/media/{fn}")
    print(f"wrote {out}  ({len(b.body)} blocks, {b.eq_no} equations, {len(b.images)} figures)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(HERE / "template" / "splnproc2510.docm"))
    ap.add_argument("--out", default=str(HERE / "paper.docx"))
    a = ap.parse_args()
    build(a.template, a.out)
