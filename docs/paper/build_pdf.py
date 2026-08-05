"""Convert issw-stabs-draft.md to LaTeX and compile with tectonic."""

import re
import subprocess
from pathlib import Path

SRC = Path(__file__).parent / "issw-stabs-draft.md"
TEX = Path(__file__).parent / "issw-stabs-draft.tex"

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1.1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{mathpazo}
\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip}
\setlength{\parskip}{6pt}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection.}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}
\begin{document}
"""

UNICODE_MAP = {
    "\u2013": "--", "\u2014": "---", "\u2018": "`", "\u2019": "'",
    "\u201c": "``", "\u201d": "''", "\u00d7": r"$\times$", "\u2265": r"$\ge$",
    "\u2264": r"$\le$", "\u2248": r"$\approx$", "\u00b1": r"$\pm$",
    "\u2208": r"$\in$", "\u00b7": r"$\cdot$", "\u2026": r"\ldots{}",
    "\u03b3": r"$\gamma$", "\u03c6": r"$\phi$", "\u03c9": r"$\omega$",
    "\u03b6": r"$\zeta$", "\u00a0": "~",
    "\u010d": r"\v{c}", "\u00e9": r"\'e", "\u00fc": r"\"u", "\u00f8": r"\o{}",
    "\u00e4": r"\"a", "\u00ed": r"\'i", "\u00f3": r"\'o", "\u00e1": r"\'a",
    "\u00f6": r"\"o", "\u00e8": r"\`e", "\u00e7": r"\c{c}", "\u00f1": r"\~n",
    "\u00b0": r"\textdegree{}",
}

SPECIALS = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"}


def esc(text: str) -> str:
    """Escape LaTeX specials in prose (not applied inside math)."""
    out = []
    for ch in text:
        out.append(SPECIALS.get(ch, ch))
    return "".join(out)


def inline(text: str) -> str:
    """Convert one line of markdown prose to LaTeX."""
    for k, v in UNICODE_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r'"([^"]+)"', r"``\1''", text)
    # protect math spans, escape everything else
    parts = re.split(r"(\$`.*?`\$|\$[^$]+\$)", text)
    for i, p in enumerate(parts):
        if p.startswith("$`") and p.endswith("`$"):
            parts[i] = "$" + p[2:-2] + "$"
        elif p.startswith("$") and p.endswith("$") and len(p) > 1:
            pass
        else:
            p = esc(p)
            p = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1\\footnote{\\url{\2}}", p)
            p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
            p = re.sub(r"(?<![\w\\])\*([^*]+)\*", r"\\emph{\1}", p)
            p = re.sub(r"`([^`]+)`", r"\\texttt{\1}", p)
            parts[i] = p
    return "".join(parts)


def convert(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    lines = md.split("\n")
    out = [PREAMBLE]
    i = 0
    in_refs = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("$$"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("$$"):
                block.append(lines[i])
                i += 1
            out.append("\\[" + "\n".join(block) + "\\]")
        elif line.startswith("# "):
            out.append("\\begin{center}{\\LARGE\\bfseries "
                       + inline(line[2:]) + "}\\end{center}")
        elif line.startswith("## "):
            title = line[3:].strip()
            m = re.match(r"(\d+)\.\s+(.*)", title)
            in_refs = title.lower() == "references"
            if m:
                out.append("\\section{" + inline(m.group(2)) + "}")
            else:
                out.append("\\section*{" + inline(title) + "}")
        elif line.startswith("- ") and in_refs:
            item = inline(line[2:])
            while i + 1 < len(lines) and lines[i + 1].startswith("  "):
                i += 1
                item += " " + inline(lines[i].strip())
            out.append("\\hangindent=1.5em\\hangafter=1\\noindent " + item + "\\par")
        elif line.strip() == "":
            out.append("")
        else:
            para = inline(line)
            while i + 1 < len(lines) and lines[i + 1].strip() != "" and not \
                    lines[i + 1].startswith(("#", "$$", "- ")):
                i += 1
                para += "\n" + inline(lines[i])
            out.append(para)
        i += 1
    out.append("\\end{document}")
    return "\n".join(out)


TEX.write_text(convert(SRC.read_text()))
subprocess.run(["tectonic", str(TEX)], check=True)
print("built", TEX.with_suffix(".pdf"))
