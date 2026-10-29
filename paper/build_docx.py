"""Build bao_cao_nghien_cuu.docx (and a PDF for proofreading) from bao_cao_nghien_cuu.md.

Windows only: it needs Microsoft Word, driven through COM via PowerShell. The Markdown is turned
into HTML with a small converter that only covers what the report uses (headings, paragraphs,
lists, pipe tables, images, bold/italic/code, inline and display math). Display equations are
rendered to PNG with Matplotlib mathtext; inline math becomes text with sub/superscripts. Word then
opens the HTML, embeds the pictures and saves .docx and .pdf.

    .venv\\Scripts\\python.exe paper\\build_docx.py
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent
SOURCE = PAPER / "bao_cao_nghien_cuu.md"
TARGET = PAPER / "bao_cao_nghien_cuu.docx"
PDF = PAPER / "bao_cao_nghien_cuu.pdf"
TEXT_WIDTH_PX = 590  # A4 minus the margins set in WORD_SCRIPT, at Word's 96 px per inch
FIGURE_WIDTH_PX = TEXT_WIDTH_PX
MATH_DPI = 300
MATH_FONTSIZE = 13

GREEK = {r"\delta": "δ", r"\dots": "…", r"\le": "≤", r"\ge": "≥", r"\in": "∈", r"\mathbb{1}": "𝟙", r"\cdot": "·"}

CSS = """
body { font-family: 'Times New Roman'; font-size: 12pt; line-height: 1.3; }
h1 { font-size: 18pt; text-align: center; margin-bottom: 4pt; }
h2 { font-size: 15pt; margin-top: 18pt; }
h3 { font-size: 13pt; margin-top: 12pt; }
p { margin: 0 0 6pt 0; text-align: justify; }
p.caption { font-style: italic; text-align: center; font-size: 10.5pt; margin-bottom: 12pt; }
p.center, p.figure, p.equation { text-align: center; }
table { border-collapse: collapse; margin: 6pt auto 10pt auto; font-size: 10pt; }
th, td { border: 1px solid #808080; padding: 1pt 4pt; vertical-align: top; }
td p, th p { margin: 0; text-align: left; line-height: 1.1; }
p.lead { text-align: left; }
p.note { margin: 0 0 0 14pt; padding: 3pt 8pt; border-left: 3pt solid #3b4fd6; background: #eef1fb; font-size: 11pt; }th { background: #e8e8e8; font-weight: bold; }
code { font-family: Consolas; font-size: 10pt; }
li { margin-bottom: 3pt; text-align: justify; }
"""


def _math_lines(tex: str) -> list[str]:
    """mathtext has no cases environment, so a cases block becomes one line per case."""
    # mathtext also spells some symbols differently from LaTeX.
    tex = tex.replace(r"\text{", r"\mathrm{").replace(r"\ge ", r"\geq ").replace(r"\le ", r"\leq ")
    if r"\begin{cases}" not in tex:
        return [tex]
    head, rest = tex.split(r"\begin{cases}", 1)
    body, tail = rest.split(r"\end{cases}", 1)
    lines = []
    for row in body.split(r"\\"):
        value, condition = (part.strip() for part in row.split("&", 1))
        lines.append(rf"{head.strip()} {value.rstrip(',')} \quad \mathrm{{nếu}}\ {condition}")
    tail = tail.strip().removeprefix(r"\qquad").strip()
    return lines + ([tail] if tail else [])


def render_math(tex: str, work: Path, stem: str) -> list[tuple[Path, int, int]]:
    """Render display math to PNGs, one per line; sizes are Word pixels (96 per inch), capped to the text width."""
    images = []
    for index, line in enumerate(_math_lines(tex)):
        path = work / f"{stem}_{index}.png"
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${line}$", fontsize=MATH_FONTSIZE)
        fig.savefig(path, dpi=MATH_DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        plt.close(fig)
        width, height = (value * 96 / MATH_DPI for value in plt.imread(path).shape[1::-1])
        scale = min(1.0, TEXT_WIDTH_PX / width)
        images.append((path, round(width * scale), round(height * scale)))
    return images


def inline_math(tex: str) -> str:
    """Approximate inline LaTeX as HTML text with sub/superscripts."""
    text = tex.replace(r"\text{", "{").replace(r"\mathrm{", "{").replace("{,}", ",")
    for command, symbol in GREEK.items():
        text = text.replace(command, symbol)
    text = text.replace(r"\,", " ").replace(r"\ ", " ")

    def script(match: re.Match) -> str:
        tag = "sub" if match.group(1) == "_" else "sup"
        return f"<{tag}>{match.group(2) or match.group(3)}</{tag}>"

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"([_^])(?:\{([^{}]*)\}|([A-Za-z0-9]))", script, text)
    text = text.replace("{", "").replace("}", "")
    return f"<i>{text}</i>"


def inline(text: str) -> str:
    """Inline Markdown: code, math, images are handled by the caller; bold, italic, links here."""
    parts = re.split(r"(`[^`]+`|\$[^$]+\$)", text)
    output = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            output.append(f"<code>{html.escape(part[1:-1])}</code>")
        elif part.startswith("$") and part.endswith("$") and len(part) > 1:
            output.append(inline_math(part[1:-1]))
        else:
            escaped = html.escape(part, quote=False)
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
            escaped = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", escaped)
            escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", escaped)
            output.append(escaped)
    return "".join(output)


def to_html(markdown: str, work: Path) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    list_stack: list[str] = []
    equation = 0

    def close_lists() -> None:
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped == "---":
            close_lists()
            i += 1
            continue
        if stripped.startswith("$$"):
            close_lists()
            tex = stripped.strip("$").strip()
            equation += 1
            for image, width, height in render_math(tex, work, f"equation_{equation}"):
                out.append(f'<p class="equation"><img src="{image.as_uri()}" width="{width}" height="{height}"></p>')
            i += 1
            continue
        if stripped.startswith(">"):
            # A blockquote is a plain-language note: one shaded paragraph per non-empty line.
            close_lists()
            notes = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                text = lines[i].strip().lstrip(">").strip()
                if text:
                    notes.append(text)
                i += 1
            for index, text in enumerate(notes):
                # Word's HTML import ignores a second class, so the gap after the note is set inline.
                gap = ' style="margin-bottom:10pt"' if index == len(notes) - 1 else ""
                out.append(f'<p class="note"{gap}>{inline(text)}</p>')
            continue
        heading = re.match(r"^(#{1,3}) (.*)$", stripped)
        if heading:
            close_lists()
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        image = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if image:
            close_lists()
            # Copy into the ASCII temp folder: Word mangles non-ASCII file URIs and then leaves the picture linked.
            path = work / Path(image.group(2)).name
            shutil.copyfile(PAPER / image.group(2), path)
            width, height = plt.imread(path).shape[1::-1]
            scale = FIGURE_WIDTH_PX / width
            out.append(f'<p class="figure"><img src="{path.as_uri()}" width="{FIGURE_WIDTH_PX}" height="{round(height * scale)}" alt="{html.escape(image.group(1))}"></p>')
            i += 1
            continue
        if stripped.startswith("|"):
            close_lists()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[cell.strip() for cell in re.split(r"(?<!\\)\|", row.strip("|"))] for row in rows]
            header, body = cells[0], [row for row in cells[2:]]
            out.append("<table><tr>" + "".join(f"<th><p>{inline(c)}</p></th>" for c in header) + "</tr>")
            for row in body:
                out.append("<tr>" + "".join(f"<td><p>{inline(c.replace(chr(92) + '|', '|'))}</p></td>" for c in row) + "</tr>")
            out.append("</table>")
            continue
        item = re.match(r"^(\s*)([-*]|\d+\.) (.*)$", line)
        if item:
            depth = len(item.group(1)) // 2
            kind = "ol" if item.group(2)[0].isdigit() else "ul"
            while len(list_stack) > depth + 1:
                out.append(f"</{list_stack.pop()}>")
            if len(list_stack) < depth + 1:
                list_stack.append(kind)
                out.append(f"<{kind}>")
            out.append(f"<li>{inline(item.group(3))}</li>")
            i += 1
            continue
        if list_stack and line.startswith("  "):  # continuation paragraph inside a list item
            out.append(f"<p>{inline(stripped)}</p>")
            i += 1
            continue
        close_lists()
        paragraph = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||!\[|\$\$|>|\s*[-*] |\s*\d+\. |---)", lines[i].strip()):
            paragraph.append(lines[i].strip())
            i += 1
        text = " ".join(paragraph)
        css = ' class="caption"' if text.startswith("*Hình") or text.startswith("*(") else ""
        if text.startswith("**Bảng"):
            css = ' class="lead"'  # table captions hold long file names; justifying them stretches the words
        if text.startswith("*") and text.endswith("*") and css:
            text = text[1:-1]
        out.append(f"<p{css}>{inline(text)}</p>")
    close_lists()
    return f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{"".join(out)}</body></html>'


WORD_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($args[0], $false, $true)
    foreach ($shape in @($doc.InlineShapes)) {
        if ($shape.LinkFormat -ne $null) { $shape.LinkFormat.SavePictureWithDocument = $true; $shape.LinkFormat.BreakLink() }
    }
    foreach ($field in @($doc.Fields)) { if ($field.Type -eq 67) { $field.Unlink() } }   # leftover INCLUDEPICTURE fields
    $doc.ActiveWindow.View.Type = 3   # print layout, so page setup applies
    $setup = $doc.PageSetup
    $setup.PaperSize = 7              # A4
    $setup.TopMargin = 72; $setup.BottomMargin = 72; $setup.LeftMargin = 79; $setup.RightMargin = 72
    $footer = $doc.Sections.Item(1).Footers.Item(1)
    $footer.PageNumbers.Add(1, $true) | Out-Null
    foreach ($table in @($doc.Tables)) { $table.Rows.Alignment = 1; $table.AllowAutoFit = $true; $table.AutoFitBehavior(1) }
    $doc.SaveAs2($args[1], 16)
    $doc.ExportAsFixedFormat($args[2], 17)
    $doc.Close($false)
} finally {
    $word.Quit()
}
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        page = work / "report.html"
        page.write_text(to_html(SOURCE.read_text(encoding="utf-8"), work), encoding="utf-8")
        script = work / "word.ps1"
        script.write_text(WORD_SCRIPT, encoding="utf-8-sig")
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(page), str(TARGET), str(PDF)],
            capture_output=True, text=True,
        )
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            return result.returncode
    print(f"wrote {TARGET.name} and {PDF.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
