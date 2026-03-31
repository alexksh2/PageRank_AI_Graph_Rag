# Skill: Markdown → Professional PDF (QuantOS Style)

Converts any `README.md` (or any Markdown file) into a polished, quant-research-style PDF with a dark navy cover page, gold accents, professional tables, and dark-terminal code blocks.

---

## Dependencies

```bash
python3 -m venv /tmp/pdfvenv
/tmp/pdfvenv/bin/pip install markdown weasyprint
```

> Only needed once. The venv persists until the system temp is cleared.

---

## Script

Save as `generate_pdf.py` and run:

```bash
/tmp/pdfvenv/bin/python3 generate_pdf.py
```

```python
#!/usr/bin/env python3
"""
generate_pdf.py
Converts README.md → README.pdf in the same directory.
Style: QuantOS professional quant-research report.

Usage:
    python3 generate_pdf.py [input.md] [output.pdf]

Defaults:
    input  → README.md  (same directory as script)
    output → README.pdf (same directory as script)
"""

import sys
import pathlib
import markdown
from weasyprint import HTML

# ── Config ──────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).parent
INPUT_MD  = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
OUTPUT_PDF = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "README.pdf"

FIRM_NAME = "QuantOS"
DIVISION  = "Quantitative Research &nbsp;|&nbsp; Technology"
DOC_DATE  = "March 3, 2026"
DOC_TYPE  = "Technical Architecture Report"
DOC_CLASS = "Submission — Public"
DOC_VER   = "1.0"
# ────────────────────────────────────────────────────────────────────────────

CSS = """
@page {
  margin: 0;
  size: A4;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── COVER ───────────────────────────────────────────────────────────────── */
.cover {
  width: 210mm;
  height: 297mm;
  background: #0a1628;
  display: flex;
  flex-direction: column;
  page-break-after: always;
  overflow: hidden;
}

.cover-accent-bar {
  width: 100%;
  height: 6px;
  background: linear-gradient(90deg, #c8a951 0%, #e8c97a 50%, #c8a951 100%);
}

.cover-top-bar {
  padding: 22px 36px 18px 36px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(200,169,81,0.25);
}

.cover-firm-name {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 13pt;
  font-weight: bold;
  color: #c8a951;
  letter-spacing: 2px;
  text-transform: uppercase;
}

.cover-division {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 8pt;
  color: #8a9bb5;
  letter-spacing: 1.5px;
  text-transform: uppercase;
}

.cover-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 36px;
}

.cover-tag {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 8pt;
  color: #c8a951;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 16px;
  border-left: 3px solid #c8a951;
  padding-left: 10px;
}

.cover-title {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 34pt;
  font-weight: bold;
  color: #ffffff;
  line-height: 1.2;
  margin-bottom: 10px;
}

.cover-subtitle {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 14pt;
  color: #c8a951;
  margin-bottom: 30px;
  font-style: italic;
}

.cover-divider {
  width: 60px;
  height: 2px;
  background: #c8a951;
  margin-bottom: 28px;
}

.cover-meta {
  display: flex;
  gap: 40px;
}

.cover-meta-item label {
  display: block;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 7pt;
  color: #8a9bb5;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 4px;
}

.cover-meta-item span {
  display: block;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 9.5pt;
  color: #e0e8f0;
  font-weight: 500;
}

.cover-bottom {
  padding: 18px 36px;
  border-top: 1px solid rgba(200,169,81,0.2);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.cover-disclaimer {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 7pt;
  color: #4a5e78;
  max-width: 70%;
  line-height: 1.4;
}

.cover-page-num {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 8pt;
  color: #4a5e78;
}

/* ── CONTENT ─────────────────────────────────────────────────────────────── */
.content {
  padding: 22mm 20mm 18mm 20mm;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 9.5pt;
  color: #1a1a1a;
  line-height: 1.65;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 10px;
  border-bottom: 2px solid #0a1628;
  margin-bottom: 22px;
}

.page-header-left {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 8pt;
  color: #0a1628;
  font-weight: bold;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.page-header-right {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 7.5pt;
  color: #8a9bb5;
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* ── HEADINGS ────────────────────────────────────────────────────────────── */
h1 {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 18pt;
  font-weight: bold;
  color: #0a1628;
  margin: 26px 0 10px 0;
  padding-bottom: 8px;
  border-bottom: 2px solid #0a1628;
}

h2 {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 13pt;
  font-weight: bold;
  color: #0a1628;
  margin: 22px 0 8px 0;
  padding: 6px 10px 6px 14px;
  border-left: 4px solid #c8a951;
  background: #f7f4ee;
}

h3 {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 10.5pt;
  font-weight: bold;
  color: #1a3a5c;
  margin: 16px 0 6px 0;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  border-bottom: 1px solid #d0d7de;
  padding-bottom: 3px;
}

h4 {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 9.5pt;
  font-weight: bold;
  color: #1a3a5c;
  margin: 12px 0 4px 0;
}

/* ── BODY TEXT ───────────────────────────────────────────────────────────── */
p  { margin: 0 0 9px 0; }
strong { font-weight: 700; color: #0a1628; }
em { font-style: italic; color: #444; }
a  { color: #1a3a5c; text-decoration: none; border-bottom: 1px dotted #c8a951; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 18px 0; }

/* ── BLOCKQUOTE ──────────────────────────────────────────────────────────── */
blockquote {
  margin: 12px 0;
  padding: 10px 16px;
  background: #f0f4f8;
  border-left: 4px solid #1a3a5c;
  border-radius: 0 4px 4px 0;
}
blockquote p { color: #2c4a6e; font-size: 9.5pt; font-style: italic; margin: 0; }

/* ── LISTS ───────────────────────────────────────────────────────────────── */
ul, ol { padding-left: 20px; margin: 6px 0 10px 0; }
li { margin-bottom: 4px; }

/* ── CODE BLOCKS ─────────────────────────────────────────────────────────── */
pre {
  background: #0f1e35;
  border-radius: 4px;
  padding: 13px 16px;
  margin: 10px 0;
  page-break-inside: avoid;
  border-left: 3px solid #c8a951;
}
pre code {
  font-family: 'Courier New', Courier, monospace;
  font-size: 7.8pt;
  color: #c8d8e8;
  background: none;
  border: none;
  padding: 0;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.55;
}
code {
  font-family: 'Courier New', Courier, monospace;
  font-size: 8.2pt;
  background: #eef2f7;
  color: #1a3a5c;
  border: 1px solid #c8d4e0;
  border-radius: 3px;
  padding: 1px 5px;
  word-break: break-all;
}

/* ── TABLES ──────────────────────────────────────────────────────────────── */
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 8.5pt; page-break-inside: avoid; }
thead tr { background: #0a1628; }
thead th {
  color: #c8a951;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-weight: 600;
  font-size: 8pt;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  padding: 7px 10px;
  border: 1px solid #1a3a5c;
  text-align: left;
}
tbody td {
  padding: 6px 10px;
  border: 1px solid #d0d7de;
  color: #1a1a1a;
  vertical-align: top;
  line-height: 1.45;
}
tbody tr:nth-child(odd)  td { background: #ffffff; }
tbody tr:nth-child(even) td { background: #f5f8fc; }
"""


def build_cover(title: str, subtitle: str, tag: str) -> str:
    return f"""
<div class="cover">
  <div class="cover-accent-bar"></div>
  <div class="cover-top-bar">
    <div class="cover-firm-name">{FIRM_NAME}</div>
    <div class="cover-division">{DIVISION}</div>
  </div>
  <div class="cover-body">
    <div class="cover-tag">{tag}</div>
    <div class="cover-title">{title}</div>
    <div class="cover-subtitle">{subtitle}</div>
    <div class="cover-divider"></div>
    <div class="cover-meta">
      <div class="cover-meta-item"><label>Document Type</label><span>{DOC_TYPE}</span></div>
      <div class="cover-meta-item"><label>Date</label><span>{DOC_DATE}</span></div>
      <div class="cover-meta-item"><label>Classification</label><span>{DOC_CLASS}</span></div>
      <div class="cover-meta-item"><label>Version</label><span>{DOC_VER}</span></div>
    </div>
  </div>
  <div class="cover-bottom">
    <div class="cover-disclaimer">
      Every technical claim in this document maps to a specific file path, endpoint, or log entry
      in the accompanying repository. Responsible AI gaps are documented explicitly.
    </div>
    <div class="cover-page-num">Page 1</div>
  </div>
</div>
"""


def md_to_pdf(input_md: pathlib.Path, output_pdf: pathlib.Path) -> None:
    md_content = input_md.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "toc"],
    )

    # Extract title & subtitle from first H1 / blockquote if present
    lines = md_content.splitlines()
    title    = next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), input_md.stem)
    subtitle = next((l.lstrip("> ").strip() for l in lines if l.startswith(">")), "")
    tag      = "Technical Report &nbsp;|&nbsp; 2026"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{build_cover(title, subtitle, tag)}
<div class="content">
  <div class="page-header">
    <div class="page-header-left">{title} — Technical Report</div>
    <div class="page-header-right">{FIRM_NAME} &nbsp;|&nbsp; Quantitative Research &nbsp;|&nbsp; 2026</div>
  </div>
  {body}
</div>
</body>
</html>"""

    HTML(string=html).write_pdf(str(output_pdf))
    print(f"✓  {input_md.name}  →  {output_pdf}  ({output_pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    md_to_pdf(INPUT_MD, OUTPUT_PDF)
```

---

## Quick usage

```bash
# Default: README.md → README.pdf in current directory
/tmp/pdfvenv/bin/python3 generate_pdf.py

# Custom paths
/tmp/pdfvenv/bin/python3 generate_pdf.py docs/report.md output/report.pdf
```

---

## Customisation knobs (top of script)

| Variable | Default | Description |
|---|---|---|
| `FIRM_NAME` | `QuantOS` | Branding shown on cover and running header |
| `DIVISION` | `Quantitative Research \| Technology` | Sub-division line on cover |
| `DOC_DATE` | `March 3, 2026` | Date shown in cover metadata grid |
| `DOC_TYPE` | `Technical Architecture Report` | Document type label |
| `DOC_CLASS` | `Submission — Public` | Classification label |
| `DOC_VER` | `1.0` | Version label |

---

## Design tokens

| Token | Value | Used for |
|---|---|---|
| Navy | `#0a1628` | Cover bg, H1 border, table header bg, page header rule |
| Gold | `#c8a951` | Accent bar, H2 left border, table header text, code block left border |
| Mid-navy | `#1a3a5c` | H3, H4, inline code text, blockquote border |
| Warm cream | `#f7f4ee` | H2 background |
| Terminal bg | `#0f1e35` | Code block background |
| Terminal text | `#c8d8e8` | Code block font colour |
