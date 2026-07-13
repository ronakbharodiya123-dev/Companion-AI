"""
Standalone test: table extraction on LG Express Cool PDF.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from typing import List
import pdfplumber
import fitz

PDF_PATH = r"D:\Companion AI\backend\data\uploads\Refrigerator_LG_20260319_180709_LG EXPRESS COOL.pdf"
EMPTY_T  = 0.40
COV_T    = 0.40


def is_lossy(rows: List[List], page_full_text: str = ""):
    total_cells = sum(len(row) for row in rows)
    if total_cells == 0:
        return True, "empty table"

    empty = sum(
        1 for row in rows for cell in row
        if cell is None or not str(cell).strip()
    )
    if (empty / total_cells) > EMPTY_T:
        return True, f"empty-cell={empty/total_cells:.0%}"

    num_cols = len(rows[0]) if rows else 0
    num_rows = len(rows)
    if page_full_text.strip() and num_cols > 1 and num_rows > 3:
        table_text = " ".join(
            str(cell) for row in rows for cell in row
            if cell is not None and str(cell).strip()
        )
        coverage = len(table_text) / max(len(page_full_text), 1)
        if coverage < COV_T:
            return True, f"coverage={coverage:.0%}"

    return False, "clean"


def table_to_markdown(rows: List[List]) -> str:
    def cell(v): return str(v).strip() if v is not None else ""
    header  = "| " + " | ".join(cell(c) for c in rows[0]) + " |"
    divider = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body    = ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows[1:]]
    return "\n".join([header, divider] + body)


# ── run ───────────────────────────────────────────────────────────────────────

print("=" * 70)
print(f"PDF: {PDF_PATH}")
print("=" * 70)

texts, metas = [], []
fallback_added = set()
diag = []

with pdfplumber.open(PDF_PATH) as pdf, fitz.open(PDF_PATH) as fitz_doc:
    print(f"Total pages: {len(pdf.pages)}\n")

    for page_num, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()
        if not tables:
            continue

        page_full_text = page.extract_text() or ""

        for table in tables:
            non_empty = [r for r in table if any(c for c in r if c)]
            if not non_empty:
                continue

            lossy, reason = is_lossy(non_empty, page_full_text)
            diag.append((page_num, len(non_empty[0]), len(non_empty), lossy, reason))

            if lossy:
                if page_num not in fallback_added:
                    raw = fitz_doc[page_num - 1].get_text("text").strip()
                    if raw:
                        texts.append(f"Specifications on page {page_num}:\n{raw}")
                        metas.append({"source": "table_text", "page": page_num})
                        fallback_added.add(page_num)
            else:
                md = table_to_markdown(non_empty)
                texts.append(f"Table on page {page_num}:\n{md}")
                metas.append({"source": "table", "page": page_num})

# ── diagnostics ───────────────────────────────────────────────────────────────

print("--- Per-table diagnostics ---")
for page_num, cols, rows, lossy, reason in diag:
    status = "FALLBACK" if lossy else "MARKDOWN"
    print(f"  Page {page_num:2d} | cols={cols} rows={rows} | {status:8s} | {reason}")

n_md  = sum(1 for m in metas if m["source"] == "table")
n_fb  = sum(1 for m in metas if m["source"] == "table_text")
print(f"\n[OK] Total chunks : {len(texts)}  ({n_md} markdown, {n_fb} raw-text fallback)\n")

# ── print every chunk ─────────────────────────────────────────────────────────

for idx, (text, meta) in enumerate(zip(texts, metas), 1):
    kind = "MARKDOWN TABLE" if meta["source"] == "table" else "RAW-TEXT FALLBACK"
    print("-" * 70)
    print(f"CHUNK {idx}  |  {kind}  |  page {meta['page']}")
    print("-" * 70)
    print(text)
    print()

print("=" * 70)
print(f"DONE -- {n_md} markdown tables  |  {n_fb} raw-text fallback pages")
print("=" * 70)
