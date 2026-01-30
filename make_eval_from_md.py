import re, json
from pathlib import Path
from datetime import datetime

PAGE_RE = re.compile(r"START OF PAGE:\s*(\d+)\s*\n", re.IGNORECASE)
TABLE_RE = re.compile(r"(\|.*\|\n\|[-:\s|]+\|\n(?:\|.*\|\n)+)", re.MULTILINE)
NUM_RE = re.compile(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?")

def split_pages(md: str):
    parts = PAGE_RE.split(md)
    pages = {}
    for i in range(1, len(parts), 2):
        pages[int(parts[i])] = parts[i+1]
    return pages

def normalize_number(s: str):
    if not s:
        return None
    m = NUM_RE.search(s)
    if not m:
        return None
    raw = m.group(0)

    # normalize EU/US separators
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    else:
        if "," in raw:
            parts = raw.split(",")
            if len(parts[-1]) == 3:
                raw = raw.replace(",", "")
            else:
                raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None

def parse_md_table(tbl_md: str):
    lines = [ln.strip() for ln in tbl_md.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return []
    def split_row(row: str):
        return [c.strip() for c in row.strip().strip("|").split("|")]
    rows = [split_row(lines[0])]
    for ln in lines[2:]:
        if ln.startswith("|"):
            rows.append(split_row(ln))
    return rows

def col_index(header, wanted_substrings):
    h = [c.lower() for c in header]
    for w in wanted_substrings:
        w = w.lower()
        for i, cell in enumerate(h):
            if w in cell:
                return i
    return None

def find_table_with_row(pages, row_contains):
    for pno, txt in pages.items():
        for tbl_md in TABLE_RE.findall(txt):
            grid = parse_md_table(tbl_md)
            if not grid:
                continue
            for r in grid[1:]:
                if any(row_contains.lower() in (c.lower()) for c in r):
                    return pno, grid
    return None, None

def row_by_name(grid, name):
    for r in grid[1:]:
        if r and name.lower() in (r[0].lower()):
            return r
    # fallback: any cell
    for r in grid[1:]:
        if any(name.lower() in c.lower() for c in r):
            return r
    return None

def build_items_for_doc(pdf_doc_name: str, md_path: str):
    md = Path(md_path).read_text(encoding="utf-8")
    pages = split_pages(md)

    items = []

    # Net sales
    pno, grid = find_table_with_row(pages, "Net sales")
    if grid:
        header = grid[0]
        row = row_by_name(grid, "Net sales")

        targets = [
            ("Q1", ["q1 2025", "q1"], "Merck Group net sales in Q1 (in € million)"),
            ("Q2", ["q2 2025", "q2"], "Merck Group net sales in Q2 (in € million)"),
            ("Q3", ["q3 2025", "q3"], "Merck Group net sales in Q3 (in € million)"),
            ("9M", ["9m 2025", "9m"], "Merck Group net sales in 9M (in € million)"),
        ]

        if row:
            for key, wants, q in targets:
                ci = col_index(header, wants)
                if ci is None or ci >= len(row):
                    continue
                val = normalize_number(row[ci])
                if val is None:
                    continue
                items.append({
                    "id": f"{Path(pdf_doc_name).stem}-NETSALES-{key}",
                    "question": f"According to the {pdf_doc_name}, what were {q}?",
                    "gold_answer": {"value": val, "unit": "€ million"},
                    "answer_format": "number",
                    "tolerance": {"absolute": 0.5},
                    "must_cite": True,
                    "evidence": [{"doc": pdf_doc_name, "page": pno}],
                    "gold_span": f"{header[ci]} | Net sales = {row[ci]}",
                })

    # EBITDA pre
    pno2, grid2 = find_table_with_row(pages, "EBITDA pre")
    if grid2:
        header = grid2[0]
        row = row_by_name(grid2, "EBITDA pre")
        targets = [
            ("Q1", ["q1 2025", "q1"], "Merck Group EBITDA pre in Q1 (in € million)"),
            ("Q2", ["q2 2025", "q2"], "Merck Group EBITDA pre in Q2 (in € million)"),
            ("Q3", ["q3 2025", "q3"], "Merck Group EBITDA pre in Q3 (in € million)"),
            ("9M", ["9m 2025", "9m"], "Merck Group EBITDA pre in 9M (in € million)"),
        ]
        if row:
            for key, wants, q in targets:
                ci = col_index(header, wants)
                if ci is None or ci >= len(row):
                    continue
                val = normalize_number(row[ci])
                if val is None:
                    continue
                items.append({
                    "id": f"{Path(pdf_doc_name).stem}-EBITDApre-{key}",
                    "question": f"According to the {pdf_doc_name}, what was {q}?",
                    "gold_answer": {"value": val, "unit": "€ million"},
                    "answer_format": "number",
                    "tolerance": {"absolute": 0.5},
                    "must_cite": True,
                    "evidence": [{"doc": pdf_doc_name, "page": pno2}],
                    "gold_span": f"{header[ci]} | EBITDA pre = {row[ci]}",
                })

    return items

def main(md_files, out_path):
    all_items = []
    for md_path in md_files:
        stem = Path(md_path).stem  # "2025-Q1"
        pdf_doc = f"{stem}-Financial-Statement-EN.pdf"
        all_items.extend(build_items_for_doc(pdf_doc, md_path))

    payload = {
        "dataset_id": "merck_eval_gold_from_llamaparse_md_2025",
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "task": "rag_qa_eval",
        "notes": "Gold numeric values extracted from LlamaParse markdown tables with page markers.",
        "items": all_items
    }

    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_items)} items -> {out_path}")

if __name__ == "__main__":
    import sys
    *mds, out = sys.argv[1:]
    main(mds, out)
