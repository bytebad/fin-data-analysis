import json
import os
import random
import time
from collections import Counter, defaultdict
import re
from math import isfinite

from rag_pipeline import RAGPipeline
import logging
import shutil
from math import isfinite

logger = logging.getLogger("evaluation")
logging.basicConfig(level=logging.INFO)

NUM_RE = re.compile(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?")


def to_gold_unit(pred_value: float, pred_unit: str, gold_unit: str, gold_type: str, raw_text: str):
    """
    Convert predicted value to match gold unit/type when possible.
    """
    if pred_value is None or not isfinite(pred_value):
        return None

    text_l = (raw_text or "").lower()
    pred_unit = pred_unit or ""

    # Percent handling: allow ratio vs percent
    if gold_type == "percent":
        # normalize to percent number (12.3 means 12.3%)
        # if model returned 0.123, treat as 12.3
        if pred_value <= 1.0 and pred_value >= -1.0:
            # only scale if it looks like a ratio AND gold is percent
            return pred_value * 100.0
        return pred_value

    # Money / € million handling
    if gold_unit == "€ million":
        # scale only if the answer indicates bn/billion near the output
        # (JSON unit may say € million already; in that case do nothing)
        if pred_unit in ("€ billion", "bn"):
            return pred_value * 1000.0
        return pred_value

    return pred_value

def has_grounding(answer_text: str) -> bool:
    try:
        obj = json.loads(answer_text)
        q = (obj.get("evidence_quote") or "").strip()
        return bool(q) and bool(NUM_RE.search(q))
    except Exception:
        return False

def within_tolerance(pred: float, gold: float, abs_tol: float, rel_tol: float):
    tol = max(abs_tol, abs(gold) * rel_tol)
    return abs(pred - gold) <= tol

def value_correct_json(answer_text: str, gold_value: float, gold_type: str, gold_unit: str, abs_tol: float):
    pred_value, pred_unit, found = parse_predicted_value(answer_text)
    if not found or pred_value is None:
        return False, None, pred_unit

    pred_value = to_gold_unit(pred_value, pred_unit, gold_unit, gold_type, answer_text)
    if pred_value is None:
        return False, None, pred_unit

    # Relative tolerance helps for large € million values
    rel_tol = 0.001 if gold_unit == "€ million" else 0.0005  # tweak as needed
    ok = within_tolerance(pred_value, gold_value, abs_tol=abs_tol, rel_tol=rel_tol)
    return ok, pred_value, pred_unit

def _norm_doc(name: str) -> str:
    """Normalize doc names to basenames to avoid path mismatches."""
    if not name:
        return ""
    return os.path.basename(name).strip()


def recall_at_k(eval_items, pipeline, k: int = 5, debug_misses: int = 0):
    """
    Global Recall@k over eval_items.
    Returns a dict with recall + useful counts.

    debug_misses: print up to N misses with retrieved (doc,page) pairs.
    """
    hits = 0
    total = 0
    skipped = 0
    miss_examples = []
    doc_miss_counter = Counter()

    t0 = time.time()

    for item in eval_items:
        # Build gold set of (doc,page)
        gold_pages = {
            (_norm_doc(ev.get("doc")), int(ev.get("page")))
            for ev in item.get("evidence", [])
            if ev.get("doc") and ev.get("page") is not None
        }

        if not gold_pages:
            skipped += 1
            continue

        total += 1
        question = item.get("question", "")

        # Retrieve top-k
        nodes = pipeline.query_documents(question, top_k=k)

        # Build retrieved set of (doc,page)
        retrieved = set()
        for n in nodes:
            meta = getattr(n, "metadata", {}) or {}
            doc = _norm_doc(meta.get("file_name"))
            page = meta.get("page_number")
            if doc and page is not None:
                try:
                    retrieved.add((doc, int(page)))
                except (TypeError, ValueError):
                    pass

        hit = bool(gold_pages & retrieved)
        if hit:
            hits += 1
        else:
            # Track misses for debugging
            gold_docs = {d for (d, _) in gold_pages}
            for d in gold_docs:
                doc_miss_counter[d] += 1
            if len(miss_examples) < debug_misses:
                miss_examples.append({
                    "id": item.get("id"),
                    "question": question,
                    "gold": sorted(list(gold_pages))[:10],
                    "retrieved_topk": sorted(list(retrieved))[:10],
                })

    elapsed = time.time() - t0
    recall = hits / max(1, total)

    return {
        "k": k,
        "recall": recall,
        "hits": hits,
        "total": total,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 3),
        "miss_examples": miss_examples,
        "misses_by_doc": doc_miss_counter,
    }


def extract_number_candidates(text: str):
    if not text:
        return []
    out = []
    for m in NUM_RE.finditer(text):
        v = normalize_number_token(m.group(0))
        if v is None:
            continue
        out.append((v, m.start(), m.end()))
    return out

def pick_best_candidate(text: str, gold_value: float, gold_type: str):
    """
    gold_type: 'percent' or 'number'
    Heuristic:
      - if percent: prefer candidates near '%' sign
      - else prefer candidates near '€', 'eur', 'million', 'bn'
      - finally pick candidate closest to gold_value
    """
    cands = extract_number_candidates(text)
    if not cands:
        return None

    text_l = text.lower()

    scored = []
    for v, s, e in cands:
        window = text_l[max(0, s-12):min(len(text_l), e+12)]
        bonus = 0.0

        if gold_type == "percent":
            if "%" in window:
                bonus += 5.0
            # if model outputs ratio (0.123) but gold is 12.3, consider scaling
            # we don't decide here; we’ll handle in value_correct
        else:
            if "€" in window or "eur" in window:
                bonus += 3.0
            if "million" in window or " m" in window:
                bonus += 2.0
            if "bn" in window or "billion" in window:
                bonus += 2.0  # unit scaling handled below

        scored.append((bonus, v, s, e, window))

    # take top few by bonus then choose closest to gold
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]
    top.sort(key=lambda x: abs(x[1] - gold_value))
    return top[0][1]

def value_correct(answer_text: str, gold_value: float, tol_abs: float, gold_type: str, gold_unit: str):
    pred = pick_best_candidate(answer_text or "", gold_value, gold_type)
    if pred is None or not isfinite(pred):
        return False, None

    text_l = (answer_text or "").lower()

    # handle percent ratio outputs: if gold is percent and pred in [0,1] while gold in [1,100]
    if gold_type == "percent":
        # try raw pred and scaled pred*100, choose closer
        options = [pred, pred*100.0]
        pred = min(options, key=lambda x: abs(x - gold_value))
        return abs(pred - gold_value) <= tol_abs, pred

    # handle € million vs bn output (very basic)
    if gold_unit == "€ million":
        # if answer contains bn/billion near number, scale
        if "bn" in text_l or "billion" in text_l:
            # ambiguous if the chosen candidate was the bn one; but this helps in practice
            pred = pred * 1000.0

    return abs(pred - gold_value) <= tol_abs, pred

def strip_json_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def eval_rag_numeric(eval_items, pipeline, k=10):
    retrieval_hits = 0
    value_hits = 0
    grounded_value_hits = 0
    total = 0

    for item in eval_items:
        gold_pages = {(_norm_doc(ev["doc"]), int(ev["page"])) for ev in item["evidence"]}
        gold = item["gold_answer"]
        gold_value = float(gold["value"])
        gold_type = gold.get("type", "number")
        gold_unit = gold.get("unit", "")
        abs_tol = float(item.get("tolerance", {}).get("absolute", 0.5 if gold_type == "number" else 0.05))

        nodes = pipeline.query_documents(item["question"], top_k=k)

        retrieved_pages = set()
        for n in nodes:
            meta = getattr(n, "metadata", {}) or {}
            doc = _norm_doc(meta.get("file_name"))
            page = meta.get("page_number")
            if doc and page is not None:
                retrieved_pages.add((doc, int(page)))

        retrieved_ok = any(
            (gd == rd) and (gp in (rp, rp-1, rp+1))
            for (gd, gp) in gold_pages
            for (rd, rp) in retrieved_pages
        )

        if retrieved_ok:
            retrieval_hits += 1
        else:
            total += 1
            continue  # skip Gemini call

        answer = pipeline.generate_answer_eval(
            item["question"],
            nodes,
            gold_type=gold_type,
            gold_unit=gold_unit
        )

        ok_val, pred_val, pred_unit = value_correct_json(
            answer_text=answer,
            gold_value=gold_value,
            gold_type=gold_type,
            gold_unit=gold_unit,
            abs_tol=abs_tol
        )

        if ok_val:
            value_hits += 1
            if is_grounded(answer):
                grounded_value_hits += 1
        total += 1

    return {
        "k": k,
        "total": total,
        "recall_at_k": retrieval_hits / max(1, total),
        "value_accuracy_given_retrieval_hit": value_hits / max(1, retrieval_hits),
        "grounded_value_accuracy": grounded_value_hits / max(1, total),
    }


def load_eval(path: str):
    """Supports either a raw list or {items:[...]}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported eval JSON format (expected list or dict with 'items').")

def reset_chroma(db_path="./chroma_db", storage_path="./storage"):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)

    os.makedirs(db_path, exist_ok=True)
    os.makedirs(storage_path, exist_ok=True)

def stratified_sample_by_doc(items, per_doc=10, seed=42):
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for it in items:
        doc = it["evidence"][0]["doc"]
        buckets[doc].append(it)

    sampled = []
    for doc, bucket in buckets.items():
        k = min(per_doc, len(bucket))
        sampled.extend(rng.sample(bucket, k))

    return sampled

def stratified_sample_doc_and_type(items, per_bucket=5, seed=42):
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for it in items:
        doc = os.path.basename(it["evidence"][0]["doc"])
        t = it["gold_answer"]["type"]
        buckets[(doc, t)].append(it)

    sampled = []
    for (_, _), bucket in buckets.items():
        sampled.extend(rng.sample(bucket, min(per_bucket, len(bucket))))

    return sampled

def normalize_number_token(raw: str):
    raw = raw.strip()
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

def is_grounded(answer_text: str) -> bool:
    try:
        obj = json.loads(answer_text)
        quote = (obj.get("evidence_quote") or "").strip()
        return bool(quote) and bool(NUM_RE.search(quote))
    except Exception:
        return False

def parse_predicted_value(answer_text: str):
    """
    Returns (pred_value, pred_unit, found).
    Tries JSON first, then regex fallback.
    """
    if not answer_text:
        return None, None, False

    # Try JSON
    try:
        answer_text = strip_json_fences(answer_text)
        obj = json.loads(answer_text)
        if isinstance(obj, dict):
            found = bool(obj.get("found", obj.get("value") is not None))
            unit = obj.get("unit")
            val = obj.get("value")
            if val is None:
                return None, unit, found
            try:
                return float(val), unit, found
            except (TypeError, ValueError):
                return None, unit, False
    except json.JSONDecodeError:
        pass

    # Fallback: regex first number
    m = NUM_RE.search(answer_text)
    if not m:
        return None, None, False
    v = normalize_number_token(m.group(0))
    if v is None:
        return None, None, False

    # heuristic unit
    txt = answer_text.lower()
    if "%" in answer_text:
        unit = "%"
    elif "€" in answer_text or "eur" in txt:
        unit = "€ million" if "million" in txt else "€"
    else:
        unit = None
    return v, unit, True

if __name__ == "__main__":
    reset_chroma()
    pipeline = RAGPipeline()

    pdfs = [
        # "files/2018-Q3-Financial-Statement-EN.pdf",
        # "files/2018-Q4-Financial-Statement-EN.pdf",
        # "files/2019-Q3-Financial-Statement-EN.pdf",
        # "files/2019-Q4-Financial-Statement-EN.pdf",
        # "files/2020-Q1-Financial-Statement-EN.pdf",
        # "files/2020-Q2-Financial-Statement-EN.pdf",
        # "files/2020-Q3-Financial-Statement-EN.pdf",
        # "files/2020-Q4-Financial-Statement-EN.pdf",
        # "files/2021-Q1-Financial-Statement-EN.pdf",
        # "files/2021-Q2-Financial-Statement-EN.pdf",
        # "files/2021-Q3-Financial-Statement-EN.pdf",
        # "files/2021-Q4-Financial-Statement-EN.pdf",
        # "files/2022-Q1-Financial-Statement-EN.pdf",
        # "files/2022-Q2-Financial-Statement-EN.pdf",
        # "files/2022-Q3-Financial-Statement-EN.pdf",
        # "files/2022-Q4-Financial-Statement-EN.pdf",
        #"files/2023-Q1-Financial-Statement-EN.pdf",
        #"files/2023-Q2-Financial-Statement-EN.pdf",
        #"files/2023-Q3-Financial-Statement-EN.pdf",
        #"files/2023-Q4-Financial-Statement-EN.pdf",
        #"files/2024-Q1-Financial-Statement-EN.pdf",
        # "files/2024-Q2-Financial-Statement-EN.pdf",
        # "files/2024-Q3-Financial-Statement-EN.pdf",
        # "files/2024-Q4-Financial-Statement-EN.pdf",
        "files/2025-Q1-Financial-Statement-EN.pdf",
        "files/2025-Q2-Financial-Statement-EN.pdf",
        "files/2025-Q3-Financial-Statement-EN.pdf"
    ]

    for pdf in pdfs:
        pipeline.ingest_pdf(pdf)

    if hasattr(pipeline, "_refresh_index_after_ingest"):
        pipeline._refresh_index_after_ingest()

    eval_items = load_eval("evaluation/merck_financial_statements_2025_eval.json")

    allowed_docs = {os.path.basename(p) for p in pdfs}
    eval_items = [
        it for it in eval_items
        if os.path.basename(it["evidence"][0]["doc"]) in allowed_docs
    ]

    rec10 = recall_at_k(eval_items, pipeline, k=10, debug_misses=5)
    print(f"Recall@10: {rec10['recall']:.3f}")

    value_sample = stratified_sample_doc_and_type(
        eval_items,
        per_bucket=5
    )

    val5 = eval_rag_numeric(value_sample, pipeline, k=5)
    print(
        f"Value@5 (given hit): {val5['value_accuracy_given_retrieval_hit']:.3f}  "
        f"Grounded@5: {val5['grounded_value_accuracy']:.3f}"
    )