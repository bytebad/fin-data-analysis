from dotenv import load_dotenv
import os
import uuid
import logging
import asyncio
import chromadb
from concurrent.futures import ThreadPoolExecutor
import time
import json
import re
import pandas as pd
import numpy as np

from llama_parse import LlamaParse
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SimpleNodeParser


from google import genai

load_dotenv()
logger = logging.getLogger("rag_pipeline")
logging.basicConfig(level=logging.INFO)

# Models
Settings.llm = Gemini(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="models/gemini-flash-lite-latest",
)
Settings.embed_model = GeminiEmbedding(
    model_name="models/text-embedding-004"
)

_executor = ThreadPoolExecutor(max_workers=1)

def run_async(coro):
    """
    Run async code safely in Streamlit by isolating it
    in a dedicated thread + event loop.
    """

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    return _executor.submit(runner).result()


class RAGPipeline:
    def __init__(self, db_path="./chroma_db", storage_path="./storage"):
        self.db_path = db_path
        self.storage_path = storage_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("documents")

    def ingest_pdf(self, pdf_path):
        logger.info("Starting PDF ingestion (isolated from uvloop)")

        def run_ingestion_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                doc_id = str(uuid.uuid4())
                file_name = os.path.basename(pdf_path)

                logger.info("Parsing PDF with LlamaParse")
                parser = LlamaParse(
                    api_key=os.getenv("LLAMA_API_KEY"),
                    result_type="markdown",
                    parsing_instruction="Reconstruct all tables in Markdown.",
                )

                documents = loop.run_until_complete(parser.aload_data(pdf_path))
                logger.info(f"Parsed {len(documents)} documents")

                for doc in documents:
                    doc.metadata = doc.metadata or {}
                    doc.metadata.update({
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_path": pdf_path,
                    })
                    

                # ✅ SAFE parser (NO async, NO LLM, NO asyncio_run)
                from llama_index.core.node_parser import SimpleNodeParser

                logger.info("Processing documents into nodes")
                node_parser = SimpleNodeParser.from_defaults(
                    chunk_size=1024,
                    chunk_overlap=100,
                )

                nodes = node_parser.get_nodes_from_documents(documents)
                logger.info(f"Created {len(nodes)} nodes")

                vector_store = ChromaVectorStore(self.collection)
                storage = StorageContext.from_defaults(vector_store=vector_store)

                VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
                storage.persist(self.storage_path)

                logger.info(f"Ingestion complete for {file_name}")

            finally:
                loop.close()

        _executor.submit(run_ingestion_in_thread).result()



    def list_documents(self):
        data = self.collection.get(include=["metadatas"])
        files = {}
        
        for meta in data.get("metadatas", []):
            if meta and "file_name" in meta:
                files[meta["file_name"]] = meta.get("doc_id", meta["file_name"])

        # return a list of dicts
        print(files)
        logger.info(f"filess {files} ")
        return [{"file_name": k, "doc_id": v} for k, v in files.items()]



    def query_documents(self, query, top_k=5):
        storage = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(self.collection),
            persist_dir=self.storage_path,
        )
        index = load_index_from_storage(storage)
        retriever = index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)

    def generate_answer(self, query, nodes):
        context = "\n\n".join(n.get_content() for n in nodes)
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Context:\n{context}\n\nQuestion: {query}\nAnswer:",
        )
        return response.text.strip()

    def llm_generate_code(self, question: str, df: pd.DataFrame, metadata: dict, last_error: str | None = None) -> str:
        CODE_PROMPT = """
        You are given:
        - question: {question}
        - a pandas DataFrame named df
        - df columns: {columns}
        - metadata: {metadata}

        Task:
        Write ONLY executable Python (pandas) code that computes the answer from df.

        Rules:
        - Do not import anything.
        - Do not read/write files.
        - Do not use network.
        - Use ONLY df, pd, np, re, to_number.
        - Put the final result into a variable named `answer`.
        - If the question cannot be answered from df, set: answer = "NOT_FOUND".
        - Return ONLY code. No markdown. No backticks. No explanation.

        Tips:
        - Values may contain commas, parentheses negatives, %, currencies. Use to_number() when needed.
        """
        prompt = CODE_PROMPT.format(
            question=question,
            columns=[str(c) for c in df.columns],
            metadata=metadata
        )
        if last_error:
            prompt += f"\n\nThe previous code failed with this error:\n{last_error}\nFix the code accordingly.\n"

        # resp = client.models.generate_content(
        #     model=GEMINI_MODEL,
        #     contents=prompt
        # )
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        # Gemini bazen kodu ``` ile sarabiliyor; temizleyelim:
        text = (resp.text or "").strip()
        text = text.replace("```python", "").replace("```", "").strip()
        return text


def save_uploaded_file(uploaded_file, upload_dir="uploads"):
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, os.path.basename(uploaded_file.name))
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# ==========================================
# (CELL 4) Numeric robustness helpers
# ==========================================
def to_number(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "n/a", "nan", "-", "—"}:
        return np.nan

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    for sym in ["€", "$", "£"]:
        s = s.replace(sym, "")
    s = s.replace("%", "").replace(" ", "")

    # Decide separator style by the LAST separator position
    last_comma = s.rfind(",")
    last_dot = s.rfind(".")

    if last_comma != -1 and last_dot != -1:
        # both exist -> choose style by which one appears last
        if last_dot > last_comma:
            # US: 20,992.9 -> 20992.9
            s = s.replace(",", "")
        else:
            # EU: 20.992,9 -> 20992.9
            s = s.replace(".", "").replace(",", ".")
    elif last_comma != -1 and last_dot == -1:
        # only comma exists: could be thousands or decimal
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 3:
            # 12,345 -> 12345
            s = parts[0] + parts[1]
        else:
            # 12,3 -> 12.3
            s = s.replace(",", ".")
    # else: only dot or none -> float can handle

    try:
        val = float(s)
        return -val if neg else val
    except:
        return np.nan


# ==========================================
# (CELL 6) Safe execution + retry
# ==========================================
FORBIDDEN = re.compile(
    r"\b(import|open\(|exec\(|eval\(|__|os\.|sys\.|subprocess|socket|requests|http|pathlib)\b",
    re.IGNORECASE
)

def safe_exec_pandas(code: str, df: pd.DataFrame):
    if FORBIDDEN.search(code):
        raise ValueError("Forbidden operation detected in generated code.")

    local_env = {"df": df, "pd": pd, "np": np, "re": re, "to_number": to_number}

    SAFE_BUILTINS = {
        "str": str, "int": int, "float": float, "bool": bool,
        "len": len, "sum": sum, "min": min, "max": max,
        "range": range, "enumerate": enumerate,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "abs": abs, "round": round,
        "any": any, "all": all,
        "sorted": sorted,
    }
    global_env = {"__builtins__": SAFE_BUILTINS}

    exec(code, global_env, local_env)
    return local_env.get("answer", None), local_env

def guess_used_columns(code: str, columns) -> list:
    used = []
    for c in columns:
        c_str = str(c)
        if f'["{c_str}"]' in code or f"['{c_str}']" in code:
            used.append(c_str)
    return used

def build_preview(df: pd.DataFrame, used_cols: list[str], n=6) -> str:
    try:
        if used_cols:
            sub = df[used_cols].head(n)
        else:
            sub = df.head(n)
        return sub.to_string(index=False)
    except:
        return df.head(n).to_string(index=False)

def answer_with_code(question: str, df: pd.DataFrame, metadata: dict, max_retries: int = 2) -> dict:
    t0 = time.time()
    retries = 0
    last_err = None
    last_code = ""

    for attempt in range(max_retries + 1):
        code = llm_generate_code(question, df, metadata, last_error=last_err)
        last_code = code

        try:
            ans, env = safe_exec_pandas(code, df)
            latency_ms = int((time.time() - t0) * 1000)

            if isinstance(ans, str) and ans.strip() == "NOT_FOUND":
                return {
                    "status": "NOT_FOUND",
                    "final_answer": "NOT_FOUND",
                    "table_id": metadata.get("table_id"),
                    "used_columns": [],
                    "pandas_code": code,
                    "result_preview": "",
                    "retries": retries,
                    "latency_ms": latency_ms
                }

            used_cols = guess_used_columns(code, df.columns)
            preview = build_preview(df, used_cols)

            return {
                "status": "OK",
                "final_answer": str(ans),
                "table_id": metadata.get("table_id"),
                "used_columns": used_cols,
                "pandas_code": code,
                "result_preview": preview,
                "retries": retries,
                "latency_ms": latency_ms
            }

        except Exception as e:
            last_err = repr(e)
            retries += 1

    latency_ms = int((time.time() - t0) * 1000)
    return {
        "status": "ERROR",
        "final_answer": "ERROR",
        "table_id": metadata.get("table_id"),
        "used_columns": [],
        "pandas_code": last_code,
        "result_preview": "",
        "retries": retries,
        "latency_ms": latency_ms,
        "error": last_err
    }

# ==========================================
# (CELL 7) Top-k candidate loop (Pinecone olsa da aynı)
# ==========================================
def is_table_compatible(df: pd.DataFrame) -> bool:
    cols = [str(c).strip().lower() for c in df.columns]
    return ("group" in cols) and ("fy" in cols)

def answer_over_candidates(question: str, candidates: list[dict], max_retries_per_table: int = 1) -> dict:
    for cand in candidates:
        df = cand["df"]
        if not is_table_compatible(df):
            continue  # skip tables without Group & FY for this kind of query

        md = dict(cand.get("metadata", {}))
        md["table_id"] = cand.get("table_id", md.get("table_id"))
        out = answer_with_code(question, df, md, max_retries=max_retries_per_table)
        out["candidate_table_id"] = cand.get("table_id")

        if out["status"] == "OK":
            return out

    return {"status": "NOT_FOUND", "final_answer": "NOT_FOUND"}