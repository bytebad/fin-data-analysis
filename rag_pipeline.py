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
import random
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
from llama_index.core.schema import Document, TextNode


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


# ==========================================
# Table Extraction and Conversion Functions
# ==========================================

def extract_markdown_tables(content: str) -> list[dict]:
    """
    Extract complete markdown tables from content.
    Returns list of dicts with table_text, start_pos, end_pos.
    """
    tables = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if line looks like a table header (contains |)
        if '|' in line and line.count('|') >= 2:
            table_start = i
            table_lines = [line]
            i += 1
            
            # Check for separator line (|---|---|)
            if i < len(lines) and re.match(r'^\s*\|[\s\-\|:]+\|\s*$', lines[i]):
                table_lines.append(lines[i])
                i += 1
            
            # Collect table rows until we hit a non-table line
            while i < len(lines):
                current_line = lines[i].strip()
                # Stop if line doesn't contain | or has too few |
                if '|' not in current_line or current_line.count('|') < 2:
                    # Allow empty lines within table
                    if not current_line:
                        i += 1
                        continue
                    break
                table_lines.append(lines[i])
                i += 1
            
            # Only add if we have at least header + separator + 1 row
            if len(table_lines) >= 3:
                table_text = '\n'.join(table_lines)
                # Calculate character positions in original content
                start_char = sum(len(lines[j]) + 1 for j in range(table_start))
                end_char = sum(len(lines[j]) + 1 for j in range(i))
                tables.append({
                    'table_text': table_text,
                    'start_pos': start_char,
                    'end_pos': end_char,
                })
        else:
            i += 1
    
    return tables


def markdown_table_to_dataframe(table_text: str) -> pd.DataFrame:
    """
    Convert markdown table string to pandas DataFrame.
    Handles various markdown table formats.
    Automatically detects and converts numeric columns for proper aggregations.
    """
    try:
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        if len(lines) < 2:
            return pd.DataFrame()
        
        # Remove separator line (|---|---|)
        lines = [line for line in lines if not re.match(r'^\|[\s\-\|:]+\|\s*$', line)]
        
        if not lines:
            return pd.DataFrame()
        
        # Parse header
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
        
        # Parse rows
        rows = []
        for line in lines[1:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) == len(headers):
                rows.append(cells)
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Auto-detect and convert numeric columns
        # This ensures aggregations work properly
        for col in df.columns:
            # Skip if column name suggests it's not numeric (e.g., "Name", "Description")
            col_lower = str(col).lower()
            if any(skip_word in col_lower for skip_word in ['name', 'description', 'note', 'comment', 'id']):
                continue
            
            # Try to convert column to numeric
            # First, try direct conversion
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            valid_count = (~numeric_series.isna()).sum()
            total_count = len(numeric_series)
            
            # If direct conversion didn't work well, try to_number for formatted numbers
            if valid_count / total_count < 0.5:
                # Most values failed direct conversion, try to_number
                numeric_series = df[col].apply(to_number)
                valid_count = (~numeric_series.isna()).sum()
            
            # If we got at least 50% valid numeric values, convert the column
            if valid_count / total_count >= 0.5:
                df[col] = numeric_series
        
        return df
    
    except Exception as e:
        logger.warning(f"Failed to convert markdown table to DataFrame: {e}")
        return pd.DataFrame()


def create_table_aware_nodes(documents: list[Document], chunk_size: int = 1024, chunk_overlap: int = 100) -> list[TextNode]:
    """
    Create nodes from documents, preserving complete tables as single nodes.
    Tables are converted to DataFrames and stored in metadata.
    """
    from llama_index.core.node_parser import SimpleNodeParser
    
    all_nodes = []
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    for doc in documents:
        content = doc.get_content()
        tables = extract_markdown_tables(content)
        
        if not tables:
            # No tables found, use regular chunking
            nodes = node_parser.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)
            continue
        
        # Process document with tables
        last_pos = 0
        table_counter = 0
        
        for table_info in tables:
            # Add text before table
            if table_info['start_pos'] > last_pos:
                text_before = content[last_pos:table_info['start_pos']].strip()
                if text_before:
                    # Create temporary document for text chunking
                    temp_doc = Document(text=text_before, metadata=doc.metadata.copy())
                    text_nodes = node_parser.get_nodes_from_documents([temp_doc])
                    all_nodes.extend(text_nodes)
            
            # Create table node
            table_text = table_info['table_text']
            table_df = markdown_table_to_dataframe(table_text)
            
            table_id = f"{doc.metadata.get('doc_id', 'unknown')}_table_{table_counter}"
            table_counter += 1
            
            # Create metadata with DataFrame info
            # ChromaDB only accepts str, int, float, None - so we serialize complex types to JSON strings
            table_metadata = doc.metadata.copy()
            table_metadata.update({
                'node_type': 'table',
                'table_id': table_id,
                'table_text': table_text,
                'has_dataframe': 1 if not table_df.empty else 0,  # Convert bool to int
                'table_rows': len(table_df),
            })
            
            # Serialize DataFrame to JSON for storage in metadata (as strings for ChromaDB)
            if not table_df.empty:
                try:
                    # Convert DataFrame to dict for JSON serialization
                    df_dict = table_df.to_dict(orient='records')
                    columns_list = list(table_df.columns)
                    dtypes_dict = {col: str(dtype) for col, dtype in table_df.dtypes.items()}
                    
                    # Serialize to JSON strings for ChromaDB compatibility
                    table_metadata['dataframe_json'] = json.dumps(df_dict)
                    table_metadata['dataframe_columns'] = json.dumps(columns_list)
                    table_metadata['dataframe_dtypes'] = json.dumps(dtypes_dict)
                except Exception as e:
                    logger.warning(f"Failed to serialize DataFrame for table {table_id}: {e}")
            
            # Create table node with full table text
            table_node = TextNode(
                text=table_text,
                metadata=table_metadata,
            )
            all_nodes.append(table_node)
            
            last_pos = table_info['end_pos']
        
        # Add remaining text after last table
        if last_pos < len(content):
            text_after = content[last_pos:].strip()
            if text_after:
                temp_doc = Document(text=text_after, metadata=doc.metadata.copy())
                text_nodes = node_parser.get_nodes_from_documents([temp_doc])
                all_nodes.extend(text_nodes)
    
    return all_nodes


def extract_dataframe_from_node(node: TextNode) -> pd.DataFrame | None:
    """
    Extract DataFrame from node metadata if it's a table node.
    Reconstructs DataFrame and preserves numeric types for proper aggregations.
    Handles JSON string deserialization from ChromaDB metadata.
    """
    metadata = node.metadata or {}
    # Check has_dataframe (stored as int: 1 or 0, or bool for backward compatibility)
    has_dataframe = metadata.get('has_dataframe', 0)
    if isinstance(has_dataframe, bool):
        has_dataframe = 1 if has_dataframe else 0
    if metadata.get('node_type') == 'table' and has_dataframe:
        try:
            # Deserialize JSON strings from ChromaDB metadata
            df_json_str = metadata.get('dataframe_json')
            columns_str = metadata.get('dataframe_columns', '[]')
            dtypes_str = metadata.get('dataframe_dtypes', '{}')
            
            if df_json_str and columns_str:
                # Parse JSON strings back to Python objects
                df_json = json.loads(df_json_str) if isinstance(df_json_str, str) else df_json_str
                columns = json.loads(columns_str) if isinstance(columns_str, str) else columns_str
                dtypes = json.loads(dtypes_str) if isinstance(dtypes_str, str) else dtypes_str
                
                if df_json and columns:
                    df = pd.DataFrame(df_json)
                    
                    # Restore numeric types that may have been lost in JSON serialization
                    for col in df.columns:
                        if col in dtypes:
                            dtype_str = dtypes[col]
                            # Convert numeric types
                            if 'int' in dtype_str or 'float' in dtype_str:
                                # Try direct conversion first
                                numeric_series = pd.to_numeric(df[col], errors='coerce')
                                if not numeric_series.isna().all():
                                    df[col] = numeric_series
                                else:
                                    # If direct conversion fails, use to_number for formatted values
                                    df[col] = df[col].apply(to_number)
                    
                    return df
        except Exception as e:
            logger.warning(f"Failed to reconstruct DataFrame from node metadata: {e}")
    return None


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
                    

                # ✅ Table-aware parser that preserves complete tables
                logger.info("Processing documents into nodes with table preservation")
                nodes = create_table_aware_nodes(
                    documents,
                    chunk_size=1024,
                    chunk_overlap=100
                )
                
                # Count table nodes vs text nodes
                table_nodes = sum(1 for n in nodes if n.metadata.get('node_type') == 'table')
                text_nodes = len(nodes) - table_nodes
                logger.info(f"Created {len(nodes)} nodes ({table_nodes} table nodes, {text_nodes} text nodes)")

                vector_store = ChromaVectorStore(self.collection)
                os.makedirs(self.storage_path, exist_ok=True)

                docstore_path = os.path.join(self.storage_path, "docstore.json")

                if os.path.exists(docstore_path):
                    storage = StorageContext.from_defaults(
                        vector_store=vector_store,
                        persist_dir=self.storage_path,
                    )
                else:
                    storage = StorageContext.from_defaults(vector_store=vector_store)

                try:
                    if os.path.exists(docstore_path):
                        index = load_index_from_storage(storage)
                        index.insert_nodes(nodes)
                    else:
                        raise FileNotFoundError(docstore_path)
                except Exception:
                    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)

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



    def sample_context(self, sample_k: int = 10, max_chars: int = 12000) -> str:
        count = 0
        try:
            count = int(self.collection.count())
        except Exception as e:
            logger.warning(f"Failed to count collection rows: {e}")

        if count <= 0:
            return ""

        k = min(sample_k, count)
        offset = 0
        if count > k:
            offset = random.randint(0, max(0, count - k))

        try:
            data = self.collection.get(include=["documents", "metadatas"], limit=k, offset=offset)
        except TypeError:
            try:
                data = self.collection.get(include=["documents", "metadatas"])
                docs_all = data.get("documents") or []
                metas_all = data.get("metadatas") or []
                data = {
                    "documents": docs_all[offset : offset + k],
                    "metadatas": metas_all[offset : offset + k],
                }
            except Exception as e:
                logger.warning(f"Failed to sample context from Chroma (fallback): {e}")
                return ""
        except Exception as e:
            logger.warning(f"Failed to sample context from Chroma: {e}")
            return ""

        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        parts = []
        for doc, meta in zip(docs, metas):
            if not doc:
                continue
            file_name = ""
            if isinstance(meta, dict):
                file_name = str(meta.get("file_name") or "")
            header = f"[Source: {file_name}]\n" if file_name else ""
            parts.append(header + str(doc).strip())

        context = "\n\n".join(parts).strip()
        if not context:
            return ""
        return context[:max_chars]



    def generate_question_recommendations(self, num_questions: int = 6, sample_k: int = 12) -> list[str]:
        context = self.sample_context(sample_k=sample_k)
        if not context.strip():
            return []

        prompt = (
            "You are generating question recommendations for a user who will ask questions about a financial PDF report.\n"
            "You are given ONLY the following extracted context from the report.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Task: Propose questions that can be answered with 100% certainty using ONLY the context above.\n"
            "Rules:\n"
            "- Do not invent facts not present in the context.\n"
            "- Each question must be answerable directly from the context.\n"
            "- Prefer specific, grounded questions (figures, dates, segments, definitions, table values) over vague ones.\n"
            "- Do not ask for opinions, future predictions, or anything requiring external data.\n"
            f"- Return exactly {num_questions} items as a JSON array of strings.\n"
        )

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = (resp.text or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                out = [str(x).strip() for x in parsed if str(x).strip()]
                return out[:num_questions]
        except Exception:
            pass

        lines = [l.strip(" -\t").strip() for l in text.splitlines() if l.strip()]
        return lines[:num_questions]



    def query_documents(self, query, top_k=5):
        docstore_path = os.path.join(self.storage_path, "docstore.json")
        if os.path.exists(docstore_path):
            storage = StorageContext.from_defaults(
                vector_store=ChromaVectorStore(self.collection),
                persist_dir=self.storage_path,
            )
        else:
            storage = StorageContext.from_defaults(
                vector_store=ChromaVectorStore(self.collection),
            )
        try:
            if not os.path.exists(docstore_path):
                raise FileNotFoundError(docstore_path)
            index = load_index_from_storage(storage)
        except Exception as e:
            logger.warning(f"Failed to load index from storage, rebuilding from vector store: {e}")
            vector_store = ChromaVectorStore(self.collection)
            if hasattr(VectorStoreIndex, "from_vector_store"):
                index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage)
            else:
                index = VectorStoreIndex([], storage_context=storage)
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        if not nodes:
            logger.info("Retriever returned 0 nodes for query")
        
        # Extract DataFrames from table nodes for pandas execution
        for node in nodes:
            df = extract_dataframe_from_node(node)
            if df is not None:
                # Store DataFrame in node metadata for easy access
                if not hasattr(node, '_dataframe'):
                    node._dataframe = df
        
        return nodes
    
    def get_table_candidates_from_nodes(self, nodes: list) -> list[dict]:
        """
        Extract table candidates (with DataFrames) from query nodes.
        Returns list of dicts with 'df', 'metadata', 'table_id', 'node'.
        """
        candidates = []
        for node in nodes:
            df = extract_dataframe_from_node(node)
            if df is not None and not df.empty:
                metadata = node.metadata or {}
                candidates.append({
                    'df': df,
                    'metadata': metadata,
                    'table_id': metadata.get('table_id'),
                    'node': node,
                })
        return candidates

    def generate_answer(self, query, nodes):
        context = "\n\n".join(n.get_content() for n in nodes)
        if not context.strip():
            return "I couldn't find relevant context in the indexed documents to answer that. Try rephrasing with more specific terms (metric, year, segment) or upload the relevant report."
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Context:\n{context}\n\nQuestion: {query}\nAnswer:",
        )
        return response.text.strip()
    
    def query_with_routing(self, query: str, top_k: int = 5, prefer_quantitative: bool = True) -> dict:
        """
        Unified query method that routes to quantitative (pandas) or qualitative (text) handlers.
        
        Args:
            query: The user's question
            top_k: Number of nodes to retrieve
            prefer_quantitative: If True, try quantitative approach first when tables are available
        
        Returns:
            dict with:
            - answer: The final answer text
            - method: 'quantitative' or 'qualitative'
            - details: Additional info (pandas_code, result_preview, etc. for quantitative)
        """
        # Retrieve relevant nodes
        nodes = self.query_documents(query, top_k=top_k)
        
        # Check if we have table nodes with DataFrames
        table_candidates = self.get_table_candidates_from_nodes(nodes)
        
        # Try quantitative approach if tables are available and query seems quantitative
        if table_candidates and prefer_quantitative:
            # Check if query seems to require calculations
            quantitative_keywords = [
                'sum', 'total', 'average', 'mean', 'max', 'min', 'calculate', 
                'compute', 'aggregate', 'compare', 'growth', 'percentage', 
                'ratio', 'difference', 'by', 'group', 'per', 'more than', 
                'less than', 'greater', 'higher', 'lower'
            ]
            query_lower = query.lower()
            is_quantitative_query = any(keyword in query_lower for keyword in quantitative_keywords)
            
            # Also check if query mentions numeric operations
            has_numbers = bool(re.search(r'\d+', query))
            
            if is_quantitative_query or has_numbers or len(table_candidates) > 0:
                # Try quantitative approach
                result = answer_over_candidates(
                    query, 
                    table_candidates, 
                    max_retries_per_table=2,
                    require_compatibility=False
                )
                
                if result.get("status") == "OK":
                    return {
                        "answer": result["final_answer"],
                        "method": "quantitative",
                        "details": {
                            "pandas_code": result.get("pandas_code", ""),
                            "result_preview": result.get("result_preview", ""),
                            "used_columns": result.get("used_columns", []),
                            "table_id": result.get("table_id", ""),
                            "latency_ms": result.get("latency_ms", 0),
                        }
                    }
                elif result.get("status") in ["NOT_FOUND", "ERROR"]:
                    # Quantitative approach didn't find answer or had error, fall back to qualitative
                    if result.get("status") == "ERROR":
                        logger.warning(f"Quantitative approach failed: {result.get('error', 'Unknown error')}, falling back to qualitative")
                    else:
                        logger.info("Quantitative approach returned NOT_FOUND, falling back to qualitative")
        
        # Fall back to qualitative approach (text-based)
        answer = self.generate_answer(query, nodes)
        return {
            "answer": answer,
            "method": "qualitative",
            "details": {}
        }



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

def llm_generate_code(question: str, df: pd.DataFrame, metadata: dict, last_error: str | None = None) -> str:
    """
    Generate pandas code to answer question using DataFrame.
    This is a standalone function (not a method) for use in answer_with_code.
    """
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

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    # Clean up code block markers
    text = (resp.text or "").strip()
    text = text.replace("```python", "").replace("```", "").strip()
    return text


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

def answer_over_candidates(question: str, candidates: list[dict], max_retries_per_table: int = 1, require_compatibility: bool = False) -> dict:
    """
    Try to answer question using table candidates.
    
    Args:
        question: The question to answer
        candidates: List of dicts with 'df', 'metadata', 'table_id'
        max_retries_per_table: Max retries per table
        require_compatibility: If True, only try tables that pass is_table_compatible check
    """
    for cand in candidates:
        df = cand["df"]
        
        if require_compatibility and not is_table_compatible(df):
            continue  # skip tables without Group & FY for this kind of query

        md = dict(cand.get("metadata", {}))
        md["table_id"] = cand.get("table_id", md.get("table_id"))
        out = answer_with_code(question, df, md, max_retries=max_retries_per_table)
        out["candidate_table_id"] = cand.get("table_id")

        if out["status"] == "OK":
            return out

    return {"status": "NOT_FOUND", "final_answer": "NOT_FOUND"}