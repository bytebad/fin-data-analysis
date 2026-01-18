from dotenv import load_dotenv
import os
import uuid
import logging
import asyncio
import chromadb
from concurrent.futures import ThreadPoolExecutor

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


def save_uploaded_file(uploaded_file, upload_dir="uploads"):
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, os.path.basename(uploaded_file.name))
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
