import streamlit as st
import os
from rag_pipeline import RAGPipeline, save_uploaded_file
import logging

st.set_page_config(
    page_title="Financial Reports Q&A",
    page_icon="📚",
    layout="wide",
)

# Session state - initialize BEFORE any handlers
for key, default in {
    "rag_pipeline": None,
    "messages": [],
    "current_file": None,
    "ingestion_complete": False,
    "logs": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Logging to Streamlit
class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        # Safely append to logs, ensuring it exists
        if "logs" not in st.session_state:
            st.session_state.logs = []
        st.session_state.logs.append(self.format(record))

logger = logging.getLogger("rag_pipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = StreamlitLogHandler()
handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(handler)


def main():
    st.title("📚 Financial Reports Q&A")

    # -----------------------------
    # Session State Initialization
    # -----------------------------
    if (
        "rag_pipeline" not in st.session_state
        or st.session_state.rag_pipeline is None
    ):
        st.session_state.rag_pipeline = RAGPipeline()


    if "documents" not in st.session_state:
        st.session_state.documents = st.session_state.rag_pipeline.list_documents()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "ingestion_complete" not in st.session_state:
        st.session_state.ingestion_complete = False

    if "current_file" not in st.session_state:
        st.session_state.current_file = None

    # -----------------------------
    # Sidebar
    # -----------------------------
    with st.sidebar:
        st.header("📄 Documents")

        # Upload
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded_file and uploaded_file.name != st.session_state.current_file:
            st.session_state.current_file = uploaded_file.name
            st.session_state.ingestion_complete = False
            st.session_state.messages = []

            file_path = save_uploaded_file(uploaded_file)

            with st.spinner("Parsing and indexing PDF…"):
                st.session_state.rag_pipeline.ingest_pdf(file_path)

            os.remove(file_path)

            # Refresh document list from persistent store
            st.session_state.documents = st.session_state.rag_pipeline.list_documents()
            st.session_state.ingestion_complete = True

            st.success("Document indexed successfully!")

        # Document list (persistent)
        if st.session_state.documents:
            st.subheader("Previously Uploaded")
            for doc in st.session_state.documents:
                st.text(f"📄 {doc['file_name']}")

        else:
            st.info("No documents indexed yet.")

    # -----------------------------
    # Main Panel
    # -----------------------------
    if not st.session_state.documents:
        st.info("Upload a document to begin.")
        return

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your uploaded documents"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.spinner("Thinking…"):
            # Use unified query routing that handles both quantitative and qualitative queries
            result = st.session_state.rag_pipeline.query_with_routing(prompt, top_k=10)
            
            answer = result["answer"]
            method = result["method"]
            details = result.get("details", {})

        # Display answer
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # Show additional details for quantitative queries
            if method == "quantitative" and details:
                with st.expander("📊 Calculation Details", expanded=False):
                    if details.get("pandas_code"):
                        st.code(details["pandas_code"], language="python")
                    
                    if details.get("result_preview"):
                        st.text("Result Preview:")
                        st.text(details["result_preview"])
                    
                    if details.get("used_columns"):
                        st.caption(f"Columns used: {', '.join(details['used_columns'])}")
                    
                    if details.get("latency_ms"):
                        st.caption(f"Response time: {details['latency_ms']}ms")



if __name__ == "__main__":
    main()
