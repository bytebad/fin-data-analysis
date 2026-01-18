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

    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None

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

            # Auto-select latest document
            st.session_state.selected_doc = st.session_state.documents[-1]
            st.session_state.ingestion_complete = True

            st.success("Document indexed successfully!")

        # Document list (persistent)
        if st.session_state.documents:
            st.subheader("Previously Uploaded")
            for doc in st.session_state.documents:
                if st.button(f"📄 {doc['file_name']}", key=doc["file_name"]):
                    st.session_state.selected_doc = doc
                    st.session_state.messages = []

        else:
            st.info("No documents indexed yet.")

    # -----------------------------
    # Main Panel
    # -----------------------------
    if not st.session_state.selected_doc:
        st.info("Upload or select a document to begin.")
        return

    st.caption(f"Selected document: **{st.session_state.selected_doc['file_name']}**")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about this document"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.spinner("Thinking…"):
            nodes = st.session_state.rag_pipeline.query_documents(prompt)
            answer = st.session_state.rag_pipeline.generate_answer(prompt, nodes)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)



if __name__ == "__main__":
    main()
