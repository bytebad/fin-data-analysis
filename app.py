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
    "question_recommendations": [],
    "pending_question": None,
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

    if "question_recommendations" not in st.session_state:
        st.session_state.question_recommendations = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

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

    def run_chat_turn(question: str):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Thinking…"):
            result = st.session_state.rag_pipeline.query_with_routing(question, top_k=10)
            answer = result["answer"]
            method = result["method"]
            details = result.get("details", {})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
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

    if not st.session_state.question_recommendations:
        try:
            st.session_state.question_recommendations = (
                st.session_state.rag_pipeline.generate_question_recommendations(num_questions=3, sample_k=12)
            )
        except Exception as e:
            logger.warning(f"Failed to generate question recommendations: {e}")
            st.session_state.question_recommendations = []

    with st.expander("Recommended questions", expanded=False):
        st.caption("Click a question to send it to chat.")
        for q in st.session_state.question_recommendations:
            if st.button(q, key=f"rec_{hash(q)}"):
                st.session_state.pending_question = q
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.pending_question:
        pending = st.session_state.pending_question
        st.session_state.pending_question = None
        run_chat_turn(pending)

    if prompt := st.chat_input("Ask a question about your uploaded documents"):
        run_chat_turn(prompt)



if __name__ == "__main__":
    main()
