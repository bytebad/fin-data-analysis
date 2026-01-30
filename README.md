# Document Q&A Assistant

A Streamlit-based RAG (Retrieval-Augmented Generation) application that allows you to upload PDF documents and ask questions about their content using AI.

## Features

- 📄 **PDF Upload**: Upload any PDF document for analysis
- 🤖 **AI-Powered Q&A**: Ask questions about your documents using Google's Gemini AI
- 💾 **Vector Storage**: Documents are processed and stored in ChromaDB for efficient retrieval
- 💬 **Chat Interface**: Interactive chat interface with conversation history
- ⚡ **Real-time Processing**: See progress indicators during document ingestion
- 📋 **Detailed Logging**: View comprehensive logs during document processing
- 🔄 **Session Management**: Start new conversations with different documents

## Architecture

The application consists of:

- **rag_pipeline.py**: Core RAG functionality including document ingestion and querying
- **app.py**: Streamlit web interface
- **main.py**: Legacy command-line interface (backward compatible)
- **run_app.py**: Convenience script to start the Streamlit app

## Prerequisites

- Python 3.8+
- API Keys for:
  - Google Gemini AI (`GEMINI_API_KEY`)
  - LlamaCloud/LlamaParse (`LLAMA_API_KEY`)

## Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   LLAMA_API_KEY=your_llama_api_key_here
   ```

## Usage

### Running the Streamlit App

**Option 1: Using the convenience script (Recommended)**
```bash
python run_app.py
```

**Option 2: Direct Streamlit command**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Running the CLI Version (Legacy)

The original command-line interface is still available:

```bash
# Process a PDF and store in database
python main.py --ingest 1

# Query without reprocessing
python main.py
```

### Using the Command Line (Legacy)

You can still use the original command-line interface:

```bash
# Process a PDF and store in database
python main.py --ingest 1

# Query without reprocessing
python main.py
```

## How It Works

1. **Document Upload**: Users upload PDF files through the Streamlit interface
2. **Document Processing**:
   - PDFs are parsed using LlamaParse
   - Content is split into chunks and processed with MarkdownElementNodeParser
   - Embeddings are created using Gemini Embedding model
   - Everything is stored in ChromaDB vector database
3. **Question Answering**:
   - User questions are converted to embeddings
   - Similar content is retrieved from the vector database
   - Retrieved context is sent to Gemini AI for answer generation
4. **Chat Interface**: Responses are displayed in a conversational format

## File Storage

- **Uploaded PDFs**: Temporarily stored in the `uploads/` directory during processing
- **Vector Database**: Persistent storage in `chroma_db/` directory
- **Index Storage**: Additional metadata stored in `storage/` directory

## API Keys Setup

### Google Gemini AI
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create an API key
3. Add it to your `.env` file as `GEMINI_API_KEY`

### LlamaParse
1. Go to [LlamaCloud](https://cloud.llamaindex.ai/)
2. Sign up and get your API key
3. Add it to your `.env` file as `LLAMA_API_KEY`

## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`

2. **API Key errors**: Verify your API keys are correctly set in the `.env` file

3. **Event loop errors (asyncio/nest_asyncio)**: The app is designed to handle asyncio properly. If you encounter event loop errors, make sure you're running it through the provided `run_app.py` script.

4. **Document processing fails**: Check that your PDF is not corrupted and contains readable text

5. **Memory issues**: Large PDFs may require more RAM. Consider splitting large documents.

### Performance Tips

- **Smaller PDFs**: Process faster and use less memory
- **Clear questions**: More specific questions yield better results
- **Relevant content**: The AI works best when the document contains information relevant to your questions

## Development

The codebase is structured as follows:

- `rag_pipeline.py`: Core RAG pipeline logic
- `app.py`: Streamlit web application
- `main.py`: Legacy CLI interface
- `requirements.txt`: Python dependencies

## License

This project is for educational and demonstration purposes.