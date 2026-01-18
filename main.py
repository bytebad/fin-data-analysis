from rag_pipeline import RAGPipeline

rag = RAGPipeline()

rag.ingest_pdf("./files/merck.pdf")

query = "Compare Net Sales growth between Life Science and Electronics sectors."
nodes = rag.query_documents(query)
answer = rag.generate_answer(query, nodes)

print(answer)
