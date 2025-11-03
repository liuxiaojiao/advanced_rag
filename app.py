'''
Advanced RAG System: 
Semantic chunking → sub-query rewrite → multi-retriever fusion (BM25 + vector) → BGE rerank + MMR diversity
Result: maximum recall, maximum precision, minimum hallucination.
'''
from chunking.chunking import process_pdfs_from_local_dir
from chunking.vector_store import create_faiss_index, load_faiss_index
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llm import llm_models 
import os

def main():
    print("\n=== RAG System Interactive Mode ===")
    # user_query = input("\nAsk a question about the PDF: ").strip()
    user_query = "Should I eat supplements?"
    all_chunks = process_pdfs_from_local_dir("input/")

    embedding_save_path = "faiss_store/"

    if os.path.exists(embedding_save_path):
        print("🔁 FAISS index found — loading...")
        vectorstore = load_faiss_index(embedding_model="BAAI/bge-large-en-v1.5", save_path=embedding_save_path)
    else:
        print("🚀 FAISS index not found — creating new one...")
        vectorstore = create_faiss_index(all_chunks, embedding_model="BAAI/bge-large-en-v1.5", save_path=embedding_save_path)

    print("\n🔍 Top results for query:", user_query)
    response = llm_models.generate_response(llm_models.llm_model(), vectorstore, all_chunks, user_query)
    print("\n=== LLM Response ===", response.answer)


if __name__ == "__main__":
    main()