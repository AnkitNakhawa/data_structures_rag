#!/usr/bin/env python3
"""
Query a FAISS RAG index with OpenAI’s ChatGPT models via LangChain.
Requires:
    pip install faiss-cpu langchain openai
"""
import os
import pickle
import argparse

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # for ChatGPT-style models

def load_vectorstore(path: str):
    """Load a pickled FAISS vectorstore from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_qa_chain(vectorstore, top_k: int = 5):
    """
    Create a RetrievalQA chain using ChatGPT (OpenAI ChatOpenAI) 
    and the provided vectorstore.
    """
    # 1) Retriever for top_k chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 2) Use ChatOpenAI to call the Chat Completions API
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-2024-05-13"),  # or "gpt-4"
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 3) Build the RAG chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

def main():
    parser = argparse.ArgumentParser(
        description="Query a RAG index using ChatGPT models."
    )
    parser.add_argument(
        "--index",
        default="ollama_deepseek_index.pkl",
        help="Path to the pickled FAISS vectorstore"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top chunks to retrieve"
    )
    parser.add_argument(
        "query",
        nargs='+',
        help="The question to ask the QA system"
    )
    args = parser.parse_args()
    question = " ".join(args.query)

    if not os.path.exists(args.index):
        print(f"Error: index file '{args.index}' not found.")
        return

    print("Loading vectorstore from disk...")
    vs = load_vectorstore(args.index)

    print(f"Building QA chain (top_k={args.k}) using {os.getenv('OPENAI_MODEL','gpt-3.5-turbo')}...")
    qa = build_qa_chain(vs, top_k=args.k)

    print(f"Asking: {question}\n")
    resp = qa(question)

    print("\n=== Answer ===")
    print(resp["result"])

    print("\n=== Source Documents ===")
    for doc in resp.get("source_documents", []):
        src = doc.metadata.get("source", "<unknown>")
        chunk = doc.metadata.get("chunk", "?")
        print(f"- {src} (chunk {chunk})")

if __name__ == "__main__":
    main()
