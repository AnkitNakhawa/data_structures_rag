#!/usr/bin/env python3
"""
Query a FAISS RAG index with Anthropic Claude via the chat‐completions “Messages API”.
Requires:
    pip install faiss-cpu langchain-anthropic
"""
import os
import pickle
import argparse

from langchain.chains import RetrievalQA
from langchain_anthropic.chat_models import ChatAnthropic  # use the chat‐completion client

def load_vectorstore(path: str):
    """Load a pickled FAISS vectorstore from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_qa_chain(vectorstore, top_k: int = 5):
    """
    Create a RetrievalQA chain using Anthropic Claude’s Messages API
    (via ChatAnthropic) and the provided vectorstore.
    """
    # 1) Retriever for top_k chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 2) Use the ChatAnthropic wrapper to call /v1/chat/completions
    llm = ChatAnthropic(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-haiku-20241022",  # exact ID from list_models
        temperature=0.0
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
        description="Query a RAG index using Anthropic Claude (Chat API)."
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

    print(f"Building QA chain (top_k={args.k})...")
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
