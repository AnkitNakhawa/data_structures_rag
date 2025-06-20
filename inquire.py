#!/usr/bin/env python3
"""
Script to load an existing FAISS vectorstore and prompt Llama2 via Ollama in a RAG pipeline.
"""
import os
import pickle
import argparse

from langchain.chains import RetrievalQA
from langchain.llms import Ollama


def load_vectorstore(path: str):
    """
    Load a pickled FAISS vectorstore from disk.
    """
    with open(path, 'rb') as f:
        vs = pickle.load(f)
    return vs


def build_qa_chain(vectorstore, top_k: int = 5):
    """
    Create a RetrievalQA chain with Ollama Llama2 using the provided vectorstore.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = Ollama(
        model="deepseek-r1",       # change to your local Ollama model
        temperature=0.0
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def main():
    parser = argparse.ArgumentParser(
        description="Query a RAG index using Ollama-powered Llama2."
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
    vectorstore = load_vectorstore(args.index)

    print(f"Building QA chain (top_k={args.k})...")
    qa_chain = build_qa_chain(vectorstore, top_k=args.k)

    print(f"Asking: {question}\n")
    result = qa_chain(question)

    print("\n=== Answer ===")
    print(result["result"])

    print("\n=== Source Documents ===")
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "<unknown>")
        chunk = doc.metadata.get("chunk", "?")
        print(f"- {src} (chunk {chunk})")


if __name__ == '__main__':
    main()
