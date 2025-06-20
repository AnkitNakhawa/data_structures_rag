# pip install faiss-cpu langchain ollama requests tqdm

import os
import glob
import pickle

import numpy as np
import faiss
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

def build_rag_with_ollama_deepseek(text_dirs, output_index_path, batch_size=32):
    """
    Uses Ollama-hosted Deepseek-R1 for embeddings (with progress bar)
    and Ollama Llama2 for QA.
    """
    # 1) Read all .txt files
    docs = []
    for d in text_dirs:
        for path in glob.glob(os.path.join(d, "*.txt")):
            with open(path, 'r', encoding='utf-8') as f:
                docs.append({
                    "page_content": f.read(),
                    "metadata": {"source": path}
                })

    # 2) Chunk each document into ~1,000-char pieces with 50-char overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks, metadatas = [], []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d["page_content"])):
            chunks.append(piece)
            metadatas.append({**d["metadata"], "chunk": i})

    # 3) Embed with Deepseek-R1 via Ollama, in batches with tqdm progress
    embeddings = OllamaEmbeddings(model="deepseek-r1")
    vectors = []
    pbar = tqdm(total=len(chunks), desc="Embedding chunks", unit="chunk")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_vecs = embeddings.embed_documents(batch)
        vectors.extend(batch_vecs)
        pbar.update(len(batch))
    pbar.close()

    # 4) Build a FAISS index using `from_texts` (it will re-embed under the hood)
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas
    )

    # 5) Persist the index to disk
    with open(output_index_path, "wb") as f:
        pickle.dump(vectorstore, f)

    # 6) Create a RetrievalQA chain using Llama2 via Ollama
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(
        model="llama2",    # or your local Ollama model name
        temperature=0.0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

if __name__ == "__main__":
    text_dirs = ["output_chapters"]
    qa = build_rag_with_ollama_deepseek(text_dirs, "ollama_deepseek_index.pkl")

    # Example query
    question = "What’s the average-case runtime of quicksort?"
    resp = qa(question)

    print("Answer:", resp["result"])
    print("\nSource chunks:")
    for doc in resp["source_documents"]:
        print(f"- {doc.metadata['source']} (chunk {doc.metadata['chunk']})")
