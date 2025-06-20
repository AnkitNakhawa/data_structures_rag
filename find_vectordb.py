import pickle
import numpy as np

# 1) Reload your saved FAISS vectorstore
with open("ollama_deepseek_index.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# 2) Grab the FAISS index itself
faiss_index = vectorstore.index
print("Total embeddings:", faiss_index.ntotal)

# 3) Locate the index→docstore mapping and the docstore
#    Depending on your langchain version these may be named with or without _
index_to_id = getattr(vectorstore, "index_to_docstore_id", None) \
              or getattr(vectorstore, "_index_to_docstore_id")
docstore   = getattr(vectorstore, "docstore", None) \
              or getattr(vectorstore, "_docstore")

# 4) The docstore holds a dict of {doc_id: Document}
#    InMemoryDocstore stores it under ._dict
docs_dict = getattr(docstore, "_dict", {})

# 5) Reconstruct embeddings and look up their texts
all_texts, all_metadata, all_embs = [], [], []
for i in range(faiss_index.ntotal):
    # 5a) pull the raw vector
    vec = faiss_index.reconstruct(i)        # shape == (dim,)
    all_embs.append(vec)

    # 5b) find which document it came from
    doc_id = index_to_id[i]
    doc: "Document" = docs_dict[doc_id]

    all_texts.append(doc.page_content)
    all_metadata.append(doc.metadata)

# 6) Convert embeddings to a single array if you like
emb_array = np.vstack(all_embs)             # shape (ntotal, dim)

print("Loaded", emb_array.shape[0], "embeddings of dim", emb_array.shape[1])
print("First chunk text:", all_texts[0][:200])
print("First chunk metadata:", all_metadata[0])
