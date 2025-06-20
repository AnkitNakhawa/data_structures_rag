# Data Structures RAG Demo

This repository contains a small Retrieval-Augmented Generation (RAG) prototype built with Python and [LangChain](https://github.com/langchain-ai/langchain).  It converts PDF textbooks into text, splits the text into manageable chunks, embeds those chunks with a local Ollama model, stores them in a FAISS vector database, and then allows querying the data with different language models.

The sample textbook used in this demo is **Data Structures and Algorithm Analysis in Java (3rd Edition)** by **Mark Allen Weiss**. Only excerpts of the book are included for demonstration purposes.

The project does **not** contain any API keys or secrets.  All API calls expect keys to be provided via environment variables such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

## Directory overview

- `pdf_textbooks/` – source PDFs used to build the knowledge base
- `output/` and `output_chapters/` – extracted text
- `split_chunks.py` – builds the FAISS index with Ollama embeddings
- `inquire.py` – query the index using Llama 2 via Ollama
- `inquire_gpt.py` – query the index using OpenAI models
- `inquire_claude.py` – query the index using Anthropic Claude
- `find_vectordb.py` – inspect an existing FAISS vectorstore
- `get_text.py` – utilities to extract and clean PDF text

## Example usage

1. **Extract text**

```bash
python get_text.py
```

2. **Build the vector index**

```bash
python split_chunks.py
```

This creates `ollama_deepseek_index.pkl` which holds the FAISS vectorstore and embeddings.

3. **Ask questions**

With a local Ollama model:

```bash
python inquire.py --k 3 "What is the average-case runtime of quicksort?"
```

Using OpenAI (requires `OPENAI_API_KEY` environment variable):

```bash
export OPENAI_API_KEY=your-token-here
python inquire_gpt.py "Explain the difference between a stack and a queue"
```

Using Anthropic Claude (requires `ANTHROPIC_API_KEY`):

```bash
export ANTHROPIC_API_KEY=your-token-here
python inquire_claude.py "How does hashing work?"
```

All scripts print the answer as well as the file and chunk numbers of the retrieved source documents.

## Model listing helpers

`list_models.py` and `list_models_http.py` show how to list available Claude models via the SDK or direct HTTP.

---

The repository ships only sample PDFs and code. It does not contain any API tokens.
