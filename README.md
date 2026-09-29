# Agentic AI Ebook RAG Chatbot

A custom Python RAG API that answers questions from the Agentic AI ebook, returns the retrieved passages, and refuses when retrieval or answer-grounding checks fail. It uses LangGraph, Pinecone, local Hugging Face embeddings, Ollama, and FastAPI. OpenAI is not required.

## Requirements

- Python 3.10 through 3.13; Python 3.11 is recommended and tested
- Pinecone API key and a Pinecone serverless region
- Ollama installed locally

## Setup

From the repository root, create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy the template with `cp .env.example .env`. Set `PINECONE_API_KEY` in `.env`; optionally change the local index name, Pinecone region, Ollama URL, or chat model. Never commit `.env`.

Install Ollama from [ollama.com](https://ollama.com), then download and start the local chat model:

```powershell
ollama pull qwen2.5:3b
```

On Windows, the Ollama app normally starts the local server automatically. If the server is not running, start it in a separate terminal with `ollama serve`.

The embedding model `sentence-transformers/all-MiniLM-L6-v2` downloads from Hugging Face on its first use and runs locally on CPU. Neither model requires a paid inference API. The Pinecone free tier may be used for vector storage; its API key is still required.

## Ingest the ebook

The ingestion command downloads the source PDF to `data/Ebook-Agentic-AI.pdf` if it is not already present, extracts pages, creates 800-character chunks with 100-character overlap, and upserts them with page/source metadata. It creates a separate Pinecone index using 384-dimensional local embeddings when needed. `PINECONE_LOCAL_INDEX_NAME` intentionally avoids reusing an index created for 1536-dimensional OpenAI embeddings.

```powershell
python -m src.ingestion
```

To use a local PDF or another index:

```powershell
python -m src.ingestion --pdf data/Ebook-Agentic-AI.pdf --index agentic-ai-local-index
```

Repeated ingestion uses deterministic chunk IDs, so matching chunks are updated rather than duplicated.

## Run the API

```powershell
uvicorn app:app --reload
```

POST to `http://127.0.0.1:8000/chat` with `{"query":"What is Agentic AI?"}`. Each response has this shape:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["..."],
  "confidence_score": 0.82
}
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Workflow

1. Ingestion loads PDF pages, splits text, and writes embeddings plus source/page metadata to Pinecone.
2. The LangGraph retrieval node requests the four most relevant chunks. Answers are refused when the strongest relevance score is below the configured threshold.
3. The local Ollama model is prompted to use only retrieved text. A separate structured-output grading call checks whether factual claims are supported; one regeneration is allowed, then unsupported answers are refused.
4. The final confidence score is the lower of retrieval relevance and the grader's grounding score. This is a heuristic, not a calibrated probability; prompt instructions and an LLM grader reduce hallucination risk but cannot guarantee perfect grounding.

## Validation

Run offline checks for the retrieval threshold and confidence calculation:

```powershell
python -m unittest discover -s tests -v
```

With the server running and the PDF ingested, run all six assignment sample queries (including the out-of-scope France check):

```powershell
python tests_sample_queries.py
```

The live checks require a running Ollama server with the selected model, a populated Pinecone index, and network access for Pinecone. The first embedding run needs network access to download the Hugging Face model.