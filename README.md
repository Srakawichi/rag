# Local RAG with Ollama

## Overview
This project demonstrates a simple local Retrieval-Augmented Generation (RAG) system using Ollama, LangChain, and Chroma.

The model answers questions based on the content of provided PDF files.

---

## Tech Stack
- Ollama
- LangChain
- ChromaDB
- Python
- Docker / Docker Compose
- WSL (recommended for Windows)

---

## Quick start (Docker)
```bash
# 1. Clone repo
git clone https://github.com/Srakawichi/rag.git
cd rag

# 2. Add PDFs
# → Put your files into data/

# 3. Build vector DB (pulls models automatically on first run)
docker compose run --rm rag python ingest.py

# 4. Ask questions
docker compose run --rm rag python query.py
```

## Quick start (local)
```bash
# 1. Clone repo
git clone https://github.com/Srakawichi/rag.git
cd rag

# 2. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start Ollama
ollama serve

# 4. Add PDFs
# → Put your files into data/

# 5. Build vector DB
python ingest.py

# 6. Ask questions
python query.py
```

---

## Features
- Fully local (no cloud required)
- Works with Ollama LLMs
- Uses embeddings for semantic search
- PDF-based knowledge base
- Models are pulled automatically on first run (Docker)

---

## Project Structure
```
rag/
├── data/                      # PDFs (not tracked by git)
├── db/                        # Vector database (not tracked by git)
├── ingest.py                  # Build knowledge base
├── query.py                   # Ask questions (CLI)
├── api.py                     # OpenAI-compatible API (for Open WebUI)
├── rag_core.py                # Shared retrieval/rerank/answer logic
├── config.py                  # Configuration
├── Dockerfile
├── docker-compose.yml         # Local/dev stack (bundles its own Ollama)
├── docker-compose.server.yml  # Production deployment (uses existing Ollama)
└── requirements.txt
```

---

## Requirements
- Docker + Docker Compose (for Docker setup)
- Python 3.10+ (for local setup)
- Ollama installed (for local setup)
- NVIDIA GPU + NVIDIA Container Toolkit (required for Docker setup)

---

## Configuration

`config.py`:
```python
DATA_PATH = "data"
DB_PATH = "db"

LLM_MODEL = "mistral"
EMBED_MODEL = "jina/jina-embeddings-v2-base-de:latest"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 5

OLLAMA_BASE_URL = "http://localhost:11434"
```

---

## How It Works
1. PDFs are loaded and split into chunks
2. Chunks are converted into embeddings
3. Stored in a vector database (Chroma)
4. Query → similar chunks retrieved via semantic + keyword search
5. LLM reranks results and generates an answer based on context

---

## Notes
- Works completely offline
- `data/` and `db/` are excluded from git — add your own PDFs after cloning
- Ollama models are stored in `${HOME}/.ollama` on the host — shared between Docker and a local Ollama installation, and persist across container restarts
- Hybrid search: combines semantic embeddings with BM25 keyword search for better retrieval

---

## Use as a model in Open WebUI

`api.py` exposes the RAG pipeline as an OpenAI-compatible API (`/v1/models`,
`/v1/chat/completions`), so it can be registered as a regular connection in
Open WebUI and shows up as a selectable model.

### Deploy on a server that already runs Ollama + Open WebUI

Assumes the `ollama-stack` setup from the infrastructure protocol (Ollama on
port `11434`, Open WebUI on port `3000`, same Ubuntu host).

```bash
# 1. Pull this repo onto the server
git clone https://github.com/Srakawichi/rag.git
cd rag

# 2. Add PDFs
# → put files into data/

# 3. Build (or rebuild) the vector DB — reuses the existing Ollama instance
docker compose -f docker-compose.server.yml run --rm rag-api python ingest.py

# 4. Start the API (stays up, restarts with the host)
docker compose -f docker-compose.server.yml up -d --build
```

`docker-compose.server.yml` does **not** start its own Ollama container — it
runs with `network_mode: host` and talks to the existing Ollama on
`localhost:11434`. This avoids depending on the internal Docker network name
of the already-running `ollama-stack`, so nothing about that stack needs to
be touched or restarted.

Since it uses host networking, port `8000` is bound directly on the server's
network interfaces (same exposure level `ollama` already has on `11434`).
Set an API key before exposing this beyond localhost:

```bash
echo "RAG_API_KEY=$(openssl rand -hex 24)" > .env
docker compose -f docker-compose.server.yml up -d --build
```

### Register in Open WebUI

1. Open `http://<Server-IP>:3000` → **Admin Settings → Connections**
2. Add a connection: **Type:** OpenAI API, **Base URL:** `http://<Server-IP>:8000/v1`, **Key:** the `RAG_API_KEY` value (or any non-empty string if unset)
3. Save — the model **"Wissensdatenbank (RAG)"** now appears in the model selector, next to `llama3.3:70b`, `qwen2.5-coder:32b`, etc.

Rebuilding the knowledge base later (new/updated PDFs) just means re-running
step 3 (`ingest.py`) — the running `rag-api` container reads `db/` fresh on
every request, no restart needed.
