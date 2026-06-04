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
├── data/                 # PDFs (not tracked by git)
├── db/                   # Vector database (not tracked by git)
├── ingest.py             # Build knowledge base
├── query.py              # Ask questions
├── config.py             # Configuration
├── Dockerfile
├── docker-compose.yml
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
- Best performance with GPU
- `data/` and `db/` are excluded from git — add your own PDFs after cloning
