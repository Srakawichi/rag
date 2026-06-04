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

# 3. Start Ollama + pull models automatically
docker compose up model-init

# 4. Build vector DB
docker compose run --rm rag python ingest.py

# 5. Ask questions
docker compose run --rm -it rag python query.py
```

## Quick start (local)
```bash
# 1. Start Ollama
ollama serve

# 2. Install models
ollama pull mistral
ollama pull jina/jina-embeddings-v2-base-de:latest

# 3. Setup project
git clone https://github.com/Srakawichi/rag.git
cd rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

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
- Simple and extendable architecture

---

## Project Structure
```bash
rag/
│
├── data/                 # PDFs
├── db/                   # Vector database (Chroma)
├── ingest.py             # Build knowledge base
├── query.py              # Ask questions
├── config.py             # Configurations
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
```

---

## Requirements
- Docker + Docker Compose (for Docker setup)
- Python 3.10+ (for local setup)
- Ollama installed (for local setup)
- GPU (optional but recommended)

Check GPU:

nvidia-smi


---

## Installation

### 1. Start Ollama

ollama serve


(Optional with GPU)

OLLAMA_NUM_GPU=1 ollama serve


---

### 2. Setup Environment
git clone https://github.com/Srakawichi/rag.git
cd rag
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt


---

### 3. Install Models

ollama pull mistral
ollama pull jina/jina-embeddings-v2-base-de:latest


---

## Configuration

Example `config.py`:

DATA_PATH = "data"
DB_PATH = "db"

LLM_MODEL = "llama3"
EMBED_MODEL = "jina/jina-embeddings-v2-base-de:latest"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

OLLAMA_BASE_URL = "http://localhost:11434"


---

## Usage

### 1. Add PDFs
Place your PDF files in:

rag/data


---

### 2. Build Knowledge Base

python ingest.py


---

### 3. Ask Questions

python query.py


---

## How It Works
1. PDFs are loaded and split into chunks  
2. Chunks are converted into embeddings  
3. Stored in a vector database (Chroma)  
4. Query → similar chunks retrieved  
5. LLM generates answer based on context  

---

## Improvements (Optional)
- Increase `TOP_K` for better recall
- Adjust `CHUNK_SIZE` for better context
- Use better models (e.g. mistral)
- Add reranking for higher accuracy

---
## Known Issues
- Answers can be too generic → adjust prompt
- Embeddings may return irrelevant chunks
- PDF encoding issues (Unicode errors possible)

---

## Future Work
- Reranking (LLM-based filtering)
- Hybrid search (Embeddings + Keywords)
- Better prompt engineering
- UI integration (OpenWebUI)

---

## Notes
- Works completely offline
- Best performance with GPU
- Designed as a simple MVP
