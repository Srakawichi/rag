import os

DATA_PATH = "data"
DB_PATH = "db"

LLM_MODEL = "mistral"
EMBED_MODEL = "jina/jina-embeddings-v2-base-de:latest"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K = 5
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI-compatible API (api.py) — used to register this RAG pipeline
# as a selectable model in Open WebUI
MODEL_ID = os.getenv("RAG_MODEL_ID", "rag-knowledgebase")
MODEL_NAME = os.getenv("RAG_MODEL_NAME", "Wissensdatenbank (RAG)")
API_KEY = os.getenv("RAG_API_KEY", "")
