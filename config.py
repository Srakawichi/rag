import os

DATA_PATH = "data"
DB_PATH = "db"

LLM_MODEL = "mistral"
EMBED_MODEL = "jina/jina-embeddings-v2-base-de:latest"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K = 5
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
