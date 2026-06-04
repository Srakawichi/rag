import os
import shutil

import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import DATA_PATH, DB_PATH, EMBED_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_BASE_URL


def ensure_models():
    client = ollama.Client(host=OLLAMA_BASE_URL)
    for model in [LLM_MODEL, EMBED_MODEL]:
        print(f"Pulling {model}...")
        client.pull(model)
        print(f"  {model} ready.")


def load_documents():
    documents = []

    for filename in os.listdir(DATA_PATH):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

    return documents


def main():
    ensure_models()
    print("Loading PDFs...")
    docs = load_documents()

    if not docs:
        print("No PDFs found.")
        return

    print(f"Loaded pages: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)
    print(f"Chunks: {len(chunks)}")

    if os.path.exists(DB_PATH):
        for item in os.listdir(DB_PATH):
            item_path = os.path.join(DB_PATH, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    print("Creating vector DB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("Done.")


if __name__ == "__main__":
    main()
