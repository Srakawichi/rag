from typing import List, Union, Generator, Iterator
import os

DB_PATH = "/app/db"
EMBED_MODEL = "jina/jina-embeddings-v2-base-de:latest"
LLM_MODEL = "mistral"
TOP_K = 5
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

PROMPT_TEMPLATE = """You are a technical assistant.

Answer using ONLY the provided context.
Be precise and detailed.

If the question asks for steps:
- give a clear step-by-step guide
- include commands if available

Context:
{context}

Question:
{question}

Answer:
"""


class Pipeline:
    def __init__(self):
        self.name = "Studiengangsbeschreibung"
        self.db = None
        self.llm = None

    async def on_startup(self):
        from langchain_ollama import OllamaEmbeddings, OllamaLLM
        from langchain_chroma import Chroma

        self.db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=OllamaEmbeddings(
                model=EMBED_MODEL,
                base_url=OLLAMA_BASE_URL,
            ),
        )
        self.llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    async def on_shutdown(self):
        pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        from rank_bm25 import BM25Okapi
        from langchain_core.prompts import PromptTemplate

        enhanced_query = (
            f"Find detailed technical instructions and steps. "
            f"Focus on commands and setup. Question: {user_message}"
        )

        semantic_results = self.db.similarity_search(enhanced_query, k=10)

        tokenized = [doc.page_content.lower().split() for doc in semantic_results]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(user_message.lower().split())
        keyword_results = [
            doc
            for doc, _ in sorted(zip(semantic_results, scores), key=lambda x: x[1], reverse=True)[:5]
        ]

        combined = list({id(doc): doc for doc in (semantic_results + keyword_results)}.values())

        scored = []
        for doc in combined:
            rerank_prompt = (
                f"Rate the relevance of the following context for answering the question.\n\n"
                f"Context:\n{doc.page_content}\n\nQuestion:\n{user_message}\n\n"
                f"Give a score from 1 to 10. Only return the number."
            )
            try:
                score = int(self.llm.invoke(rerank_prompt).strip())
            except Exception:
                score = 0
            scored.append((doc, score))

        top_docs = [doc for doc, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:3]]

        context = "".join(
            f"[Source {i+1}]\n{doc.page_content}\n\n" for i, doc in enumerate(top_docs)
        )

        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context,
            question=user_message,
        )

        return self.llm.invoke(prompt)
