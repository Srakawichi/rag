import ollama
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi

from config import DB_PATH, EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL


def ensure_models():
    client = ollama.Client(host=OLLAMA_BASE_URL)
    for model in [LLM_MODEL, EMBED_MODEL]:
        print(f"Pulling {model}...")
        client.pull(model)
        print(f"  {model} ready.")


def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8")


def keyword_search(docs, query):
    tokenized_docs = [clean_text(doc.page_content).lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored[:5]]


def rerank_chunks(llm, docs, question):
    scored_docs = []

    for doc in docs:
        prompt = f"""
        Rate the relevance of the following context for answering the question.

        Context:
        {clean_text(doc.page_content)}

        Question:
        {question}

        Give a score from 1 to 10. Only return the number.
        """

        try:
            score = llm.invoke(prompt).strip()
            score = int(score)
        except Exception:
            score = 0

        scored_docs.append((doc, score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:3]]


PROMPT_TEMPLATE = """
You are a technical assistant.

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


def answer_question(question: str):
    enhanced_query = clean_text(f"""
    Find detailed technical instructions and steps.
    Focus on commands and setup.

    Question: {question}
    """)

    embedding_function = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function
    )

    embedding_results = db.similarity_search(enhanced_query, k=10)

    keyword_results = keyword_search(embedding_results, question)

    combined = list({id(doc): doc for doc in (embedding_results + keyword_results)}.values())

    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    results = rerank_chunks(llm, combined, question)

    context_text = ""
    for i, doc in enumerate(results):
        context_text += f"[Source {i+1}]\n{clean_text(doc.page_content)}\n\n"

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=question
    )

    response = llm.invoke(prompt)
    sources = [str(doc.metadata) for doc in results]

    return response, sources
