import json
import time
import uuid
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import API_KEY, MODEL_ID, MODEL_NAME
from rag_core import answer_question, ensure_models

app = FastAPI(title="RAG OpenAI-compatible API")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False


def check_auth(authorization: Optional[str] = Header(default=None)):
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.on_event("startup")
def startup_event():
    ensure_models()


@app.get("/v1/models")
def list_models(_=Depends(check_auth)):
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "rag",
                "name": MODEL_NAME,
            }
        ],
    }


def build_content(question: str) -> str:
    answer, sources = answer_question(question)
    if sources:
        answer += "\n\n**Quellen:**\n" + "\n".join(f"- {s}" for s in sources)
    return answer


def build_completion(content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def stream_completion(content: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    delta_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(delta_chunk)}\n\n"

    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, _=Depends(check_auth)):
    question = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

    if not question.strip():
        raise HTTPException(status_code=400, detail="No user message found")

    content = build_content(question)

    if req.stream:
        return StreamingResponse(stream_completion(content), media_type="text/event-stream")

    return build_completion(content)
