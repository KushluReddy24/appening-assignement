from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import INDEX_NAME
from src.graph import build_rag_graph, create_initial_state

app = FastAPI(title="Agentic AI Ebook RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float = Field(ge=0, le=1)


@lru_cache(maxsize=1)
def get_graph():
    return build_rag_graph(INDEX_NAME)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        result = get_graph().invoke(create_initial_state(request.query))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return QueryResponse(
        query=request.query,
        final_answer=result["answer"],
        retrieved_context_chunks=[chunk["text"] for chunk in result["context"]],
        confidence_score=result["confidence_score"],
    )