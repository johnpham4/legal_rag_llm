from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import traceback
from loguru import logger

from llm_engineering.application.rag.qa import CohereInference
from llm_engineering.application.evaluation.llm_judge import LLMJudge
from llm_engineering.domain.evaluation import JudgmentScore
from llm_engineering.infrastructure.openapi_config import apply_custom_openapi

app = FastAPI(title="Legal Q&A API")

class QueryRequest(BaseModel):
    query: str
    k: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.3, ge=0.0, le=1.0)
    use_sparse: bool = Field(True, description="Enable hybrid search with sparse embeddings")
    expand_to_n_queries: int = Field(3, ge=1, le=5, description="Number of expanded queries")

class EvaluateRequest(BaseModel):
    query: str
    expected_answer: Optional[str | None]
    k: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.3, ge=0.0, le=1.0)
    use_sparse: bool = Field(True, description="Enable hybrid search")
    expand_to_n_queries: int = Field(3, ge=1, le=5, description="Query expansion")

class SourceInfo(BaseModel):
    document_id: str | None
    document_type: str | None
    field: str | None
    document_number: str | None
    content_preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    metadata: dict

qa_service = CohereInference(mock=False)
llm_judge = LLMJudge()

app = apply_custom_openapi(app)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/rag", response_model=QueryResponse)
def rag_endpoint(request: QueryRequest):
    try:
        result = qa_service.execute(
            query=request.query,
            k=request.k,
            temperature=request.temperature,
            use_sparse=request.use_sparse,
            expand_to_n_queries=request.expand_to_n_queries
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceInfo(**src) for src in result["sources"]],
            metadata=result["metadata"]
        )
    except Exception as e:
        logger.error(f"RAG endpoint error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/rag/evaluate", response_model=JudgmentScore)
def rag_evaluate_endpoint(request: EvaluateRequest):
    try:
        result = qa_service.execute(
            query=request.query,
            k=request.k,
            temperature=request.temperature,
            use_sparse=request.use_sparse,
            expand_to_n_queries=request.expand_to_n_queries
        )

        judgment = llm_judge.judge_query(
            query=request.query,
            generated_answer=result["answer"],
            sources=result["sources"],
        )

        logger.info(f"Judgment completed: Overall {judgment.avg_score:.2f}/10")
        return judgment

    except Exception as e:
        logger.error(f"Evaluation endpoint error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
