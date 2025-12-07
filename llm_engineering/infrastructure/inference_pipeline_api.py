from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_engineering.application.rag.qa import CohereInference

app = FastAPI(title="Legal Q&A API")

class QueryRequest(BaseModel):
    query: str
    k: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.3, ge=0.0, le=1.0)

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

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/rag", response_model=QueryResponse)
def rag_endpoint(request: QueryRequest):
    try:
        result = qa_service.execute(
            query=request.query,
            k=request.k,
            temperature=request.temperature
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceInfo(**src) for src in result["sources"]],
            metadata=result["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
