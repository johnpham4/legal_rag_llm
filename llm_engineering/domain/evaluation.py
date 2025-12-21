from dataclasses import dataclass
from pydantic import BaseModel, Field

@dataclass
class EvaluationResult:
    query: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    k: int

    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float

    true_positives: int
    false_positives: int
    false_negatives: int

    def __str__(self) -> str:
        return f"""
            Query: {self.query[:60]}...
            Relevant docs: {len(self.relevant_doc_ids)}
            Retrieved docs: {len(self.retrieved_doc_ids)}

            Metrics:
            Precision@{self.k}: {self.precision:.3f}
            Recall@{self.k}:    {self.recall:.3f}
            MRR:            {self.reciprocal_rank:.3f}
            NDCG@{self.k}:      {self.ndcg:.3f}

            Details:
            True Positives:  {self.true_positives}
            False Positives: {self.false_positives}
            False Negatives: {self.false_negatives}
        """


class JudgmentScore(BaseModel):
    factual_accuracy: int = Field(..., ge=0, le=10, description="Độ chính xác nội dung (0-10)")
    completeness: int = Field(..., ge=0, le=10, description="Độ đầy đủ thông tin (0-10)")
    legal_correctness: int = Field(..., ge=0, le=10, description="Tính chính xác về mặt pháp lý (0-10)")
    hallucination: bool = Field(..., description="Có thông tin sai lệch/bịa đặt không?")
    reasoning: str = Field(..., description="Lý do chấm điểm")

    @property
    def avg_score(self) -> float:
        return (self.factual_accuracy + self.completeness + self.legal_correctness) / 3.0