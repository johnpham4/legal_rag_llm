from typing import Dict, List
from llm_engineering.domain.evaluation import JudgmentScore
from loguru import logger
import json

from llm_engineering.infrastructure.llm.cohere_client import CohereLLMClient


class LLMJudge:
    """LLM as judge để đánh giá chất lượng RAG answers."""

    JUDGE_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ của bạn là đánh giá chất lượng câu trả lời RAG dựa trên tài liệu nguồn.

**CÂU HỎI:**
{query}

**TÀI LIỆU NGUỒN (Retrieved Chunks):**
{sources}

**CÂU TRẢ LỜI CỦA HỆ THỐNG:**
{generated_answer}

---

Đánh giá câu trả lời theo 4 tiêu chí sau (chỉ trả về JSON):

1. **factual_accuracy** (0-10): Độ chính xác so với thông tin trong TÀI LIỆU NGUỒN
   - 10: Thông tin hoàn toàn khớp với tài liệu nguồn, không sai lệch
   - 7-9: Chính xác phần lớn, có thể diễn đạt khác nhưng nghĩa không đổi
   - 4-6: Có một số thông tin không khớp với nguồn
   - 0-3: Sai lệch nghiêm trọng so với tài liệu nguồn

2. **completeness** (0-10): Độ đầy đủ - Có khai thác hết thông tin liên quan từ TÀI LIỆU NGUỒN không?
   - 10: Tổng hợp đầy đủ tất cả thông tin liên quan từ các chunks
   - 7-9: Bao phủ hầu hết thông tin chính, bỏ sót ít
   - 4-6: Chỉ dùng một phần thông tin từ nguồn
   - 0-3: Thiếu hầu hết thông tin có trong nguồn

3. **legal_correctness** (0-10): Tính đúng đắn về mặt pháp lý dựa trên TÀI LIỆU NGUỒN
   - 10: Diễn giải đúng nguyên tắc pháp luật từ tài liệu
   - 7-9: Đúng nhưng có thể diễn đạt không chuẩn xác
   - 4-6: Có sai sót về thuật ngữ/diễn giải
   - 0-3: Vi phạm hoặc hiểu sai nguyên tắc pháp luật

4. **hallucination** (true/false): Có thông tin BỊA ĐẶT/KHÔNG CÓ trong TÀI LIỆU NGUỒN không?
   - false: Tất cả thông tin đều có trong tài liệu nguồn
   - true: Có thông tin không tồn tại trong tài liệu nguồn (bịa số liệu, chính sách không có, etc.)

5. **reasoning**: Giải thích ngắn gọn tại sao chấm điểm như vậy, đặc biệt chỉ ra hallucination nếu có (1-3 câu)

**QUAN TRỌNG:**
- Chỉ đánh giá dựa trên TÀI LIỆU NGUỒN được cung cấp
- Không so sánh với kiến thức bên ngoài
- Nếu câu trả lời tổng hợp từ nhiều chunks → điểm completeness cao
- Nếu có thông tin không có trong nguồn → hallucination = true

**CHỈ TRẢ VỀ JSON:**
{{
    "factual_accuracy": <0-10>,
    "completeness": <0-10>,
    "legal_correctness": <0-10>,
    "hallucination": <true/false>,
    "reasoning": "<giải thích>"
}}
"""

    def __init__(self):
        self.llm = CohereLLMClient()

    def judge_query(
        self,
        query: str,
        generated_answer: str,
        sources: List[Dict],
        expected_answer: str | None = None
    ) -> JudgmentScore:

        sources_text = "\n\n".join([
            f"**Chunk {i+1}:** [{src.get('document_number', 'N/A')}]\n{src.get('content_preview', '')[:500]}"
            for i, src in enumerate(sources[:3])
        ])

        prompt = self.JUDGE_PROMPT.format(
            query=query,
            generated_answer=generated_answer,
            sources=sources_text
        )

        try:

            response = self.llm.generate(prompt, temperature=0.0)

            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                judgment_data = json.loads(json_str)
                return JudgmentScore(**judgment_data)
            else:
                logger.error(f"Cannot parse JSON from LLM response: {response}")
                raise ValueError("Invalid JSON format")

        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            raise


    @staticmethod
    def aggregate_scores(judgments: List[Dict]) -> Dict:
        """Tính trung bình các metrics."""
        if not judgments:
            return {}

        factual_scores = [j['scores']['factual_accuracy'] for j in judgments]
        completeness_scores = [j['scores']['completeness'] for j in judgments]
        legal_scores = [j['scores']['legal_correctness'] for j in judgments]
        hallucination_count = sum(1 for j in judgments if j['scores']['hallucination'])

        return {
            'avg_factual_accuracy': sum(factual_scores) / len(factual_scores),
            'avg_completeness': sum(completeness_scores) / len(completeness_scores),
            'avg_legal_correctness': sum(legal_scores) / len(legal_scores),
            'overall_score': sum(j['avg_score'] for j in judgments) / len(judgments),
            'hallucination_rate': hallucination_count / len(judgments),
            'total_cases': len(judgments)
        }
