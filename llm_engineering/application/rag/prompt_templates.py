from langchain_core.prompts import PromptTemplate
from llm_engineering.application.rag.base import PromptTemplateFactory

class QueryExpansionTemplate(PromptTemplateFactory):
    prompt: str = """Bạn là trợ lý AI hỗ trợ tìm kiếm văn bản pháp luật. Nhiệm vụ của bạn là tạo ra {expand_to_n}
    phiên bản khác nhau của câu hỏi người dùng để truy xuất các văn bản liên quan từ cơ sở dữ liệu vector.
    Bằng cách tạo nhiều góc nhìn khác nhau về câu hỏi, bạn giúp người dùng vượt qua những hạn chế
    của tìm kiếm dựa trên độ tương đồng khoảng cách.
    Hãy cung cấp các câu hỏi thay thế được phân tách bởi '{separator}'.
    Câu hỏi gốc: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question"

    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "expand_to_n": expand_to_n,
            },
        )

class SelfQueryTemplate(PromptTemplateFactory):
    """Extract structured metadata from Vietnamese legal queries for better filtering."""

    prompt: str = """Bạn là trợ lý AI chuyên về pháp luật Việt Nam. Nhiệm vụ của bạn là trích xuất thông tin có cấu trúc từ câu hỏi của người dùng để tìm kiếm văn bản pháp luật chính xác hơn.

Hãy trích xuất 3 thông tin SAU (nếu có trong câu hỏi):

1. **document_type**: Loại văn bản - CHỈ chọn MỘT trong các giá trị:
   • "Luật" (bao gồm: Bộ luật, Luật)
   • "Nghị định"
   • "Thông tư"
   • "Quyết định"
   • "Nghị quyết"
   • "Chỉ thị"
   • "Công văn"

2. **field**: Lĩnh vực - CHỈ chọn MỘT trong các giá trị:
   • "Lao động"
   • "Thuế"
   • "Đất đai"
   • "Doanh nghiệp"
   • "Hình sự"
   • "Dân sự"
   • "Hành chính"
   • "Giáo dục"
   • "Y tế"
   • "Tài chính"
   • "Xây dựng"
   • "Văn hóa"
   • "Thương mại"
   • "Công nghệ thông tin"
   • "Tài nguyên"

3. **document_number**: Số hiệu văn bản (VD: 45/2019/QH14, 86/2015/NĐ-CP, 01/2021/TT-BCA)

LƯU Ý:
- Nếu KHÔNG tìm thấy → để null
- document_type và field PHẢI khớp CHÍNH XÁC danh sách trên
- "Bộ luật Lao động" → document_type="Luật", field="Lao động"
- CHỈ trả về JSON, KHÔNG giải thích

Format:
{{
    "document_type": "...",
    "field": "...",
    "document_number": "..."
}}

VÍ DỤ:

QUESTION: Điều 97 Bộ luật Lao động quy định về thời giờ làm việc?
RESPONSE:
{{
    "document_type": "Luật",
    "field": "Lao động",
    "document_number": null
}}

QUESTION: Nghị định 86/2015/NĐ-CP về bảo hiểm xã hội
RESPONSE:
{{
    "document_type": "Nghị định",
    "field": "Lao động",
    "document_number": "86/2015/NĐ-CP"
}}

QUESTION: Quy định về thuế thu nhập cá nhân như thế nào?
RESPONSE:
{{
    "document_type": null,
    "field": "Thuế",
    "document_number": null
}}

QUESTION: Thông tư 01/2021/TT-BCA về đăng ký cư trú
RESPONSE:
{{
    "document_type": "Thông tư",
    "field": "Hành chính",
    "document_number": "01/2021/TT-BCA"
}}

QUESTION: Quyền lợi của người lao động khi bị sa thải?
RESPONSE:
{{
    "document_type": null,
    "field": "Lao động",
    "document_number": null
}}

---
User question: {question}
Response (JSON only):"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])