from typing import Dict, Any
from fastapi.openapi.utils import get_openapi

def custom_openapi_schema(app) -> Dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Legal Q&A API - Điện Biên",
        version="1.0.0",
        description="""
        # API Hỏi Đáp Pháp Luật - Retrieval-Augmented Generation (RAG)

        ## Tính năng
        - **Hybrid Search**: Kết hợp Dense (semantic) + Sparse (BM25) embeddings
        - **Query Expansion**: Tự động mở rộng câu hỏi để tăng độ phủ
        - **Reranking**: Sử dụng Cross-Encoder để tái xếp hạng kết quả

        ## Cấu hình
        - `use_sparse=true`: Bật hybrid search (khuyến nghị)
        - `use_sparse=false`: Chỉ dùng dense search
        - `expand_to_n_queries`: Số câu hỏi mở rộng (1-5)
        - `k`: Số kết quả trả về (1-10)
        """,
        routes=app.routes,
    )

    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }

    # Examples dùng chung cho cả /rag và /rag/evaluate
    query_examples = {
        "quản_lý_nhân_lực": {
            "summary": "Quản lý phát triển nguồn nhân lực",
            "value": {
                "query": "Các giải pháp đổi mới quản lý nhà nước về phát triển nguồn nhân lực của tỉnh Điện Biên là gì?"
            }
        },
        "thu_hút_nhân_tài": {
            "summary": "Chính sách thu hút nhân lực cao cấp",
            "value": {
                "query": "Tỉnh Điện Biên có chính sách gì để thu hút nguồn nhân lực trình độ cao?"
            }
        },
        "cơ_cấu_lao_động": {
            "summary": "Cơ cấu lao động theo ngành",
            "value": {
                "query": "Cơ cấu lao động theo ngành kinh tế của tỉnh Điện Biên được đặt ra như thế nào?"
            }
        },
        "ngân_sách": {
            "summary": "Phân bổ ngân sách giáo dục",
            "value": {
                "query": "Ngân sách nhà nước được phân bổ như thế nào cho phát triển nguồn nhân lực tỉnh Điện Biên?"
            }
        },
        "đào_tạo_nghề": {
            "summary": "Mục tiêu đào tạo nghề",
            "value": {
                "query": "Mục tiêu đào tạo nghề và tạo việc làm hàng năm của tỉnh Điện Biên là bao nhiêu?"
            }
        }
    }

    # Apply examples cho cả 2 endpoints
    if "/rag" in openapi_schema["paths"]:
        openapi_schema["paths"]["/rag"]["post"]["requestBody"]["content"]["application/json"]["examples"] = query_examples

    if "/rag/evaluate" in openapi_schema["paths"]:
        openapi_schema["paths"]["/rag/evaluate"]["post"]["requestBody"]["content"]["application/json"]["examples"] = query_examples

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def apply_custom_openapi(app):
    app.openapi = lambda: custom_openapi_schema(app)
    return app
