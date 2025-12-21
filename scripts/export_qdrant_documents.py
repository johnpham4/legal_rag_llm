"""
Export unique documents from Qdrant to JSON for evaluation dataset creation.

Usage:
    python scripts/export_qdrant_documents.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
from collections import defaultdict
from qdrant_client import QdrantClient
from loguru import logger

from llm_engineering.settings import settings


def export_documents():

    if settings.USE_QDRANT_CLOUD:
        client = QdrantClient(url=settings.QDRANT_CLOUD_URL, api_key=settings.QDRANT_APIKEY)
    else:
        client = QdrantClient(host=settings.QDRANT_DATABASE_HOST, port=settings.QDRANT_DATABASE_PORT)

    collection_name = "embedded_chunks"
    logger.info(f"Connecting to Qdrant collection: {collection_name}")

    # Collect unique documents
    all_docs = defaultdict(lambda: {"chunks": []})
    offset = None
    total_points = 0

    while True:
        results = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True
        )

        points, next_offset = results
        total_points += len(points)

        for point in points:
            payload = point.payload
            doc_num = payload.get("document_number")

            if doc_num:
                # Aggregate info from first chunk
                if not all_docs[doc_num]["chunks"]:
                    all_docs[doc_num].update({
                        "document_number": doc_num,
                        "document_type": payload.get("document_type"),
                        "field": payload.get("field"),
                        "link": payload.get("link"),
                        "platform": payload.get("platform"),
                    })

                # Collect chunk content
                content = payload.get("content", "")
                if content:
                    all_docs[doc_num]["chunks"].append(content[:300])  # First 300 chars

        logger.info(f"Processed {total_points} points, found {len(all_docs)} unique documents...")

        if next_offset is None:
            break
        offset = next_offset

    # Convert to list and add sample content
    documents = []
    for doc_num, doc_data in all_docs.items():
        chunks = doc_data.pop("chunks", [])
        doc_data["num_chunks"] = len(chunks)
        doc_data["content_sample"] = chunks[0] if chunks else ""
        documents.append(doc_data)

    # Sort by document_number
    documents.sort(key=lambda x: x["document_number"])

    # Save to JSON
    output_path = Path(__file__).parent.parent / "data" / "qdrant_documents.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    logger.info(f"Exported {len(documents)} unique documents to {output_path}")
    logger.info(f"Total points processed: {total_points}")

if __name__ == "__main__":
    export_documents()
