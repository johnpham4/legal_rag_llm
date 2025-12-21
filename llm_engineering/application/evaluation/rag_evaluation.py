from loguru import logger
import statistics
from ranx import Qrels, Run, evaluate

from llm_engineering.domain.evaluation import EvaluationResult

class RetrievalEvaluator:
    def __init__(self, retriever=None):
        self.retriever = retriever

    @staticmethod
    def evaluate_query(
        query_id: str,
        retrieved_doc_ids: list[str],
        relevant_doc_ids: list[str],
        k: int = 10
    ) -> dict:
        """Evaluate retrieval for a single query using ranx.

        Args:
            query_id: Query identifier
            retrieved_doc_ids: List of retrieved document IDs (ranked by score)
            relevant_doc_ids: Ground truth relevant document IDs
            k: Cutoff for evaluation metrics

        Returns:
            Dict with metrics: precision@k, recall@k, mrr, ndcg@k, map@k
        """
        # Create Qrels (ground truth)
        qrels_dict = {
            query_id: {doc_id: 1 for doc_id in relevant_doc_ids}
        }
        qrels = Qrels(qrels_dict)

        # Create Run (retrieved results with scores)
        run_dict = {
            query_id: {
                doc_id: 1.0 / (rank + 1)  # Score decreases with rank
                for rank, doc_id in enumerate(retrieved_doc_ids[:k])
            }
        }
        run = Run(run_dict)

        # Evaluate with ranx - optimized metrics for small dataset
        # MRR: primary metric for datasets with few relevant docs per query
        # Hit Rate@k: binary - có tìm thấy ÍT NHẤT 1 relevant doc không? (0 hoặc 1)
        # Recall@k: coverage - tìm được bao nhiêu % trong tổng relevant docs
        # NDCG: standard ranking quality metric
        # MAP: reference metric
        metrics = evaluate(
            qrels=qrels,
            run=run,
            metrics=[
                "mrr",  # Mean Reciprocal Rank (PRIMARY)
                f"hit_rate@1",  # Binary: có relevant doc ở rank 1?
                f"hit_rate@3",  # Binary: có relevant doc trong top 3?
                f"hit_rate@5",  # Binary: có relevant doc trong top 5?
                f"hit_rate@{k}",  # Binary: có relevant doc trong top K?
                f"recall@3",  # Coverage: % relevant docs tìm được trong top 3
                f"recall@5",  # Coverage: % relevant docs tìm được trong top 5
                f"recall@{k}",  # Coverage: % relevant docs tìm được trong top K
                f"ndcg@5",  # NDCG at 5
                f"ndcg@{k}",  # NDCG at K
                "map"  # Mean Average Precision (reference)
            ]
        )

        return metrics

    def compare(
        self,
        query: str,
        relevant_doc_ids: list[str],
        k: int = 10
    ) -> dict:
        """Compare dense vs hybrid search for a single query.

        Args:
            query: User query string
            relevant_doc_ids: Ground truth document IDs (e.g., ["12/2017/QH14"])
            k: Number of results to evaluate

        Returns:
            Dict with dense and hybrid results + improvement metrics
        """
        from llm_engineering.domain.queries import EmbeddedQuery

        logger.info(f"Evaluating query: {query[:60]}...")
        query_id = f"q_{hash(query) % 10000}"  # Generate consistent query ID

        # Test 1: Dense only
        logger.info("Testing Dense-only search...")
        original_sparse_config = EmbeddedQuery.Config.use_sparse_vector_index
        EmbeddedQuery.Config.use_sparse_vector_index = False

        dense_results = self.retriever.search(
            query=query,
            k=k,
            expand_to_n_queries=1
        )
        dense_doc_ids = [r.document_number for r in dense_results if r.document_number]

        # Test 2: Hybrid (Dense + Sparse)
        logger.info("Testing Hybrid search (Dense + Sparse)...")
        EmbeddedQuery.Config.use_sparse_vector_index = True

        hybrid_results = self.retriever.search(
            query=query,
            k=k,
            expand_to_n_queries=1
        )
        hybrid_doc_ids = [r.document_number for r in hybrid_results if r.document_number]

        # Restore original config
        EmbeddedQuery.Config.use_sparse_vector_index = original_sparse_config

        # Evaluate both with ranx
        dense_metrics = RetrievalEvaluator.evaluate_query(
            query_id=query_id,
            retrieved_doc_ids=dense_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=k
        )

        hybrid_metrics = RetrievalEvaluator.evaluate_query(
            query_id=query_id,
            retrieved_doc_ids=hybrid_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=k
        )

        # Calculate improvement
        def calc_improvement(hybrid_val, dense_val):
            if dense_val == 0:
                return 0.0
            return ((hybrid_val - dense_val) / dense_val) * 100

        improvements = {}
        for metric_name in dense_metrics:
            dense_val = dense_metrics[metric_name]
            hybrid_val = hybrid_metrics[metric_name]
            improvements[f"{metric_name}_pct"] = calc_improvement(hybrid_val, dense_val)

        # Winner based on MRR (best metric for datasets with few relevant docs)
        winner = "hybrid" if hybrid_metrics["mrr"] > dense_metrics["mrr"] else "dense"

        return {
            "query": query,
            "dense": {
                **dense_metrics,
                "retrieved_docs": dense_doc_ids[:5]  # Top 5 for inspection
            },
            "hybrid": {
                **hybrid_metrics,
                "retrieved_docs": hybrid_doc_ids[:5]
            },
            "improvement": improvements,
            "winner": winner
        }

    def compare_batch(
        self,
        test_queries: list[dict],
        k: int = 10
    ) -> dict:
        """Compare across multiple queries and aggregate results.

        Args:
            test_queries: List of {"query": str, "relevant_doc_ids": list[str]}
            k: Number of results to evaluate

        Returns:
            Summary statistics with aggregated metrics
        """
        results = []

        for test_case in test_queries:
            try:
                result = self.compare(
                    query=test_case["query"],
                    relevant_doc_ids=test_case["relevant_doc_ids"],
                    k=k
                )
                results.append(result)
                logger.info(f"{test_case['query'][:40]}... | Winner: {result['winner']}")
            except Exception as e:
                logger.error(f"Failed to evaluate query '{test_case['query'][:40]}...': {e}")
                continue

        if not results:
            return {"error": "No results", "num_queries": 0}

        # Aggregate metrics using ranx structure
        metric_names = [k for k in results[0]["dense"].keys() if k != "retrieved_docs"]

        dense_metrics = {
            metric: [r["dense"][metric] for r in results]
            for metric in metric_names
        }

        hybrid_metrics = {
            metric: [r["hybrid"][metric] for r in results]
            for metric in metric_names
        }

        improvements = {
            f"{metric}_pct": [
                ((r["hybrid"][metric] - r["dense"][metric]) / r["dense"][metric] * 100)
                if r["dense"][metric] > 0 else 0.0
                for r in results
            ]
            for metric in metric_names
        }

        # Winner count
        hybrid_wins = sum(1 for r in results if r["winner"] == "hybrid")
        dense_wins = len(results) - hybrid_wins

        summary = {
            "num_queries": len(results),
            "dense_avg": {k: statistics.mean(v) for k, v in dense_metrics.items()},
            "hybrid_avg": {k: statistics.mean(v) for k, v in hybrid_metrics.items()},
            "avg_improvement": {k: statistics.mean(v) for k, v in improvements.items()},
            "wins": {
                "hybrid": hybrid_wins,
                "dense": dense_wins,
                "win_rate_hybrid": hybrid_wins / len(results) * 100
            },
            "recommendation": "hybrid" if hybrid_wins > dense_wins else "dense" if dense_wins > hybrid_wins else "inconclusive",
            "details": results  # Full results for inspection
        }

        return summary