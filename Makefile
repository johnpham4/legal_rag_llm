# LLM Engineering Makefile

.PHONY: sync run-etl-maxime run-etl-paul lint format test clean

uv:
	uv sync --all-groups

lab:
	uv run jupyter lab --ip=127.0.0.1 --port 8000

zenml_up:
	docker compose up -d

zenml:
	uv run zenml connect --url http://localhost:8080 --username admin --password Admin@123

etl:
	uv run python -m tools.run --run-legal-etl --no-cache

fe:
	uv run python -m tools.run --run-feature-engineering --no-cache

train:
	uv run python -m tools.run --run-train-sparse-model --no-cache

del:
	uv run python scripts/delete_collections.py

api:
	uv run python llm_engineering/infrastructure/inference_pipeline_api.py

eval:
	uv run python 001-evaluate_search.py

