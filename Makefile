# LLM Engineering Makefile

.PHONY: sync run-etl-maxime run-etl-paul lint format test clean

uv:
	uv sync --all-groups

zenml_up:
	docker compose up -d

zenml_server:
	uv run zenml connect --url http://localhost:8080 --username admin --password Admin@123

etl:
	uv run python -m tools.run --run-legal-etl --no-cache

fe:
	uv run python -m tools.run --run-feature-engineering --no-cache

train:
	uv run python -m tools.run --run-train-sparse-model --no-cache

test:
	uv run pytest tests/

del:
	uv run python scripts/delete_collections.py

test_unit:
	uv run pytest tests/integration/
