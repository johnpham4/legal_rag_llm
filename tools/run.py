from datetime import datetime as dt
from pathlib import Path

import click
from loguru import logger

from llm_engineering import settings
from pipelines import legal_data_etl, feature_engineering, train_sparse_model

@click.command(
    help="""
LLM Engineering project CLI v0.0.1.

Main entry point for the pipeline execution.
This entrypoint is where everything comes together.

Run the ZenML LLM Engineering project pipelines with various options.

Run a pipeline with the required parameters. This executes
all steps in the pipeline in the correct order using the orchestrator
stack component that is configured in your active ZenML stack.

Examples:

  \b
  # Run the pipeline with default options
  python run.py

  \b
  # Run the pipeline without cache
  python run.py --no-cache

  \b
  # Run only the ETL pipeline
  python run.py --only-etl
    """
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--run-legal-etl",
    is_flag=True,
    default=False,
    help="Run the Legal ETL pipeline for Vietnamese legal documents.",
)
@click.option(
    "--run-feature-engineering",
    is_flag=True,
    default=False,
    help="Whether to run the FE pipeline.",
)
@click.option(
    "--run-train-sparse-model",
    is_flag=True,
    default=False,
    help="Run the training sparse model pipline.",
)
def main(
    no_cache: bool = False,
    run_legal_etl: bool = False,
    run_feature_engineering: bool = False,
    run_train_sparse_model: bool = False,
):
    assert run_legal_etl or run_feature_engineering or run_train_sparse_model, "Please use one of the options"

    pipeline_args = {
        "enable_cache": not no_cache,
    }

    root_dir = Path(__file__).resolve().parent.parent

    if run_legal_etl:
        pipeline_args["config_path"] = root_dir / "configs" / "legal_data_etl.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"legal_data_etl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting Legal Data ETL Pipeline for Vietnamese legal documents")
        legal_data_etl.with_options(**pipeline_args)()

    if run_feature_engineering:
        pipeline_args["config_path"] = root_dir / "configs" / "feature_engineering.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"feature_engineering_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting Feature Engineering Pipeline for Vietnamese legal documents")
        feature_engineering.with_options(**pipeline_args)()

    if run_train_sparse_model:
        pipeline_args["config_path"] = root_dir / "configs" / "train_sparse_embedding.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"feature_engineering_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting Feature Engineering Pipeline for Vietnamese legal documents")
        train_sparse_model.with_options(**pipeline_args)()

if __name__ == "__main__":
    main()
