import papermill as pm
from datetime import datetime
from loguru import logger
from pathlib import Path
from llm_engineering.settings import settings

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pm.execute_notebook(
        input_path="001-evaluate_search.ipynb",
        output_path=f"outputs/001-evaluate_{settings.SPARSE_ALGORITHM}_{timestamp}.ipynb",
    )

    logger.info(f"Done! Output: outputs/001-evaluate_{settings.SPARSE_ALGORITHM}_{timestamp}.ipynb")
