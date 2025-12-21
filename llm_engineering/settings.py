from pathlib import Path
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
# from zenml.client import Client
# from zenml.exceptions import EntityExistsError

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    # MongoDB database
    DATABASE_HOST: str = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "legal"

    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "keepitreal/vietnamese-sbert"  # Vietnamese dense embedding
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"

    # Hugging face
    HF_TOKEN: str | None = None
    HF_REPO_ID: str | None = None

    # Cohere API
    COHERE_MODEL_ID: str = "command-r-08-2024"
    COHERE_API_KEY: str | None = None

    SPARSE_ALGORITHM: str = "tfidf"

    @property
    def SPARSE_MODEL_PATH(self) -> str:
        """Return absolute path to sparse model."""
        project_root = Path(__file__).parent.parent
        return str((project_root / f"models/sparse_{self.SPARSE_ALGORITHM}_model.pkl").resolve())

    # QdrantDB Vector DB
    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    # AWS S3
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str | None = None

    @classmethod
    def load_settings(cls) -> "Settings":

        # try:
        #     logger.info("Loading settings from the ZenML secret store.")

        #     settings_secrets = Client().get_secret("settings")
        #     settings = Settings(**settings_secrets.secret_values)
        # except (RuntimeError, KeyError):
        #     logger.warning(
        #         "Failed to load settings from the ZenML secret store. Defaulting to loading the settings from the '.env' file."
        #     )
        settings = Settings()

        return settings

    def export(self) -> None:
        """
        Exports the settings to the ZenML secret store.
        """

        env_vars = self.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()

        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning(
                "Secret 'scope' already exists. Delete it manually by running 'zenml secret delete settings', before trying to recreate it."
            )

settings = Settings.load_settings()