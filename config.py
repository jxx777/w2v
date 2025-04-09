from pydantic_settings import BaseSettings
from pathlib import Path

# Base class for settings, allowing values to be overridden by environment variables (.env)
class Settings(BaseSettings):
    # Wikipedia dump
    DATASET_URL: str = "https://dumps.wikimedia.org/rowiki/latest/rowiki-latest-pages-articles.xml.bz2"
    DATASET_PATH: Path = "datasets/rowiki-latest-pages-articles.xml.bz2"

    # Model configuration
    MODEL_RETRAIN: bool = False
    MODEL_DIR: Path = Path("models")
    MODEL_NAME: str = "word2vec_en"

    # Word2Vec hyperparameters
    VECTOR_SIZE: int = 300
    WINDOW: int = 5
    MIN_COUNT: int = 3
    EPOCHS: int = 10

    # Qdrant integration
    UPLOAD_TO_VECTORDB: bool = False
    VECTORDB_HOST: str = "127.0.0.1"
    VECTORDB_PORT: int = 6333
    VECTORDB_COLLECTION: str = "word2vec_en"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"