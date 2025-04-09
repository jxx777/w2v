# config.py
from pydantic_settings import BaseSettings  # Updated import location
from pathlib import Path


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
    UPLOAD_TO_QDRANT: bool = False
    QDRANT_HOST: str = "127.0.0.1"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "word2vec_en"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"