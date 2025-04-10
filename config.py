from pydantic_settings import BaseSettings
from pathlib import Path

# Base class for settings, allowing values to be overridden by environment variables (.env)
class Settings(BaseSettings):
    # Dataset
    DATASET_URL: str = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    DATASET_PATH: Path = "./datasets/enwiki-latest-pages-articles.xml.bz2"

    # Corpus storing strategy
    CHECKPOINT_STRATEGY = "streaming"

    # Model configuration
    MODEL_TYPE: str = "Word2Vec"
    MODEL_TRAIN: bool = False
    MODEL_DIR: Path = Path("models")
    MODEL_NAME: str = "word2vec_enwiki-latest-pages-articles"
    MODEL_RESUME: bool = False  # Or True to resume by default

    # Model hyperparameters (Gensim)
    VECTOR_SIZE: int = 300
    WINDOW: int = 5
    MIN_COUNT: int = 3
    EPOCHS: int = 10

    # VectorDb integration (for now Quadrant)
    UPLOAD_TO_VECTORDB: bool = False
    VECTORDB_HOST: str = "127.0.0.1"
    VECTORDB_PORT: int = 6333
    VECTORDB_COLLECTION: str = "word2vec_enwiki-latest-pages-articles"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"