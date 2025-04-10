from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Directories and file paths
    MODEL_DIR: Path = Path("models")
    DATASET_DIR: Path = Path("datasets")
    CHECKPOINT_DIR: Path = Path("checkpoints")

    # Dataset configuration
    DATASET_URL: str = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    DATASET_FILE: str = "enwiki-latest-pages-articles.xml.bz2"
    DATASET_PATH: Path = DATASET_DIR / DATASET_FILE

    # Model configuration
    MODEL_TYPE: str = "Word2Vec"  # or "fasttext"
    MODEL_TRAIN: bool = False
    MODEL_NAME: str = "word2vec_enwiki-latest-pages-articles"
    MODEL_RESUME: bool = False

    # Model hyperparameters
    VECTOR_SIZE: int = 300
    WINDOW: int = 5
    MIN_COUNT: int = 3
    EPOCHS: int = 10

    # Corpus checkpoint strategy (choices: "streaming" or "serialized")
    CORPUS_CHECKPOINT_STRATEGY: str = "streaming"

    # Vector DB integration (for Qdrant/Quadrant)
    UPLOAD_TO_VECTORDB: bool = False
    VECTORDB_HOST: str = "127.0.0.1"
    VECTORDB_PORT: int = 6333
    VECTORDB_COLLECTION: str = "word2vec_enwiki-latest-pages-articles"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
