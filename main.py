import json
import logging
from pathlib import Path

from gensim.models import Word2Vec

from config import Settings
from scripts.evaluate import run_simple_queries
from scripts.train import train_embedding_model
from scripts.upload_qdrant import upload_embedding_model_to_quadrant
from utils.corpus_loader import load_or_tokenize_wiki
from utils.download import download

# Load centralized configuration
settings = Settings()

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("main")

if __name__ == "__main__":
    logger.info("Current configuration settings:\n%s", settings.model_dump_json(indent=2))

    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Path(settings.DATASET_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Model exists: Skip corpus extraction & training, run queries + optionally upload to vectordb
    model_path: Path = settings.MODEL_DIR / f"{settings.MODEL_NAME}.model" # Relative to cd
    if model_path.exists() and not settings.MODEL_TRAIN:
        logger.info(f"Using existing model at {model_path}. Skipping corpus extraction & training.")
        model = Word2Vec.load(str(model_path))

        # Dynamically load parameters to review based on Word2Vec params
        with open('./utils/review_model_parameters.json') as f:model_params_to_review = json.load(f) # Load external JSON config
        loaded_model_settings_to_review = {
            key: getattr(model, key)
            for key, include in model_params_to_review.items() if include
        }
        logger.info("Loaded model settings:\n%s", loaded_model_settings_to_review)

        run_simple_queries(model)

        if settings.UPLOAD_TO_VECTORDB:
            logger.info(f"Uploading existing vectors to vectordb: {settings.VECTORDB_COLLECTION}")
            upload_embedding_model_to_quadrant(model_path=str(model_path))  # Quadrant specific implementation

        exit(0)

    # Download Wikipedia dump if not found
    if not Path(settings.DATASET_PATH).exists():
        logger.info("Dataset / dump not found. Downloading...")
        download(settings.DATASET_URL, str(settings.DATASET_PATH))
        logger.info(f"Download complete: {settings.DATASET_PATH}")

    # We're 'caching' the resulted sentences so subsequent runs are faster
    corpus_checkpoint = Path(f"./checkpoints/{settings.MODEL_NAME}.pkl")

    # Should pickle checkpoint exist - we return that early, else iterate corpus
    sentences = load_or_tokenize_wiki(
        settings.DATASET_PATH,
        corpus_checkpoint
    )

    # Embedding model training entrypoint
    model = train_embedding_model(
        model_type=settings.MODEL_TYPE,
        sentences=sentences,
        save_dir=settings.MODEL_DIR,
        model_name=settings.MODEL_NAME,
        vector_size=settings.VECTOR_SIZE,
        window=settings.WINDOW,
        min_count=settings.MIN_COUNT,
        epochs=settings.EPOCHS,
        resume=settings.MODEL_RESUME
    )

    # After training the model
    run_simple_queries(model)

    if settings.UPLOAD_TO_VECTORDB:
        logger.info(f"Uploading existing vectors to vectordb: {settings.VECTORDB_COLLECTION}")
        upload_embedding_model_to_quadrant(model_path=str(model_path))  # Quadrant specific implementation