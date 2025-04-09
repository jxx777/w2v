import json
import logging
import os
from pathlib import Path

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from tqdm import tqdm

from config import Settings
from scripts.evaluate import run_simple_queries
from scripts.upload_qdrant import upload_word2vec_to_qdrant
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

    model_path = settings.MODEL_DIR / f"{settings.MODEL_NAME}.model"

    # Skip training, run queries + optionally upload to vectordb
    if model_path.exists() and not settings.MODEL_RETRAIN:
        logger.info(f"Using existing model at {model_path}. Skipping corpus extraction & training.")
        model = Word2Vec.load(str(model_path))

        # Dynamically load parameters to review based on WORD2VEC_PARAMS
        # Load external JSON config
        with open('./utils/review_model_parameters.json') as f:model_params_to_review = json.load(f)
        loaded_model_settings_to_review = {
            key: getattr(model, key)
            for key, include in model_params_to_review.items() if include
        }
        logger.info("Loaded model settings:\n%s", loaded_model_settings_to_review)

        run_simple_queries(model)

        if settings.UPLOAD_TO_VECTORDB:
            logger.info(f"Uploading existing vectors to Qdrant: {settings.VECTORDB_COLLECTION}")
            upload_word2vec_to_qdrant(model_path=str(model_path)) # Quadrant specific implementation

        exit(0)

    # Download Wikipedia dump if not found
    if not Path(settings.DATASET_PATH).exists():
        logger.info("Wikipedia dump not found. Downloading...")
        download(settings.DATASET_URL, str(settings.DATASET_PATH))
        logger.info(f"Download complete: {settings.DATASET_PATH}")

    logger.info("Parsing and tokenizing Wikipedia dump...")
    corpus = WikiCorpus(
        fname=str(settings.DATASET_PATH),
        processes=os.cpu_count(),
        article_min_tokens=15,
        token_min_len=2,
        token_max_len=30,
        lower=True,
    )

    sentences = tqdm(corpus.get_texts(), desc="Forming sentences from Wikipedia articles")

    model = Word2Vec(
        sentences,
        vector_size=settings.VECTOR_SIZE,
        window=settings.WINDOW,
        min_count=settings.MIN_COUNT,
        epochs=settings.EPOCHS,
        workers=os.cpu_count()
    )

    # Save your model after training for later reuse
    model.save(str(model_path))

    if settings.UPLOAD_TO_VECTORDB:
        logger.info(f"Uploading vectors to Qdrant: {settings.QDRANT_COLLECTION}")
        upload_word2vec_to_qdrant(model_path=str(model_path))