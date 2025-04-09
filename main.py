import logging
import os
from pathlib import Path

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from tqdm import tqdm

from config import Settings
from scripts.evaluate import run_full_evaluation, themes
from scripts.upload_qdrant import upload_word2vec_to_qdrant
from utils.download import download

# Load centralized configuration for better management
settings = Settings()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger("main")

if __name__ == "__main__":

    logger.info("Current configuration settings:\n%s", settings.model_dump_json(indent=2))
    # Ensure that necessary directories exist
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Path(settings.DATASET_PATH).parent.mkdir(parents=True, exist_ok=True)

    model_path = settings.MODEL_DIR / f"{settings.MODEL_NAME}.model"

    # If a pre-trained model exists and override is not enabled, load and evaluate it
    if model_path.exists() and not settings.MODEL_OVERRIDE:
        logger.info(f"Using existing model at {model_path}. Skipping corpus extraction & training.")
        model = Word2Vec.load(str(model_path))

        logger.info("Running evaluation on loaded model...")
        run_full_evaluation(model, themes)

        if settings.UPLOAD_TO_QDRANT:
            logger.info(f"Uploading existing vectors to Qdrant: {settings.QDRANT_COLLECTION}")
            upload_word2vec_to_qdrant(model_path=str(model_path))
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
        article_min_tokens=15,  # Adjust for quality articles
        token_min_len=2,        # Include short but meaningful words
        token_max_len=30,
        lower=True,
    )

    sentences = tqdm(
        corpus.get_texts(),
        desc="Forming sentences from Wikipedia articles"
    )

    # Train the model using settings directly
    model = Word2Vec(
        sentences,
        vector_size=settings.VECTOR_SIZE,
        window=settings.WINDOW,
        min_count=settings.MIN_COUNT,
        epochs=settings.EPOCHS,
        workers=os.cpu_count()
    )

    logger.info("Running evaluation on freshly trained model...")
    run_full_evaluation(model, themes)  # Pass the themes list

    if settings.UPLOAD_TO_QDRANT:
        logger.info(f"Uploading vectors to Qdrant: {settings.QDRANT_COLLECTION}")
        upload_word2vec_to_qdrant(model_path=str(model_path))