import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from tqdm import tqdm

from scripts.evaluate import run_full_evaluation
from scripts.train import train_word2vec_model
from scripts.upload_qdrant import upload_word2vec_to_qdrant

from utils.download import download

#  Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger("main")

# Load env config
WIKI_URL: str = os.getenv("WIKI_URL")
WIKI_PATH: Path = Path(os.getenv("WIKI_PATH"))
MODEL_DIR: Path = Path(os.getenv("MODEL_DIR"))
MODEL_NAME: str = os.getenv("MODEL_NAME")
MODEL_PATH: str = str(MODEL_DIR / f"{MODEL_NAME}.model")

# Hyperparameters
VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", 300))
WINDOW: int = int(os.getenv("WINDOW", 5))
MIN_COUNT: int = int(os.getenv("MIN_COUNT", 3))
EPOCHS: int = int(os.getenv("EPOCHS", 10))

# Qdrant config
UPLOAD_TO_QDRANT: bool = os.getenv("UPLOAD_TO_QDRANT", "false").lower() == "true"
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", MODEL_NAME)
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))

MODEL_OVERRIDE: bool = os.getenv("MODEL_OVERRIDE", "false").lower() == "true"

# Main logic
if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_PATH.parent.mkdir(parents=True, exist_ok=True)

    if Path(MODEL_PATH).exists() and not MODEL_OVERRIDE:
        logger.info(f"Using existing model at {MODEL_PATH}. Skipping corpus extraction & training.")
        model = Word2Vec.load(MODEL_PATH)

        logger.info("Running evaluation on loaded model...")
        run_full_evaluation(model)

        if UPLOAD_TO_QDRANT:
            logger.info(f"Uploading existing vectors to Qdrant: {QDRANT_COLLECTION}")
            upload_word2vec_to_qdrant(
                model_path=MODEL_PATH,
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                collection_name=QDRANT_COLLECTION
            )

        exit(0)  # skip rest of training pipeline


    if not WIKI_PATH.exists():
        logger.info("Wikipedia dump not found. Downloading...")

        download(WIKI_URL, str(WIKI_PATH))
        logger.info(f"Download complete: {WIKI_PATH}")

    logger.info("Parsing and tokenizing Wikipedia dump...")

    wiki = WikiCorpus(
        fname=str(WIKI_PATH),
        processes=os.cpu_count(),
        article_min_tokens=15,  # Adjusted threshold to include quality articles (tweak based on corpus statistics)
        token_min_len=2,  # Lower bound lowered to keep short but meaningful words, e.g., "vin"
        token_max_len=30,
        lower=True,
    )

    sentences = [tokens for tokens in tqdm(wiki.get_texts(), desc="Forming sentences from wikipedia articles")]
    logger.info(f"Extracted {len(sentences):,} tokenized articles.")

    model = train_word2vec_model(
        sentences=sentences,
        save_dir=MODEL_DIR,
        model_name=MODEL_NAME,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        epochs=EPOCHS
    )

    logger.info("Running evaluation on freshly trained model...")
    run_full_evaluation(model)

    if UPLOAD_TO_QDRANT:
        logger.info(f"Uploading vectors to Qdrant: {QDRANT_COLLECTION}")
        upload_word2vec_to_qdrant(
            model_path=str(MODEL_DIR / f"{MODEL_NAME}.model"),
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=QDRANT_COLLECTION
        )