import logging
from pathlib import Path

from config import Settings
from scripts.evaluate import run_simple_queries
from scripts.train import train_embedding_model
from scripts.upload_qdrant import upload_embedding_model_to_quadrant
from utils.corpus_loader import load_or_tokenize_wiki
from utils.download import download
from utils.model_loader import load_embedding_model  # New import

# Load configuration
settings = Settings()

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("main")

# Ensure required directories exist
for directory in [settings.MODEL_DIR, settings.DATASET_DIR, settings.CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory created: {directory}")

if __name__ == "__main__":
    logger.info("Current configuration settings:\n%s", settings.model_dump_json(indent=2))

    # Build the model path based on settings
    model_path: Path = settings.MODEL_DIR / f"{settings.MODEL_NAME}.model"

    # Use existing model if present and training is disabled
    if model_path.exists() and not settings.MODEL_TRAIN:
        logger.info(f"Using existing model at {model_path}. Skipping training.")
        model = load_embedding_model(model_path, settings.MODEL_TYPE)
        run_simple_queries(model)

        if settings.UPLOAD_TO_VECTORDB:
            logger.info(f"Uploading vectors to vectordb: {settings.VECTORDB_COLLECTION}")
            upload_embedding_model_to_quadrant(
                model=model,
                collection_name=settings.VECTORDB_COLLECTION
            )
        exit(0)

    # Download dataset if it does not exist
    if not settings.DATASET_PATH.exists():
        logger.info("Dataset not found. Downloading...")
        download(settings.DATASET_URL, str(settings.DATASET_PATH))
        logger.info(f"Download complete: {settings.DATASET_PATH}")

    # Determine the checkpoint strategy
    match settings.CORPUS_CHECKPOINT_STRATEGY:
        case "serialized":
            corpus_checkpoint = settings.CHECKPOINT_DIR / f"{settings.DATASET_FILE}.pkl"
            use_streaming = False
        case "streaming":
            corpus_checkpoint = settings.CHECKPOINT_DIR / f"{settings.DATASET_FILE}.txt"
            use_streaming = True
        case _:
            raise ValueError(f"Invalid checkpoint strategy: {settings.CORPUS_CHECKPOINT_STRATEGY}")

    # Load or create the tokenized corpus for training
    sentences = load_or_tokenize_wiki(
        dataset_path=settings.DATASET_PATH,
        checkpoint_path=corpus_checkpoint,
        use_streaming=use_streaming
    )

    # Train the embedding model and extract metadata via spaCy
    model, corpus_metadata = train_embedding_model(
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

    # Evaluate the trained model
    run_simple_queries(model)

    # Optionally upload vectors to the vector database
    if settings.UPLOAD_TO_VECTORDB:
        logger.info(f"Uploading vectors to vectordb: {settings.VECTORDB_COLLECTION}")
        # Here, if you wish, you can use `corpus_metadata` to enrich payloads in your upload.
        upload_embedding_model_to_quadrant(
            model=model,
            collection_name=settings.VECTORDB_COLLECTION,
            metadata=corpus_metadata,
        )