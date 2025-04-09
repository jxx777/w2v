import logging
from gensim.models import Word2Vec
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

logger = logging.getLogger("qdrant-upload")


def upload_word2vec_to_qdrant(
        model_path: str,
        batch_size: int = 20000,
        distance: Distance = Distance.COSINE
):
    """
    Uploads a Word2Vec model's vectors to Qdrant using only configuration parameters
    specific to Qdrant from the centralized config.

    Parameters:
        model_path: Path to the Word2Vec model.
        batch_size: Number of vectors to upload in each batch.
        distance: Distance metric for vector similarity.
    """
    # Load configuration inside the function
    from config import Settings
    settings = Settings()

    logger.info(f"Loading model from: {model_path}")
    model: Word2Vec = Word2Vec.load(model_path)
    vector_size = model.vector_size
    vocab = model.wv.index_to_key

    # Use Qdrant-specific configuration from settings
    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT
    )

    logger.info(f"Creating Qdrant collection: '{settings.QDRANT_COLLECTION}' with vector size {vector_size}")
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance
        )
    )

    logger.info(f"Uploading {len(vocab):,} vectors to Qdrant...")

    points = []
    for idx, word in enumerate(vocab):
        vector = model.wv[word].tolist()
        # Point struct
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={"word": word}
        ))

        if len(points) == batch_size or idx == len(vocab) - 1:
            client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
            logger.info(f"Uploaded {idx + 1:,} / {len(vocab):,}")
            points = []

    logger.info("Upload complete.")
