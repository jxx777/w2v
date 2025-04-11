import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from config import Settings

settings = Settings()
logger = logging.getLogger("qdrant-upload")

def upload_embedding_model_to_quadrant(
        model,
        collection_name: str,
        metadata: dict = None,    # Additional metadata mapping word -> attributes
        batch_size: int = 15000,
        distance: Distance = Distance.COSINE
):
    """
    Uploads vectors from a preloaded embedding model to Qdrant,
    along with optional metadata for each word.

    Parameters:
        model: Pre-loaded Word2Vec or FastText model.
        collection_name: The target collection name for Qdrant.
        metadata: A dictionary mapping words to additional attributes (e.g., POS, lemma, frequency)
        batch_size: Number of vectors to upload in each batch.
        distance: Distance metric for vector similarity.
    """
    logger.info("Starting model upload to Qdrant...")

    vector_size = model.vector_size
    vocab = model.wv.index_to_key

    client = QdrantClient(
        host=settings.VECTORDB_HOST,
        port=settings.VECTORDB_PORT
    )

    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' already exists. Deleting it.")
        client.delete_collection(collection_name=collection_name)

    logger.info(f"Creating Qdrant collection: '{collection_name}' with vector size {vector_size}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance
        )
    )

    logger.info(f"Uploading {len(vocab):,} vectors to Qdrant...")
    points = []
    for idx, word in enumerate(vocab):
        # Get additional metadata for the word if provided
        extra_meta = metadata.get(word, {}) if metadata is not None else {}
        # Construct payload including the basic word key
        payload = {"word": word}
        # Merge extra metadata into payload (if any)
        payload.update(extra_meta)

        points.append(PointStruct(
            id=idx,
            vector=model.wv[word].tolist(),
            payload=payload
        ))
        # Upload in batches
        if len(points) == batch_size or idx == len(vocab) - 1:
            client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Uploaded {idx + 1:,} / {len(vocab):,}")
            points = []

    logger.info("Upload complete.")
