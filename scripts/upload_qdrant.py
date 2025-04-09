import logging
from gensim.models import Word2Vec, FastText
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config import Settings

settings = Settings()
logger = logging.getLogger("qdrant-upload")


def upload_embedding_model_to_quadrant(
        model_path: str,
        batch_size: int = 20000,
        distance: Distance = Distance.COSINE
):
    """
    Uploads a vector embeddings model's vectors to Qdrant using configuration
    parameters specific to Qdrant from the centralized config.

    Parameters:
        model_path: Path to the embedding model (.model).
        batch_size: Number of vectors to upload in each batch.
        distance: Distance metric for vector similarity.
    """

    logger.info(f"Loading model from: {model_path}")

    model_type = settings.MODEL_TYPE.lower()
    match model_type:
        case "word2vec":
            model = Word2Vec.load(model_path)
        case "fasttext":
            model = FastText.load(model_path)
        case _:
            raise ValueError(f"Unsupported model type '{settings.MODEL_TYPE}'")

    vector_size = model.vector_size
    vocab = model.wv.index_to_key

    client = QdrantClient(
        host=settings.VECTORDB_HOST,
        port=settings.VECTORDB_PORT
    )

    collection_name = settings.VECTORDB_COLLECTION

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
        points.append(PointStruct(
            id=idx,
            vector=model.wv[word].tolist(),
            payload={"word": word}
        ))

        if len(points) == batch_size or idx == len(vocab) - 1:
            client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Uploaded {idx + 1:,} / {len(vocab):,}")
            points = []

    logger.info("Upload complete.")
