import logging

from gensim.models import Word2Vec
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

logger = logging.getLogger("qdrant-upload")


def upload_word2vec_to_qdrant(
        model_path: str,
        collection_name: str,
        host: str = "127.0.0.1",
        port: int = 6333,
        distance: Distance = Distance.COSINE,
        batch_size: int = 1000
):
    # ✅ Load the trained model
    logger.info(f"Loading model from: {model_path}")
    model: Word2Vec = Word2Vec.load(model_path)
    vector_size = model.vector_size
    vocab = model.wv.index_to_key

    # ✅ Connect to Qdrant
    client = QdrantClient(host=host, port=port)

    # ✅ Create or recreate the collection with correct params
    logger.info(f"Creating Qdrant collection: '{collection_name}' with vector size {vector_size}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance
        )
    )

    # ✅ Build & upload points in batches
    logger.info(f"📤 Uploading {len(vocab):,} vectors to Qdrant...")

    points = []
    for idx, word in enumerate(vocab):
        vector = model.wv[word].tolist()
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={"word": word})
        )

        # Upload in batches
        if len(points) == batch_size or idx == len(vocab) - 1:
            client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Uploaded {idx + 1:,} / {len(vocab):,}")
            points = []

    logger.info("Upload complete.")
