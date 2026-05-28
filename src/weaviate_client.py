"""Helper for connecting to Weaviate (BYO-vectors, no vectorizer module)."""
import weaviate
from weaviate.classes.config import Configure, Property, DataType

from .config import (
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_GRPC_PORT,
)


def connect():
    return weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_HTTP_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )


def ensure_collection(client, name: str, recreate: bool = True):
    """Create (or recreate) a collection with no vectorizer (BYO vectors).
    The `text` property is automatically indexed for BM25 by Weaviate.
    """
    if client.collections.exists(name):
        if recreate:
            client.collections.delete(name)
        else:
            return client.collections.get(name)

    client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(),  # default ANN, no tuning
        properties=[
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
        ],
    )
    return client.collections.get(name)
