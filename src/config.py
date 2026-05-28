"""Global configuration for the RAG pipeline."""

DATASETS = ["nfcorpus", "scifact", "fiqa"]

# Embedding model (bi-encoder)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Chunking: fixed size in tokens, 10% overlap.
# MiniLM has max_seq_length=256 tokens, so 256 is the safe default for indexing.
# Requirement 4 (sweep) will test 256/512/1024.
CHUNK_SIZE = 256
CHUNK_OVERLAP_RATIO = 0.1

# Weaviate (running via docker-compose)
WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8080
WEAVIATE_GRPC_PORT = 50051

# Folder where BEIR downloads the archives
DATA_DIR = "datasets"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def collection_name(dataset: str) -> str:
    """Weaviate class name (must start with an uppercase letter)."""
    return "Beir_" + dataset.capitalize()
