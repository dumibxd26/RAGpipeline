"""Download and loading of BEIR datasets."""
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from .config import BEIR_URL, DATA_DIR


def ensure_dataset(name: str) -> str:
    """Return the dataset folder path, downloading it if missing."""
    data_path = Path(DATA_DIR) / name
    if not (data_path / "corpus.jsonl").exists():
        util.download_and_unzip(BEIR_URL.format(name), DATA_DIR)
    return str(data_path)


def load(name: str, split: str = "test"):
    """Return (corpus, queries, qrels) for the requested split."""
    data_path = ensure_dataset(name)
    return GenericDataLoader(data_folder=data_path).load(split=split)
