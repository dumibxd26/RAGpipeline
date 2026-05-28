"""Fixed-size token chunking with percentage overlap."""
from typing import List


def chunk_text(tokenizer, text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split `text` into chunks of at most `chunk_size` tokens, with `overlap`
    tokens shared between consecutive chunks. Uses the embedding model's
    tokenizer so the chunk boundaries match the units the model will actually
    see at encoding time.
    """
    if not text or not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0
    while start < len(ids):
        window = ids[start : start + chunk_size]
        if not window:
            break
        chunks.append(tokenizer.decode(window, skip_special_tokens=True).strip())
        if start + chunk_size >= len(ids):
            break
        start += step
    return [c for c in chunks if c]
