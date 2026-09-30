"""
embed_index.py

Builds both halves of your hybrid search from Module 2:
- Dense vector index: a persistent ChromaDB collection (index/chroma/) holding
  each chunk's text, embedding (sentence-transformers), and metadata. Chroma
  does the nearest-neighbour search and the topic pre-filtering.
- BM25 keyword index (rank_bm25): NOT stored — it's rebuilt from the chunk
  texts in Chroma at load time, which takes well under a second at this
  corpus size and means Chroma is the single source of truth.

Embeddings are computed here with sentence-transformers rather than by
Chroma's built-in embedding function, so indexing and querying are
guaranteed to use the same model (see EMBEDDING_MODEL).
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, solid baseline; swap for a
# stronger model (e.g. "BAAI/bge-base-en-v1.5") once you're validating
# retrieval quality rather than just wiring things up

COLLECTION_NAME = "ethics_chunks"


def _to_metadata(chunk: Dict) -> Dict:
    """
    Chroma metadata values must be str/int/float/bool (no None, no lists), so
    unit_range is split into unit_start/unit_end and unset fields are dropped.
    """
    meta = {k: v for k, v in chunk.items() if k not in ("text", "unit_range") and v is not None}
    meta["unit_start"], meta["unit_end"] = chunk["unit_range"]
    return meta


def _from_record(text: str, meta: Dict) -> Dict:
    chunk = {k: v for k, v in meta.items() if k not in ("unit_start", "unit_end")}
    chunk["text"] = text
    chunk["unit_range"] = [meta["unit_start"], meta["unit_end"]]
    return chunk


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_indices(chunks: List[Dict], out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=str(out_path / "chroma"))
    # Full rebuild every run — simpler and safer than diffing against the
    # previous index when files are added/removed/renamed in data/.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )

    batch = client.get_max_batch_size()
    for start in range(0, len(chunks), batch):
        end = start + batch
        collection.add(
            ids=[str(i) for i in range(start, min(end, len(chunks)))],
            documents=texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=[_to_metadata(c) for c in chunks[start:end]],
        )

    print(f"Indexed {collection.count()} chunks -> {out_path / 'chroma'}")


@lru_cache(maxsize=None)
def load_indices(index_dir: str):
    """
    Returns (collection, bm25, chunks, model). Cached, so the embedding model
    and BM25 index are only built once per process rather than per query.
    chunks[i] corresponds to Chroma id str(i) and to BM25 document i.
    """
    client = chromadb.PersistentClient(path=str(Path(index_dir) / "chroma"))
    collection = client.get_collection(COLLECTION_NAME)

    records = collection.get(include=["documents", "metadatas"])
    order = sorted(range(len(records["ids"])), key=lambda i: int(records["ids"][i]))
    chunks = [_from_record(records["documents"][i], records["metadatas"][i]) for i in order]

    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    model = SentenceTransformer(EMBEDDING_MODEL)
    return collection, bm25, chunks, model
