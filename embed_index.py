"""
embed_index.py

Builds both halves of your hybrid search from Module 2:
- Dense vector index (sentence-transformers, stored as a plain numpy array —
  no vector DB needed yet at this corpus size)
- BM25 keyword index (rank_bm25)

Both are saved to disk alongside the chunk metadata so retrieval.py can
load everything back without recomputing.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, solid baseline; swap for a
# stronger model (e.g. "BAAI/bge-base-en-v1.5") once you're validating
# retrieval quality rather than just wiring things up


def build_indices(chunks: List[Dict], out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    texts = [c["text"] for c in chunks]

    # --- dense vectors ---
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    np.save(out_path / "embeddings.npy", embeddings)

    # --- BM25 ---
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(out_path / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # --- chunk metadata (aligned by index to both indices above) ---
    with open(out_path / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Indexed {len(chunks)} chunks -> {out_path}")


def load_indices(index_dir: str):
    index_path = Path(index_dir)
    embeddings = np.load(index_path / "embeddings.npy")
    with open(index_path / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    chunks = [json.loads(line) for line in open(index_path / "chunks.jsonl")]
    model = SentenceTransformer(EMBEDDING_MODEL)
    return embeddings, bm25, chunks, model
