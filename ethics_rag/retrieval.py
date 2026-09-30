"""
retrieval.py

Hybrid retrieval combining BM25 (keyword) and dense vector search using
Reciprocal Rank Fusion (RRF) — same technique from Module 2 of the course.

Topic filtering happens BEFORE scoring, not after: for
topic="surrogate_decision_making" both rankers only ever see matching
chunks, so a semantically-similar chunk from another topic can't outrank
the right one.
"""

from typing import List, Dict, Optional

from .config import CANDIDATES_PER_RANKER, INDEX_DIR, TOP_K
from .vectorstore import load_indices, tokenize


def _rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """Standard Reciprocal Rank Fusion: score = sum(1 / (k + rank))."""
    scores: Dict[int, float] = {}
    for rank_list in rank_lists:
        for rank, idx in enumerate(rank_list):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_search(
    query: str,
    index_dir=INDEX_DIR,
    topic: Optional[str] = None,
    top_k: int = TOP_K,
):
    collection, bm25, chunks, model = load_indices(str(index_dir))

    candidate_idxs = [i for i, c in enumerate(chunks) if not topic or c.get("topic") == topic]
    if not candidate_idxs:
        return []  # nothing matches the filter — caller should handle the
        # "no answer found, refer to X" fallback here rather than falling
        # back to unfiltered search
    n_candidates = min(CANDIDATES_PER_RANKER, len(candidate_idxs))

    # --- dense ranking within candidates (Chroma, topic pre-filtered) ---
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    dense = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=n_candidates,
        where={"topic": topic} if topic else None,
        include=[],
    )
    dense_order = [int(i) for i in dense["ids"][0]]

    # --- BM25 ranking within candidates ---
    tokenized_query = tokenize(query)
    all_bm25_scores = bm25.get_scores(tokenized_query)
    candidate_bm25 = [(i, all_bm25_scores[i]) for i in candidate_idxs]
    bm25_order = [i for i, _ in sorted(candidate_bm25, key=lambda x: -x[1])][:n_candidates]

    # --- fuse ---
    fused_scores = _rrf_fuse([dense_order, bm25_order])
    ranked = sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]

    results = []
    for idx, score in ranked:
        c = chunks[idx]
        results.append({**c, "score": score})
    return results

