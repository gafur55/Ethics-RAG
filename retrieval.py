"""
retrieval.py

Hybrid retrieval combining BM25 (keyword) and dense vector search using
Reciprocal Rank Fusion (RRF) — same technique from Module 2 of the course.

Metadata filtering happens BEFORE scoring, not after: for a query tagged
with state="Illinois", topic="surrogate_decision_making", we narrow to
matching chunks first, then rank within that subset. This matters for legal
content — you don't want a semantically-similar-but-wrong-state chunk
outranking the correct one.
"""

from typing import List, Dict, Optional
from embed_index import load_indices, _tokenize

# How deep each ranker (dense, BM25) looks before fusion. Plenty for a top_k
# of ~4 — anything below rank 50 in both lists can't win under RRF anyway.
CANDIDATES_PER_RANKER = 50


def _filter_indices(chunks: List[Dict], topic: Optional[str], state: Optional[str]) -> List[int]:
    idxs = []
    for i, c in enumerate(chunks):
        if topic and c.get("topic") != topic:
            continue
        if state and c.get("state") not in (None, state):
            # allow chunks with no state tag through (general guidance),
            # exclude chunks explicitly tagged for a DIFFERENT state
            continue
        idxs.append(i)
    return idxs


def _rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """Standard Reciprocal Rank Fusion: score = sum(1 / (k + rank))."""
    scores: Dict[int, float] = {}
    for rank_list in rank_lists:
        for rank, idx in enumerate(rank_list):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_search(
    query: str,
    index_dir: str,
    topic: Optional[str] = None,
    state: Optional[str] = None,
    top_k: int = 4,
):
    collection, bm25, chunks, model = load_indices(index_dir)

    candidate_idxs = _filter_indices(chunks, topic, state)
    if not candidate_idxs:
        return []  # nothing matches the filter — caller should handle the
        # "no answer found, refer to X" fallback here rather than falling
        # back to unfiltered search
    candidate_set = set(candidate_idxs)
    n_candidates = min(CANDIDATES_PER_RANKER, len(candidate_idxs))

    # --- dense ranking within candidates (Chroma, topic pre-filtered) ---
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    dense = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=n_candidates,
        where={"topic": topic} if topic else None,
        include=[],
    )
    dense_order = [int(i) for i in dense["ids"][0] if int(i) in candidate_set]

    # --- BM25 ranking within candidates ---
    tokenized_query = _tokenize(query)
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


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "how do I approach a surrogate decision conversation"
    results = hybrid_search(q, index_dir="./index")
    for r in results:
        print(f"[{r['source_type']}] {r['source_title']} ({r.get('topic')}, {r.get('state')}) — score={r['score']:.4f}")
        print(r["text"][:200], "...\n")
