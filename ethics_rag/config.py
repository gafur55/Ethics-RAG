"""
config.py

Every path, model name, and tuning knob in one place. Paths are resolved
from the project root, so commands work no matter which directory you run
them from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"

# --- chunking ---
CHUNK_WORDS = 350
OVERLAP_WORDS = 50

# --- embedding / vector store ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, solid baseline; swap for a
# stronger model (e.g. "BAAI/bge-base-en-v1.5") once you're validating
# retrieval quality — requires a rebuild, since stored vectors must come
# from the same model as query vectors
COLLECTION_NAME = "ethics_chunks"

# --- retrieval ---
# How deep each ranker (dense, BM25) looks before fusion. Plenty for a top_k
# of ~4 — anything below rank 50 in both lists can't win under RRF anyway.
CANDIDATES_PER_RANKER = 50

# Deliberately small — this is a point-of-care tool, not a literature
# review. More context just means more for the resident to read mid-case.
TOP_K = 4

# --- generation ---
GROQ_MODEL = "openai/gpt-oss-20b"
