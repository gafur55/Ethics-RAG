"""
generate.py

The last step: takes a clinician's question, retrieves the most relevant
chunks (via retrieval.py's hybrid_search), and asks an LLM to synthesize
them into a short, structured, point-of-care answer — with every claim
cited by passage number, and an explicit "insufficient evidence" fallback
if retrieval didn't find enough to work with.

Uses Groq since it's free. Swap GROQ_MODEL (config.py) or the client
entirely if you want a stronger model later — this is the actual text a
clinician reads, so generation quality matters.
"""

import os

from dotenv import load_dotenv
from groq import Groq

from .citations import format_citation
from .config import GROQ_MODEL, INDEX_DIR, PROJECT_ROOT, TOP_K
from .retrieval import hybrid_search

# Picks up GROQ_API_KEY from the project's .env file (gitignored) if present.
# Real environment variables — e.g. Hugging Face Space secrets — take
# precedence, since load_dotenv doesn't override existing ones.
load_dotenv(PROJECT_ROOT / ".env")

SYSTEM_PROMPT = """You are a point-of-care assistant for surgery residents facing clinical ethics
situations in real time — think "glance at your phone before walking into the room," not a
literature review.

You will be given a clinician's question and a set of retrieved source passages, each numbered
[1], [2], ... and labeled with its source. Using ONLY the information in those passages:

1. Give a short, direct answer (3-6 sentences) — practical guidance on how to approach the
   situation, not an essay on ethical theory.
2. If the passages contain concrete facts, scripts, or phrases a clinician could actually use,
   include them.
3. Cite the passage number(s) inline right after each key claim, e.g. "... [1]" or "... [2][3]".
   Use only these bracketed numbers — never invent author names, years, or page numbers.
   Bracketed numbers appearing INSIDE a passage's text are that paper's own reference list,
   not passage numbers — never cite those.
4. If the passages don't actually contain enough relevant information to answer the question,
   say so explicitly rather than filling gaps from general knowledge — say something like
   "The retrieved sources don't directly address this; consider consulting [topic] literature
   or an ethics consult" instead of guessing.

Never state a fact that isn't traceable to one of the provided passages. This is used in real
clinical situations — accuracy and honesty about the limits of the retrieved material matter
more than sounding complete.
"""


def generate_answer(query: str, index_dir=INDEX_DIR, topic: str = None):
    """
    Returns (answer, results). results are the numbered passages the answer's
    [n] citations refer to — results[0] is [1] — so callers can display
    exactly what the model saw.
    """
    results = hybrid_search(query, index_dir=index_dir, topic=topic, top_k=TOP_K)

    if not results:
        return (
            "No relevant material was found in the indexed sources"
            + (f" for topic '{topic}'" if topic else "")
            + ". Consider broadening the search, checking a different topic, "
            + "or consulting an ethics consult directly."
        ), []

    context_blocks = []
    for n, r in enumerate(results, start=1):
        context_blocks.append(f"[{n}] {format_citation(r)}\n{r['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    user_content = f"Clinician's question: {query}\n\nRetrieved passages:\n\n{context}"

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=600,
        reasoning_effort="low",
        reasoning_format="hidden",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip(), results

