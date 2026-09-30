"""
generate.py

The last step: takes a clinician's question, retrieves the most relevant
chunks (via retrieval.py's hybrid_search), and asks an LLM to synthesize
them into a short, structured, point-of-care answer — with every claim
attributed to a source, and an explicit "insufficient evidence" fallback
if retrieval didn't find enough to work with.

Uses Groq (same setup as tagging.py) since it's already configured and
free. Swap GROQ_MODEL or the client entirely if you want a stronger model
for this step later — generation quality matters more here than in
tagging, since this is the actual text a clinician reads.

Usage:
    python generate.py "how do I approach a surrogate decision conversation"
    python generate.py "how do I approach a DNR conversation" --topic dnr_in_or
"""

import argparse
import math
import os

from groq import Groq

from retrieval import hybrid_search

GROQ_MODEL = "openai/gpt-oss-20b"

# Deliberately small — this is a point-of-care tool, not a literature
# review. More context just means more for the resident to read mid-case.
TOP_K = 4

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


UNIT_LABELS = {  # unit_type -> (singular, plural) location label
    "page": ("PDF p.", "PDF pp."),
    "slide": ("slide", "slides"),
    "slide_notes": ("slide notes", "slide notes"),
}


def format_citation(r: dict) -> str:
    """
    'Surgical Ethics (Ferreres), "Informed Consent", PDF pp. 143-145'
    unit_range holds fractional indices for split pages (e.g. 3.01) — floor
    them back to the page/slide number. Paragraph/table indices aren't
    meaningful to a reader, so docx sources get no location.
    """
    parts = [r["source_title"]]
    if r.get("source_year") and r["source_year"] not in r["source_title"]:
        parts[0] += f" ({r['source_year']})"
    if r.get("section_heading") and r.get("source_type") == "textbook":
        parts.append(f"\"{r['section_heading']}\"")
    if r.get("unit_type") in UNIT_LABELS:
        start, end = (math.floor(x) for x in r["unit_range"])
        singular, plural = UNIT_LABELS[r["unit_type"]]
        parts.append(f"{singular} {start}" if start == end else f"{plural} {start}-{end}")
    return ", ".join(parts)


def generate_answer(query: str, index_dir: str = "./index", topic: str = None):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--index-dir", default="./index")
    parser.add_argument("--topic", default=None)
    args = parser.parse_args()

    answer, results = generate_answer(args.query, index_dir=args.index_dir, topic=args.topic)
    print(answer)
    if results:
        print("\nSources:")
        for n, r in enumerate(results, start=1):
            print(f"  [{n}] {format_citation(r)}")