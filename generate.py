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

You will be given a clinician's question and a set of retrieved source passages, each labeled
with its source title. Using ONLY the information in those passages:

1. Give a short, direct answer (3-6 sentences) — practical guidance on how to approach the
   situation, not an essay on ethical theory.
2. If the passages contain concrete facts, scripts, or phrases a clinician could actually use,
   include them.
3. After the answer, list which source(s) each key claim came from, e.g. "(Torke 2008)".
4. If the passages don't actually contain enough relevant information to answer the question,
   say so explicitly rather than filling gaps from general knowledge — say something like
   "The retrieved sources don't directly address this; consider consulting [topic] literature
   or an ethics consult" instead of guessing.

Never state a fact that isn't traceable to one of the provided passages. This is used in real
clinical situations — accuracy and honesty about the limits of the retrieved material matter
more than sounding complete.
"""


def generate_answer(query: str, index_dir: str = "./index", topic: str = None) -> str:
    results = hybrid_search(query, index_dir=index_dir, topic=topic, top_k=TOP_K)

    if not results:
        return (
            "No relevant material was found in the indexed sources"
            + (f" for topic '{topic}'" if topic else "")
            + ". Consider broadening the search, checking a different topic, "
            + "or consulting an ethics consult directly."
        )

    context_blocks = []
    for r in results:
        context_blocks.append(f"[Source: {r['source_title']}]\n{r['text']}")
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
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--index-dir", default="./index")
    parser.add_argument("--topic", default=None)
    args = parser.parse_args()

    answer = generate_answer(args.query, index_dir=args.index_dir, topic=args.topic)
    print(answer)