"""
app.py

Gradio web app for the surgical ethics RAG system — same pattern as your
Mohs Surgery RAG space, wired up to this project's retrieval + generation
modules instead of rebuilding anything from scratch.

Run locally:
    python app.py

Deploy to Hugging Face Spaces:
    1. Create a new Space (SDK: Gradio) at huggingface.co/new-space
    2. Push this whole project folder to it (see deploy instructions at
       the bottom of this file's docstring, or the README section on
       Spaces deployment)
    3. Set GROQ_API_KEY as a Space secret (Settings -> Repository secrets)
       — never commit it into a file
    4. Make sure index/ (chunks.jsonl, embeddings.npy, bm25.pkl) is
       committed to the repo, since the Space container starts fresh
       every time and has no access to your local data/ folder — it needs
       the prebuilt index, not the raw source files.
"""

import json

import gradio as gr

from retrieval import hybrid_search
from generate import generate_answer, TOP_K

INDEX_DIR = "./index"


def _load_topics():
    try:
        chunks = [json.loads(l) for l in open(f"{INDEX_DIR}/chunks.jsonl")]
        topics = sorted({c.get("topic") for c in chunks if c.get("topic")})
        return topics
    except FileNotFoundError:
        return []


TOPICS = _load_topics()


def ask(question: str, topic: str):
    if not question.strip():
        return "Enter a question above.", ""

    topic_filter = None if topic == "All topics" else topic
    answer = generate_answer(question, index_dir=INDEX_DIR, topic=topic_filter)

    # Also show the raw retrieved sources underneath, collapsed — useful
    # for a resident (or a reviewer) who wants to sanity-check where the
    # answer actually came from, without cluttering the main answer.
    results = hybrid_search(question, index_dir=INDEX_DIR, topic=topic_filter, top_k=TOP_K)
    if results:
        sources_md = "\n\n".join(
            f"**{r['source_title']}** (score={r['score']:.4f})\n\n{r['text'][:400]}..."
            for r in results
        )
    else:
        sources_md = "No sources retrieved."

    return answer, sources_md


with gr.Blocks(title="Surgical Ethics Point-of-Care Assistant") as demo:
    gr.Markdown(
        "# Surgical Ethics Point-of-Care Assistant\n"
        "Ask a question about a clinical ethics situation — surrogate decision "
        "making, DNR conversations, goals of care, and more. Answers are "
        "generated only from the indexed source material below, with citations."
    )

    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            placeholder="How do I approach a surrogate decision conversation?",
            lines=2,
        )

    topic = gr.Dropdown(
        choices=["All topics"] + TOPICS,
        value="All topics",
        label="Topic filter (optional)",
    )

    submit = gr.Button("Ask", variant="primary")

    answer_box = gr.Markdown(label="Answer")

    with gr.Accordion("Retrieved sources", open=False):
        sources_box = gr.Markdown()

    submit.click(fn=ask, inputs=[question, topic], outputs=[answer_box, sources_box])
    question.submit(fn=ask, inputs=[question, topic], outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()