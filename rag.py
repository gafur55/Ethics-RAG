"""
rag.py — the one command-line entry point.

    python rag.py build                          # rebuild index/ from data/
    python rag.py search "question" [--topic T]  # retrieval only, no LLM
    python rag.py ask "question" [--topic T]     # retrieval + Groq answer
    python rag.py topics                         # list topic values + counts
    python rag.py eval-questions                 # one-time: build the AI test set
    python rag.py eval [--answers 30]            # score quality, compare with last run
"""

import argparse
from collections import Counter

from ethics_rag.citations import format_citation
from ethics_rag.config import DATA_DIR, INDEX_DIR, TOP_K


def cmd_build(args):
    from ethics_rag.pipeline import run_pipeline
    run_pipeline(args.data_dir, args.index_dir)


def cmd_search(args):
    from ethics_rag.retrieval import hybrid_search
    results = hybrid_search(args.query, index_dir=args.index_dir, topic=args.topic, top_k=args.top_k)
    if not results:
        print("No results" + (f" for topic '{args.topic}'" if args.topic else "") + ".")
    for n, r in enumerate(results, start=1):
        print(f"[{n}] {format_citation(r)}")
        print(f"    topic={r.get('topic')}  type={r['source_type']}  score={r['score']:.4f}")
        print("    " + " ".join(r["text"].split())[:args.chars] + "...\n")


def cmd_ask(args):
    from ethics_rag.generate import generate_answer
    answer, results = generate_answer(args.query, index_dir=args.index_dir, topic=args.topic)
    print(answer)
    if results:
        print("\nSources:")
        for n, r in enumerate(results, start=1):
            print(f"  [{n}] {format_citation(r)}")


def cmd_topics(args):
    from ethics_rag.vectorstore import load_indices
    _, _, chunks, _ = load_indices(str(args.index_dir))
    for topic, count in Counter(c.get("topic") for c in chunks).most_common():
        print(f"{count:6d}  {topic}")


def cmd_eval_questions(args):
    from ethics_rag.evaluation import QUESTIONS_FILE, generate_questions
    if QUESTIONS_FILE.exists() and not args.overwrite:
        raise SystemExit(f"{QUESTIONS_FILE.name} already exists — scores are only comparable on the "
                         "same questions. Pass --overwrite to replace it anyway.")
    generate_questions(n=args.n, index_dir=args.index_dir)


def cmd_eval(args):
    from ethics_rag.evaluation import run_eval
    run_eval(index_dir=args.index_dir, top_k=args.top_k, answers=args.answers, note=args.note)


def main():
    parser = argparse.ArgumentParser(description="Clinical ethics RAG")
    parser.add_argument("--index-dir", default=INDEX_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="rebuild the vector database from data/")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.set_defaults(func=cmd_build)

    for name, func, help_text in [
        ("search", cmd_search, "show retrieved passages only (no LLM call)"),
        ("ask", cmd_ask, "answer a question with cited sources (calls Groq)"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("query")
        p.add_argument("--topic", default=None)
        p.set_defaults(func=func)
    sub.choices["search"].add_argument("--top-k", type=int, default=TOP_K)
    sub.choices["search"].add_argument("--chars", type=int, default=400,
                                       help="how much of each passage to print")

    p = sub.add_parser("topics", help="list topic values and chunk counts")
    p.set_defaults(func=cmd_topics)

    p = sub.add_parser("eval-questions", help="one-time: have AI write the test question set")
    p.add_argument("--n", type=int, default=100, help="number of on-topic questions")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_eval_questions)

    p = sub.add_parser("eval", help="score retrieval (and optionally answers) on the test set")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--answers", type=int, default=0, metavar="N",
                   help="also generate and AI-grade N answers (calls Groq; ~2 calls each)")
    p.add_argument("--note", default="", help="label for this run, e.g. 'bge embeddings'")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
