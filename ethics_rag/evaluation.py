"""
evaluation.py

Measures quality without a human expert, in two parts:

1. Retrieval (free, fast): an LLM writes a question from a randomly chosen
   passage, so the "right answer" is known by construction — that passage.
   A question is a hit if search puts that passage (or an overlapping piece
   of the same pages) in the top k.

2. Answers (optional, calls Groq): the real system answers each question,
   then a larger "judge" model checks every claim against the passages the
   answer was given — is it supported, is its [n] citation right, did it
   answer the question, and did it honestly refuse the off-topic questions.

The question set is generated once and saved (eval/questions.jsonl), so
every run is scored on the same test and runs can be compared.
"""

import json
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from .citations import format_citation
from .config import (
    EMBEDDING_MODEL, GROQ_MODEL, INDEX_DIR, JUDGE_MODEL, PROJECT_ROOT, TOP_K,
)
from .generate import generate_answer
from .retrieval import hybrid_search
from .vectorstore import load_indices

load_dotenv(PROJECT_ROOT / ".env")

EVAL_DIR = PROJECT_ROOT / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
RUNS_DIR = EVAL_DIR / "runs"

MIN_PASSAGE_WORDS = 120  # shorter passages rarely make a self-contained question

# Questions the library should NOT be able to answer — clinical technique
# and pharmacology, not ethics. The right behaviour is an honest "the
# sources don't cover this", so these test refusal, not retrieval.
OFF_TOPIC_QUESTIONS = [
    "What is the weight-based dosing for heparin after a DVT?",
    "What suture should I use for fascial closure after a midline laparotomy?",
    "What are the steps to achieve the critical view of safety in a laparoscopic cholecystectomy?",
    "What antibiotic prophylaxis is recommended before an appendectomy?",
    "How do I place an ultrasound-guided internal jugular central line?",
    "What is the vancomycin dose for an MRSA wound infection?",
    "How do I read a chest x-ray for a pneumothorax?",
    "What propofol infusion rate should I start for ICU sedation?",
    "What CPT code do I use to bill an inpatient surgical consult?",
    "How long should I wait after eating before an elective hernia repair?",
]

QUESTION_PROMPT = """You are helping build a test set for a search tool used by surgery residents
facing clinical ethics situations.

Below is one passage from the tool's library. Write ONE question that a surgery resident might
realistically ask — in their own words, the way they'd ask a colleague in the hallway — that
THIS passage specifically helps answer.

Rules:
- Do not copy distinctive phrases from the passage; paraphrase the situation.
- Do not mention "the passage", "the text", the authors, the title, or the book.
- The question must be answerable from this passage, not just loosely related to it.
- If the passage can't support a realistic question (reference list, table of contents,
  acknowledgments, author bios, copyright notices, fragments), set usable to false.

Return JSON: {"usable": true or false, "question": "..."}

Passage (from: SOURCE_LABEL):
PASSAGE_TEXT"""

JUDGE_PROMPT = """You are grading an answer produced by a clinical-ethics assistant. The assistant
was only allowed to use the numbered passages below. You are NOT judging whether the advice is
clinically wise — only whether it faithfully reflects the passages.

1. Split the answer into its distinct factual claims or recommendations about the clinical or
   ethical situation. Do NOT count as claims: statements about what the sources do or don't
   contain ("the passages don't address X"), suggestions to consult other resources or an
   ethics consult, or pure filler. An answer that only says the sources don't cover the
   question has zero claims. For each claim record:
   - "cited": the passage numbers cited for it (empty list if none)
   - "supported_by_cited": true only if the cited passage(s) actually state or clearly imply it
   - "supported_by_any": true if ANY of the passages state or clearly imply it
2. "refused": true if the answer's main message is that the retrieved sources don't address
   the question (e.g. "The retrieved passages do not provide any information about X").
   Example: an answer saying "The sources don't cover heparin dosing; consult a pharmacology
   reference" -> refused: true, claims: [].
3. "relevance": 1-5, how directly the answer addresses the question asked
   (5 = directly and fully, 1 = off-target). If refused, rate how appropriate refusing was
   given the passages (5 = passages clearly don't cover it, 1 = they clearly do).

Return JSON: {"claims": [{"claim": "...", "cited": [1], "supported_by_cited": true,
"supported_by_any": true}], "refused": false, "relevance": 5}

Question: QUESTION

Passages:
PASSAGES

Answer to grade:
ANSWER"""


def _client() -> Groq:
    # generous retries: the free tier rate-limits bursts, and the client
    # backs off automatically between attempts
    return Groq(max_retries=8)


def _ask_json(client: Groq, prompt: str, max_tokens: int = 1500) -> dict:
    for attempt in range(3):
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=max_tokens,
            reasoning_effort="low",
            reasoning_format="hidden",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            if attempt == 2:
                raise
            time.sleep(1)


def _pages(chunk: dict) -> tuple:
    start, end = chunk["unit_range"]
    return int(start), int(end)


def _same_passage(a: dict, b: dict) -> bool:
    """
    Same file and overlapping page/slide range. Compared this way rather than
    by chunk id so a question set stays valid after re-chunking or switching
    embedding models (chunk ids change on every rebuild).
    """
    if a["source_path"] != b["source_path"]:
        return False
    (a0, a1), (b0, b1) = _pages(a), _pages(b)
    return a0 <= b1 and b0 <= a1


# ---------------------------------------------------------------------------
# 1. Build the question set (run once)
# ---------------------------------------------------------------------------

def generate_questions(n: int = 100, index_dir=INDEX_DIR, seed: int = 0):
    """
    Samples passages evenly across (source_type, topic) groups — otherwise
    ~60% would come from the textbooks — and has the judge model write one
    question per passage. Writes eval/questions.jsonl.
    """
    _, _, chunks, _ = load_indices(str(index_dir))
    rng = random.Random(seed)

    groups = defaultdict(list)
    for c in chunks:
        if len(c["text"].split()) >= MIN_PASSAGE_WORDS:
            groups[(c["source_type"], c["topic"])].append(c)
    for g in groups.values():
        rng.shuffle(g)

    # round-robin across groups so small topics are represented too
    queue = []
    for i in range(max(len(g) for g in groups.values())):
        for key in sorted(groups):
            if i < len(groups[key]):
                queue.append(groups[key][i])

    client = _client()
    questions = []
    for c in queue:
        if len(questions) >= n:
            break
        prompt = (QUESTION_PROMPT
                  .replace("SOURCE_LABEL", format_citation(c))
                  .replace("PASSAGE_TEXT", c["text"]))
        result = _ask_json(client, prompt, max_tokens=800)
        if not result.get("usable") or not result.get("question", "").strip():
            continue
        questions.append({
            "id": f"q{len(questions) + 1:03d}",
            "kind": "on_topic",
            "question": result["question"].strip(),
            "source_path": c["source_path"],
            "unit_range": c["unit_range"],
            "source_label": format_citation(c),
            "source_type": c["source_type"],
            "topic": c["topic"],
        })
        print(f"  [{len(questions):3d}/{n}] {questions[-1]['question']}")

    for i, q in enumerate(OFF_TOPIC_QUESTIONS, start=1):
        questions.append({"id": f"x{i:03d}", "kind": "off_topic", "question": q})

    EVAL_DIR.mkdir(exist_ok=True)
    with open(QUESTIONS_FILE, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(f"\nSaved {len(questions)} questions -> {QUESTIONS_FILE.relative_to(PROJECT_ROOT)}")


def load_questions():
    if not QUESTIONS_FILE.exists():
        raise SystemExit("No question set yet — run: python rag.py eval-questions")
    return [json.loads(line) for line in open(QUESTIONS_FILE)]


# ---------------------------------------------------------------------------
# 2. Score a run
# ---------------------------------------------------------------------------

def _retrieval_metrics(questions, index_dir, top_k):
    per_question, by_group = [], defaultdict(list)
    for q in questions:
        if q["kind"] != "on_topic":
            continue
        results = hybrid_search(q["question"], index_dir=index_dir, top_k=top_k)
        rank = next((i for i, r in enumerate(results, start=1) if _same_passage(r, q)), None)
        doc_hit = any(r["source_path"] == q["source_path"] for r in results)
        per_question.append({"id": q["id"], "rank": rank, "doc_hit": doc_hit})
        group = q["topic"] if q["source_type"] == "journal_article" else q["source_type"]
        by_group[group].append(rank is not None)

    n = len(per_question)
    return {
        "questions": n,
        "passage_hit_rate": sum(p["rank"] is not None for p in per_question) / n,
        "document_hit_rate": sum(p["doc_hit"] for p in per_question) / n,
        "mrr": sum(1 / p["rank"] for p in per_question if p["rank"]) / n,
        "passage_hit_rate_by_group": {g: sum(v) / len(v) for g, v in sorted(by_group.items())},
    }, per_question


def _judge(client, question, answer, results):
    passages = "\n\n".join(f"[{n}] {format_citation(r)}\n{r['text']}"
                           for n, r in enumerate(results, start=1))
    prompt = (JUDGE_PROMPT
              .replace("QUESTION", question)
              .replace("PASSAGES", passages or "(none retrieved)")
              .replace("ANSWER", answer))
    return _ask_json(client, prompt)


def _answer_metrics(questions, index_dir, limit, retrieval_per_question):
    rng = random.Random(0)
    on_topic = [q for q in questions if q["kind"] == "on_topic"]
    sample = rng.sample(on_topic, min(limit, len(on_topic))) + [q for q in questions if q["kind"] == "off_topic"]
    rng.shuffle(sample)  # mix off-topic in, so a rate-limit stop still leaves some graded
    hit = {p["id"]: p["rank"] is not None for p in retrieval_per_question}

    client = _client()
    graded, stopped_early = [], None
    for i, q in enumerate(sample, start=1):
        print(f"  grading answer {i}/{len(sample)}: {q['question'][:70]}")
        try:
            answer, results = generate_answer(q["question"], index_dir=index_dir)
            verdict = _judge(client, q["question"], answer, results)
        except RateLimitError as e:
            # Free tier daily token caps — keep what's graded rather than
            # losing the whole run
            stopped_early = str(e)[:300]
            print(f"\n  ⚠️  Groq rate limit hit — stopping after {len(graded)} graded answers.")
            break
        graded.append({"id": q["id"], "kind": q["kind"], "question": q["question"],
                       "answer": answer, "verdict": verdict})

    claims = [c for g in graded if g["kind"] == "on_topic" for c in g["verdict"].get("claims", [])]
    on = [g for g in graded if g["kind"] == "on_topic"]
    off = [g for g in graded if g["kind"] == "off_topic"]
    answered = [g for g in on if not g["verdict"].get("refused")]

    def rate(xs, pred):
        return sum(map(pred, xs)) / len(xs) if xs else None

    return {
        "answers_graded": len(graded),
        "answers_planned": len(sample),
        "stopped_early": stopped_early,
        "claims_graded": len(claims),
        # share of claims backed by SOME passage — the rest are hallucinated
        "faithfulness": rate(claims, lambda c: bool(c.get("supported_by_any"))),
        # share of claims whose own [n] citation backs them
        "citation_accuracy": rate(claims, lambda c: bool(c.get("cited")) and bool(c.get("supported_by_cited"))),
        "uncited_claim_rate": rate(claims, lambda c: not c.get("cited")),
        "relevance_avg": rate(answered, lambda g: g["verdict"].get("relevance", 0)),
        "off_topic_refusal_rate": rate(off, lambda g: bool(g["verdict"].get("refused"))),
        # refused even though the source passage WAS retrieved — over-cautious
        "false_refusal_rate": rate([g for g in on if hit.get(g["id"])],
                                   lambda g: bool(g["verdict"].get("refused"))),
    }, graded


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


def run_eval(index_dir=INDEX_DIR, top_k: int = TOP_K, answers: int = 0, note: str = ""):
    questions = load_questions()
    previous = _latest_run()

    print(f"Scoring retrieval on {sum(q['kind'] == 'on_topic' for q in questions)} questions...")
    retrieval, per_question = _retrieval_metrics(questions, str(index_dir), top_k)

    answer_metrics, graded = None, []
    if answers:
        print(f"\nGrading {answers} answers + {len(OFF_TOPIC_QUESTIONS)} off-topic with {JUDGE_MODEL}...")
        answer_metrics, graded = _answer_metrics(questions, str(index_dir), answers, per_question)

    run = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "note": note,
        "config": {"top_k": top_k, "embedding_model": EMBEDDING_MODEL,
                   "answer_model": GROQ_MODEL, "judge_model": JUDGE_MODEL},
        "retrieval": retrieval,
        "answers": answer_metrics,
        "per_question": per_question,
        "graded_answers": graded,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(run, indent=2))

    _print_report(run, previous)
    print(f"\nFull results (every question, answer, and verdict): {out.relative_to(PROJECT_ROOT)}")


def _latest_run():
    runs = sorted(RUNS_DIR.glob("*.json")) if RUNS_DIR.exists() else []
    return json.loads(runs[-1].read_text()) if runs else None


def _print_report(run, previous):
    def line(label, key, section, pct=True):
        value = (run[section] or {}).get(key)
        if value is None:
            return
        prev = ((previous or {}).get(section) or {}).get(key)
        fmt = (lambda v: f"{v:6.1%}") if pct else (lambda v: f"{v:6.2f}")
        delta = ""
        if prev is not None:
            diff = value - prev
            delta = f"   (was {fmt(prev).strip()}, {'+' if diff >= 0 else ''}{diff * 100 if pct else diff:.1f}{'pts' if pct else ''})"
        print(f"  {label:<34}{fmt(value)}{delta}")

    k = run["config"]["top_k"]
    print(f"\n=== Retrieval ({run['retrieval']['questions']} questions, top {k}) ===")
    line("Source passage found", "passage_hit_rate", "retrieval")
    line("Source document found", "document_hit_rate", "retrieval")
    line("MRR (1.0 = always ranked #1)", "mrr", "retrieval", pct=False)
    print("  By group (passage found):")
    for g, v in run["retrieval"]["passage_hit_rate_by_group"].items():
        print(f"    {g:<42}{v:6.1%}")

    if run["answers"]:
        a = run["answers"]
        print(f"\n=== Answers ({a['answers_graded']} graded, {a['claims_graded']} claims) ===")
        if a.get("stopped_early"):
            print(f"  ⚠️  INCOMPLETE — only {a['answers_graded']}/{a['answers_planned']} graded "
                  "(Groq rate limit). Scores are less reliable; rerun later.")
        line("Faithfulness (claims supported)", "faithfulness", "answers")
        line("Citation accuracy", "citation_accuracy", "answers")
        line("Uncited claims", "uncited_claim_rate", "answers")
        line("Relevance (1-5)", "relevance_avg", "answers", pct=False)
        line("Off-topic correctly refused", "off_topic_refusal_rate", "answers")
        line("False refusals", "false_refusal_rate", "answers")
    if previous:
        print(f"\n  (compared with run from {previous['timestamp']}"
              + (f", note: {previous['note']}" if previous.get("note") else "") + ")")
