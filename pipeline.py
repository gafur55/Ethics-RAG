"""
pipeline.py

End-to-end orchestrator, tuned to your actual data/ layout:

    data/
      Curriculum Facilitator Guides/   -> source_type=internal_curriculum
      Curriculum Presentations/        -> source_type=internal_curriculum
      Ethics Consult Cases/            -> source_type=case_example_source
      McCullough Chapter.pdf           -> loose file, source_type=textbook (DEFAULT_ROOT_SOURCE_TYPE)
      Relevant Literature/             -> special-cased: source_type=journal_article,
        Surrogate Decision Making/         AND topic_hint pulled from the subfolder name
        DNR in OR/                         (e.g. "surrogate_decision_making")
        ... (any subfolder)

Because your literature subfolders are already organized by topic, that
topic_hint gets carried through extraction -> chunking -> tagging and
overrides whatever the LLM would otherwise guess. This is both cheaper and
more accurate than free-form topic inference for those ~60 papers.

Run:

    python pipeline.py --data-dir ./data --index-dir ./index --tag

Use --tag only once you're ready to spend the API calls — it's the slow,
costly step. Skip it while you're just testing extraction/chunking on a
new batch of files.
"""

import argparse
import json
import re
from pathlib import Path

from extractors import extract_file
from chunker import chunk_units
from embed_index import build_indices

# Keys are matched against folder.name.strip() so trailing spaces in your
# actual folder names (e.g. "Curriculum Facilitator Guides ") don't cause
# a silent mismatch.
FOLDER_TO_SOURCE_TYPE = {
    "Curriculum Facilitator Guides": "internal_curriculum",
    "Curriculum Presentations": "internal_curriculum",
    "Ethics Consult Cases": "case_example_source",
    "books": "textbook",
}

# Optional per-file metadata manifests sitting in --data-dir (e.g. the AMA
# Journal of Ethics download manifest). Each has a "documents" list whose
# entries carry "file" (path relative to the project root) plus "title",
# "year", "source_url" — used to give those files real citation titles
# instead of their slugified filenames.
METADATA_MANIFEST_GLOB = "*_sources.json"

# The "Relevant Literature" folder is handled specially (see below): each of
# ITS subfolders is a topic label, not a source_type. Everything under it is
# journal_article (papers) unless it's obviously a society statement — adjust
# per-file below if you want finer granularity (e.g. ACS/AMA statements as
# "society_guideline" instead of "journal_article").
LITERATURE_FOLDER_NAME = "Relevant Literature"
LITERATURE_SOURCE_TYPE = "journal_article"

# source_type to use for files sitting loose directly in --data-dir
# (e.g. McCullough Chapter.pdf), rather than in one of the mapped folders.
DEFAULT_ROOT_SOURCE_TYPE = "textbook"


def _normalize_topic_folder_name(name: str) -> str:
    """
    'Disagreements w Colleagues-Attendings' -> 'disagreements_with_colleagues_attendings'
    'Futility (Inappropriate Treatment) '   -> 'futility'   (parenthetical dropped)
    'DNR in OR'                             -> 'dnr_in_or'
    Must line up with TOPIC_TAXONOMY in tagging.py — if a mapping here doesn't
    match an entry there, tagging.py will still accept it (topic_hint bypasses
    the taxonomy check), but retrieval filtering works best when they align.
    """
    name = name.strip()
    name = re.sub(r"\s*\([^)]*\)", "", name)  # drop "(Inappropriate Treatment)"
    name = name.replace("w Colleagues-Attendings", "with Colleagues Attendings")
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name


def _load_manifests(data_path: Path) -> dict:
    """resolved file path -> {"source_title", "source_year", "source_url"}"""
    meta = {}
    for manifest in data_path.glob(METADATA_MANIFEST_GLOB):
        for doc in json.load(open(manifest)).get("documents", []):
            meta[str(Path(doc["file"]).resolve())] = {
                "source_title": doc.get("title"),
                "source_year": doc.get("year"),
                "source_url": doc.get("source_url"),
            }
    return meta


def _clean_book_title(stem: str) -> str:
    """
    'Surgical Ethics -- edited by Laurence B_ McCullough, ... -- 1, FR, 1998 -- ...'
      -> 'Surgical Ethics (edited by Laurence B. McCullough, ...)'
    Filenames from the books/ folder pack title -- authors -- edition -- ...
    into the stem; keep just title and authors.
    """
    parts = [p.strip() for p in stem.split(" -- ")]
    if len(parts) < 2:
        return stem
    return f"{parts[0]} ({parts[1].replace('_', '.')})"


def run_pipeline(data_dir: str, index_dir: str, do_tag: bool):
    data_path = Path(data_dir)
    all_units = []
    manifest_meta = _load_manifests(data_path)

    def process_file(file_path: Path, source_type: str, topic_hint: str = None):
        if file_path.suffix.lower() not in (".pptx", ".docx", ".pdf"):
            return
        print(f"Extracting: {file_path}" + (f"  [topic_hint={topic_hint}]" if topic_hint else ""))
        try:
            units = extract_file(str(file_path), source_type, topic_hint)
        except Exception as e:
            print(f"  FAILED: {e}")
            return

        overrides = manifest_meta.get(str(file_path.resolve()), {})
        if not overrides and source_type == "textbook":
            overrides = {"source_title": _clean_book_title(file_path.stem)}
        for u in units:
            u.update({k: v for k, v in overrides.items() if v})
        all_units.extend(units)

    for item in data_path.iterdir():
        if item.is_dir() and item.name.strip() == LITERATURE_FOLDER_NAME:
            # Each subfolder here is a topic, not a source_type.
            for topic_folder in item.iterdir():
                if not topic_folder.is_dir():
                    continue
                topic_hint = _normalize_topic_folder_name(topic_folder.name)
                for file_path in topic_folder.rglob("*"):
                    process_file(file_path, LITERATURE_SOURCE_TYPE, topic_hint)
        elif item.is_dir():
            source_type = FOLDER_TO_SOURCE_TYPE.get(item.name.strip())
            if source_type is None:
                print(f"Skipping unrecognized folder: '{item.name}' "
                      f"(add it to FOLDER_TO_SOURCE_TYPE if it should be included)")
                continue
            for file_path in item.rglob("*"):
                process_file(file_path, source_type)  # no topic_hint — these
                # folders span multiple topics per file, so topic is left for
                # the LLM tagging step to infer
        else:
            # loose file directly under data_dir, e.g. McCullough Chapter.pdf
            process_file(item, DEFAULT_ROOT_SOURCE_TYPE)

    print(f"\nExtracted {len(all_units)} raw units total.")

    chunks = chunk_units(all_units)
    print(f"Chunked into {len(chunks)} chunks.")

    if do_tag:
        from tagging import tag_chunks
        print("Tagging chunks with LLM (batched requests, ~8 chunks per call)...")
        chunks = tag_chunks(chunks)
    else:
        # No LLM call needed: topic_hint already came from the literature
        # subfolder names, so use it directly as topic. Chunks with no
        # topic_hint (curriculum/facilitator/case files, which span multiple
        # topics per file) get "other" — retrieval can still find them via
        # BM25/vector search, just not via topic filtering.
        # content_type/state/authority_tier are simply left unset; retrieval
        # only filters on topic/state, and state was mostly empty in your
        # corpus anyway (no state statute text yet).
        print("Skipping LLM tagging — using topic_hint directly as topic "
              "(no content_type/state/authority_tier metadata).")
        for c in chunks:
            c["topic"] = c.get("topic_hint") or "other"

    build_indices(chunks, index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--tag", action="store_true")
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.index_dir, args.tag)