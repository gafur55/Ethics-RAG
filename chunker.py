"""
chunker.py

Groups the common-schema units from extractors.py into retrieval-sized
chunks. Uses a simple word-count proxy for tokens (good enough to start —
swap in a real tokenizer later if you want tighter control over embedding
model context limits).

Design choices worth knowing about:
- Chunks never span two source files, and by default never span two
  section_headings, so a chunk doesn't mix unrelated content.
- unit_type == "table" and "slide_notes" are NOT merged with neighboring
  units — tables and notes tend to be dense/self-contained and merging
  them with prose neighbors muddies both.
- Overlap is applied only within a merge-run, not across section boundaries.
- Units larger than chunk_words on their own (e.g. a dense two-column PDF
  page, easily 700-1000+ words) get pre-split into smaller paragraph-based
  pieces BEFORE grouping/merging. Without this, a single oversized unit
  sails through untouched, and the overlap carryover then drags that whole
  oversized unit into the NEXT chunk too — which is why early runs produced
  multi-page chunks with references/acknowledgments bundled in, roughly
  double the intended size.
"""

from typing import List, Dict
import itertools

DEFAULT_CHUNK_WORDS = 350
DEFAULT_OVERLAP_WORDS = 50
NO_MERGE_TYPES = {"table", "slide_notes"}


def _word_count(text: str) -> int:
    return len(text.split())


def _split_oversized_unit(unit: Dict, max_words: int) -> List[Dict]:
    """
    Splits a single unit's text down to paragraph-level pieces, so nothing
    entering the merge/overlap loop below is ever much larger than a
    paragraph. This is deliberately fine-grained rather than pre-grouped —
    the merge loop already knows how to combine small pieces up to
    chunk_words with a clean overlap tail; it just can't do that job if the
    pieces handed to it are already chunk_words-sized themselves (a full
    paragraph-grouped piece is too coarse for the overlap window to trim
    meaningfully). Preserves all metadata; gives each piece a fractional
    unit_index (e.g. page 4 -> 4.0, 4.1, 4.2) so unit_range in the final
    chunk still reads sensibly.
    """
    words = unit["text"].split()
    if len(words) <= max_words:
        return [unit]

    paragraphs = [p for p in unit["text"].split("\n\n") if p.strip()]
    pieces = []
    for para in paragraphs:
        para_words = para.split()
        if len(para_words) > max_words:
            # a single paragraph alone exceeds the cap — hard-split by words
            for i in range(0, len(para_words), max_words):
                pieces.append(" ".join(para_words[i:i + max_words]))
        else:
            pieces.append(para)

    return [
        {**unit, "text": piece, "unit_index": unit["unit_index"] + i / 100}
        for i, piece in enumerate(pieces)
    ]


def chunk_units(
    units: List[Dict],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> List[Dict]:
    """
    units: output of extractors.extract_file (already grouped per source file)
    Returns a list of chunk dicts:
    {
        "text": str,
        "source_path": str,
        "source_title": str,
        "source_type": str,
        "section_heading": str|None,
        "unit_range": [start_idx, end_idx],   # e.g. slide/page/paragraph range covered
        "unit_type": str,
    }
    """
    chunks = []

    # Pre-split any oversized unit BEFORE grouping, so nothing entering the
    # merge loop below can exceed chunk_words on its own.
    split_units = []
    for u in units:
        if u["unit_type"] in NO_MERGE_TYPES:
            split_units.append(u)  # tables/notes are exempt — kept whole deliberately
        else:
            split_units.extend(_split_oversized_unit(u, chunk_words))
    units = split_units

    # Group by (source_path, section_heading) so merging never crosses a
    # heading boundary. Units with no heading form their own runs by
    # contiguous unit_type instead.
    def group_key(u):
        return (u["source_path"], u.get("section_heading"), u["unit_type"])

    for (source_path, heading, unit_type), group_iter in itertools.groupby(
        sorted(units, key=lambda u: (u["source_path"], str(u.get("section_heading")), u["unit_index"])),
        key=group_key,
    ):
        group = list(group_iter)

        if unit_type in NO_MERGE_TYPES:
            # keep these as their own chunks, one per unit, no merging
            for u in group:
                chunks.append(_make_chunk([u], source_path, heading, unit_type))
            continue

        buffer, buffer_words = [], 0
        for u in group:
            u_words = _word_count(u["text"])
            if buffer and buffer_words + u_words > chunk_words:
                chunks.append(_make_chunk(buffer, source_path, heading, unit_type))
                # Overlap: a bounded synthetic fragment containing just the
                # last overlap_words words of the emitted chunk, not whole
                # units. This is deliberately word-sliced rather than
                # unit-based — if we instead tried to carry whole units
                # forward, a single ~350-word unit (very possible even
                # post-split, e.g. a hard-split PDF block) would blow the
                # overlap budget by 7x and re-inflate the next chunk right
                # back toward double size, which is exactly what happened
                # before this fix (chunks landing at a suspicious, constant
                # ~700 words — 2x chunk_words — across the corpus).
                combined_words = " ".join(pu["text"] for pu in buffer).split()
                overlap_text = " ".join(combined_words[-overlap_words:])
                buffer = [{**buffer[-1], "text": overlap_text}] if overlap_text else []
                buffer_words = _word_count(overlap_text) if overlap_text else 0

            buffer.append(u)
            buffer_words += u_words

        if buffer:
            chunks.append(_make_chunk(buffer, source_path, heading, unit_type))

    return chunks


def _make_chunk(unit_group: List[Dict], source_path: str, heading, unit_type: str) -> Dict:
    first, last = unit_group[0], unit_group[-1]
    return {
        "text": "\n\n".join(u["text"] for u in unit_group),
        "source_path": source_path,
        "source_title": first["source_title"],
        "source_type": first["source_type"],
        "section_heading": heading,
        "unit_type": unit_type,
        "unit_range": [first["unit_index"], last["unit_index"]],
        "topic_hint": first.get("topic_hint"),
        "source_year": first.get("source_year"),
        "source_url": first.get("source_url"),
    }