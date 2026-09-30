"""
extractors.py

Converts heterogeneous source files (pptx, docx, pdf) into a single common
schema so downstream chunking/tagging/embedding doesn't need to know what
format the content originally came from.

Common unit schema (one dict per paragraph/slide/page-block):
{
    "text": str,
    "source_path": str,          # original file path
    "source_title": str,         # human-readable title, e.g. filename without ext
    "source_type": str,          # "curriculum_pptx" | "curriculum_docx" | "textbook" | "journal_article" | "book"
    "unit_type": str,            # "slide" | "paragraph" | "page"
    "unit_index": int,           # slide number / paragraph index / page number
    "section_heading": str|None, # best-effort nearest heading, if detectable
    "topic_hint": str|None,      # topic taken from folder structure, if known —
                                 # see pipeline.py's "Relevant Literature" handling
}

You are expected to set `source_type` per-file (or per-folder) based on where
the file came from — see pipeline.py for how that gets assigned. `topic_hint`
is optional: pass it when the folder structure already tells you the topic
(e.g. a "Surrogate Decision Making" literature subfolder) so downstream
tagging can trust it instead of re-inferring topic from scratch.
"""

from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------

def extract_pptx(path: str, source_type: str, topic_hint: Optional[str] = None) -> List[Dict]:
    from pptx import Presentation

    prs = Presentation(path)
    title = Path(path).stem
    units = []

    for slide_idx, slide in enumerate(prs.slides, start=1):
        slide_texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        slide_texts.append(line)

        slide_body = "\n".join(slide_texts).strip()
        if slide_body:
            units.append({
                "text": slide_body,
                "source_path": path,
                "source_title": title,
                "source_type": source_type,
                "unit_type": "slide",
                "unit_index": slide_idx,
                "section_heading": slide_texts[0] if slide_texts else None,
                "topic_hint": topic_hint,
            })

        # Speaker notes carry a lot of the actual reasoning — keep as a
        # separate unit rather than merging into the slide body, since notes
        # tend to be prose while slide text is bullet fragments.
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()
            if notes_text:
                units.append({
                    "text": notes_text,
                    "source_path": path,
                    "source_title": title,
                    "source_type": source_type,
                    "unit_type": "slide_notes",
                    "unit_index": slide_idx,
                    "section_heading": slide_texts[0] if slide_texts else None,
                    "topic_hint": topic_hint,
                })

    return units


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def extract_docx(path: str, source_type: str, topic_hint: Optional[str] = None) -> List[Dict]:
    import docx

    doc = docx.Document(path)
    title = Path(path).stem
    units = []
    current_heading = None
    para_idx = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style_name = (para.style.name or "").lower()
        if style_name.startswith("heading") or style_name == "title":
            current_heading = text
            continue  # headings become metadata, not standalone chunks

        para_idx += 1
        units.append({
            "text": text,
            "source_path": path,
            "source_title": title,
            "source_type": source_type,
            "unit_type": "paragraph",
            "unit_index": para_idx,
            "section_heading": current_heading,
            "topic_hint": topic_hint,
        })

    # Tables often hold the actual reference material (e.g. state-by-state
    # comparisons) — extract them as their own units, row-joined.
    for t_idx, table in enumerate(doc.tables, start=1):
        rows_text = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows_text.append(" | ".join(cells))
        if rows_text:
            units.append({
                "text": "\n".join(rows_text),
                "source_path": path,
                "source_title": title,
                "source_type": source_type,
                "unit_type": "table",
                "unit_index": t_idx,
                "section_heading": current_heading,
                "topic_hint": topic_hint,
            })

    return units


# ---------------------------------------------------------------------------
# PDF (papers, book chapters, books)
# ---------------------------------------------------------------------------

# Front/back matter entries in a book's table of contents — pages under these
# are dropped rather than indexed, since they'd only add noise to retrieval.
SKIP_TOC_TITLES = {"contents", "index", "contributors", "about the editors", "about the authors"}


def _chapter_by_page(doc) -> Dict[int, str]:
    """
    Maps page number -> chapter title using the PDF's embedded table of
    contents (books have one; most papers don't, in which case this returns
    {} and section_heading stays None). Uses TOC level 1 as the chapter
    level, unless level 1 is "Part I/II/..." groupings, in which case the
    chapters are one level down.
    """
    toc = doc.get_toc()
    if not toc:
        return {}
    chapter_level = 2 if any(t[1].strip().lower().startswith("part ") for t in toc if t[0] == 1) else 1

    entries = sorted(
        [(page, " ".join(title.split())) for level, title, page in toc if level <= chapter_level and page > 0],
        key=lambda e: e[0],
    )
    mapping, current = {}, None
    for page_idx in range(1, doc.page_count + 1):
        for page, title in entries:
            if page == page_idx:
                current = title
        mapping[page_idx] = current
    return mapping


def extract_pdf(path: str, source_type: str, topic_hint: Optional[str] = None) -> List[Dict]:
    """
    Uses PyMuPDF (fitz). Chosen over pdfplumber/pypdf because it handles
    multi-column academic layouts noticeably better via `get_text("blocks")`,
    which returns text blocks in reading order rather than raw stream order.
    """
    import fitz  # pymupdf

    doc = fitz.open(path)
    title = Path(path).stem
    units = []
    chapters = _chapter_by_page(doc)

    for page_idx, page in enumerate(doc, start=1):
        chapter = chapters.get(page_idx)
        if chapter and chapter.lower() in SKIP_TOC_TITLES:
            continue

        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        # Sort top-to-bottom, then left-to-right — approximates correct
        # reading order for 2-column layouts better than raw stream order.
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1], 0), b[0]))

        page_text_parts = []
        for b in blocks_sorted:
            text = b[4].strip()
            if text:
                page_text_parts.append(text)

        page_text = "\n\n".join(page_text_parts).strip()
        if page_text:
            units.append({
                "text": page_text,
                "source_path": path,
                "source_title": title,
                "source_type": source_type,
                "unit_type": "page",
                "unit_index": page_idx,
                "section_heading": chapter,  # from the PDF's TOC when present (books), else None
                "topic_hint": topic_hint,
            })

    doc.close()
    return units


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

EXTENSION_MAP = {
    ".pptx": extract_pptx,
    ".docx": extract_docx,
    ".pdf": extract_pdf,
}


def extract_file(path: str, source_type: str, topic_hint: Optional[str] = None) -> List[Dict]:
    ext = Path(path).suffix.lower()
    if ext not in EXTENSION_MAP:
        raise ValueError(f"Unsupported file type: {ext} ({path})")
    return EXTENSION_MAP[ext](path, source_type, topic_hint)