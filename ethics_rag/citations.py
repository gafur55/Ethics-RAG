"""
citations.py

Human-readable citation labels for retrieved chunks — shared by the LLM
prompt (generate.py), the CLI, and the web app so all three number and
label sources identically.
"""

import math

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
    meaningful to a reader, so docx sources get no location. Pages are PDF
    page positions, not printed page numbers — the source PDFs don't carry
    reliable page labels.
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
