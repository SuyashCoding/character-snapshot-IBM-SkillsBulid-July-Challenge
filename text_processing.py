"""
text_processing.py

Loads a public-domain book (plain .txt, Gutenberg-formatted) and splits it
into natural bookmark units: Letters and Chapters for Frankenstein, or a
word-count fallback for any other book that doesn't have that structure.

Each unit is the atomic step the extraction pipeline processes in order,
and the atomic step the Streamlit bookmark slider can stop at.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Unit:
    index: int          # 0-based position in the book
    label: str          # e.g. "Letter 1", "Chapter 5"
    text: str           # raw text of just this unit


GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)

# Matches a standalone line like "Letter 1", "Letter I.", "Chapter 12", "Chapter IV"
UNIT_HEADING_RE = re.compile(
    r"^\s*(Letter|Chapter)\s+([IVXLCDM]+|\d+)\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def load_book(path: str) -> str:
    """Read the raw text file off disk."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return text


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Cut the Project Gutenberg license header/footer, keeping just the
    actual novel. If the markers aren't found (e.g. you're using a
    non-Gutenberg source), returns the text unchanged.
    """
    start_match = GUTENBERG_START_RE.search(text)
    end_match = GUTENBERG_END_RE.search(text)

    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(text)

    return text[start_idx:end_idx].strip()


def split_into_units(text: str, fallback_words_per_chunk: int = 3000) -> list[Unit]:
    """
    Split on Letter/Chapter headings. If fewer than 3 are found (book doesn't
    use this convention, or you fed it something other than Frankenstein),
    fall back to fixed-size word chunks so the pipeline still works on any
    plain-text book.
    """
    matches = list(UNIT_HEADING_RE.finditer(text))

    if len(matches) >= 3:
        # Gutenberg texts often have a table of contents up top that also
        # matches "Letter 1" / "Chapter 1" style headings, back to back with
        # no real text between them (or sometimes just a page number). Filter
        # those out by requiring a real amount of prose, not just a truthy
        # string, so a TOC entry like "Chapter 1 ..... 5" doesn't sneak in
        # as a unit. Then re-number indices over the SURVIVING units only --
        # numbering off the raw match position is what caused real content
        # to start at index 28 instead of 0 and break the progress bar.
        candidates = []
        for i, m in enumerate(matches):
            label = f"{m.group(1).title()} {m.group(2)}"
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            if len(body.split()) >= 30:
                candidates.append((label, body))
        return [
            Unit(index=idx, label=label, text=body)
            for idx, (label, body) in enumerate(candidates)
        ]

    # Fallback: fixed-size chunks by word count
    words = text.split()
    units = []
    for i in range(0, len(words), fallback_words_per_chunk):
        chunk = " ".join(words[i : i + fallback_words_per_chunk])
        units.append(Unit(index=len(units), label=f"Section {len(units) + 1}", text=chunk))
    return units


def load_and_split(path: str) -> list[Unit]:
    """Convenience wrapper: load, strip boilerplate, split into units."""
    raw = load_book(path)
    cleaned = strip_gutenberg_boilerplate(raw)
    return split_into_units(cleaned)