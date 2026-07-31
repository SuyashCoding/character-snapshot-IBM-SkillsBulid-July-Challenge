"""
extraction.py

Runs the rolling character-sheet extraction over a list of book Units,
in order, using the Gemini API (free tier). Each step feeds the model:
  - the character sheet as it stood after the previous unit
  - the raw text of the new unit
...and asks it to return an updated sheet, plus a short note on what
changed. Results are cached to disk per-book/per-character so re-running
the Streamlit app doesn't re-burn API calls.

Requires: pip install google-genai
Get a free key at https://aistudio.google.com/apikey (no card required).
"""

import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Free-tier model as of July 2026. Google renames/rotates which models sit on
# the free tier fairly often (gemini-2.0-flash, the previous default here,
# was shut down in June 2026) -- if this 404s or you get a "not found" error,
# check https://aistudio.google.com for the current free Flash model name
# and swap it in here. "gemini-flash-latest" is a self-updating alias Google
# maintains specifically so code doesn't break every time they rotate models,
# it's the safest default for exactly this reason.
DEFAULT_MODEL = "gemini-flash-latest"

EMPTY_SHEET = {
    "name": None,
    "aliases": [],          # [{term, first_seen, connotation, note}]
    "physical_description": {"summary": "", "last_updated": None, "notes": []},
    "personality": {"summary": "", "last_updated": None, "notable_shifts": []},
    "relationships": {},    # {other_character: {status, last_updated, note}}
    "knowledge_state": {"knows": [], "does_not_know": [], "last_updated": None},
    "voice": {"notes": "", "sample_lines": []},
    "key_events": [],       # [{unit, event}]
    "dangling_threads": [], # [{planted_at, description, resolved}]
    "trivia": [],
}

SCHEMA_INSTRUCTIONS = """
Return ONLY a single JSON object (no markdown fences, no commentary) matching
this exact shape:

{
  "name": string,
  "aliases": [
    {"term": string, "first_seen": string, "connotation": "affectionate"|"neutral"|"pejorative"|"formal"|"other", "note": string}
  ],
  "physical_description": {"summary": string, "last_updated": string, "notes": [string]},
  "personality": {"summary": string, "last_updated": string, "notable_shifts": [string]},
  "relationships": {
    "<other character name>": {"status": string, "last_updated": string, "note": string}
  },
  "knowledge_state": {"knows": [string], "does_not_know": [string], "last_updated": string},
  "voice": {"notes": string, "sample_lines": [string]},
  "key_events": [{"unit": string, "event": string}],
  "dangling_threads": [{"planted_at": string, "description": string, "resolved": boolean}],
  "trivia": [string],
  "changes_this_unit": [string]
}

Rules:
- This is a ROLLING update. You are given the sheet as it stood before this
  unit, plus the new unit's text. Merge, don't discard: keep everything still
  true, update anything that changed, and add anything new.
- If personality or relationships shift, don't just overwrite the old summary
  silently -- add a short entry to "notable_shifts" or the relevant "note"
  describing the change, so the arc is visible, not just the current state.
- "aliases" should capture every distinct way this character is referred to
  (names, titles, epithets, pronouns used pointedly) with the connotation of
  that specific term as used in THIS unit. If a term already exists from a
  prior unit, keep it in the list; don't duplicate, but you may update its
  note if usage shifted.
- "knowledge_state" is relative to the character's own head, as of THIS unit
  only -- do not include anything that hasn't happened in the text yet.
- "changes_this_unit" is a short bullet list (3-6 items) summarizing what's
  new or different in THIS unit specifically, for a reader who wants a diff.
- Every "last_updated" / "first_seen" / "unit" / "planted_at" field should
  reference the unit label you're given (e.g. "Letter 2", "Chapter 5").
- If the character does not appear at all in this unit, return the sheet
  unchanged except for an empty "changes_this_unit" list.
"""


def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def build_prompt(character_name: str, previous_sheet: dict, unit_label: str, unit_text: str) -> str:
    return f"""You are maintaining a rolling character sheet for "{character_name}"
as a novel is read in order, one unit (letter/chapter) at a time.

{SCHEMA_INSTRUCTIONS}

SHEET BEFORE THIS UNIT:
{json.dumps(previous_sheet, indent=2)}

NEW UNIT: {unit_label}
TEXT OF THIS UNIT:
\"\"\"
{unit_text}
\"\"\"

Return the updated JSON object now."""


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def call_gemini(client: genai.Client, prompt: str, model_name: str = DEFAULT_MODEL, max_retries: int = 5) -> dict:
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            cleaned = _strip_fences(response.text)
            return json.loads(cleaned)
        except genai_errors.ClientError as e:
            # 429 quota-exhausted is not transient within this session --
            # free-tier daily quotas reset on a ~24h cycle, not within the
            # few seconds our backoff waits, so retrying just wastes time.
            # Surface it immediately instead of burning all 5 attempts.
            if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                raise RuntimeError(
                    "Gemini free-tier daily quota exhausted for this model. "
                    "Retrying won't help today. Either switch to a different "
                    "model in the sidebar (lighter models generally carry a "
                    "higher daily cap) and hit Process/Resume again, or wait "
                    "for the quota to reset -- everything processed so far is "
                    "already cached, so resuming picks up right where this "
                    "stopped."
                ) from e
            last_err = e
            time.sleep(delay)
            delay *= 2
        except json.JSONDecodeError as e:
            last_err = e
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            last_err = e
            # Other transient errors (5xx, network blips) -- back off and retry
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


def cache_path_for(cache_dir: str, book_id: str, character_name: str) -> Path:
    safe_char = character_name.replace(" ", "_").lower()
    return Path(cache_dir) / f"{book_id}__{safe_char}.json"


def load_cache(cache_dir: str, book_id: str, character_name: str) -> list[dict]:
    p = cache_path_for(cache_dir, book_id, character_name)
    if p.exists():
        return json.loads(p.read_text())
    return []


def save_cache(cache_dir: str, book_id: str, character_name: str, snapshots: list[dict]) -> None:
    p = cache_path_for(cache_dir, book_id, character_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snapshots, indent=2))


def process_book(
    units,
    character_name: str,
    api_key: str,
    book_id: str = "book",
    cache_dir: str = "cache",
    model_name: str = DEFAULT_MODEL,
    progress_callback=None,
) -> list[dict]:
    """
    Runs (or resumes) the rolling extraction across all units for one
    character. Returns a list of snapshots, one per unit, in order --
    snapshots[i] is the character sheet as it stood right after unit i.
    This list IS the bookmark timeline the Streamlit slider walks through.
    """
    snapshots = load_cache(cache_dir, book_id, character_name)
    start_index = len(snapshots)

    if start_index >= len(units):
        return snapshots  # already fully processed

    client = get_client(api_key)
    previous_sheet = snapshots[-1] if snapshots else dict(EMPTY_SHEET, name=character_name)

    for unit in units[start_index:]:
        prompt = build_prompt(character_name, previous_sheet, unit.label, unit.text)
        updated = call_gemini(client, prompt, model_name=model_name)
        snapshots.append(updated)
        previous_sheet = updated
        save_cache(cache_dir, book_id, character_name, snapshots)  # save as we go
        if progress_callback:
            progress_callback(unit.index + 1, len(units), unit.label)

    return snapshots