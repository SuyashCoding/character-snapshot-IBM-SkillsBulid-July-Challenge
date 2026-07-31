"""
extraction.py

Runs the rolling character-sheet extraction over a list of book Units,
in order, using IBM watsonx.ai (Granite). Each step feeds the model:
  - the character sheet as it stood after the previous unit
  - the raw text of the new unit
...and asks it to return an updated sheet, plus a short note on what
changed. Results are cached to disk per-book/per-character so re-running
the Streamlit app doesn't re-burn API calls.

Requires: pip install ibm-watsonx-ai
You'll need three things from IBM Cloud, all free on the Lite plan:
  1. An IAM API key: https://cloud.ibm.com/iam/apikeys
  2. A watsonx.ai project ID: create/open a project at https://dataplatform.cloud.ibm.com/wx,
     then Project -> Manage -> General -> Details.
  3. The region your project lives in (Dallas/us-south is the default for new
     Lite-plan projects unless you picked something else at creation).
"""

import json
import time
from pathlib import Path

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure

# ibm/granite-3-3-8b-instruct is the current stable mid-size Granite instruct
# model on watsonx.ai as of mid-2026 -- ibm/granite-4-h-small is newer and
# worth trying too, but 3-3-8b has been out longer and is the safer default
# for an extraction pipeline that needs to hold a JSON shape reliably across
# 20+ chained calls. IBM does deprecate older Granite versions over time
# (granite-3-8b-instruct, one generation back, already carries a deprecated
# flag) -- if this 404s, check the current list with:
#   api_client.foundation_models.ChatModels
# or browse https://dataplatform.cloud.ibm.com/wx/samples for what's live
# in your project's region, and swap the name in here or the sidebar field.
DEFAULT_MODEL = "ibm/granite-3-3-8b-instruct"

# watsonx.ai is regional -- your project lives in exactly one of these, and
# calls have to go to the matching endpoint or you'll get a 404. Find yours
# under Project -> Manage -> General -> Details -> "Region".
WATSONX_REGIONS = {
    "Dallas (us-south)": "https://us-south.ml.cloud.ibm.com",
    "Frankfurt (eu-de)": "https://eu-de.ml.cloud.ibm.com",
    "London (eu-gb)": "https://eu-gb.ml.cloud.ibm.com",
    "Tokyo (jp-tok)": "https://jp-tok.ml.cloud.ibm.com",
    "Toronto (ca-tor)": "https://ca-tor.ml.cloud.ibm.com",
}
DEFAULT_REGION = "Dallas (us-south)"

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


def get_client(api_key: str, project_id: str, model_name: str = DEFAULT_MODEL, url: str = WATSONX_REGIONS[DEFAULT_REGION]) -> ModelInference:
    credentials = Credentials(url=url, api_key=api_key)
    params = TextChatParameters(
        temperature=0.3,
        max_completion_tokens=8000,
        # Granite 3.3 supports native JSON-object mode through the chat API,
        # same role response_mime_type played for Gemini. Kept alongside the
        # prompt-level instruction and _strip_fences below as a second layer,
        # since json_object mode still occasionally wraps output in fences.
        response_format={"type": "json_object"},
    )
    return ModelInference(
        model_id=model_name,
        credentials=credentials,
        project_id=project_id,
        params=params,
    )


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


def call_watsonx(model: ModelInference, prompt: str, max_retries: int = 5) -> dict:
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            response = model.chat(messages=[{"role": "user", "content": prompt}])
            raw_text = response["choices"][0]["message"]["content"]
            cleaned = _strip_fences(raw_text)
            return json.loads(cleaned)
        except ApiRequestFailure as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                raise RuntimeError(
                    "watsonx.ai rejected the credentials (401/403). Regenerate "
                    "the API key at https://cloud.ibm.com/iam/apikeys and make "
                    "sure it belongs to the same IBM Cloud account as the "
                    "project ID entered in the sidebar."
                ) from e
            if status == 404:
                raise RuntimeError(
                    "watsonx.ai returned 404 -- either the project ID is wrong "
                    "(check Project -> Manage -> General -> Details) or the "
                    "selected region doesn't match where that project actually "
                    "lives, or the model isn't enabled in that region."
                ) from e
            # Unlike Gemini's hard daily free-tier quota, a watsonx.ai 429 is
            # a rolling per-instance requests-per-second throttle (its docs
            # cite ~30 RPS on typical instances) that clears within seconds,
            # so backing off and retrying is the right move here, not failing
            # fast the way the Gemini quota case did.
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
    raise RuntimeError(f"watsonx.ai call failed after {max_retries} attempts: {last_err}")


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
    project_id: str,
    book_id: str = "book",
    cache_dir: str = "cache",
    model_name: str = DEFAULT_MODEL,
    url: str = WATSONX_REGIONS[DEFAULT_REGION],
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

    model = get_client(api_key, project_id, model_name, url)
    previous_sheet = snapshots[-1] if snapshots else dict(EMPTY_SHEET, name=character_name)

    for unit in units[start_index:]:
        prompt = build_prompt(character_name, previous_sheet, unit.label, unit.text)
        updated = call_watsonx(model, prompt)
        snapshots.append(updated)
        previous_sheet = updated
        save_cache(cache_dir, book_id, character_name, snapshots)  # save as we go
        if progress_callback:
            progress_callback(unit.index + 1, len(units), unit.label)

    return snapshots