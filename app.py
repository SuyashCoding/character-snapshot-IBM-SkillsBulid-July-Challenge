"""
Character Snapshot -- a tool for writers picking a book back up after time away.

Feed it a book. Pick a character. Drag a slider to the exact chapter where
you stopped. Get the character exactly as they stood at that point: no
foreshadowing of what hasn't happened, no flattened whole-book summary,
just a save-state.

Run: streamlit run app.py
"""

import tempfile
import time
from pathlib import Path

import streamlit as st

from text_processing import load_book, strip_gutenberg_boilerplate, split_into_units
from extraction import process_book, DEFAULT_MODEL, WATSONX_REGIONS, DEFAULT_REGION

st.set_page_config(page_title="Character Snapshot", page_icon="🔖", layout="wide")

CONNOTATION_COLORS = {
    "affectionate": "#3a7d44",
    "neutral": "#5c5c5c",
    "pejorative": "#a63d40",
    "formal": "#3f5f8a",
    "other": "#8a6d3f",
}

# ---------------------------------------------------------------------------
# Sidebar: setup
# ---------------------------------------------------------------------------
st.sidebar.title("Setup")

api_key = st.sidebar.text_input(
    "IBM Cloud API key",
    type="password",
    help="Free on the Lite plan, no card required: https://cloud.ibm.com/iam/apikeys",
)
project_id = st.sidebar.text_input(
    "watsonx.ai project ID",
    help="From your project: Manage tab -> General -> Details. "
         "Create a project at https://dataplatform.cloud.ibm.com/wx if you don't have one.",
)
region_label = st.sidebar.selectbox(
    "Region",
    options=list(WATSONX_REGIONS.keys()),
    index=list(WATSONX_REGIONS.keys()).index(DEFAULT_REGION),
    help="Must match the region your project was created in, or calls will 404.",
)
watsonx_url = WATSONX_REGIONS[region_label]
model_name = st.sidebar.text_input(
    "Model name",
    value=DEFAULT_MODEL,
    help="If this 404s, the model may not be enabled in your project's region -- "
         "check what's available under Project -> Manage -> Foundation models.",
)

st.sidebar.markdown("---")

default_book_path = Path("data/frankenstein.txt")
uploaded = st.sidebar.file_uploader("Upload a public-domain .txt book", type=["txt"])

book_path = None
book_id = None
if uploaded is not None:
    tmp = Path(tempfile.gettempdir()) / uploaded.name
    tmp.write_bytes(uploaded.getvalue())
    book_path = tmp
    book_id = uploaded.name
elif default_book_path.exists():
    book_path = default_book_path
    book_id = "frankenstein"
    st.sidebar.caption(f"Using {default_book_path}")
else:
    st.sidebar.warning(
        "No book found. Upload a .txt file, or drop one at data/frankenstein.txt.\n\n"
        "Frankenstein (public domain): https://www.gutenberg.org/ebooks/84 -- "
        "grab the 'Plain Text UTF-8' link."
    )

character_name = st.sidebar.text_input(
    "Character to track",
    value="the creature",
    help='Try "the creature" or "Victor Frankenstein". Use whatever term you want as the anchor -- aliases get tracked automatically.',
)

process_clicked = st.sidebar.button("Process / Resume", type="primary", disabled=not (api_key and project_id and book_path))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Processing runs chapter by chapter and caches to disk as it goes. "
    "If you stop partway or hit a rate limit, just hit the button again to resume."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🔖 Character Snapshot")
st.caption("Not \"what happens to this character.\" What did I know about them the day I stopped writing.")

if "units" not in st.session_state:
    st.session_state.units = None
if "snapshots" not in st.session_state:
    st.session_state.snapshots = None

if book_path and st.session_state.units is None:
    raw = load_book(str(book_path))
    cleaned = strip_gutenberg_boilerplate(raw)
    st.session_state.units = split_into_units(cleaned)

if process_clicked and book_path and api_key and project_id and character_name:
    progress_bar = st.progress(0.0)
    status = st.empty()

    def on_progress(done, total, label):
        progress_bar.progress(done / total)
        status.text(f"Processed {label} ({done}/{total})")

    try:
        snapshots = process_book(
            st.session_state.units,
            character_name=character_name,
            api_key=api_key,
            project_id=project_id,
            book_id=book_id,
            cache_dir="cache",
            model_name=model_name,
            url=watsonx_url,
            progress_callback=on_progress,
        )
        st.session_state.snapshots = snapshots
        status.text(f"Done. {len(snapshots)} units processed.")
    except Exception as e:
        st.error(f"Extraction failed: {e}")

# Try loading from cache even without clicking the button, so re-opening
# the app doesn't force a re-process.
if st.session_state.snapshots is None and book_id and character_name:
    from extraction import load_cache
    cached = load_cache("cache", book_id, character_name)
    if cached:
        st.session_state.snapshots = cached

# ---------------------------------------------------------------------------
# Bookmark view
# ---------------------------------------------------------------------------
snapshots = st.session_state.snapshots
units = st.session_state.units

if not snapshots:
    st.info("Enter an API key, load a book, and hit **Process / Resume** to get started.")
else:
    max_idx = len(snapshots) - 1
    labels = [u.label for u in units[: len(snapshots)]]

    bookmark_idx = st.select_slider(
        "Your bookmark -- where did you stop?",
        options=list(range(max_idx + 1)),
        value=max_idx,
        format_func=lambda i: labels[i],
    )

    sheet = snapshots[bookmark_idx]
    st.subheader(f"{sheet.get('name') or character_name} — as of {labels[bookmark_idx]}")

    if sheet.get("changes_this_unit"):
        with st.expander(f"What's new since the last unit", expanded=True):
            for change in sheet["changes_this_unit"]:
                st.markdown(f"- {change}")

    tabs = st.tabs([
        "Overview", "Aliases / Epithets", "Relationships",
        "Knowledge state", "Voice", "Key events", "Dangling threads", "Trivia",
    ])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Physical description**")
            st.write(sheet["physical_description"]["summary"] or "_Nothing established yet._")
            if sheet["physical_description"]["notes"]:
                for n in sheet["physical_description"]["notes"]:
                    st.caption(f"• {n}")
        with col2:
            st.markdown("**Personality**")
            st.write(sheet["personality"]["summary"] or "_Nothing established yet._")
            if sheet["personality"]["notable_shifts"]:
                st.markdown("_Notable shifts so far:_")
                for s in sheet["personality"]["notable_shifts"]:
                    st.caption(f"• {s}")

    with tabs[1]:
        st.caption("How this character has been referred to, and the judgment baked into each term.")
        aliases = sheet.get("aliases", [])
        if not aliases:
            st.write("_None tracked yet._")
        for a in aliases:
            color = CONNOTATION_COLORS.get(a.get("connotation", "neutral"), "#5c5c5c")
            st.markdown(
                f"<span style='background-color:{color};color:white;padding:2px 8px;"
                f"border-radius:10px;font-size:0.85em'>{a.get('connotation','')}</span> "
                f"**{a.get('term','')}** — first seen {a.get('first_seen','?')}"
                f"<br><span style='color:#888'>{a.get('note','')}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")

    with tabs[2]:
        rels = sheet.get("relationships", {})
        if not rels:
            st.write("_None tracked yet._")
        for other, info in rels.items():
            st.markdown(f"**{other}** — {info.get('status','')}")
            st.caption(f"{info.get('note','')} (as of {info.get('last_updated','?')})")

    with tabs[3]:
        ks = sheet.get("knowledge_state", {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Knows**")
            for k in ks.get("knows", []) or ["_Nothing tracked yet._"]:
                st.write(f"- {k}")
        with col2:
            st.markdown("**Does not know**")
            for k in ks.get("does_not_know", []) or ["_Nothing tracked yet._"]:
                st.write(f"- {k}")

    with tabs[4]:
        voice = sheet.get("voice", {})
        st.write(voice.get("notes") or "_Nothing established yet._")
        if voice.get("sample_lines"):
            st.markdown("_Sample lines:_")
            for line in voice["sample_lines"]:
                st.markdown(f"> {line}")

    with tabs[5]:
        events = sheet.get("key_events", [])
        if not events:
            st.write("_None yet._")
        for e in events:
            st.markdown(f"**{e.get('unit','')}** — {e.get('event','')}")

    with tabs[6]:
        threads = sheet.get("dangling_threads", [])
        if not threads:
            st.write("_None yet._")
        for t in threads:
            status = "✅ resolved" if t.get("resolved") else "🧵 still open"
            st.markdown(f"**{t.get('planted_at','')}** — {t.get('description','')} · {status}")

    with tabs[7]:
        trivia = sheet.get("trivia", [])
        if not trivia:
            st.write("_None yet._")
        for t in trivia:
            st.markdown(f"- {t}")