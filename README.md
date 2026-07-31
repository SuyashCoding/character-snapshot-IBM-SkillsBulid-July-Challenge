# Character Snapshot

Reload a character exactly as they stood the day you stopped writing.

Feed it a book, pick a character, drag a slider to the exact chapter where you left off. It hands you back the character sheet as it stood at that point: no plot from later chapters leaking backward, no whole-book average that blurs who they were on the page you actually stopped at.

## The problem

Come back to a manuscript after two weeks away and you've lost the character, not the plot. Does she know about the affair yet, at the point the draft stopped? Which of her three names is anyone using right now, and does she still trust the person she was starting to open up to a page ago? Rereading from the top to reload that costs the one thing a returning writer is short on: momentum. A whole-book summary makes it worse, not better, because it either flattens the character into who they end up as by the final page or averages across the whole arc and loses the exact version that existed where the draft stopped. Character bibles fail for a related reason: they need manual upkeep every time something shifts, and updating them is the first thing that gets skipped when the writing stalls.

## How it works

The book gets split into units (Letters and Chapters for a Gutenberg-formatted text; a 3,000-word fallback chunk for anything without that structure) and processed in order. Each unit is handed to Granite, via IBM watsonx.ai, along with the character sheet as it stood *before* that unit, and the model is asked to revise it rather than regenerate it: keep what's still true, update what changed, note anything new. That revision instruction is the whole trick. It's what keeps a personality shift or a broken relationship from silently overwriting the old read, and it's what makes the output a real timeline instead of 24 disconnected snapshots.

Each unit's result is written to disk as soon as it's produced (`cache/<book_id>__<character>.json`), so a throttling error or a killed process doesn't cost the work already done. Hitting Process again resumes from the last cached unit instead of starting over.

## What you actually see

Eight tabs per snapshot: overview, aliases/epithets, relationships, knowledge state, voice, key events, dangling threads, trivia. The aliases tab is the one that earns its keep, since it tracks not just what a character is called but the judgment baked into each term, tagged and dated. Running the pipeline on Frankenstein against "the creature," Letter 4 picks up "dæmon" as a pejorative the moment the stranger first mentions him, alongside the flatly neutral "traveller." By the point Justine is executed, the changes-this-unit log for that chapter reads:

> Victor uses new epithets for the creature, referring to him as 'fiend', 'monster', and 'destroyer'.

That's the arc becoming visible in the data instead of just in your memory of having read it.

## Setup

```
pip install -r requirements.txt
```

You'll need three things from IBM Cloud, all free on the Lite plan:

1. An IAM API key: https://cloud.ibm.com/iam/apikeys
2. A watsonx.ai project ID, from a project you create at https://dataplatform.cloud.ibm.com/wx (Project -> Manage -> General -> Details)
3. The region that project lives in (Dallas/us-south by default)

```
streamlit run app.py
```

## Usage

1. Paste your IBM Cloud API key and watsonx.ai project ID into the sidebar, and pick the region your project lives in.
2. Upload a public-domain `.txt` book, or drop one at `data/frankenstein.txt` and leave the uploader empty.
3. Enter the character to track (aliases are picked up automatically, so "the creature" or "Victor Frankenstein" both work as anchors).
4. Hit **Process / Resume**. It runs chapter by chapter and caches as it goes; if you stop partway or hit a throttling error, hitting the button again picks up where it left off.
5. Drag the bookmark slider to wherever you stopped writing.

## A note on the model

Default model is `ibm/granite-3-3-8b-instruct`. `ibm/granite-4-h-small` is newer and worth trying, but 3.3 has more runway behind it for holding a JSON shape reliably across a chain of 20+ calls where each one builds on the last. IBM does retire older Granite versions over time, so if the default 404s, check what's currently enabled for your project's region under Project -> Manage -> Foundation models and swap the name into the sidebar field.

Region matters as much as model name here: watsonx.ai calls have to hit the endpoint your project actually lives in, so a wrong region selection in the sidebar also shows up as a 404, not just a bad model name. Rate limiting (HTTP 429) on watsonx.ai is a rolling per-instance requests-per-second cap rather than a hard daily quota, so unlike a free-tier daily limit, it clears within seconds and the app backs off and retries automatically instead of failing immediately.

## Project structure

```
app.py             Streamlit UI
extraction.py       Rolling extraction loop, watsonx.ai calls, disk cache
text_processing.py  Book loading, Gutenberg boilerplate stripping, unit splitting
requirements.txt
cache/              Generated per book/character on first run
```

## Limitations

One character per run. Plain-text public-domain books only, Gutenberg-formatted or not. Books without Letter/Chapter headings fall back to fixed-size word chunks, so bookmarks land on "Section 4" instead of a chapter name. No cross-character relationship view beyond what's captured in each tracked character's own `relationships` field.
