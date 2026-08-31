# Document Parsing & Advanced Chunking

Day 16 of the "UAV AI Application Development" series. Replaces Day15's hard-coded manual with a **real file**, then compares three chunking strategies to show how chunk quality drives retrieval quality — the foundation of any RAG system.

## What's inside

- `day16_doc_parsing.py` — a fully offline (no API key) script that:
  - generates a sample UAV battery manual (`uav_battery_manual.md`) if missing, then reads it back with proper encoding;
  - chunks it three ways:
    - **A — by paragraph** (`\n\n` split): semantically whole, but uneven and titles detach from body;
    - **B — fixed window** (60 chars + 15 overlap): uniform length, may cut a sentence, overlap mitigates;
    - **C — by heading** (`##` sections): keeps chapter structure, each chunk carries its title → friendliest for retrieval;
  - prints block-count / average-length stats for each strategy;
  - reuses Day15's `search()` to verify which chunking hits the right section for *"电池温度太高会怎样？"*;
  - persists the **heading-based** chunks to `chunks_parsed.json` (with id / text / source / strategy) as input for Day17 (Chroma).
  - depends on `day15_rag/day15_rag_basics.py` (reused for the retrieval check); degrades to "chunking only" if Day15 isn't found.
- `uav_battery_manual.md` — the sample manual used as input.
- `chunks_parsed.json` — the committed chunking output (what Day17 will ingest).

## Requirements

- Python 3.12 (Day16 itself uses only the standard library — `os`, `json`, `re`)
- `requests` — **only** needed for the optional retrieval check in section ③, because it reuses Day15's `search()`. Without `requests` the script still runs and does everything except the verification line.

```bash
pip install requests
```

## Run

From this folder (`professional/day16_doc_parsing/`):

```bash
python day16_doc_parsing.py
```

It writes `uav_battery_manual.md` and `chunks_parsed.json` next to itself.

## Key pattern

**Three chunking strategies, one trade-off space** — length uniformity vs semantic integrity:

| Strategy | Pros | Cons | Best for |
|---|---|---|---|
| By paragraph | semantically whole | uneven size; title/body split | clean docs already split well |
| Fixed window | uniform size, controllable | may cut a sentence | huge flat text |
| By heading | keeps structure, chunk carries title | needs heading markup | manuals / docs / wikis |

**Persist chunks for reuse** — never re-parse every run:

```python
json.dump(
    [{"id": i, "text": c, "source": src, "chunk_strategy": strat} for i, c in enumerate(chunks)],
    f, ensure_ascii=False, indent=2,
)
```

> Note: real systems usually **combine** strategies — split by heading first, then re-split any over-long chapter by window. Heading chunking wins here because the retrieved chunk already tells you *which* section it came from.

## Why it matters

Retrieval quality is capped by chunking quality. A great embedding model still fails if chunks are cut badly. Learning to choose and combine chunking strategies is what separates a toy RAG from a production one — and it's the data-prep step before Day17's vector database.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI System**. Full roadmap: see repo root `README.md`.
