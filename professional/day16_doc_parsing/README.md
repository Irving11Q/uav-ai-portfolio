# Document Parsing & Advanced Chunking

Day 16 of the "UAV AI Application Development" series. Upgrades the RAG pipeline from Day 15 by replacing the hard-coded manual with a **real markdown file**, then comparing **three chunking strategies** to see which one retrieves better.

## What's inside

- `day16_doc_parsing.py` — a self-contained script that:
  - auto-generates a sample manual file `uav_battery_manual.md` (7 sections with `##` headings, ready to be replaced by your own real document);
  - parses the file with `open()` (handles encoding, `\r\n` normalization, empty-line cleaning);
  - chunks the text with **three strategies**:
    - `chunk_by_paragraph` — split on blank lines, keeps semantics but uneven size;
    - `chunk_by_window` — fixed-size windows with overlap, uniform but may cut sentences;
    - `chunk_by_heading` — split on `##` headings, keeps chapter structure (recommended);
  - verifies retrieval quality with character-frequency vectors + cosine similarity (a minimal self-contained replica of Day 15's retrieval);
  - persists the best chunks to `chunks_parsed.json` (id / text / source / strategy) — the input for the Day 17 vector database.

## Requirements

- Python 3.12 (standard library only — no extra packages)

## Setup

No API key needed. Everything runs offline.

## Run

From this folder (`professional/day16_doc_parsing/`):

```bash
python day16_doc_parsing.py
```

The program prints: parse stats → chunk stats of the three strategies → retrieval hit for the question "电池温度太高会怎样？" → saves `chunks_parsed.json`.

## Key pattern

**Chunking by heading** — each markdown heading starts a new chunk, keeping the title with its body so hits are self-explanatory:

```python
def chunk_by_heading(text):
    chunks, current = [], []
    for line in text.split("\n"):
        if re.match(r"^#{1,3}\s", line):   # a heading line
            if current:
                chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return chunks
```

**Persisting chunks** — every record carries `id`, `text`, `source` and `chunk_strategy`, so the downstream vector DB can ingest it directly and answers stay traceable:

```python
records = [
    {"id": i, "text": c, "source": source, "chunk_strategy": strategy}
    for i, c in enumerate(chunks)
]
json.dump(records, open("chunks_parsed.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
```

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI System**. Full roadmap: see repo root `README.md`.
