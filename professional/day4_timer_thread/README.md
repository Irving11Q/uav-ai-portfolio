# Day 4: QTimer Periodic Refresh & QThread Background Task

Demo of two PySide6 fundamentals that keep a desktop app responsive: periodic refresh with `QTimer`, and long-running work offloaded to `QThread` with progress reported back via a `Signal`.

## What's inside

- `day4_timer_thread.py` — a window with:
  - a counter label incremented every second by a `QTimer`;
  - a toggle button to start/stop the timer;
  - a background `QThread` worker counting to 100, emitting progress to the main thread without freezing the UI.

## Requirements

- Python 3.12
- PySide6 >= 6.0

```bash
pip install PySide6
```

## Run

```bash
python day4_timer_thread.py
```

Click **Start periodic refresh** to see the counter tick every second (QTimer). Then click **Start background task** — the progress label counts to 100 while the window stays fully responsive (QThread + Signal).

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.
