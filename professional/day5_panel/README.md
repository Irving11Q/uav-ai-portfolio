# UAV Flight Parameter Monitoring Panel

A PySide6 desktop panel that refreshes a set of simulated UAV flight parameters every second, with data generated on a background thread so the UI stays responsive.

## What's inside

- `day5_panel.py` — a window with:
  - a table showing 4 UAV flight parameters (battery voltage, motor current, body temperature, flight altitude);
  - a 1-second `QTimer` driving periodic refresh;
  - a background `QThread` sensor that generates readings and reports them back via a `Signal`.

## Requirements

- Python 3.12
- PySide6 >= 6.0

```bash
pip install PySide6
```

## Run

```bash
python day5_panel.py
```

Click **开始刷新** (Start refresh) to begin. The parameter values update every second; try dragging the window or clicking while it runs — the UI stays responsive because data is produced on a background thread.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.
