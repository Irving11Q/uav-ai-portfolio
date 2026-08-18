# UAV Flight Parameter Monitoring Panel (with Alerts & Data Logging)

An extended version of the flight monitoring panel: refreshes simulated UAV flight parameters every second on a background thread, flags out-of-range values in red, shows status in a bottom status bar, and appends every reading to a CSV file.

## What's inside

- `day6_panel.py` — a window with:
  - a table of 4 UAV flight parameters (battery voltage, motor current, body temperature, flight altitude);
  - per-parameter alert thresholds — voltage alerts when below limit, the others when above — highlighted in red;
  - a 1-second `QTimer` driving periodic refresh;
  - a background `QThread` sensor reporting readings via a `Signal`;
  - a `QStatusBar` showing refresh state and last receive time;
  - each reading appended to `flight_data.csv` (UTF-8, Excel-compatible).

## Requirements

- Python 3.12
- PySide6 >= 6.0

```bash
pip install PySide6
```

## Run

```bash
python day6_panel.py
```

Click **开始刷新** (Start refresh). Parameter values update every second; rows turn red when a value exceeds its threshold. While it runs, drag the window — the UI stays responsive. When you close it, open `flight_data.csv` in Excel to review the logged readings.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.
