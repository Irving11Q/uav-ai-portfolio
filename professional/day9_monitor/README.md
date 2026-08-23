# UAV Data Acquisition Monitor

A complete PySide6 + pyqtgraph data-acquisition desktop app: displays 4 UAV flight parameters in a live table, flags out-of-range values in red, plots battery voltage as a scrolling curve, and appends every reading to a CSV file. Simulated data is generated on a background thread so the UI stays responsive.

## What's inside

- `data_monitor.py` — self-contained demo combining:
  - a live parameter table (battery voltage, motor current, body temperature, flight altitude);
  - per-parameter alert thresholds — voltage alerts when below the limit, others when above — highlighted in red;
  - a scrolling voltage curve (pyqtgraph) with a rolling window of the last 50 points;
  - a `QStatusBar` showing refresh state and last receive time;
  - CSV logging (UTF-8, Excel-compatible).

## Requirements

- Python 3.12
- PySide6 >= 6.0
- pyqtgraph >= 0.13

```bash
pip install PySide6 pyqtgraph
```

## Run

```bash
python data_monitor.py
```

Click **开始刷新** (Start refresh). Watch the table update, rows turn red when values exceed thresholds, and the voltage curve scroll. Close the app and open `flight_data.csv` in Excel to review the log.

## Notes

The data source is a simulated thread. To use real hardware, replace the `SensorThread` usage with a serial connection, e.g.:

```python
import serial
ser = serial.Serial("COM4", 115200)
line = ser.readline()
```

The table / alert / curve / CSV logic stays unchanged.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.
