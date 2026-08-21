# UAV Battery Voltage Live Plot

A PySide6 + pyqtgraph desktop app that plots UAV battery voltage as a scrolling real-time curve. Simulated data is generated on a background thread so the UI stays responsive.

## What's inside

- `live_plot.py` — self-contained demo:
  - a `SensorThread` background thread appends a simulated voltage value every 0.3 s;
  - a `QTimer` pulls new values and updates the curve via `curve.setData(x, y)`;
  - a rolling window keeps the last 50 points so the curve scrolls like a live ECG.

## Requirements

- Python 3.12
- PySide6 >= 6.0
- pyqtgraph >= 0.13

```bash
pip install PySide6 pyqtgraph
```

## Run

```bash
python live_plot.py
```

A window opens showing a yellow voltage curve that scrolls continuously.

## Key pattern

```python
plot_widget = pg.PlotWidget()          # create the plot area
curve = plot_widget.plot(pen="y")      # create a curve
curve.setData(x_data, y_data)          # update the curve
```

The same pattern applies when feeding a real serial/network data source instead of the simulated thread.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.
