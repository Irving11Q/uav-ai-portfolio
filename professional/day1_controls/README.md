# Day 1: PySide6 Widgets & Layouts

First-day demo of PySide6 desktop GUI basics: common widgets, layout nesting, and signals/slots.

## What's inside

- `day1_controls.py` — a window demonstrating 5 commonly used widgets (QLabel, QLineEdit, QComboBox, QCheckBox, QTableWidget), VBox/HBox layout nesting, and a `clicked` signal connected to a slot.

## Requirements

- Python 3.12
- PySide6 >= 6.0

```bash
pip install PySide6
```

## Run

```bash
python day1_controls.py
```

A window opens. Type in the input box, pick a combo box option, toggle the checkbox, and click the buttons to see the label update with the current widget states.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.
