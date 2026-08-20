# Day7: Serial Data Reading & Parsing

Reads a stream of UAV flight parameters (battery voltage, motor current, body temperature, flight altitude) from a serial connection and parses each comma-separated line into numeric values.

## Files

| File | Description |
|---|---|
| `serial_reader.py` | Opens the serial link, reads one line per second, and parses it into 4 parameters. |
| `mock_sensor.py` | Simulates the device: sends one line of parameters every second over a local TCP connection. |

## Requirements

- Python 3.12

## Run

Run the simulated device first (Terminal 1):

```bash
python mock_sensor.py
```

Then read and parse (Terminal 2):

```bash
python serial_reader.py
```

Terminal 2 prints the parsed parameters each second:

```
电压=24.6V  电流=1.2A  温度=38.5°C  高度=120m
```

## Notes

`MockSerial` exposes the same `readline` / `close` interface as `pyserial.Serial`. To switch to a real device, replace the connection line in `serial_reader.py` with:

```python
import serial
ser = serial.Serial("COM4", 115200)
```

The reading and parsing logic stays unchanged.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.
