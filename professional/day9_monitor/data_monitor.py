"""
UAV 数据采集上位机。

串口数据采集监控的完整示例：表格实时显示 4 个飞行参数（电池电压、电机电流、
机身温度、飞行高度），参数超限时标红告警，电压以滚动曲线展示，数据写入 CSV。
模拟数据由内置后台线程生成，界面保持流畅。真实接入时可将数据源替换为串口。

运行：直接执行本文件即可，无需外部数据源。
"""

import csv
import random
import sys
import threading
import time

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class SensorThread(threading.Thread):
    """后台线程：定时生成一组模拟飞行参数。"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.running = True
        self.lock = threading.Lock()
        self.lines: list[str] = []

    def run(self) -> None:
        while self.running:
            values = [
                round(random.uniform(20.0, 26.0), 1),   # 电池电压 V
                round(random.uniform(0.5, 3.0), 1),     # 电机电流 A
                round(random.uniform(30.0, 65.0), 1),   # 机身温度 °C
                round(random.uniform(50, 200), 0),      # 飞行高度 m
            ]
            with self.lock:
                self.lines.append(",".join(str(v) for v in values))
            time.sleep(0.3)


class MainWindow(QMainWindow):
    """数据采集上位机主窗口。"""

    PARAM_NAMES = ["电池电压", "电机电流", "机身温度", "飞行高度"]
    PARAM_UNITS = ["V", "A", "°C", "m"]
    THRESHOLDS = [22.0, 2.5, 50.0, 150.0]  # 电压为下限，其余为上限
    MAX_POINTS = 50
    CSV_PATH = "flight_data.csv"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UAV 数据采集上位机")
        self.resize(760, 640)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("无人机飞行参数监控（串口数据）"))

        # 参数表格
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["参数", "数值", "单位"])
        for row in range(4):
            self.table.setItem(row, 0, QTableWidgetItem(self.PARAM_NAMES[row]))
            self.table.setItem(row, 1, QTableWidgetItem("--"))
            self.table.setItem(row, 2, QTableWidgetItem(self.PARAM_UNITS[row]))
        layout.addWidget(self.table)

        # 电压实时曲线
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "电压 (V)")
        self.plot_widget.setLabel("bottom", "采样序号")
        self.curve = self.plot_widget.plot(pen="y")
        layout.addWidget(self.plot_widget)

        self.x_data: list[int] = []
        self.y_data: list[float] = []

        # 控制按钮与状态栏
        self.toggle_btn = QPushButton("开始刷新")
        self.toggle_btn.clicked.connect(self.on_toggle)
        layout.addWidget(self.toggle_btn)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("状态：等待开始")

        # 数据源与定时器
        self.sensor = SensorThread()
        self.sensor.start()

        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()

        # CSV 表头
        self._create_csv_header()

    def on_toggle(self) -> None:
        """启动或停止定时刷新。"""
        if self.timer.isActive():
            self.timer.stop()
            self.toggle_btn.setText("开始刷新")
            self.status_bar.showMessage("状态：已停止")
        else:
            self.timer.start()
            self.toggle_btn.setText("停止刷新")
            self.status_bar.showMessage("状态：刷新中")

    def on_tick(self) -> None:
        """定时刷新：解析数据，更新表格/告警/曲线，写入 CSV。"""
        with self.sensor.lock:
            lines = self.sensor.lines.copy()
            self.sensor.lines.clear()
        if not lines:
            return

        line = lines[-1]
        values = [float(v) for v in line.split(",")]

        # 更新表格并标红超限项
        for row in range(4):
            value = values[row]
            item = QTableWidgetItem(f"{value:.1f}")
            alarm = value < self.THRESHOLDS[row] if row == 0 else value > self.THRESHOLDS[row]
            if alarm:
                item.setBackground(QBrush(QColor(255, 80, 80)))
            self.table.setItem(row, 1, item)

        # 更新电压曲线（滚动窗口）
        voltage = values[0]
        self.x_data.append(len(self.x_data))
        self.y_data.append(voltage)
        if len(self.x_data) > self.MAX_POINTS:
            self.x_data = self.x_data[-self.MAX_POINTS:]
            self.y_data = self.y_data[-self.MAX_POINTS:]
        self.curve.setData(self.x_data, self.y_data)

        # 写入 CSV
        self._save_to_csv(values)
        self.status_bar.showMessage(f"已接收：{time.strftime('%H:%M:%S')}")

    def _create_csv_header(self) -> None:
        with open(self.CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["时间", "电压V", "电流A", "温度C", "高度m"])

    def _save_to_csv(self, values: list) -> None:
        with open(self.CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%H:%M:%S")] + values)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.sensor.running = False
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
