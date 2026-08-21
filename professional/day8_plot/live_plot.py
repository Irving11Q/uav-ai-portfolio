"""
UAV 电压实时曲线监控。

从模拟数据源读取电池电压，用 pyqtgraph 绘制滚动的实时曲线。
模拟数据在后台线程生成，界面保持流畅。

运行：直接执行本文件即可，无需外部数据源。
"""

import random
import sys
import threading
import time

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow


class SensorThread(threading.Thread):
    """后台线程：定时向共享列表追加一组模拟电压值。"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.running = True
        self.lock = threading.Lock()
        self.values: list[float] = []

    def run(self) -> None:
        while self.running:
            value = round(random.uniform(23.0, 25.0), 1)
            with self.lock:
                self.values.append(value)
            time.sleep(0.3)


class MainWindow(QMainWindow):
    """实时电压曲线主窗口。"""

    MAX_POINTS = 50

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UAV 电压实时曲线")
        self.resize(700, 500)

        # 绘图区与曲线
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)
        self.plot_widget.setLabel("left", "电压 (V)")
        self.plot_widget.setLabel("bottom", "采样序号")
        self.curve = self.plot_widget.plot(pen="y")

        # 数据缓冲
        self.x_data: list[int] = []
        self.y_data: list[float] = []

        # 后台数据源 + 定时器
        self.sensor = SensorThread()
        self.sensor.start()

        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()

    def on_tick(self) -> None:
        """定时刷新：取后台生成的最新值并更新曲线。"""
        with self.sensor.lock:
            values = self.sensor.values.copy()
            self.sensor.values.clear()
        if not values:
            return

        # 追加新点，滚动窗口裁剪
        for v in values:
            self.x_data.append(len(self.x_data))
            self.y_data.append(v)
        if len(self.x_data) > self.MAX_POINTS:
            self.x_data = self.x_data[-self.MAX_POINTS:]
            self.y_data = self.y_data[-self.MAX_POINTS:]

        self.curve.setData(self.x_data, self.y_data)

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
