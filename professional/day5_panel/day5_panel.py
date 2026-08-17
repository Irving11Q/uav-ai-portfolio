"""
无人机飞行参数监控面板。

每 1 秒自动刷新一组模拟的飞行参数（电池电压、电机电流、机身温度、飞行高度），
数据由后台线程生成，界面保持响应。
"""

import random
import sys

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """飞行参数监控面板主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("无人机飞行参数监控")
        self.resize(500, 420)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("无人机飞行参数（每秒自动刷新）"))

        # 参数表格：4 行 3 列（参数名 / 数值 / 单位）
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["参数", "数值", "单位"])

        self.param_names = ["电池电压", "电机电流", "机身温度", "飞行高度"]
        self.param_units = ["V", "A", "°C", "m"]
        for row in range(4):
            self.table.setItem(row, 0, QTableWidgetItem(self.param_names[row]))
            self.table.setItem(row, 1, QTableWidgetItem("--"))
            self.table.setItem(row, 2, QTableWidgetItem(self.param_units[row]))
        layout.addWidget(self.table)

        self.status_label = QLabel("状态：等待开始")
        layout.addWidget(self.status_label)

        self.toggle_btn = QPushButton("开始刷新")
        self.toggle_btn.clicked.connect(self.on_toggle)
        layout.addWidget(self.toggle_btn)

        # 定时器：每秒触发一次刷新
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.on_refresh)

        # 后台传感器线程
        self.sensor = SensorWorker()
        self.sensor.data_ready.connect(self.on_data_ready)

    def on_toggle(self) -> None:
        """启动或停止刷新。"""
        if self.timer.isActive():
            self.timer.stop()
            self.sensor.stop()
            self.toggle_btn.setText("开始刷新")
            self.status_label.setText("状态：已停止")
        else:
            self.timer.start()
            self.sensor.start()
            self.toggle_btn.setText("停止刷新")
            self.status_label.setText("状态：刷新中")

    def on_refresh(self) -> None:
        """定时器触发：请求后台线程生成一组新数据。"""
        self.sensor.request_data()

    def on_data_ready(self, values: list) -> None:
        """接收后台数据并更新表格。"""
        for row in range(4):
            self.table.setItem(row, 1, QTableWidgetItem(f"{values[row]:.1f}"))


class SensorWorker(QThread):
    """模拟传感器：收到请求后生成一组随机读数。"""

    data_ready = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._pending = False

    def request_data(self) -> None:
        self._pending = True

    def run(self) -> None:
        self._running = True
        while self._running:
            if self._pending:
                self._pending = False
                values = [
                    round(random.uniform(23.0, 25.0), 1),
                    round(random.uniform(0.5, 2.0), 1),
                    round(random.uniform(30.0, 60.0), 1),
                    round(random.uniform(50, 150), 0),
                ]
                self.data_ready.emit(values)
            self.msleep(50)

    def stop(self) -> None:
        self._running = False


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
