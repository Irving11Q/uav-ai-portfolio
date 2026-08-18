"""
无人机飞行参数监控面板（含告警与数据记录）。

每 1 秒刷新一组模拟的飞行参数（电池电压、电机电流、机身温度、飞行高度），
数据由后台线程生成，界面保持响应。参数超过阈值时该行标红告警，
刷新数据实时写入 CSV 文件，可用 Excel 打开查看。
"""

import csv
import random
import sys
import time

from PySide6.QtCore import QThread, QTimer, Signal
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


class MainWindow(QMainWindow):
    """飞行参数监控面板主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("无人机飞行参数监控")
        self.resize(520, 480)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("无人机飞行参数（每秒自动刷新）"))

        # 参数表格：4 行 3 列（参数名 / 数值 / 单位）
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["参数", "数值", "单位"])

        self.param_names = ["电池电压", "电机电流", "机身温度", "飞行高度"]
        self.param_units = ["V", "A", "°C", "m"]
        # 各参数告警阈值：电压为下限，其余为上限
        self.thresholds = [22.0, 2.5, 50.0, 150.0]

        for row in range(4):
            self.table.setItem(row, 0, QTableWidgetItem(self.param_names[row]))
            self.table.setItem(row, 1, QTableWidgetItem("--"))
            self.table.setItem(row, 2, QTableWidgetItem(self.param_units[row]))
        layout.addWidget(self.table)

        # 状态栏：显示当前刷新状态
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("状态：等待开始")

        self.toggle_btn = QPushButton("开始刷新")
        self.toggle_btn.clicked.connect(self.on_toggle)
        layout.addWidget(self.toggle_btn)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.on_refresh)

        self.sensor = SensorWorker()
        self.sensor.data_ready.connect(self.on_data_ready)

        # CSV 数据记录
        self.csv_path = "flight_data.csv"
        self._create_csv_header()

    def on_toggle(self) -> None:
        """启动或停止刷新。"""
        if self.timer.isActive():
            self.timer.stop()
            self.sensor.stop()
            self.toggle_btn.setText("开始刷新")
            self.status_bar.showMessage("状态：已停止")
        else:
            self.timer.start()
            self.sensor.start()
            self.toggle_btn.setText("停止刷新")
            self.status_bar.showMessage("状态：刷新中")

    def on_refresh(self) -> None:
        """定时器触发：请求后台线程生成一组新数据。"""
        self.sensor.request_data()

    def on_data_ready(self, values: list) -> None:
        """接收数据，更新表格、标红超限项并写入 CSV。"""
        for row in range(4):
            value = values[row]
            item = QTableWidgetItem(f"{value:.1f}")

            # 电压低于阈值告警，其余参数高于阈值告警
            alarm = value < self.thresholds[row] if row == 0 else value > self.thresholds[row]
            if alarm:
                item.setBackground(QBrush(QColor(255, 0, 0)))

            self.table.setItem(row, 1, item)

        self._save_to_csv(values)
        self.status_bar.showMessage(f"已接收：{time.strftime('%H:%M:%S')}")

    def _create_csv_header(self) -> None:
        """创建 CSV 文件并写入表头。"""
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["时间", "电压V", "电流A", "温度C", "高度m"])

    def _save_to_csv(self, values: list) -> None:
        """将一组数据追加写入 CSV。"""
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%H:%M:%S")] + values)


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
                    round(random.uniform(20.0, 26.0), 1),
                    round(random.uniform(0.5, 3.0), 1),
                    round(random.uniform(30.0, 65.0), 1),
                    round(random.uniform(50, 200), 0),
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
