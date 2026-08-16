"""
PySide6 演示：QTimer（定时刷新）和 QThread（后台任务）。

窗口功能：
    1. 一个计数器标签，每 1 秒自动 +1，由 QTimer 驱动。
    2. 一个按钮，控制定时器的启动/停止。
    3. 一个后台任务（QThread）从 1 数到 100，界面不卡顿，
       并通过 Signal 把进度汇报回主线程。
"""

import sys

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """主窗口：演示 QTimer 定时刷新和 QThread 后台任务。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Day4 Timer + Thread")
        self.resize(420, 300)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- QTimer：定时刷新 ---
        self.count = 0
        self.label = QLabel("刷新次数：0")
        layout.addWidget(self.label)

        self.timer_btn = QPushButton("开始定时刷新")
        self.timer_btn.clicked.connect(self.on_toggle_timer)
        layout.addWidget(self.timer_btn)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)          # 每 1000 毫秒（1 秒）触发一次
        self.timer.timeout.connect(self.on_tick)

        # --- QThread：后台任务（不能阻塞界面）---
        self.worker_label = QLabel("后台任务：还没开始")
        layout.addWidget(self.worker_label)

        self.worker_btn = QPushButton("启动后台任务")
        self.worker_btn.clicked.connect(self.on_start_worker)
        layout.addWidget(self.worker_btn)

        self.worker = Worker()
        self.worker.progress.connect(self.on_worker_progress)

    def on_toggle_timer(self) -> None:
        """启动或停止定时器。"""
        if self.timer.isActive():
            self.timer.stop()
            self.timer_btn.setText("开始定时刷新")
        else:
            self.timer.start()
            self.timer_btn.setText("停止定时刷新")

    def on_tick(self) -> None:
        """定时器每秒调用：计数器 +1 并刷新标签。"""
        self.count += 1
        self.label.setText(f"刷新次数：{self.count}")

    def on_start_worker(self) -> None:
        """启动后台线程，运行期间界面保持响应。"""
        self.worker_label.setText("后台任务：进行中…")
        self.worker.start()

    def on_worker_progress(self, value: int) -> None:
        """连接 worker 的 progress 信号的槽函数。"""
        self.worker_label.setText(f"后台任务：{value}/100")


class Worker(QThread):
    """在独立线程中执行的耗时任务。"""

    progress = Signal(int)

    def run(self) -> None:
        """循环 100 次，每次短暂休眠并发出进度信号。"""
        for i in range(1, 101):
            self.msleep(50)
            self.progress.emit(i)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
