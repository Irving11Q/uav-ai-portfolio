"""
PySide6 常用控件与布局示例。

集中展示上位机开发最常用的 5 类控件（标签、输入框、下拉框、复选框、表格），
以及 VBox / HBox 布局嵌套的用法，并演示 clicked 信号到槽函数的连接方式。
"""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """主窗口：集中展示常用控件与信号槽绑定。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Day3 常用控件展示")
        self.resize(560, 520)

        central = QWidget(self)
        self.setCentralWidget(central)

        # 主布局为垂直布局，控件自上而下依次排列
        layout = QVBoxLayout(central)

        # 标签：回显交互结果
        self.label = QLabel("1. 这是 QLabel 标签：用来显示文字")
        layout.addWidget(self.label)

        # 输入框：接收用户输入，作为 on_show 的数据源
        layout.addWidget(QLabel("2. 这是 QLineEdit 输入框：输入内容"))
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("在这里输入，点按钮会显示到标签上")
        layout.addWidget(self.line_edit)

        # 下拉框：提供可选项（示例为无人机飞行参数）
        layout.addWidget(QLabel("3. 这是 QComboBox 下拉框：从选项里选一个"))
        self.combo = QComboBox()
        self.combo.addItems(["电池电量", "电机电流", "机身温度", "飞行高度"])
        layout.addWidget(self.combo)

        # 复选框：布尔开关，默认开启
        layout.addWidget(QLabel("4. 这是 QCheckBox 复选框：勾选/取消"))
        self.checkbox = QCheckBox("开启实时监控")
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)

        # HBox 嵌套进 VBox，使两个按钮水平并排
        row = QHBoxLayout()
        self.show_btn = QPushButton("把输入显示到标签")
        self.show_btn.clicked.connect(self.on_show)
        row.addWidget(self.show_btn)

        self.clear_btn = QPushButton("清空输入框")
        self.clear_btn.clicked.connect(self.on_clear)
        row.addWidget(self.clear_btn)
        layout.addLayout(row)

        # 表格：数据面板核心，双层循环逐格填充
        layout.addWidget(QLabel("5. 这是 QTableWidget 表格：数据面板核心"))
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["参数", "数值", "单位"])
        data = [
            ["电池电量", "24.6", "V"],
            ["电机电流", "1.2", "A"],
            ["机身温度", "38.5", "°C"],
            ["飞行高度", "120", "m"],
        ]
        for r in range(4):
            for c in range(3):
                self.table.setItem(r, c, QTableWidgetItem(data[r][c]))
        layout.addWidget(self.table)

    def on_show(self) -> None:
        """汇总输入框、下拉框、复选框状态并回显到标签。"""
        text = self.line_edit.text() or "（没输入内容）"
        msg = (
            f"输入框：{text}｜下拉框：{self.combo.currentText()}"
            f"｜监控：{'开' if self.checkbox.isChecked() else '关'}"
        )
        self.label.setText(msg)

    def on_clear(self) -> None:
        """清空输入框。"""
        self.line_edit.clear()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
