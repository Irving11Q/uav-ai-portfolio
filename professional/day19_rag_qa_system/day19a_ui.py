#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 19-A：问答界面骨架 —— 先把「看得见」做出来
═══════════════════════════════════════════════════════════════

【这个文件做什么】
    第 4 周前四天全是命令行：跑完只蹦出一个数字，看不见摸不着。
    今天把它变成一个真正的桌面问答窗口 —— 问一句、答一句，
    而且**把检索到的参考资料和分数直接摆在右边**。

【为什么先做界面，不先接检索】
    这是「跑最小」的做法：界面不依赖检索，先让它跑起来、看得见，
    你就知道成品最后长什么样。Day19-B 再把假检索换成真的。
    先有骨架再填肉，比一次性写 800 行再调试省力得多。

【怎么读这个文件】按这个顺序：
    1. 直接跑，先看窗口长什么样（不用读代码）
    2. FakeRetriever   —— 假检索：返回写死的几条结果
    3. AnswerWorker    —— 为什么必须开线程（不开界面会卡死）
    4. MainWindow      —— 界面是怎么搭起来的
    5. main()

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day19a_ui.py

    必须用 chroma-env：它是 --system-site-packages 建的 venv，
    能同时 import 到 PySide6（系统装的）和 chromadb（venv 里装的）。
    系统 Python 只有 PySide6，到 Day19-B 会 import 不到 chromadb。

【为什么学这个】
    ① 第 4 周的成品要求就是「可发布的问答系统」，命令行不算成品。
    ② 把检索过程摆在界面上 = 给 RAG 装了一块仪表盘。
       你终于能一眼看见「它到底找了什么」，这正是 Day15~18 最缺的东西。
    ③ W1/W2 学的 PySide6 在这里派上用场，前后知识接上了。
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：拿工具
# ════════════════════════════════════════════════════════════════

import sys
import time

from PySide6.QtCore import Qt, QThread, Signal
# 【解释】QThread 把耗时任务挪到后台；Signal 是后台通知界面的唯一安全方式。

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QSplitter,
)
# 【解释】这些控件 W1/W2 都用过，这里只是换个场景重新组合。


# ════════════════════════════════════════════════════════════════
# 第 2 部分：假检索 —— 先让界面有东西可显示
# ════════════════════════════════════════════════════════════════

FAKE_CHUNKS = [
    ("1. 电池规格", "本机采用 3S 锂电池，单节额定电压 3.7V，整机额定电压 11.1V，电池容量 5200mAh。"),
    ("3. 电流与功耗", "悬停电流约 12A，巡航电流约 8A，起飞瞬间电流可达 20A。"),
    ("5. 续航估算", "满电悬停约 25 分钟；负载每增加 100g，续航减少约 1.5 分钟。"),
    ("7. 安全预警", "单节电压低于 3.3V 必须立即降落，过放会永久性损伤电池。"),
]
# 【解释】四条写死的资料占位。Day19-B 会把这里换成真正的 Chroma 检索。
#         关键是：界面代码一行都不用改，因为两者的接口完全一样。


class FakeRetriever:
    """
    假检索 —— 数问题里的字命中了多少，排序后返回 Top-K。

    ★ 接口约定（Day19-B 的真检索会一模一样）：
        retrieve(question, top_k) -> [(标题, 正文, 分数), ...]

    只要守住这个接口，界面那边完全不需要知道背后是真是假 ——
    这就是「面向接口编程」，也是后面能无痛换真检索的原因。
    """

    def __init__(self):
        self.mode_name = "假检索（Day19-B 换成真检索）"

    def retrieve(self, question, top_k=3):
        scored = []
        for title, body in FAKE_CHUNKS:
            hit = sum(1 for ch in set(question) if ch in body or ch in title)
            # 【解释】数问题里有几个不同的字出现在资料里 —— 最朴素的匹配方式。
            #         真检索用向量余弦，但「返回什么结构」完全一样。

            score = min(0.95, hit / max(1, len(set(question))) * 1.5)
            # 【解释】除以问题长度做归一化，再封顶 0.95，免得算出大于 1 的怪值。
            scored.append((title, body, score))

        scored.sort(key=lambda x: -x[2])
        # 【解释】按分数从高到低排。sort 是稳定的，同分保持原有顺序。
        return scored[:top_k]
        # 【解释】切片只留前 top_k 条 —— 这就是「Top-K 检索」里的那个 K。


def fake_generate(question, refs):
    """假生成 —— 不调大模型，拼一段看起来像答案的话。"""
    # 【解释】Day19-B 会把这里换成真的 ask_model 调用。

    if not refs:
        return "（参考资料里没有相关内容，这时候应该说「我不知道」，而不是硬答。）"

    top = refs[0]
    # 【解释】取最相关的那一条作为回答依据。
    return ("【Day19-A 占位答案，Day19-B 换成大模型真实生成】\n\n"
            "你问的是「%s」。\n\n"
            "我在手册里找到最相关的一段是「%s」，内容是：\n    %s\n\n"
            "命中分数 %.2f。" % (question, top[0], top[1], top[2]))


# ════════════════════════════════════════════════════════════════
# 第 3 部分：工作线程 —— 为什么不能直接在按钮里干活
# ════════════════════════════════════════════════════════════════

class AnswerWorker(QThread):
    """
    后台线程：检索 + 生成。

    为什么必须开线程？
        检索要跑 embedding（几百毫秒到几秒），生成还要调 API（几秒）。
        如果直接写在按钮的槽函数里，Qt 的事件循环被堵死，
        界面会变成「未响应」，连窗口都拖不动 —— 新手最常踩的坑之一。
    """

    done = Signal(str, list)
    # 【解释】信号：干完活把 (答案, 参考列表) 发回主线程。

    failed = Signal(str)
    # 【解释】出错也发信号，让界面显示错误，而不是默默崩掉。

    def __init__(self, retriever, generator, question, parent=None):
        super().__init__(parent)
        self.retriever = retriever
        self.generator = generator
        self.question = question
        # 【解释】把「用什么检索、怎么生成、问什么」三件事传进来。
        #         于是这个线程不关心背后是假的还是真的 —— 这就叫依赖倒置。

    def run(self):
        """线程入口。QThread.start() 后自动调用 run()，不要手动调它。"""
        try:
            time.sleep(0.4)
            # 【解释】故意睡 0.4 秒模拟真实耗时。
            #         这样你能亲眼看到按钮变成「检索中…」，才相信线程真在后台跑。

            refs = self.retriever.retrieve(self.question)
            # 【解释】第一步：检索，拿到参考资料和分数。

            answer = self.generator(self.question, refs)
            # 【解释】第二步：生成，让模型照着资料回答。

            self.done.emit(answer, refs)
            # 【解释】emit 发出信号，主线程收到后才更新界面。
        except Exception as e:
            # 【解释】任何异常都要接住：子线程里抛异常不会弹窗，只会静默死掉。
            self.failed.emit("%s: %s" % (type(e).__name__, e))


# ════════════════════════════════════════════════════════════════
# 第 4 部分：主窗口 —— 界面怎么搭起来
# ════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """问答窗口：左边对话，右边参考资料。"""

    def __init__(self, retriever, generator):
        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.worker = None
        # 【解释】必须存住线程对象的引用！不存的话线程会被 Python 垃圾回收，
        #         程序直接崩 —— 这是 QThread 第二个常见坑。

        self.setWindowTitle("无人机电池/能耗手册问答助手 · Day19-A")
        self.resize(920, 580)

        self._build_ui()
        # 【解释】界面搭建拆成单独方法，__init__ 才不会又臭又长。

    def _build_ui(self):
        central = QWidget(self)
        # 【解释】QMainWindow 必须有一个中心部件，不能直接往它上面塞布局。
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        # 【解释】最外层竖着排：提示条 + 中间区 + 输入区。

        tips = QLabel("左边是对话，右边是本次检索到的参考资料 —— "
                      "把检索过程摆出来，就是这个界面最大的价值。")
        tips.setWordWrap(True)
        # 【解释】自动换行，免得长句子把窗口撑宽。
        root.addWidget(tips)

        splitter = QSplitter(Qt.Horizontal)
        # 【解释】QSplitter 允许用户拖动中间的分界线，自己调整左右宽度。

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        # 【解释】只读：对话区只能看，不能编辑。
        splitter.addWidget(self.chat_view)

        self.ref_view = QTextEdit()
        self.ref_view.setReadOnly(True)
        splitter.addWidget(self.ref_view)

        splitter.setSizes([580, 340])
        # 【解释】初始左右宽度（像素），右边窄一点够用。
        root.addWidget(splitter, 1)
        # 【解释】第二个参数 1 是拉伸因子，让中间区吃掉所有剩余高度。

        input_row = QHBoxLayout()
        # 【解释】输入区横着排：输入框 + 按钮。
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("问点什么，比如：电池能飞多久？")
        self.input_box.returnPressed.connect(self.on_send)
        # 【解释】回车即发送 —— 这个细节决定手感。
        input_row.addWidget(self.input_box, 1)
        # 【解释】拉伸因子 1：窗口变宽时输入框跟着变宽，按钮保持原样。

        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.on_send)
        # 【解释】点按钮和按回车走同一个 on_send，逻辑只写一份。
        input_row.addWidget(self.send_btn)

        root.addLayout(input_row)
        # 【解释】addLayout 把子布局塞进父布局，和 addWidget 相对。

        self._say("系统", "你好，我是无人机手册问答助手。当前用的是%s，"
                         "答案只是占位，Day19-B 会接入真正的检索和大模型。"
                  % self.retriever.mode_name)

    def _say(self, who, text):
        """往对话区追加一段话。"""
        color = "#185FA5" if who == "我" else "#0F6E56"
        # 【解释】提问和回答用不同颜色，一眼分得清谁说的。
        self.chat_view.append(
            '<p style="color:%s;"><b>%s：</b>%s</p>'
            % (color, who, text.replace("\n", "<br>"))
        )
        # 【解释】QTextEdit 支持富文本，所以换行要手动换成 <br>。
        self.chat_view.verticalScrollBar().setValue(
            self.chat_view.verticalScrollBar().maximum()
        )
        # 【解释】把滚动条拉到最底，始终显示最新内容。

    def _show_refs(self, refs):
        """把参考资料和分数画到右边 —— 这就是 RAG 的仪表盘。"""
        if not refs:
            self.ref_view.setHtml(
                '<p style="color:#A32D2D;"><b>本次没有检索到任何资料</b></p>'
                "<p>这种情况系统应该直接说「我不知道」，而不是硬答。</p>"
            )
            return

        rows = []
        for i, (title, body, score) in enumerate(refs, 1):
            filled = int(round(score * 20))
            # 【解释】分数 0~1 映射成 0~20 个方块，肉眼一眼比出高低。
            bar = "█" * filled
            rows.append(
                '<p style="margin-bottom:12px;">'
                '<b>%d. %s</b>　<span style="color:#888780;">%.2f</span><br>'
                '<span style="color:#378ADD;">%s</span><br>'
                '<span style="color:#5F5E5A;">%s</span></p>'
                % (i, title, score, bar, body[:70])
            )
            # 【解释】正文只截前 70 字，界面上够看就行，太长反而找不到重点。
        self.ref_view.setHtml("<h3>本次检索到的资料</h3>" + "".join(rows))

    def on_send(self):
        """点发送 / 按回车 —— 这里只负责「派活」，自己不干活。"""
        question = self.input_box.text().strip()
        # 【解释】strip 去掉首尾空格，避免误发空内容。
        if not question:
            return
        if self.worker and self.worker.isRunning():
            return
            # 【解释】上一轮还没跑完就别重复发，否则起两个线程互相打架。

        self._say("我", question)
        self.input_box.clear()
        # 【解释】先显示自己说的话并清空输入框，界面立刻有响应感。

        self.send_btn.setEnabled(False)
        self.send_btn.setText("检索中…")
        # 【解释】禁用按钮 + 改文字，明确告诉用户「正在忙」。
        #         因为开了线程，改完这两行界面不会卡住。

        self.worker = AnswerWorker(self.retriever, self.generator, question)
        # 【解释】每轮都新建一个线程对象 —— QThread 不能重复 start。
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        # 【解释】把两个信号分别接到对应的槽函数上。
        self.worker.start()
        # 【解释】start() 在新线程里调用 run()，然后立刻返回，不等它跑完。

    def on_done(self, answer, refs):
        """后台干完了 —— 信号回到主线程，这才可以改界面。"""
        self._show_refs(refs)
        self._say("助手", answer)
        self._finish()

    def on_failed(self, msg):
        self._say("助手", "出错了：%s" % msg)
        self._finish()

    def _finish(self):
        """收尾：把按钮恢复原样，焦点还给输入框。"""
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.input_box.setFocus()
        # 【解释】焦点回到输入框，可以接着问下一句。


# ════════════════════════════════════════════════════════════════
# 第 5 部分：main —— 启动
# ════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    # 【解释】每个 Qt 程序都得先有且只有一个 QApplication 对象。

    retriever = FakeRetriever()
    # 【解释】★ 这就是「假检索 → 真检索」的唯一替换点。
    #         Day19-B 只改这一行，界面代码完全不动。

    win = MainWindow(retriever, fake_generate)
    win.show()
    # 【解释】show() 之后窗口才真正出现在屏幕上。

    sys.exit(app.exec())
    # 【解释】app.exec() 进入事件循环，窗口关掉才返回。
    #         外面套 sys.exit 是为了把退出码正确传出去。


if __name__ == "__main__":
    main()
