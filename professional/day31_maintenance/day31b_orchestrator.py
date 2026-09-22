"""════════════════════════════════════════════════════════════════
Day31b  编排器（把运维智能体接进真实服务，跑通端到端闭环）
════════════════════════════════════════════════════════════════
做什么：
    轮询 Day28 实时巡检服务的 /live/alerts（已发生告警）和 /live/status 的 prediction
    （预测性预警），自动触发 Day31a 的 MaintenanceAgent 出处置单。
    「板子报警 → 模型查手册 → 出处置单」的端到端闭环在这里合龙。

怎么读：
    MaintenanceLoop.poll_once() 读一轮（返回这一轮新生成的处置单）。
    run_selftest() 在进程内用 TestClient 起 Day25+Day28，真跑一遍闭环。
    run_demo()    起真实端口(8000/8001)的常驻进程，给面试官现场演示。

怎么跑：
    python day31b_orchestrator.py --selftest     # 进程内起两服务，真跑闭环
    python day31b_orchestrator.py --demo         # 起真实端口，常驻轮询

为什么学：
    ① 感知→决策→行动 的闭环怎么落地成一个常驻进程（不是一次性脚本）。
    ② 同指标告警去重——温度刷几十条也只出一份处置单，避免刷屏。
    ③ 进程内 TestClient 既能自测又能演示，零端口冲突、零子进程。
【读完之后】→ professional/day31_maintenance/ 是发布目录（自包含，拷出去就能跑）
════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import subprocess

import day31a_maintenance_agent as a31
from day31a_maintenance_agent import MaintenanceAgent, HttpBackend

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


# ════════════════════════════════════════════════════════════════
# 第 1 部分：维护闭环（轮询 + 触发 + 去重）
# ════════════════════════════════════════════════════════════════
class MaintenanceLoop:
    """持有一个 Day28 客户端(c28) 和一个 backend(查 Day25)，反复 poll_once()。

    【解释】★ 去重是工程必须，不是锦上添花：
        合成数据一秒产生几十条温度告警，每条都调一次模型/检索既慢又刷屏。
        已发生告警按 frame 去重（同一帧不重复处理）；
        同类指标（温度/电压…）只出一份处置单，后面只计数不重复生成。
    """

    def __init__(self, c28, backend, have_key=False, verbose=True):
        self.c28 = c28                       # TestClient 或带 .get() 的对象
        self.backend = backend
        self.have_key = have_key
        self.verbose = verbose
        self.seen_frames = set()             # 已处理的帧（避免重复）
        self.done_metrics = set()            # 已出处置单的指标（避免刷屏）

    def poll_once(self):
        new = []
        # ── 分支一：已发生的告警 ──
        try:
            d = self.c28.get("/live/alerts").json()
        except Exception:
            return new
        for al in d.get("alerts", []):
            f = al.get("frame")
            if f in self.seen_frames:
                continue
            self.seen_frames.add(f)
            if al.get("level") not in ("warn", "alarm"):
                continue
            metric = a31.parse_metric(al.get("msg", ""))
            if metric in self.done_metrics:      # 同类指标已出单，跳过
                continue
            self.done_metrics.add(metric)
            order = MaintenanceAgent.generate_order(al, self.backend, self.have_key)
            new.append(order)
            if self.verbose:
                print("  🛠 告警→处置单[%s]：%s" % (metric, al.get("msg", "")))
                for s in order["steps"]:
                    print("     - %s" % s)
        # ── 分支二：预测性预警（续航模型算出即将跌破安全线）──
        try:
            st = self.c28.get("/live/status").json()
        except Exception:
            return new
        pred = st.get("prediction", {}) or {}
        stt = pred.get("seconds_to_threshold")
        if pred.get("trustworthy") and stt is not None and stt < 120:
            key = "pred@%s" % st.get("frames")
            if key in self.seen_frames:
                return new
            self.seen_frames.add(key)
            if "电压" in self.done_metrics:
                return new
            self.done_metrics.add("电压")
            evt = {"frame": st.get("frames"), "level": "warn",
                   "msg": "预测约 %.0f 秒后电压跌破 %.1fV"
                          % (stt, pred.get("threshold_v", 23)),
                   "t": st.get("latest_t")}
            order = MaintenanceAgent.generate_order(evt, self.backend, self.have_key)
            new.append(order)
            if self.verbose:
                print("  🔮 预测预警→处置单[电压]：%s" % evt["msg"])
                for s in order["steps"]:
                    print("     - %s" % s)
        return new


# ════════════════════════════════════════════════════════════════
# 第 2 部分：自测（进程内起 Day25+Day28，真跑闭环）
# ════════════════════════════════════════════════════════════════
def run_selftest():
    if not HAS_FASTAPI:
        print("✗ 缺 fastapi，无法自测")
        return 1
    # 延迟 import：发布目录才有 day25/day28；且要在 import 前决定是否预热
    import day25_api_service as s25
    import day28_live_api as s28
    # ★ 关掉 Day28 自带的「降频 AI 巡检」：本机 LLM 限流时会卡住采集线程，
    #   导致 frames 卡在第 10 帧、温度告警出不来。Day31 的运维 Agent 是独立一层，
    #   不依赖它，关掉只影响 Day28 自己的自由文本洞察，不影响我们的闭环演示。
    s28.core.HAS_LLM = False

    checks = []

    def check(name, cond, extra=""):
        print("  %s %s%s" % ("✅" if cond else "❌", name,
                             ("  " + extra) if extra else ""))
        checks.append(bool(cond))

    print("=" * 64)
    print("  Day31b 自测：进程内起 Day25 + Day28，真跑闭环")
    print("=" * 64)

    with TestClient(s25.app) as c25, TestClient(s28.app) as c28:
        check("Day25 /health", c25.get("/health").status_code == 200)
        check("Day28 /health", c28.get("/health").status_code == 200)

        r = c28.post("/live/start", json={"source": "synth", "speed": 10})
        check("Day28 start(synth)", r.status_code == 200, r.json().get("source", ""))

        # ★ HttpBackend 用 TestClient 当 client：同一个进程内，零端口冲突
        backend = HttpBackend("/manual/search", "/data/filter", client=c25)
        loop = MaintenanceLoop(c28, backend, have_key=False)
        orders = []
        for _ in range(80):                  # 最多 ~8s（每次睡 0.1s）
            time.sleep(0.1)
            orders += loop.poll_once()
            if any(o["metric"] == "温度" for o in orders):
                break

        temp = [o for o in orders if o["metric"] == "温度"]
        check("synth 触发温度告警→处置单", bool(temp))
        if temp:
            o = temp[0]
            check("处置单含根因", bool(o["root_cause"]), o["root_cause"][:16])
            check("处置单含步骤≥2", len(o["steps"]) >= 2, "steps=%d" % len(o["steps"]))
            check("手册引用真查 Day25", len(o["manual_refs"]) >= 1,
                  o["manual_refs"][0][:24])
            check("风险等级非空", bool(o["risk"]), o["risk"])

        # 预测性预警分支（synth 电压不跌破，这里直接构造事件验证代码路径）
        po = MaintenanceAgent.generate_order(
            {"frame": 999, "level": "warn",
             "msg": "预测约 60 秒后电压跌破 23.0V", "t": 99.0},
            backend, have_key=False)
        check("预测性预警分支→处置单", po["metric"] == "电压")

        c28.post("/live/stop")
        check("Day28 stop", c28.get("/live/status").json()["running"] is False)

    passed = sum(1 for x in checks if x)
    print("=" * 64)
    print("  自测完成：%d / %d 项通过" % (passed, len(checks)))
    return 0 if passed == len(checks) else 1


# ════════════════════════════════════════════════════════════════
# 第 3 部分：常驻演示（起真实端口，给面试官现场看）
# ════════════════════════════════════════════════════════════════
def run_demo():
    import requests

    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    procs = []

    def spawn(mod, port):
        p = subprocess.Popen([py, "-m", "uvicorn", mod + ":app",
                              "--port", str(port), "--host", "127.0.0.1"],
                             cwd=here)
        procs.append(p)
        return p

    spawn("day25_api_service", 8000)
    spawn("day28_live_api", 8001)

    # 轮询健康，等服务就绪
    for port in (8000, 8001):
        for _ in range(60):
            try:
                if requests.get("http://127.0.0.1:%d/health" % port,
                                 timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)

    backend = HttpBackend("http://127.0.0.1:8000/manual/search",
                          "http://127.0.0.1:8000/data/filter", client=None)
    have_key = bool(os.environ.get("ZHIPU_API_KEY")
                    or os.environ.get("DEEPSEEK_API_KEY"))
    loop = MaintenanceLoop(None, backend, have_key=have_key, verbose=True)

    # 用 requests 包一层，给 MaintenanceLoop 一个带 .get() 的对象
    class RC:
        def __init__(self, base):
            self.base = base

        def get(self, path):
            return requests.get(self.base + path, timeout=10)

    loop.c28 = RC("http://127.0.0.1:8001")

    requests.post("http://127.0.0.1:8001/live/start",
                  json={"source": "synth", "speed": 5})
    print("Day31 运维闭环已启动（Day25:8000  Day28:8001），Ctrl-C 退出")
    try:
        while True:
            time.sleep(1)
            loop.poll_once()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            requests.post("http://127.0.0.1:8001/live/stop", timeout=5)
        except Exception:
            pass
        for p in procs:
            p.terminate()
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    if "--demo" in sys.argv:
        sys.exit(run_demo())
    print("用法：python day31b_orchestrator.py --selftest | --demo")
