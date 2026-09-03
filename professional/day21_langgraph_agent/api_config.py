"""
════════════════════════════════════════════════════════════════
【配置文件】第 3 周所有代码统一从这里拿：API Key + 接口地址 + 模型名
════════════════════════════════════════════════════════════════

这个文件不做任何事，它只负责"存配置"。
第 3 周每个程序第一行都会写：
    from api_config import API_KEY, BASE_URL, MODEL_NAME
这样你只需要改这一个文件，所有程序都跟着变。

【怎么切换后端（最重要）】
    所有主流大模型 API 都是"OpenAI 兼容格式"——换模型 = 只改 3 个东西：
        接口地址(base_url) + 密钥(api_key) + 模型名(model)
    所以本项目把它们做成"可切换"：改下面这一行 PROVIDER 即可。

        PROVIDER = "glm"      # 默认：智谱 GLM-4-Flash（注册送免费额度，学习零成本）
        PROVIDER = "deepseek" # 切回 DeepSeek（性能好、便宜，需 DEEPSEEK_API_KEY）

    day10~day13 全部自动跟着变，不用改别的文件。

【重要】你的 API Key 填在哪里？
    ⚠️ 千万不要把 Key 直接写死在 day10/day11 那些代码文件里！
      因为它们将来可能被你传到 GitHub（学习路径里有发布计划），
      Key 一旦公开，别人就能用你的额度花钱。
    正确做法（本项目已经帮你配好）：
        Key 存在"系统环境变量"里，名字见下面每个后端的 key_env。
        Python 用 os.environ.get(变量名) 去拿。
    检查/设置方法（Windows）：
        设置 → 系统 → 关于 → 高级系统设置 → 环境变量 → 用户变量 → 新建
    如果环境变量没配好，代码会在启动时明确提示你。

【模型名怎么选】
    glm-4-flash   = 智谱免费档（默认，够用，推荐）
    deepseek-chat = DeepSeek 通用对话（便宜、快、中文好）
    想换具体模型，只改下面 _PROVIDERS 里对应的 "model" 一行。
════════════════════════════════════════════════════════════════
"""

import os
# 【解释】os = 操作系统工具。这里用它读环境变量。

# ─────────────────────────────────────────────────────────────
# 1. 选择后端（只改这一行就能换大模型）
# ─────────────────────────────────────────────────────────────
PROVIDER = "glm"
# 【解释】当前默认用哪个大模型后端。
#         "glm"      = 智谱 GLM-4-Flash，注册送免费额度，学习零成本（推荐）
#         "deepseek" = 深度求索，性能好、便宜，需 DEEPSEEK_API_KEY

# ─────────────────────────────────────────────────────────────
# 2. 各后端的"身份证"：接口地址 + 模型名 + Key 的环境变量名
# ─────────────────────────────────────────────────────────────
_PROVIDERS = {
# _cfg = _PROVIDERS[PROVIDER]：比如 PROVIDER="glm"，就取到 glm 那一组字典。
# 前三行把字典里的字段"解包"成 BASE_URL / MODEL_NAME / API_KEY 三个独立常量。
# API_KEY = os.environ.get(..., "")：os.environ.get 从系统环境变量读 Key，"" 是兜底（读不到就返回空字符串，不报错）。
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        # 【解释】DeepSeek 官方 OpenAI 兼容接口地址。

        "model": "deepseek-chat",
        # 【解释】DeepSeek 通用对话模型（便宜、快、中文好）。

        "key_env": "DEEPSEEK_API_KEY",
        # 【解释】它的 Key 存在这个环境变量里。

        "key_hint": (
            "platform.deepseek.com 注册 → 创建 API Key →\n"
            "   系统环境变量里新建 DEEPSEEK_API_KEY，值为 sk-xxx"
        ),
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        # 【解释】智谱 AI 的 OpenAI 兼容接口地址（GLM 系列）。

        "model": "glm-4-flash",
        # 【解释】glm-4-flash = 免费额度档，学习零成本；glm-4-air 更强但计费。

        "key_env": "ZHIPU_API_KEY",
        # 【解释】它的 Key 存在这个环境变量里。

        "key_hint": (
            "open.bigmodel.cn 注册 → API 密钥 → 复制;\n"
            "   系统环境变量里新建 ZHIPU_API_KEY，值为你复制的密钥\n"
            "   （glm-4-flash 长期有免费额度，注册即用）"
        ),
    },
}
# 【解释】把两个平台的配置都预先写好，靠 PROVIDER 这个"开关"选一个。
#         以后想加通义、硅基流动，照着往这个字典里加一项就行。

# ─────────────────────────────────────────────────────────────
# 3. 根据 PROVIDER 取出当前配置
# ─────────────────────────────────────────────────────────────
_cfg = _PROVIDERS[PROVIDER]
BASE_URL = _cfg["base_url"]
MODEL_NAME = _cfg["model"]
API_KEY = os.environ.get(_cfg["key_env"], "")
# 【解释】从对应环境变量拿 Key。
#         os.environ.get(名字, "") = 拿不到就返回空字符串 ""。

# ─────────────────────────────────────────────────────────────
# 4. 启动时检查：Key 配好了没有（防止你忘配，一脸懵）
# ─────────────────────────────────────────────────────────────
if not API_KEY:
    # 【解释】如果 API_KEY 是空的（环境变量没配）。

    raise RuntimeError(
        f"没找到 {_cfg['key_env']}！\n"
        f"当前后端 = {PROVIDER}，需要它的 API Key。\n"
        f"获取方式：{_cfg['key_hint']}\n"
        f"（配好环境变量后，重新打开终端再运行本程序。）"
    )
    # 【解释】raise = 主动报错。这样你一眼就知道问题在哪，而不是等请求失败。
