"""Embedding 抽象层：默认使用通义千问 DashScope（OpenAI 兼容），
同时保留本地 bge 选项（设置 EMBEDDING_BACKEND=local 并安装 sentence-transformers 即可启用）。"""
from functools import lru_cache

from ..config import settings

@lru_cache(maxsize=1)
def get_embeddings():
    """返回 langchain Embeddings 实例（按配置懒加载，避免无 key 时启动报错）。"""
    if settings.EMBEDDING_BACKEND == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    if not settings.DASHSCOPE_API_KEY:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法生成向量")

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        check_embedding_ctx_length=False,
        # 关键：显式限制单次批量条数。默认值过大，文档分块数 >10 时
        # DashScope 会报 400「batch size is invalid, it should not be larger than 10」。
        # langchain 会按此值自动分批调用，文档再大也不会失败。
        chunk_size=settings.EMBEDDING_BATCH_SIZE,
    )


@lru_cache(maxsize=8)
def _build_llm(model: str, temperature: float, max_tokens: int):
    """按「模型 + 温度 + 长度」三个维度缓存 ChatModel 实例。

    原来这里是 maxsize=1 且参数固定，导致管理员在后台换了模型也不生效。
    现在把模型名作为缓存键的一部分：改配置后新键自然多出一个实例，旧实例留在缓存里，
    既不破坏正在进行的调用，也不需要重建整个进程。
    """
    if not settings.DASHSCOPE_API_KEY:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法调用大模型")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm(temperature: float | None = None, max_tokens: int | None = None):
    """返回 langchain ChatModel（通义千问）。

    模型名 / 温度 / 最大长度都从**运行时配置**读取，
    因此管理员在后台切换模型或调温度后，下一次提问即刻生效，无需重启服务。

    显式传参时以传参为准（便于测试与特殊链路覆写）。
    """
    from .. import appconfig

    model = str(appconfig.get("LLM_MODEL", settings.LLM_MODEL))
    if temperature is None:
        temperature = float(appconfig.get("LLM_TEMPERATURE", 0.3))
    if max_tokens is None:
        max_tokens = int(appconfig.get("LLM_MAX_TOKENS", 1200))
    return _build_llm(model, float(temperature), int(max_tokens))
