# Day30 收官：把 W6 两个后端服务打包成可一键部署的镜像
# 同一份镜像起两个容器，靠 compose 的 command 区分跑哪个服务
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖：healthcheck 用 python -c 探活，镜像自带 python；
# sentence-transformers 走预编译 wheel，不需要 gcc
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# 拷贝整套自包含服务（day25/day28 及其依赖 day27/day23a/day21b/api_config/手册/数据）
COPY . /app

# 默认起 agent 服务；compose 会用 command 覆盖成 live 服务
EXPOSE 8000 8001
CMD ["python", "-m", "uvicorn", "day25_api_service:app", "--host", "0.0.0.0", "--port", "8000"]
