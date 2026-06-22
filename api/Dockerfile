# 知册多模态客服智能体 · 线上服务镜像
# 构建（在源码包根目录执行）：  docker build -f api/Dockerfile -t zhice-cs-agent .
# 运行：                        docker run -d -p 8000:8000 --env-file .env --name zhice zhice-cs-agent
FROM python:3.11-slim

WORKDIR /app
# 拷贝整个源码包（src/ + knowledge_base/ + api/ …）；路径以包根为基准自动定位
COPY . /app

# 仅安装 Web 框架依赖；核心智能体仅用标准库，无重型依赖
RUN pip install --no-cache-dir -r api/requirements.txt

EXPOSE 8000
WORKDIR /app/api

# 生产多进程启动（2 worker，可按 CPU 调整）
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
