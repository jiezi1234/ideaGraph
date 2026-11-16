FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.cloud.tencent.com/debian|g' /etc/apt/sources.list.d/debian.sources && apt-get update && apt-get install -y --no-install-recommends default-jre-headless && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml /tmp/pyproject.toml
# 小服务器只安装 CPU runtime，不下载 CUDA 或 PDF hybrid 模型。
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple filelock typing-extensions networkx jinja2 fsspec setuptools sympy==1.13.1 numpy pillow && pip install --no-cache-dir --no-deps torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN python -c "import tomllib; p=tomllib.load(open('/tmp/pyproject.toml','rb')); open('/tmp/requirements.txt','w').write('\n'.join(d.replace('[hybrid]', '') for d in p['project']['dependencies'] if not d.startswith(('torch','graphistry'))))" && pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple -r /tmp/requirements.txt fastapi uvicorn pytest
COPY src /app/src
COPY tests /app/tests
COPY main.py /app/main.py
COPY scripts /app/scripts
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple 'graphistry>=0.56.1' uv
CMD ["uvicorn", "src.web.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
