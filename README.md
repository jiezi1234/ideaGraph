# IdeaGraph

从文献引用图谱出发的研究选题工作台：检索论文 → 构建引用图与语义实体图 → 工具调用 Agent 沿图谱探索 → 生成可追溯的研究方案与实验计划。

## 功能

- **建立文献图谱**：关键词检索、引用扩展、PDF 全文解析、实体与关系抽取，写入 Neo4j
- **生成研究想法**：多轮图谱工具调用，输出「问题 → 方法 → 实验计划」，附评审意见与改进对比
- **图谱浏览**：论文引用图与实体关系网络，支持搜索、缩放与节点详情
- **过程追踪**：任务阶段进度，可选 Langfuse 调用链查看
- **Markdown 报告**：网页内渲染，支持下载 `.md`

## 快速开始

依赖：

- Python 3.12+，[uv](https://docs.astral.sh/uv/)
- Neo4j 5
- 一个 OpenAI 兼容的模型服务和一个 embedding 服务（如百炼、DeepSeek、硅基流动）

### 1. 安装与配置

```bash
uv sync
cp .env.example .env
```

编辑 `.env`：

| 变量 | 说明 |
| --- | --- |
| `LLM_BASE_URL` / `LLM_API_KEY` | 模型服务端点与密钥 |
| `LLM_MODEL_EXTRACTION` / `LLM_MODEL_GENERATION` | 抽取与生成使用的模型档位 |
| `EMBED_MODEL` | embedding 模型；同厂商时 `EMBED_BASE_URL` / `EMBED_API_KEY` 留空即可复用 LLM 配置 |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | 图数据库连接 |
| `WEB_ACCESS_TOKEN` | Web 工作台访问口令，至少 24 字符 |

### 2. 启动 Web 工作台

```bash
uv run uvicorn src.web.server:app --host 127.0.0.1 --port 8080
```

打开 <http://127.0.0.1:8080>，输入 `WEB_ACCESS_TOKEN` 登录。页面提供两个入口：建立文献图谱、生成研究想法。

### 3. Docker 部署

```bash
docker compose --env-file .env -f deploy/compose.yml up -d --build
```

服务器部署、反向代理与数据目录说明见 [docs/web-deployment.md](docs/web-deployment.md)。

## 命令行

```bash
# 关键词建图
uv run python main.py build-graph --keyword "graph neural network" --max-seed-papers 3

# 指定 OpenAlex 论文建图
uv run python main.py build-openalex-seed-graph --openalex-id W1234567890

# 本地论文文件夹建图
uv run python main.py build-local-paper-graph --paper-dir ./papers/example

# 生成研究想法
uv run python main.py chat-idea --query "如何提升公交调度的鲁棒性？" --seed-paper-id "Paper.id"

# 预览实体抽取结果（不写库）
uv run python main.py preview-entity-extraction --md paper.md --output table

# 导出当前图谱为 GraphML
uv run python export_graphml.py
```

## 测试

```bash
uv run pytest -q
```

## 目录结构

```
src/
  agent/          Agent 图与节点
  app/            依赖装配（bootstrap）
  config/         配置加载
  extraction/     摘要与实体抽取
  ingestion/      OpenAlex / OpenReview / PDF 获取
  knowledge/      实体与关系对齐
  models/         OpenAI 兼容模型客户端
  observability/  日志、任务进度、Langfuse
  prompts/        提示词模块
  retrieval/      Chroma 向量检索与切块
  schemas/        Pydantic 数据结构
  storage/        Neo4j 仓储
  tools/          图谱查询工具
  web/            FastAPI 工作台与前端
  workflows/      建图、chat-idea、评测流程
scripts/          离线 benchmark 与运维脚本
tests/            单元测试
deploy/           Dockerfile 与 compose
docs/             设计与部署文档
main.py           CLI 入口
```

## 文档

- [Web 部署与运维](docs/web-deployment.md)
- [全文抽取](docs/full-text-extraction.md)
- [AI Idea Bench 评测](docs/ai-idea-bench-evaluation.md)

## 许可

Apache-2.0
