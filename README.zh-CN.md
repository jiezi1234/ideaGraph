## MVP 目标

从 OpenAlex 构建 Neo4j 论文引用图，可选地抽取语义实体和关系，然后使用本地图谱生成有依据的研究想法。

## 快速开始

1. 先安装 `uv`，安装说明见 https://docs.astral.sh/uv/getting-started/installation/。
2. 在项目根目录执行 `uv sync` 安装依赖。
3. 将 `.env.example` 复制为 `.env`，并填写本地运行所需的配置值。
4. 使用 `uv run` 启动命令，例如：

```bash
uv run python main.py build-graph --keyword "idea graph" --max-seed-papers 3 --require-references
```

## 配置

将 `.env.example` 复制为 `.env`，并填写本地运行所需的配置值。

## 运行

```bash
uv run python main.py build-graph --keyword "idea graph" --max-seed-papers 3 --require-references
```

启用全文抽取时，需要先在单独的 shell 中启动 OpenDataLoader hybrid 后端：

```bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
CUDA_VISIBLE_DEVICES=1 uv run opendataloader-pdf-hybrid --host 127.0.0.1 --port 15002 --device cuda --log-level info
```

然后使用常规应用命令构建图谱：

```bash
FULL_TEXT_EXTRACTION_ENABLED=true \
PDF_HYBRID_URL=http://127.0.0.1:15002 \
uv run python main.py build-graph --keyword "idea graph" --max-seed-papers 3 --require-references
```

```bash
uv run python main.py chat-idea --query "graph-based scientific idea generation for low-resource domains"
```

### 偶发工作流

准备并评测一个小规模 AI Idea Bench target-paper 对齐 pilot。该流程不会把
target paper 写入图谱，只用引用论文构建参考图谱，然后用审核后的研究问题运行
`chat-idea`，再用官方 I2I 风格的 motivation / experiment plan judge 打分，
并统计检索到的论文对 target 引用论文的命中率：

```bash
python3 scripts/run_ai_idea_bench_queries.py \
  --problem-drafts exports/ai-idea-bench-lidar-perception-10/problem_drafts_v2.jsonl \
  --archive-output-dir exports/ai-idea-bench-lidar-perception-10/chat_idea_runs_v2 \
  --runs-output exports/ai-idea-bench-lidar-perception-10/runs_v2.jsonl \
  --query-field research_problem \
  --limit 10
```

```bash
python3 scripts/evaluate_ai_idea_bench_i2i.py \
  --judge-mode official_i2i \
  --dataset-json .cache/ai_idea_bench/Idea_bench_data/target_paper_data.json \
  --run-manifest exports/ai-idea-bench-lidar-perception-10/runs_v2.jsonl \
  --output-json exports/ai-idea-bench-lidar-perception-10/i2i_official_v2.json \
  --output-csv exports/ai-idea-bench-lidar-perception-10/i2i_official_v2.csv
```

```bash
python3 scripts/evaluate_ai_idea_bench_retrieval_hits.py \
  --run-manifest exports/ai-idea-bench-lidar-perception-10/runs_v2.jsonl \
  --reference-availability exports/ai-idea-bench-lidar-perception-10/reference_availability.json \
  --output-json exports/ai-idea-bench-lidar-perception-10/retrieval_hits_v2.json \
  --output-csv exports/ai-idea-bench-lidar-perception-10/retrieval_hits_v2.csv
```

完整 pilot 流程和输出契约见 `docs/ai-idea-bench-evaluation.md`。

对 IdeaGraph、ResearchAgent 和 AI-Researcher 的离线 idea 生成结果先做标准化，
再做盲评两两对比。标准化后的 `method_idea_v1` 是 judge 的唯一输入边界，
避免盲评脚本直接耦合各系统原始输出格式。若标准化结果中有 `original`，
judge 只看 `Original`；否则只看 `Title`、`Summary`、`Problem`、`Method`
和 `Experiment`。AI-Researcher 会先从 `selected_idea` 中解析被选中的
`Idea X`，再用 `ideas[X-1]` 作为 `original`；如果解析不到编号或编号越界，
会直接报错。完整三系统本地运行手册见
`docs/ai-researcher-researchagent-bus-scheduling-runbook.md`：

```bash
python3 scripts/normalize_method_idea_outputs.py \
  --input exports/chat_idea_runs/20260604-055159-query-c194029a3dcd.json \
  --output-json exports/testbus-method-comparison/normalized/ideagraph.json \
  --method-name IdeaGraph \
  --question "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？"

python3 scripts/normalize_method_idea_outputs.py \
  --input exports/testbus-local-baselines/researchagent/ideas.jsonl \
  --output-json exports/testbus-method-comparison/normalized/researchagent.json \
  --method-name ResearchAgent \
  --question "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？"

python3 scripts/normalize_method_idea_outputs.py \
  --input exports/testbus-local-baselines/ai_researcher/outputs/local_idea_agent/ideas_bus_sched_bus_evo_nn.json \
  --output-json exports/testbus-method-comparison/normalized/ai_researcher.json \
  --method-name AI-Researcher \
  --question "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？"
```

对所有标准化后的方法运行盲评。传入三个方法时，脚本会自动生成所有两两组合，
并且每个 pair 都正序、反序各评一次：

```bash
python3 scripts/compare_normalized_method_ideas.py \
  --idea-json exports/testbus-method-comparison/normalized/ideagraph.json \
  --idea-json exports/testbus-method-comparison/normalized/researchagent.json \
  --idea-json exports/testbus-method-comparison/normalized/ai_researcher.json \
  --question "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？" \
  --output-json exports/testbus-method-comparison/blind_comparison_3methods.json \
  --output-md exports/testbus-method-comparison/blind_comparison_3methods.md
```

从 OpenReview venue 中拉取论文，将每篇论文转换为与 `chat-idea`
一致的 `IdeaOutput` 结构，再用现有评判 prompt 打分，并导出
JSON/CSV/HTML 结果：

```bash
uv run python main.py benchmark-accepted-ideas \
  --venue ICLR.cc/2026/Conference \
  --query rag \
  --max-papers 50 \
  --decision-scope accepted \
  --output-dir exports/iclr2026-rag-benchmark
```

输出目录中会写入 `results.json`、`results.csv` 和可交互的 `report.html`；报告会逐篇
展示转换后的 idea、评分明细、雷达图、ICLR 分数与各维度分数的相关性，以及按
accepted / rejected / other 切换的状态筛选。
如果 `results.json` 已存在，该 workflow 会把它作为增量基线：本次运行中
OpenReview ID 相同的论文会被覆盖，本次未触及的旧论文会继续保留在重新生成的
JSON/CSV/HTML 输出中。

如果要专门看被拒论文，可以把 `--decision-scope` 改成 `rejected`：

```bash
uv run python main.py benchmark-accepted-ideas \
  --venue ICLR.cc/2026/Conference \
  --query rag \
  --max-papers 50 \
  --decision-scope rejected \
  --output-dir exports/iclr2026-rag-benchmark-rejected
```

可以直接打开 HTML 报告查看可视化结果：

```bash
xdg-open exports/iclr2026-rag-benchmark/report.html
```

命令行查看时，`results.csv` 适合快速浏览紧凑分数，`results.json` 包含完整的
idea 转换结果和评判明细：

```bash
head -n 5 exports/iclr2026-rag-benchmark/results.csv
```

从一个或多个明确的 OpenAlex work 构建图谱。该命令只从给定 seed
扩展一跳引用，然后复用现有全文抽取、实体抽取和图写入配置。显式
指定的 seed 即使 OpenAlex 无法提供摘要或可解析全文，也会保留为
`Paper` 节点并继续扩展引用；这类无内容 seed 只跳过实体抽取，后续
可通过本地论文导入补全文：

```bash
uv run python main.py build-openalex-seed-graph \
  --openalex-id https://openalex.org/Wxxxx \
  --openalex-id Wyyyy
```

从一个或多个本地论文文件夹构建图谱。每个文件夹需要包含一个 PDF；
文件夹名会作为论文 `document_id`，摘要会先由模型从 PDF 中抽取，再复用
现有的实体抽取和图写入流程：

```bash
uv run python main.py build-local-paper-graph \
  --paper-dir /path/to/paper-a /path/to/paper-b
```

预览一篇 Markdown 论文的实体抽取结果，不写入 Neo4j。`--output json` 输出归一化
后的机器可读字段；`--output table` 输出同一批实体和实体关系字段的终端表格：

```bash
uv run python main.py preview-entity-extraction \
  --md /path/to/paper.md \
  --output json
```

该 preview 命令会固定输出两组结果：`full_text` 是整篇 Markdown 的基线抽取，
`incremental` 会先抽取 abstract，再对过滤引用后的正文按 tokenizer 切块，并在
各 chunk 之间维护当前实体和关系集合。

图谱构建完成后，用 `chat-idea` 查询已构建的图谱：

```bash
uv run python main.py chat-idea --query "linear graph retrieval augmented generation"
```

如果已经明确要以图谱中的某篇论文作为生成种子，可以传入精确的 `Paper.id`；
`chat-idea` 会跳过查询理解、embedding 召回和论文选择，直接从 agent 生成阶段
继续：

```bash
uv run python main.py chat-idea \
  --query "linear graph retrieval augmented generation" \
  --seed-paper-id "Graph-guided discovery"
```

每次 `chat-idea` 查询都会默认在 `exports/chat_idea_runs/` 下写入一份 JSON
留档，供后续程序化查看。需要按运行覆盖输出目录时可使用：

```bash
uv run python main.py chat-idea \
  --query "linear graph retrieval augmented generation" \
  --archive-output-dir exports/my-query-runs
```

导出当前 Neo4j 图为 GraphML：

```bash
uv run python export_graphml.py
```

将当前 Neo4j 图上传到 Graphistry 做交互式可视化时，把
`GRAPHISTRY_PERSONAL_KEY_ID` 和 `GRAPHISTRY_PERSONAL_KEY_SECRET` 保存在
`.env` 中，然后运行：

```bash
uv run python scripts/visualize_graphistry.py --name "IdeaGraph full graph"
```

该脚本复用主应用的 `NEO4J_*` 配置，并把 Graphistry 的 Bolt session 固定到
`NEO4J_DATABASE`，避免误查 Neo4j 默认数据库。不传 `--limit` 会上传包含孤立
节点在内的完整图；第一次冒烟可以先用 `--limit 1000`。只有使用非默认
Graphistry 服务端时，才需要设置 `GRAPHISTRY_SERVER` 或传入
`--graphistry-server`。

## 当前行为

- 配置从 `.env` 加载。
- 引用扩展支持 `one_hop` 和 `priority_expand`。
- OpenAlex 请求可以使用可选的逗号分隔 API key 池。
- 网络失败会被记录并跳过，不会中止整个运行流程。
- 如果 Neo4j 数据库不存在，会自动创建。
- 启用后，论文 PDF 会按 OpenAlex 外部位置、OpenAlex `content_urls.pdf`、arXiv 标题查询的顺序尝试获取。若 `data/{NEO4J_DATABASE}/{paper}/` 下已有 Markdown，会先复用该解析结果，不再重复下载；成功尝试会保存为 `source_N.pdf` 和 `attempt_N/*.md`，并用于摘要补全和实体/关系抽取。
- 全文实体抽取采用 abstract-first 增量流程：先用摘要初始化当前图状态，再去除 References/Bibliography 后的正文内容，并按 tokenizer 对齐 chunk 逐段返回 patch 更新。
- 本地论文文件夹导入会使用文件夹名作为 `document_id`，先覆盖重建同 ID 的已有论文并保留已有入/出 `CITES` 边，再由模型从 PDF 中抽取摘要，最后复用同一套实体抽取和图写入流程。
- 如果全文不可用，则从标题和摘要中抽取实体与关系。
- `preview-entity-extraction` 是不落库的对比 workflow，会同时输出整篇 Markdown
  抽取结果和 abstract-first 增量切块抽取结果；两组结果都只包含 `entities` 和
  `relations` 字段。
- 实体和关系包含简短描述；当模型提供方返回 usage 元数据时，抽取日志会记录模型输入和输出 token 数。
- 当 `ENTITY_RESOLUTION_ENABLED=true` 时，抽取出的实体只会和 Neo4j 中已存在的实体做对齐。已有 `Entity.id` 保持稳定；别名会累积，resolver 生成的融合描述会替换旧实体描述，避免重复追加。
- 关系端点 canonical 化之后，source、relation type、target 完全相同的已有实体关系可以由模型融合 description。关系 `source_refs` 会跨论文累积，模型融合失败时回退到追加去重写入。
- 每篇论文会被逐篇抽取并增量写入 Neo4j。
- 如果已抓取的被引论文之间存在引用关系，也会创建相应引用边。
- 当 OpenAlex 标题存在时，论文 ID 会被规范化为可读的论文标题。
- `chat-idea` 会将用户查询改写为 `retrieval_query` 和 `query_keywords`；论文检索节点使用 `retrieval_query` 通过 ChromaDB 持久向量索引召回候选论文，`query_keywords` 仅作为 agent 图谱工具规划上下文传入，不会被主流程自动逐个检索。Chroma 索引按 Neo4j 库名落在 `.cache/chroma/{NEO4J_DATABASE}`，并在数据变化时从 Neo4j 的论文、实体和 chunks 重新同步。chunk-vector 检索采用段落优先切块，默认 `CHUNK_MIN_TOKENS=256`、`CHUNK_MAX_TOKENS=2048`，并通过 `CHUNK_TOKENIZER_MODEL_PATH` 指向本地 HuggingFace tokenizer，让 chunk 长度更贴近生成上下文模型。随后通过 Ollama 的 OpenAI-compatible chat completions API 运行工具调用 agent 循环；该循环可以调用 `get_node_neighborhood` 获取一跳图谱上下文，最后返回按 `Problem -> Method -> Experiment` 组织的结构化中文想法。
- 当传入 `--seed-paper-id` 时，`chat-idea` 会按精确 `Paper.id` 从 Neo4j 加载该论文作为已选论文，并直接进入 agent 生成阶段，不再运行查询理解、embedding 召回和论文选择。
- 当图谱工具循环不再返回 `tool_calls` 时，`chat-idea` 会先尝试把该轮 assistant 响应直接解析为 `IdeaOutput` JSON；只有内容为空或 JSON 非法时，才会回退到单独的 `final_idea_generation` 结构化生成。
- `chat-idea` 的图谱工具查找会先尝试精确匹配 `Paper.title` 或 `Entity.id`，然后回退到不区分大小写的 `Entity.aliases`；图谱工具节点载荷会压缩为生成所需的属性。
- `chat-idea` 随后会仅基于生成的想法运行单模型想法评估节点，并打印总分、各评估项得分、诊断和修改指令。
- `chat-idea` 会为每次查询写入 JSON 留档，内容包括查询理解、候选论文、选中论文、图谱工具调用轨迹、合并后的图谱上下文、生成想法、评估结果和可选 thinking trace。
- 如果设置了 `LANGFUSE_PUBLIC_KEY` 和 `LANGFUSE_SECRET_KEY`，`chat-idea` 还会把整次运行写入 Langfuse Cloud；模型调用通过 `langfuse.openai.OpenAI` 记录 generation 和 token 用量，检索、选择、agent 和图谱工具步骤保留为工作流 observation。
- 为了排查轮次级响应，workflow 还会额外记录 `agent_generation_round_N_result` observation，其中包含该轮返回的 `content`、`reasoning`、`finish_reason` 和归一化后的 `{input, output, total}` token 用量。
- 当 Ollama 模型暴露 `message.thinking` 时，`chat-idea` 会打印最终想法生成阶段的 thinking trace，用于调试。
- 查询理解、论文到 idea 转换、想法 agent 行为、实体抽取和想法评估的 prompt
  模板以 Python 模块形式位于 `src/prompts/`，应通过 `src.prompts` 导入，
  不再读取 `.txt` 文件。
- 结构化模型输出由代码里的 OpenAI-compatible `structured_json` 调用和 Pydantic
  schema 约束；prompt 只保留任务和约束说明，不再重复完整 JSON schema。
- GraphML 导出和 `chat-idea` 查询留档都会写入 `exports/`；生成的导出文件属于本地产物，除非明确要求，否则不应提交。
- `scripts/visualize_graphistry.py` 会通过 Bolt 将当前 Neo4j 图上传到
  Graphistry。它从 `.env` 读取 Graphistry 凭据，将 Neo4j 查询固定到
  `NEO4J_DATABASE`，并通过项目日志输出 Graphistry URL。

对已有 Neo4j 数据库回填规范化小写实体别名：

```bash
uv run python scripts/backfill_entity_aliases.py --dry-run
uv run python scripts/backfill_entity_aliases.py
```

全文抽取配置和运行说明见 `docs/full-text-extraction.md`。
