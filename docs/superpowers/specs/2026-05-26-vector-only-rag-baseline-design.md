# 纯向量 RAG 基线改造设计

## 背景

IdeaGraph 已经在候选论文召回阶段接入 Chroma，但当前 `chat-idea`
运行时默认仍然是图谱增强路径：

- `generation_profile` 负责选择提示词和输出 schema，例如 `ideagraph`
  或 `ai_idea_bench`。
- Chroma 目前只存 paper 级向量和 entity 级向量。
- 默认 workflow 仍然注入 `VectorFallbackGraphContextProvider`，所以模型在
  vector 召回候选论文后，仍然可以继续调用图工具。

因此，当前 Chroma 路径还不能作为“纯向量数据库 baseline”。为了评估图谱
是否真正有效，需要一个只使用向量检索、不暴露 Neo4j 图邻居、并能归档向量
证据的运行模式。

## 目标

1. 增加一个真正的纯向量 RAG baseline，使用 Chroma 做 chunk 级检索。
2. 让 `generation_profile` 只负责提示词和输出 schema。
3. 新增独立的 `retrieval_mode`，专门控制检索行为。
4. 防止图谱上下文泄漏到纯向量实验里。
5. 归档检索到的 chunks，使召回命中率和生成质量分析可以复用同一份 run
   artifact。

## 非目标

- 不替换现有 IdeaGraph 图工具路径。
- 第一版不引入 LlamaIndex，也不再增加第二套向量框架。
- 不修改 `IdeaOutput` 或 `AIIdeaBenchIdeaOutput` 的 schema 契约。
- 本阶段不做完整 benchmark dashboard。

## 检索模式

新增运行参数 `retrieval_mode`。

支持三个取值：

- `graph`：当前 IdeaGraph 行为。先用 Chroma 召回候选论文，再允许图工具
  检索 Paper/Entity 邻域。
- `paper_vector`：只用 Chroma 召回候选论文，禁用图工具。这是快速的摘要级
  vector baseline。
- `chunk_vector`：用 Chroma 召回候选论文和全文 chunks，禁用图工具。这是
  正式的纯向量 RAG baseline。

`generation_profile` 保持独立：

- `ideagraph`：使用当前面向 `IdeaOutput` 的中文 idea 生成 profile。
- `ai_idea_bench`：使用兼容 AI Idea Bench 的英文字段。

预期实验组合：

```text
--generation-profile ai_idea_bench --retrieval-mode graph
--generation-profile ai_idea_bench --retrieval-mode paper_vector
--generation-profile ai_idea_bench --retrieval-mode chunk_vector
```

## Chroma Collections

沿用当前按 Neo4j database 隔离的持久化路径：

```text
.cache/chroma/{NEO4J_DATABASE}/
```

保留现有 collections：

- `papers`
- `entities`

新增第三个 collection：

- `chunks`

Chunk metadata：

```json
{
  "kind": "chunk",
  "neo4j_database": "testidea",
  "paper_id": "Paper title or id",
  "title": "Paper title or id",
  "chunk_id": "stable hash",
  "chunk_index": 0,
  "paragraph_start": 0,
  "paragraph_end": 2,
  "text_source": "full_text_md",
  "text_hash": "sha256..."
}
```

Chunk document text：

```text
Title: <paper title>
Chunk:
<chunk text>
```

`chunks` collection 使用和现有 paper/entity collection 相同的 embedding
模型。这样 graph baseline 和 vector baseline 的差异主要来自检索结构，而不是
模型选择。

## Chunk 来源

repository 需要暴露一个面向向量索引的 paper projection，至少包含：

- `document_id`
- `title`
- `abstract`
- `year`
- `cited_by_count`
- `full_text_markdown_path`
- `text_source`

Chunk 文本来源优先级：

1. 如果 `full_text_markdown_path` 存在且文件可读，使用 Markdown 全文。
2. 否则，如果 `abstract` 非空，用 `Title + Abstract` 生成一个 fallback
   chunk。
3. 如果全文和摘要都没有，该 paper 不进入 chunk 索引。

这样即使语料不完整，baseline 也能跑起来；同时通过 `text_source` 明确标出
每个 chunk 是全文来源还是摘要 fallback。

## 切块策略

第一版使用一个轻量本地 Markdown chunker，采用“段落优先”的策略。每个
chunk 必须绑定到原始论文，通过 metadata 中的 `paper_id`、`title`、
`chunk_index`、`paragraph_start`、`paragraph_end` 保留可追溯关系。

默认参数：

- 最小 chunk 大小：256 tokens
- 目标 chunk 大小：尽量贴近 2048 tokens
- 最大 chunk 大小：2048 tokens
- overlap：80 tokens，仅用于拆分超长段落或超长段落组
- 切分单元：Markdown 段落；heading 可作为段落边界保留在相邻 chunk 中
- fallback：单段超过最大 token 数时，按句子或 token 窗口拆分

切块规则：

1. 正常段落优先，向后合并相邻段落，尽量贴近最大 token 数。
2. 如果加入下一个段落会超过最大 token 数，则停止合并。例如 3 个段落超过
   2048 tokens 而前 2 个段落未超过时，chunk 使用前 2 个段落。
3. 文档末尾最后一个 chunk 如果低于最小 token 数，可以保留，但必须记录
   真实 `paragraph_start` 和 `paragraph_end`。
4. 如果单个段落本身超过最大 token 数，则按句子或 token 窗口拆分，并加入
   overlap。

token 计数使用本地 HuggingFace tokenizer，默认路径为
`/home/zb/models/Qwen3.6-35B-A3B-GPTQ-Int4`。该 tokenizer 对齐生成上下文
模型，避免无空格中文段落被空白切词严重低估。

chunker 必须是确定性的。同一个 paper id、同一份源文本、同样的 token 阈值
和 overlap，必须生成相同的 `chunk_id`。`chunk_id` 建议由
`paper_id + chunk_index + text_hash` 派生。

## 运行流程

### `graph`

1. 执行 query understanding。
2. 从 Chroma `papers` 召回候选论文。
3. 选择 top candidate paper。
4. 启用图工具，运行 agent generation。
5. 归档 candidate papers、selected paper、graph context、tool trace、输出结果
   和 canonical idea。

### `paper_vector`

1. 执行 query understanding。
2. 从 Chroma `papers` 召回候选论文。
3. 选择 top candidate paper。
4. 禁用图工具，直接运行 structured generation。
5. 归档 candidate papers、selected paper、输出结果和 canonical idea。

该模式下 `graph_context` 和 `tool_trace` 不应包含任何图谱证据。

### `chunk_vector`

1. 执行 query understanding。
2. 从 Chroma `papers` 召回候选论文。
3. 选择 top candidate paper。
4. 使用同一个 `retrieval_query` 从 Chroma `chunks` 召回 top-k chunks。
5. 禁用图工具，运行 structured generation，并把 retrieved chunks 作为纯向量
   证据注入 user prompt。
6. 归档 candidate papers、selected paper、retrieved chunks、输出结果和
   canonical idea。

该模式下不能向模型暴露 graph tool schema。

## Prompt 边界

不要增加第三套输出 schema。`generation_profile` 继续决定输出格式。

对 `chunk_vector`，user prompt 应包含类似以下区块：

```text
Retrieved vector context:
[1] Paper: ...
Score: ...
Text source: ...
Chunk:
...
```

system prompt 仍然由 profile 控制。`retrieval_mode` 只改变提供给模型的上下文，
不改变输出 schema。

## Archive 契约

在 `chat-idea` archive 记录里增加字段：

- `retrieval_mode`
- `retrieved_chunks`

每个 retrieved chunk 包含：

- `paper_id`
- `title`
- `chunk_id`
- `chunk_index`
- `score`
- `text_source`
- `text`

对于 `graph`，`retrieved_chunks` 可以为空或不存在。对于 `chunk_vector`，
`retrieved_chunks` 必须存在。

## CLI 和批跑脚本

给 `chat-idea` 增加参数：

```text
--retrieval-mode graph|paper_vector|chunk_vector
```

默认值：

```text
graph
```

更新 `scripts/run_ai_idea_bench_queries.py`，让批跑脚本转发：

```text
--retrieval-mode <mode>
```

run manifest 需要记录：

- `variant`
- `generation_profile`
- `retrieval_mode`

这样同一批 AI Idea Bench 输入可以生成可对比的 `graph`、`paper_vector`、
`chunk_vector` manifests。

## 测试

实现前先加聚焦测试：

1. `ChromaVectorIndex` 能创建并同步 `chunks`。
2. chunk sync 能跳过未变化 chunks，并删除 stale chunks。
3. `ChromaChunkRetriever` 能返回带 metadata 的排序 chunk records。
4. `paper_vector` workflow 禁用图工具，并且不返回 tool trace。
5. `chunk_vector` workflow 会把 retrieved chunks 注入生成 prompt。
6. CLI 能解析 `--retrieval-mode`。
7. AI Idea Bench 批跑脚本能转发并记录 `retrieval_mode`。
8. retrieval hit 脚本能从 `retrieved_chunks` 读取 paper titles。

完成前完整验证：

```bash
python3 -m pytest -q
```

## 风险

- 全文 Markdown 路径可能只是本地 artifact，换机器后文件可能不存在。用
  title/abstract fallback chunk 可以保证模式可运行，但结果报告必须暴露
  `text_source`。
- 第一次建立 chunk 索引可能较慢。应复用现有 paper/entity Chroma 的
  hash-based lazy sync 思路。
- 如果 `graph_context_provider` 仍然被注入，纯向量模式可能意外变成图谱增强。
  测试需要明确断言 `retrieval_mode != graph` 时不会暴露 tool schema。

## 验收标准

满足以下条件后，改造才算完成：

1. `chat-idea --retrieval-mode chunk_vector` 可以在不使用图工具的情况下运行。
2. 当存在匹配的 vector chunks 时，archive 包含
   `retrieval_mode: "chunk_vector"` 和非空 `retrieved_chunks`。
3. `chat-idea --retrieval-mode graph` 保持当前图工具行为。
4. `generation_profile` 和 `retrieval_mode` 可以独立组合。
5. AI Idea Bench 批跑脚本可以分别生成 graph 和 chunk-vector variant 的
   manifests。
6. `python3 -m pytest -q` 通过。
