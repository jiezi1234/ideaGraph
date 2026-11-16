# Agent Idea Chat Design

## Goal

在现有论文引用图 MVP 之上，新增一条“对话式 idea 生成”链路：

1. 用户输入自然语言问题
2. Agent 理解问题并生成检索查询
3. 系统从本地论文库中召回候选论文
4. Agent 从候选中选择一篇核心论文
5. Agent 可在最终生成前检索一跳图谱上下文
6. Agent 基于用户问题、核心论文标题/摘要和可用图谱上下文生成一条 idea
7. Agent 使用同一个大模型对生成出的 idea 做结构化评估

截至 2026-05-06，代码中的当前闭环已经包含查询理解输出 `retrieval_query` 和 `query_keywords`、候选论文召回、Ollama 原生工具调用的一跳图谱上下文检索、中文结构化 idea 输出、仅基于生成 idea 的单模型评估，以及可选的 Ollama `message.thinking` 调试输出。

本阶段不要求：

- 多轮记忆真正生效
- 使用引用图做复杂多跳查询
- 使用实体图做推理

LangGraph 作为编排框架使用，但第一版保持节点职责简单、可调试。

## Scope

### In Scope

- 单轮对话式 idea 生成
- LangGraph 编排的单轮节点链路
- 基于摘要向量检索召回候选论文
- 从候选论文中选择一篇核心论文
- 基于“用户问题 + 核心论文标题 + 核心论文摘要 + 一跳图谱上下文”生成 idea
- 通过单一 Ollama tool-calling agent loop 构造 `get_node_neighborhood` 调用
- 用中文返回按 `Problem -> Method -> Experiment` 组织的结构化 idea
- 对生成出的 idea 运行单模型结构化评估，输出总分、分项分数、问题诊断和修改建议
- 暴露最终 idea 生成阶段的 Ollama thinking trace 作为调试信息
- 为后续多轮和工具调用预留 state 结构

### Out of Scope

- 多轮会话历史压缩、长期记忆
- 引用图上的多跳查询增强
- 实体图上的复杂推理
- 基于评估结果自动补查或自修正
- 前端 UI

## Design Principles

1. Build and query stay separate
当前 `MvpGraphBuildWorkflow` 只负责建图。新的 Agent workflow 只负责用图和论文数据回答问题，不得混入建图逻辑。

2. Retrieval and selection stay separate
论文召回和核心论文选择分成两个独立节点，避免“检索是否命中”和“选择是否合理”混在同一个 prompt 里。

3. Minimal state, expandable shape
第一版只跑单轮，但 state 结构按后续可扩展方式设计，避免后面重写 LangGraph state。

4. Tool use stays bounded
当前只允许 `get_node_neighborhood` 这类一跳图谱上下文查询。工具调用用于补充生成上下文，不负责建图、扩展 OpenAlex 数据或执行多跳推理。

## User Flow

### Current v1 Flow

1. 用户输入一个问题
2. `query_understanding` 节点生成检索查询
3. `paper_retrieval` 节点用摘要向量检索召回 top-k 候选论文
4. `paper_selection` 节点从候选中选出一篇核心论文
5. `agent_generation` 节点维护同一段消息历史，按需调用 `get_node_neighborhood`，把工具结果作为 `role=tool` 消息追加回历史，并在模型不再返回 `message.tool_calls` 时解析最终中文 idea
6. `idea_evaluation` 节点只读取生成出的 `idea_output`，输出结构化评估结果
7. 返回结构化结果；如果 Ollama 返回 `message.thinking`，同时返回 `idea_thinking`

### Later Flow

后续在不改主链路的前提下，可以增加：

- 会话历史
- 自主工具调用
- 引用图相关论文补充
- 根据评估结果触发补查和修正

## LangGraph Shape

### Graph Nodes

1. `query_understanding`
输入：`user_query`
输出：`retrieval_query`, `query_keywords`

职责：
- 把用户自然语言问题转成更适合召回的查询文本
- 保留关键词列表作为后续 agent 图谱工具规划上下文；主流程不会自动逐个用关键词检索节点
- 第一版可以先做轻量 prompt 改写
- 如果后续验证发现收益不明显，可退化为直接透传

2. `paper_retrieval`
输入：`retrieval_query`
输出：`candidate_papers`

职责：
- 使用摘要向量检索召回候选论文
- 返回统一论文对象列表
- 不负责决定哪篇是核心论文

3. `paper_selection`
输入：`user_query`, `candidate_papers`
输出：`selected_paper`

职责：
- 从候选论文中选出最适合作为当前问题起点的一篇核心论文
- 第一版可以先按向量相似度最高选第一篇，或保留一个轻量 LLM 选择实现
- 推荐实现为可替换策略接口

4. `agent_generation`
输入：`user_query`, `retrieval_query`, `query_keywords`, `candidate_papers`, `selected_paper`
输出：`graph_context`, `idea_output`, `idea_thinking`

职责：
- 维护一份完整的 Ollama messages 历史
- 用原生 `message.tool_calls` 判断是否调用 `get_node_neighborhood`
- 将工具结果作为 `role=tool` 消息追加回同一段历史
- 工具循环最多 10 个 action round，模型不再返回工具调用时解析当前 assistant content 为最终 idea
- 输出一条中文结构化 idea；如果模型返回 Markdown fenced JSON，解析前剥掉最外层代码块
- 如果模型返回 thinking trace，将其作为调试字段传出

5. `idea_evaluation`
输入：`idea_output`
输出：`idea_evaluation`

职责：
- 只评估已经生成出的 idea，不读取用户原始问题、核心论文、图谱上下文或隐藏检索状态
- 按 `novelty/clarity/feasibility/relevance/expected_effectiveness/evidence_grounding` 输出 1-5 分
- 给出中文诊断、主要问题和修改指令

### Graph State

第一版 state 建议至少包含：

- `user_query`
- `retrieval_query`
- `query_keywords`
- `candidate_papers`
- `selected_paper`
- `graph_context`
- `idea_output`
- `idea_evaluation`
- `idea_thinking`
- `errors`

建议保留但第一版可不使用：

- `conversation_id`
- `history`
- `tool_trace`

## Data Contracts

### CandidatePaper

统一候选论文对象至少包含：

- `paper_id`
- `title`
- `abstract`
- `cited_by_count`
- `score`

说明：
- `score` 表示 retrieval 阶段的相似度或排序分
- 第一版不要求包含引用邻居和实体信息

### SelectedPaper

在 `CandidatePaper` 基础上可直接复用，不必额外定义新对象。核心要求是：

- 能明确被选中的唯一论文
- 可回显标题和摘要

### IdeaOutput

第一版建议结构化输出：

- `title`
- `summary`
- `problem`
- `method`
- `experiment`
- `core_paper_id`
- `core_paper_title`

这样既方便 CLI 输出，也方便后续接 API。

### IdeaEvaluation

当前实现的评估输出包含：

- `overall_score`
- `summary`
- `novelty`
- `clarity`
- `feasibility`
- `relevance`
- `expected_effectiveness`
- `evidence_grounding`
- `strengths`
- `major_issues`
- `revision_instruction`

每个分项指标使用统一的 `score/comment/revision_suggestion` 结构。评估节点的输入上下文只包含 `<generated_idea>` 包裹的 `idea_output` JSON。

### GraphContext

图谱上下文来自 Neo4j 的一跳邻域查询，包含：

- `entities`
- `relationships`
- `page`
- `page_size`
- `total_entities`
- `total_relationships`

该上下文反映当前数据库中的结构化事实，不代表对论文全文的重新验证。
Graph tool 查询先尝试精确 `Paper.title` 或 `Entity.id`，未命中时对
`Entity.id` 和 `Entity.aliases` 做大小写不敏感匹配。返回给工具的节点
属性保持紧凑：`Paper` 只暴露 `id` 和 `abstract`；`Entity` 只暴露 `id`、
`entity_type` 和 `description`。

## Retrieval Design

### v1 Retrieval

第一版召回策略：

- 使用论文摘要向量索引
- 输入为 `retrieval_query`
- 返回 top-k 候选论文

注意：

- 论文召回仍以摘要向量检索为主
- 图谱工具只在召回和选择后补充一跳上下文，不改变候选论文选择流程
- query keywords 可作为候选实体 id 交给 agent 决定是否查图；系统不再单独运行实体向量检索器

## Selection Design

### v1 Selection

第一版选择策略建议保持可替换：

- 默认策略：取 retrieval top-1
- 可选策略：让 LLM 在 top-k 中选最相关的一篇

推荐实现边界：

- `PaperSelector` 接口
- `TopScorePaperSelector` 作为默认实现

这样后续把选择逻辑切到 LLM 或引用图增强时，不需要改 LangGraph 主链。

## Prompt Design

### Query Understanding Prompt

目标：

- 把用户问题变成适合向量检索的短查询文本
- 不求复杂结构化，只求稳定可用

建议输出：

- `keywords`，进入 workflow state 后保存为 `query_keywords`
- `retrieval_query`

实现要求：如果模型输出无效，节点应记录 warning 并退回原始用户问题。

### Chat Idea Agent Prompt

工具使用和最终生成共用 `src/prompts/chat_idea_agent.py`。prompt 模块暴露
`PROMPT` 常量；工具 schema 通过 OpenAI-compatible `tools` 参数动态提供，
prompt 只描述角色、任务、工具使用准则和最终输出字段，不重复完整工具参数
schema。

该 prompt 保持图谱化的三段式生成结构：将选中论文视为 target paper，将候选
论文和图谱邻居论文视为 related papers，将 `Concept`、`Domain`、`Problem`、
`Method`、`Scenario`、`Innovation` 等语义节点视为 entities。最终 idea 必须
围绕 `Problem -> Method -> Experiment` 展开，并在三段中覆盖 Problem
Statement / Motivation、Proposed Method、Step-by-Step Experiment Plan、Test
Case Examples 和 Fallback Plan 等研究方案要素。

Agent 可通过原生 `message.tool_calls` 调用 `get_node_neighborhood`。模型可以指定：

- `node_id`
- `node_types`
- `relationship_types`
- `page`

不支持 depth、page_size 或直接请求 totals。当前实现会把多个工具结果去重合并。
不要使用 JSON fallback 工具协议；没有工具调用时，当前 assistant content 就是
最终 idea 响应。

### Final Idea Output

输入上下文包含：

- 用户问题
- 核心论文标题
- 核心论文摘要
- 同一 messages 历史中的工具调用和工具结果

输出要求：

- 给出一条研究 idea
- 结构化为 `title/summary/problem/method/experiment`
- 面向用户的 idea 字段使用中文
- `core_paper_id` 和 `core_paper_title` 保留可追溯核心论文
- 结构化 fallback 生成通过 `structured_json` 和 `IdeaOutput` schema 约束；
  prompt 不重复完整 JSON schema

当前评估开启时会运行一次改进轮：先评估初始 idea，再把初始 idea、评估反馈、
上一轮可读 `graph_context` 和 `tool_trace` 摘要传给新的 `agent_generation`，
但不复用上一轮原始 OpenAI message history。

### Idea Evaluation Prompt

输入上下文只包含：

- `<generated_idea>` 中的 `idea_output` JSON

输出要求：

- 使用 `src/prompts/evaluate_research_idea.py`
- prompt 使用 XML-like tags 分段，包括 `<role>`、`<context>`、`<task>`、`<scoring_rules>`、`<evaluation_criteria>`、`<output_requirements>`
- 通过 `structured_json` 和 `IdeaEvaluation` schema 约束输出
- 不包含工具策略段，除非后续真的允许评估器调用工具
- 不读取或假设用户问题、核心论文、图谱上下文、外部检索结果

## Module Layout

建议新增模块：

- `src/agent/`
- `src/retrieval/`
- `src/workflows/idea_chat.py`

建议结构：

```text
src/
  agent/
    state.py
    graph.py
    nodes/
      query_understanding.py
      paper_selection.py
      idea_evaluation.py

  retrieval/
    base.py
    vector_retriever.py
    selector.py

  workflows/
    idea_chat.py
```

### Responsibilities

`agent/state.py`
- 定义 LangGraph state

`agent/graph.py`
- 定义 graph 拓扑和节点连接关系

`agent/nodes/*`
- 保留轻量节点函数；主 tool-calling agent loop 位于 `agent/graph.py`

`retrieval/*`
- 负责候选论文召回与核心论文选择策略

`workflows/idea_chat.py`
- 为 CLI/API 提供统一调用入口

## Integration Plan

当前对现有系统的集成边界：

1. 不改现有建图 workflow
2. OpenAlex 建图逻辑继续只负责发现和写入论文/引用
3. 新增一条 idea-chat workflow
4. idea-chat workflow 通过 Neo4j read API 获取论文、实体和一跳邻域

## Error Handling

### Retrieval Returns Empty

如果召回不到候选论文：

- 直接返回结构化错误
- 不进入 selection 和 generation

### Selection Fails

如果候选论文为空或选择器异常：

- 返回明确错误
- 记录日志

### Generation Fails

如果生成模型失败：

- 返回失败信息
- 保留 `selected_paper` 供调试

### Evaluation Fails

如果评估模型输出不符合 `IdeaEvaluation` schema：

- 由 pydantic 校验抛出错误
- 当前不触发补查或自动修正

### Logging

建议新增日志阶段：

- `agent:query_understanding_done`
- `agent:paper_retrieval_done`
- `agent:paper_selection_done`
- `agent:idea_generation_done`
- `agent:idea_evaluation_done`
- `agent:workflow_done`

## Testing Strategy

第一版至少覆盖：

1. `query_understanding` 节点输出 state 更新正确
2. `paper_retrieval` 在 mock retriever 下能返回候选论文
3. `paper_selection` 能从候选里稳定选出核心论文
4. `idea_generation` 能输出结构化 idea
5. `graph_context_retrieval` 能执行并合并一跳邻域查询
6. `idea_evaluation` 只接收生成出的 idea，并返回结构化评估
7. graph 主流程能从输入走到完整输出
8. 空召回和节点失败能返回错误

## Risks

1. Retrieval quality may be weak at first
初期如果只靠摘要向量召回，可能出现候选不稳定。但这是可接受的，因为第一阶段目标是先跑通闭环。

2. Query rewriting may not help
`query_understanding` 可能对召回帮助有限，所以这一步要保持可退化，不要让系统依赖复杂 query rewrite。

3. LangGraph may be overkill for v1
第一版 graph 很短，但仍然建议使用 LangGraph，因为后面会接工具和多轮状态，现在先把状态边界定下来是值得的。

## Non-Goals

这一版明确不做：

- 多轮自主工具循环
- 实体图上的复杂推理
- 引用图上的多跳 related papers 扩展
- 根据评估结果自动修正 idea
- 前端聊天界面

## Recommendation

建议按“最小 LangGraph 闭环”先落地：

- 单轮
- 摘要向量召回
- top-k 候选
- 选一篇核心论文
- 用标题、摘要和一跳图谱上下文生成中文 idea
- 对生成出的 idea 做一次只读结构化评估

这样能最快把“从建图到用图”的下一条主链跑通，并且为下一阶段的 agent tool 使用保留稳定接口。
