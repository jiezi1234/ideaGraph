# IdeaGraph 与 AI-Researcher、ResearchAgent 的 idea generation 实验方案

## 1. 实验目标

本实验只比较三种方法：

- **IdeaGraph**：我们的 typed scientific evidence graph + runtime graph exploration 方法。
- **AI-Researcher**：`HKUDS/AI-Researcher`，仓库地址为 `https://github.com/HKUDS/AI-Researcher`。
- **ResearchAgent**：`JinheonBaek/ResearchAgent`，仓库地址为 `https://github.com/JinheonBaek/ResearchAgent`。

本方案主要参考两篇论文的评估思想：

- **Chain-of-Ideas: Revolutionizing Research via Novel Idea Development with LLM Agents**：使用 Idea Arena，对不同方法生成的 idea 和 experiment design 做匿名 pairwise ranking，并用 Elo / win rate 统计方法优劣。
- **HindSight: Evaluating LLM-Generated Research Ideas via Future Impact**：使用 time-split，把系统限制在 cutoff 前信息，再用 cutoff 后真实论文的引用影响力和顶会发表情况评估 generated ideas。

## 2. 基准方法定位

### 2.1 AI-Researcher
`HKUDS/AI-Researcher`，对应论文 **AI-Researcher: Autonomous Scientific Innovation**。该论文于 2025 年 5 月公开在 arXiv，编号为 `arXiv:2505.18705`。

AI-Researcher 是端到端 autonomous scientific innovation 系统，完整流程包括：

1. **Literature review**：读取用户给定的研究想法或参考论文，整理相关工作和研究背景。
2. **Hypothesis / idea generation**：基于文献和任务描述生成候选研究想法。
3. **Algorithm design and implementation**：把候选想法发展成算法方案，并生成实现代码。
4. **Validation and refinement**：运行实验、分析结果，并根据结果迭代修改方案。
5. **Manuscript creation**：整理实验、方法和结果，生成接近论文格式的 manuscript。

本实验只取其 **literature review + hypothesis / idea generation** 阶段作为基准方法；代码实现、真实实验运行和 manuscript creation 不进入主评分。

### 2.2 ResearchAgent

`JinheonBaek/ResearchAgent`，对应论文 **ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models**。该论文发表在 **NAACL 2025 Long Papers**，ACL Anthology ID 为 `2025.naacl-long.342`。

ResearchAgent 是 literature-grounded iterative ideation 系统，主要流程包括：

1. **Core paper input**：从一篇核心论文出发，作为生成 research idea 的起点。
2. **Academic graph retrieval**：沿论文引用图检索相关 publications，扩展文献上下文。
3. **Knowledge store retrieval**：从论文中挖掘共享概念和实体，构建并检索知识存储。
4. **Problem / method / experiment generation**：生成 research problem、proposed method 和 experiment design。
5. **ReviewingAgents refinement**：使用多个 LLM-based reviewing agents 对 idea 给出反馈，并迭代修订生成结果。

### 2.3 两个 baseline 的论文检索方式

**AI-Researcher** 主要从给定的 reference papers 出发。运行时先读取输入中的 `source_papers`，然后围绕这些论文生成 idea。完整流程里还会让 Prepare Agent 从参考论文中选择 `reference_papers`，并按论文标题到 OpenAlex / arXiv 查找和下载对应全文。它默认不是从一个开放 research question 去全网搜索大量新论文，而是围绕输入参考论文做补充读取。

**ResearchAgent** 主要从 core paper 出发做引用扩展。运行时先读取输入中的 seed / core papers，然后获取这些论文的 title 和 abstract；接着查询每篇 core paper 的 references，再用 embedding similarity 从这些 references 中选出最相关的若干篇论文。同时，它会从知识库中取与 core paper 和 retrieved references 相关的 entities，用于后续 idea 生成和 refinement。

因此，实验中需要把两者的检索空间统一到每个问题的 visible papers：AI-Researcher 的 `source_papers` / `reference_papers` 来自 visible papers；ResearchAgent 的 core papers 和 retrieved references 也只能来自 visible papers。

## 3. 评估框架总览

本方案只采用两种评估方法，顺序如下：

| 顺序 | 评估方法 | 核心问题 | 主指标 |
|---|---|---|---|
| 1 | Chain-of-Ideas / Idea Arena | 在盲评对比中，哪种方法生成的 idea 与实验设计综合质量更高 | Elo、pairwise win rate、dimension win |
| 2 | HindSight | idea 是否预测到未来真实且有影响力的研究 | HindSight score、match rate、impact-weighted match |

两种方法的分工如下：

- **Idea Arena** 先比较生成结果的综合科研质量，评价 novelty、significance、clarity、feasibility、expected effectiveness，以及 experiment design；pairwise preference 只是统计形式，不是评价目标本身。
- **HindSight** 再检查 idea 是否能匹配 cutoff 后真实发表且有影响力的论文，作为更客观的 future-impact 信号。

## 4. 评估方法一：Chain-of-Ideas / Idea Arena

### 4.1 核心思想

Chain-of-Ideas 使用 Idea Arena 做 pairwise evaluation。相比给每个 idea 打绝对分，pairwise ranking 更稳定，也更接近 human preference。

本实验采用三方 round-robin：

```text
IdeaGraph vs AI-Researcher
IdeaGraph vs ResearchAgent
AI-Researcher vs ResearchAgent
```

每个 research question、每对方法都比较两次，交换 A/B 顺序，降低 position bias。评估时隐藏方法名称，只给 judge 展示同一个问题下两种方法生成的 idea 和实验设计。

### 4.2 Idea 与实验设计评价维度

Idea Arena 的评估对象不是单独一句 idea，而是同一个 research question 下完整的 idea proposal。每个 proposal 至少包含 problem、motivation、proposed method 和 experiment plan。因此，评估维度同时覆盖 idea 本身和实验设计。

idea 质量评价维度：

| 维度 | 评价问题 |
|---|---|
| Novelty | idea 是否相对已有文献有新意 |
| Significance | 研究问题是否重要 |
| Clarity | problem、motivation、method 是否清楚 |
| Feasibility | 方法是否现实可执行 |
| Expected Effectiveness | 如果实施，是否有望带来有效结果 |

实验设计评价维度：

| 维度 | 评价问题 |
|---|---|
| Experiment Feasibility | 实验是否能实际执行 |
| Technical Quality | dataset、baseline、metric、ablation 是否合理 |
| Experiment Clarity | 实验步骤是否清楚 |

为了避免 judge 只偏好文本长度或格式更整齐的输出，输入给 judge 前需要统一字段顺序，并限制每个字段的最大长度。

judge 输出：

```json
{
  "question_id": "q001",
  "pair": ["IdeaGraph", "AI-Researcher"],
  "order": "A_first",
  "winner": "A|B|tie",
  "dimension_winners": {
    "novelty": "A|B|tie",
    "significance": "A|B|tie",
    "clarity": "A|B|tie",
    "feasibility": "A|B|tie",
    "expected_effectiveness": "A|B|tie",
    "experiment_feasibility": "A|B|tie",
    "technical_quality": "A|B|tie",
    "experiment_clarity": "A|B|tie"
  },
  "rationale": "..."
}
```

### 4.3 结果汇总

Idea Arena 的基本观测单位是一次匿名 pairwise battle。每次 battle 记录 A/B 两个 proposal 的 overall winner 和各维度 winner。汇总时先保留原始 battle 结果，再计算方法级统计量。

主要统计量：

- **overall win rate**：某方法在所有 pairwise battles 中获胜的比例，tie 按 0.5 计。
- **overall Elo**：使用所有 pairwise battle 的 overall winner 更新 Elo，作为 Idea Arena 的主排序指标。
- **dimension Elo**：分别使用 Novelty、Significance、Clarity、Feasibility、Expected Effectiveness、Experiment Feasibility、Technical Quality、Experiment Clarity 等维度的 winner 更新 Elo。
- **dimension win rate**：每个维度上的胜率，作为对应 dimension Elo 的直观解释。

主结论以 overall Elo 和 overall win rate 为主；各维度的 Elo 与 win rate 用来解释方法优势来自 idea 新颖性、可行性，还是实验设计质量。

报告表：

Overall 表：

| 方法 | Overall Elo | Overall win rate |
|---|---:|---:|
| IdeaGraph | | |
| AI-Researcher | | |
| ResearchAgent | | |

维度表：

| 维度 | IdeaGraph Elo | IdeaGraph win rate | AI-Researcher Elo | AI-Researcher win rate | ResearchAgent Elo | ResearchAgent win rate |
|---|---:|---:|---:|---:|---:|---:|
| Novelty | | | | | | |
| Significance | | | | | | |
| Clarity | | | | | | |
| Feasibility | | | | | | |
| Expected Effectiveness | | | | | | |
| Experiment Feasibility | | | | | | |
| Technical Quality | | | | | | |
| Experiment Clarity | | | | | | |

## 5. 评估方法二：HindSight-style future impact

### 5.1 核心思想

HindSight-style 评估把 idea generation 看成一种对未来研究方向的预测：

```text
系统只能使用 cutoff 前信息生成 idea
-> 将 generated idea 与 cutoff 后真实论文匹配
-> 匹配到越高影响的未来论文，分数越高
```

这比单纯 LLM judge 更适合作为主指标，因为它直接问：这个 idea 是否接近后来真实发生且有影响力的研究。

### 5.2 Matching 输入

每个 generated idea 编码为：

```text
idea_text = problem + proposed_method
```

每篇 future paper 编码为：

```text
paper_text = title + abstract
```

优先使用 SPECTER2 或同类 scientific paper embedding。匹配流程：

1. 用同一 embedding model 编码所有 idea 和 future papers。
2. 用 FAISS 建立 future paper 向量索引。
3. 每个 idea 检索 top-K future papers，建议 `K=20`。
4. 保留相似度超过阈值 `theta` 的 matches。

HindSight 原文使用 SPECTER2、FAISS、top-K=20。本实验固定一个匹配阈值用于主结果，并在报告中记录该阈值，保证结果可复现。

### 5.3 Future impact score

每篇 future paper 的 impact score：

```text
h(p) = 0.6 * normalized_citation_count(p) + 0.4 * top_venue_indicator(p)
```

其中：

- `normalized_citation_count(p)` 是在同一 future pool 内 min-max normalized 的 citation count；
- `top_venue_indicator(p)` 为 1 表示论文发表在固定顶会列表中，否则为 0；
- 顶会列表建议固定为 ICLR、NeurIPS、ICML、ACL、EMNLP、CVPR、AAAI。

每个 idea 的 HindSight-style score：

```text
score(i) = max h(p), p in matched_future_papers(i)
```

如果没有匹配到 future paper，则：

```text
score(i) = 0
```

使用 max 而不是 average，是因为一个 idea 只要准确预测到一个高影响未来方向，就应得到较高分。

### 5.4 方法级指标

对每种方法统计：

| 指标 | 含义 |
|---|---|
| mean HindSight score | 所有 idea 的平均 future-impact 分数 |
| median HindSight score | 中位数，避免少数极高引用论文主导 |
| match rate | 至少匹配一个 future paper 的 idea 比例 |
| average matches | 每个 idea 平均匹配 future paper 数 |
| top-impact match rate | 匹配到 top venue 或高引用 future paper 的比例 |
| best-of-3 score | 每个问题三个 ideas 中最高分，用于模拟实际使用者挑选最佳 idea |
| question win rate | 每个问题上哪种方法 best-of-3 分数最高 |

## 6. 数据构建

### 6.1 测试输入就是 research question

第一版实验不需要构造复杂数据集。每个测试样本可以只是一条 research question，再附上必要背景材料：

```json
{
  "question_id": "q001",
  "research_question": "...",
  "background": "... optional",
  "cutoff_date": "2024-01-01",
  "visible_papers": ["paper title / id / abstract ... optional"]
}
```

最小可运行输入是：

```json
{
  "question_id": "q001",
  "research_question": "How can retrieval-augmented LLM agents generate more grounded and testable scientific hypotheses?"
}
```

如果某个问题没有整理好的可见论文，也可以先让三种方法只基于同一个问题生成 idea。这样能先跑通 Idea Arena。HindSight 评估再为该问题单独补充 cutoff 后论文池。

### 6.2 问题来源

问题来源分两类即可：

1. **我们提供的问题**  
   由我们手工整理，优先覆盖 IdeaGraph 关心的方向，例如 交通调度之类。

2. **从最新发表论文中收集的问题**  
   从近期 AI / CS 论文的 title、abstract、introduction、limitations 和 future work 中抽取 research questions。可以参考 Hugging Face Daily Papers、arXiv cs.AI / cs.CL / cs.LG / cs.IR、OpenReview 近期会议论文，以及 Semantic Scholar / OpenAlex 检索结果。

### 6.3 可见论文

可见论文不是强制数据字段，而是为了让三种方法有相同背景。第一版可以按问题灵活提供：

- 如果我们已经有 seed papers，就给每个问题 3-10 篇 cutoff 前论文的 title + abstract。
- 如果问题来自某篇新论文，就把该论文的 references 或相关工作中 cutoff 前论文作为可见材料。
- 如果暂时没有整理论文，只给 research question，先做 Idea Arena。

公平性要求只有一条：同一个问题下，三种方法看到的 background 和 visible papers 必须完全相同。

为了让三种方法的“可见性”一致，可以把每个问题的 visible papers 聚在一起，构造成一个独立的 question-specific 检索库或轻量搜索引擎。这个检索库只包含该问题允许看到的 cutoff 前论文，提供统一的 title / abstract / keyword / embedding search 接口。IdeaGraph、AI-Researcher 和 ResearchAgent 如果需要检索论文，都只能访问这个独立检索库，而不能访问全量 Semantic Scholar、OpenAlex、arXiv 或各自默认数据库。

示例：

```text
question q001
-> visible_papers_q001.jsonl
-> local_search_index_q001
-> three methods query the same local_search_index_q001
```

这样可以避免某个 baseline 因为默认检索范围更大而看到额外论文，也方便记录每个问题实际暴露给方法的论文集合。

### 6.4 HindSight future paper pool

HindSight 需要额外为每个问题收集一小批 cutoff 后真实论文，但这些论文只用于评估，不能给生成方法看到。

future paper pool 的构造方式：

1. 根据 research question 设计检索词。
2. 在 Semantic Scholar、OpenAlex、OpenReview、arXiv 中检索 cutoff date 之后发表的相关论文。
3. 保留 title、abstract、publication date、venue、citation count。
4. 标记是否为顶会论文，例如 ICLR、NeurIPS、ICML、ACL、EMNLP、CVPR、AAAI。
5. 冻结采集日期，因为 citation count 会随时间变化。

这部分不需要大规模。只要每个问题能收集到若干相关 post-cutoff 论文，就可以用于 HindSight-style matching。没有 future papers 的问题仍可参与 Idea Arena，但不进入 HindSight 主统计。

## 7. 统一生成协议

### 7.1 Shared input

三种方法必须看到相同输入。推荐统一输入格式：

```json
{
  "question_id": "q001",
  "research_question": "...",
  "background": "... optional",
  "cutoff_date": "2024-01-01 optional",
  "visible_papers": [
    {
      "paper_id": "... optional",
      "title": "...",
      "abstract": "...",
      "year": 2023
    }
  ]
}
```

第一版可以不强制 `visible_papers`。如果某个问题只有 `research_question`，三种方法就都只使用该问题本身。如果某个问题有 visible papers，则建议同时提供一个该问题专属的本地检索接口；三种方法都通过这个接口查论文，保证检索空间一致。

### 7.2 Unified output

三种方法都输出 3 个 idea，每个 idea 使用统一 schema：

```json
{
  "title": "...",
  "problem": "...",
  "existing_methods": "...",
  "motivation": "...",
  "proposed_method": "...",
  "experiment_plan": "...",
  "expected_results": "...",
  "key_difference_from_prior_work": "...",
  "supporting_papers": [
    {
      "paper_id": "...",
      "claim_supported": "..."
    }
  ],
  "risks_or_limitations": "..."
}
```

该 schema 对齐 Chain-of-Ideas 的 idea + experiment design 评估，也保留 HindSight 所需的 `problem + proposed_method` semantic matching 字段。

### 7.3 IdeaGraph

IdeaGraph 使用同一个 research question 运行 `chat-idea`。如果该问题有 visible papers，可以先构建局部图谱；如果没有 visible papers，则使用当前图谱和检索流程，但必须记录实际可见信息。

记录：

- graph statistics：Paper、CITES、Entity、Entity relation 数量；
- `tool_trace`；
- `graph_context`；
- supporting papers；
- generated ideas；
- 每个 idea 使用到的 graph paths。

### 7.4 AI-Researcher

AI-Researcher 使用 reference-based ideation 模式。它的论文输入主要来自我们提供的 reference papers。实验时，把每个问题的 visible papers 写成它需要的 reference-paper 输入；如果它需要进一步读取论文全文，也只允许从这些 visible papers 对应的本地文件或本地检索库中读取。

检索设置：

- `source_papers` 来自该问题的 visible papers。
- `reference_papers` 只能从 `source_papers` 中选择。
- 论文全文读取只使用 visible papers 的本地副本。
- 不允许通过 OpenAlex / arXiv 查找 shared input 之外的新论文。

记录：

```json
{
  "repo": "https://github.com/HKUDS/AI-Researcher",
  "mode": "reference_based_ideation",
  "paper_source": "visible_papers_only",
  "external_paper_search_allowed": false
}
```

### 7.5 ResearchAgent

ResearchAgent 使用 core paper + reference retrieval + knowledge store + reviewer refinement。它的检索方式是先给定 core papers，再从这些 core papers 的 references 中找相关论文。实验时，从每个问题的 visible papers 中选 1-3 篇作为 core papers；后续 reference retrieval 也只能在该问题的 visible papers 内完成。

检索设置：

- core papers 来自该问题的 visible papers。
- retrieved references 来自该问题的 visible papers。
- knowledge store 只由该问题的 visible papers 构建。
- 不允许通过 OpenAlex / Semantic Scholar 扩展到 visible papers 之外的论文。

记录：

```json
{
  "repo": "https://github.com/JinheonBaek/ResearchAgent",
  "core_papers": ["... optional"],
  "reference_source": "visible_papers_only",
  "knowledge_store_source": "visible_papers_only"
}
```

## 8. 数据披露与泄漏控制

### 8.1 主实验规则

主实验必须满足：

1. 三种方法看到相同 research question、background 和 visible papers。
2. 如果某个问题没有 visible papers，三种方法都不能额外使用可见论文。
3. cutoff date 之后的论文不能进入生成过程。
4. 如果方法需要搜索论文，只能访问该问题对应的本地检索库。
5. AI-Researcher 不能通过 OpenAlex/arXiv 下载 shared input 之外的论文全文。
6. ResearchAgent 不能通过 OpenAlex / Semantic Scholar API 扩展到 shared input 之外的 references。
7. 如果保留 GitHub repo search，必须记录 query 和 repo 结果，并确认没有引入额外论文。
8. AI-Researcher 的代码执行和论文写作不进入主评分。
9. judge 只能看到匿名 idea、实验设计和评价 rubric。
10. judge 不能联网搜索。
11. 所有输出在评估前冻结。

每次运行保存：

```json
{
  "method": "IdeaGraph",
  "question_id": "q001",
  "setting": "shared_input",
  "cutoff_date": "2024-01-01 optional",
  "visible_paper_count": 0,
  "future_papers_visible_during_generation": false,
  "external_search_allowed": false,
  "generation_model": "...",
  "judge_model": "...",
  "code_commit": "...",
  "run_timestamp": "..."
}
```

### 8.2 Future paper leakage

HindSight-style 评估最重要的风险是 future paper leakage。

必须检查：

| 检查项 | 方法 |
|---|---|
| future paper title 是否出现在输入 | 字符串匹配 |
| future paper abstract 是否混入 visible papers | embedding / fuzzy matching |
| post-cutoff paper 是否在生成输入中 | paper_id + publication_date 检查 |
| output 是否近似复述 future paper | title / abstract / method 相似度 |
| baseline 原始数据库是否含 future paper | 运行前过滤或记录风险 |

如果某个 research question 发现泄漏：

- 从 HindSight 主统计剔除，或
- 标记为 contaminated，只做定性分析。

### 8.3 Native-corpus 补充实验

Native-corpus 可以让各方法按默认方式运行，但必须单独报告：

- 是否联网检索；
- 使用了哪个数据库；
- 数据库覆盖到什么时间；
- 是否可能包含 future papers；
- 是否执行代码；
- 输出中引用了哪些 shared input 之外的论文。

Native-corpus 结果只能回答“真实默认使用时哪个系统更强”，不能作为“方法机制更强”的主证据。


## 9. 统计与结果呈现

### 9.1 Idea Arena 主表

Overall 表：

| 方法 | Overall Elo | Overall win rate |
|---|---:|---:|
| IdeaGraph | | |
| AI-Researcher | | |
| ResearchAgent | | |

维度表：

| 维度 | IdeaGraph Elo | IdeaGraph win rate | AI-Researcher Elo | AI-Researcher win rate | ResearchAgent Elo | ResearchAgent win rate |
|---|---:|---:|---:|---:|---:|---:|
| Novelty | | | | | | |
| Significance | | | | | | |
| Clarity | | | | | | |
| Feasibility | | | | | | |
| Expected Effectiveness | | | | | | |
| Experiment Feasibility | | | | | | |
| Technical Quality | | | | | | |
| Experiment Clarity | | | | | | |

### 9.2 HindSight 主表

| 方法 | Mean score | Median score | Match rate | Avg matches | Best-of-3 score | Question win rate |
|---|---:|---:|---:|---:|---:|---:|
| IdeaGraph | | | | | | |
| AI-Researcher | | | | | | |
| ResearchAgent | | | | | | |

### 9.3 方法两两对比表

| 对比 | Arena winner | HindSight winner | 主要差异 |
|---|---|---|---|
| IdeaGraph vs AI-Researcher | | | |
| IdeaGraph vs ResearchAgent | | | |
| AI-Researcher vs ResearchAgent | | | |

### 9.4 Correlation analysis

参考 HindSight，分析 objective impact 与 judge preference 是否一致：

| 分析 | 目的 |
|---|---|
| HindSight score vs Novelty win | judge 是否高估新奇但不落地的 idea |
| HindSight score vs Feasibility win | 可行性是否更接近真实未来研究 |
| HindSight score vs Expected effectiveness | judge 预期影响是否与真实影响一致 |

### 9.5 Case study

选择少量案例：

1. High HindSight + high Arena：两个指标都认可。
2. High HindSight + low Arena：judge 低估但实际匹配未来研究。
3. Low HindSight + high Arena：看起来很新颖但未匹配未来研究。
4. IdeaGraph 赢的案例。
5. AI-Researcher 赢的案例。
6. ResearchAgent 赢的案例。

每个 case 展示：

- research question；
- 三种方法的 best idea；
- matched future paper；
- HindSight score；
- Arena judgment；
- 为什么某方法赢。

## 10. 参考资料

- HindSight: [HindSight: Evaluating LLM-Generated Research Ideas via Future Impact](https://arxiv.org/abs/2603.15164)
- Chain-of-Ideas: [Chain of Ideas: Revolutionizing Research via Novel Idea Development with LLM Agents](https://arxiv.org/abs/2410.13185)
- Chain-of-Ideas code: [DAMO-NLP-SG/CoI-Agent](https://github.com/DAMO-NLP-SG/CoI-Agent)
- AI-Researcher code: [HKUDS/AI-Researcher](https://github.com/HKUDS/AI-Researcher)
- ResearchAgent code: [JinheonBaek/ResearchAgent](https://github.com/JinheonBaek/ResearchAgent)
- ResearchAgent paper: [Iterative Research Idea Generation over Scientific Literature with Large Language Models](https://aclanthology.org/2025.naacl-long.342/)
- Existing IdeaGraph related-work report: [scientific-idea-generation-related-work.md](scientific-idea-generation-related-work.md)
