# AI-Researcher / ResearchAgent 公交调度本地对比运行手册

本手册记录当前在 IdeaGraph 仓库内复跑三系统 idea generation 对比的命令。示例问题：

```text
利用进化神经网络解决公交车辆调度问题，有哪些创新想法？
```

目标论文目录：

```text
/home/zb/code/IdeaGraph/data/testbus/A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling
```

## 1. 准备共享本地论文输入

先把本地 PDF 论文解析成 AI-Researcher 和 ResearchAgent 都能消费的输入。该脚本会读取 `.env` 中的 Neo4j 配置；如果图谱中已有该论文实体，会把实体一起写入 ResearchAgent 本地上下文。

```bash
cd /home/zb/code/IdeaGraph

QUESTION="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？"
TARGET_PAPER_DIR="/home/zb/code/IdeaGraph/data/testbus/A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling"
OUTPUT_DIR="/home/zb/code/IdeaGraph/exports/testbus-local-baselines"

python3 scripts/prepare_local_baseline_inputs.py \
  --paper-dir "$TARGET_PAPER_DIR" \
  --question "$QUESTION" \
  --output-dir "$OUTPUT_DIR" \
  --category bus_sched \
  --instance-id bus_evo_nn
```

主要输出：

- `exports/testbus-local-baselines/ai_researcher/`
- `exports/testbus-local-baselines/researchagent/local_contexts.jsonl`
- `exports/testbus-local-baselines/manifest.json`

## 2. 运行三个系统

### 2.1 IdeaGraph

`build-local-paper-graph` 会使用本地论文文件夹名作为 `Paper.id`。如果要让 `chat-idea` 固定以这篇论文作为生成依据，`--seed-paper-id` 需要传同一个 id。

```bash
cd /home/zb/code/IdeaGraph

uv run python main.py build-local-paper-graph \
  --paper-dir "$TARGET_PAPER_DIR"

uv run python main.py chat-idea \
  --query "$QUESTION" \
  --seed-paper-id "A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling" \
  --archive-output-dir "$OUTPUT_DIR/ideagraph"
```

`chat-idea` 会在 `$OUTPUT_DIR/ideagraph/` 下写出 IdeaGraph archive JSON。后续标准化时传入具体 archive 文件路径。

### 2.2 AI-Researcher

使用本仓库的本地论文 idea-agent adapter。它复用 AI-Researcher 原始 idea agent 和本地文件浏览工具，但只做论文阅读、idea 生成和最终 idea 选择，不进入代码实现和实验阶段。

```bash
cd /home/zb/code/IdeaGraph

API_BASE_URL=http://localhost:11435 \
CHEEP_MODEL=ollama_chat/qwen3.6:35b \
/home/zb/code/AI-Researcher/.venv/bin/python \
  scripts/run_ai_researcher_local_idea_agent.py \
  --ai-researcher-root /home/zb/code/AI-Researcher \
  --input-dir "$OUTPUT_DIR/ai_researcher" \
  --category bus_sched \
  --instance-id bus_evo_nn \
  --task-level task2
```

输出：

```text
exports/testbus-local-baselines/ai_researcher/outputs/local_idea_agent/ideas_bus_sched_bus_evo_nn.json
exports/testbus-local-baselines/ai_researcher/outputs/local_idea_agent/ideas_bus_sched_bus_evo_nn.md
```

标准化时，AI-Researcher adapter 会从 `selected_idea` 中解析 `Idea X`，再取 `ideas[X-1]` 作为 `original`。解析不到编号或编号越界会直接报错。

### 2.3 ResearchAgent

使用本仓库的 local-context runner，跳过 ResearchAgent 原始 OpenAlex 检索入口，直接把准备好的本地论文上下文喂给它的 `Problem -> Method -> Experiment` 生成管线。

```bash
cd /home/zb/code/IdeaGraph

PYTHONPATH=/home/zb/code/ResearchAgent/code \
/home/zb/code/ResearchAgent/.venv/bin/python \
  scripts/run_researchagent_local_context.py \
  --context-jsonl "$OUTPUT_DIR/researchagent/local_contexts.jsonl" \
  --results-path "$OUTPUT_DIR/researchagent/ideas.jsonl" \
  --researchagent-code-dir /home/zb/code/ResearchAgent/code \
  --model-name qwen3.6:35b \
  --iterations 3
```

输出：

```text
exports/testbus-local-baselines/researchagent/ideas.jsonl
```

`--iterations` 是 ResearchAgent 每个阶段生成和验证的迭代次数；正式对比建议用 `3`，冒烟可用 `1`。

## 3. 标准化输出

盲评脚本只读取统一的 `method_idea_v1` JSON，不直接解析三个系统的原始输出。

标准化后的 judge 输入规则：

- 如果 `original` 非空，judge 只看 `Original`。
- 如果 `original` 为空，judge 只看 `Title`、`Summary`、`Problem`、`Method`、`Experiment`。
- `source_payload` 保留原始结果用于追溯，但不进入盲评 prompt。

```bash
cd /home/zb/code/IdeaGraph

python3 scripts/normalize_method_idea_outputs.py \
  --input exports/chat_idea_runs/20260604-055159-query-c194029a3dcd.json \
  --output-json exports/testbus-method-comparison/normalized/ideagraph.json \
  --method-name IdeaGraph \
  --question "$QUESTION"

python3 scripts/normalize_method_idea_outputs.py \
  --input "$OUTPUT_DIR/researchagent/ideas.jsonl" \
  --output-json exports/testbus-method-comparison/normalized/researchagent.json \
  --method-name ResearchAgent \
  --question "$QUESTION"

python3 scripts/normalize_method_idea_outputs.py \
  --input "$OUTPUT_DIR/ai_researcher/outputs/local_idea_agent/ideas_bus_sched_bus_evo_nn.json" \
  --output-json exports/testbus-method-comparison/normalized/ai_researcher.json \
  --method-name AI-Researcher \
  --question "$QUESTION"
```

注意：IdeaGraph 的 `--input` 应替换成你本次 `chat-idea` 实际生成的 archive JSON。上面的 `exports/chat_idea_runs/20260604-055159-query-c194029a3dcd.json` 是已跑过的示例文件。

## 4. 运行盲评

三个系统一起盲评时，脚本不是一次比较三个，而是生成所有两两组合：

```text
IdeaGraph vs ResearchAgent
IdeaGraph vs AI-Researcher
ResearchAgent vs AI-Researcher
```

每个 pair 都正序和反序各评一次，用于检查 A/B 顺序敏感性。

```bash
cd /home/zb/code/IdeaGraph

python3 scripts/compare_normalized_method_ideas.py \
  --idea-json exports/testbus-method-comparison/normalized/ideagraph.json \
  --idea-json exports/testbus-method-comparison/normalized/researchagent.json \
  --idea-json exports/testbus-method-comparison/normalized/ai_researcher.json \
  --question "$QUESTION" \
  --output-json exports/testbus-method-comparison/blind_comparison_3methods.json \
  --output-md exports/testbus-method-comparison/blind_comparison_3methods.md
```

输出：

- `blind_comparison_3methods.json`：完整机器可读结果，包含每个 pair 的正反序结果、维度胜者和顺序敏感性。
- `blind_comparison_3methods.md`：便于人工查看的摘要。

## 5. 本地文件说明

以下目录是运行产物，不应提交：

- `exports/`
- `data/`
- `paper_db/`
- `.cache/`
- `logs/`
