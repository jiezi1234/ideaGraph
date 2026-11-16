# LinearRAG 相关论文的 OpenAlex 链接

> 说明：下面整理的是前面用于逆向 LinearRAG 的核心论文种子。  
> OpenAlex 页面链接统一使用 `https://openalex.org/W...`。

| 角色 | 论文 | OpenAlex |
|---|---|---|
| 目标论文 | LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora | https://openalex.org/W4417100168 |
| 基础方法 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | https://openalex.org/W3098425262 |
| 图检索主线 | Retrieval-Augmented Generation with Graphs (GraphRAG) | https://openalex.org/W4406031095 |
| 图检索综述 | A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models | https://openalex.org/W4406840433 |
| 评测基准 | GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation | https://openalex.org/W4417307696 |
| 层次检索 | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | https://openalex.org/W4391463108 |
| 轻量图检索 | LightRAG: Simple and Fast Retrieval-Augmented Generation | https://openalex.org/W4416034797 |
| 长期记忆 / 多跳检索 | HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models | https://openalex.org/W4398886972 |
| HippoRAG2 对应条目 | From RAG to Memory: Non-Parametric Continual Learning for Large Language Models | https://openalex.org/W4407831980 |
| 图理解检索 | G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering | https://openalex.org/W4391799277 |
| 图基础模型路线 | GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation | https://openalex.org/W4407129024 |
| 轻量 GraphRAG 变体 | E^2GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness | https://openalex.org/W4414888065 |
| 何时使用图 | When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation | https://openalex.org/W4417097190 |

## 建图建议

如果你要把这些文献作为 LinearRAG 的 seed graph，建议至少分成三层：

1. **方法主线**：RAG → GraphRAG → LinearRAG
2. **对照方法**：RAPTOR / LightRAG / HippoRAG / G-Retriever / GFM-RAG / E^2GraphRAG
3. **证据与评测**：GraphRAG-Bench / When to use Graphs in RAG / GraphRAG survey

## 建图命令（不含目标论文）

下面命令排除了表格中的“目标论文” LinearRAG，只将其余 OpenAlex 论文作为 seed：

```bash
uv run python main.py build-openalex-seed-graph \
  --openalex-id https://openalex.org/W3098425262 \
  --openalex-id https://openalex.org/W4406031095 \
  --openalex-id https://openalex.org/W4406840433 \
  --openalex-id https://openalex.org/W4417307696 \
  --openalex-id https://openalex.org/W4391463108 \
  --openalex-id https://openalex.org/W4416034797 \
  --openalex-id https://openalex.org/W4398886972 \
  --openalex-id https://openalex.org/W4407831980 \
  --openalex-id https://openalex.org/W4391799277 \
  --openalex-id https://openalex.org/W4407129024 \
  --openalex-id https://openalex.org/W4414888065 \
  --openalex-id https://openalex.org/W4417097190
```

## 备注

- `HippoRAG2` 在 OpenAlex 中更接近对应到 **From RAG to Memory: Non-Parametric Continual Learning for Large Language Models**。
- `E2 GraphRAG` 在 OpenAlex 中对应的规范标题是 **E^2GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness**。
