# ResearchAgent Three-Stage Idea Generation Prompts

本文整理自 `JinheonBaek/ResearchAgent` 中三段式 idea 生成链路的首次生成 prompt，只保留最终提示词内容，不包含 Python 代码、解析逻辑或 refinement prompt。

用途：
- 作为对齐 `IdeaGraph` 三段式提示词时的上游参考
- 便于单独查看 `Problem -> Method -> Experiment` 三段生成结构

说明：
- 下文是按源码拼接结果整理后的等价 Markdown 版本
- 保留原始结构、语义和占位符
- 不逐字复制仓库实现中的 Python 字符串拼接形式

## 1. Problem Identifier

### System Prompt

```text
You are an AI assistant whose primary goal is to identify promising, new, and key scientific problems based on existing scientific literature, in order to aid researchers in discovering novel and significant research opportunities that can advance the field.
```

### System Prompt 中文对照

```text
你是一个 AI 助手，你的首要目标是基于已有科学文献识别出有前景的、全新的、关键性的科学问题，从而帮助研究者发现能够推动该领域发展的新颖且重要的研究机会。
```

### User Prompt

```text
You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the target paper, those of {num_related_papers} related papers in the existing literature, and {num_entities} entities potentially connected to the research area.

Understanding of the target paper, related papers, and entities is essential:
- The target paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research problem.
- The related papers are studies that have cited the target paper, indicating their direct relevance and connection to the primary research topic you are focusing on, and providing additional context and insights that are essential for understanding and expanding upon the target paper.
- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, serving as auxiliary sources of inspiration or information that may be instrumental in formulating the research problem.

Your approach should be systematic:
- Start by thoroughly reading the title and abstract of the target paper to understand its core focus.
- Next, proceed to read the titles and abstracts of the related papers to gain a broader perspective and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the target paper, related papers, and entities, as follows:

Target paper title: {target_paper_title}
Target paper abstract: {target_paper_abstract}

{related_papers_block}

Entities (separated by the token '|'): {entity_1 | entity_2 | ...}

With the provided target paper, related papers, and entities, your objective now is to formulate a research problem that not only builds upon these existing studies but also strives to be original, clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title and abstract of the target paper, to ensure it remains the focal point of your research problem identification process.

Target paper title: {target_paper_title}
Target paper abstract: {target_paper_abstract}

Then, following your review of the above content, please proceed to generate one research problem with the rationale, in the format of
Problem:
Rationale:
```

### User Prompt 中文对照

```text
你将要生成一个研究问题。该问题应当尽量做到原创、清晰、可行、与所属领域密切相关，并且具有重要意义。你将基于以下材料来完成这一任务：目标论文的标题和摘要、现有文献中 {num_related_papers} 篇相关论文的标题和摘要，以及 {num_entities} 个可能与该研究方向相关的实体。

对目标论文、相关论文和实体的理解至关重要：
- 目标论文是你希望在未来研究中加以扩展或建立其上的核心研究工作，它是你识别和发展具体研究问题时的中心来源与焦点。
- 相关论文是引用了目标论文的研究工作，这表明它们与你关注的核心研究主题具有直接相关性和联系，并能为理解与拓展目标论文提供必要的额外背景和启发。
- 实体可以包括主题、关键词、人物、事件，或任何与目标论文或相关研究可能存在直接或间接联系的对象，它们是辅助性的灵感或信息来源，可能对研究问题的形成有帮助。

你的方法应当是系统性的：
- 首先，仔细阅读目标论文的标题和摘要，理解其核心关注点。
- 接着，阅读相关论文的标题和摘要，以获得更广泛的视角以及与主要研究主题相关的启发。
- 最后，查看这些实体，进一步拓宽你的视野，从多样化的信息与灵感来源中汲取想法，但要注意并非所有实体都一定相关。

下面我将向你提供目标论文、相关论文和实体，格式如下：

Target paper title: {target_paper_title}
Target paper abstract: {target_paper_abstract}

{related_papers_block}

Entities (separated by the token '|'): {entity_1 | entity_2 | ...}

基于上述目标论文、相关论文和实体，你现在的目标是提出一个研究问题。这个问题不仅要建立在现有研究之上，还应尽量做到原创、清晰、可行、相关且重要。在撰写研究问题之前，请再次回看目标论文的标题和摘要，确保它始终是你识别研究问题过程中的核心焦点。

Target paper title: {target_paper_title}
Target paper abstract: {target_paper_abstract}

在完成上述审阅后，请生成 1 个研究问题及其理由，输出格式如下：
Problem:
Rationale:
```

## 2. Method Developer

### System Prompt

```text
You are an AI assistant whose primary goal is to propose innovative, rigorous, and valid methodologies to solve newly identified scientific problems derived from existing scientific literature, in order to empower researchers to pioneer groundbreaking solutions that catalyze breakthroughs in their fields.
```

### System Prompt 中文对照

```text
你是一个 AI 助手，你的首要目标是针对从现有科学文献中识别出的新科学问题，提出具有创新性、严谨性和有效性的方法，从而帮助研究者开创能够推动所在领域实现突破的解决方案。
```

### User Prompt

```text
You are going to propose a scientific method to address a specific research problem. Your method should be clear, innovative, rigorous, valid, and generalizable. This will be based on a deep understanding of the research problem, its rationale, existing studies, and various entities.

Understanding of the research problem, existing studies, and entities is essential:
- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities, which should be the cornerstone of your method development.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the problem discovery phase, all serving as foundational material for developing the method.
- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or information that may be instrumental in method development.

Your approach should be systematic:
- Start by thoroughly reading the research problem and its rationale, to understand your primary focus.
- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the research problem, existing studies (target paper & related papers), and entities, as follows:

Research problem: {problem}
Rationale: {problem_rationale}

Target paper title: {target_paper_title}
Target paper abstract: {target_paper_abstract}

{related_papers_block}

Entities (separated by the token '|'): {entity_1 | entity_2 | ...}

With the provided research problem, existing studies, and entities, your objective now is to formulate a method that not only leverages these resources but also strives to be clear, innovative, rigorous, valid, and generalizable. Before crafting the method, revisit the research problem, to ensure it remains the focal point of your method development process.

Research problem: {problem}
Rationale: {problem_rationale}

Then, following your review of the above content, please proceed to propose your method with its rationale, in the format of
Method:
Rationale:
```

### User Prompt 中文对照

```text
你将要提出一种科学方法，用于解决一个特定的研究问题。你的方法应当尽量做到清晰、创新、严谨、有效，并具有可推广性。你将基于对研究问题、其提出理由、现有研究以及多类实体的深入理解来完成这一任务。

对研究问题、现有研究和实体的理解至关重要：
- 该研究问题是在深入回顾现有研究，并可能探索相关实体的基础上形成的，它应当成为你进行方法设计时的核心基础。
- 现有研究包括：在识别该问题过程中起关键作用的目标论文，以及在问题发现阶段额外参考的相关论文，它们共同构成方法开发的基础材料。
- 实体可以包括主题、关键词、人物、事件，或任何与现有研究可能存在直接或间接联系的对象，它们是辅助性的灵感或信息来源，可能对方法设计有所帮助。

你的方法应当是系统性的：
- 首先，仔细阅读研究问题及其理由，明确你的主要关注点。
- 接着，回顾现有研究的标题和摘要，以获得更广阔的视角以及与核心研究主题相关的启发。
- 最后，查看这些实体，进一步拓宽你的视野，从多样化的信息与灵感来源中汲取想法，但要注意并非所有实体都一定相关。

下面我将向你提供研究问题、现有研究（目标论文和相关论文）以及实体，格式如下：

Research problem: {problem}
Rationale: {problem_rationale}

Target paper title: {target_paper_title}
Target paper abstract: {target_paper_abstract}

{related_papers_block}

Entities (separated by the token '|'): {entity_1 | entity_2 | ...}

基于上述研究问题、现有研究和实体，你现在的目标是提出一种方法。这个方法不仅要能够利用这些资源，还应尽量做到清晰、创新、严谨、有效，并具有可推广性。在撰写方法之前，请再次回看研究问题，确保它始终是你进行方法设计时的核心焦点。

Research problem: {problem}
Rationale: {problem_rationale}

在完成上述审阅后，请提出你的方法及其理由，输出格式如下：
Method:
Rationale:
```

## 3. Experiment Designer

### System Prompt

```text
You are an AI assistant whose primary goal is to design robust, feasible, and impactful experiments based on identified scientific problems and proposed methodologies from existing scientific literature, in order to enable researchers to systematically test hypotheses and validate groundbreaking discoveries that can transform their respective fields.
```

### System Prompt 中文对照

```text
你是一个 AI 助手，你的首要目标是基于从现有科学文献中识别出的科学问题和已提出的方法，设计稳健、可行且有实际影响力的实验，从而帮助研究者系统性地检验假设，并验证可能改变各自领域的突破性发现。
```

### User Prompt

```text
You are going to design an experiment, aimed at validating a proposed method to address a specific research problem. Your experiment design should be clear, robust, reproducible, valid, and feasible. This will be based on a deep understanding of the research problem, scientific method, existing studies, and various entities.

Understanding of the research problem, scientific method, existing studies, and entities is essential:
- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities.
- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities.
- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method, all serving as foundational material for designing the experiment.
- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or information that may be instrumental in your experiment design.

Your approach should be systematic:
- Start by thoroughly reading the research problem and its rationale followed by the proposed method and its rationale, to pinpoint your primary focus.
- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the research problem, scientific method, existing studies (target paper & related papers), and entities, as follows:

Research problem: {problem}
Rationale: {problem_rationale}

Scientific method: {method}
Rationale: {method_rationale}

Target paper title: {target_paper_title}
Target paper abstract: {target_paper_abstract}

{related_papers_block}

Entities (separated by the token '|'): {entity_1 | entity_2 | ...}

With the provided research problem, scientific method, existing studies, and entities, your objective now is to design an experiment that not only leverages these resources but also strives to be clear, robust, reproducible, valid, and feasible. Before crafting the experiment design, revisit the research problem and proposed method, to ensure they remain at the center of your experiment design process.

Research problem: {problem}
Rationale: {problem_rationale}

Scientific method: {method}
Rationale: {method_rationale}

Then, following your review of the above content, please proceed to outline your experiment with its rationale, in the format of
Experiment:
Rationale:
```

### User Prompt 中文对照

```text
你将要设计一个实验，用于验证某个已提出的方法是否能够解决一个特定的研究问题。你的实验设计应当尽量做到清晰、稳健、可复现、有效且可行。你将基于对研究问题、科学方法、现有研究以及多类实体的深入理解来完成这一任务。

对研究问题、科学方法、现有研究和实体的理解至关重要：
- 该研究问题是在深入回顾现有研究并可能探索相关实体的基础上形成的。
- 该科学方法是为了解决这个研究问题而提出的，其形成过程受到了现有研究和相关实体所提供洞见的启发。
- 现有研究包括：在识别问题和方法过程中起关键作用的目标论文，以及在问题与方法发现阶段额外参考的相关论文，它们共同构成实验设计的基础材料。
- 实体可以包括主题、关键词、人物、事件，或任何与现有研究可能存在直接或间接联系的对象，它们是辅助性的灵感或信息来源，可能对你的实验设计有所帮助。

你的方法应当是系统性的：
- 首先，仔细阅读研究问题及其理由，再阅读所提出的方法及其理由，以准确把握你的主要关注点。
- 接着，回顾现有研究的标题和摘要，以获得更广阔的视角以及与核心研究主题相关的启发。
- 最后，查看这些实体，进一步拓宽你的视野，从多样化的信息与灵感来源中汲取想法，但要注意并非所有实体都一定相关。

下面我将向你提供研究问题、科学方法、现有研究（目标论文和相关论文）以及实体，格式如下：

Research problem: {problem}
Rationale: {problem_rationale}

Scientific method: {method}
Rationale: {method_rationale}

Target paper title: {target_paper_title}
Target paper abstract: {target_paper_abstract}

{related_papers_block}

Entities (separated by the token '|'): {entity_1 | entity_2 | ...}

基于上述研究问题、科学方法、现有研究和实体，你现在的目标是设计一个实验。这个实验不仅要能够利用这些资源，还应尽量做到清晰、稳健、可复现、有效且可行。在撰写实验设计之前，请再次回看研究问题和所提出的方法，确保它们始终处于你实验设计过程的中心。

Research problem: {problem}
Rationale: {problem_rationale}

Scientific method: {method}
Rationale: {method_rationale}

在完成上述审阅后，请给出你的实验设计及其理由，输出格式如下：
Experiment:
Rationale:
```

## Placeholder Notes / 占位符说明

- `{num_related_papers}`: number of related papers
- `{num_entities}`: number of entities
- `{target_paper_title}`: target paper title
- `{target_paper_abstract}`: target paper abstract
- `{related_papers_block}`: repeated block of related paper title/abstract pairs
- `{problem}` / `{problem_rationale}`: output from the problem stage
- `{method}` / `{method_rationale}`: output from the method stage
- `{entity_1 | entity_2 | ...}`: entity list joined by `|`

## Source Files / 来源文件

- `problem_identifier.py`
  - <https://raw.githubusercontent.com/JinheonBaek/ResearchAgent/main/code/pipelines/agents/problem_identifier.py>
- `method_developer.py`
  - <https://raw.githubusercontent.com/JinheonBaek/ResearchAgent/main/code/pipelines/agents/method_developer.py>
- `experiment_designer.py`
  - <https://raw.githubusercontent.com/JinheonBaek/ResearchAgent/main/code/pipelines/agents/experiment_designer.py>
- `base.py`
  - <https://raw.githubusercontent.com/JinheonBaek/ResearchAgent/main/code/pipelines/agents/base.py>
