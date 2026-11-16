# IdeaGraph Architecture Design

- Date: 2026-04-21
- Status: Draft
- Scope: High-level architecture and core system objects for v1

## 1. Context

IdeaGraph is a frontier-technology discovery and idea-generation system. It takes user needs, technical topics, or natural-language questions as input, retrieves relevant knowledge from a graph-backed knowledge base, and produces structured idea candidates through generation and evaluation.

At the current stage, the project direction is:

- first complete high-level logical architecture design
- then refine into implementable module boundaries
- architecture is service-oriented, but early validation can be driven by CLI/scripts

## 2. v1 Scope

The system keeps three user-facing modes, but does not split them into three different pipelines:

1. reverse mining of technical routes from capability needs
2. forward discovery of potential application scenarios from frontier technologies
3. combinational innovation after decomposing user requirements

For v1, these three modes share the same core processing chain. Their difference is mainly in prompt/profile and query strategy, not in system architecture.

## 3. Confirmed Requirements

### 3.1 P0

- multi-source ingestion
- model decoupling and replaceability
- Neo4j as graph storage
- configurable prompts
- configurable filtering and reranking strategy
- unified configuration system
- unified output schema

### 3.2 P1

- traceable output at document/path level
- task-style execution for ingestion and processing jobs
- failure recovery and idempotent processing

### 3.3 P2

- optional relational database support in the future

### 3.4 Explicit Non-Goals for v1

- no version-management subsystem for prompts/parsing/indexing
- no chunk-based full-text processing in the initial implementation
- no dedicated evidence object in the initial implementation

## 4. Architecture Principles

1. Single shared main pipeline
   All three business modes share one pipeline: input understanding, query planning, graph query, generation, evaluation, output.

2. External capability decoupling
   Models, data sources, graph storage, and future relational storage are accessed through abstractions instead of being hardcoded into business logic.

3. Graph as the core knowledge substrate
   Neo4j is the primary structured knowledge store. Idea generation should be organized around graph query results instead of raw free-form document stuffing.

4. Configuration over hardcoding
   Prompt selection, model selection, filtering/reranking strategy, running mode, online expansion switch, query parameters, and data-source settings should all be configurable.

5. Service-oriented design, script-driven validation
   The system is designed as an API-oriented backend, while CLI/scripts can drive the internal workflow during early development and testing.

6. Simplicity first
   v1 should prioritize a short and reliable end-to-end chain, with clear boundaries and replaceable components, instead of introducing heavyweight orchestration too early.

## 5. High-Level Module Boundaries

The system is organized as four core layers plus one cross-cutting layer.

### 5.1 Access Layer

Responsibilities:

- expose HTTP API
- expose CLI/script entrypoints
- receive future frontend requests
- validate input and trigger workflows/tasks

Non-responsibilities:

- no core business logic
- no direct graph operations

### 5.2 Idea Workflow Layer

Responsibilities:

- input understanding
- query-plan generation
- calling the knowledge query layer
- generation context construction
- candidate idea generation
- idea evaluation
- iterative revision based on low-score dimensions
- final output formatting

Notes:

- the three business modes are represented here as workflow profiles or prompt profiles
- this layer is an orchestrator, not a storage layer

### 5.3 Knowledge Query Layer

Responsibilities:

- keyword and semantic retrieval
- entity matching and normalization
- neighbor expansion
- path querying
- query-result filtering and reranking
- configurable online expansion

Contract:

- returns unified query results to the workflow layer
- hides Neo4j and other retrieval implementation details from upper layers

Notes:

- online expansion must be configurable
- the system must support both online-expand mode and offline-only mode for restricted environments
- filtering and reranking should be implemented behind a replaceable strategy interface instead of being tied to a single formula or policy

### 5.4 Ingestion and Graph-Building Layer

Responsibilities:

- connect to multiple data sources
- standardize documents
- extract structured objects
- write graph data into Neo4j
- update retrieval indexes
- manage retries, idempotency, and task status

Notes:

- this layer produces knowledge
- it does not generate ideas

### 5.5 Configuration and Governance Layer

Responsibilities:

- model configuration
- prompt configuration
- data-source configuration
- run-mode configuration
- task configuration
- shared schema definitions
- logs and trace metadata

Notes:

- this is a cross-cutting support layer
- it should not contain direct business decision logic

## 6. Boundary Rules

1. The workflow layer must not access Neo4j directly.
2. The query layer must not perform heavy graph-building work during normal query execution.
3. The ingestion layer must not own idea-generation logic.
4. CLI and API are different entrances to the same application services, not two different systems.
5. Configuration support should stay separate from workflow logic.

## 7. Main System Flows

### 7.1 User Request Flow

1. request enters through the access layer
2. idea workflow layer interprets the input and creates a query request
3. knowledge query layer retrieves documents, entities, relations, and paths
4. workflow layer constructs generation context
5. workflow layer runs generation, evaluation, and revision loop
6. access layer returns unified final output

### 7.2 Knowledge Ingestion Flow

1. source data arrives from API or uploaded document
2. ingestion layer standardizes the document
3. structured objects are extracted
4. graph data is written into Neo4j
5. retrieval indexes are updated
6. knowledge query layer can query the new knowledge

## 8. Core System Objects

The initial design uses document-level knowledge organization. Chunk and evidence objects are intentionally out of scope for v1, but the architecture should leave room for future extension.

### 8.1 Document

Represents one source document.

Fields:

- `document_id`: internal unique identifier; recommended to be derived from normalized title, abstract, source type, and source reference
- `source_type`: source category such as API or PDF
- `source_ref`: original external identifier, path, or URL
- `title`: document title
- `abstract`: document abstract or normalized summary
- `raw_text`: optional full text for future use
- `authors`: optional author list
- `publisher` / `venue`: optional publication/source metadata
- `year`: optional but recommended time field
- `ingestion_status`: current ingestion state
- `extra_metadata`: source-specific extension fields

### 8.2 Entity

Represents a graph node.

Fields:

- `entity_id`: internal identifier; in v1 it may be derived from normalized entity name
- `name`: display name of the entity
- `entity_type`: high-level type such as `Technology`, `Problem`, `Scenario`, `Evaluation`, or `Claim`
- `aliases`: optional aliases or abbreviations
- `description`: optional short description
- `score`: optional score derived from retrieval/ranking formula
- `source_refs`: references to supporting source documents

### 8.3 Relation

Represents a graph edge between entities.

Fields:

- `source_entity_id`: source entity identifier
- `target_entity_id`: target entity identifier
- `relation_type`: edge type
- `attributes`: extension fields such as counts or trend-related values
- `score`: optional ranking or relation score

### 8.4 QueryRequest

Standard input object sent from the workflow layer to the query layer.

Fields:

- `query_text`: normalized query text
- `intent_type`: business intent category
- `keywords`: extracted keywords or anchor entities
- `constraints`: user preferences and hard constraints
- `query_config`: query configuration object, including policy and runtime switches

### 8.5 QueryResult

Standard result object returned by the query layer.

Fields:

- `documents`: matched document set
- `entities`: matched or expanded entities
- `relations`: key relations
- `paths`: structured paths used for reasoning and generation

### 8.6 IdeaDraft

Internal candidate idea object used inside the workflow.

Fields:

- `problem_statement`: what problem the idea addresses
- `method_proposal`: the proposed technical route
- `validation_plan`: how the idea should be validated
- `supporting_context`: supporting documents/entities/paths used during generation

### 8.7 EvaluationReport

Evaluation result for an idea draft.

Fields:

- `problem_scores`: scores for the problem-definition dimension
- `method_scores`: scores for the method dimension
- `experiment_scores`: scores for the validation/experiment dimension

### 8.8 FinalIdeaOutput

Unified external result object for API, CLI, and frontend use.

Fields:

- `idea_id`: result identifier
- `title`: final title
- `summary`: short overview
- `problem_statement`: final problem definition
- `method_proposal`: final technical route
- `validation_plan`: final validation path

## 9. Deferred Objects

The following objects are intentionally deferred and not part of the v1 core object set:

- `DocumentChunk`
- `Evidence`

They may be introduced later when the system moves from title/abstract-level processing to full-text chunk-based retrieval and fine-grained traceability.

## 10. Open Topics

The following parts are not finalized yet and should be discussed in subsequent iterations:

- src-level module decomposition
- concrete abstraction for model providers
- concrete abstraction for filtering/reranking strategy providers
- prompt configuration structure
- retrieval index composition beyond Neo4j
- whether and how to adopt LangGraph in the workflow layer

## 11. Core Process Design

This section defines the three main runtime flows at the architecture level. It focuses on logical stages and ownership boundaries rather than implementation details.

### 11.1 Multi-Source Ingestion and Graph-Building Flow

Goal:

- ingest heterogeneous source data
- normalize it into internal document objects
- extract graph-structured knowledge
- make the knowledge queryable through the graph layer

Flow:

1. `Source Intake`
   Receive input from supported sources.
   Current priority is paper-style sources. At the architecture level, the system should still reserve space for multiple source types such as API-based papers and uploaded PDFs.

2. `Document Standardization`
   Convert incoming source data into the internal `Document` object.
   This stage handles:
   - field alignment
   - missing-field fallback
   - source metadata retention
   - document identifier generation
   - ingestion status initialization

3. `Structured Extraction`
   Extract `Entity` and `Relation` objects from standardized documents.
   The extraction strategy may vary by source type:
   - for paper APIs, v1 mainly uses title and abstract
   - for PDFs or other richer sources, later extraction may use full text

   The key rule is:
   source-specific parsing and extraction are allowed, but all outputs must converge into the same internal graph schema.

4. `Graph Write`
   Write extracted nodes and edges into Neo4j.
   This includes:
   - node merge/upsert
   - relation merge/upsert
   - source association
   - graph-side scores or attributes when applicable

5. `Task Finalization`
   Update task status and record success or failure.
   The ingestion flow must support retry and idempotent re-execution.

Notes:

- v1 does not require chunk-based full-text processing
- v1 may begin with paper ingestion only
- later source types can use different preprocessing/extraction strategies as long as they converge to the same internal objects
- a separate retrieval-index refresh step is not required as a hard architectural stage if knowledge becomes queryable immediately after graph write; extra indexing can be introduced later if needed

### 11.2 Query and Graph Retrieval Flow

Goal:

- turn a user request into high-value graph context for idea generation

Flow:

1. `Input Understanding`
   Identify:
   - the main problem or theme
   - intent type
   - constraints and preferences
   - key technical terms

2. `Query Request Construction`
   Convert interpreted input into a `QueryRequest`.
   The request includes:
   - normalized query text
   - intent type
   - keywords
   - constraints
   - query configuration

3. `Initial Retrieval`
   The knowledge query layer performs initial retrieval from the local graph-backed knowledge base.
   It may include:
   - document retrieval
   - entity matching
   - relation recall
   - path candidate construction

4. `Filtering and Scoring`
   Results are filtered and ranked according to graph/query-layer logic.
   This stage is expected to happen primarily in the knowledge-graph side rather than the upper workflow side when possible, because graph-native scoring and filtering should be faster and more coherent with the graph schema.

   Typical signals include:
   - semantic relevance
   - graph connectivity
   - path completeness
   - entity/relation score
   - user constraints

   Architectural rule:
   this stage should be exposed behind a replaceable strategy interface so that future scoring formulas or filtering policies can be swapped without changing the workflow-layer contract.

5. `Optional Tool-Based Expansion`
   The system should always support local graph-query tools that can be called by the model or workflow when needed.
   This is different from optional external online tools.

   The architecture therefore distinguishes:
   - local graph query tools: core capability, always available
   - external online tools: optional capability, enabled or disabled by configuration

   This design supports restricted environments where only local knowledge-graph access is allowed.

6. `Query Result Assembly`
   Assemble the final `QueryResult` and return it to the idea workflow layer.

Notes:

- the query layer returns structured knowledge context, not raw database records
- local graph-tool access is part of the core design
- external online expansion is optional and configuration-controlled
- lower-level retrieval implementation details stay hidden behind the query-layer interface

### 11.3 Generation-Evaluation-Revision Loop

Goal:

- generate candidate ideas from graph context
- evaluate them along multiple dimensions
- iteratively revise them within configurable limits

Flow:

1. `Generation Context Build`
   The workflow layer organizes the `QueryResult` into model-ready generation context.
   This context typically includes:
   - selected documents
   - selected entities
   - selected relations
   - selected paths
   - user intent and constraints

2. `Idea Draft Generation`
   The generation model produces an `IdeaDraft`:
   - `problem_statement`
   - `method_proposal`
   - `validation_plan`

3. `Distributed Evaluation`
   Evaluation agents score and explain the draft along multiple dimensions.
   At the current design level, the three main dimensions are:
   - problem
   - method
   - experiment

   Each evaluation step produces both:
   - scores
   - reasons/diagnostics

   These are generated together in one evaluation pass per evaluator/agent.

4. `Evaluation Aggregation`
   The workflow aggregates the outputs from evaluation agents and passes the combined result back into the revision step.
   The aggregation result is not just a single score; it is structured feedback for controlled improvement.

5. `Revision`
   A model revises the idea based on aggregated evaluation feedback.
   Revision can target:
   - problem definition
   - method design
   - validation plan

6. `Loop Control`
   The iteration is controlled by configuration such as:
   - maximum iteration rounds
   - stopping threshold
   - whether to stop on plateau or insufficient improvement

7. `Final Output Materialization`
   When loop conditions are satisfied, the system outputs a `FinalIdeaOutput`.

Notes:

- evaluation is part of the core generation loop, not a display-only stage
- scoring and rationale are produced together by evaluation agents
- revision is driven by aggregated evaluation feedback
- generation models and evaluation models/agents should remain replaceable

## 12. Key Abstractions and Replaceable Components

This section identifies the system components that should be abstracted early so the architecture does not become tightly coupled to one implementation path.

### 12.1 Data Source Adapter

Purpose:

- connect to heterogeneous external sources
- read raw source data
- expose source-specific metadata

Boundary:

- responsible for fetching/reading source data
- not responsible for producing final graph objects

### 12.2 Document Standardizer

Purpose:

- convert raw source data into the internal `Document` object
- align fields and preserve extra metadata
- generate stable document identifiers

Boundary:

- responsible for normalization into internal document schema
- independent from both raw source adapters and graph extraction

### 12.3 Extraction Strategy

Purpose:

- extract `Entity` and `Relation` objects from standardized documents
- support different extraction strategies for different source types

Examples:

- paper metadata extraction from title and abstract
- later full-text extraction for PDF-like sources

Requirement:

- source-specific extraction is allowed
- outputs must converge into the same internal graph schema

### 12.4 Graph Repository

Purpose:

- encapsulate Neo4j read/write operations
- hide driver and Cypher details from upper layers

Boundary:

- low-level graph storage access
- not the main business-facing query abstraction

### 12.5 Knowledge Query Provider

Purpose:

- provide unified knowledge-query capability to the workflow layer
- organize candidate documents, entities, relations, and paths
- call filtering/reranking logic and return a `QueryResult`

Requirement:

- upper layers must depend on this abstraction instead of direct graph queries

### 12.6 Filtering and Reranking Strategy

Purpose:

- score, filter, and rerank query candidates
- support changes in formula, policy, or model-assisted ranking later

Requirement:

- must be implemented behind a strategy interface
- must be replaceable without changing the workflow-layer contract

Examples:

- formula-based ranking
- graph-score ranking
- model-assisted reranking
- hybrid strategy

### 12.7 Graph Tool Provider

Purpose:

- expose local graph-query tools to the model or workflow
- standardize tool input/output
- separate local graph tools from optional external online tools

Requirement:

- local graph tools are a core capability
- external online tools remain optional and configuration-controlled

### 12.8 Model Provider

Purpose:

- abstract model invocation
- route different workflow stages to different models if needed
- shield upper layers from vendor SDK details

Expected usages:

- extraction model
- generation model
- evaluation model
- revision model

### 12.9 Prompt Provider

Purpose:

- load prompt templates by workflow profile and node
- keep prompt assets separate from workflow logic
- support prompt configuration and later extension

### 12.10 Evaluation Coordinator

Purpose:

- coordinate multiple evaluation agents
- aggregate dimension-level scores and reasons
- provide structured evaluation output to the revision stage

### 12.11 Workflow Engine / Orchestrator

Purpose:

- orchestrate query, generation, evaluation, revision, and loop control
- keep workflow definition separate from execution framework choice

Requirement:

- architecture should not be tightly bound to a specific orchestration framework
- this leaves room for future adoption of frameworks such as LangGraph

### 12.12 Priority of Abstractions

Must abstract early:

- `Model Provider`
- `Prompt Provider`
- `Knowledge Query Provider`
- `Filtering and Reranking Strategy`
- `Extraction Strategy`

Recommended to abstract:

- `Data Source Adapter`
- `Document Standardizer`
- `Graph Tool Provider`
- `Evaluation Coordinator`

Can remain thin in v1:

- `Graph Repository`
- `Workflow Engine`

### 12.13 Replaceable vs Configurable

These two concepts must stay separate:

- replaceable component: the implementation class can change
- configurable component: the same implementation runs with different parameters or profiles

This distinction is especially important for models, prompts, reranking strategies, and workflow behavior.

## 13. Suggested `src/` Module Decomposition

v1 should be implemented as a single application repository with clear internal boundaries, rather than being split into separate deployable services too early.

### 13.1 Top-Level Modules

Suggested top-level modules under `src/`:

- `app`
- `workflows`
- `knowledge`
- `ingestion`
- `models`
- `prompts`
- `config`
- `schemas`
- `storage`
- `tasks`

Possible future interface modules:

- `interfaces/api`
- `interfaces/cli`

### 13.2 Responsibilities by Module

`app`

- application assembly layer
- compose configuration, providers, repositories, and workflows

`workflows`

- idea workflow orchestration
- input understanding, generation, evaluation, revision loop

`knowledge`

- knowledge-query layer implementation
- query providers
- filtering/reranking strategies
- graph tool provider

`ingestion`

- multi-source intake and graph-building flow
- source adapters
- standardizers
- extraction strategies

`models`

- model invocation abstraction
- provider and model routing logic

`prompts`

- prompt templates and loading logic
- organized by workflow profile and workflow node

`config`

- system configuration definitions and loading

`schemas`

- core object definitions such as `Document`, `Entity`, `Relation`, `QueryRequest`, `QueryResult`, `IdeaDraft`, `EvaluationReport`, and `FinalIdeaOutput`

`storage`

- storage access encapsulation
- Neo4j repository first
- optional future relational-storage support

`tasks`

- task triggering, status, retry, and batch-process coordination

### 13.3 Concrete Directory Sketch

```text
src/
  app/
    bootstrap.py
    container.py

  schemas/
    document.py
    entity.py
    relation.py
    query.py
    idea.py
    evaluation.py

  config/
    settings.py
    model_config.py
    prompt_config.py
    query_config.py

  models/
    base.py
    provider.py
    registry.py

  prompts/
    loader.py
    profiles/
      reverse_route.yaml
      forward_scenario.yaml
      combinational_innovation.yaml
    nodes/
      generate.yaml
      evaluate_problem.yaml
      evaluate_method.yaml
      evaluate_experiment.yaml
      revise.yaml

  storage/
    graph/
      neo4j_repository.py
      cypher_builder.py
    relational/
      README.md

  ingestion/
    adapters/
      base.py
      paper_api.py
      pdf.py
    standardizers/
      base.py
      document_standardizer.py
    extractors/
      base.py
      paper_metadata_extractor.py
      pdf_fulltext_extractor.py
    services/
      ingest_service.py

  knowledge/
    providers/
      base.py
      graph_query_provider.py
    ranking/
      base.py
      formula_strategy.py
    tools/
      graph_tool_provider.py
    services/
      query_service.py

  workflows/
    idea/
      orchestrator.py
      context_builder.py
      generator.py
      evaluator.py
      reviser.py

  tasks/
    ingest_tasks.py
    rebuild_tasks.py
```

### 13.4 Module Design Rules

1. `schemas` should stay independent.
2. `storage` and `knowledge` should remain separate.
3. `ingestion` and `workflows` should remain separate.
4. `prompts` and `models` should remain separate.
5. `app` should only handle wiring and assembly.

### 13.5 Suggested Minimum Build Order

1. `schemas`
2. `config`
3. `storage/graph`
4. `ingestion` with paper adapter and metadata extractor
5. `knowledge` with graph query provider and formula ranking strategy
6. `models` and `prompts`
7. `workflows/idea`
8. CLI or simple API interface
