# Preview Incremental Entity Extraction Design

## Goal

Make `preview-entity-extraction` compare two extraction strategies in one run:
the existing whole-Markdown extraction and a new abstract-first incremental
extraction over pre-reference body chunks.

## Scope

- Only the preview workflow changes behavior.
- No CLI flag is added; JSON and table output both include two named results:
  `full_text` and `incremental`.
- The main graph-build, local-paper graph build, and `chat-idea` workflows keep
  their existing extraction behavior.
- The preview workflow must not connect to or write Neo4j.

## Design

The preview workflow reads the Markdown once and resolves the paper title with
the existing title-resolution behavior. It first runs the old full-text
extraction over the complete Markdown to preserve a direct baseline. It then
runs an incremental strategy:
extract the Abstract section if present, strip references from the Markdown,
remove the Abstract section from the remaining body, split the body with the
same paragraph/token helper and tokenizer configuration used by chunk-vector
retrieval, and process each chunk sequentially.

Each incremental chunk call receives the current normalized entities and
relations in a compact line format to reduce prompt tokens:

```text
[entity_id]----[entity_type]----[description]
[source_entity_id]----[target_entity_id]----[relation_type]----[description]
```

The model is instructed to return only changed entities and relations grounded
in the current chunk. The output shape remains the existing structured
`entities` / `relations` schema; the preview workflow interprets those entries
as a patch.

The preview workflow applies each patch locally. A new entity or relation is
added. If an existing entity or relation appears again, its `description` is
overwritten by the current chunk's description instead of being merged or
appended. Existing entity ids and types are preserved locally. Relations whose
endpoints do not exist in the current entity set or the current patch are
ignored.

## Output

JSON output shape:

```json
{
  "full_text": {"entities": [], "relations": []},
  "incremental": {"entities": [], "relations": []}
}
```

Table output renders two sections with the same entity and relation columns
used by the existing preview command.

## Validation

Tests cover the dual output shape, abstract-first call ordering, references
truncation for incremental chunks, current-state passing into the extractor,
local patch application, description overwrite semantics, and prompt wording
for compact incremental patch extraction.
