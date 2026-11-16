# Full-Text Extraction

IdeaGraph can optionally enrich each OpenAlex paper with PDF-derived Markdown
before entity and relation extraction. This is controlled by
`FULL_TEXT_EXTRACTION_ENABLED`; when disabled or when a paper PDF cannot be
resolved, extraction continues with title and abstract only.

## Runtime Flow

1. Check `data/{NEO4J_DATABASE}/{paper}/` for an existing non-empty Markdown
   artifact. A cache hit is treated as an already successful parse and skips PDF
   download plus conversion.
2. Build OpenAlex PDF candidates from external locations first:
   `best_oa_location.pdf_url`, `primary_location.pdf_url`,
   `locations[*].pdf_url`, and any `open_access.oa_url` or location
   `landing_page_url` that already looks like a direct `.pdf`.
3. Add OpenAlex `content_urls.pdf` as the final OpenAlex candidate. This URL
   uses the configured `OPENALEX_API_KEYS` pool because OpenAlex content
   downloads require an API key.
4. Try candidates in order. Each attempt downloads to
   `data/{NEO4J_DATABASE}/{paper}/source_N.pdf`, verifies the file starts with
   `%PDF`, and converts into `data/{NEO4J_DATABASE}/{paper}/attempt_N/`. A
   failed download, non-PDF response, or conversion failure is logged and the
   next candidate is tried.
5. If all OpenAlex candidates fail, query the arXiv API by paper title and run
   the same download/verify/convert loop once for the arXiv PDF.
6. Recover a missing abstract from the PDF-derived Markdown when needed.
7. Initialize entity/relation state from the abstract, strip the Markdown from
   common reference headings onward, and process tokenizer-aligned body chunks
   as incremental patch updates. The reference matcher handles Markdown
   headings, numbered headings such as `6. REFERENCES`, and Chinese headings
   such as `参考文献` or `引用文献`.
8. Store extracted entity and relation descriptions, plus token usage when the
   model provider returns it.

The generated PDF, Markdown, and JSON files are local artifacts and should not
be committed.

## Hybrid Backend

The application calls the `opendataloader-pdf` Python API directly. It does not
start the hybrid backend itself. Start the backend in another shell before
running graph construction:

```bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
CUDA_VISIBLE_DEVICES=1 uv run opendataloader-pdf-hybrid --host 127.0.0.1 --port 15002 --device cuda --log-level info
```

The default application settings expect:

```env
PDF_HYBRID_URL=http://127.0.0.1:15002
PDF_HYBRID_BACKEND=docling-fast
PDF_HYBRID_MODE=auto
```

Hybrid extraction is attempted first. If it fails, or if page coverage looks
incomplete, the extractor logs a warning and retries with fast mode. The helper
script `scripts/check_reference_headings.py` can scan generated Markdown files
to check which files match the current reference-heading rule.

## OpenAlex API Keys

Set `OPENALEX_API_KEYS` to a comma-separated list when API keys are available.
The client appends the current key as an `api_key` query parameter. If a key hits
an authentication or rate-limit error, it is marked unavailable for
`OPENALEX_API_KEY_COOLDOWN_SECONDS` and the next key is tried. The same key pool
is used for OpenAlex content PDF downloads under `https://content.openalex.org/`.

Normal environment proxy variables are left to `httpx` defaults. There is no
project-specific OpenAlex proxy setting.
