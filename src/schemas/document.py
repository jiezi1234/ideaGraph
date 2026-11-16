from pydantic import BaseModel, Field


class PaperDocument(BaseModel):
    document_id: str
    source_type: str
    source_ref: str
    abstract: str
    year: int | None = None
    cited_by_count: int | None = None
    pdf_url: str | None = None
    pdf_urls: list[str] = Field(default_factory=list)
    full_text_markdown_path: str | None = None
    full_text_markdown: str | None = None
    text_source: str = "title_abstract"
    citation_openalex_ids: list[str] = Field(default_factory=list)
    ingestion_status: str


class RetrievedPaperDocument(BaseModel):
    document_id: str
    title: str
    abstract: str
    year: int | None = None
    cited_by_count: int | None = None
    full_text_markdown_path: str | None = None
    text_source: str = "title_abstract"


class RetrievedContextChunk(BaseModel):
    paper_id: str
    title: str
    chunk_id: str
    chunk_index: int
    paragraph_start: int
    paragraph_end: int
    score: float
    text_source: str
    text: str
