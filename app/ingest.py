"""PDF → clean, page-aware chunks (with a JSON cache).

Problems found in these specific government PDFs and how they are handled:

1. InDesign printer slugs such as
   "厚厚労労22002233__概概要要版版..iinndddd 11 22002233//0077//1133" appear on 58 pages.
   → `page.dedupe_chars()` removes the doubled glyphs, then the slug line is dropped.
2. Bold headings rendered twice ("第第１１部部") → also fixed by `dedupe_chars()`.
3. Two-column layout (文部科学白書) made pdfplumber interleave both columns line by line
   ("…果た 計画について…").  → text is extracted in content-stream order.
4. Unmapped glyphs appear as "(cid:21)" → removed (numbers in those charts are lost;
   documented as a known limitation instead of being silently "fixed").
"""

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pdfplumber

from app.catalog import get_meta
from app.config import Settings

logger = logging.getLogger(__name__)

INGEST_VERSION = "3"  # bump when cleaning/chunking logic changes → cache is rebuilt

_SLUG_LINE = re.compile(r"^.*\.indd\s+\d+.*$", re.MULTILINE)
_CID = re.compile(r"\(cid:\d+\)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_SENTENCE_END = "。！？!?\n"


@dataclass
class Chunk:
    chunk_id: str  # e.g. "2023_gaiyo:p17:c0"
    doc_id: str
    title: str
    ministry: str
    page: int  # 1-based page index inside the PDF (used for "#page=N" links)
    text: str


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def extract_page_text(page) -> str:
    """`dedupe_chars()` removes doubled glyphs; `use_text_flow=True` follows the PDF
    content-stream order, so two-column pages are read column by column instead of
    line-by-line across both columns."""
    return page.dedupe_chars().extract_text(use_text_flow=True) or ""


def clean_text(text: str) -> str:
    text = _SLUG_LINE.sub("", text)
    text = _CID.sub("", text)
    return _MULTI_BLANK.sub("\n\n", text).strip()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window that prefers to cut at a sentence end (。) or line break."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start + size // 2 : end]
            cut = max(window.rfind(ch) for ch in _SENTENCE_END)
            if cut != -1:
                end = start + size // 2 + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always make progress
    return chunks


# --------------------------------------------------------------------------- #
# Corpus build + cache
# --------------------------------------------------------------------------- #
def _fingerprint(pdfs: list[Path], settings: Settings) -> str:
    h = hashlib.sha256(INGEST_VERSION.encode())
    h.update(f"{settings.chunk_size}:{settings.chunk_overlap}:{settings.min_chunk_chars}".encode())
    for p in pdfs:  # content hash, not mtime: a fresh git clone or a copy keeps the cache valid
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def load_corpus(settings: Settings) -> tuple[list[Chunk], str]:
    """Return (chunks, fingerprint). Uses a JSON cache (never pickle: safe to load)."""
    pdfs = sorted(settings.documents_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in '{settings.documents_dir}'.")

    fingerprint = _fingerprint(pdfs, settings)
    cache_file = settings.cache_dir / f"corpus-{fingerprint}.json"
    if cache_file.exists():
        chunks = [Chunk(**c) for c in json.loads(cache_file.read_text(encoding="utf-8"))]
        logger.info("Loaded %d chunks from cache (%s)", len(chunks), cache_file.name)
        return chunks, fingerprint

    chunks: list[Chunk] = []
    for pdf_path in pdfs:
        meta = get_meta(pdf_path.stem)
        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                text = clean_text(extract_page_text(page))
                pieces = split_text(text, settings.chunk_size, settings.chunk_overlap)
                pieces = [p for p in pieces if len(p) >= settings.min_chunk_chars]
                for i, piece in enumerate(pieces):
                    chunks.append(
                        Chunk(f"{meta.doc_id}:p{page_no}:c{i}", meta.doc_id, meta.title, meta.ministry, page_no, piece)
                    )
            logger.info("Ingested %-45s %3d pages", pdf_path.name, len(pdf.pages))

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8")
    logger.info("Built %d chunks from %d PDFs → %s", len(chunks), len(pdfs), cache_file.name)
    return chunks, fingerprint
