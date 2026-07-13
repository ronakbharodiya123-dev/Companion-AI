"""
Document processing service for extracting and indexing manual content.
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from app.models.database import ManualDocument, DocumentStatus
from app.services.rag_service import rag_service
from app.core.config import settings

logger = logging.getLogger(__name__)

# Pattern to detect page-number-only chunks (e.g. "English - 14", "- 14 -", or "14")
_PAGE_NUM_RE = re.compile(
    r"^(?:[A-Za-z]+ - \d+|- \d+ -|\d+)$"
)


class DocumentProcessor:
    """Process uploaded documents and index them in the vector store."""

    def __init__(self):
        """Initialize document processor."""
        self.text_splitter = rag_service.text_splitter

    # ------------------------------------------------------------------
    # Text extraction (fitz / PyMuPDF)
    # ------------------------------------------------------------------

    def _extract_text_chunks(self, file_path: str) -> List[str]:
        """
        Extract text from a digital PDF page by page using PyMuPDF (fitz).
        Splits each page's text into overlapping chunks of 1000 chars / 200 overlap,
        then filters out noise chunks.

        Returns:
            List of clean text chunk strings.
        """
        chunk_size = 1000
        overlap = 200
        min_len = 30

        chunks: List[str] = []

        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                if not page_text or not page_text.strip():
                    continue

                # Slide a window over the page text
                start = 0
                while start < len(page_text):
                    end = start + chunk_size
                    chunk = page_text[start:end].strip()

                    if len(chunk) >= min_len and not _PAGE_NUM_RE.match(chunk):
                        chunks.append(chunk)

                    start += chunk_size - overlap  # advance with overlap

        logger.info(f"Extracted {len(chunks)} text chunks via PyMuPDF (fitz)")
        return chunks

    # ------------------------------------------------------------------
    # Table extraction (pdfplumber)
    # ------------------------------------------------------------------

    def _extract_table_chunks(
        self, file_path: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract tables from each page using pdfplumber.

        - Clean / simple tables  → converted to GitHub-flavoured markdown.
        - Lossy / complex tables → raw page text extracted via fitz is used
          instead, so that multi-column spec tables (e.g. Samsung QLED with
          3 model columns) are stored without data loss.

        A table is considered lossy when >40 % of its cells are empty after
        pdfplumber parsing, which indicates a complex layout it cannot handle.

        Returns:
            (table_texts, table_metadatas) — parallel lists ready for Qdrant.
        """
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        # Pages already emitted as fitz fallback — avoid duplicates
        fallback_pages_added: set = set()

        with pdfplumber.open(file_path) as pdf, fitz.open(file_path) as fitz_doc:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if not tables:
                    continue

                # Full page text for coverage comparison
                page_full_text = page.extract_text() or ""

                for table in tables:
                    # Skip rows that are entirely empty / None
                    non_empty_rows = [
                        row for row in table if any(cell for cell in row if cell)
                    ]
                    if not non_empty_rows:
                        continue

                    if self._is_table_lossy(non_empty_rows, page_full_text):
                        # Complex multi-column layout — fall back to fitz raw text
                        if page_num not in fallback_pages_added:
                            fitz_page = fitz_doc[page_num - 1]
                            raw_text = fitz_page.get_text("text").strip()
                            if raw_text:
                                texts.append(
                                    f"Specifications on page {page_num}:\n{raw_text}"
                                )
                                metadatas.append(
                                    {"source": "table_text", "page": page_num}
                                )
                                fallback_pages_added.add(page_num)
                                logger.debug(
                                    f"Page {page_num}: lossy table — stored as raw text fallback"
                                )
                    else:
                        # Clean table — store as markdown
                        markdown = self._table_to_markdown(non_empty_rows)
                        texts.append(f"Table on page {page_num}:\n{markdown}")
                        metadatas.append({"source": "table", "page": page_num})

        clean = sum(1 for m in metadatas if m["source"] == "table")
        fallback = sum(1 for m in metadatas if m["source"] == "table_text")
        logger.info(
            f"Table extraction: {clean} markdown tables, "
            f"{fallback} complex-table pages stored as raw text"
        )
        return texts, metadatas

    @staticmethod
    def _is_table_lossy(
        rows: List[List],
        page_full_text: str = "",
        empty_threshold: float = 0.4,
        coverage_threshold: float = 0.40,
    ) -> bool:
        """
        Return True if the table appears to have lost data during parsing.

        Two complementary checks:

        1. **Empty-cell ratio** — if >*empty_threshold* of cells are blank,
           pdfplumber likely failed on a merged-cell layout.

        2. **Text-coverage ratio** — only applied to real multi-column data
           tables (cols > 1 and rows > 3).  Compares the total characters
           captured inside the table against the full page text. If the table
           covers <*coverage_threshold* of the page text, columns were silently
           dropped (e.g. a 3-column spec table parsed as 2 columns).
        """
        total_cells = sum(len(row) for row in rows)
        if total_cells == 0:
            return True

        # Check 1: too many empty cells
        empty = sum(
            1 for row in rows for cell in row
            if cell is None or not str(cell).strip()
        )
        if (empty / total_cells) > empty_threshold:
            return True

        # Check 2: coverage, but only for real multi-column data tables
        # (skip small UI / navigation tables with 1 col or <= 3 rows)
        num_cols = len(rows[0]) if rows else 0
        num_rows = len(rows)
        if page_full_text.strip() and num_cols > 1 and num_rows > 3:
            table_text = " ".join(
                str(cell) for row in rows for cell in row
                if cell is not None and str(cell).strip()
            )
            coverage = len(table_text) / max(len(page_full_text), 1)
            if coverage < coverage_threshold:
                return True

        return False

    @staticmethod
    def _table_to_markdown(rows: List[List]) -> str:
        """Convert a list of rows into a GitHub-flavoured markdown table."""
        if not rows:
            return ""

        def cell(val) -> str:
            return str(val).strip() if val is not None else ""

        header = "| " + " | ".join(cell(c) for c in rows[0]) + " |"
        divider = "| " + " | ".join("---" for _ in rows[0]) + " |"
        body_rows = [
            "| " + " | ".join(cell(c) for c in row) + " |"
            for row in rows[1:]
        ]
        return "\n".join([header, divider] + body_rows)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract lightweight metadata heuristics from document text.

        Args:
            text: Document text (first portion used for heuristics)

        Returns:
            Dictionary of extracted metadata.
        """
        metadata: Dict[str, Any] = {}
        lines = text.split("\n")[:50]  # inspect first 50 lines only

        for line in lines:
            line_lower = line.lower()

            if "model" in line_lower and len(line) < 100:
                metadata["detected_model"] = line.strip()

            if "troubleshooting" in line_lower:
                metadata["section_type"] = "troubleshooting"
            elif "installation" in line_lower:
                metadata["section_type"] = "installation"
            elif "user guide" in line_lower or "user manual" in line_lower:
                metadata["section_type"] = "user_guide"

        return metadata

    # ------------------------------------------------------------------
    # Main processing entry-point
    # ------------------------------------------------------------------

    async def process_document(self, document_id: str) -> bool:
        """
        Process a document: extract text & table chunks, then index in Qdrant.

        Args:
            document_id: MongoDB document ID

        Returns:
            True if successful, False otherwise.
        """
        try:
            document = await ManualDocument.get(document_id)
            if not document:
                logger.error(f"Document {document_id} not found")
                return False

            logger.info(f"Processing document: {document.filename}")

            document.status = DocumentStatus.PROCESSING
            await document.save()

            file_ext = Path(document.file_path).suffix.lower()

            if file_ext == ".pdf":
                import asyncio
                from functools import partial

                loop = asyncio.get_running_loop()

                # Run both extractions in executor (blocking I/O)
                text_chunks = await loop.run_in_executor(
                    None, partial(self._extract_text_chunks, document.file_path)
                )
                table_texts, table_chunk_metadatas = await loop.run_in_executor(
                    None, partial(self._extract_table_chunks, document.file_path)
                )

            elif file_ext in (".txt", ".text"):
                with open(document.file_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                text_chunks = self.text_splitter.split_text(raw)
                table_texts, table_chunk_metadatas = [], []
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Combine all chunks
            all_chunks = text_chunks + table_texts

            if not all_chunks:
                raise ValueError("No content could be extracted from the document")

            # Build per-chunk metadata; text chunks get doc-level metadata,
            # table chunks get doc-level + table-specific metadata merged in.
            auto_metadata = self.extract_metadata_from_text(
                " ".join(text_chunks[:5])  # quick heuristic over first few chunks
            )

            metadatas: List[Dict[str, Any]] = []

            for i, chunk in enumerate(text_chunks):
                metadatas.append(
                    {
                        "source_file": document.filename,
                        "device_type": document.device_type,
                        "brand": document.brand,
                        "model": document.model or "Unknown",
                        "document_id": document.document_id,
                        "chunk_index": i,
                        "total_chunks": len(all_chunks),
                        **auto_metadata,
                    }
                )

            for i, (table_text, tbl_meta) in enumerate(
                zip(table_texts, table_chunk_metadatas)
            ):
                metadatas.append(
                    {
                        "source_file": document.filename,
                        "device_type": document.device_type,
                        "brand": document.brand,
                        "model": document.model or "Unknown",
                        "document_id": document.document_id,
                        "chunk_index": len(text_chunks) + i,
                        "total_chunks": len(all_chunks),
                        **auto_metadata,
                        **tbl_meta,  # source="table", page=X override
                    }
                )

            logger.info(
                f"Indexing {len(all_chunks)} chunks "
                f"({len(text_chunks)} text, {len(table_texts)} tables)"
            )

            chunks_added = await rag_service.add_documents(
                texts=all_chunks,
                metadatas=metadatas,
            )

            document.status = DocumentStatus.INDEXED
            document.chunks_count = chunks_added
            document.processed_at = datetime.now(timezone.utc)
            await document.save()

            logger.info(f"Successfully processed document: {document.filename}")
            return True

        except Exception as e:
            logger.error(f"Error processing document: {e}")

            try:
                document = await ManualDocument.get(document_id)
                if document:
                    document.status = DocumentStatus.FAILED
                    document.error_message = str(e)
                    await document.save()
            except Exception as save_error:
                logger.error(f"Error updating document status: {save_error}")

            return False


# ---------------------------------------------------------------------------
# Background task helper
# ---------------------------------------------------------------------------

async def process_document_task(document_id: str):
    """
    Background task to process a document.

    Args:
        document_id: Document ID to process
    """
    processor = DocumentProcessor()
    await processor.process_document(document_id)
