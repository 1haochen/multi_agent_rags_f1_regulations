"""
Chunking utilities for FIA-style regulations JSON produced by PaddleOCR-VL.

The goal is to turn page-level OCR output (lists of layout blocks) into
RAG-ready chunks with rich metadata (article / clause ids, page indices,
tables, etc.), without depending on any specific vector DB.

See `regulations_chunking/pipeline.py` for the main entry points.
"""

