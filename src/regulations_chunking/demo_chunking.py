from __future__ import annotations

"""
Minimal example for running the regulations chunker on one JSON file.

Usage (from repo root, after creating the venv and installing deps):

    source .venv/bin/activate
    python -m src.regulations_chunking.demo_chunking \\
        --json-path "data/FIA 2026 F1 Regulations - Section E [Financial Regulations - Power Unit Manufacturers] - Iss 03 - 2025-12-10.pdf_by_PaddleOCR-VL_no_strike.json" \\
        --max-chunks 5

This will:
  - Load the PaddleOCR-VL JSON into `Page` / `Block` structures.
  - Build RAG-ready `Chunk` objects.
  - Print a small sample of chunks (text + key metadata) so you can inspect
    how articles, clauses, and tables were grouped.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

from .model import ChunkingConfig
from .parser import load_pages
from .pipeline import build_chunks


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regulations chunking on a single PaddleOCR-VL JSON file."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        required=True,
        help="Path to a *_by_PaddleOCR-VL*.json file.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=5,
        help="Number of chunks to print for inspection.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("example_chunked/chunks.jsonl"),
        help=(
            "Where to write chunk output as JSONL. "
            "Parent directory is created automatically."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    json_path: Path = args.json_path
    max_chunks: int = max(args.max_chunks, 1)
    out_path: Path = args.out_path

    if not json_path.exists():
        print(f"[demo_chunking] JSON file not found: {json_path}")
        return 1

    print(f"[demo_chunking] Loading pages from {json_path}")
    pages = load_pages(json_path)
    print(f"[demo_chunking] Loaded {len(pages)} page(s)")

    cfg = ChunkingConfig()
    chunks = build_chunks(pages, cfg)
    print(f"[demo_chunking] Built {len(chunks)} chunk(s)")

    # Persist chunk output so it can be inspected and reused by indexing code.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            row = {"text": ch.text, "metadata": ch.metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[demo_chunking] Saved JSONL chunks to {out_path}")

    print(f"\n[demo_chunking] Showing first {max_chunks} chunk(s):\n")
    for idx, ch in enumerate(chunks[:max_chunks]):
        meta = ch.metadata
        print("=" * 80)
        print(f"Chunk #{idx}")
        print(
            f"  article_id={meta.get('article_id')} "
            f"clause_ids={meta.get('clause_ids')} "
            f"page_indices={meta.get('page_indices')} "
            f"has_table={meta.get('has_table')}"
        )
        print("-" * 80)
        # Limit displayed text length so the sample fits comfortably in a
        # terminal; you still get the full text in `ch.text` for indexing.
        snippet = ch.text
        if len(snippet) > 1000:
            snippet = snippet[:1000] + "\n... [truncated for display] ..."
        print(snippet)
        print()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

