from __future__ import annotations

"""
Batch runner for regulation chunking.

This script scans an input directory for PaddleOCR-VL JSON files, runs the
chunking pipeline on each file, and writes one JSONL output per input under an
output directory (default: `chunks/`).
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from .model import ChunkingConfig
from .parser import load_pages
from .pipeline import build_chunks


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clause-aware chunking on all OCR JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing *_by_PaddleOCR-VL*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chunks"),
        help="Directory where chunk JSONL files are written.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*_by_PaddleOCR-VL*.json",
        help="Glob pattern used to discover input JSON files.",
    )
    parser.add_argument(
        "--prefer-no-strike",
        action="store_true",
        help=(
            "When both base and _no_strike JSON exist for the same PDF, "
            "only process the _no_strike variant."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def discover_inputs(input_dir: Path, pattern: str, prefer_no_strike: bool) -> List[Path]:
    all_files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not prefer_no_strike:
        return all_files

    chosen: dict[str, Path] = {}
    for path in all_files:
        key = path.name.replace("_no_strike", "")
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = path
            continue
        # Prefer _no_strike when both exist.
        if "_no_strike" in path.name:
            chosen[key] = path
    return sorted(chosen.values())


def output_path_for(input_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_path.relative_to(input_root)
    out_name = rel.name.replace(".json", ".chunks.jsonl")
    return output_root / rel.parent / out_name


def process_file(input_path: Path, output_path: Path) -> int:
    pages = load_pages(input_path)
    cfg = ChunkingConfig()
    chunks = build_chunks(pages, cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            row = {"text": ch.text, "metadata": ch.metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(chunks)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        print(f"[run_chunking] Input directory not found: {input_dir}")
        return 1

    inputs = discover_inputs(input_dir, args.pattern, args.prefer_no_strike)
    if not inputs:
        print(f"[run_chunking] No files matched pattern: {args.pattern} under {input_dir}")
        return 0

    print(f"[run_chunking] Processing {len(inputs)} file(s)")
    total_chunks = 0
    success = 0
    failed = 0

    for src in inputs:
        dst = output_path_for(src, input_dir, output_dir)
        try:
            n = process_file(src, dst)
            total_chunks += n
            success += 1
            print(f"[ok] {src} -> {dst} ({n} chunks)")
        except Exception as exc:
            failed += 1
            print(f"[fail] {src}: {exc}")

    print(
        f"[run_chunking] done | success={success} failed={failed} total_chunks={total_chunks} "
        f"output_dir={output_dir}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

