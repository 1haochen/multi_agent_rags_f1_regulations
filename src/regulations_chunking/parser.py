from __future__ import annotations

"""
Helpers to read PaddleOCR-VL JSON into `Page` / `Block` objects.

This layer knows about the JSON structure (keys like `parsing_res_list`,
`block_label`, etc.) and intentionally hides that from higher-level
chunking code so that it can remain focused on articles / clauses.
"""

import json
from pathlib import Path
from typing import Iterable, List

from .model import Block, Page


def load_pages(json_path: Path) -> List[Page]:
    """
    Load a PaddleOCR-VL JSON file and convert it into `Page` objects.

    Assumes the JSON top level is a list where each element represents a
    page (as produced by `src.ocr.paddle_ocr.run_ocr_on_pdf`).
    """
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    pages: List[Page] = []
    for page_obj in raw:
        pages.append(_page_from_raw(page_obj))
    return pages


def _page_from_raw(obj: dict) -> Page:
    """
    Convert one page dict into a `Page` with `Block`s.
    """
    input_path = Path(obj.get("input_path", ""))
    page_index = int(obj.get("page_index", 0))
    page_count = int(obj.get("page_count", 1))
    width = int(obj.get("width", 0))
    height = int(obj.get("height", 0))

    blocks: List[Block] = []
    for b in obj.get("parsing_res_list", []) or []:
        label = str(b.get("block_label", "") or "")
        content = str(b.get("block_content", "") or "")
        bbox = tuple(b.get("block_bbox", [0, 0, 0, 0]))  # type: ignore[assignment]
        block_id = int(b.get("block_id", -1))

        # Some blocks (headers / images) do not participate in logical order.
        block_order = b.get("block_order")
        block_order_int = int(block_order) if block_order is not None else None

        group_id = b.get("group_id")
        group_id_int = int(group_id) if group_id is not None else None

        original_content = b.get("block_content_ocr_original")

        # Keep unknown keys in case they become useful later.
        known = {
            "block_label",
            "block_content",
            "block_bbox",
            "block_id",
            "block_order",
            "group_id",
            "block_content_ocr_original",
        }
        extra = {k: v for k, v in b.items() if k not in known}

        blocks.append(
            Block(
                page_index=page_index,
                block_id=block_id,
                label=label,
                content=content,
                bbox=bbox,  # type: ignore[arg-type]
                block_order=block_order_int,
                group_id=group_id_int,
                original_content=original_content,
                extra=extra,
            )
        )

    # Sort by block_order when present, falling back to block_id.
    blocks.sort(key=lambda bl: (bl.block_order is None, bl.block_order or bl.block_id))

    return Page(
        input_path=input_path,
        page_index=page_index,
        page_count=page_count,
        width=width,
        height=height,
        blocks=blocks,
    )


def iter_blocks(pages: Iterable[Page]):
    """
    Convenience generator for (page, block) pairs in document order.
    """
    for page in pages:
        for block in page.blocks:
            yield page, block

