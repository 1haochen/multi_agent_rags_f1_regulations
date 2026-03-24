from __future__ import annotations

"""
Typed representations for the OCR JSON blocks and the derived RAG chunks.

These dataclasses are intentionally decoupled from any specific embedding
or vector database library. Downstream code can turn `Chunk` objects into
whatever index format it prefers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Block:
    """
    Lightweight wrapper around a single entry in `parsing_res_list`.

    The PaddleOCR-VL JSON structure is not strictly documented as a stable
    schema, so this class only models the fields we actually use for
    chunking. Any additional keys remain available via `extra`.
    """

    page_index: int
    block_id: int
    label: str
    content: str
    bbox: Tuple[int, int, int, int]
    block_order: Optional[int]
    group_id: Optional[int]
    original_content: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    """
    Normalised representation of a single page in the OCR JSON.

    `blocks` is already filtered to the labels we care about; callers can
    keep the raw JSON around if they need full fidelity for debugging.
    """

    input_path: Path
    page_index: int
    page_count: int
    width: int
    height: int
    blocks: List[Block]


@dataclass
class Chunk:
    """
    A RAG-ready text chunk with metadata.

    - `text` is what you embed and feed to the LLM.
    - `metadata` carries article / clause ids, page indices, etc.
    """

    text: str
    metadata: Dict[str, Any]


@dataclass
class ChunkingConfig:
    """
    Tunable parameters for the regulations chunking pipeline.
    """

    # Max token-ish length per chunk. We approximate tokens using a simple
    # character-count heuristic to avoid binding to a specific tokenizer.
    max_chars: int = 4000
    overlap_ratio: float = 0.1  # for long spans that need overlapping splits

    # Which block labels should be ignored entirely when building text.
    ignore_labels: Sequence[str] = (
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
        "number",
        "footnote",
    )

    # Clause id pattern, e.g. "E1.1", "E2.3.4". This is applied on text
    # lines when trying to detect clause boundaries inside article text.
    clause_prefix: str = "E"  # specific to FIA Section E; can be generalised

