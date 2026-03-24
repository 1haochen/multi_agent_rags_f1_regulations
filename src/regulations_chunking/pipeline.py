from __future__ import annotations

"""
High-level chunking pipeline for FIA-style regulations JSON.

Typical usage from a notebook or script:

    from pathlib import Path
    from regulations_chunking.parser import load_pages
    from regulations_chunking.pipeline import ChunkingConfig, build_chunks

    pages = load_pages(Path("data/..._by_PaddleOCR-VL_no_strike.json"))
    cfg = ChunkingConfig()
    chunks = build_chunks(pages, cfg)

`chunks` is then ready to be fed into your embedding / vector DB layer.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .model import Block, Chunk, ChunkingConfig, Page


# Generic FIA section patterns: A1..., B2..., C3..., ... E9...
ARTICLE_RE = re.compile(r"^ARTICLE\s+(?P<id>[A-Z]\d+)\b", re.IGNORECASE)
APPENDIX_RE = re.compile(r"^APPENDIX\s+(?P<id>[A-Z]\d+)\b", re.IGNORECASE)
CLAUSE_RE = re.compile(r"\b[A-Z]\d+(?:\.\d+)*(?:\.[a-z])?\b", re.IGNORECASE)
APPENDIX_REF_RE = re.compile(r"\bAPPENDIX\s+([A-Z]\d+)\b", re.IGNORECASE)
# Top-level clause heading for grouping retrieval units, e.g. E1.2 / A1.2.
# IMPORTANT: negative lookahead `(?!\.)` prevents matching E1.2.1.
TOP_CLAUSE_HEADING_RE = re.compile(r"^(?P<id>[A-Z]\d+\.\d+)(?!\.)\b")
SUBCLAUSE_LINE_RE = re.compile(r"^(?P<id>[A-Z]\d+\.\d+\.\d+(?:\.\d+)*)\b", re.IGNORECASE)
LETTERED_ITEM_LINE_RE = re.compile(r"^(?P<id>[a-z])\.\s", re.IGNORECASE)


def build_chunks(pages: Iterable[Page], cfg: ChunkingConfig) -> List[Chunk]:
    """
    End-to-end chunk construction.

    Steps:
      1. Traverse all pages / blocks.
      2. Group blocks into articles and intra-article spans.
      3. Within each article, split into clause- or paragraph-level spans.
      4. Turn each span into one or more size-bounded `Chunk` objects.
      5. Handle tables either inline or as separate chunks with links.
    """
    spans = _group_blocks_by_sections(pages, cfg)
    chunks: List[Chunk] = []
    for span_idx, span in enumerate(spans):
        if span["scope"] == "article":
            span_chunks = _article_retrieval_chunks(
                article_id=span["section_id"],
                article_title=span["section_title"],
                span_index=span_idx,
                blocks=span["blocks"],
                cfg=cfg,
            )
        else:
            span_chunks = _appendix_chunks(
                appendix_id=span["section_id"],
                appendix_title=span["section_title"],
                span_index=span_idx,
                blocks=span["blocks"],
                cfg=cfg,
            )
        chunks.extend(span_chunks)
    return chunks


def _group_blocks_by_sections(pages: Iterable[Page], cfg: ChunkingConfig) -> List[Dict[str, Any]]:
    """
    Group blocks into ordered section spans.

    Sections can be either:
      - article: ARTICLE Ex...
      - appendix: APPENDIX Ex...
    The returned list preserves document order to avoid article reordering
    artifacts from dictionary insertion order.
    """
    spans: List[Dict[str, Any]] = []
    current_scope: Optional[str] = None
    current_section_id: Optional[str] = None
    current_section_title: Optional[str] = None
    current_span: List[Block] = []
    in_contents = False
    contents_start_page: Optional[int] = None

    def flush_span() -> None:
        nonlocal current_span
        if current_scope is None or current_section_id is None or not current_span:
            current_span = []
            return
        spans.append(
            {
                "scope": current_scope,
                "section_id": current_section_id,
                "section_title": current_section_title,
                "blocks": current_span,
            }
        )
        current_span = []

    for page in pages:
        for block in page.blocks:
            if block.label in cfg.ignore_labels:
                continue

            heading_text = " ".join((block.content or "").split())
            article_match = ARTICLE_RE.search(heading_text) if block.label in ("paragraph_title", "doc_title", "figure_title") else None
            appendix_match = APPENDIX_RE.search(heading_text) if block.label in ("paragraph_title", "doc_title", "figure_title") else None

            # Explicitly skip TOC/contents area.
            if _is_contents_heading(block):
                flush_span()
                in_contents = True
                contents_start_page = page.page_index
                continue

            if in_contents:
                # Exit contents mode once we move to a later page and hit an
                # actual section heading (article or appendix).
                if (article_match or appendix_match) and contents_start_page is not None and page.page_index > contents_start_page:
                    in_contents = False
                else:
                    continue

            # Appendix heading starts a new appendix section.
            if appendix_match:
                flush_span()
                current_scope = "appendix"
                current_section_id = appendix_match.group("id").upper()
                current_section_title = heading_text
                # Keep the heading with following appendix content.
                current_span = [block]
                continue

            # Article heading starts a new article section.
            if block.label in ("paragraph_title", "doc_title"):
                if article_match:
                    flush_span()
                    current_scope = "article"
                    current_section_id = article_match.group("id").upper()
                    current_section_title = heading_text
                    # Keep the heading with following article content.
                    current_span = [block]
                    continue

            # Treat tables as part of the nearest text span by default.
            if block.label == "table":
                # If a table appears before we enter any section, ignore it.
                if current_scope is None:
                    continue
                current_span.append(block)
                continue

            # For regular text-like labels, just keep extending the span.
            if block.label in ("text", "paragraph_title", "figure_title"):
                if current_scope is None:
                    continue
                current_span.append(block)
                continue

            # Unknown labels: conservatively start/end spans around them.
            if current_span:
                flush_span()

    # Final trailing span.
    flush_span()

    return spans


def _article_retrieval_chunks(
    *,
    article_id: str,
    article_title: Optional[str],
    span_index: int,
    blocks: List[Block],
    cfg: ChunkingConfig,
) -> List[Chunk]:
    """
    Build retrieval chunks for one article span.

    Strategy:
      1. Split article blocks into top-level clause units (E?.? headings).
      2. Keep each clause unit as the primary semantic boundary.
      3. If a clause unit is still large, split by lower-level structure
         (subclauses first, then lettered items).
      4. Emit tables as separate chunks linked to their parent clause.
    """
    clause_units = _build_clause_units(blocks, article_id)
    if not clause_units:
        return []

    out: List[Chunk] = []
    chunk_order = 0
    for unit in clause_units:
        text_blocks = [b for b in unit["blocks"] if b.label != "table"]
        table_blocks = [b for b in unit["blocks"] if b.label == "table"]
        clause_title = unit["clause_title"]
        parent_clause_id = unit["parent_clause_id"]

        text = _join_block_text(text_blocks)
        if text and not _is_front_matter_span(base_text=text, clause_ids=[parent_clause_id]):
            # Long clause units get finer-grained splits by subclauses / items.
            text_pieces = _split_clause_text(text, parent_clause_id)
            for piece_idx, piece in enumerate(text_pieces):
                piece_clause_ids = _extract_clause_ids(piece)
                piece_scope = "clause" if len(text_pieces) == 1 else "subclause"
                out.append(
                    Chunk(
                        text=piece,
                        metadata={
                            "scope": "article",
                            "chunk_scope": piece_scope,
                            "article_id": article_id,
                            "article_title": article_title,
                            "appendix_id": None,
                            "appendix_title": None,
                            "span_index": span_index,
                            "chunk_index_within_span": piece_idx,
                            "chunk_order_within_article": chunk_order,
                            "parent_clause_id": parent_clause_id,
                            "chunk_title": clause_title,
                            "page_indices": _sorted_unique([b.page_index for b in text_blocks]),
                            "block_ids": _sorted_unique([b.block_id for b in text_blocks]),
                            "clause_ids": piece_clause_ids,
                            "has_table": False,
                            "appendix_refs": _extract_appendix_refs(piece),
                        },
                    )
                )
                chunk_order += 1

        # Tables should travel with nearby context for better retrieval.
        # We append each table to the latest text chunk emitted for this clause.
        for _t_idx, table_block in enumerate(table_blocks):
            table_text = (table_block.original_content or table_block.content or "").strip()
            if not table_text:
                continue

            # Find the latest chunk for the same parent clause.
            attached = False
            for i in range(len(out) - 1, -1, -1):
                meta = out[i].metadata
                if (
                    meta.get("scope") == "article"
                    and meta.get("parent_clause_id") == parent_clause_id
                    and meta.get("chunk_scope") in {"clause", "subclause"}
                ):
                    out[i].text = f"{out[i].text}\n\nTABLE:\n{table_text}"
                    meta["has_table"] = True
                    meta["chunk_scope"] = "clause_with_table"
                    merged_block_ids = _sorted_unique(list(meta.get("block_ids", [])) + [table_block.block_id])
                    merged_pages = _sorted_unique(list(meta.get("page_indices", [])) + [table_block.page_index])
                    meta["block_ids"] = merged_block_ids
                    meta["page_indices"] = merged_pages
                    attached = True
                    break

            # Fallback: if no textual context chunk exists, keep a standalone
            # table chunk with clause title context.
            if not attached:
                out.append(
                    Chunk(
                        text=f"{clause_title}\n\nTABLE:\n{table_text}",
                        metadata={
                            "scope": "article",
                            "chunk_scope": "table",
                            "article_id": article_id,
                            "article_title": article_title,
                            "appendix_id": None,
                            "appendix_title": None,
                            "span_index": span_index,
                            "chunk_index_within_span": 0,
                            "chunk_order_within_article": chunk_order,
                            "parent_clause_id": parent_clause_id,
                            "chunk_title": f"Table for {parent_clause_id}",
                            "page_indices": [table_block.page_index],
                            "block_ids": [table_block.block_id],
                            "clause_ids": [parent_clause_id],
                            "has_table": True,
                            "appendix_refs": [],
                        },
                    )
                )
                chunk_order += 1

    return out


def _appendix_chunks(
    *,
    appendix_id: str,
    appendix_title: Optional[str],
    span_index: int,
    blocks: List[Block],
    cfg: ChunkingConfig,
) -> List[Chunk]:
    """
    Keep appendix as a separate scope. We still split by paragraph-aware
    packing to keep chunks manageable for retrieval.
    """
    paragraph_units: List[str] = []
    table_units: List[str] = []
    for b in blocks:
        if b.label == "table":
            table_text = (b.original_content or b.content or "").strip()
            if table_text:
                table_units.append(f"TABLE:\n{table_text}")
            continue
        cleaned = _clean_block_text(b.content)
        if cleaned:
            paragraph_units.append(cleaned)

    all_units = paragraph_units + table_units
    if not all_units:
        return []

    pieces = _pack_units_by_word_budget(all_units, min_words=250, max_words=700)
    out: List[Chunk] = []
    for idx, piece in enumerate(pieces):
        out.append(
            Chunk(
                text=piece,
                metadata={
                    "scope": "appendix",
                    "chunk_scope": "appendix",
                    "article_id": None,
                    "article_title": None,
                    "appendix_id": appendix_id,
                    "appendix_title": appendix_title,
                    "span_index": span_index,
                    "chunk_index_within_span": idx,
                    "chunk_order_within_article": idx,
                    "parent_clause_id": None,
                    "chunk_title": appendix_title,
                    "page_indices": _sorted_unique([b.page_index for b in blocks]),
                    "block_ids": _sorted_unique([b.block_id for b in blocks]),
                    "clause_ids": _extract_clause_ids(piece),
                    "has_table": "TABLE:" in piece,
                    "appendix_refs": _extract_appendix_refs(piece),
                },
            )
        )
    return out


def _snap_end_to_boundary(text: str, start: int, end_limit: int) -> int:
    """
    Move the chunk end left to a natural boundary if possible.

    Preference order:
      1) newline between start and end_limit
      2) whitespace between start and end_limit
      3) fallback to end_limit
    """
    if end_limit >= len(text):
        return len(text)

    nl = text.rfind("\n", start, end_limit)
    if nl > start:
        return nl

    ws = text.rfind(" ", start, end_limit)
    if ws > start:
        return ws

    return end_limit


def _snap_start_to_boundary(text: str, start: int) -> int:
    """
    Move the chunk start right to the next natural boundary, then skip
    boundary characters. This avoids leading partial words like "e ..." or
    "u ..." caused by overlap beginning in the middle of a token.
    """
    if start <= 0:
        return 0
    if start >= len(text):
        return len(text)

    i = start
    while i < len(text) and text[i] not in {" ", "\n", "\t"}:
        i += 1
    while i < len(text) and text[i] in {" ", "\n", "\t"}:
        i += 1
    return i


def _is_contents_heading(block: Block) -> bool:
    if block.label not in {"paragraph_title", "doc_title", "figure_title"}:
        return False
    text = " ".join((block.content or "").split()).upper()
    return text.startswith("CONTENTS") or text.startswith("TABLE OF CONTENTS")


def _is_front_matter_span(*, base_text: str, clause_ids: List[str]) -> bool:
    text = " ".join(base_text.split())
    upper = text.upper()
    if not text:
        return True
    # Standalone heading lines.
    if ARTICLE_RE.fullmatch(text) is not None:
        return True
    if text.upper().startswith("ARTICLE E") and len(text.split()) <= 8:
        return True
    # Cover-page convention/publishing metadata.
    if "CONVENTION:" in upper:
        return True
    if not clause_ids and ("WMSC APPROVAL DATE" in upper or "STATUS:" in upper):
        return True
    return False


def _split_oversized_unit(unit: str, max_chars: int) -> List[str]:
    """
    Fallback splitter for a single overlong paragraph.
    """
    pieces: List[str] = []
    start = 0
    while start < len(unit):
        end_limit = min(len(unit), start + max_chars)
        end = _snap_end_to_boundary(unit, start, end_limit)
        piece = unit[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(unit):
            break
        start = _snap_start_to_boundary(unit, end)
    return pieces


def _build_clause_units(blocks: List[Block], article_id: str) -> List[Dict[str, Any]]:
    """
    Build clause units keyed by top-level clause headings (E?.?).

    Each unit collects:
      - heading (e.g. E1.2 Objectives)
      - immediate child content until next top-level clause heading.
    """
    units: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    article_heading_seen = False

    for b in blocks:
        text = _clean_block_text(b.content)
        if not text and b.label != "table":
            continue

        # Skip standalone article heading blocks as retrieval units.
        if b.label in {"paragraph_title", "doc_title", "figure_title"}:
            flat = " ".join((b.content or "").split())
            if ARTICLE_RE.search(flat):
                article_heading_seen = True
                continue

        top_clause_id = _extract_top_clause_heading_id(text, article_id)
        if top_clause_id:
            if current is not None:
                units.append(current)
            current = {
                "parent_clause_id": top_clause_id,
                "clause_title": text.splitlines()[0].strip(),
                "blocks": [b],
            }
            continue

        if current is None:
            # Ignore any preamble before the first top-level clause.
            if article_heading_seen:
                continue
            continue

        current["blocks"].append(b)

    if current is not None:
        units.append(current)
    return units


def _split_clause_text(text: str, parent_clause_id: str) -> List[str]:
    """
    Split a clause text into retrieval chunks while preserving coherence.

    - Keep clause intact if short (<=700 words).
    - If long, split by subclause headings under the same parent clause.
    - If still long, split by lettered item lines.
    - Final fallback: paragraph budget split.
    """
    if _word_count(text) <= 700:
        return [text.strip()]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    subclause_segments = _split_by_subclause_headings(paragraphs, parent_clause_id)
    subclause_segments = _merge_heading_only_segments(subclause_segments, parent_clause_id)
    if len(subclause_segments) > 1:
        packed = _pack_units_by_word_budget(subclause_segments, min_words=250, max_words=700)
        if packed:
            return packed

    letter_segments = _split_by_lettered_items(paragraphs)
    letter_segments = _merge_heading_only_segments(letter_segments, parent_clause_id)
    if len(letter_segments) > 1:
        packed = _pack_units_by_word_budget(letter_segments, min_words=250, max_words=700)
        if packed:
            return packed

    return _pack_units_by_word_budget(paragraphs, min_words=250, max_words=700)


def _split_by_subclause_headings(paragraphs: List[str], parent_clause_id: str) -> List[str]:
    segments: List[str] = []
    current: List[str] = []
    wanted_prefix = f"{parent_clause_id}."
    for p in paragraphs:
        first_line = p.splitlines()[0].strip()
        m = SUBCLAUSE_LINE_RE.match(first_line)
        if m and m.group("id").upper().startswith(wanted_prefix.upper()):
            if current:
                segments.append("\n\n".join(current))
            current = [p]
        else:
            current.append(p)
    if current:
        segments.append("\n\n".join(current))
    return segments


def _split_by_lettered_items(paragraphs: List[str]) -> List[str]:
    """
    Split at lettered list items (a., b., c., ...) when present.
    """
    segments: List[str] = []
    current: List[str] = []
    for p in paragraphs:
        first_line = p.splitlines()[0].strip()
        if LETTERED_ITEM_LINE_RE.match(first_line):
            if current:
                segments.append("\n\n".join(current))
            current = [p]
        else:
            current.append(p)
    if current:
        segments.append("\n\n".join(current))
    return segments


def _merge_heading_only_segments(segments: List[str], parent_clause_id: str) -> List[str]:
    """
    Prevent tiny heading-only chunks like "E3.1 Exclusions" or "E4.1 Adjustments"
    from being emitted alone. If a segment is only the parent heading line (or
    very short heading text), merge it into the following segment.
    """
    if len(segments) <= 1:
        return segments

    merged: List[str] = []
    i = 0
    while i < len(segments):
        seg = segments[i].strip()
        if i < len(segments) - 1 and _is_heading_only_segment(seg, parent_clause_id):
            combined = f"{seg}\n\n{segments[i + 1].strip()}"
            merged.append(combined)
            i += 2
            continue
        merged.append(seg)
        i += 1
    return merged


def _is_heading_only_segment(segment: str, parent_clause_id: str) -> bool:
    lines = [ln.strip() for ln in segment.splitlines() if ln.strip()]
    if not lines:
        return False
    if len(lines) > 2:
        return False
    first = lines[0]
    if not first.upper().startswith(parent_clause_id.upper()):
        return False
    # If there is a second line, it should be very short heading-like text.
    if len(lines) == 2 and _word_count(lines[1]) > 6:
        return False
    return True


def _pack_units_by_word_budget(units: List[str], min_words: int, max_words: int) -> List[str]:
    """
    Pack sequential units into chunks that target 250-700 words.
    """
    out: List[str] = []
    pending: List[str] = []
    pending_words = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        w = _word_count(unit)
        if pending and pending_words + w > max_words:
            out.append("\n\n".join(pending).strip())
            pending = [unit]
            pending_words = w
        else:
            pending.append(unit)
            pending_words += w

    if pending:
        out.append("\n\n".join(pending).strip())
    return out


def _extract_top_clause_heading_id(text: str, article_id: str) -> Optional[str]:
    first = text.splitlines()[0].strip()
    m = TOP_CLAUSE_HEADING_RE.match(first)
    if not m:
        return None
    clause_id = m.group("id").upper()
    # Guardrail: keep only clauses belonging to current article family.
    if not clause_id.startswith(f"{article_id}."):
        return None
    return clause_id


def _clean_block_text(text: str) -> str:
    """
    Normalize OCR block text while keeping line boundaries.
    """
    if not text:
        return ""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _join_block_text(blocks: List[Block]) -> str:
    parts = []
    for b in blocks:
        cleaned = _clean_block_text(b.content)
        if cleaned:
            parts.append(cleaned)
    return "\n\n".join(parts).strip()


def _extract_clause_ids(text: str) -> List[str]:
    return sorted(set(m.upper() for m in CLAUSE_RE.findall(text or "")))


def _extract_appendix_refs(text: str) -> List[str]:
    return sorted(set(m.upper() for m in APPENDIX_REF_RE.findall(text or "")))


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _sorted_unique(items: List[int]) -> List[int]:
    return sorted(set(items))

