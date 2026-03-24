from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.rag.retriever import RagRetriever


ID_PATTERN = re.compile(r"\b(?:E\d+(?:\.\d+)*(?:\.[a-z])?|Article\s+[A-Z]?\d+(?:\.\d+)*)\b", re.I)
APPENDIX_PATTERN = re.compile(r"\bAppendix\s+\d+\b", re.I)


@dataclass
class ChunkDocObj:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    source: str


def _metadata_matches_ids(meta: Dict[str, Any], wanted: set[str]) -> bool:
    searchable = {
        str(meta.get("parent_clause_id", "") or ""),
        str(meta.get("article_id", "") or ""),
        str(meta.get("appendix_id", "") or ""),
    }
    searchable.update(str(x) for x in (meta.get("clause_ids") or []) if x is not None)
    searchable.discard("")
    return bool(wanted.intersection(searchable))


class LocalClauseIndex:
    """
    On-disk exact lookup by clause / article / appendix id.

    **Important:** We stream JSONL files and stop after ``limit`` hits. We do **not**
    load the entire ``chunks/`` tree into RAM (that was causing multi-minute stalls on
    large corpora — and ``exact_lookup`` used to call full load even when ``ids`` was empty).
    """

    def __init__(self, chunks_root: str | Path = "chunks") -> None:
        self.chunks_root = Path(chunks_root)

    def exact_lookup(self, ids: Iterable[str], limit: int = 6) -> List[ChunkDocObj]:
        wanted = {x.strip() for x in ids if x.strip()}
        if not wanted:
            return []

        hits: List[ChunkDocObj] = []
        if not self.chunks_root.is_dir():
            return hits

        for path in sorted(self.chunks_root.glob("**/*.chunks.jsonl")):
            rel = str(path.relative_to(self.chunks_root))
            try:
                with path.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(row, dict):
                            continue
                        text = (row.get("text") or "").strip()
                        meta_raw = row.get("metadata")
                        if not text or not isinstance(meta_raw, dict):
                            continue
                        if not _metadata_matches_ids(meta_raw, wanted):
                            continue
                        metadata = {
                            "source_file": rel,
                            "source_row": idx,
                            **meta_raw,
                        }
                        hits.append(
                            ChunkDocObj(
                                id=f"local::{rel}::{idx}",
                                score=1.2,
                                text=text,
                                metadata=metadata,
                                source="exact",
                            )
                        )
                        if len(hits) >= limit:
                            return hits
            except OSError:
                continue
        return hits


def extract_ids(text: str) -> List[str]:
    ids = [m.group(0).strip() for m in ID_PATTERN.finditer(text or "")]
    ids.extend(m.group(0).strip() for m in APPENDIX_PATTERN.finditer(text or ""))
    out: List[str] = []
    seen = set()
    for x in ids:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


class HybridRetriever:
    def __init__(
        self,
        *,
        index_name: Optional[str] = None,
        host: Optional[str] = None,
        namespace: Optional[str] = None,
        chunks_root: str | Path = "chunks",
        model_name: str = "BAAI/bge-large-en-v1.5",
    ) -> None:
        self.semantic = RagRetriever(
            model_name=model_name,
            index_name=index_name,
            host=host,
            namespace=namespace,
            chunks_root=chunks_root,
        )
        self.local_index = LocalClauseIndex(chunks_root=chunks_root)

    def retrieve(self, query: str, explicit_ids: List[str], top_k: int = 8) -> List[ChunkDocObj]:
        sem = self.semantic.retrieve(query, top_k=top_k)
        sem_docs = [
            ChunkDocObj(
                id=r.id,
                score=r.score,
                text=r.text or "",
                metadata=r.metadata,
                source="semantic",
            )
            for r in sem
            if (r.text or "").strip()
        ]
        exact_docs = self.local_index.exact_lookup(explicit_ids, limit=max(2, top_k // 2))
        merged: Dict[str, ChunkDocObj] = {}
        for d in sem_docs + exact_docs:
            if d.id not in merged or d.score > merged[d.id].score:
                merged[d.id] = d
        return sorted(merged.values(), key=lambda x: x.score, reverse=True)[: top_k + 4]
