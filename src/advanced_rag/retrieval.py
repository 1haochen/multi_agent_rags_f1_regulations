from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        model_name: str = (os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-base-en"),
    ) -> None:
        self.semantic = RagRetriever(
            model_name=model_name,
            index_name=index_name,
            host=host,
            namespace=namespace,
            chunks_root=chunks_root,
        )

    def retrieve_references_from_pinecone(self, refs: List[str], *, limit: int = 6) -> List[ChunkDocObj]:
        """
        Expand context by retrieving referenced clause/article/appendix IDs from Pinecone metadata.

        This avoids using local `chunks/` files during reference expansion.
        """
        wanted = [x.strip() for x in (refs or []) if x and x.strip()]
        if not wanted:
            return []

        docs: List[ChunkDocObj] = []
        seen: set[str] = set()
        for rid in wanted:
            hits = self.semantic.retrieve_by_reference_id(rid, top_k=max(2, limit // 2))
            for h in hits:
                if h.id in seen:
                    continue
                if not (h.text or "").strip():
                    continue
                seen.add(h.id)
                docs.append(
                    ChunkDocObj(
                        id=h.id,
                        score=float(h.score),
                        text=h.text or "",
                        metadata=h.metadata,
                        source="reference",
                    )
                )
                if len(docs) >= limit:
                    return docs
        return docs

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
        # Expand any explicit IDs (clause/article/appendix) via Pinecone metadata filters only.
        exact_docs = self.retrieve_references_from_pinecone(explicit_ids, limit=max(2, top_k // 2))
        merged: Dict[str, ChunkDocObj] = {}
        for d in sem_docs + exact_docs:
            if d.id not in merged or d.score > merged[d.id].score:
                merged[d.id] = d
        return sorted(merged.values(), key=lambda x: x.score, reverse=True)[: top_k + 4]
