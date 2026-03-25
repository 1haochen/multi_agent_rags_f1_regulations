from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from src.regulations_chunking.embed_and_upsert import _safe_metadata, make_vector_id


@dataclass
class RetrievalResult:
    id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str]


def _derive_index_name(index_name: Optional[str], host: Optional[str]) -> Optional[str]:
    if index_name:
        return index_name
    if not host:
        return None
    cleaned = host.replace("https://", "").replace("http://", "")
    return cleaned.split(".", maxsplit=1)[0] if cleaned else None


class RagRetriever:
    def __init__(
        self,
        *,
        model_name: str = (os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-base-en"),
        index_name: Optional[str] = None,
        host: Optional[str] = None,
        namespace: Optional[str] = None,
        chunks_root: Union[Path, str] = Path("chunks"),
    ) -> None:
        load_dotenv()
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing PINECONE_API_KEY in environment or .env")

        self.host = host or os.environ.get("PINECONE_HOST")
        self.index_name = _derive_index_name(
            index_name or os.environ.get("PINECONE_INDEX_NAME"),
            self.host,
        )
        if not self.index_name:
            raise RuntimeError("Missing index name (pass index_name or set PINECONE_INDEX_NAME)")

        self.namespace = namespace if namespace is not None else os.environ.get("PINECONE_NAMESPACE")
        self.chunks_root = Path(chunks_root)
        self.model = SentenceTransformer(model_name)
        pc = Pinecone(api_key=api_key)
        self.index = (
            pc.Index(name=self.index_name, host=self.host)
            if self.host
            else pc.Index(name=self.index_name)
        )

    def _hydrate_text(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Return chunk text from Pinecone metadata only.

        This project now treats Pinecone as the sole source of retrieved text; we do not
        read local `chunks/` JSONL files to "hydrate" missing text.
        """
        chunk_text = metadata.get("chunk_text")
        return chunk_text.strip() if isinstance(chunk_text, str) and chunk_text.strip() else None

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        query_vec = self.model.encode(
            [f"query: {query.strip()}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].tolist()

        kwargs: Dict[str, Any] = {
            "vector": query_vec,
            "top_k": top_k,
            "include_metadata": True,
            "include_values": False,
        }
        if self.namespace:
            kwargs["namespace"] = self.namespace
        response = self.index.query(**kwargs)
        matches = (
            response.get("matches", [])
            if isinstance(response, dict)
            else getattr(response, "matches", [])
        )

        out: List[RetrievalResult] = []
        for m in matches:
            row = m if isinstance(m, dict) else m.to_dict()
            metadata = row.get("metadata", {}) or {}
            vector_id = row.get("id", "")
            out.append(
                RetrievalResult(
                    id=vector_id,
                    score=float(row.get("score", 0.0)),
                    metadata=metadata,
                    text=self._hydrate_text(metadata),
                )
            )
        return out

    def retrieve_by_reference_id(self, ref_id: str, *, top_k: int = 3) -> List[RetrievalResult]:
        """
        Retrieve chunks from Pinecone by matching common regulation id fields in metadata.

        This is meant for **reference expansion** (e.g. "see Article 3.2", "E12.4", "Appendix 4"),
        and intentionally does **not** fall back to reading local `chunks/` files for text. It
        expects the index to have `chunk_text` stored in metadata (see upsert flag
        `--store-text-in-metadata`).
        """
        rid = (ref_id or "").strip()
        if not rid:
            return []

        # A vector is still required by Pinecone's query API; we use a neutral vector and rely on metadata filtering.
        dim = int(getattr(self.model, "get_sentence_embedding_dimension", lambda: 0)() or 0)
        if dim <= 0:
            probe = self.model.encode(
                ["query: dimension probe"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0].tolist()
            dim = len(probe)
        vector = [0.0] * dim

        filters = [
            {"parent_clause_id": {"$eq": rid}},
            {"article_id": {"$eq": rid}},
            {"appendix_id": {"$eq": rid}},
            {"clause_ids": {"$in": [rid]}},
        ]

        out: List[RetrievalResult] = []
        seen: set[str] = set()
        for f in filters:
            kwargs: Dict[str, Any] = {
                "vector": vector,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False,
                "filter": f,
            }
            if self.namespace:
                kwargs["namespace"] = self.namespace
            response = self.index.query(**kwargs)
            matches = (
                response.get("matches", [])
                if isinstance(response, dict)
                else getattr(response, "matches", [])
            )
            for m in matches:
                row = m if isinstance(m, dict) else m.to_dict()
                vector_id = row.get("id", "")
                if not vector_id or vector_id in seen:
                    continue
                seen.add(vector_id)
                metadata = row.get("metadata", {}) or {}
                text = metadata.get("chunk_text")
                out.append(
                    RetrievalResult(
                        id=vector_id,
                        score=float(row.get("score", 0.0)),
                        metadata=metadata,
                        text=text.strip() if isinstance(text, str) and text.strip() else None,
                    )
                )
        return out
