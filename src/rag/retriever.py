from __future__ import annotations

import json
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
        model_name: str = "BAAI/bge-large-en-v1.5",
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

    def _hydrate_text(self, vector_id: str, metadata: Dict[str, Any]) -> Optional[str]:
        chunk_text = metadata.get("chunk_text")
        if isinstance(chunk_text, str) and chunk_text.strip():
            return chunk_text.strip()

        source_file = metadata.get("source_file")
        if not source_file:
            return None
        chunk_path = self.chunks_root / str(source_file)
        if not chunk_path.exists():
            return None

        source_row = metadata.get("source_row")
        if isinstance(source_row, int):
            with chunk_path.open("r", encoding="utf-8") as f:
                for row_idx, line in enumerate(f):
                    if row_idx != source_row:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        return None
                    text = (row.get("text") or "").strip() if isinstance(row, dict) else ""
                    return text or None
            return None

        # Backward compatibility when older vectors have no source_row.
        with chunk_path.open("r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                row_meta = _safe_metadata(
                    row.get("metadata") if isinstance(row, dict) else {},
                    source_file=str(source_file),
                    source_row=row_idx,
                )
                row_vector_id = make_vector_id(str(source_file), row_idx, row_meta)
                if row_vector_id == vector_id:
                    return text
        return None

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
                    text=self._hydrate_text(vector_id, metadata),
                )
            )
        return out
