from __future__ import annotations

"""
Embed chunked regulations data and upsert vectors to Pinecone.

Expected chunk JSONL rows:
    {"text": "...", "metadata": {...}}
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunk JSONL files and upsert to Pinecone."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("chunks"),
        help="Directory containing *.chunks.jsonl files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.chunks.jsonl",
        help="Glob pattern used to find chunk files.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Pinecone index name (or set PINECONE_INDEX_NAME).",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Pinecone namespace for upsert (or set PINECONE_NAMESPACE).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Pinecone index host URL (or set PINECONE_HOST).",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size for sentence-transformer encoding.",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=100,
        help="Number of vectors per Pinecone upsert request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of records processed.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum retry attempts for transient upsert failures.",
    )
    parser.add_argument(
        "--create-index-if-missing",
        action="store_true",
        help="Create index with cosine metric when missing.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=("cosine", "dotproduct", "euclidean"),
        help="Metric used when creating a new index.",
    )
    parser.add_argument(
        "--cloud",
        type=str,
        default="aws",
        help="Cloud provider used for serverless index creation.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="Region used for serverless index creation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run loading/embedding only, no Pinecone writes.",
    )
    parser.add_argument(
        "--store-text-in-metadata",
        action="store_true",
        help="Store truncated chunk text in Pinecone metadata for direct retrieval.",
    )
    parser.add_argument(
        "--metadata-text-max-chars",
        type=int,
        default=1200,
        help="Max characters for text stored in metadata when enabled.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


@dataclass
class ChunkRecord:
    vector_id: str
    text: str
    metadata: Dict[str, Any]


def discover_chunk_files(input_dir: Path, pattern: str) -> List[Path]:
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def _safe_metadata(raw: Any, source_file: str, source_row: int) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    meta: Dict[str, Any] = {
        "source_file": source_file,
        "source_row": int(source_row),
        "scope": raw.get("scope"),
        "chunk_scope": raw.get("chunk_scope"),
        "article_id": raw.get("article_id"),
        "appendix_id": raw.get("appendix_id"),
        "parent_clause_id": raw.get("parent_clause_id"),
        "has_table": bool(raw.get("has_table", False)),
    }

    page_indices = raw.get("page_indices", [])
    clause_ids = raw.get("clause_ids", [])
    if isinstance(page_indices, list):
        # Pinecone metadata supports list[str], not list[number].
        meta["page_indices"] = [str(int(x)) for x in page_indices[:100] if isinstance(x, int)]
    if isinstance(clause_ids, list):
        meta["clause_ids"] = [str(x)[:64] for x in clause_ids[:200]]

    return {k: v for k, v in meta.items() if v is not None}


def make_vector_id(source_file: str, row_index: int, metadata: Dict[str, Any]) -> str:
    article_id = metadata.get("article_id") or ""
    appendix_id = metadata.get("appendix_id") or ""
    clause = metadata.get("parent_clause_id") or ""
    payload = f"{source_file}|{row_index}|{article_id}|{appendix_id}|{clause}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"ch_{digest}"


def load_records(
    chunk_files: Sequence[Path], input_root: Path, limit: Optional[int]
) -> Tuple[List[ChunkRecord], int]:
    records: List[ChunkRecord] = []
    skipped = 0
    for path in chunk_files:
        rel_source = str(path.relative_to(input_root))
        with path.open("r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                if limit is not None and len(records) >= limit:
                    return records, skipped
                line = line.strip()
                if not line:
                    skipped += 1
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                text = (row.get("text") or "").strip() if isinstance(row, dict) else ""
                if not text:
                    skipped += 1
                    continue
                metadata = _safe_metadata(
                    row.get("metadata") if isinstance(row, dict) else {},
                    source_file=rel_source,
                    source_row=row_idx,
                )
                rec_id = make_vector_id(rel_source, row_idx, metadata)
                records.append(ChunkRecord(vector_id=rec_id, text=text, metadata=metadata))
    return records, skipped


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def derive_index_name(index_name: Optional[str], host: Optional[str]) -> Optional[str]:
    if index_name:
        return index_name
    if not host:
        return None
    cleaned = host.replace("https://", "").replace("http://", "")
    return cleaned.split(".", maxsplit=1)[0] if cleaned else None


def ensure_index(
    pc: Pinecone,
    index_name: str,
    embedding_dim: int,
    metric: str,
    create_if_missing: bool,
    cloud: str,
    region: str,
) -> None:
    names = {entry["name"] for entry in pc.list_indexes()}
    if index_name not in names:
        if not create_if_missing:
            raise RuntimeError(
                f"Index '{index_name}' not found. Re-run with --create-index-if-missing."
            )
        print(
            f"[index] creating index name={index_name} dim={embedding_dim} "
            f"metric={metric} cloud={cloud} region={region}"
        )
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    desc = pc.describe_index(index_name)
    got_dim = getattr(desc, "dimension", None)
    if got_dim is None and isinstance(desc, dict):
        got_dim = desc.get("dimension")
    if got_dim != embedding_dim:
        raise RuntimeError(
            f"Index dimension mismatch for '{index_name}': expected "
            f"{embedding_dim}, got {got_dim}"
        )


def with_retries_upsert(
    *,
    index: Any,
    vectors: List[Dict[str, Any]],
    namespace: Optional[str],
    max_retries: int,
) -> None:
    for attempt in range(max_retries + 1):
        try:
            kwargs: Dict[str, Any] = {"vectors": vectors}
            if namespace:
                kwargs["namespace"] = namespace
            index.upsert(**kwargs)
            return
        except Exception:
            if attempt >= max_retries:
                raise
            delay_s = 0.5 * (2**attempt)
            time.sleep(delay_s)


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    input_dir = args.input_dir
    if not input_dir.exists():
        print(f"[error] input directory not found: {input_dir}")
        return 1

    chunk_files = discover_chunk_files(input_dir, args.pattern)
    if not chunk_files:
        print(f"[info] no files found: pattern={args.pattern} input_dir={input_dir}")
        return 0

    print(f"[load] found {len(chunk_files)} chunk file(s)")
    records, skipped = load_records(chunk_files, input_dir, args.limit)
    if not records:
        print(f"[info] no usable records loaded (skipped={skipped})")
        return 0
    print(f"[load] records={len(records)} skipped={skipped}")

    print(f"[embed] loading model {args.model_name}")
    model = SentenceTransformer(args.model_name)
    embed_dim = model.get_sentence_embedding_dimension()
    if embed_dim is None:
        raise RuntimeError("Could not determine embedding dimension from model.")

    vectors: List[Dict[str, Any]] = []
    for group in batched(records, args.embed_batch_size):
        passages = [f"passage: {r.text}" for r in group]
        embeds = model.encode(
            passages,
            batch_size=args.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for rec, emb in zip(group, embeds):
            metadata = dict(rec.metadata)
            if args.store_text_in_metadata:
                metadata["chunk_text"] = rec.text[: max(0, args.metadata_text_max_chars)]
            vectors.append(
                {
                    "id": rec.vector_id,
                    "values": emb.tolist(),
                    "metadata": metadata,
                }
            )
    print(f"[embed] encoded vectors={len(vectors)} dim={embed_dim}")

    if args.dry_run:
        print("[dry-run] skipping Pinecone upsert")
        return 0

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("[error] missing PINECONE_API_KEY in environment/.env")
        return 1

    host = args.host or os.environ.get("PINECONE_HOST")
    namespace = args.namespace if args.namespace is not None else os.environ.get("PINECONE_NAMESPACE")
    index_name = derive_index_name(
        args.index_name or os.environ.get("PINECONE_INDEX_NAME"),
        host,
    )
    if not index_name:
        print(
            "[error] index name is required (set --index-name or PINECONE_INDEX_NAME)."
        )
        return 1

    pc = Pinecone(api_key=api_key)
    ensure_index(
        pc=pc,
        index_name=index_name,
        embedding_dim=embed_dim,
        metric=args.metric,
        create_if_missing=args.create_index_if_missing,
        cloud=args.cloud,
        region=args.region,
    )

    index = pc.Index(name=index_name, host=host) if host else pc.Index(name=index_name)
    upserted = 0
    for group in batched(vectors, args.upsert_batch_size):
        with_retries_upsert(
            index=index,
            vectors=list(group),
            namespace=namespace,
            max_retries=args.max_retries,
        )
        upserted += len(group)
        print(f"[upsert] {upserted}/{len(vectors)}")

    print(
        f"[done] upserted={upserted} namespace={namespace or '(default)'} "
        f"index={index_name}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

