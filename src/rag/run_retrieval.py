from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from .retriever import RagRetriever
import os


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal RAG retrieval CLI.")
    parser.add_argument("--query", type=str, required=True, help="Natural language query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=(os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-base-en"),
        help="Embedding model for query encoding.",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Pinecone index name (or set PINECONE_INDEX_NAME).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Pinecone host URL (or set PINECONE_HOST).",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Pinecone namespace (or set PINECONE_NAMESPACE).",
    )
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=Path("chunks"),
        help="(Deprecated) Previously used for local text hydration; retrieval now uses Pinecone metadata only.",
    )
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=600,
        help="Maximum characters of hydrated text to print per result.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    retriever = RagRetriever(
        model_name=args.model_name,
        index_name=args.index_name,
        host=args.host,
        namespace=args.namespace,
        chunks_root=args.chunks_root,
    )

    results = retriever.retrieve(args.query, top_k=args.top_k)
    if not results:
        print("[info] no results")
        return 0

    print(
        f"[ok] query='{args.query}' index={retriever.index_name} "
        f"namespace={retriever.namespace or '(default)'} top_k={args.top_k}"
    )
    for idx, r in enumerate(results, start=1):
        meta = r.metadata
        clause = meta.get("parent_clause_id") or meta.get("article_id") or meta.get("appendix_id") or ""
        source = meta.get("source_file", "(unknown)")
        scope = f"{meta.get('scope', '')}/{meta.get('chunk_scope', '')}"
        print(f"[{idx}] score={r.score:.4f} id={r.id} scope={scope} clause={clause} source={source}")
        if r.text:
            print(f"    text: {r.text[: max(0, args.text_max_chars)]}")
        else:
            print("    text: (not available)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

