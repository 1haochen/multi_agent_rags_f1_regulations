from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .retriever import RagRetriever, RetrievalResult


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG QA with Pinecone + OpenAI.")
    parser.add_argument("--query", type=str, required=True, help="User question.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieved chunks to use.")
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model for final answer.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=(os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-base-en"),
        help="Embedding model for retrieval queries.",
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
        type=str,
        default="chunks",
        help="(Deprecated) Previously used for local text hydration; retrieval now uses Pinecone metadata only.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help="Hard cap for assembled retrieval context.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def build_context(results: List[RetrievalResult], max_context_chars: int) -> str:
    sections: List[str] = []
    used = 0
    for i, r in enumerate(results, start=1):
        text = (r.text or "").strip()
        if not text:
            continue
        m = r.metadata
        clause = m.get("parent_clause_id") or m.get("article_id") or m.get("appendix_id") or ""
        source = m.get("source_file", "(unknown)")
        header = f"[Chunk {i}] source={source} clause={clause} score={r.score:.4f}"
        block = f"{header}\n{text}\n"
        if used + len(block) > max_context_chars:
            remaining = max_context_chars - used
            if remaining <= 0:
                break
            block = (block[:remaining]).rstrip() + "\n"
        sections.append(block)
        used += len(block)
        if used >= max_context_chars:
            break
    return "\n".join(sections).strip()


def answer_with_rag(
    *,
    query: str,
    context: str,
    model: str,
    client: OpenAI,
) -> str:
    system = (
        "You are a precise regulations assistant. Answer ONLY from provided context. "
        "If the context is insufficient, say so explicitly. Include brief citations in "
        "the form [Chunk N]."
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Provide a concise, direct answer with citations."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    return (resp.choices[0].message.content or "").strip()


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("[error] missing OPENAI_API_KEY in environment/.env")
        return 1

    retriever = RagRetriever(
        model_name=args.embedding_model,
        index_name=args.index_name,
        host=args.host,
        namespace=args.namespace,
        chunks_root=args.chunks_root,
    )
    results = retriever.retrieve(args.query, top_k=args.top_k)
    if not results:
        print("[info] no retrieval results")
        return 0

    context = build_context(results, max_context_chars=args.max_context_chars)
    if not context:
        print("[info] retrieval results had no usable text")
        return 0

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    answer = answer_with_rag(
        query=args.query,
        context=context,
        model=args.llm_model,
        client=client,
    )

    print(f"[query] {args.query}")
    print(f"[model] {args.llm_model}")
    print("\n[answer]")
    print(answer)
    print("\n[citations]")
    for i, r in enumerate(results, start=1):
        m = r.metadata
        clause = m.get("parent_clause_id") or m.get("article_id") or m.get("appendix_id") or ""
        print(f"[Chunk {i}] score={r.score:.4f} clause={clause} source={m.get('source_file')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

