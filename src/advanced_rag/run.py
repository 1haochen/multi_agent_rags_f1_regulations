"""
CLI for the multi-agent regulation RAG.

From repository root:
    python -m src.advanced_rag.run --query "..."
"""

from __future__ import annotations

import argparse

from .service import AdvancedRegulationAssistant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run advanced multi-agent regulation RAG.")
    p.add_argument("--query", required=True, type=str)
    p.add_argument("--session-id", default="cli", type=str)
    p.add_argument("--top-k", default=8, type=int)
    p.add_argument("--llm-model", default="gpt-4o-mini", type=str)
    p.add_argument(
        "--synthesizer-openai-model",
        default=None,
        type=str,
        help="OpenAI model name for the synthesizer only (default: same as --llm-model).",
    )
    p.add_argument("--chunks-root", default="chunks", type=str)
    p.add_argument("--index-name", default=None, type=str)
    p.add_argument("--namespace", default=None, type=str)
    p.add_argument("--host", default=None, type=str)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    assistant = AdvancedRegulationAssistant(
        llm_model=args.llm_model,
        top_k=args.top_k,
        chunks_root=args.chunks_root,
        index_name=args.index_name,
        namespace=args.namespace,
        host=args.host,
        answer_synthesizer_openai_model=args.synthesizer_openai_model,
    )
    resp = assistant.ask(args.query, session_id=args.session_id)
    print(f"[synthesizer] {assistant.answer_synthesizer_backend}")
    print(f"[resolved_query] {resp.resolved_query}")
    print(f"[query_type] {resp.query_type}")
    print("\n[answer]")
    print(resp.answer)
    print(f"\n[supported] {resp.answer_supported} notes={resp.support_notes}")
    print("\n[citations]")
    for i, c in enumerate(resp.citations, start=1):
        md = c.get("metadata", {})
        clause = md.get("parent_clause_id") or md.get("article_id") or md.get("appendix_id") or ""
        print(f"[Chunk {i}] score={c['score']:.4f} clause={clause} source={md.get('source_file')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
