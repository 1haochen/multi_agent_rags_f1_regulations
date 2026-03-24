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
        "--answer-synthesizer",
        choices=("openai", "local_qwen"),
        default="openai",
        help="Answer synthesizer: OpenAI JSON (gpt-4o-mini by default) or local fine-tuned Qwen LoRA.",
    )
    p.add_argument(
        "--synthesizer-openai-model",
        default=None,
        type=str,
        help="OpenAI model name for the synthesizer only (default: same as --llm-model).",
    )
    p.add_argument(
        "--qwen-adapter-path",
        default=None,
        type=str,
        help="PEFT adapter dir (flat export or Trainer output; e.g. models/Qwen2.5-1.5B-lora).",
    )
    p.add_argument(
        "--qwen-base-only",
        action="store_true",
        help="Use Hugging Face base Qwen only (no LoRA). Implies --answer-synthesizer local_qwen.",
    )
    p.add_argument(
        "--qwen-base-model",
        default=None,
        type=str,
        help="HF model id or path for base-only mode (default: Qwen/Qwen2.5-1.5B-Instruct or ADVANCED_RAG_QWEN_BASE_MODEL).",
    )
    p.add_argument("--qwen-max-new-tokens", default=1024, type=int)
    p.add_argument("--chunks-root", default="chunks", type=str)
    p.add_argument("--index-name", default=None, type=str)
    p.add_argument("--namespace", default=None, type=str)
    p.add_argument("--host", default=None, type=str)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    synth = args.answer_synthesizer
    if args.qwen_base_only:
        synth = "local_qwen"
    assistant = AdvancedRegulationAssistant(
        llm_model=args.llm_model,
        top_k=args.top_k,
        chunks_root=args.chunks_root,
        index_name=args.index_name,
        namespace=args.namespace,
        host=args.host,
        answer_synthesizer_backend=synth,
        answer_synthesizer_openai_model=args.synthesizer_openai_model,
        qwen_adapter_path=args.qwen_adapter_path,
        qwen_use_base_only=args.qwen_base_only,
        qwen_base_model=args.qwen_base_model,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
    )
    resp = assistant.ask(args.query, session_id=args.session_id)
    print(
        f"[synthesizer] {assistant.answer_synthesizer_backend}"
        + (" (base Qwen, no LoRA)" if assistant.qwen_use_base_only else "")
    )
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
