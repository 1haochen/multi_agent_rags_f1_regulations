from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from .prompts import (
    ANSWER_CHECK_PROMPT,
    ANSWER_SYNTHESIZER_PROMPT,
    CONVERSATION_MEMORY_PROMPT,
    QUERY_PLANNER_PROMPT,
    REFERENCE_RESOLVER_PROMPT,
    RELEVANCE_JUDGE_PROMPT,
)
from .retrieval import ChunkDocObj, extract_ids
from .state import AdvancedRagState, ChunkDoc


def _json_response(client: OpenAI, *, model: str, system: str, user: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = (resp.choices[0].message.content or "{}").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _obj_to_doc(obj: ChunkDocObj) -> ChunkDoc:
    return {
        "id": obj.id,
        "score": float(obj.score),
        "text": obj.text,
        "metadata": obj.metadata,
        "source": obj.source,  # type: ignore[typeddict-item]
    }


def _render_chunks(chunks: List[ChunkDoc], max_chars: int = 9000) -> str:
    lines: List[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        md = c.get("metadata", {})
        cid = md.get("parent_clause_id") or md.get("article_id") or md.get("appendix_id") or ""
        block = (
            f"[Chunk {i}] id={c['id']} score={c['score']:.4f} clause={cid} "
            f"source={md.get('source_file', '')}\n{c['text'].strip()}\n"
        )
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)


def conversation_memory_node(state: AdvancedRagState, *, client: OpenAI, model: str) -> Dict[str, Any]:
    user = json.dumps(
        {
            "user_query": state["user_query"],
            "chat_history": state.get("chat_history", []),
            "active_topic": state.get("active_topic", ""),
            "active_article_ids": state.get("active_article_ids", []),
            "active_clause_ids": state.get("active_clause_ids", []),
        },
        ensure_ascii=True,
    )
    data = _json_response(client, model=model, system=CONVERSATION_MEMORY_PROMPT, user=user)
    return {
        "is_follow_up": bool(data.get("is_follow_up", False)),
        "resolved_query": str(data.get("resolved_query") or state["user_query"]),
        "active_topic": str(data.get("active_topic") or state.get("active_topic", "")),
        "active_article_ids": list(data.get("active_article_ids") or state.get("active_article_ids", [])),
        "active_clause_ids": list(data.get("active_clause_ids") or state.get("active_clause_ids", [])),
    }


def query_planner_node(state: AdvancedRagState, *, client: OpenAI, model: str) -> Dict[str, Any]:
    user = json.dumps(
        {
            "resolved_query": state["resolved_query"],
            "active_topic": state.get("active_topic", ""),
            "active_article_ids": state.get("active_article_ids", []),
            "active_clause_ids": state.get("active_clause_ids", []),
        },
        ensure_ascii=True,
    )
    data = _json_response(client, model=model, system=QUERY_PLANNER_PROMPT, user=user)
    likely_ids = list(data.get("likely_ids") or [])
    likely_ids.extend(extract_ids(state["resolved_query"]))
    return {
        "rewritten_query": str(data.get("rewritten_query") or state["resolved_query"]),
        "query_type": str(data.get("query_type") or "direct_lookup"),
        "needs_reference_expansion": bool(data.get("needs_reference_expansion", False)),
        "active_clause_ids": list(dict.fromkeys(state.get("active_clause_ids", []) + likely_ids)),
    }


def retriever_node(state: AdvancedRagState, *, retriever: Any, top_k: int) -> Dict[str, Any]:
    explicit_ids = list(dict.fromkeys(state.get("active_clause_ids", []) + state.get("active_article_ids", [])))
    effective_top_k = int(state.get("top_k", top_k))
    docs = retriever.retrieve(state["rewritten_query"], explicit_ids=explicit_ids, top_k=effective_top_k)
    return {"retrieved_chunks": [_obj_to_doc(x) for x in docs]}


def relevance_judge_node(state: AdvancedRagState, *, client: OpenAI, model: str) -> Dict[str, Any]:
    chunks = state.get("retrieved_chunks", [])
    payload = json.dumps(
        {
            "query": state["resolved_query"],
            "chunks": [
                {"id": c["id"], "text": c["text"][:5000], "metadata": c.get("metadata", {})}
                for c in chunks
            ],
        },
        ensure_ascii=True,
    )
    data = _json_response(client, model=model, system=RELEVANCE_JUDGE_PROMPT, user=payload)
    kept_ids = set(data.get("kept_ids") or [])
    if not kept_ids:
        ranked = sorted(chunks, key=lambda c: c["score"], reverse=True)
        return {"filtered_chunks": ranked[: min(5, len(ranked))]}
    return {"filtered_chunks": [c for c in chunks if c["id"] in kept_ids]}


def reference_resolver_node(state: AdvancedRagState, *, client: OpenAI, model: str, retriever: Any) -> Dict[str, Any]:
    filtered = state.get("filtered_chunks", [])
    payload = json.dumps(
        {
            "query": state["resolved_query"],
            "chunks": [{"id": c["id"], "text": c["text"][:5000]} for c in filtered],
        },
        ensure_ascii=True,
    )
    data = _json_response(client, model=model, system=REFERENCE_RESOLVER_PROMPT, user=payload)
    refs = list(data.get("references") or [])
    # Regex fallback catches explicit clause/article mentions.
    for c in filtered:
        refs.extend(extract_ids(c["text"]))
    refs = list(dict.fromkeys(refs))
    ref_docs = retriever.retrieve_references_from_pinecone(refs, limit=6) if refs else []
    referenced_chunks = [_obj_to_doc(x) for x in ref_docs]
    merged: Dict[str, ChunkDoc] = {c["id"]: c for c in filtered}
    for c in referenced_chunks:
        merged[c["id"]] = c
    return {
        "extracted_references": refs,
        "referenced_chunks": referenced_chunks,
        "final_context_chunks": list(merged.values()),
    }


def answer_synthesizer_node(
    state: AdvancedRagState,
    *,
    client: OpenAI,
    openai_model: str,
) -> Dict[str, Any]:
    context = _render_chunks(state.get("final_context_chunks", []), max_chars=20000)
    payload = json.dumps(
        {"query": state["resolved_query"], "query_type": state.get("query_type", ""), "context": context},
        ensure_ascii=True,
    )
    data = _json_response(client, model=openai_model, system=ANSWER_SYNTHESIZER_PROMPT, user=payload)
    ans_out = (data.get("answer") or "").strip()
    if not ans_out:
        ans_out = "I do not know based on the retrieved context."
    return {
        "draft_answer": str(ans_out),
        "cited_clause_ids": list(data.get("cited_clause_ids") or []),
    }


def answer_check_node(state: AdvancedRagState, *, client: OpenAI, model: str) -> Dict[str, Any]:
    context = _render_chunks(state.get("final_context_chunks", []), max_chars=8000)
    payload = json.dumps(
        {
            "query": state["resolved_query"],
            "draft_answer": state.get("draft_answer", ""),
            "context": context,
            "retry_count": state.get("retry_count", 0),
        },
        ensure_ascii=True,
    )
    data = _json_response(client, model=model, system=ANSWER_CHECK_PROMPT, user=payload)
    supported = bool(data.get("answer_supported", False))
    notes = str(data.get("missing_evidence", "")).strip()
    hint = str(data.get("additional_query_hint", "")).strip()

    updates: Dict[str, Any] = {"answer_supported": supported, "support_notes": notes}
    if not supported and state.get("retry_count", 0) < 1:
        base = state.get("rewritten_query", state["resolved_query"])
        updates["rewritten_query"] = f"{base}. Also resolve: {hint}" if hint else base
        updates["retry_count"] = state.get("retry_count", 0) + 1
        updates["top_k"] = max(int(state.get("top_k", 8)), 12)
    return updates
