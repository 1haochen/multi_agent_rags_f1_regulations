from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .graph import build_advanced_rag_graph
from .memory import SessionMemoryStore
from .retrieval import HybridRetriever
from .state import AdvancedRagState, ChunkDoc

# Repository root (parent of ``src/``)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class AdvancedRagResponse:
    answer: str
    resolved_query: str
    query_type: str
    citations: List[ChunkDoc]
    answer_supported: bool
    support_notes: str


class AdvancedRegulationAssistant:
    def __init__(
        self,
        *,
        llm_model: str = "gpt-4o-mini",
        index_name: Optional[str] = None,
        host: Optional[str] = None,
        namespace: Optional[str] = None,
        chunks_root: str = "chunks",
        top_k: int = 8,
        answer_synthesizer_backend: Literal["openai", "local_qwen"] = "openai",
        answer_synthesizer_openai_model: Optional[str] = None,
        qwen_adapter_path: Optional[str] = None,
        qwen_use_base_only: bool = False,
        qwen_base_model: Optional[str] = None,
        qwen_max_new_tokens: int = 1024,
        qwen_device_map: Optional[str] = "auto",
        qwen_torch_dtype: Optional[str] = None,
        qwen_merge_adapters: bool = True,
        qwen_do_sample: bool = False,
    ) -> None:
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
        self.client = OpenAI(api_key=api_key)
        self.model = llm_model
        self.answer_synthesizer_backend = answer_synthesizer_backend
        self.qwen_use_base_only = False
        self.memory = SessionMemoryStore(max_turns=6)
        self.retriever = HybridRetriever(
            index_name=index_name or os.environ.get("PINECONE_INDEX_NAME"),
            host=host or os.environ.get("PINECONE_HOST"),
            namespace=namespace if namespace is not None else os.environ.get("PINECONE_NAMESPACE"),
            chunks_root=chunks_root,
        )

        qwen_synthesizer = None
        if answer_synthesizer_backend == "local_qwen":
            from .local_qwen_synthesizer import get_shared_qwen_synthesizer

            base_only = qwen_use_base_only or os.environ.get(
                "ADVANCED_RAG_QWEN_BASE_ONLY", ""
            ).strip().lower() in ("1", "true", "yes")
            synth_kw = dict(
                base_model_name_or_path=qwen_base_model,
                max_new_tokens=qwen_max_new_tokens,
                device_map=qwen_device_map,
                torch_dtype=qwen_torch_dtype,
                merge_adapters=qwen_merge_adapters,
                do_sample=qwen_do_sample,
            )
            if base_only:
                self.qwen_use_base_only = True
                qwen_synthesizer = get_shared_qwen_synthesizer(None, **synth_kw)
            else:
                adapter = (
                    Path(qwen_adapter_path).expanduser().resolve()
                    if qwen_adapter_path
                    else (REPO_ROOT / "models" / "Qwen2.5-1.5B-lora")
                )
                qwen_synthesizer = get_shared_qwen_synthesizer(adapter, **synth_kw)
        elif answer_synthesizer_backend != "openai":
            raise ValueError(
                f"answer_synthesizer_backend must be 'openai' or 'local_qwen', got {answer_synthesizer_backend!r}"
            )

        synth_openai = answer_synthesizer_openai_model or llm_model
        self.graph = build_advanced_rag_graph(
            client=self.client,
            model=self.model,
            retriever=self.retriever,
            top_k=top_k,
            synthesizer_openai_model=synth_openai,
            qwen_synthesizer=qwen_synthesizer,
        )

    def _initial_state(
        self,
        *,
        query: str,
        session_id: str,
        chat_history: Optional[List[Dict[str, str]]],
    ) -> AdvancedRagState:
        short_mem = self.memory.recent_history(session_id)
        merged_history = list(chat_history or [])
        for t in short_mem:
            merged_history.append({"role": "user", "content": t["user_query"]})
            merged_history.append({"role": "assistant", "content": t["answer"]})

        active_clause_ids: List[str] = []
        for t in short_mem:
            active_clause_ids.extend(t.get("cited_clause_ids", []))
        active_topic = short_mem[-1]["active_topic"] if short_mem else ""

        return {
            "user_query": query,
            "chat_history": merged_history[-10:],
            "session_id": session_id,
            "resolved_query": query,
            "is_follow_up": False,
            "active_topic": active_topic,
            "active_article_ids": [],
            "active_clause_ids": list(dict.fromkeys(active_clause_ids)),
            "rewritten_query": query,
            "query_type": "direct_lookup",
            "needs_reference_expansion": False,
            "retrieved_chunks": [],
            "filtered_chunks": [],
            "extracted_references": [],
            "referenced_chunks": [],
            "final_context_chunks": [],
            "draft_answer": "",
            "cited_clause_ids": [],
            "answer_supported": False,
            "support_notes": "",
            "retry_count": 0,
        }

    def ask(
        self,
        query: str,
        *,
        session_id: str = "default",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> AdvancedRagResponse:
        import time

        state = self._initial_state(query=query, session_id=session_id, chat_history=chat_history)
        t0 = time.perf_counter()
        out = self.graph.invoke(state)
        if os.environ.get("ADVANCED_RAG_DEBUG", "").strip() in ("1", "true", "yes"):
            print(
                f"[advanced_rag] graph.invoke total: {time.perf_counter() - t0:.1f}s "
                f"(synthesizer={self.answer_synthesizer_backend})",
                flush=True,
            )
        self.memory.add_turn(
            session_id=session_id,
            user_query=query,
            answer=out.get("draft_answer", ""),
            cited_clause_ids=list(out.get("cited_clause_ids", [])),
            active_topic=out.get("active_topic", ""),
        )
        return AdvancedRagResponse(
            answer=out.get("draft_answer", ""),
            resolved_query=out.get("resolved_query", query),
            query_type=out.get("query_type", "direct_lookup"),
            citations=list(out.get("final_context_chunks", [])),
            answer_supported=bool(out.get("answer_supported", False)),
            support_notes=out.get("support_notes", ""),
        )
