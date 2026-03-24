from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict


class ChunkDoc(TypedDict):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    source: Literal["semantic", "exact", "reference"]


class AdvancedRagState(TypedDict):
    user_query: str
    chat_history: List[Dict[str, str]]
    session_id: str

    resolved_query: str
    is_follow_up: bool
    active_topic: str
    active_article_ids: List[str]
    active_clause_ids: List[str]

    rewritten_query: str
    query_type: str
    needs_reference_expansion: bool

    retrieved_chunks: List[ChunkDoc]
    filtered_chunks: List[ChunkDoc]
    extracted_references: List[str]
    referenced_chunks: List[ChunkDoc]
    final_context_chunks: List[ChunkDoc]

    draft_answer: str
    cited_clause_ids: List[str]
    answer_supported: bool
    support_notes: str
    retry_count: int
