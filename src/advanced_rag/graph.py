from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from .nodes import (
    answer_check_node,
    answer_synthesizer_node,
    conversation_memory_node,
    query_planner_node,
    reference_resolver_node,
    relevance_judge_node,
    retriever_node,
)
from .state import AdvancedRagState


def build_advanced_rag_graph(
    *,
    client: OpenAI,
    model: str,
    retriever: object,
    top_k: int = 8,
    synthesizer_openai_model: Optional[str] = None,
) -> Callable[[AdvancedRagState], AdvancedRagState]:
    graph = StateGraph(AdvancedRagState)

    graph.add_node("conversation_memory_agent", lambda s: conversation_memory_node(s, client=client, model=model))
    graph.add_node("query_planner_agent", lambda s: query_planner_node(s, client=client, model=model))
    graph.add_node("retriever_agent", lambda s: retriever_node(s, retriever=retriever, top_k=top_k))
    graph.add_node("relevance_judge_agent", lambda s: relevance_judge_node(s, client=client, model=model))
    graph.add_node(
        "reference_resolver_agent",
        lambda s: reference_resolver_node(s, client=client, model=model, retriever=retriever),
    )
    synth_model = synthesizer_openai_model if synthesizer_openai_model is not None else model

    graph.add_node(
        "answer_synthesizer_agent",
        lambda s: answer_synthesizer_node(
            s,
            client=client,
            openai_model=synth_model,
        ),
    )
    graph.add_node("answer_check_agent", lambda s: answer_check_node(s, client=client, model=model))

    graph.add_edge(START, "conversation_memory_agent")
    graph.add_edge("conversation_memory_agent", "query_planner_agent")
    graph.add_edge("query_planner_agent", "retriever_agent")
    graph.add_edge("retriever_agent", "relevance_judge_agent")
    graph.add_edge("relevance_judge_agent", "reference_resolver_agent")
    graph.add_edge("reference_resolver_agent", "answer_synthesizer_agent")
    graph.add_edge("answer_synthesizer_agent", "answer_check_agent")

    def _route_after_check(state: AdvancedRagState) -> str:
        if state.get("answer_supported", False):
            return END
        if state.get("retry_count", 0) >= 1:
            return END
        return "retriever_agent"

    graph.add_conditional_edges("answer_check_agent", _route_after_check, {END: END, "retriever_agent": "retriever_agent"})
    return graph.compile()
