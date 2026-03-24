from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, TypedDict


class SessionTurn(TypedDict):
    user_query: str
    answer: str
    cited_clause_ids: List[str]
    active_topic: str


class SessionMemoryStore:
    def __init__(self, max_turns: int = 6) -> None:
        self._sessions: Dict[str, Deque[SessionTurn]] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )

    def add_turn(
        self,
        *,
        session_id: str,
        user_query: str,
        answer: str,
        cited_clause_ids: List[str],
        active_topic: str,
    ) -> None:
        self._sessions[session_id].append(
            {
                "user_query": user_query,
                "answer": answer,
                "cited_clause_ids": list(cited_clause_ids),
                "active_topic": active_topic,
            }
        )

    def recent_history(self, session_id: str) -> List[SessionTurn]:
        return list(self._sessions.get(session_id, []))
