"""
Flask app: F1-themed chat UI for the multi-agent regulation assistant.

Run from repository root (with .env and venv):
    pip install flask
    python -m frontend.app

Or:
    flask --app frontend.app run --host 0.0.0.0 --port 5000

Optional env (default synthesizer if UI omits it; per-request override in JSON body):
    ADVANCED_RAG_SYNTHESIZER=openai|local_qwen|local_qwen_base
    ADVANCED_RAG_QWEN_ADAPTER_PATH=/path/to/Qwen2.5-1.5B-lora  # or .../checkpoint-370
    ADVANCED_RAG_QWEN_BASE_ONLY=1
    ADVANCED_RAG_QWEN_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

JSON body may include ``synthesizer``: ``"openai"`` | ``"local_qwen"`` | ``"local_qwen_base"`` (answer step only).
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List

from flask import Flask, Response, render_template, request

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = Path(__file__).resolve().parent

# One assistant per synthesizer backend (lazy init; first request pays load cost for that backend)
_assistants: Dict[str, Any] = {}


def _normalize_synthesizer(raw: str | None) -> str:
    v = (raw or "").strip().lower()
    if v in ("qwen_base", "local_qwen_base", "base_qwen"):
        return "local_qwen_base"
    if v in ("qwen", "local_qwen", "local", "lora"):
        return "local_qwen"
    return "openai"


def _get_assistant(synthesizer: str | None) -> Any:
    """Return cached AdvancedRegulationAssistant for openai vs local_qwen answer synthesizer."""
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.advanced_rag import AdvancedRegulationAssistant

    key = _normalize_synthesizer(synthesizer)
    if key not in _assistants:
        kw: dict = {}
        if key == "local_qwen":
            kw["answer_synthesizer_backend"] = "local_qwen"
            ap = os.environ.get("ADVANCED_RAG_QWEN_ADAPTER_PATH")
            if ap:
                kw["qwen_adapter_path"] = ap
        elif key == "local_qwen_base":
            kw["answer_synthesizer_backend"] = "local_qwen"
            kw["qwen_use_base_only"] = True
            bm = os.environ.get("ADVANCED_RAG_QWEN_BASE_MODEL")
            if bm:
                kw["qwen_base_model"] = bm
        _assistants[key] = AdvancedRegulationAssistant(**kw)
    return _assistants[key]


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(FRONTEND_DIR / "templates"),
        static_folder=str(FRONTEND_DIR / "static"),
    )

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/api/chat")
    def chat_json() -> tuple[Any, int]:
        """Non-streaming JSON fallback for clients that do not use SSE."""
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        session_id = (data.get("session_id") or "default").strip() or "default"
        synth_raw = data.get("synthesizer") or os.environ.get("ADVANCED_RAG_SYNTHESIZER", "openai")
        if not message:
            return {"error": "message required"}, 400
        try:
            r = _get_assistant(synth_raw).ask(message, session_id=session_id)
        except Exception as e:  # pragma: no cover
            return {"error": str(e)}, 500

        citations: List[Dict[str, Any]] = []
        for c in r.citations[:20]:
            if not isinstance(c, dict):
                continue
            md = c.get("metadata") or {}
            citations.append(
                {
                    "id": c.get("id"),
                    "score": c.get("score"),
                    "clause": md.get("parent_clause_id")
                    or md.get("article_id")
                    or md.get("appendix_id"),
                    "source_file": md.get("source_file"),
                }
            )

        return (
            {
                "answer": r.answer,
                "resolved_query": r.resolved_query,
                "query_type": r.query_type,
                "answer_supported": r.answer_supported,
                "support_notes": r.support_notes,
                "citations": citations,
                "synthesizer": _normalize_synthesizer(synth_raw),
            },
            200,
        )

    @app.post("/api/chat/stream")
    def chat_stream() -> Response:
        """
        Server-Sent Events while the pipeline runs:
        - Rotating phase labels (thinking) until `ask()` completes
        - Then token events (word chunks) for progressive answer display
        - Final meta event with citations and query metadata
        """
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        session_id = (data.get("session_id") or "default").strip() or "default"
        synth_raw = data.get("synthesizer") or os.environ.get("ADVANCED_RAG_SYNTHESIZER", "openai")
        synth_key = _normalize_synthesizer(synth_raw)
        if not message:
            return Response(
                json.dumps({"error": "message required"}),
                status=400,
                mimetype="application/json",
            )

        phases = [
            "Resolving conversation context…",
            "Planning retrieval targets…",
            "Retrieving regulation chunks…",
            "Judging relevance…",
            "Following cross-references…",
            "Synthesizing grounded answer…",
            "Verifying evidence support…",
        ]

        def generate() -> Generator[str, None, None]:
            result_q: queue.Queue = queue.Queue()
            assistant = _get_assistant(synth_raw)

            def worker() -> None:
                try:
                    result_q.put(("ok", assistant.ask(message, session_id=session_id)))
                except Exception as exc:  # pragma: no cover
                    result_q.put(("err", str(exc)))

            threading.Thread(target=worker, daemon=True).start()

            phase_idx = 0
            while True:
                try:
                    kind, payload = result_q.get(timeout=0.4)
                    break
                except queue.Empty:
                    label = phases[phase_idx % len(phases)]
                    phase_idx += 1
                    yield f"data: {json.dumps({'type': 'phase', 'label': label})}\n\n"

            if kind == "err":
                yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                yield "data: [DONE]\n\n"
                return

            r = payload
            answer = (r.answer or "").strip()
            if not answer:
                answer = "No answer was produced. Check retrieval and API keys."

            # Progressive display: stream by words (keeps punctuation on words)
            words = answer.split(" ")
            for i, w in enumerate(words):
                chunk = w + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"

            citations: List[Dict[str, Any]] = []
            for c in r.citations[:12]:
                if not isinstance(c, dict):
                    continue
                md = c.get("metadata") or {}
                citations.append(
                    {
                        "clause": md.get("parent_clause_id")
                        or md.get("article_id")
                        or md.get("appendix_id"),
                        "article": md.get("article_id"),
                        "appendix": md.get("appendix_id"),
                        "source_file": md.get("source_file"),
                    }
                )

            meta = {
                "type": "meta",
                "resolved_query": r.resolved_query,
                "query_type": r.query_type,
                "answer_supported": r.answer_supported,
                "support_notes": r.support_notes or "",
                "citations": citations,
                "synthesizer": synth_key,
            }
            yield f"data: {json.dumps(meta)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
