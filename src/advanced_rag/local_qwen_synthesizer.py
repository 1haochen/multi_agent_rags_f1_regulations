"""
Local Qwen for the answer synthesizer agent only (base model and/or LoRA).

- **LoRA:** load PEFT adapters from ``models/Qwen2.5-1.5B-lora/`` (flat) or ``.../checkpoint-*`` (Trainer).
  Base id is read from ``adapter_config.json`` unless you pass ``base_model_name_or_path``.
- **Base only:** pass ``adapter_path=None`` and set ``base_model_name_or_path`` (or rely on
  ``ADVANCED_RAG_QWEN_BASE_MODEL`` / default ``Qwen/Qwen2.5-1.5B-Instruct``).

Performance notes (why it can feel "slow"):
- **First call** downloads/loads the base model from Hugging Face (~1.5B) unless already cached
  under ``HF_HOME`` / ``~/.cache/huggingface`` — can take many minutes on a cold cache.
- **CPU inference** is much slower than GPU; use CUDA if available.
- Use **one shared synthesizer** (``get_shared_qwen_synthesizer``) so the weights load once per process.
- Set ``ADVANCED_RAG_DEBUG=1`` to print load/generate timings.

Environment:
- ``ADVANCED_RAG_DEBUG=1`` — log load and generation seconds to stderr.
- ``ADVANCED_RAG_QWEN_CONTEXT_CHARS`` — max characters of regulation context passed to the
  synthesizer (see ``nodes.answer_synthesizer_node``); default 12000 for Qwen.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.I)

_LOG = logging.getLogger(__name__)

# One loaded model per cache key per process — avoids duplicate base loads.
_SYNTH_BY_KEY: Dict[str, "LoRAQwenSynthesizer"] = {}


def resolve_peft_adapter_dir(adapter_dir: str | Path) -> Path:
    """
    Return a directory that contains ``adapter_config.json``.

    Supports:
    - Flat export: ``models/Qwen2.5-1.5B-lora/`` with config next to weights
    - Trainer layout: parent dir with ``checkpoint-*`` subfolders (picks latest step)
    - Stale path: ``.../checkpoint-370`` missing config but parent has a flat export
    """
    p = Path(adapter_dir).expanduser().resolve()
    if p.is_file():
        p = p.parent
    if (p / "adapter_config.json").is_file():
        return p
    if p.name.startswith("checkpoint-") and (p.parent / "adapter_config.json").is_file():
        return p.parent.resolve()
    if p.is_dir():
        checkpoints: list[tuple[int, Path]] = []
        try:
            for d in p.iterdir():
                if not d.is_dir() or not d.name.startswith("checkpoint-"):
                    continue
                if not (d / "adapter_config.json").is_file():
                    continue
                suffix = d.name[len("checkpoint-") :]
                try:
                    step = int(suffix)
                except ValueError:
                    step = 0
                checkpoints.append((step, d))
        except OSError:
            pass
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0])
            return checkpoints[-1][1].resolve()
    return p


def get_shared_qwen_synthesizer(adapter_path: str | Path | None, **kwargs: Any) -> "LoRAQwenSynthesizer":
    """
    Return a process-wide cached synthesizer.

    Pass ``adapter_path=None`` for **base model only** (no PEFT). Otherwise pass the adapter
    checkpoint directory. The first call may **load weights** (slow). Later calls reuse the model.
    Runtime overrides: ``max_new_tokens``, ``do_sample``.
    """
    merge = bool(kwargs.get("merge_adapters", True))
    if adapter_path is None:
        base = (
            (kwargs.get("base_model_name_or_path") or "").strip()
            or os.environ.get("ADVANCED_RAG_QWEN_BASE_MODEL", "").strip()
            or "Qwen/Qwen2.5-1.5B-Instruct"
        )
        dm = kwargs.get("device_map", "auto")
        dt = kwargs.get("torch_dtype")
        key = f"BASE|{base}|dm={dm}|dtype={dt}"
    else:
        key = f"{resolve_peft_adapter_dir(adapter_path)}|merge={merge}"
    if key not in _SYNTH_BY_KEY:
        _SYNTH_BY_KEY[key] = LoRAQwenSynthesizer(adapter_path, **kwargs)
    inst = _SYNTH_BY_KEY[key]
    if "max_new_tokens" in kwargs:
        inst.max_new_tokens = int(kwargs["max_new_tokens"])
    if "do_sample" in kwargs:
        inst.do_sample = bool(kwargs["do_sample"])
    return inst


def _debug_timing() -> bool:
    return os.environ.get("ADVANCED_RAG_DEBUG", "").strip() in ("1", "true", "yes")


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Extract a single JSON object from model output (handles fences, preambles, truncation)."""
    raw = (text or "").strip()
    m = _JSON_BLOCK.search(raw)
    if m:
        raw = m.group(1).strip()
    start = raw.find("{")
    if start == -1:
        return {}
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(raw, start)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    end = raw.rfind("}")
    if end == -1 or end <= start:
        return {}
    try:
        out = json.loads(raw[start : end + 1])
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


class LoRAQwenSynthesizer:
    """
    Lazy-loads Qwen once (base-only or base + LoRA), then runs chat-formatted generation for JSON.
    Prefer ``get_shared_qwen_synthesizer`` so the same process does not load weights twice.
    """

    def __init__(
        self,
        adapter_path: str | Path | None,
        *,
        base_model_name_or_path: Optional[str] = None,
        max_new_tokens: int = 1024,
        device_map: str | None = "auto",
        torch_dtype: Optional[str] = None,
        merge_adapters: bool = True,
        do_sample: bool = False,
    ) -> None:
        self.adapter_path = (
            resolve_peft_adapter_dir(adapter_path) if adapter_path is not None else None
        )
        self._base_override = base_model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self._dtype_name = torch_dtype  # "auto", "bfloat16", "float16", "float32"
        self.merge_adapters = merge_adapters
        self.do_sample = do_sample

        self._model = None
        self._tokenizer = None

    def _read_base_model(self) -> str:
        if self.adapter_path is None:
            if self._base_override and str(self._base_override).strip():
                return str(self._base_override).strip()
            env_base = os.environ.get("ADVANCED_RAG_QWEN_BASE_MODEL", "").strip()
            if env_base:
                return env_base
            return "Qwen/Qwen2.5-1.5B-Instruct"
        if self._base_override:
            return str(self._base_override).strip()
        cfg_path = self.adapter_path / "adapter_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"Missing adapter_config.json under {self.adapter_path}. "
                "Unzip trained weights (e.g. models.zip) or set qwen_adapter_path."
            )
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path")
        if not base:
            raise ValueError(f"adapter_config.json has no base_model_name_or_path: {cfg_path}")
        return str(base)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        base_id = self._read_base_model()

        if self._dtype_name == "bfloat16":
            dtype = torch.bfloat16
        elif self._dtype_name == "float16":
            dtype = torch.float16
        elif self._dtype_name == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if _debug_timing():
            has_cuda = torch.cuda.is_available()
            if self.adapter_path is None:
                print(
                    f"[advanced_rag.qwen] Loading base model only {base_id!r} (cuda={has_cuda})…",
                    flush=True,
                )
            else:
                print(
                    f"[advanced_rag.qwen] Loading base + LoRA from {self.adapter_path} "
                    f"(cuda={has_cuda})…",
                    flush=True,
                )

        if self.adapter_path is not None and not self.adapter_path.is_dir():
            raise FileNotFoundError(
                f"Qwen adapter directory not found: {self.adapter_path}. "
                "Point qwen_adapter_path to a checkpoint folder with adapter weights."
            )

        if self.adapter_path is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.adapter_path),
                    trust_remote_code=True,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if self.adapter_path is None:
            model = base
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, str(self.adapter_path))
            if self.merge_adapters:
                try:
                    model = model.merge_and_unload()
                except Exception as exc:
                    if _debug_timing():
                        print(
                            f"[advanced_rag.qwen] merge_and_unload failed ({exc}); "
                            "keeping LoRA layers (slightly slower inference).",
                            flush=True,
                        )
                    _LOG.warning("merge_and_unload failed; using PeftModel: %s", exc)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer

        if _debug_timing():
            dt = time.perf_counter() - t0
            print(f"[advanced_rag.qwen] Load finished in {dt:.1f}s", flush=True)
        if not torch.cuda.is_available():
            _LOG.warning(
                "Qwen synthesizer running on CPU — generation will be slow. Use a CUDA GPU if possible."
            )

    def synthesize_json(self, system_prompt: str, user_content: str) -> Dict[str, Any]:
        """Return a dict with at least ``answer`` and optionally ``cited_clause_ids``."""
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Do not truncate here: HF defaults can clip long regulation context and hide the answer.
        inputs = self._tokenizer(text, return_tensors="pt", truncation=False)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kw: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.do_sample:
            gen_kw["temperature"] = 0.2
            gen_kw["top_p"] = 0.9

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kw)

        if _debug_timing():
            dt = time.perf_counter() - t0
            n_in = inputs["input_ids"].shape[1]
            n_out = out.shape[1] - n_in
            print(
                f"[advanced_rag.qwen] generate: {dt:.2f}s (prompt_tokens≈{n_in}, new_tokens≈{n_out})",
                flush=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, prompt_len:]
        decoded = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        data = _parse_json_object(decoded)
        ans = (data.get("answer") or "").strip()
        if ans:
            return data
        if _debug_timing() and decoded.strip():
            preview = decoded.strip().replace("\n", " ")[:240]
            print(f"[advanced_rag.qwen] empty answer after parse; gen preview: {preview!r}", flush=True)
        # Model sometimes returns plain text or truncated JSON; use non-empty decode as answer.
        plain = decoded.strip()
        if plain and len(plain) > 15 and "{" not in plain[:30]:
            return {"answer": plain, "cited_clause_ids": list(data.get("cited_clause_ids") or [])}
        if not data and plain and len(plain) > 15:
            return {"answer": plain[:8000], "cited_clause_ids": []}
        return data
