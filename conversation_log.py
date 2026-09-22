"""Append metadata-only conversation telemetry for local diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Optional
from uuid import uuid4


DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".workshop", "conversations.jsonl")


def _text_metadata(value: str) -> dict[str, Any]:
    value = value or ""
    return {
        "length": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def _step_record(number: int, turn: dict[str, Any]) -> dict[str, Any]:
    usage = turn.get("usage") or (0, 0)
    return {
        "step": number,
        "elapsed": round(turn.get("elapsed") or 0, 3),
        "input_tokens": usage[0],
        "output_tokens": usage[1],
        "stop_reason": turn.get("stop_reason"),
        "tool_names": [call.get("name") for call in turn.get("tool_calls", [])],
    }


def append_conversation_log(
    *,
    source: str,
    pnr: str,
    last_name: str,
    prompt: str,
    response: str = "",
    error: Optional[str] = None,
    tracer: Any = None,
    path: str = DEFAULT_PATH,
) -> dict[str, Any]:
    """Append one conversation record without persisting its text bodies."""
    summary = tracer.summary() if tracer else {}
    turns = list(getattr(tracer, "turns", []) or [])
    record = {
        "run_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "pnr": pnr,
        "last_name_present": bool(last_name),
        "prompt": _text_metadata(prompt),
        "response": _text_metadata(response),
        "status": "error" if error else "success",
        "error": error,
        "summary": {
            "turns": summary.get("turns", len(turns)),
            "tool_calls": summary.get("tool_calls", 0),
            "tokens": summary.get("tokens", {}),
            "cache_hit_ratio": summary.get("cache_hit_ratio"),
            "elapsed": summary.get("elapsed"),
            "stop_reason": turns[-1].get("stop_reason") if turns else None,
        },
        "steps": [_step_record(index, turn) for index, turn in enumerate(turns, 1)],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return record
