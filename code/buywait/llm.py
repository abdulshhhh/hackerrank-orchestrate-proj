from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent


def load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(CODE_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)


load_environment()


@dataclass
class UsageBucket:
    calls: int = 0
    failed_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelUsage:
    provider: str = "Anthropic"
    model: str = "none"
    by_call_type: dict[str, UsageBucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, call_type: str, *, failed=False, cache_hit=False, input_tokens=0, output_tokens=0) -> None:
        with self._lock:
            bucket = self.by_call_type.setdefault(call_type, UsageBucket())
            bucket.calls += int(not cache_hit)
            bucket.failed_calls += int(failed)
            bucket.cache_hits += int(cache_hit)
            bucket.input_tokens += int(input_tokens or 0)
            bucket.output_tokens += int(output_tokens or 0)

    @property
    def calls(self) -> int:
        return sum(item.calls for item in self.by_call_type.values())

    @property
    def failed_calls(self) -> int:
        return sum(item.failed_calls for item in self.by_call_type.values())

    @property
    def cache_hits(self) -> int:
        return sum(item.cache_hits for item in self.by_call_type.values())

    @property
    def input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.by_call_type.values())

    @property
    def output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.by_call_type.values())


class AnthropicRuntime:
    """Thread-safe forced-tool caller with bounded retries and a SHA-256 disk cache."""

    _warning_lock = threading.Lock()
    _warned_missing_key = False

    def __init__(self, model_env: str = "BUYWAIT_MODEL", default_model: str = "claude-sonnet-5") -> None:
        load_environment()
        requested = os.getenv("BUYWAIT_USE_LLM", "1") == "1"
        self.enabled = requested and bool(os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv(model_env, os.getenv("BUYWAIT_MODEL", default_model))
        self.usage = ModelUsage(model=self.model if self.enabled else "none")
        self.cache_dir = CODE_ROOT / ".cache" / "anthropic"
        self.timeout_seconds = max(1.0, float(os.getenv("BUYWAIT_LLM_TIMEOUT_SECONDS", "20")))
        self.rebuild_cache = os.getenv("BUYWAIT_REBUILD_CACHE", "0") == "1"
        self._local = threading.local()
        if requested and not os.getenv("ANTHROPIC_API_KEY"):
            with self._warning_lock:
                if not self.__class__._warned_missing_key:
                    print("warning: ANTHROPIC_API_KEY is missing; using deterministic fallbacks", file=sys.stderr)
                    self.__class__._warned_missing_key = True

    def _client(self):
        if not hasattr(self._local, "client"):
            from anthropic import Anthropic

            self._local.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0)
        return self._local.client

    def forced_tool(
        self,
        call_type: str,
        system: str,
        payload: Any,
        tool: dict[str, Any],
        *,
        max_tokens: int = 1024,
        deadline: float | None = None,
        content_prefix: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        material = {
            "model": self.model,
            "system": system,
            "payload": payload,
            "tool": tool,
            "content_prefix": content_prefix or [],
        }
        digest = hashlib.sha256(
            json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{digest}.json"
        if cache_path.is_file() and not self.rebuild_cache:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                self.usage.record(call_type, cache_hit=True)
                return cached
            except (OSError, json.JSONDecodeError):
                pass

        request_deadline = deadline or time.monotonic() + self.timeout_seconds
        retryable = self._retryable_exceptions()
        for attempt in range(3):
            remaining = request_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                content = list(content_prefix or [])
                content.append({"type": "text", "text": json.dumps(payload, ensure_ascii=False, sort_keys=True)})
                response = self._client().messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": content}],
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    # No temperature is sent: it is deprecated for this model and passing it
                    # returns HTTP 400. Run-to-run reproducibility therefore comes from the
                    # SHA-256 disk cache below rather than from sampling control, so a
                    # published run replays exactly while a cache rebuild may vary.
                    timeout=max(1.0, remaining),
                )
                result = next(
                    (block.input for block in response.content if block.type == "tool_use" and block.name == tool["name"]),
                    None,
                )
                if not isinstance(result, dict):
                    raise ValueError(f"forced tool {tool['name']} returned no tool_use input")
                self.usage.record(
                    call_type,
                    input_tokens=getattr(response.usage, "input_tokens", 0),
                    output_tokens=getattr(response.usage, "output_tokens", 0),
                )
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
                temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
                os.replace(temporary, cache_path)
                return result
            except retryable:
                if attempt < 2:
                    time.sleep(min(2**attempt, max(0.0, request_deadline - time.monotonic())))
                    continue
                self.usage.record(call_type, failed=True)
                return None
            except Exception:
                self.usage.record(call_type, failed=True)
                return None
        self.usage.record(call_type, failed=True)
        return None

    @staticmethod
    def _retryable_exceptions():
        try:
            from anthropic import APIConnectionError, InternalServerError, OverloadedError, RateLimitError

            return (APIConnectionError, InternalServerError, OverloadedError, RateLimitError)
        except ImportError:
            return (TimeoutError,)
