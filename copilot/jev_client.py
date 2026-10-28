"""Thin async wrapper around the TypeSafe SDK: one speculative fan-out request per utterance.

Returns plain dicts so the engine (and tests, via a mock) never depend on SDK types.
Never logs the API key. Logs model, usage, latency and request id per call.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from . import constants as C

log = logging.getLogger("copilot.jev")


def resolve_api_key() -> str | None:
    """TYPESAFE_API_KEY with fallback to JEV_API_KEY (never printed)."""
    for name in C.API_KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


@dataclass
class JudgeResult:
    answers: dict[str, dict[str, Any]]
    model: str = C.MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    request_id: str | None = None
    cost_usd: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# The engine only needs something with this signature; tests inject a mock.
Judge = Callable[[Any, Mapping[str, Any]], Awaitable[JudgeResult]]


def normalize_answers(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """SDK answer objects -> plain dicts. Score probabilities keyed by int level."""
    out: dict[str, dict[str, Any]] = {}
    for qid, a in raw.items():
        if isinstance(a, Mapping):
            kind = a.get("type")
        elif hasattr(a, "noul"):
            kind = "noul"
        elif hasattr(a, "choice"):
            kind = "choice"
        elif hasattr(a, "score"):
            kind = "score"
        else:
            kind = None
        if kind == "noul":
            out[qid] = {"type": "noul", "noul": float(a.noul if hasattr(a, "noul") else a["noul"])}
        elif kind == "choice":
            probs = a.probabilities if hasattr(a, "probabilities") else a["probabilities"]
            out[qid] = {
                "type": "choice",
                "choice": str(a.choice if hasattr(a, "choice") else a["choice"]),
                "confidence": float(a.confidence if hasattr(a, "confidence") else a["confidence"]),
                "probabilities": {str(k): float(v) for k, v in probs.items()},
            }
        elif kind == "score":
            probs = a.probabilities if hasattr(a, "probabilities") else a["probabilities"]
            probs_int = {int(k): float(v) for k, v in probs.items()}
            out[qid] = {
                "type": "score",
                "score": float(a.score if hasattr(a, "score") else a["score"]),
                "confidence": float(a.confidence if hasattr(a, "confidence") else a["confidence"]),
                "probabilities": probs_int,
                "levels": max(probs_int) + 1 if probs_int else 0,
            }
    return out


def cost_for_tokens(input_tokens: int) -> float:
    return input_tokens / 1_000_000 * C.PRICE_PER_M_INPUT_TOKENS_USD


class HostedJevJudge:
    """Same interface as JevJudge, against a hosted SystemOne-compatible HTTP API (jevmodel.org, BeatAPI).

    Question names are renamed to safe aliases (`[A-Za-z0-9_]`) on the wire. Providers that cap the number of
    questions per request get the set split into chunks sent concurrently; answers are merged back under the
    original names. Usage is summed over the chunks.
    """

    def __init__(self, api_key: str, base_url: str, model: str, max_questions: int | None = None,
                 max_criteria_chars: int | None = None, path: str = "/v1/systemone", timeout: float = C.HOSTED_REQUEST_TIMEOUT_S):
        import asyncio

        import httpx

        self.model = model
        self._max_questions = max_questions
        self._path = path
        self._max_criteria_chars = max_criteria_chars
        self._sem = asyncio.Semaphore(C.HOSTED_MAX_CONCURRENCY)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def warm(self) -> None:
        """Open the TLS connection ahead of the first utterance, so it doesn't pay the handshake."""
        try:
            await self._client.head("/")
        except Exception:  # noqa: BLE001
            pass

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        import hashlib
        import json
        from pathlib import Path

        key = hashlib.sha256(json.dumps([str(self._client.base_url), body], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = Path(C.JEV_CACHE_DIR) / f"{key}.json"
        if path.exists():
            resp = json.loads(path.read_text(encoding="utf-8"))
            return {**resp, "usage": {"input_tokens": 0, "output_tokens": 0}, "cached": True}
        resp = await self._post_http(body)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:  # a read-only or full disk only costs the cache, never the answer
            log.warning("jev cache write failed: %s", exc)
        return resp

    async def _post_http(self, body: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        delay = 0.3
        async with self._sem:
            for attempt in range(4):
                r = await self._client.post(self._path, json=body)
                if r.status_code in (429, 502) and attempt < 3:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if r.status_code != 200:
                    try:
                        err = r.json().get("error", {})
                        msg = f"{err.get('type') or err.get('code') or 'error'}: {err.get('message', '')}"
                    except Exception:  # noqa: BLE001
                        msg = r.text[:200]
                    raise RuntimeError(f"HTTP {r.status_code} {msg}")
                return r.json()
        raise RuntimeError("unreachable")

    async def __call__(self, state: Any, questions: Mapping[str, Any]) -> JudgeResult:
        import asyncio

        t0 = time.perf_counter()
        names = list(questions)
        alias = {f"q{i}": name for i, name in enumerate(names)}
        items = [(a, fit_criteria(questions[n], self._max_criteria_chars) if self._max_criteria_chars else questions[n]) for a, n in alias.items()]
        size = self._max_questions or len(items)
        chunks = [dict(items[i:i + size]) for i in range(0, len(items), size)]
        try:
            resps = await asyncio.gather(*(self._post({"model": self.model, "state": state, "questions": ch}) for ch in chunks))
        except Exception as exc:  # noqa: BLE001 - surface every failure to the UI, keep the loop alive
            latency = (time.perf_counter() - t0) * 1000
            log.warning("hosted jev request failed after %.0f ms: %s", latency, exc)
            return JudgeResult(answers={}, latency_ms=latency, error=str(exc))
        latency = (time.perf_counter() - t0) * 1000
        raw: dict[str, Any] = {}
        in_tok = out_tok = 0
        model = self.model
        for resp in resps:
            for a, ans in resp.get("answers", {}).items():
                raw[alias.get(a, a)] = ans
            usage = resp.get("usage", {})
            in_tok += int(usage.get("input_tokens", 0) or 0)
            out_tok += int(usage.get("output_tokens", 0) or 0)
            model = resp.get("model", model)
        result = JudgeResult(
            answers=normalize_answers(raw),
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency,
            cost_usd=cost_for_tokens(in_tok),
            extra={"requests": len(chunks), "cached": sum(1 for r in resps if r.get("cached"))},
        )
        log.info("hosted jev model=%s latency=%.0fms requests=%d in_tokens=%d cost=$%.6f", model, latency, len(chunks), in_tok, result.cost_usd)
        return result


def fit_criteria(q: Mapping[str, Any], limit: int) -> dict[str, Any]:
    """jevmodel.org caps serialized criteria per question. Over the cap: keep only each option's `what`, trimmed evenly."""
    import json

    q = dict(q)
    crit = q.get("criteria")
    if not isinstance(crit, Mapping) or len(json.dumps(crit, ensure_ascii=False)) <= limit:
        return q
    flat = {k: (v.get("what", "") if isinstance(v, Mapping) else str(v)) for k, v in crit.items()}
    overhead = sum(len(json.dumps(k)) + 4 for k in flat) + 2
    per = max(20, (limit - overhead) // len(flat) - 2)
    q["criteria"] = {k: v if len(v) <= per else v[: per - 1].rsplit(" ", 1)[0] + "…" for k, v in flat.items()}
    return q


def make_judge(api_key: str | None = None, model: str = C.MODEL) -> "JevJudge | HostedJevJudge":
    """Pick the backend: OPENROUTER_API_KEY → OpenRouter; BEATAPI_API_KEY → BeatAPI;
    JEVMODEL_API_KEY or an `sk-…` key → jevmodel.org; else the TypeSafe SDK."""
    openrouter = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key and openrouter:
        return HostedJevJudge(openrouter, C.OPENROUTER_BASE_URL, path=C.OPENROUTER_DECISIONS_PATH,
                              model=os.environ.get("OPENROUTER_JEV_MODEL", C.OPENROUTER_JEV_MODEL))
    beat = os.environ.get("BEATAPI_API_KEY", "").strip()
    if not api_key and beat:
        return HostedJevJudge(beat, C.BEATAPI_BASE_URL, model=os.environ.get("BEATAPI_MODEL", C.BEATAPI_MODEL))
    key = api_key or os.environ.get("JEVMODEL_API_KEY", "").strip() or resolve_api_key()
    if key and key.startswith("sk-"):
        return HostedJevJudge(key, C.JEVMODEL_BASE_URL, model=model,
                              max_questions=C.JEVMODEL_MAX_QUESTIONS_PER_REQUEST, max_criteria_chars=C.JEVMODEL_MAX_CRITERIA_CHARS)
    return JevJudge(key, model=model)


class JevJudge:
    """Async callable: (state, questions) -> JudgeResult. Reuses one HTTP client."""

    def __init__(self, api_key: str | None = None, model: str = C.MODEL, timeout: float = C.REQUEST_TIMEOUT_S):
        key = api_key or resolve_api_key()
        if not key:
            raise RuntimeError("No API key: set TYPESAFE_API_KEY (or JEV_API_KEY). See .env.example / run.sh.")
        self.model = model
        self._client = AsyncTypeSafeClient(
            api_key=key,
            model=model,
            timeout=timeout,
            retry=RetryPolicy(max_retries=2, backoff_initial=0.2, backoff_max=1.0, timeout=timeout * 2),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __call__(self, state: Any, questions: Mapping[str, Any]) -> JudgeResult:
        t0 = time.perf_counter()
        try:
            resp = await self._client.system_one(state, dict(questions))
        except Exception as exc:  # noqa: BLE001 - surface every failure to the UI, keep the loop alive
            latency = (time.perf_counter() - t0) * 1000
            log.warning("jev request failed after %.0f ms: %s", latency, type(exc).__name__)
            return JudgeResult(answers={}, latency_ms=latency, error=f"{type(exc).__name__}: {exc}")
        latency = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        result = JudgeResult(
            answers=normalize_answers(resp.answers),
            model=resp.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency,
            request_id=resp.request_id,
            cost_usd=cost_for_tokens(in_tok),
        )
        log.info(
            "jev model=%s latency=%.0fms in_tokens=%d out_tokens=%d cost=$%.6f request_id=%s",
            resp.model, latency, in_tok, out_tok, result.cost_usd, resp.request_id,
        )
        return result
