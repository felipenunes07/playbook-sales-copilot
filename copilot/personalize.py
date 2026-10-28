"""Tailored phrasing: a small, fast generative LLM rewrites the chosen playbook line for this conversation.

Division of labour (see README "Tailored phrasing"):
  Jev   decides which move, gates it on confidence, and afterwards judges the candidate sentence.
  LLM   writes exactly one sentence, only when the move changed or a new fact locked in.
  code  decides when to call, validates the output, discards stale or invented lines, keeps the generic line as fallback.

The LLM call never blocks the per-utterance loop: the server fires it as a background task and pushes a
`phrasing` message to the browser when (and if) a usable line comes back.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

import httpx

from . import constants as C
from .jev_client import Judge

log = logging.getLogger("copilot.personalize")


def resolve_llm_key() -> str | None:
    for name in C.LLM_API_KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# Trigger rule (pure, unit-tested)
# ---------------------------------------------------------------------------


@dataclass
class PersonalizeState:
    last_move_id: str | None = None
    last_index: int = -10_000  # utterance index of the last rewrite
    last_facts: frozenset[str] = frozenset()


def should_personalize(coaching: Mapping[str, Any], facts: Mapping[str, Any], index: int, st: PersonalizeState) -> str | None:
    """Return a reason string when a rewrite is warranted for this utterance, else None."""
    if not coaching or coaching.get("gated") or not coaching.get("move_id"):
        return None
    move = coaching["move_id"]
    if move in C.PERSONALIZE_SKIP_MOVES or move not in C.PLAYBOOK_BY_ID:
        return None
    if index - st.last_index < C.PERSONALIZE_COOLDOWN_UTTERANCES:
        return None
    facts_now = frozenset(facts)
    if move != st.last_move_id:
        return "move changed"
    if facts_now - st.last_facts:
        return "new fact"
    if index - st.last_index >= C.PERSONALIZE_REFRESH_UTTERANCES:
        return "refresh"
    return None


# ---------------------------------------------------------------------------
# Prompt + output validation (pure)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are whispering one sentence into a sales rep's ear during a live call. "
    "Write exactly ONE sentence the rep can say next, in the rep's own voice, natural and spoken. "
    "It must carry out the coaching move you are given. Use only details that appear in the transcript: "
    "never invent numbers, prices, names, dates, features, or promises. Prefer a question over a statement. "
    "Write in the language of the transcript (Brazilian Portuguese if the call is in Portuguese). "
    "Output the sentence only: no quotes, no preamble, no explanation."
)


def max_words_for(move_id: str) -> int:
    return C.PERSONALIZE_MAX_WORDS_BY_MOVE.get(move_id, C.PERSONALIZE_MAX_WORDS)


def build_prompt(move: Mapping[str, Any], transcript: list[Mapping[str, Any]], facts: list[str], objection: Mapping[str, Any] | None) -> str:
    lines = [f"{u['speaker']}: {u['text']}" for u in transcript]
    parts = [
        f"Coaching move: {move['title']}",
        f"What it means: {move['what']}",
        "Generic example lines for this move (match their tone, but make yours specific to this call):",
        *[f"- {p}" for p in move["phrasings"]],
        "",
        f"Facts already established on this call: {', '.join(facts) if facts else 'none yet'}",
    ]
    if objection and objection.get("open"):
        parts.append(f"Open objection: {objection.get('type') or 'unknown'}")
    parts += ["", "Recent transcript (rep = you):", *lines, "", f"Your one sentence (at most {max_words_for(move['id'])} words):"]
    return "\n".join(parts)


_QUOTE_CHARS = "\"'“”‘’«»"


def clean_line(text: str, max_words: int = C.PERSONALIZE_MAX_WORDS) -> str | None:
    """Normalize the LLM output to one plain sentence; None if it does not look like one."""
    if not text:
        return None
    line = text.strip().splitlines()[0].strip()
    line = line.strip(_QUOTE_CHARS + " ").strip()
    line = re.sub(r"^(rep|you|say|sentence)\s*:\s*", "", line, flags=re.IGNORECASE).strip(_QUOTE_CHARS + " ")
    if not line or len(line) < 8:
        return None
    words = line.split()
    if len(words) > max_words + 6:
        return None
    if line[-1] not in ".?!":
        line += "?" if line.lower().startswith(("what", "how", "who", "when", "where", "which", "would", "could", "can", "is", "are", "do", "does", "did", "if")) else "."
    return line


# ---------------------------------------------------------------------------
# LLM client (Anthropic Messages API over plain HTTP; no SDK)
# ---------------------------------------------------------------------------


@dataclass
class PhrasingResult:
    text: str | None
    model: str = C.LLM_MODEL
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    verification: dict[str, float] = field(default_factory=dict)
    rejected_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


Personalizer = Callable[[Mapping[str, Any], list[Mapping[str, Any]], list[str], Mapping[str, Any] | None], Awaitable[PhrasingResult]]


def llm_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1e6 * C.LLM_PRICE_PER_M_INPUT_TOKENS_USD + output_tokens / 1e6 * C.LLM_PRICE_PER_M_OUTPUT_TOKENS_USD


class AnthropicPersonalizer:
    """Async callable: (move, transcript, facts, objection) -> PhrasingResult. Reuses one HTTP client."""

    def __init__(self, api_key: str | None = None, model: str = C.LLM_MODEL, timeout: float = C.LLM_TIMEOUT_S):
        key = api_key or resolve_llm_key()
        if not key:
            raise RuntimeError("No LLM key: set ANTHROPIC_API_KEY to enable tailored phrasing.")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url="",
            timeout=timeout,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __call__(self, move: Mapping[str, Any], transcript: list[Mapping[str, Any]], facts: list[str], objection: Mapping[str, Any] | None) -> PhrasingResult:
        body = {
            "model": self.model,
            "max_tokens": C.LLM_MAX_OUTPUT_TOKENS,
            "temperature": 0.4,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_prompt(move, transcript, facts, objection)}],
        }
        t0 = time.perf_counter()
        try:
            resp = await self._client.post(C.LLM_ENDPOINT, json=body)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - never let the LLM path break the call loop
            latency = (time.perf_counter() - t0) * 1000
            log.warning("llm request failed after %.0f ms: %s", latency, type(exc).__name__)
            return PhrasingResult(text=None, model=self.model, latency_ms=latency, error=f"{type(exc).__name__}: {exc}")
        latency = (time.perf_counter() - t0) * 1000
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        in_tok, out_tok = int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)
        truncated = data.get("stop_reason") == "max_tokens"
        line = None if truncated else clean_line(text, max_words_for(move["id"]))
        log.info("llm model=%s latency=%.0fms in=%d out=%d cost=$%.6f ok=%s", data.get("model", self.model), latency, in_tok, out_tok, llm_cost(in_tok, out_tok), line is not None)
        return PhrasingResult(
            text=line,
            model=str(data.get("model", self.model)),
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=llm_cost(in_tok, out_tok),
            rejected_reason=None if line else ("truncated" if truncated else "unusable output"),
            extra={"raw": text[:400], "stop_reason": data.get("stop_reason")},
        )


class OpenRouterPersonalizer:
    """Same interface as AnthropicPersonalizer, over OpenRouter's OpenAI-compatible chat completions."""

    def __init__(self, api_key: str, model: str = C.OPENROUTER_LLM_MODEL, timeout: float = C.LLM_TIMEOUT_S):
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=C.OPENROUTER_BASE_URL,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def warm(self) -> None:
        """Open the TLS connection ahead of the first rewrite."""
        try:
            await self._client.head("/")
        except Exception:  # noqa: BLE001
            pass

    async def __call__(self, move: Mapping[str, Any], transcript: list[Mapping[str, Any]], facts: list[str], objection: Mapping[str, Any] | None) -> PhrasingResult:
        body = {
            "model": self.model,
            "max_tokens": C.LLM_MAX_OUTPUT_TOKENS,
            "temperature": 0.4,
            "reasoning": {"enabled": False},  # one spoken sentence needs no thinking; reasoning models otherwise run out of tokens
            "provider": {"sort": "latency"},  # the line has to land while the moment is still on
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(move, transcript, facts, objection)},
            ],
        }
        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/api/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - never let the LLM path break the call loop
            latency = (time.perf_counter() - t0) * 1000
            log.warning("llm request failed after %.0f ms: %s", latency, type(exc).__name__)
            return PhrasingResult(text=None, model=self.model, latency_ms=latency, error=f"{type(exc).__name__}: {exc}")
        latency = (time.perf_counter() - t0) * 1000
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage", {})
        in_tok, out_tok = int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
        truncated = choice.get("finish_reason") == "length"
        line = None if truncated else clean_line(text, max_words_for(move["id"]))
        cost = float(usage.get("cost") or 0.0)
        log.info("llm model=%s latency=%.0fms in=%d out=%d cost=$%.6f ok=%s", data.get("model", self.model), latency, in_tok, out_tok, cost, line is not None)
        return PhrasingResult(
            text=line,
            model=self.model,
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            rejected_reason=None if line else ("truncated" if truncated else "unusable output"),
            extra={"raw": text[:400], "stop_reason": choice.get("finish_reason")},
        )


def make_personalizer() -> "AnthropicPersonalizer | OpenRouterPersonalizer | None":
    """ANTHROPIC_API_KEY → Anthropic directly; else OPENROUTER_API_KEY → OpenRouter; else off."""
    if resolve_llm_key():
        return AnthropicPersonalizer()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return OpenRouterPersonalizer(key, model=os.environ.get("OPENROUTER_LLM_MODEL", C.OPENROUTER_LLM_MODEL))
    return None


# ---------------------------------------------------------------------------
# Jev as the guardrail on the generated line
# ---------------------------------------------------------------------------


async def verify_with_jev(judge: Judge, line: str, move: Mapping[str, Any], transcript: list[Mapping[str, Any]], facts: list[str]) -> tuple[bool, dict[str, float], str | None]:
    """Ask Jev whether the candidate line invents a fact / fits the move. Returns (ok, probabilities, reason)."""
    state = {
        "candidate_line": line,
        "move": {"title": move["title"], "what": move["what"]},
        "recent_transcript": [{"speaker": u["speaker"], "text": u["text"]} for u in transcript],
        "known_facts": list(facts),
    }
    result = await judge(state, C.VERIFY_QUESTIONS)
    if result.error or not result.answers:
        return True, {}, None  # verification unavailable: do not block on it
    probs = {k: float(v.get("noul", 0.0)) for k, v in result.answers.items() if v.get("type") == "noul"}
    if probs.get("invents_fact", 0.0) >= C.PERSONALIZE_INVENTED_FACT_THRESHOLD:
        return False, probs, "invents a fact"
    if probs.get("on_move", 1.0) < 0.5:
        return False, probs, "off move"
    return True, probs, None
