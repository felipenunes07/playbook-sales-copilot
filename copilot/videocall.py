"""Video call inside the copilot: rooms, WebRTC signaling, public tunnel, per-speaker transcription.

The rep's copilot page and the prospect's page (/sala/<room>) connect peer-to-peer over WebRTC; this server only
relays the signaling messages (HTTP post + poll, see `Signaling`). Speakers are known by channel, not by voice: the rep's microphone is always `rep`,
the remote stream is always `prospect`. The browser cuts each stream into speech segments and posts them to
/api/transcribe, which asks a small audio model on OpenRouter for the text.

Anyone holding the public URL reaches this server, so everything except the guest page and its signaling answers
only to this machine or to a signed-in rep (see `is_local` and copilot.auth).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

from . import constants as C

log = logging.getLogger("copilot.videocall")
ROOT = Path(__file__).resolve().parent.parent
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def is_local(headers: Any) -> bool:
    """True only for requests made on this machine. Tunnelled requests carry Cloudflare headers and a public Host."""
    if headers.get("cf-ray") or headers.get("cf-connecting-ip"):
        return False
    host = headers.get("host", "")
    host = host[: host.find("]") + 1] if host.startswith("[") else host.split(":", 1)[0]
    return host in LOCAL_HOSTS


# ---------------------------------------------------------------------------
# Rooms + signaling
#
# Hosted on serverless functions, the rep and the prospect may reach different instances, so nothing about a room
# lives in process memory: a room id is a signed token any instance can verify, and signaling messages wait in a
# per-recipient queue in Redis (Upstash REST) — or in memory when running locally. Browsers post and poll over HTTP.
# ---------------------------------------------------------------------------

ROOM_TTL_S = 6 * 3600
SIGNAL_TYPES = ("peer-joined", "offer", "answer", "ice", "bye")


def app_secret() -> bytes:
    """Signs room ids and login cookies. Must be the same on every instance: set APP_SECRET in production."""
    explicit = os.environ.get("APP_SECRET", "").strip()
    if explicit:
        return explicit.encode()
    password = os.environ.get("APP_PASSWORD", "").strip()
    if password:
        return hashlib.sha256(("copilot:" + password).encode()).digest()
    # No secret configured (e.g. OPEN_ACCESS deployments): derive one from the API keys, which every instance shares,
    # so a room created on one serverless instance is still valid on another. Only a random per-process secret
    # when nothing is configured (a local run without keys).
    shared = "".join(os.environ.get(k, "").strip() for k in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "JEV_API_KEY", "JEVMODEL_API_KEY", "BEATAPI_API_KEY", "BLOB_READ_WRITE_TOKEN"))
    if shared:
        return hashlib.sha256(("copilot-rooms:" + shared).encode()).digest()
    return _LOCAL_SECRET


_LOCAL_SECRET = secrets.token_bytes(32)


def sign(payload: str) -> str:
    return hmac.new(app_secret(), payload.encode(), hashlib.sha256).hexdigest()[:24]


def new_room(now: float | None = None) -> str:
    expires = int((now or time.time()) + ROOM_TTL_S)
    body = f"{secrets.token_urlsafe(9)}.{expires}"
    return f"{body}.{sign(body)}"


def room_valid(room: str, now: float | None = None) -> bool:
    parts = room.split(".")
    if len(parts) != 3 or not parts[1].isdigit():
        return False
    body = f"{parts[0]}.{parts[1]}"
    return hmac.compare_digest(parts[2], sign(body)) and int(parts[1]) > (now or time.time())


class MemoryStore:
    def __init__(self) -> None:
        self.queues: dict[str, list[str]] = {}

    async def push(self, key: str, item: str, ttl: int) -> None:
        self.queues.setdefault(key, []).append(item)

    async def drain(self, key: str) -> list[str]:
        return self.queues.pop(key, [])

    async def aclose(self) -> None:
        pass


class UpstashStore:
    """Redis over Upstash's REST API (what the Vercel Marketplace "Upstash Redis" integration provides)."""

    def __init__(self, url: str, token: str) -> None:
        self._client = httpx.AsyncClient(base_url=url.rstrip("/"), headers={"Authorization": f"Bearer {token}"}, timeout=10.0)

    async def push(self, key: str, item: str, ttl: int) -> None:
        r = await self._client.post("/pipeline", json=[["RPUSH", key, item], ["EXPIRE", key, str(ttl)]])
        r.raise_for_status()

    async def drain(self, key: str) -> list[str]:
        r = await self._client.post("/multi-exec", json=[["LRANGE", key, "0", "-1"], ["DEL", key]])
        r.raise_for_status()
        return list(r.json()[0].get("result") or [])

    async def aclose(self) -> None:
        await self._client.aclose()


def make_store() -> MemoryStore | UpstashStore:
    url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
    return UpstashStore(url, token) if url and token else MemoryStore()


class Signaling:
    def __init__(self, store: MemoryStore | UpstashStore) -> None:
        self.store = store

    @staticmethod
    def _queue(room: str, role: str) -> str:
        return f"copilot:sala:{room}:to:{role}"

    async def send(self, room: str, role: str, msg: dict[str, Any]) -> None:
        other = "guest" if role == "host" else "host"
        await self.store.push(self._queue(room, other), json.dumps(msg), ROOM_TTL_S)

    async def receive(self, room: str, role: str) -> list[dict[str, Any]]:
        return [json.loads(x) for x in await self.store.drain(self._queue(room, role))]


# ---------------------------------------------------------------------------
# Public link: Cloudflare quick tunnel (free, no account)
# ---------------------------------------------------------------------------

_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def find_cloudflared() -> str | None:
    candidates = [
        shutil.which("cloudflared"),
        str(ROOT / "tools" / "cloudflared.exe"),
        str(ROOT / "tools" / "cloudflared"),
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
        r"C:\Program Files\cloudflared\cloudflared.exe",
    ]
    return next((c for c in candidates if c and Path(c).is_file()), None)


class Tunnel:
    def __init__(self, port: int) -> None:
        self.port = port
        self.url: str | None = None
        self.error: str | None = None
        self.proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def ensure(self, timeout: float = 25.0) -> str | None:
        async with self._lock:
            if self.url and self.proc and self.proc.returncode is None:
                return self.url
            exe = find_cloudflared()
            if not exe:
                self.error = "cloudflared não encontrado"
                return None
            self.url, self.error = None, None
            self.proc = await asyncio.create_subprocess_exec(
                exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{self.port}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            deadline = time.monotonic() + timeout
            assert self.proc.stdout is not None
            while time.monotonic() < deadline:
                try:
                    line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=max(0.1, deadline - time.monotonic()))
                except asyncio.TimeoutError:
                    break
                if not line:
                    break
                m = _TUNNEL_URL.search(line.decode(errors="replace"))
                if m:
                    self.url = m.group(0)
                    asyncio.create_task(self._drain())
                    log.info("public tunnel up: %s", self.url)
                    return self.url
            self.error = "o túnel não respondeu a tempo"
            await self.stop()
            return None

    async def _drain(self) -> None:
        # keep reading cloudflared's output so its pipe never fills up and blocks it
        assert self.proc is not None and self.proc.stdout is not None
        while await self.proc.stdout.readline():
            pass

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()
        self.proc = None
        self.url = None


# ---------------------------------------------------------------------------
# Speech-to-text (OpenRouter audio model)
# ---------------------------------------------------------------------------

STT_PROMPT = (
    "Transcreva exatamente o que é falado neste áudio, em português do Brasil, palavra por palavra. "
    "Responda só com a transcrição, sem comentários. Se não houver fala compreensível, responda exatamente: (vazio)"
)


class Transcriber:
    def __init__(self, api_key: str, model: str = C.OPENROUTER_STT_MODEL, timeout: float = 20.0) -> None:
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=C.OPENROUTER_BASE_URL, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
        )
        self.total_cost_usd = 0.0
        self.total_seconds = 0.0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __call__(self, wav: bytes) -> dict[str, Any]:
        seconds = max(0.0, (len(wav) - 44) / 32000)  # 16 kHz, 16-bit mono
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": STT_PROMPT},
                {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
            ]}],
        }
        t0 = time.perf_counter()
        try:
            r = await self._client.post("/api/v1/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("transcription failed: %s", exc)
            return {"text": "", "error": f"{type(exc).__name__}: {exc}"}
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        text = clean_transcript(text)
        cost = float((data.get("usage") or {}).get("cost") or 0.0)
        self.total_cost_usd += cost
        self.total_seconds += seconds
        return {"text": text, "latency_ms": round((time.perf_counter() - t0) * 1000), "seconds": round(seconds, 1),
                "cost_usd": round(cost, 6), "total_cost_usd": round(self.total_cost_usd, 5), "model": self.model}


def clean_transcript(text: str) -> str:
    text = text.strip().strip('"“”').strip()
    if not text or "(vazio)" in text.lower() or len(re.sub(r"[\W_]+", "", text)) < 2:
        return ""
    return text


def make_transcriber() -> Transcriber | None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return Transcriber(key, model=os.environ.get("OPENROUTER_STT_MODEL", C.OPENROUTER_STT_MODEL)) if key else None
