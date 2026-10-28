"""FastAPI app: static UI + one WebSocket per browser tab (replay or live utterances).

    uv run uvicorn copilot.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from . import constants as C
from .engine import CallSession
from .jev_client import JevJudge, make_judge, resolve_api_key
from .personalize import PersonalizeState, make_personalizer, Personalizer, resolve_llm_key, should_personalize, verify_with_jev
from .replay import list_calls, load_call, load_remote_calls
from .videocall import SIGNAL_TYPES, Signaling, Tunnel, find_cloudflared, is_local, make_store, make_transcriber, new_room, room_valid
from . import auth

log = logging.getLogger("copilot.server")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
NO_CACHE = {"Cache-Control": "no-store"}  # the UI changes often; never serve a stale page from the browser cache
MEDIA_DIR = Path(__file__).resolve().parent.parent / "data" / "media"

MAX_REAL_GAP_S = 6.0  # never wait longer than this between utterances, even at 1x
MIN_GAP_S = 0.15


def ensure_state(app: FastAPI) -> None:
    """Create the shared clients once per process (also when a host does not run the ASGI lifespan)."""
    if getattr(app.state, "ready", False):
        return
    app.state.judge = make_judge() if resolve_api_key() else None
    if app.state.judge is None:
        log.error("No API key found (TYPESAFE_API_KEY / JEV_API_KEY / OPENROUTER_API_KEY). The UI will load but Jev calls will fail.")
    app.state.personalizer = make_personalizer()
    if app.state.personalizer is None:
        log.info("No LLM key: tailored phrasing disabled, generic playbook lines only.")
    app.state.transcriber = make_transcriber()
    app.state.signaling = Signaling(make_store())
    app.state.tunnel = Tunnel(int(os.environ.get("PORT", "8000")))
    if os.environ.get("PRIVATE_CALLS_URLS"):
        log.info("private calls loaded: %s", load_remote_calls(os.environ["PRIVATE_CALLS_URLS"]))
    app.state.ready = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_state(app)
    yield
    await app.state.tunnel.stop()
    for obj in (app.state.judge, app.state.personalizer, app.state.transcriber, app.state.signaling.store):
        if obj is not None and hasattr(obj, "aclose"):
            await obj.aclose()
    app.state.ready = False  # the clients above are closed; build new ones if the app starts again


app = FastAPI(title="Sales Copilot (Jev)", lifespan=lifespan)
PUBLIC_PATHS = ("/sala/", "/login")


@app.middleware("http")
async def require_login(request: Request, call_next):
    """The prospect's call page is public; everything else needs this machine or a signed-in rep."""
    ensure_state(request.app)
    path = request.url.path
    if path.startswith(PUBLIC_PATHS) or auth.authorized(request.headers, request.cookies):
        return await call_next(request)
    if request.method == "GET" and not path.startswith(("/api/", "/media/")):
        return RedirectResponse("/login", status_code=303)
    return JSONResponse({"error": "login required"}, status_code=401)


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/login")
async def login_page(error: str = "") -> HTMLResponse:
    msg = ""
    if error == "1":
        msg = '<div class="err">Senha incorreta.</div>'
    elif not auth.password():
        msg = '<div class="err">O acesso externo está desligado: defina APP_PASSWORD no servidor.</div>'
    return HTMLResponse(auth.LOGIN_PAGE.replace("{error}", msg), headers=NO_CACHE)


@app.post("/login")
async def login(request: Request) -> RedirectResponse:
    form = parse_qs((await request.body()).decode(errors="replace"))
    if not auth.password_ok((form.get("password") or [""])[0]):
        return RedirectResponse("/login?error=1", status_code=303)
    resp = RedirectResponse("/", status_code=303)
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp.set_cookie(auth.COOKIE, auth.make_cookie(), max_age=auth.SESSION_DAYS * 86400, httponly=True, secure=secure, samesite="lax")
    return resp


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "model": C.MODEL, "has_api_key": bool(resolve_api_key()), "questions": len(C.QUESTIONS), "llm": app.state.personalizer.model if getattr(app.state, "personalizer", None) else None}


# ---------------------------------------------------------------------------
# Video call: room + link, guest page, transcription, signaling (HTTP post + poll)
# ---------------------------------------------------------------------------
def ice_servers() -> list[dict[str, Any]]:
    """STUN by default; set ICE_SERVERS (JSON list, e.g. a TURN service) for networks that block peer-to-peer."""
    try:
        extra = json.loads(os.environ.get("ICE_SERVERS", "") or "[]")
    except json.JSONDecodeError:
        extra = []
    return [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}, *extra]


@app.post("/api/sala")
async def create_room(request: Request) -> dict[str, Any]:
    room = new_room()
    if is_local(request.headers):
        public = await app.state.tunnel.ensure()  # on this machine, a Cloudflare quick tunnel makes the link reachable
        guest_url, error = (f"{public}/sala/{room}" if public else None), app.state.tunnel.error
    else:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        guest_url, error = f"{proto}://{request.headers.get('host')}/sala/{room}", None
    return {
        "room": room,
        "guest_url": guest_url,
        "local_guest_url": f"http://localhost:{app.state.tunnel.port}/sala/{room}",
        "tunnel_error": error,
        "has_cloudflared": find_cloudflared() is not None,
        "stt": app.state.transcriber.model if app.state.transcriber else None,
        "ice": ice_servers(),
    }


@app.get("/sala/{room}")
async def guest_page(room: str) -> Any:
    if not room_valid(room):
        return JSONResponse({"error": "sala não encontrada"}, status_code=404)
    return FileResponse(STATIC_DIR / "sala.html", headers=NO_CACHE)


@app.get("/sala/{room}/ice")
async def guest_ice(room: str) -> Any:
    if not room_valid(room):
        return JSONResponse({"error": "sala não encontrada"}, status_code=404)
    return ice_servers()


def _signal_role(request: Request, room: str) -> str | None:
    """`host` only for the rep (this machine or signed in); anyone with the link may be the guest."""
    if not room_valid(room):
        return None
    role = "host" if request.query_params.get("role") == "host" else "guest"
    if role == "host" and not auth.authorized(request.headers, request.cookies):
        return None
    return role


@app.post("/sala/{room}/send")
async def signal_send(room: str, request: Request) -> Any:
    role = _signal_role(request, room)
    if role is None:
        return JSONResponse({"error": "sala não encontrada"}, status_code=404)
    try:
        msg = await request.json()
    except ValueError:
        return JSONResponse({"error": "json inválido"}, status_code=400)
    if not isinstance(msg, dict) or msg.get("type") not in SIGNAL_TYPES:
        return JSONResponse({"error": "tipo de mensagem não permitido"}, status_code=400)
    if msg["type"] == "peer-joined":
        msg = {"type": "peer-joined", "role": role}
    await app.state.signaling.send(room, role, msg)
    return {"ok": True}


@app.get("/sala/{room}/poll")
async def signal_poll(room: str, request: Request) -> Any:
    role = _signal_role(request, room)
    if role is None:
        return JSONResponse({"error": "sala não encontrada"}, status_code=404)
    return await app.state.signaling.receive(room, role)


@app.post("/api/transcribe")
async def transcribe(request: Request) -> dict[str, Any]:
    if app.state.transcriber is None:
        return {"text": "", "error": "defina OPENROUTER_API_KEY para transcrever a chamada"}
    wav = await request.body()
    if len(wav) < 44 + 16000:  # under ~0.5 s of audio
        return {"text": ""}
    return await app.state.transcriber(wav)


@app.get("/media/{name}")
async def media(name: str) -> Any:
    """Local call recordings (data/media/, git-ignored); FileResponse serves byte ranges so the player can seek."""
    path = (MEDIA_DIR / name).resolve()
    if path.parent != MEDIA_DIR.resolve() or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/calls")
async def calls() -> list[dict[str, Any]]:
    return list_calls()


@app.get("/api/calls/{call_id}")
async def call(call_id: str) -> JSONResponse:
    try:
        return JSONResponse(load_call(call_id))
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.get("/api/playbook")
async def playbook() -> list[dict[str, Any]]:
    return C.PLAYBOOK


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "model": C.MODEL,
        "weights": [{"signal": k, **v} for k, v in C.SIGNAL_WEIGHTS.items()],
        "stage_prior": C.STAGE_PRIOR,
        "ema_alpha": C.EMA_ALPHA,
        "thresholds": {
            "fact_persist": C.FACT_PERSIST_THRESHOLD,
            "objection_open": C.OBJECTION_OPEN_THRESHOLD,
            "next_move_min_confidence": C.NEXT_MOVE_MIN_CONFIDENCE,
            "signal_on": C.SIGNAL_ON_THRESHOLD,
        },
        "questions": {k: v for k, v in C.QUESTIONS.items()},
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
class Connection:
    def __init__(self, ws: WebSocket, judge: JevJudge | None, personalizer: Personalizer | None = None):
        self.ws = ws
        self.session = CallSession(judge) if judge else None
        self.personalizer = personalizer
        self.pstate = PersonalizeState()
        self.ptask: asyncio.Task | None = None
        self.llm_stats = {"requests": 0, "shown": 0, "rejected": 0, "errors": 0, "cum_cost_usd": 0.0, "last_ms": 0.0}
        self.replay_task: asyncio.Task | None = None
        self.speed = 1.0
        self.paused = asyncio.Event()
        self.paused.set()  # set == running
        self.send_lock = asyncio.Lock()

    async def send(self, payload: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(payload))

    async def stop_replay(self) -> None:
        if self.replay_task and not self.replay_task.done():
            self.replay_task.cancel()
            try:
                await self.replay_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.replay_task = None
        self.paused.set()

    def reset_personalization(self) -> None:
        if self.ptask and not self.ptask.done():
            self.ptask.cancel()
        self.ptask = None
        self.pstate = PersonalizeState()

    # -- tailored phrasing: decide in code, generate in the background, verify with Jev, push if still current --
    def maybe_personalize(self, snap: dict[str, Any]) -> None:
        if self.personalizer is None or self.session is None:
            return
        reason = should_personalize(snap.get("coaching") or {}, snap.get("facts") or {}, snap["index"], self.pstate)
        if not reason:
            return
        if self.ptask and not self.ptask.done():
            self.ptask.cancel()  # a newer moment supersedes the in-flight request
        move = C.PLAYBOOK_BY_ID[snap["coaching"]["move_id"]]
        self.pstate.last_move_id, self.pstate.last_index, self.pstate.last_facts = move["id"], snap["index"], frozenset(snap["facts"])
        transcript = [u.to_state() for u in self.session.utterances[-C.PERSONALIZE_CONTEXT_UTTERANCES :]]
        facts = sorted(snap["facts"])
        self.ptask = asyncio.create_task(self._personalize(snap["index"], move, transcript, facts, snap.get("objection"), reason))

    async def _personalize(self, index: int, move: dict[str, Any], transcript: list[dict[str, Any]], facts: list[str], objection: dict[str, Any] | None, reason: str) -> None:
        assert self.session is not None and self.personalizer is not None
        self.llm_stats["requests"] += 1
        await self.send({"type": "phrasing_pending", "index": index, "move_id": move["id"]})  # the UI shows "personalizando…" only while this runs
        try:
            res = await self.personalizer(move, transcript, facts, objection)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.llm_stats["errors"] += 1
            log.warning("personalize failed: %s", e)
            return
        self.llm_stats["cum_cost_usd"] += res.cost_usd
        self.llm_stats["last_ms"] = round(res.latency_ms)
        if res.error:
            self.llm_stats["errors"] += 1
        verified: dict[str, float] = {}
        rejected = res.rejected_reason
        if res.text and C.PERSONALIZE_VERIFY_WITH_JEV:
            ok, verified, why = await verify_with_jev(self.session.judge, res.text, move, transcript, facts)
            if not ok:
                rejected = why
        # stale-protection: the call has moved on to a different move since we started
        latest = self.session.steps[-1] if self.session.steps else None
        if latest is None or latest.coaching.get("move_id") != move["id"] or latest.coaching.get("gated"):
            rejected = rejected or "stale"
        if rejected:
            self.llm_stats["rejected"] += 1
        else:
            self.llm_stats["shown"] += 1
        await self.send(
            {
                "type": "phrasing",
                "index": index,
                "move_id": move["id"],
                "text": None if rejected else res.text,
                "rejected": rejected,
                "reason": reason,
                "llm": {"model": res.model, "latency_ms": round(res.latency_ms), "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "cost_usd": round(res.cost_usd, 6), "error": res.error, **self.llm_stats},
                "verification": verified,
                "candidate": res.text or (res.extra.get("raw") or None),
            }
        )

    async def run_replay(self, call: dict[str, Any], start: int = 0) -> None:
        assert self.session is not None
        utterances = call["utterances"]
        prev_t = float(utterances[start - 1]["t"]) if 0 < start <= len(utterances) else None
        try:
            for u in utterances[start:]:
                if prev_t is not None:
                    gap = min(MAX_REAL_GAP_S, max(0.0, float(u["t"]) - prev_t))
                    await asyncio.sleep(max(MIN_GAP_S, gap / max(0.1, self.speed)))
                prev_t = float(u["t"])
                await self.paused.wait()
                snap = await self.session.ingest(u["speaker"], u["text"], float(u["t"]))
                snap["replay"] = {"call": call["id"], "index": snap["index"], "total": len(utterances)}
                await self.send(snap)
                self.maybe_personalize(snap)
            await self.send({"type": "replay_done", "call": call["id"], "final_probability": self.session.probability})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("replay failed")
            await self.send({"type": "error", "message": f"replay failed: {e}"})


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    ensure_state(ws.app)
    if not auth.authorized(ws.headers, ws.cookies):
        await ws.close(code=4403)
        return
    await ws.accept()
    conn = Connection(ws, ws.app.state.judge, getattr(ws.app.state, "personalizer", None))
    await conn.send(
        {
            "type": "hello",
            "model": C.MODEL,
            "has_api_key": conn.session is not None,
            "llm": getattr(conn.personalizer, "model", C.LLM_MODEL) if conn.personalizer is not None else None,
            "personalize_skip_moves": sorted(C.PERSONALIZE_SKIP_MOVES),
            "calls": list_calls(),
            "playbook": C.PLAYBOOK,
            "weights": conn.session.weights_table() if conn.session else [],
            "stage_prior": C.STAGE_PRIOR,
            "ema_alpha": C.EMA_ALPHA,
            "thresholds": {"next_move_min_confidence": C.NEXT_MOVE_MIN_CONFIDENCE, "objection_open": C.OBJECTION_OPEN_THRESHOLD, "signal_on": C.SIGNAL_ON_THRESHOLD},
        }
    )
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await conn.send({"type": "error", "message": "invalid json"})
                continue
            await handle_message(conn, msg)
    except WebSocketDisconnect:
        pass
    finally:
        await conn.stop_replay()
        conn.reset_personalization()


async def handle_message(conn: Connection, msg: dict[str, Any]) -> None:
    kind = msg.get("type")
    if conn.session is None and kind not in ("ping",):
        await conn.send({"type": "error", "message": "O servidor está sem chave de API: defina OPENROUTER_API_KEY (ou TYPESAFE_API_KEY) e reinicie ou republique."})
        return
    session = conn.session
    assert session is not None

    if kind == "start_replay":
        await conn.stop_replay()
        try:
            call = load_call(str(msg.get("call", "good")))
        except FileNotFoundError as e:
            await conn.send({"type": "error", "message": str(e)})
            return
        conn.speed = float(msg.get("speed", 1.0))
        start = int(msg.get("from") or 0)
        if not (msg.get("keep") and start):  # `keep` resumes a replay after a reconnect, on the restored session
            start = 0
            session.reset()
            conn.reset_personalization()
            await conn.send({"type": "reset", "call": call, "weights": session.weights_table()})
        conn.replay_task = asyncio.create_task(conn.run_replay(call, start))
    elif kind == "pause":
        conn.paused.clear()
        await conn.send({"type": "paused"})
    elif kind == "resume":
        conn.paused.set()
        await conn.send({"type": "resumed"})
    elif kind == "set_speed":
        conn.speed = max(0.25, min(50.0, float(msg.get("speed", 1.0))))
    elif kind == "stop":
        await conn.stop_replay()
        await conn.send({"type": "stopped"})
    elif kind == "reset":
        await conn.stop_replay()
        session.reset()
        conn.reset_personalization()
        await conn.send({"type": "reset", "call": None, "weights": session.weights_table()})
    elif kind == "utterance":
        text = str(msg.get("text", "")).strip()
        speaker = "prospect" if str(msg.get("speaker", "rep")).lower().startswith("p") else "rep"
        if not text:
            return
        snap = await session.ingest(speaker, text, msg.get("t"))
        await conn.send(snap)
        conn.maybe_personalize(snap)
    elif kind == "restore":
        await conn.stop_replay()
        conn.reset_personalization()
        session.restore(list(msg.get("steps") or []))
        snap = session.snapshot()
        snap["type"] = "restored"
        await conn.send(snap)
    elif kind == "set_weights":
        weights = msg.get("weights") or {}
        session.set_weights({k: float(v) for k, v in weights.items()})
        if "stage_prior" in msg and isinstance(msg["stage_prior"], dict):
            for k, v in msg["stage_prior"].items():
                if k in session.stage_prior:
                    session.stage_prior[k] = float(v)
        session.recompute()
        snap = session.snapshot()
        snap["type"] = "recomputed"
        await conn.send(snap)
    elif kind == "ping":
        await conn.send({"type": "pong"})
    else:
        await conn.send({"type": "error", "message": f"unknown message type {kind!r}"})
