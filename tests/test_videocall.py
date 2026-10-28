from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from copilot import auth
from copilot.server import app
from copilot.videocall import clean_transcript, is_local, new_room, room_valid

PUBLIC = {"host": "copilot.example.com", "x-forwarded-proto": "https"}
LOCAL = {"host": "localhost"}


def test_is_local():
    assert is_local({"host": "localhost:8000"})
    assert is_local({"host": "127.0.0.1"})
    assert is_local({"host": "[::1]:8000"})
    assert not is_local({"host": "abc.trycloudflare.com"})
    assert not is_local({"host": "localhost:8000", "cf-ray": "x"})  # tunnelled, whatever the Host says


def test_clean_transcript():
    assert clean_transcript(' "Achei caro." ') == "Achei caro."
    assert clean_transcript("(vazio)") == ""
    assert clean_transcript("…") == ""


def test_rooms_are_signed_and_expire():
    room = new_room(now=1000)
    assert room_valid(room, now=1001)
    assert not room_valid(room, now=1000 + 7 * 3600)  # expired
    assert not room_valid(room[:-1] + ("0" if room[-1] != "0" else "1"), now=1001)  # tampered
    assert not room_valid("nope", now=1001)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "segredo-do-time")
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def test_public_requests_need_login_except_the_guest_page(client):
    room = client.post("/api/sala").json()["room"]
    r = client.get("/", headers=PUBLIC, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.post("/api/sala", headers=PUBLIC).status_code == 401
    assert client.post("/api/transcribe", headers=PUBLIC, content=b"x").status_code == 401
    assert client.get("/login", headers=PUBLIC).status_code == 200
    assert client.get(f"/sala/{room}", headers=PUBLIC).status_code == 200
    assert client.get("/sala/nope", headers=PUBLIC).status_code == 404
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers=PUBLIC) as ws:
            ws.receive_json()
    # nobody outside can take the rep's seat in a room
    assert client.get(f"/sala/{room}/poll?role=host", headers=PUBLIC).status_code == 404


def test_login_sets_a_cookie_that_opens_the_copilot(client):
    r = client.post("/login", headers=PUBLIC, content="password=errada", follow_redirects=False)
    assert r.headers["location"] == "/login?error=1"
    r = client.post("/login", headers=PUBLIC, content="password=segredo-do-time", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.cookies.get(auth.COOKIE)
    assert cookie and auth.cookie_valid(cookie)
    headers = {**PUBLIC, "cookie": f"{auth.COOKIE}={cookie}"}
    assert client.get("/", headers=headers).status_code == 200
    room = client.post("/api/sala", headers=headers).json()
    assert room["guest_url"].startswith("https://copilot.example.com/sala/")
    with client.websocket_connect("/ws", headers=headers) as ws:
        assert ws.receive_json()["type"] == "hello"


def test_no_password_means_no_way_in_from_outside(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert not auth.cookie_valid(auth.make_cookie())
    assert not auth.password_ok("")


def test_signaling_relays_between_host_and_guest(client):
    room = client.post("/api/sala").json()["room"]
    host, guest = f"/sala/{room}/{{}}?role=host", f"/sala/{room}/{{}}?role=guest"
    assert client.post(guest.format("send"), headers=PUBLIC, json={"type": "peer-joined"}).json() == {"ok": True}
    assert client.get(host.format("poll"), headers=LOCAL).json() == [{"type": "peer-joined", "role": "guest"}]
    client.post(host.format("send"), headers=LOCAL, json={"type": "offer", "sdp": "v=0"})
    client.post(host.format("send"), headers=LOCAL, json={"type": "ice", "candidate": None})
    assert client.get(guest.format("poll"), headers=PUBLIC).json() == [{"type": "offer", "sdp": "v=0"}, {"type": "ice", "candidate": None}]
    assert client.get(guest.format("poll"), headers=PUBLIC).json() == []  # drained
    assert client.post(guest.format("send"), headers=PUBLIC, json={"type": "hack"}).status_code == 400
    client.post(guest.format("send"), headers=PUBLIC, json={"type": "answer", "sdp": "v=0"})
    assert client.get(host.format("poll"), headers=LOCAL).json() == [{"type": "answer", "sdp": "v=0"}]


def test_open_access_drops_the_login(monkeypatch):
    monkeypatch.setenv("OPEN_ACCESS", "1")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    with TestClient(app, base_url="http://localhost") as c:
        assert c.get("/", headers=PUBLIC).status_code == 200
        assert c.post("/api/sala", headers=PUBLIC).status_code == 200
        with c.websocket_connect("/ws", headers=PUBLIC) as ws:
            assert ws.receive_json()["type"] == "hello"


def test_rooms_survive_another_instance(monkeypatch):
    """Serverless: the guest may hit a different process. Same keys => same secret => the room is still valid."""
    import copilot.videocall as V

    for k in ("APP_SECRET", "APP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    room = V.new_room()
    monkeypatch.setattr(V, "_LOCAL_SECRET", b"another process")
    assert V.room_valid(room)
