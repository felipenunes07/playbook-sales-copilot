"""Unit tests for everything numeric (mock Jev)."""

from __future__ import annotations

import pytest

from copilot import constants as C
from copilot.engine import (
    CallSession,
    Utterance,
    clamp,
    composite,
    ema,
    normalize_score,
    pace_wpm,
    percentile,
    sensitivity,
    talk_ratio,
    word_count,
)
from copilot.replay import list_calls, load_call, parse_text_transcript


@pytest.fixture(autouse=True)
def no_backchannel(monkeypatch, request):
    """Most tests script Jev answers on short placeholder lines ("Sure.", "hmm"); treat every line as a real turn
    there. Tests marked `backchannel` keep the real rule."""
    if "backchannel" not in request.keywords:
        monkeypatch.setattr(C, "BACKCHANNEL_MAX_WORDS", 0)

# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_word_count_and_clamp():
    assert word_count("  hello   world ") == 2
    assert word_count("") == 0
    assert clamp(1.7) == 1.0 and clamp(-2) == 0.0 and clamp(0.4) == 0.4


def test_talk_ratio():
    us = [Utterance("rep", "one two three four", 0), Utterance("prospect", "five six", 5), Utterance("rep", "seven eight", 9)]
    ratio, rep, pro = talk_ratio(us)
    assert (rep, pro) == (6, 2)
    assert ratio == pytest.approx(0.75)
    assert talk_ratio([])[0] == 0.5  # no words -> neutral


def test_pace_wpm_window():
    us = [Utterance("rep", " ".join(["w"] * 30), 0), Utterance("rep", " ".join(["w"] * 30), 30), Utterance("prospect", "hi", 59)]
    # 60 rep words in a 60 s window -> 60 wpm
    assert pace_wpm(us, "rep", 60.0) == pytest.approx(60.0)
    # older words fall out of the window
    us.append(Utterance("prospect", "ok", 130))
    assert pace_wpm(us, "rep", 130.0) == pytest.approx(0.0)


def test_ema():
    assert ema(0.3, 0.8, alpha=0.5) == pytest.approx(0.55)
    assert ema(0.3, 0.3) == pytest.approx(0.3)


def test_normalize_score():
    assert normalize_score({"score": 2.0, "levels": 3}) == pytest.approx(1.0)
    assert normalize_score({"score": 1.0, "levels": 3}) == pytest.approx(0.5)
    assert normalize_score({"score": 0.0, "probabilities": {0: 1.0, 1: 0.0, 2: 0.0}}) == pytest.approx(0.0)


def test_composite_arithmetic_and_centering():
    weights = {
        "a": {"kind": "noul", "weight": 0.2, "label": "a"},
        "s": {"kind": "score", "weight": 0.1, "label": "s"},
        "n": {"kind": "code", "weight": -0.3, "label": "n"},
    }
    prior = {"discovery": 0.3}
    inst, contrib = composite({"a": 0.5, "s": 1.0, "n": 1.0}, "discovery", weights, prior)
    # 0.3 + 0.2*0.5 + 0.1*(1.0-0.5)*2 + (-0.3*1.0) = 0.3 + 0.1 + 0.1 - 0.3 = 0.2
    assert inst == pytest.approx(0.2)
    assert contrib["s"] == pytest.approx(0.1)  # centered: high level is +w
    inst_mid, contrib_mid = composite({"s": 0.5}, "discovery", weights, prior)
    assert contrib_mid["s"] == pytest.approx(0.0)  # middle level is neutral
    assert inst_mid == pytest.approx(0.3)
    # missing features are skipped, unknown stage falls back to INITIAL_PROBABILITY
    inst_unknown, _ = composite({}, "nope", weights, prior)
    assert inst_unknown == pytest.approx(C.INITIAL_PROBABILITY)


def test_composite_clamps():
    weights = {"a": {"kind": "noul", "weight": 5.0, "label": "a"}}
    inst, _ = composite({"a": 1.0}, "opening", weights, {"opening": 0.3})
    assert inst == C.P_MAX
    weights = {"a": {"kind": "noul", "weight": -5.0, "label": "a"}}
    inst, _ = composite({"a": 1.0}, "opening", weights, {"opening": 0.3})
    assert inst == C.P_MIN


def test_default_weights_are_sane():
    positives = sum(v["weight"] for v in C.SIGNAL_WEIGHTS.values() if v["weight"] > 0)
    negatives = sum(v["weight"] for v in C.SIGNAL_WEIGHTS.values() if v["weight"] < 0)
    assert 0.5 <= positives <= 1.2
    assert -0.8 <= negatives <= -0.3
    assert set(C.STAGE_PRIOR) == set(C.STAGES)
    for name, spec in C.SIGNAL_WEIGHTS.items():
        if spec["kind"] in ("noul", "fact"):
            assert C.QUESTIONS[name]["type"] == "noul", name
        elif spec["kind"] == "score":
            assert C.QUESTIONS[name]["type"] == "score", name


def test_sensitivity_top_n():
    prev = {"stage": 0.3, "a": 0.1, "b": 0.0, "c": 0.05}
    curr = {"stage": 0.4, "a": 0.0, "b": 0.02, "c": 0.05}
    top = sensitivity(prev, curr, top_n=2)
    assert [d["signal"] for d in top] == ["stage", "a"]
    assert top[0]["delta"] == pytest.approx(0.1)
    assert top[0]["label"] == "Etapa da call"


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([100, 200], 0.95) == pytest.approx(195)
    assert percentile([], 0.95) == 0.0


# ---------------------------------------------------------------------------
# session behaviour with a scripted judge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fact_persistence_and_state(mock_judge_factory, helpers):
    n = helpers["noul"]
    judge = mock_judge_factory(
        [
            {"budget_discussed": n(0.9)},  # becomes durable
            {"budget_discussed": n(0.1)},  # model forgets; code remembers
            {},
        ]
    )
    s = CallSession(judge)
    await s.ingest("prospect", "We have about a thousand a month for this.", 0)
    assert "budget_discussed" in s.facts
    snap = await s.ingest("rep", "Great, thanks.", 5)
    assert snap["signals"]["budget_discussed"]["value"] == 1.0
    assert snap["signals"]["budget_discussed"]["persisted"] is True
    await s.ingest("prospect", "Sure.", 9)
    # the fact is fed back to Jev via call_facts.known_facts
    assert "budget_discussed" in judge.states[-1]["call_facts"]["known_facts"]
    assert judge.calls == 3


@pytest.mark.asyncio
async def test_prospect_only_facts_ignore_rep_turns(mock_judge_factory, helpers):
    n = helpers["noul"]
    judge = mock_judge_factory([{"next_step_agreed": n(0.95)}])
    s = CallSession(judge)
    snap = await s.ingest("rep", "Thursday works for me.", 0)
    assert "next_step_agreed" not in s.facts
    assert snap["signals"]["next_step_agreed"]["value"] == 0.0


@pytest.mark.asyncio
async def test_speaker_masking(mock_judge_factory, helpers):
    n = helpers["noul"]
    judge = mock_judge_factory([{"buying_signal": n(0.9), "rep_pitching": n(0.9)}, {"buying_signal": n(0.9), "rep_pitching": n(0.9)}])
    s = CallSession(judge)
    rep_snap = await s.ingest("rep", "Our product does X and Y.", 0)
    assert rep_snap["signals"]["buying_signal"]["value"] == 0.0
    assert rep_snap["signals"]["rep_pitching"]["value"] == 0.9
    pro_snap = await s.ingest("prospect", "How do we get started?", 4)
    assert pro_snap["signals"]["buying_signal"]["value"] == 0.9
    assert pro_snap["signals"]["rep_pitching"]["value"] == 0.0


@pytest.mark.asyncio
async def test_objection_lifecycle(mock_judge_factory, helpers):
    n, ch = helpers["noul"], helpers["choice"]
    opts = list(C.OBJECTION_TYPES)
    judge = mock_judge_factory(
        [
            {"prospect_objecting": n(0.9), "objection_type": ch("price", 0.8, opts)},  # opens
            {},  # rep turn: stays open
            {"prospect_accepts": n(0.8), "prospect_disengaging": n(0.9)},  # "sure, gotta run" must NOT clear
            {"prospect_accepts": n(0.8)},  # genuine acceptance clears
        ]
    )
    s = CallSession(judge)
    snap = await s.ingest("prospect", "That's too expensive.", 0)
    assert snap["objection"] == {"open": True, "type": "price", "confidence": 0.8, "since": 0}
    assert snap["signals"]["objection_open"]["value"] == 1.0
    snap = await s.ingest("rep", "Let's compare it to what the problem costs.", 5)
    assert snap["objection"]["open"] is True
    snap = await s.ingest("prospect", "Sure. Okay, I have to run.", 10)
    assert snap["objection"]["open"] is True
    snap = await s.ingest("prospect", "That makes sense.", 15)
    assert snap["objection"]["open"] is False
    assert snap["signals"]["objection_open"]["value"] == 0.0


@pytest.mark.asyncio
async def test_objection_expires(mock_judge_factory, helpers):
    n = helpers["noul"]
    judge = mock_judge_factory([{"prospect_objecting": n(0.9)}] + [{} for _ in range(C.OBJECTION_MAX_AGE_UTTERANCES + 1)])
    s = CallSession(judge)
    await s.ingest("prospect", "Not the right time.", 0)
    assert s.objection.open
    for i in range(C.OBJECTION_MAX_AGE_UTTERANCES + 1):
        await s.ingest("rep" if i % 2 == 0 else "prospect", "blah", 5 * (i + 1))
    assert not s.objection.open


@pytest.mark.asyncio
async def test_coaching_confidence_gate(mock_judge_factory, helpers):
    ch = helpers["choice"]
    moves = [m["id"] for m in C.PLAYBOOK]
    judge = mock_judge_factory(
        [
            {"next_move": ch("ask_about_budget", C.NEXT_MOVE_MIN_CONFIDENCE - 0.05, moves)},
            {"next_move": ch("ask_about_budget", C.NEXT_MOVE_MIN_CONFIDENCE + 0.2, moves), "phrasing::ask_about_budget": ch("p2", 0.9, ["p0", "p1", "p2"])},
        ]
    )
    s = CallSession(judge)
    snap = await s.ingest("prospect", "hmm", 0)
    assert snap["coaching"]["gated"] is True
    snap = await s.ingest("prospect", "hmm", 3)
    c = snap["coaching"]
    assert c["gated"] is False
    assert c["title"] == "Pergunte sobre o orçamento"
    assert c["best_phrasing"] == C.PLAYBOOK_BY_ID["ask_about_budget"]["phrasings"][2]
    assert len(c["phrasings"]) == 3


@pytest.mark.asyncio
async def test_talk_ratio_penalty_and_monologue(mock_judge_factory):
    judge = mock_judge_factory()
    s = CallSession(judge)
    long_text = " ".join(["feature"] * (C.REP_MONOLOGUE_WORDS + 10))
    for i in range(C.TALK_RATIO_MIN_UTTERANCES):
        snap = await s.ingest("rep", long_text, 10 * i)
    assert snap["talk"]["rep_monologue"] is True
    assert snap["talk"]["ratio_rep"] == 1.0
    assert snap["signals"]["rep_talking_too_much"]["value"] == 1.0
    assert snap["contributions"]["rep_talking_too_much"] == pytest.approx(C.SIGNAL_WEIGHTS["rep_talking_too_much"]["weight"])


@pytest.mark.asyncio
async def test_ema_over_utterances_and_timeline(mock_judge_factory, helpers):
    n = helpers["noul"]
    judge = mock_judge_factory([{"commitment": n(1.0), "buying_signal": n(1.0)}, {}])
    s = CallSession(judge)
    snap1 = await s.ingest("prospect", "Let's do it, send the contract.", 0)
    inst1 = snap1["instant"]
    assert snap1["probability"] == pytest.approx(ema(C.INITIAL_PROBABILITY, inst1), abs=1e-3)
    snap2 = await s.ingest("rep", "Great.", 4)
    assert snap2["probability"] < snap1["probability"]  # transient signal fades through the EMA
    assert [p["index"] for p in snap2["timeline"]] == [0, 1]


@pytest.mark.asyncio
async def test_recompute_with_new_weights_makes_no_jev_calls(mock_judge_factory, helpers):
    n = helpers["noul"]
    judge = mock_judge_factory([{"buying_signal": n(1.0)}, {}, {}])
    s = CallSession(judge)
    for i, (spk, txt) in enumerate([("prospect", "How do we start?"), ("rep", "Great."), ("prospect", "ok")]):
        await s.ingest(spk, txt, 5 * i)
    before = s.timeline()
    calls_before = judge.calls
    s.set_weights({"buying_signal": 0.5})
    after_probs = s.recompute()
    assert judge.calls == calls_before
    assert after_probs[0] > before[0]["probability"]
    assert s.weights_table()[[w["signal"] for w in s.weights_table()].index("buying_signal")]["weight"] == 0.5
    # default constants untouched
    assert C.SIGNAL_WEIGHTS["buying_signal"]["weight"] != 0.5


@pytest.mark.asyncio
async def test_sensitivity_reported_per_utterance(mock_judge_factory, helpers):
    n = helpers["noul"]
    judge = mock_judge_factory([{}, {"prospect_disengaging": n(1.0)}])
    s = CallSession(judge)
    await s.ingest("prospect", "hi", 0)
    snap = await s.ingest("prospect", "just send me info", 4)
    assert snap["sensitivity"][0]["signal"] == "prospect_disengaging"
    assert snap["sensitivity"][0]["delta"] < 0
    assert len(snap["sensitivity"]) <= C.SENSITIVITY_TOP_N


@pytest.mark.asyncio
async def test_state_window_and_shape(mock_judge_factory):
    judge = mock_judge_factory()
    s = CallSession(judge)
    for i in range(C.RECENT_WINDOW + 5):
        await s.ingest("rep" if i % 2 == 0 else "prospect", f"turn {i}", 3 * i)
    state = judge.states[-1]
    assert len(state["recent_transcript"]) == C.RECENT_WINDOW
    assert state["latest_utterance"]["text"] == f"turn {C.RECENT_WINDOW + 4}"
    assert set(state["call_facts"]) == {"duration_min", "talk_ratio_rep", "stage_history", "known_facts", "objection"}


@pytest.mark.asyncio
async def test_judge_error_keeps_session_alive(helpers):
    from copilot.jev_client import JudgeResult

    async def failing(state, questions):
        return JudgeResult(answers={}, error="boom")

    s = CallSession(failing)
    snap = await s.ingest("prospect", "hello", 0)
    assert snap["jev_last"]["error"] == "boom"
    assert snap["coaching"]["gated"] is True
    assert 0 < snap["probability"] < 1
    assert s.errors == 1


# ---------------------------------------------------------------------------
# replay data
# ---------------------------------------------------------------------------


def test_replay_calls_load_and_are_well_formed():
    calls = list_calls()
    assert [c["id"] for c in calls][:3] == ["good", "bad", "mixed"]
    for meta in calls:
        call = load_call(meta["id"])
        assert 35 <= len(call["utterances"]) <= 200  # hand-written calls ~40, real recordings up to ~150
        ts = [u["t"] for u in call["utterances"]]
        assert ts == sorted(ts)
        assert {u["speaker"] for u in call["utterances"]} == {"rep", "prospect"}


def test_parse_text_transcript():
    call = parse_text_transcript("rep: Hi there\n[12] prospect: Hello back\nnoise line\nREP: ok")
    assert [u["speaker"] for u in call["utterances"]] == ["rep", "prospect", "rep"]
    assert call["utterances"][1]["t"] == 12.0
    assert call["utterances"][2]["t"] > 12.0


def test_questions_are_complete_and_literal():
    """Every weighted signal that Jev answers has a full question; every question names the state path it reads."""
    for qid, q in C.QUESTIONS.items():
        text = str(q["instructions"])
        assert "`recent_transcript`" in text or "`latest_utterance" in text, qid
        if q["type"] == "choice":
            assert 2 <= len(q["criteria"]) <= 255
        if q["type"] == "score":
            assert 2 <= len(q["criteria"]) <= 10
    assert set(C.QUESTIONS["next_move"]["criteria"]) == set(C.PLAYBOOK_BY_ID)
    assert "none" in C.QUESTIONS["objection_type"]["criteria"]


# ---------------------------------------------------------------------------
# listening noises, prospect-only scores, coaching streak, qualification checklist
# ---------------------------------------------------------------------------


@pytest.mark.backchannel
async def test_backchannel_skips_jev_and_keeps_the_reading(mock_judge_factory):
    judge = mock_judge_factory()
    s = CallSession(judge)
    first = await s.ingest("prospect", "Our follow-ups keep falling through every week.", 0)
    snap = await s.ingest("prospect", "Legal.", 4)
    assert judge.calls == 1  # no Jev call for a listening noise
    assert snap["backchannel"] is True
    assert snap["stage"] == first["stage"] and snap["coaching"] == first["coaching"]
    assert snap["sensitivity"] == []
    snap = await s.ingest("prospect", "Isso funciona?", 6)  # a short question is a real turn
    assert judge.calls == 2 and snap["backchannel"] is False


async def test_engagement_only_moves_on_prospect_turns(mock_judge_factory, helpers):
    score = helpers["score"]
    judge = mock_judge_factory([{"engagement": score(2.0)}, {"engagement": score(0.0)}])
    s = CallSession(judge)
    await s.ingest("prospect", "Tell me more about how it works for us.", 0)
    await s.ingest("rep", "It runs every follow-up for you automatically.", 5)
    assert s.steps[-1].features["engagement"] == 1.0  # the rep's turn keeps the prospect's last reading


async def test_coaching_streak_counts_repeats(mock_judge_factory):
    s = CallSession(mock_judge_factory())
    for i in range(3):
        snap = await s.ingest("rep", f"Explaining part {i} of the offer.", 5 * i)
    assert snap["coaching"]["streak"] == 3


async def test_rep_streak_seconds(mock_judge_factory, monkeypatch):
    monkeypatch.setattr(C, "BACKCHANNEL_MAX_WORDS", 3)
    s = CallSession(mock_judge_factory())
    await s.ingest("prospect", "What would you suggest for me?", 0)
    await s.ingest("rep", "Let me show you the system.", 10)
    await s.ingest("prospect", "Legal.", 40)
    snap = await s.ingest("rep", "And this part handles the follow-ups for you.", 70)
    assert snap["talk"]["rep_streak_s"] == 60.0  # "Legal." does not give the floor back


async def test_qualification_nudge_after_discovery(mock_judge_factory, helpers):
    choice, noul = helpers["choice"], helpers["noul"]
    judge = mock_judge_factory([{"pain_identified": noul(0.9)}, {"stage": choice("pricing", 0.9, list(C.STAGES))}])
    s = CallSession(judge)
    snap = await s.ingest("prospect", "Prospecting by hand eats my whole week.", 0)
    assert snap["qualification"]["nudge"] is None  # still in discovery
    snap = await s.ingest("rep", "The investment is five thousand, in two payments.", 5)
    q = snap["qualification"]
    assert [i["done"] for i in q["items"]][:2] == [True, False]
    assert q["nudge"]["fact"] == "decision_maker_identified"


async def test_stop_talking_waits_for_a_real_monologue(mock_judge_factory, helpers):
    ids = [m["id"] for m in C.PLAYBOOK]
    stop = {"next_move": helpers["choice"]("stop_talking_ask_open_question", 0.9, ids)}
    s = CallSession(mock_judge_factory([{}, stop, stop]))
    await s.ingest("prospect", "What would you suggest for someone like me?", 0)
    snap = await s.ingest("rep", "Let me show you how the system works for you.", 10)
    assert snap["coaching"]["move_id"] != "stop_talking_ask_open_question"  # 10 s answering a question
    snap = await s.ingest("rep", "And it also handles every follow-up automatically.", 75)
    assert snap["coaching"]["move_id"] == "stop_talking_ask_open_question"  # 75 s holding the floor


async def test_restore_rebuilds_the_call_without_jev(mock_judge_factory, helpers):
    noul = helpers["noul"]
    judge = mock_judge_factory([{"pain_identified": noul(0.9)}, {}, {}])
    s = CallSession(judge)
    for spk, txt, t in (("prospect", "Following up by hand eats my whole week.", 0), ("rep", "How many leads go cold?", 5), ("prospect", "Maybe ten a month.", 9)):
        await s.ingest(spk, txt, t)
    history = [{"speaker": st.utterance.speaker, "text": st.utterance.text, "t": st.utterance.t, "answers": st.answers} for st in s.steps]
    before = s.snapshot()
    fresh_judge = mock_judge_factory()
    r = CallSession(fresh_judge)
    r.restore(history)
    after = r.snapshot()
    assert fresh_judge.calls == 0
    assert after["probability"] == before["probability"]
    assert after["facts"] == before["facts"] and after["stage"] == before["stage"]
    assert [x["probability"] for x in after["timeline"]] == [x["probability"] for x in before["timeline"]]


def test_private_calls_load_from_urls(monkeypatch):
    import copilot.replay as R

    class Resp:
        def raise_for_status(self):
            return self

        def json(self):
            return {"id": "secret_call", "title": "Real call", "video": {"url": "https://blob.example/v.mp4"},
                    "utterances": [{"t": 0, "speaker": "Rep", "text": "Oi"}, {"t": 3, "speaker": "prospect", "text": "Olá"}]}

    monkeypatch.setattr("httpx.get", lambda *a, **k: Resp())
    monkeypatch.setattr(R, "_REMOTE", {})
    assert R.load_remote_calls("https://blob.example/call.json, ") == ["secret_call"]
    assert "secret_call" in [c["id"] for c in R.list_calls()]
    call = R.load_call("secret_call")
    assert call["utterances"][0]["speaker"] == "rep" and call["video"]["url"].endswith("v.mp4")


async def test_roteiro_guides_step_by_step(mock_judge_factory, helpers):
    noul, choice = helpers["noul"], helpers["choice"]
    stages = list(C.STAGES)
    judge = mock_judge_factory([
        {"stage": choice("opening", 0.9, stages)},
        {"stage": choice("discovery", 0.9, stages), "pain_identified": noul(0.9)},
        {"stage": choice("pricing", 0.9, stages)},
    ])
    s = CallSession(judge)
    r = (await s.ingest("rep", "Oi, obrigado pelo tempo, o que te fez marcar a conversa?", 0))["roteiro"]
    assert r["current"] == 0 and r["next_ask"]  # abertura, with a question ready
    r = (await s.ingest("prospect", "Faço a prospecção sozinho e perco muito lead sem follow-up.", 5))["roteiro"]
    steps = {st["id"]: st for st in r["steps"]}
    assert steps["abertura"]["done"]
    assert [i["done"] for i in steps["descoberta"]["items"]] == [True, False]  # pain yes, numbers not yet
    assert r["steps"][r["current"]]["id"] == "descoberta" and "Quantas horas" in r["next_ask"]
    r = (await s.ingest("rep", "O investimento é cinco mil, em duas vezes.", 10))["roteiro"]
    assert r["steps"][r["current"]]["id"] == "preco"  # the conversation jumped ahead
    assert {g["label"] for g in r["gaps"]} >= {"Impacto em números", "Decisor", "Prazo"}  # what was skipped
