"""Single reviewable module: model pin, every Jev question, composite weights, thresholds.

Nothing in here does I/O except loading the hand-written playbook JSON.
Change a coefficient here, not a prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model & pricing
# ---------------------------------------------------------------------------
MODEL = "jev-1.13.0"  # pinned; `jev-latest` moves on release and would shift thresholds
PRICE_PER_M_INPUT_TOKENS_USD = 0.042  # output tokens are free
API_KEY_ENV_VARS = ("TYPESAFE_API_KEY", "JEV_API_KEY", "JEVMODEL_API_KEY", "BEATAPI_API_KEY", "OPENROUTER_API_KEY")
# Hosted SystemOne-compatible providers (`sk-…` keys), see jev_client.make_judge
HOSTED_REQUEST_TIMEOUT_S = 30.0
HOSTED_MAX_CONCURRENCY = 6
# OpenRouter: real Jev (provider TypeSafe) on its decisions endpoint, same request shape as SystemOne
OPENROUTER_BASE_URL = "https://openrouter.ai"
OPENROUTER_DECISIONS_PATH = "/api/alpha/decisions"
OPENROUTER_JEV_MODEL = "typesafe/jev-1.13"
OPENROUTER_STT_MODEL = "openai/gpt-audio-mini"  # video call transcription: ~$0.0008/min of speech, PT-BR verified
OPENROUTER_LLM_MODEL = "qwen/qwen3.7-flash"  # tailored phrasing when no ANTHROPIC_API_KEY: ~5x cheaper than Claude Haiku 5.5 with the same quality in PT-BR (tested 2026-10-28)
# BeatAPI: one request per utterance; free accounts only get the `-free` models
BEATAPI_BASE_URL = "https://api.beatapi.io"
BEATAPI_MODEL = "jev-1.13-free"
# jevmodel.org: max 8 questions per request
JEVMODEL_BASE_URL = "https://jevmodel.org"
JEVMODEL_MAX_QUESTIONS_PER_REQUEST = 8
JEVMODEL_MAX_CRITERIA_CHARS = 1950  # API cap is 2,000 serialized chars per question
REQUEST_TIMEOUT_S = 8.0

# ---------------------------------------------------------------------------
# State shaping (kept in code: rolling window, arithmetic)
# ---------------------------------------------------------------------------
RECENT_WINDOW = 12  # utterances of transcript sent to Jev
# Economy mode (COPILOT_ECONOMY=1): fewer input tokens per utterance for free-tier keys. Skips the 15 phrasing
# Choices, questions masked for this speaker anyway, and facts already locked in; shorter transcript window.
ECONOMY = __import__("os").environ.get("COPILOT_ECONOMY", "") == "1"
RECENT_WINDOW_ECONOMY = 6
# Hosted providers: identical requests are answered from this on-disk cache (replaying a call twice costs nothing)
JEV_CACHE_DIR = __import__("os").environ.get("COPILOT_CACHE_DIR") or ("/tmp/jev-cache" if __import__("os").environ.get("VERCEL") else ".cache/jev")  # only /tmp is writable on Vercel
PACE_WINDOW_S = 60.0  # words-per-minute window for pace
TALK_RATIO_MIN_UTTERANCES = 6  # talk-ratio penalty only after this many utterances
REP_MONOLOGUE_WORDS = 70  # a single rep utterance longer than this is flagged in code
BACKCHANNEL_MAX_WORDS = 3  # "legal", "uhum", "perfeito": no Jev call, the number does not move
STOP_TALKING_MIN_STREAK_S = 60.0  # below this, "stop talking" yields to the next most likely move
REP_STREAK_WARN_S = 90.0  # the rep has held the floor this long without a real prospect turn -> UI warning

# Qualification checklist shown to the rep. Once the call is past discovery (or long enough), the first missing item
# becomes a nudge with a ready question, so gaps are flagged before the price, not after.
QUALIFICATION: list[dict[str, str]] = [
    {"fact": "pain_identified", "label": "Dor", "ask": "Qual parte do teu processo comercial mais te toma tempo hoje?"},
    {"fact": "decision_maker_identified", "label": "Decisor", "ask": "Além de você, mais alguém participa dessa decisão, um sócio por exemplo?"},
    {"fact": "timeline_known", "label": "Prazo", "ask": "Para quando você gostaria de ter isso rodando?"},
    {"fact": "budget_discussed", "label": "Orçamento", "ask": "Hoje você já investe em alguma ferramenta ou pessoa para prospecção? Quanto mais ou menos?"},
    {"fact": "next_step_agreed", "label": "Próximo passo", "ask": "Que tal já deixarmos marcada a próxima conversa, com dia e hora?"},
]
QUALIFICATION_LATE_STAGES = ("pitch_demo", "pricing", "objection_handling", "closing", "next_steps")
QUALIFICATION_AFTER_UTTERANCES = 20

# ---------------------------------------------------------------------------
# Playbook (hand-written text; Jev only picks)
# ---------------------------------------------------------------------------
PLAYBOOK_PATH = Path(__file__).resolve().parent.parent / "data" / "playbook.json"


def load_playbook(path: Path = PLAYBOOK_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["moves"]


PLAYBOOK: list[dict[str, Any]] = load_playbook()

# The call roadmap shown to the rep (data/roteiro.json): ordered steps whose items are ticked from Jev's answers.
ROTEIRO_PATH = Path(__file__).resolve().parent.parent / "data" / "roteiro.json"
with open(ROTEIRO_PATH, encoding="utf-8") as _f:
    ROTEIRO: list[dict[str, Any]] = json.load(_f)["steps"]
PLAYBOOK_BY_ID: dict[str, dict[str, Any]] = {m["id"]: m for m in PLAYBOOK}

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
STAGES: dict[str, dict[str, str]] = {
    "opening": {
        "what": "Greetings, small talk, agenda setting, or thanking for the time. Nothing about the prospect's business problem yet.",
        "not_for": "Turns that describe the prospect's process or problems.",
    },
    "discovery": {
        "what": "The rep asks about, or the prospect describes, their current process, problems, team, or goals.",
        "not_for": "Turns about budget, decision makers, or timelines (that is qualification), or about the product itself.",
    },
    "qualification": {
        "what": "Talk about budget, who decides, approval process, or the timeline for deciding or implementing.",
        "not_for": "Describing the problem itself, or the product's features.",
    },
    "pitch_demo": {
        "what": "The rep explains or shows what the product does, its features, or how it would work for the prospect.",
        "not_for": "Turns where the prospect is pushing back or asking about price.",
    },
    "objection_handling": {
        "what": "The prospect has voiced a concern, doubt, or pushback (about price, timing, need, trust, or a competitor) and the rep is responding to it.",
        "not_for": "Neutral questions about how the product works.",
    },
    "pricing": {
        "what": "Discussion of the price, plans, tiers, discounts, or contract terms of the product being sold.",
        "not_for": "The prospect's own budget in general (qualification).",
    },
    "closing": {
        "what": "The rep asks for the commitment, or the prospect says yes to buying, signing, or starting.",
        "not_for": "Scheduling a demo or follow-up (that is next_steps).",
    },
    "next_steps": {
        "what": "Agreeing on what happens after the call: a demo, trial, follow-up meeting, sending a proposal, or introducing other people.",
        "not_for": "Asking for the purchase itself.",
    },
}

# Baseline closing probability contributed by the stage the call is in (before signals).
STAGE_PRIOR: dict[str, float] = {
    "opening": 0.25,
    "discovery": 0.30,
    "qualification": 0.35,
    "pitch_demo": 0.35,
    "objection_handling": 0.30,
    "pricing": 0.38,
    "closing": 0.45,
    "next_steps": 0.42,
}

OBJECTION_TYPES: dict[str, dict[str, str]] = {
    "price": {"what": "The cost, price, fees, or value for money is too high or hard to justify."},
    "timing": {"what": "Not the right time: too busy, mid-project, revisit next quarter or after the season."},
    "authority": {"what": "The prospect cannot decide alone; someone else (boss, owner, committee, IT) must approve."},
    "need": {"what": "The prospect doubts they have the problem, or thinks the current way is fine."},
    "trust": {"what": "Doubts about the vendor, the product working for them, implementation risk, or disruption."},
    "competitor": {"what": "Prefers, uses, or is evaluating another vendor instead of buying from the rep. Tools, apps, or AI models that the rep's own solution runs on, includes, or recommends are not competitors."},
    "none": {"what": "The prospect has not voiced any concern or pushback about buying."},
}

# ---------------------------------------------------------------------------
# Questions (one speculative fan-out request per utterance)
# ids are never sent to the model: the complete question is in `instructions`.
# ---------------------------------------------------------------------------
_PROSPECT = "spoken by the prospect (`latest_utterance.speaker` is prospect)"


def _noul(question: str, true: str | None = None, false: str | None = None, **extra: Any) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "noul", "instructions": {"question": question, **extra}}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return q


QUESTIONS: dict[str, dict[str, Any]] = {
    # ---- prospect-turn Nouls (masked in code when the speaker is the rep) ----
    "buying_signal": _noul(
        "Does `latest_utterance.text`, " + _PROSPECT + ", express interest in moving forward with the product?",
        true="Asks how to get started, about onboarding, implementation, contract terms, or rollout; says the product would solve their problem; asks to see it or try it; positive reaction such as 'that's exactly what we need'.",
        false="Neutral information, a question about how a feature works, a concern, or small talk.",
    ),
    "commitment": _noul(
        "Does `latest_utterance.text`, " + _PROSPECT + ", explicitly commit to buying, signing, or starting a paid engagement?",
        true="'Let's do it', 'send over the contract', 'we're in', 'sign us up', agreeing to a purchase or paid pilot.",
        false="Interest without commitment, agreeing only to a demo or follow-up call, or any concern.",
    ),
    "next_step_agreed": _noul(
        "Does `latest_utterance.text`, " + _PROSPECT + ", agree to a concrete next step such as a demo, trial, follow-up meeting, proposal review, or introducing a colleague?",
        true="'Thursday works', 'yes, send the proposal', 'let's get my boss on the next call', 'sure, set up the demo'.",
        false="Vague 'send me some info', 'we'll think about it', a question, or a concern.",
    ),
    "prospect_objecting": _noul(
        "Does `latest_utterance.text`, " + _PROSPECT + ", raise a concern, hesitation, or pushback about buying the product?",
        true="Says it is too expensive, not the right time, needs someone else's approval, doubts they need it or that it will work, prefers a competitor, or asks a skeptical 'why would we...' question.",
        false="Neutral or clarifying questions about how something works, what it includes, its prerequisites, plans, or limits (even if detailed); sharing information about their situation; agreement; or small talk.",
    ),
    "prospect_accepts": _noul(
        "Does `latest_utterance.text`, " + _PROSPECT + ", accept, agree with, or acknowledge as reasonable what the rep just said?",
        true="'That makes sense', 'ok, fair enough', 'that would work for us', 'good point', 'I see'.",
        false="Disagreement, a new concern, a neutral question, or unrelated information.",
    ),
    "prospect_disengaging": _noul(
        "Does `latest_utterance.text`, " + _PROSPECT + ", signal that they want to end or shorten the call, or brush the rep off?",
        true="'Just send me some info', 'I need to run', 'we'll think about it and get back to you', 'not really a priority right now', one-word dismissive answers.",
        false="Engaged questions, detailed answers about their situation, or agreeing to next steps.",
    ),
    "buyer_confused": _noul(
        "Does `latest_utterance.text`, " + _PROSPECT + ", show that they are confused or did not follow what the rep said?",
        true="Asks the rep to repeat or clarify, asks what a term means, says 'I'm not sure I follow', or asks a question the rep's previous turn already addressed.",
        false="A clear question about something new, a statement, or agreement.",
    ),
    "prospect_asked_price": _noul(
        "Does `latest_utterance.text` ask what the product costs, its price, plans, or pricing model?",
    ),
    # ---- rep-turn Nouls (masked in code when the speaker is the prospect) ----
    "rep_discovery_question": _noul(
        "Is `latest_utterance.text`, spoken by the rep (`latest_utterance.speaker` is rep), an open-ended question inviting the prospect to describe their situation, current process, problems, goals, or how they make decisions?",
        true="'Walk me through how your prospecting works today', 'what's the most painful part of that?', 'what happens when a lead goes cold?', 'who else is involved in a decision like this?'",
        false="A yes/no question, a statement, describing product features, quoting a price, or proposing a meeting.",
    ),
    "rep_pitching": _noul(
        "Is `latest_utterance.text`, spoken by the rep (`latest_utterance.speaker` is rep), volunteering product features, capabilities, or benefits that the prospect did not ask about, without asking the prospect a question?",
        true="Lists what the product does, how it works, its integrations, or benefits on the rep's own initiative, and ends without a question to the prospect.",
        false="Answers a question or request the prospect made in `recent_transcript`, asks the prospect a question, talks about the prospect's situation, quotes a price, or proposes a next step.",
    ),
    "next_step_proposed": _noul(
        "Does `latest_utterance.text` propose a concrete next step such as a demo, a trial or pilot, a follow-up meeting with a day or time, sending a proposal, or bringing in another stakeholder?",
        true="'Can we set up a demo Thursday?', 'I'll send the proposal tonight', 'let's get your ops lead on the next call', 'we could start with a two-week pilot'.",
        false="Describing features, asking about the prospect's situation, or a vague 'let's stay in touch'.",
    ),
    # ---- transcript-window Nouls (durable facts, persisted in code once confident) ----
    "pain_identified": _noul(
        "In `recent_transcript`, has the prospect described a specific problem, frustration, inefficiency, or cost they are currently experiencing in their business?",
        true="Prospect mentions things like hours of manual work, leads or follow-ups falling through, no time to prospect, losing customers or deals, a process that does not scale, or paying for a tool that does not work.",
        false="Only the rep talks about problems in general, or the prospect says things are fine.",
    ),
    "pain_quantified": _noul(
        "In `recent_transcript`, has the prospect put a number on their problem or its cost: hours spent, money, leads, deals, customers, or jobs lost per week or month?",
        true="'About ten hours a week', 'we lose maybe ten leads a month', 'it costs us around fifteen thousand a month', 'half my week goes into this'.",
        false="The problem was described only in words, or only the rep mentioned numbers.",
    ),
    "budget_discussed": _noul(
        "In `recent_transcript`, has money been discussed: the price or investment of the rep's offer, or the prospect's budget, what they pay today, or whether funds are available?",
        true="The rep states the price or investment of the offer, or the prospect states a budget, a range, what they currently pay, or that funds are or are not approved.",
        false="Nobody has mentioned the price of the offer or the prospect's budget.",
    ),
    "decision_maker_identified": _noul(
        "In `recent_transcript`, has the prospect stated who makes the purchase decision (themselves, a named person, or a role) or how the decision gets approved?",
        true="'I can sign off on this', 'the owner has the final say', 'my boss and I decide together', 'it needs to go through our GM'.",
        false="Nothing said about who decides or approves.",
    ),
    "timeline_known": _noul(
        "In `recent_transcript`, has the prospect stated when they want to decide, buy, or have a solution in place?",
        true="'Before the summer season', 'this quarter', 'we need something by March', 'not until next year'.",
        false="No timing for a decision or rollout has been mentioned by the prospect.",
    ),
    "competitor_mentioned": _noul(
        "Does `latest_utterance.text` mention another vendor or an alternative that the prospect uses or is evaluating INSTEAD of buying from the rep?",
        true="Names a competing company or service the prospect is considering instead, 'we're also looking at two other agencies', 'we already pay another vendor for this'.",
        false="Mentions only the rep's offer, manual processes, or tools, apps, and AI models that the rep's own solution uses, includes, or recommends (those are part of the offer, not alternatives to it).",
    ),
    # ---- Scores (levels are concrete situations, one dimension each) ----
    "rapport": {
        "type": "score",
        "instructions": {
            "question": "How would you describe the tone between rep and prospect in `recent_transcript`?",
            "focus": "Judge the human tone, not whether the deal is going well.",
        },
        "criteria": [
            {"summary": "Tense or cold", "signals": ["Curt answers", "Irritation, sarcasm, or impatience", "Interrupting or talking past each other"]},
            {"summary": "Neutral and businesslike", "signals": ["Polite and factual", "No personal remarks or humor", "Efficient question-and-answer"]},
            {"summary": "Warm and friendly", "signals": ["Humor or personal remarks", "Mutual acknowledgement ('great question', 'I appreciate that')", "Relaxed, collaborative language"]},
        ],
    },
    "engagement": {
        "type": "score",
        "instructions": {
            "question": "How engaged is the prospect, judging by the prospect's turns in `recent_transcript`?",
            "focus": "Look only at the prospect's turns.",
        },
        "criteria": [
            {"summary": "Disengaged", "signals": ["One-word or one-line answers", "Deflecting ('just send info')", "Trying to end the call"]},
            {"summary": "Passive", "signals": ["Answers what is asked", "Volunteers nothing extra", "No questions back"]},
            {"summary": "Active", "signals": ["Asks questions about the product or process", "Gives detail about their situation", "Volunteers information or ideas"]},
        ],
    },
    "urgency": {
        "type": "score",
        "instructions": {
            "question": "How urgent is the prospect's need to solve their problem, judging by the prospect's turns in `recent_transcript`?",
        },
        "criteria": [
            {"summary": "No urgency", "signals": ["Exploring or curious", "'Someday', 'eventually'", "No deadline or pressure mentioned"]},
            {"summary": "Moderate", "signals": ["The problem hurts and is mentioned as ongoing", "Wants to fix it but no deadline"]},
            {"summary": "Pressing", "signals": ["A deadline, season, mandate, or contract renewal is named", "Acute pain: losing customers or staff now"]},
        ],
    },
    # ---- Choices ----
    "stage": {
        "type": "choice",
        "instructions": {
            "question": "Which stage of a sales call best describes the last few turns of `recent_transcript`, weighting `latest_utterance` most?",
        },
        "criteria": STAGES,
    },
    "objection_type": {
        "type": "choice",
        "instructions": {
            "question": "If the prospect's most recent turns in `recent_transcript` express a concern or pushback about buying, which kind of concern is it? Choose `none` if no concern has been voiced.",
        },
        "criteria": OBJECTION_TYPES,
    },
    "next_move": {
        "type": "choice",
        "instructions": {
            "question": "Which coaching move should the rep make next, given `recent_transcript`, `latest_utterance`, and `call_facts` (`call_facts.known_facts` lists what is already established, `call_facts.objection` describes any open concern)?",
            "focus": "Pick the move whose `what` matches the current moment; respect each move's `not_for`.",
        },
        "criteria": {m["id"]: {"what": m["what"], "not_for": m["not_for"]} for m in PLAYBOOK},
    },
}

# Speculative per-move phrasing Choices: Jev picks which hand-written line fits the moment.
for _m in PLAYBOOK:
    QUESTIONS[f"phrasing::{_m['id']}"] = {
        "type": "choice",
        "instructions": {
            "question": f"If the rep's next move is '{_m['title']}', which of these lines fits best as the rep's next sentence given `recent_transcript`?",
        },
        "criteria": {f"p{i}": text for i, text in enumerate(_m["phrasings"])},
    }

# Which Nouls only make sense for a given speaker of `latest_utterance`; others are masked to 0 in code.
PROSPECT_ONLY_SIGNALS = (
    "buying_signal",
    "commitment",
    "next_step_agreed",
    "prospect_objecting",
    "prospect_accepts",
    "prospect_disengaging",
    "buyer_confused",
    "prospect_asked_price",
)
REP_ONLY_SIGNALS = ("rep_discovery_question", "rep_pitching")
PROSPECT_SCORES = ("engagement", "urgency")  # judged on prospect turns only; rep turns carry the last value

# ---------------------------------------------------------------------------
# Composite closing probability (all arithmetic in code)
#
# inst = STAGE_PRIOR[stage] + sum_i weight_i * transform_i(x_i)
#   noul  : x in [0,1], contribution = w * x
#   score : x = score/(levels-1) in [0,1], contribution = w * (x - 0.5) * 2   (centered: middle level is neutral)
#   code  : x computed from the transcript in [0,1], contribution = w * x
#   fact  : like noul but once >= FACT_PERSIST_THRESHOLD it is persisted at 1.0 for the rest of the call
# p_t = EMA_ALPHA * inst + (1 - EMA_ALPHA) * p_{t-1}
# ---------------------------------------------------------------------------
SIGNAL_WEIGHTS: dict[str, dict[str, Any]] = {
    # positive, transient (prospect turns)
    "commitment": {"kind": "noul", "weight": 0.15, "label": "Prospect se comprometeu"},
    "buying_signal": {"kind": "noul", "weight": 0.10, "label": "Sinal de compra"},
    "prospect_accepts": {"kind": "noul", "weight": 0.03, "label": "Prospect concorda com o vendedor"},
    # positive, durable facts
    "next_step_agreed": {"kind": "fact", "weight": 0.09, "label": "Próximo passo combinado"},
    "pain_identified": {"kind": "fact", "weight": 0.05, "label": "Dor identificada"},
    "pain_quantified": {"kind": "fact", "weight": 0.03, "label": "Impacto em números"},
    "budget_discussed": {"kind": "fact", "weight": 0.04, "label": "Preço/orçamento discutido"},
    "decision_maker_identified": {"kind": "fact", "weight": 0.04, "label": "Decisor identificado"},
    "timeline_known": {"kind": "fact", "weight": 0.04, "label": "Prazo conhecido"},
    # relationship scores (centered: middle level is neutral)
    "engagement": {"kind": "score", "weight": 0.08, "label": "Engajamento do comprador"},
    "rapport": {"kind": "score", "weight": 0.04, "label": "Rapport"},
    "urgency": {"kind": "score", "weight": 0.05, "label": "Urgência"},
    # negative, transient
    "objection_open": {"kind": "code", "weight": -0.14, "label": "Objeção em aberto"},
    "prospect_disengaging": {"kind": "noul", "weight": -0.16, "label": "Prospect se desengajando"},
    "buyer_confused": {"kind": "noul", "weight": -0.05, "label": "Comprador confuso"},
    "competitor_mentioned": {"kind": "noul", "weight": -0.04, "label": "Concorrente mencionado"},
    "rep_pitching": {"kind": "noul", "weight": -0.04, "label": "Vendedor despejando funcionalidades"},
    "rep_talking_too_much": {"kind": "code", "weight": -0.10, "label": "Vendedor falando demais"},
    # positive rep behaviour
    "rep_discovery_question": {"kind": "noul", "weight": 0.03, "label": "Vendedor fez pergunta de descoberta"},
}

EMA_ALPHA = 0.40  # weight of the newest instantaneous estimate
P_MIN, P_MAX = 0.03, 0.95
INITIAL_PROBABILITY = 0.30

# ---------------------------------------------------------------------------
# Thresholds (confidence gating)
# ---------------------------------------------------------------------------
FACT_PERSIST_THRESHOLD = 0.70  # noul >= this persists the fact for the rest of the call
OBJECTION_OPEN_THRESHOLD = 0.60  # prospect_objecting >= this opens an objection
OBJECTION_CLEAR_THRESHOLD = 0.60  # prospect_accepts / buying_signal / commitment >= this clears it
OBJECTION_MAX_AGE_UTTERANCES = 8  # an objection nobody returns to expires after this many utterances
OBJECTION_TYPE_MIN_CONFIDENCE = 0.40
NEXT_MOVE_MIN_CONFIDENCE = 0.35  # below this the coaching card shows "listening..."
PHRASING_MIN_CONFIDENCE = 0.30
SIGNAL_ON_THRESHOLD = 0.60  # UI: a noul >= this lights up as "on"
TALK_RATIO_WARN = 0.65  # rep share of words above which "talking too much" starts
TALK_RATIO_MAX = 0.90  # ...and reaches full penalty
SENSITIVITY_TOP_N = 3

# ---------------------------------------------------------------------------
# Optional: tailored phrasing from a small generative LLM (off the critical path)
#
# Jev still decides *which* move and gates it on confidence. Only then, and only when the move
# changed (or a new fact locked in), a cheap fast LLM rewrites the chosen playbook line for this
# specific conversation. The generic line is shown immediately; the tailored one arrives ~1 s later.
# Jev then judges the candidate sentence and code drops it if it invents a fact.
# ---------------------------------------------------------------------------
LLM_API_KEY_ENV_VARS = ("ANTHROPIC_API_KEY",)
LLM_MODEL = "claude-haiku-4-5"  # Anthropic's fastest/cheapest; alias of claude-haiku-4-5-20251001
LLM_ENDPOINT = "https://api.anthropic.com/v1/messages"
LLM_TIMEOUT_S = 3.0  # past this, keep the generic line
LLM_MAX_OUTPUT_TOKENS = 140  # a 50-word sentence is ~70 tokens; output that hits this limit is discarded as truncated
LLM_PRICE_PER_M_INPUT_TOKENS_USD = 1.0
LLM_PRICE_PER_M_OUTPUT_TOKENS_USD = 5.0
PERSONALIZE_CONTEXT_UTTERANCES = 6  # transcript window sent to the LLM
PERSONALIZE_COOLDOWN_UTTERANCES = 2  # never rewrite twice within this many utterances
PERSONALIZE_REFRESH_UTTERANCES = 8  # same move for this long -> allow one refresh
PERSONALIZE_MAX_WORDS = 32  # default cap for a spoken sentence
PERSONALIZE_MAX_WORDS_BY_MOVE = {"summarize_and_check_understanding": 55, "state_price_with_context": 40, "handle_price_objection_roi": 40}
PERSONALIZE_VERIFY_WITH_JEV = True
PERSONALIZE_INVENTED_FACT_THRESHOLD = 0.5  # Jev noul >= this -> discard the tailored line
# Moves whose generic line is already fine are not rewritten (saves calls; the LLM adds little there).
PERSONALIZE_SKIP_MOVES = frozenset({"dig_into_the_problem", "stop_talking_ask_open_question", "clarify_simply"})

VERIFY_QUESTIONS: dict[str, dict[str, Any]] = {
    "invents_fact": _noul(
        "Does `candidate_line`, a sentence the rep is about to say, state a number, price, name, deadline, product feature, "
        "or commitment that does not appear anywhere in `recent_transcript` or `known_facts`?",
        true="The line quotes a figure, a date, a person, a feature, or a promise that nobody said on the call.",
        false="Every specific detail in the line was said on the call, or the line contains no specific details (a plain open question).",
    ),
    "on_move": _noul(
        "Is `candidate_line` a reasonable way for the rep to carry out the coaching move described in `move`?",
        true="The line does what `move.what` describes, in the rep's voice, as one natural sentence.",
        false="The line does something else, addresses the wrong concern, or is not something a rep would say aloud.",
    ),
}
