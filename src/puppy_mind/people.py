"""People at the booth: greet them, talk with them, and (only after a spoken yes) remember them as leads.

Three layers, deliberately separate:
  * pure helpers  - name matching, consent detection, lead scoring, greetings (unit-tested, no network)
  * PeopleService - the conversation: an LLM with a few tools, run per `Encounter`, plus the greeting trigger and the speech outbox
  * people_graph  - where the facts end up (Aura, or RAM in dev)

Privacy rules enforced IN CODE, not only in the prompt:
  * save_person_facts refuses unless the visitor's own last words were a clear yes (consent_from), whatever the model decides;
  * raw conversation text is never stored: only the extracted facts and a one-sentence summary per encounter;
  * recognition is by what the person says or by a name badge they wear (read only when PEOPLE_CONSENT=true), never by face;
  * forget_me erases the person and their encounters.
"""
from __future__ import annotations

import json
import re
import queue
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .config import Settings
from .wizard import system_prompt

# ------------------------------------------------------------------ pure helpers

_AFFIRM = re.compile(r"\b(yes|yeah|yep|yup|sure|ok|okay|of course|please do|go ahead|absolutely|definitely|why not|sounds good|that'?s fine|i agree)\b", re.I)
_NEGATE = re.compile(r"\b(no|nope|nah|don'?t|do not|not really|rather not|never|stop|negative)\b", re.I)


def consent_from(text: str) -> bool | None:
    """True / False when the visitor clearly said yes / no, None when unclear or mixed (then the robot must ask again)."""
    yes, no = bool(_AFFIRM.search(text or "")), bool(_NEGATE.search(text or ""))
    if yes == no:
        return None
    return yes


_ASK = re.compile(r"\b(may|can|could|shall)\s+i\s+(remember|save|keep|store|note)\b", re.I)


_INTRO = re.compile(r"\b(my name is|my name's|call me|i work|i'm working|i am working|i'm from|i am from|i'm with|i am with|"
                    r"i'm (a|an|the) |i am (a|an|the) |i'm in |i am in )", re.I)
_IM_NAME = re.compile(r"\b(?i:i'm|i am|this is) ([A-Z][a-z]+)")  # "I'm Ana" yes, "I'm fine" no: only the name must be capitalised
CONSENT_QUESTION = ("May I remember your name, company and interests so the Neo4j team can follow up with you? Just say yes or no, "
                    "and if I misheard your name, please spell it.")
UNKNOWN_FIELDS = (  # what a good lead record needs, in the order the robot asks about it (one per answer)
    ("company", "where they work"), ("role", "their role"), ("use_case", "what they are building or working on"),
    ("familiarity", "how well they know Neo4j"), ("enjoyed", "what they enjoyed most at the event"),
    ("wants_follow_up", "whether they would like the Neo4j team to follow up"))
QUESTIONS = {  # the exact question that ends the robot's reply, one unknown fact at a time (the model picked its own and skipped company)
    "name": "What's your name, by the way?",
    "company": "Which company are you with?",
    "role": "And what is your role there?",
    "use_case": "What are you working on, or hoping to build with graphs?",
    "familiarity": "How familiar are you with Neo4j so far?",
    "enjoyed": "What have you enjoyed most at {event} so far?",
    "wants_follow_up": "Would you like the Neo4j team to follow up with you after the event?",
}
_BYE = re.compile(r"\b(bye|goodbye|have to (go|run|leave)|need to (go|run|leave)|gotta (go|run)|got to (go|run)|see you|"
                  r"that'?s (it|all)( for now)?|that is (it|all)|next (one|session|talk)|let'?s (close|stop|end) (it|here)|i'?m off)\b", re.I)
CONSENT_AFTER_TURNS = 6  # ask at the end; at the latest after this many answers (a visitor may walk off without a goodbye)
ASK_ORDER = ("name", "company", "role", "use_case", "familiarity", "enjoyed")  # the follow-up wish is the consent question itself


def consent_question(name: str = "", leaving: bool = True) -> str:
    """Asked at the end of the conversation (or mid-way after a while): remembering the visitor is for sending material and following up."""
    who = f"you as {name}" if name else "your name"
    tail = ", and spell your name if I misheard it." if name else ". If yes, just tell me your name."
    lead = "Before you go" if leaving else "By the way"
    return (f"{lead}: may I remember {who} and what we talked about, so the Neo4j team can send you material and follow up? "
            f"Just say yes or no{tail}")


def says_goodbye(text: str) -> bool:
    return bool(_BYE.search(text or ""))


def end_with_question(reply: str, question: str) -> str:
    """Make `question` the reply's last sentence, replacing a question the model ended with itself (one question per answer)."""
    reply = (reply or "").strip()
    if question.lower() in reply.lower():
        return reply
    parts = re.split(r"(?<=[.!?])\s+", reply)
    if parts and parts[-1].endswith("?"):
        parts = parts[:-1]
    return (" ".join(parts) + " " + question).strip()


_WISH = {
    "wants_demo": re.compile(r"\b(demo|demonstration|show (me|us)|walk (me|us) through|see it (live|in action)|a trial)\b", re.I),
    "wants_follow_up": re.compile(r"\b(follow[- ]?up|contact (me|us)|reach (out|me|us)|get in touch|stay in touch|send me|email me|"
                                  r"call me|connect me|talk to (your|the|a) (team|sales|someone)|keep me posted)\b", re.I),
}
_NOT = re.compile(r"\b(no|not|don'?t|do not|never|rather not|no need|no thanks)\b", re.I)


def explicit_wishes(text: str) -> dict:
    """wants_demo / wants_follow_up only from the visitor's own words ("I'd love a demo" -> True, "no demo, thanks" -> False).
    Seen at the booth: a demo wish appeared that nobody voiced, then flipped to no on a goodbye; neither may come from the model."""
    out = {}
    for key, rx in _WISH.items():
        m = rx.search(text or "")
        if m:
            before = (text or "")[max(0, m.start() - 25):m.start()]
            out[key] = not _NOT.search(before)
    return out


_CONFIRM = re.compile(r"\b(yes|yeah|yep|correct|exactly|right|that'?s (right|correct)|perfect|spot on)\b", re.I)  # not "that's it": a goodbye


def confirms(text: str) -> bool:
    """The visitor confirmed what the robot just read back (an email)."""
    return bool(_CONFIRM.search(text or "")) and consent_from(text) is not False


# "Hey PuppyPi" as speech-to-text writes it: "Hey Puppy Pi", "hey puppy pie", "Hi PuppyPi", "Okay poppy pi", "Hey, Puppy!"
_HEY_PUPPY = re.compile(r"\b(hey|hi|hello|hallo|ok(ay)?|yo)\b\W*(puppy|poppy|puppi|pappy|papi|puppie)\W*(pi|pie|py|pye|pai|pee)?\b", re.I)


def has_wake_word(text: str, wake: str = "Hey PuppyPi") -> bool:
    """The visitor addressed the robot. Fuzzy for the default "Hey PuppyPi"; any other wake word must appear as words."""
    if not (wake or "").strip():
        return True
    if name_key(wake).replace(" ", "") == "heypuppypi":
        return bool(_HEY_PUPPY.search(text or ""))
    return name_key(wake) in name_key(text)


def introduces_self(text: str) -> bool:
    """The visitor said who they are or what they do ("I'm Zaid", "I work at Neo4j")."""
    return bool(_INTRO.search(text or "") or _IM_NAME.search(text or ""))


def asks_consent(reply: str) -> bool:
    """The robot's reply asked for permission to remember the visitor."""
    return bool(_ASK.search(reply or ""))


def name_key(name: str) -> str:
    """Case, accent and punctuation-insensitive key: 'Zoe  Oneil' and 'zoë o'neil' match."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z\s-]", "", s).split())


def first_name(name: str) -> str:
    return (name or "").strip().split(" ")[0].title() if (name or "").strip() else ""


def _norm_company(c: str | None) -> str:
    return re.sub(r"\b(inc|ltd|llc|gmbh|ag|corp|corporation|co|plc|sa|bv)\b\.?", "", (c or "").lower().replace(",", " ")).strip(" .")


def match_person(candidates: list[dict], company: str | None) -> dict | None:
    """Pick the stored person a spoken/badge name refers to. Two 'John Smith's from different companies stay two people."""
    if not candidates:
        return None
    want = _norm_company(company)
    same = [c for c in candidates if want and _norm_company(c.get("company")) == want]
    if same:
        return same[0]
    open_ = [c for c in candidates if not want or not _norm_company(c.get("company"))]
    return open_[0] if len(open_) == 1 else None


CORE_TOPICS = ("neo4j", "graph", "graphrag", "rag", "genai", "knowledge graph", "aura", "cypher", "vector", "agent", "llm", "memory")
_EXEC = re.compile(r"\b(cto|cio|cdo|ceo|coo|chief|vp|vice president|svp|evp|head of|director|founder|co-?founder|owner|partner)\b", re.I)
_SENIOR = re.compile(r"\b(principal|staff|architect|lead|manager|senior|sr)\b", re.I)
_TECH = re.compile(r"\b(engineer|developer|scientist|analyst|researcher|consultant|data|ml|ai)\b", re.I)


def lead_score(p: dict) -> tuple[int, str]:
    """A transparent 0-100 score and stage (cold / warm / hot). Rules, not a model, so the sales team can see why."""
    score = 0
    role = f"{p.get('role') or ''} {p.get('function') or ''}"
    if _EXEC.search(role):
        score += 20
    elif _SENIOR.search(role):
        score += 12
    elif _TECH.search(role):
        score += 6
    if p.get("company"):
        score += 5
    topics = [t.lower() for t in (p.get("interests") or []) + (p.get("enjoyed") or [])]
    score += min(15, 5 * sum(any(c in t for c in CORE_TOPICS) for t in topics))
    if (p.get("use_case") or "").strip():
        score += 20
    if (p.get("familiarity") or "") in ("user", "expert"):
        score += 5
    if p.get("wants_demo"):
        score += 15
    if p.get("wants_follow_up"):
        score += 30
    if p.get("email"):
        score += 10
    score = min(100, score)
    return score, "hot" if score >= 60 else "warm" if score >= 30 else "cold"


def _clean_topics(items: Any) -> list[str]:
    out: list[str] = []
    for t in items or []:
        t = " ".join(str(t).lower().split())[:60]
        if t and t not in out:
            out.append(t)
    return out[:20]


def merge_person(existing: dict | None, facts: dict, name: str) -> dict:
    """Existing record + newly learned facts -> the record to store (with lead score). Only fields the visitor gave are changed."""
    p = dict(existing) if existing else {"person_id": "psn-" + uuid.uuid4().hex[:8], "interests": [], "enjoyed": []}
    p["name"], p["name_key"] = name.strip(), name_key(name)
    for k in ("role", "function", "company", "email", "use_case", "notes"):
        v = (facts.get(k) or "").strip()
        if v:
            p[k] = v[:300]
    if (facts.get("next_step") or "").strip() and p.get("next_step_by") != "team":  # never overwrite what the team wrote
        p["next_step"], p["next_step_by"] = facts["next_step"].strip()[:500], "ai"
    if facts.get("familiarity") in ("none", "curious", "user", "expert"):
        p["familiarity"] = facts["familiarity"]
    for k in ("wants_follow_up", "wants_demo"):
        if facts.get(k) is not None:
            p[k] = bool(facts[k])
    for k in ("interests", "enjoyed"):
        p[k] = _clean_topics(list(p.get(k) or []) + list(facts.get(k) or []))
    p["lead_score"], p["lead_stage"] = lead_score(p)
    return p


def greeting(person: dict | None, name: str = "", robot: str = "PuppyPi", wake: str = "") -> str:
    """What the robot says when it notices someone. `person` is a remembered visitor, `name` only what a badge says. With a wake
    word, a stranger is told how to start talking (a remembered visitor can just answer: the greeting opens the conversation)."""
    if person:
        who = first_name(person.get("name", ""))
        bits = f"Welcome back, {who}!" + (f" Great to see you again from {person['company']}." if person.get("company") else " Great to see you again.")
        topic = (person.get("interests") or person.get("enjoyed") or [""])[0]
        if topic:
            bits += f" Last time we talked about {topic}."
        return bits + " What shall we explore today?"
    ask = f"Say \"{wake}\" and ask me anything about Neo4j, GenAI or GraphRAG." if wake else "Ask me anything about Neo4j, GenAI or GraphRAG."
    if name:
        return f"Hello {first_name(name)}! I'm {robot}, the Graph Wizard. {ask}"
    return f"Hi there! I'm {robot}, the Graph Wizard. {ask}" + ("" if wake else " Or tell me your name.")


def safe_view(p: dict) -> dict:
    """What the robot may say about a remembered person: no email, notes or lead data."""
    return {"first_name": first_name(p.get("name", "")), "role": p.get("role"), "company": p.get("company"),
            "interests": p.get("interests", []), "enjoyed": p.get("enjoyed", []), "times_met": p.get("meet_count", 0),
            "last_met": (p.get("last_met") or "")[:10]}


# ------------------------------------------------------------------ conversation

@dataclass
class Encounter:
    id: str
    started: float = field(default_factory=time.monotonic)
    last: float = field(default_factory=time.monotonic)
    consent: bool | None = None
    person_id: str | None = None
    name: str = ""
    badge: dict | None = None
    greeted: str = ""
    place: str | None = None
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    consent_asked: bool = False  # the robot has asked the permission question (so it does not ask on every turn)
    asked_about: list[str] = field(default_factory=list)  # unknown fields the robot already asked about: never the same question twice
    pending_email: str = ""  # an email the robot heard but the visitor has not confirmed yet
    record: dict | None = None  # the visitor's lead record during the conversation: in memory only until they say yes
    turns: int = 0          # answers in this conversation
    closing: bool = False   # the consent question came with their goodbye: no more questions after it
    reasked: bool = False   # asked mid-way, unanswered: asked once more when they say goodbye
    unclear_at: float = 0.0  # when the robot last said it did not catch something (at most every UNCLEAR_EVERY_S)
    lock: threading.Lock = field(default_factory=threading.Lock)


class PersonView(BaseModel):
    zone: Literal["left", "center", "right"] = Field(description="third of the frame the person is in")
    facing_robot: bool = Field(description="turned toward the robot: face, body or feet point at the camera, or they look down at it")
    close: bool = Field(description="within about two metres of the robot; from floor level their legs and feet look large")
    badge_name: str = Field(description="name printed on a clearly legible conference badge they wear, or empty")
    badge_company: str = Field(description="company printed on that badge, or empty")


class Facts(BaseModel):
    """What one visitor said about themselves in a stretch of conversation (empty when not said). Used after consent."""
    name: str = Field(description="the visitor's name as they said it, else empty")
    corrected_name: str = Field(description="their name if they corrected or spelled it (spelled letters joined, Z-A-I-D = Zaid), else empty")
    role: str = Field(description="job title they said, else empty")
    function: str = Field(description="area they work in (data engineering, security, product...), else empty")
    company: str = Field(description="company they said they work for, else empty")
    use_case: str = Field(description="what they are building, evaluating or working on, else empty")
    familiarity: Literal["", "none", "curious", "user", "expert"] = Field(description="how well they know Neo4j, only if they said it")
    interests: list[str] = Field(description="topics they want to learn about or asked about (GraphRAG, agent memory...), short phrases")
    enjoyed: list[str] = Field(description="what they enjoyed at the event, short phrases")
    wants_follow_up: Literal["", "yes", "no"] = Field(description="yes/no only if they said whether they want a follow-up")
    wants_demo: Literal["", "yes", "no"] = Field(description="yes/no only if they said whether they want a demo")
    conversation_summary: str = Field(description="one sentence for the sales team: who they are and what the whole conversation was about")
    next_step: str = Field(description="ONE concrete next action for the Neo4j team, based only on what they said they need, e.g. 'Send "
                                       "the aerospace knowledge-graph case study and offer a 30-min call on traceability use cases'")
    email: str = Field(description="an email address the Visitor said, written normally (andreas.baum@maxbaum.com), else empty")


EXTRACT_PROMPT = """Extract what a conference visitor said about THEMSELVES from the whole conversation below (speech-to-text, may be
imperfect). Only facts the Visitor actually stated or clearly implied: never guess, never take facts from the Robot's lines. Topics they
asked the robot about count as interests. If they spelled or corrected their name, put the corrected name in corrected_name. Leave
everything else empty. conversation_summary: one sentence for the sales team about who they are and what they wanted, covering the
whole conversation, not only its start."""


def facts_to_update(f: dict) -> dict:
    """Facts (extraction output) -> the keys merge_person understands; empty values dropped."""
    out = {k: (f.get(k) or "").strip() for k in ("role", "function", "company", "use_case") if (f.get(k) or "").strip()}
    if f.get("familiarity"):
        out["familiarity"] = f["familiarity"]
    for k in ("interests", "enjoyed"):
        if f.get(k):
            out[k] = [t for t in f[k] if str(t).strip()]
    for k in ("wants_follow_up", "wants_demo"):
        if f.get(k) in ("yes", "no"):
            out[k] = f[k] == "yes"
    return out


def unknown_fields(p: dict) -> list[str]:
    """What the robot still does not know about a remembered visitor, in asking order."""
    return [label for key, label in UNKNOWN_FIELDS if not p.get(key) and p.get(key) is not False]


class Crowd(BaseModel):
    people: list[PersonView]


class Outbox:
    """Things the robot should say out loud (greetings). The robot's voice client polls it."""

    def __init__(self, maxage_s: float = 45.0):
        self._items: list[dict] = []
        self._lock = threading.Lock()
        self.maxage_s = maxage_s

    def say(self, text: str, kind: str = "greeting") -> dict:
        item = {"id": uuid.uuid4().hex[:8], "text": text, "kind": kind, "t": time.monotonic()}
        with self._lock:
            self._items.append(item)
            del self._items[:-10]
        return item

    def next(self) -> dict | None:
        """Oldest item that is still worth saying (a greeting a minute late is just odd)."""
        with self._lock:
            while self._items:
                it = self._items.pop(0)
                if time.monotonic() - it["t"] <= self.maxage_s:
                    return it
        return None


TOOLS = [
    {"type": "function", "function": {"name": "record_consent",
     "description": "Record whether the visitor allowed you to remember their details. Only after you asked and they answered yes or no.",
     "parameters": {"type": "object", "properties": {"granted": {"type": "boolean"}}, "required": ["granted"]}}},
    {"type": "function", "function": {"name": "save_person_facts",
     "description": "Remember facts the visitor told you (needs consent). Only what they said; omit anything unknown.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"}, "role": {"type": "string", "description": "job title"},
         "function": {"type": "string", "description": "area they work in, e.g. data engineering, security, product"},
         "company": {"type": "string"}, "email": {"type": "string", "description": "only after they confirmed it"},
         "use_case": {"type": "string", "description": "what they are building or evaluating"},
         "familiarity": {"type": "string", "enum": ["none", "curious", "user", "expert"], "description": "how well they know Neo4j"},
         "interests": {"type": "array", "items": {"type": "string"}, "description": "topics they want to learn about or work with"},
         "enjoyed": {"type": "array", "items": {"type": "string"}, "description": "what they enjoyed most at the event"},
         "wants_follow_up": {"type": "boolean"}, "wants_demo": {"type": "boolean"},
         "notes": {"type": "string"}, "encounter_summary": {"type": "string", "description": "one sentence on this conversation"}}}}},
    {"type": "function", "function": {"name": "identify_person",
     "description": "Look up a returning visitor by the name they gave, to welcome them back.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "company": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "forget_me",
     "description": "Erase everything remembered about this visitor, when they ask to be forgotten.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "do_trick",
     "description": "Perform a trick when a visitor asks for one (boxing = stands on its hind legs and pumps its front paws, wave, shake_hands, bow, nod).",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": ["boxing", "wave", "shake_hands", "bow", "nod"]}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "new_visitor",
     "description": "A DIFFERENT person has started talking to you (they give another name, or say it is their turn). Call this FIRST, so nothing "
                    "about the previous visitor is mixed up with them. Consent must then be asked again.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "ask_world",
     "description": "Ask your own memory about the room: what you can see, where things are, what changed. For any question about your surroundings.",
     "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
]


class PeopleService:
    def __init__(self, s: Settings, mem, *, client=None, agent=None, bus=None, look_fn: Callable[[bytes], list[dict]] | None = None,
                 gesture_fn: Callable[[str], dict] | None = None, extract_fn: Callable[[str], dict] | None = None):
        self.s, self.mem, self.agent, self.bus, self.client = s, mem, agent, bus, client
        self.outbox = Outbox()
        self.encounters: dict[str, Encounter] = {}
        self._lock = threading.Lock()
        self._greeted: dict[str, float] = {}
        self._greeting = threading.Lock()
        self._recent_greeting: tuple[float, str, dict | None] | None = None  # (when, text, badge) for the next conversation to pick up
        self._greet_times: list[float] = []
        self._skip_note = -1e9  # when the UI was last told why a person in view was not greeted
        self.active_fn: Callable[[], bool] = lambda: True  # set by Brain: greet only while someone has the People tab open
        self.look_fn = look_fn if look_fn is not None else (self._look if client is not None else None)
        self.gesture_fn = gesture_fn  # set by Brain: plays a trick (booth mode only; rate-limited in drive.py)
        # after each answer with consent, facts are pulled from what the visitor said (the model alone saved only at the moment of consent)
        self.extract_fn = extract_fn if extract_fn is not None else (self._extract if client is not None else None)
        self.learn_in_background = True  # tests set False to learn synchronously
        # Saves go to the graph in the background, in order: the visitor never waits for the database (at the booth each Aura round
        # trip took ~1 s and a 'yes' turn made 17 of them: 22 s, past the robot's 60 s once the camera was busy too)
        self.write_in_background = True
        self._writes: queue.Queue = queue.Queue()
        self._writer: threading.Thread | None = None

    # ---- plumbing
    def _publish(self, kind: str, data: dict) -> None:
        if self.bus:
            self.bus.publish(kind, data)

    def _encounter(self, encounter_id: str | None, place: str | None) -> Encounter:
        now = time.monotonic()
        with self._lock:
            for k in [k for k, e in self.encounters.items() if now - e.last > self.s.encounter_idle_s]:
                del self.encounters[k]
            enc = self.encounters.get(encounter_id or "")
            if enc is None:
                enc = Encounter(id=encounter_id or uuid.uuid4().hex[:10], place=place)
                self.encounters[enc.id] = enc
                if self._recent_greeting and now - self._recent_greeting[0] < 120:  # they were just greeted out loud: continue from that
                    _, enc.greeted, enc.badge = self._recent_greeting
            enc.last = now
            if place:
                enc.place = place
            return enc

    def _state(self, enc: Encounter) -> str:
        lines = ["", "STATE OF THIS CONVERSATION"]
        if enc.consent is None and not enc.consent_asked:
            lines.append("- Do NOT ask for permission to remember them: that question is asked automatically at the end of the conversation.")
        elif enc.consent is None:
            lines.append("- You asked whether you may remember them; they have not answered yet. Do not ask again yourself.")
        elif enc.consent is True:
            lines.append("- They said YES to being remembered and followed up: thank them" + (
                ", then say a warm goodbye (they are leaving: no questions)." if enc.closing else "."))
        else:
            lines.append("- They said NO to being remembered: save nothing" + (", and say a friendly goodbye." if enc.closing else "."))
        if enc.consent is not False and not enc.closing:
            lines.append("- Do NOT end with a question of your own about them: one question about them is added after your answer.")
        if enc.name:
            lines.append(f"- Their name: {enc.name}.")
        if enc.greeted:
            lines.append(f"- You already greeted them out loud with: \"{enc.greeted}\" Do not greet again; carry on from there.")
        if enc.badge and enc.badge.get("name"):
            lines.append(f"- Their name badge seemed to read \"{enc.badge['name']}\"" + (f", {enc.badge['company']}" if enc.badge.get("company") else "")
                         + " (unverified). Confirm it with them before using it.")
        if enc.person_id and enc.consent is True:
            p = self._record(enc)
            if p:
                lines.append(f"- A visitor you remember: {json.dumps(safe_view(p))}")
        return "\n".join(lines)

    # ---- the visitor's record: read once, kept in memory, written behind
    def _record(self, enc: Encounter) -> dict:
        if enc.record is None and enc.person_id:
            enc.record = self.mem.get_person(enc.person_id) or None
        return enc.record or {}

    def _store(self, enc: Encounter, person: dict) -> dict:
        """Keep the record for the conversation; save it to the graph only after a yes and with a name (in the background unless
        write_in_background is off). Before the yes it is a draft in memory, discarded on a no."""
        enc.record = person
        if enc.consent is not True or not (person.get("name") or "").strip():
            return person
        enc.person_id = person["person_id"]
        job = (person, enc.id, enc.place, enc.summary, datetime.now(timezone.utc).isoformat())
        if not self.write_in_background:
            self._write_one(enc, *job)
            return enc.record
        self._writes.put((enc, job))
        if self._writer is None or not self._writer.is_alive():
            self._writer = threading.Thread(target=self._write_loop, daemon=True, name="people-writer")
            self._writer.start()
        return person

    def _write_loop(self) -> None:
        while True:
            enc, job = self._writes.get()
            self._write_one(enc, *job)

    def _write_one(self, enc: Encounter, person: dict, encounter_id: str, place: str | None, summary: str, ts: str) -> None:
        try:
            saved = self.mem.save_person(person, encounter_id=encounter_id, place=place, summary=summary, ts=ts)
            if enc.record is not None and enc.record.get("person_id") == person["person_id"]:
                for k in ("meet_count", "first_met", "last_met", "followed_up"):
                    if saved.get(k) is not None:
                        enc.record[k] = saved[k]
        except Exception as e:
            self._publish("error", {"error": f"could not save {person.get('name')} to the graph: {e}"[:200]})

    # ---- edits made in the People tab while the visitor may still be talking: open conversations must see them at once
    def _open_records(self, person_id: str) -> list[Encounter]:
        return [e for e in list(self.encounters.values()) if e.person_id == person_id]

    def set_next_step(self, person_id: str, text: str) -> bool:
        """The team's next step: saved, and put into any open conversation's record, so the next save cannot undo it."""
        self.flush()
        ok = self.mem.set_next_step(person_id, text)
        for e in self._open_records(person_id):
            if e.record is not None:
                e.record["next_step"], e.record["next_step_by"] = text.strip()[:500], "team"
        return ok

    def set_followed_up(self, person_id: str, done: bool) -> bool:
        self.flush()
        ok = self.mem.set_followed_up(person_id, done)
        for e in self._open_records(person_id):
            if e.record is not None:
                e.record["followed_up"] = done
        return ok

    def forget(self, person_id: str) -> bool:
        """Erase a person; an open conversation forgets them too, so a later save cannot bring them back."""
        for e in self._open_records(person_id):
            e.person_id, e.record, e.consent, e.name = None, None, False, ""
        self.flush()
        return self.mem.forget_person(person_id)

    def flush(self, timeout: float = 10.0) -> None:
        """Wait until queued saves reached the graph (tests, shutdown)."""
        end = time.monotonic() + timeout
        while not self._writes.empty() and time.monotonic() < end:
            time.sleep(0.05)

    # ---- tools
    def _tool(self, enc: Encounter, name: str, a: dict, last_user: str) -> Any:
        if name == "record_consent":
            heard = consent_from(last_user)
            if heard is None or heard != bool(a.get("granted")):
                return {"error": "The visitor has not clearly said yes or no in their last answer. Ask them again, plainly."}
            enc.consent = heard
            return {"consent": heard}
        if name == "identify_person":
            key = name_key(a.get("name", ""))
            p = match_person(self.mem.find_people(key), a.get("company")) if key else None
            if not p:
                return {"known": False}
            enc.person_id, enc.name, enc.consent = p["person_id"], p["name"], True  # they said yes when we first met
            enc.record = p
            return {"known": True, **safe_view(p)}
        if name == "forget_me":
            key = name_key(a.get("name") or enc.name)
            pid = enc.person_id
            if not pid and key:
                p = match_person(self.mem.find_people(key), None)
                pid = p["person_id"] if p else None
            if not pid:
                return {"forgotten": False, "reason": "no one by that name is remembered"}
            ok = self.mem.forget_person(pid)
            enc.person_id, enc.consent, enc.record = None, False, None
            self._publish("people", {"event": "forgotten", "person_id": pid})
            return {"forgotten": ok}
        if name == "save_person_facts":
            if enc.consent is not True:
                return {"error": "Not saved: the visitor has not given consent. Ask for permission first."}
            a = dict(a)
            for key in ("wants_demo", "wants_follow_up"):  # only the visitor's own words decide these, never the model
                a.pop(key, None)
            a.update(explicit_wishes(last_user))
            held = ""
            if a.get("email"):
                email = a.pop("email").strip()
                if enc.pending_email and email.lower() == enc.pending_email.lower() and confirms(last_user):
                    a["email"], enc.pending_email = email, ""
                else:  # never saved before the visitor confirmed it read back
                    enc.pending_email, held = email, "email held back: read it back letter by letter and save it again once they confirm"
            who = (a.get("name") or enc.name or "").strip()
            if not name_key(who):
                return {"error": "Not saved: I need their name first."}
            existing = self._record(enc) or None
            if existing is None:
                existing = match_person(self.mem.find_people(name_key(who)), a.get("company"))
            person = merge_person(existing, a, who)
            enc.person_id, enc.name = person["person_id"], person["name"]
            enc.summary = (a.get("encounter_summary") or enc.summary or "")[:500]
            saved = self._store(enc, person)
            self._publish("people", {"event": "saved", "person_id": saved["person_id"], "name": saved["name"], "stage": saved["lead_stage"]})
            out = {"saved": True, "name": saved["name"], "remembered": [k for k in a if k != "encounter_summary" and a[k] not in (None, "", [])]}
            if held:
                out["note"] = held
            return out
        if name == "do_trick":
            if not self.gesture_fn:
                return {"done": False, "reason": "tricks are not available"}
            return self.gesture_fn(a.get("name", ""))
        if name == "new_visitor":
            old = enc.id
            with self._lock:
                self.encounters.pop(old, None)
                enc.id = uuid.uuid4().hex[:10]
                self.encounters[enc.id] = enc
            enc.consent, enc.person_id, enc.name, enc.summary, enc.badge, enc.greeted = None, None, "", "", None, ""
            enc.consent_asked = False
            enc.asked_about, enc.pending_email, enc.record = [], "", None
            enc.messages.clear()
            return {"new_conversation": True, "note": "Treat them as someone new: ask their name, and ask for consent before saving anything."}
        if name == "ask_world":
            if not self.agent:
                return {"error": "I cannot look around right now."}
            r = self.agent.ask(a["question"], session_id=f"booth-{enc.id}")
            return {"answer": r["answer"]}
        return {"error": f"unknown tool {name}"}

    UNCLEAR_EVERY_S = 12.0

    def unclear_reply(self, encounter_id: str | None) -> str | None:
        """Something in an OPEN conversation was not understood (seen at the booth: a quiet 'yes' to the consent question came out
        as 'Thanks for watching!', was dropped, the robot said nothing and the conversation lapsed). Say so instead, at most every
        UNCLEAR_EVERY_S, and keep the conversation open. Outside a conversation, hall noise stays unanswered (None)."""
        enc = self.encounters.get(encounter_id or "")
        now = time.monotonic()
        if not enc or not enc.messages or now - enc.last >= self.s.wake_window_s or now - enc.unclear_at < self.UNCLEAR_EVERY_S:
            return None
        enc.unclear_at = enc.last = now
        if enc.consent is None and enc.consent_asked:
            return "Sorry, I didn't catch that. Was that a yes or a no?"
        return "Sorry, I didn't catch that. Could you say it again?"

    # ---- wake word (robot microphone only: Hold to talk and typing are addressed to the robot by definition)
    def conversation_active(self) -> bool:
        """Someone talked with the robot less than WAKE_WINDOW_S ago."""
        now = time.monotonic()
        return any(now - e.last < self.s.wake_window_s for e in list(self.encounters.values()) if e.messages)

    def addressed(self, text: str, encounter_id: str | None) -> tuple[bool, str | None]:
        """Should the robot answer what its microphone heard? Yes if it contains the wake word, or a conversation is open (the
        last exchange or a greeting less than WAKE_WINDOW_S ago). Returns (answer?, encounter id to use): after a closed
        conversation, a wake word starts a fresh one, since it is most likely someone new."""
        now = time.monotonic()
        enc = self.encounters.get(encounter_id or "")
        if enc and now - enc.last < self.s.wake_window_s:
            return True, enc.id
        if self._recent_greeting and now - self._recent_greeting[0] < self.s.wake_window_s and not enc:
            return True, None  # they may simply answer the greeting
        if has_wake_word(text, self.s.wake_word):
            return True, None
        return False, None

    # ---- one conversational turn
    def turn(self, text: str, encounter_id: str | None = None, place: str | None = None) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not set")
        enc = self._encounter(encounter_id, place)
        with enc.lock:
            enc.turns += 1
            if enc.consent is None and enc.consent_asked:  # the answer to "may I remember you?": decided in code, no model round trip
                heard = consent_from(text)
                if heard is not None:
                    enc.consent = heard
                    if heard is False:
                        enc.record = None  # the draft is discarded: nothing about them is kept
            consent_before = enc.consent if not enc.consent_asked or enc.consent is None else None
            learner = None
            if enc.consent is not False and self.extract_fn:  # learn from this message while the reply is prepared (in memory before a yes)
                transcript = "\n".join(("Visitor: " if m["role"] == "user" else "Robot: ") + m["content"]
                                       for m in [*enc.messages[-15:], {"role": "user", "content": text}])
                learner = threading.Thread(target=self._learn, args=(enc, transcript, text, enc.pending_email), daemon=True)
                learner.start()
            email_before = enc.pending_email  # only an email read back in an EARLIER turn can be confirmed by this message
            sysmsg = system_prompt(self.s.robot_name, self.s.event_name, self._state(enc), self.s.voice_language, self.s.booth_info)
            history = enc.messages[-12:]
            msgs: list[dict] = [{"role": "system", "content": sysmsg}, *history, {"role": "user", "content": text}]
            trace: list[dict] = []
            reply = ""
            for _ in range(6):
                r = self.client.chat.completions.create(model=self.s.chat_model, messages=msgs, tools=TOOLS, temperature=0.4)
                msg = r.choices[0].message
                if not msg.tool_calls:
                    reply = (msg.content or "").strip()
                    break
                msgs.append(msg.model_dump(exclude_none=True))
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    try:
                        res = self._tool(enc, tc.function.name, args, text)
                    except Exception as e:
                        res = {"error": str(e)[:300]}
                    trace.append({"tool": tc.function.name, "args": _redact(tc.function.name, args), "result": res})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(res, default=str)[:6000]})
                    if tc.function.name == "new_visitor":  # forget the previous visitor's conversation, in this turn too
                        msgs = [{"role": "system", "content": system_prompt(self.s.robot_name, self.s.event_name, self._state(enc),
                                                                            self.s.voice_language, self.s.booth_info)},
                                *msgs[1 + len(history):]]
                        history = []
            else:
                reply = "Sorry, my circuits got tangled on that one. Could you ask me again?"
            enc.messages += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            enc.last = time.monotonic()
        if enc.consent is not False:
            if learner is not None:  # started beside the reply: wait for it, so the question below never asks what they just said
                learner.join(timeout=8)
            if enc.consent is True and enc.record is not None and consent_before is None:  # just said yes: keep everything said so far
                enc.record = merge_person(enc.record, {"wants_follow_up": True}, enc.record.get("name") or "")
                self._store(enc, enc.record)
            question = None
            name = (self._record(enc).get("name") or "").strip()
            if enc.consent is None and not enc.consent_asked and self._consent_due(enc, text):
                enc.closing = says_goodbye(text)
                question = consent_question(name, leaving=enc.closing)
            elif enc.consent is None and enc.consent_asked and not enc.reasked and says_goodbye(text):
                enc.closing = enc.reasked = True  # asked mid-way and left unanswered: once more, now that they are leaving
                question = consent_question(name, leaving=True)
            elif not enc.closing:
                question = self._next_question(enc, text)
            if question:
                reply = end_with_question(reply, question)
                enc.messages[-1]["content"] = reply
        if asks_consent(reply):
            enc.consent_asked = True
        person = dict(self._record(enc)) if enc.person_id and enc.consent is True else None
        out = {"encounter_id": enc.id, "reply": reply, "consent": enc.consent, "person": person, "evidence": trace}
        self._publish("converse", {"encounter_id": enc.id, "heard": text if enc.consent is not False else "", "reply": reply,
                                   "tools": [t["tool"] for t in trace], "consent": enc.consent})
        return out

    def _next_question(self, enc: Encounter, text: str) -> str | None:
        """The one question that ends this reply: the first fact still unknown, never one already asked, nothing on a goodbye."""
        if says_goodbye(text):
            return None
        p = self._record(enc)
        for key in ASK_ORDER:
            if not p.get(key) and key not in enc.asked_about:
                enc.asked_about.append(key)
                return QUESTIONS[key].format(event=self.s.event_name)
        return None

    def _consent_due(self, enc: Encounter, text: str) -> bool:
        """Ask for permission at the end: when they say goodbye, when they ask for material or a follow-up, or after
        CONSENT_AFTER_TURNS answers (people walk off without a goodbye)."""
        return says_goodbye(text) or any(explicit_wishes(text).values()) or enc.turns >= CONSENT_AFTER_TURNS

    def _learn(self, enc: Encounter, transcript: str, latest: str = "", confirmable_email: str = "") -> None:
        """Add what the visitor said about themselves to their record (consent given). Never raises: a missed fact is not an error.
        Facts and the summary come from the extraction over the whole conversation; a demo/follow-up wish only from the visitor's own
        latest words; an email only once the visitor confirmed it read back."""
        try:
            if enc.consent is False:
                return
            f = (self.extract_fn(transcript) if self.extract_fn else {}) or {}
            update = facts_to_update(f)
            for key in ("wants_demo", "wants_follow_up"):
                update.pop(key, None)
            update.update(explicit_wishes(latest))
            if confirmable_email and enc.pending_email == confirmable_email and confirms(latest):
                update["email"], enc.pending_email = enc.pending_email, ""
            heard_email = (f.get("email") or "").strip().lower()
            if "@" in heard_email and not update.get("email") and not enc.pending_email:
                current = self._record(enc).get("email") if enc.person_id else None
                if heard_email != (current or "").lower():
                    enc.pending_email = heard_email  # held: saved only when they confirm it read back, in a later answer
            new_name = (f.get("corrected_name") or "").strip() or (f.get("name") or "").strip()
            summary = (f.get("conversation_summary") or "").strip()
            if (f.get("next_step") or "").strip():
                update["next_step"] = f["next_step"].strip()
            if not update and not new_name and not summary:
                return
            with enc.lock:
                if enc.consent is False:
                    return
                existing = self._record(enc) or {"person_id": "psn-" + uuid.uuid4().hex[:8], "interests": [], "enjoyed": []}
                corrected = (f.get("corrected_name") or "").strip()
                name = corrected or existing.get("name") or new_name  # a correction always wins; otherwise keep the name we have
                person = merge_person(existing, update, name or "")
                enc.name = person["name"]
                if summary:
                    enc.summary = summary[:500]
                saved = self._store(enc, person)
            if enc.consent is not True:
                return  # a draft: nothing to show until they said yes
            self._publish("people", {"event": "learned", "person_id": saved["person_id"], "name": saved["name"],
                                     "stage": saved["lead_stage"], "fields": sorted(update) + (["name"] if new_name else [])})
        except Exception as e:
            self._publish("error", {"error": f"could not learn from the conversation: {e}"[:200]})

    def _extract(self, transcript: str) -> dict:
        r = self.client.chat.completions.parse(
            model=self.s.chat_model, temperature=0, response_format=Facts,
            messages=[{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": transcript}])
        f = r.choices[0].message.parsed
        return f.model_dump() if f else {}

    # ---- greeting when the camera sees someone
    def on_view(self, scene, jpeg: bytes | None) -> bool:
        """Called after each analysed frame. Cheap checks first (a person in view, rate limits, nobody already talking to the robot);
        only then one vision call to see WHO is there (look_fn) and the crowd policy (choose_greeting) picks whom to greet.
        Runs in the background. Returns True if a greeting attempt was started."""
        s = self.s
        if not s.people_greet or not self.active_fn() or not any(o.label == "person" and o.confidence >= s.min_confidence for o in scene.objects):
            return False
        now = time.monotonic()
        self._greet_times = [t for t in self._greet_times if now - t < 3600]
        wait = s.greet_cooldown_s - (now - max(self._greeted.values(), default=-1e9))
        reason = ("already talking to someone" if any(now - e.last < 60 for e in self.encounters.values()) else
                  f"greeted someone {int(s.greet_cooldown_s - wait)} s ago (cooldown {int(s.greet_cooldown_s)} s)" if wait > 0 else
                  "hourly greeting limit reached" if len(self._greet_times) >= s.max_greetings_per_hour else
                  "still looking at the previous view" if self._greeting.locked() else "")
        if reason:
            if now - self._skip_note > 20:  # say why, but not on every frame
                self._skip_note = now
                self._publish("greeting", {"skipped": True, "reason": reason})
            return False
        if not self._greeting.acquire(blocking=False):
            return False
        threading.Thread(target=self._greet, args=(jpeg,), daemon=True).start()
        return True

    def _recently(self, key: str) -> bool:
        return time.monotonic() - self._greeted.get(key, -1e9) < self.s.person_regreet_s

    def _greet(self, jpeg: bytes | None) -> None:
        try:
            people = [{"zone": "center", "facing_robot": True, "close": True, "badge_name": "", "badge_company": ""}]  # no eyes: assume one
            if jpeg and self.look_fn:
                try:
                    people = self.look_fn(jpeg)
                except Exception as e:
                    self._publish("error", {"error": f"looking at people failed: {e}"[:200]})
                    return
            if not self.s.people_consent:  # reading worn name badges is opt-in (PEOPLE_CONSENT): drop anything read anyway
                people = [{**p, "badge_name": "", "badge_company": ""} for p in people]
            pick = choose_greeting(people, lambda k, c: match_person(self.mem.find_people(k), c), self._recently, self.s.group_min)
            if pick is None:
                engaged = sum(bool(p["facing_robot"] and p["close"]) for p in people)
                self._publish("greeting", {"skipped": True, "people": len(people), "engaged": engaged,
                                           "reason": "nobody in view" if not people else "everyone was greeted recently" if engaged else
                                           "nobody close to the robot and turned toward it"})
                return
            text = (group_greeting(self.s.robot_name, self.s.wake_word) if pick["kind"] == "group"
                    else greeting(pick["person"], pick["name"], self.s.robot_name, self.s.wake_word))
            now = time.monotonic()
            self._greeted[pick["key"]] = now
            self._greet_times.append(now)
            self._recent_greeting = (now, text, {"name": pick["name"], "company": pick["company"]} if pick["name"] else None)
            self.outbox.say(text)
            trick = self.gesture_fn(self.s.greet_action) if (self.gesture_fn and self.s.greet_action) else None
            self._publish("greeting", {"text": text, "kind": pick["kind"], "returning": pick["kind"] == "returning",
                                       "name": first_name(pick["name"]), "people": len(people), "trick": trick})
        finally:
            self._greeting.release()

    def _look(self, jpeg: bytes) -> list[dict]:
        """One vision call: who is in view, are they facing the robot and close (actually at the booth, not walking past),
        and (only with PEOPLE_CONSENT) what their name badge says. Never identifies anyone from their face."""
        from .perception import _prepare

        badge = ("If a conference name badge is clearly legible, copy the printed name and company, else leave them empty. Never guess a name."
                 if self.s.people_consent else "Leave badge_name and badge_company empty: do not read badges.")
        r = self.client.chat.completions.parse(
            model=self.s.vision_model, temperature=0, response_format=Crowd,
            messages=[{"role": "system", "content": LOOK_PROMPT + badge + " Never identify anyone from their face."},
                      {"role": "user", "content": [{"type": "text", "text": "Who is here?"},
                                                   {"type": "image_url", "image_url": {"url": _prepare(jpeg), "detail": "high"}}]}])
        c = r.choices[0].message.parsed
        return [v.model_dump() for v in (c.people if c else [])][:8]


LOOK_PROMPT = (
    "You are the eyes of a small robot dog. Its camera is at FLOOR LEVEL, about 15 cm high and tilted slightly up, so a person "
    "near it often shows only as legs and feet, maybe a lower body, and their face is usually out of frame. List each person "
    "visible, even if only their legs or feet are, at most 8, nearest first. Sitting people count. "
    "facing_robot: they are turned toward the robot, i.e. their toes, knees or body point at the camera, their face is turned to "
    "it, or they are looking down at it. Someone seen from the side or back, or walking across the view, is not facing it. "
    "close: within about two metres; from the floor their legs and feet look large, and someone in the background looks small. "
)


def choose_greeting(people: list[dict], lookup: Callable[[str, str], dict | None], recently: Callable[[str], bool],
                    group_min: int = 3) -> dict | None:
    """Whom to greet when several people are around. Pure policy (unit-tested):
      * only people who are close AND facing the robot count: passers-by in a busy hall are ignored;
      * a group (group_min or more engaged) gets ONE group hello instead of a hello each;
      * otherwise the person in front (centre) first, then a returning visitor; someone greeted recently is skipped.
    Returns {kind: group|returning|named|anon, key, name, company, person} or None."""
    engaged = [p for p in people if p.get("facing_robot") and p.get("close")]
    if not engaged:
        return None
    if len(engaged) >= group_min:
        return {"kind": "group", "key": "group", "name": "", "company": "", "person": None}
    cands = []
    for p in engaged:
        name, company = (p.get("badge_name") or "").strip(), (p.get("badge_company") or "").strip()
        key = name_key(name)
        if key and recently(key):
            continue
        person = lookup(key, company) if key else None
        cands.append((p.get("zone") != "center", person is None, {"kind": "returning" if person else "named" if key else "anon",
                                                                   "key": key or "anon", "name": name, "company": company, "person": person}))
    if not cands:
        return None
    return sorted(cands, key=lambda c: c[:2])[0][2]


def group_greeting(robot: str = "PuppyPi", wake: str = "") -> str:
    start = f"Say \"{wake}\" and ask" if wake else "Ask"
    return f"Hello everyone! I'm {robot}, the Graph Wizard. {start} me anything about Neo4j, GenAI or GraphRAG. One at a time, please!"


def _redact(tool: str, args: dict) -> dict:
    """The evidence shown in the UI never repeats an email address."""
    return {k: ("***" if k == "email" and v else v) for k, v in args.items()} if tool == "save_person_facts" else args
