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
    if facts.get("familiarity") in ("none", "curious", "user", "expert"):
        p["familiarity"] = facts["familiarity"]
    for k in ("wants_follow_up", "wants_demo"):
        if facts.get(k) is not None:
            p[k] = bool(facts[k])
    for k in ("interests", "enjoyed"):
        p[k] = _clean_topics(list(p.get(k) or []) + list(facts.get(k) or []))
    p["lead_score"], p["lead_stage"] = lead_score(p)
    return p


def greeting(person: dict | None, name: str = "", robot: str = "PuppyPi") -> str:
    """What the robot says when it notices someone. `person` is a remembered visitor, `name` only what a badge says."""
    if person:
        who = first_name(person.get("name", ""))
        bits = f"Welcome back, {who}!" + (f" Great to see you again from {person['company']}." if person.get("company") else " Great to see you again.")
        topic = (person.get("interests") or person.get("enjoyed") or [""])[0]
        if topic:
            bits += f" Last time we talked about {topic}."
        return bits + " What shall we explore today?"
    if name:
        return f"Hello {first_name(name)}! I'm {robot}, the Graph Wizard. Ask me anything about Neo4j, GenAI or GraphRAG."
    return f"Hi there! I'm {robot}, the Graph Wizard. Ask me anything about Neo4j, GenAI or GraphRAG, or tell me your name."


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
    lock: threading.Lock = field(default_factory=threading.Lock)


class PersonView(BaseModel):
    zone: Literal["left", "center", "right"] = Field(description="third of the frame the person is in")
    facing_robot: bool = Field(description="turned toward the robot: face, body or feet point at the camera, or they look down at it")
    close: bool = Field(description="within about two metres of the robot; from floor level their legs and feet look large")
    badge_name: str = Field(description="name printed on a clearly legible conference badge they wear, or empty")
    badge_company: str = Field(description="company printed on that badge, or empty")


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
                 gesture_fn: Callable[[str], dict] | None = None):
        self.s, self.mem, self.agent, self.bus, self.client = s, mem, agent, bus, client
        self.outbox = Outbox()
        self.encounters: dict[str, Encounter] = {}
        self._lock = threading.Lock()
        self._greeted: dict[str, float] = {}
        self._greeting = threading.Lock()
        self._recent_greeting: tuple[float, str, dict | None] | None = None  # (when, text, badge) for the next conversation to pick up
        self._greet_times: list[float] = []
        self._skip_note = -1e9  # when the UI was last told why a person in view was not greeted
        self.look_fn = look_fn if look_fn is not None else (self._look if client is not None else None)
        self.gesture_fn = gesture_fn  # set by Brain: plays a trick (booth mode only; rate-limited in drive.py)

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
            lines.append("- Consent to remember them: NOT ASKED YET. If they have told you their name or anything about themselves "
                         "(in this message or before), END this reply with the permission question.")
        elif enc.consent is None:
            lines.append("- Consent to remember them: you asked and they have not answered yet. If this message is their yes or no, call "
                         "record_consent. Otherwise answer them and do not ask again in this reply.")
        else:
            lines.append({True: "- Consent to remember them: granted. Save what they told you with save_person_facts.",
                          False: "- Consent to remember them: they said NO. Save nothing."}[enc.consent])
        if enc.name:
            lines.append(f"- Their name: {enc.name}.")
        if enc.greeted:
            lines.append(f"- You already greeted them out loud with: \"{enc.greeted}\" Do not greet again; carry on from there.")
        if enc.badge and enc.badge.get("name"):
            lines.append(f"- Their name badge seemed to read \"{enc.badge['name']}\"" + (f", {enc.badge['company']}" if enc.badge.get("company") else "")
                         + " (unverified). Confirm it with them before using it.")
        if enc.person_id:
            p = self.mem.get_person(enc.person_id)
            if p:
                lines.append(f"- A returning visitor you remember: {json.dumps(safe_view(p))}")
        return "\n".join(lines)

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
            enc.person_id, enc.consent = None, False
            self._publish("people", {"event": "forgotten", "person_id": pid})
            return {"forgotten": ok}
        if name == "save_person_facts":
            if enc.consent is not True:
                return {"error": "Not saved: the visitor has not given consent. Ask for permission first."}
            who = (a.get("name") or enc.name or "").strip()
            if not name_key(who):
                return {"error": "Not saved: I need their name first."}
            existing = self.mem.get_person(enc.person_id) if enc.person_id else None
            if existing is None:
                existing = match_person(self.mem.find_people(name_key(who)), a.get("company"))
            person = merge_person(existing, a, who)
            enc.person_id, enc.name = person["person_id"], person["name"]
            enc.summary = (a.get("encounter_summary") or enc.summary or "")[:500]
            saved = self.mem.save_person(person, encounter_id=enc.id, place=enc.place, summary=enc.summary,
                                         ts=datetime.now(timezone.utc).isoformat())
            self._publish("people", {"event": "saved", "person_id": saved["person_id"], "name": saved["name"], "stage": saved["lead_stage"]})
            return {"saved": True, "name": saved["name"], "remembered": [k for k in a if k != "encounter_summary" and a[k] not in (None, "", [])]}
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
            enc.messages.clear()
            return {"new_conversation": True, "note": "Treat them as someone new: ask their name, and ask for consent before saving anything."}
        if name == "ask_world":
            if not self.agent:
                return {"error": "I cannot look around right now."}
            r = self.agent.ask(a["question"], session_id=f"booth-{enc.id}")
            return {"answer": r["answer"]}
        return {"error": f"unknown tool {name}"}

    # ---- one conversational turn
    def turn(self, text: str, encounter_id: str | None = None, place: str | None = None) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not set")
        enc = self._encounter(encounter_id, place)
        with enc.lock:
            sysmsg = system_prompt(self.s.robot_name, self.s.event_name, self._state(enc))
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
                        msgs = [{"role": "system", "content": system_prompt(self.s.robot_name, self.s.event_name, self._state(enc))},
                                *msgs[1 + len(history):]]
                        history = []
            else:
                reply = "Sorry, my circuits got tangled on that one. Could you ask me again?"
            enc.messages += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            if asks_consent(reply):
                enc.consent_asked = True
            enc.last = time.monotonic()
        person = self.mem.get_person(enc.person_id) if enc.person_id else None
        out = {"encounter_id": enc.id, "reply": reply, "consent": enc.consent, "person": person, "evidence": trace}
        self._publish("converse", {"encounter_id": enc.id, "heard": text if enc.consent is not False else "", "reply": reply,
                                   "tools": [t["tool"] for t in trace], "consent": enc.consent})
        return out

    # ---- greeting when the camera sees someone
    def on_view(self, scene, jpeg: bytes | None) -> bool:
        """Called after each analysed frame. Cheap checks first (a person in view, rate limits, nobody already talking to the robot);
        only then one vision call to see WHO is there (look_fn) and the crowd policy (choose_greeting) picks whom to greet.
        Runs in the background. Returns True if a greeting attempt was started."""
        s = self.s
        if not s.people_greet or not any(o.label == "person" and o.confidence >= s.min_confidence for o in scene.objects):
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
            text = group_greeting(self.s.robot_name) if pick["kind"] == "group" else greeting(pick["person"], pick["name"], self.s.robot_name)
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


def group_greeting(robot: str = "PuppyPi") -> str:
    return f"Hello everyone! I'm {robot}, the Graph Wizard. Ask me anything about Neo4j, GenAI or GraphRAG. One at a time, please!"


def _redact(tool: str, args: dict) -> dict:
    """The evidence shown in the UI never repeats an email address."""
    return {k: ("***" if k == "email" and v else v) for k, v in args.items()} if tool == "save_person_facts" else args
