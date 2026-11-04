"""The booth host: PuppyPi greets visitors, answers as the Graph Wizard, and keeps what the booth talked about, never who it talked to.

Three layers, deliberately separate:
  * pure helpers - wake word, goodbye and question detection, the scrubber, greetings (unit-tested, no network)
  * BoothHost    - the conversation: an LLM with a few tools, run per `Encounter`, plus the greeting trigger and the speech outbox
  * booth_graph  - where the anonymous conversation notes end up (Aura, or RAM in dev)

No personal data, enforced IN CODE, not only in the prompt (lead generation is on hold for compliance; it is parked on branch
parked/booth-lead-generation):
  * the robot never asks for a name, company, role or email, and a closing question of that kind from the model is removed;
  * what is kept per conversation is only topics, the need in a few words, Neo4j familiarity, what they enjoyed and their questions
    rephrased: the extraction is told to leave out anything identifying, and `scrub` then removes emails, phone numbers and any
    name or company the visitor said about themselves;
  * the conversation text itself lives in RAM only while the conversation is open (so follow-up questions work), and is never stored;
  * the camera never reads badges or identifies anyone: it only tells whether someone is close and turned toward the robot.
"""
from __future__ import annotations

import json
import queue
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
from .drive import TRICKS
from .voice import fix_terms
from .wizard import system_prompt

# ------------------------------------------------------------------ pure helpers

QUESTIONS = {  # the exact question that ends the robot's reply, one at a time, about what they need (never who they are)
    "use_case": "What are you working on, or hoping to build with graphs?",
    "familiarity": "How familiar are you with Neo4j so far?",
    "enjoyed": "What have you enjoyed most at {event} so far?",
}
ASK_ORDER = ("use_case", "familiarity", "enjoyed")
NOTE_KEY = {"use_case": "need"}  # where the answer to each question ends up in the note
ENJOYED_AFTER_TURNS = 8  # "what have you enjoyed?" is a wind-down question, not one for the middle of their questions
_BYE = re.compile(r"\b(bye|goodbye|have to (go|run|leave)|need to (go|run|leave)|gotta (go|run)|got to (go|run)|see you|"
                  r"that'?s (it|all)( for now)?|that is (it|all)|next (one|session|talk)|let'?s (close|stop|end) (it|here)|i'?m off)\b", re.I)
_QUESTION = re.compile(r"\?\s*$|^\W*(what|why|how|when|where|which|who|can|could|would|will|do|does|did|is|are|should|tell me|"
                       r"explain|show me|i'?d like to know|i want to know)\b", re.I)
# a closing question the model asked on its own about who they are, or about remembering / contacting them: never asked
_OWN_ASK = re.compile(r"\benjoy|\bconference\b|\bthe event\b|\bremember\b|\bfollow[- ]?up\b|\bstay in touch\b|\b(your|their) (name|"
                      r"company|employer|role|job|title|position|email|e-mail|contact|number|linkedin)\b|\bwho are you\b|"
                      r"\bwhere do you work\b|\bwhich company\b|\b(save|store|keep|note)\s+(your|you|this|our)\b", re.I)


_THANKS_ONLY = re.compile(r"\W*(?:(?:ok(?:ay)?|great|cool|perfect|alright|all right|awesome|nice)\W+)?(?:thanks?|thank you)"
                          r"(?:\W+(?:so|very) much)?(?:\W+(?:puppy ?pi|robot|wizard))?\W*", re.I)
_END = re.compile(r"\b(stop( talking)?|end (the |this )?conversation|that'?s enough|we'?re done|i'?m done|no more questions)\b", re.I)


def says_goodbye(text: str) -> bool:
    """The visitor is done: a goodbye, "I need to run", a plain "thanks" on its own, or "stop" / "end the conversation"."""
    t = (text or "").strip()  # "stop" ends it only as a short command: "how do I stop a running query?" is a question
    return bool(_BYE.search(t) or _THANKS_ONLY.fullmatch(t) or (_END.search(t) and len(t.split()) <= 6))


def asks_something(text: str) -> bool:
    """The visitor asked a question ("Why should I use Neo4j compared to SQL?", "Tell me more about knowledge graphs.")."""
    return bool(_QUESTION.search((text or "").strip()))


def end_with_question(reply: str, question: str) -> str:
    """Make `question` the reply's last sentence, replacing a question the model ended with itself (one question per answer)."""
    reply = (reply or "").strip()
    if question.lower() in reply.lower():
        return reply
    parts = re.split(r"(?<=[.!?])\s+", reply)
    if parts and parts[-1].endswith("?"):
        parts = parts[:-1]
    return (" ".join(parts) + " " + question).strip()


def drop_own_ask(reply: str) -> str:
    """Remove closing questions the model asked on its own about the visitor's identity, remembering them or the event: the
    code decides what is asked. (In a replay of a real booth talk the model asked "May I remember you?" three times.)"""
    parts = re.split(r"(?<=[.!?])\s+", (reply or "").strip())
    while parts and parts[-1].endswith("?") and _OWN_ASK.search(parts[-1]):
        parts = parts[:-1]
    return " ".join(parts) or "Anything else you'd like to know about graphs?"  # the whole reply was that question


# "Hey PuppyPi" as speech-to-text writes it: "Hey Puppy Pi", "hey puppy pie", "Hi PuppyPi", "Okay poppy pi", "Hey, Puppy!"
_HEY_PUPPY = re.compile(r"\b(hey|hi|hello|hallo|ok(ay)?|yo)\b\W*(puppy|poppy|puppi|pappy|papi|puppie)\W*(pi|pie|py|pye|pai|pee)?\b", re.I)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z\s-]", "", s).split())


def has_wake_word(text: str, wake: str = "Hey PuppyPi") -> bool:
    """The visitor addressed the robot. Fuzzy for the default "Hey PuppyPi"; any other wake word must appear as words."""
    if not (wake or "").strip():
        return True
    if _norm(wake).replace(" ", "") == "heypuppypi":
        return bool(_HEY_PUPPY.search(text or ""))
    return _norm(wake) in _norm(text)


# ---- the scrubber: what the visitor said about who they are never reaches what is kept
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_PHONE = re.compile(r"\+?\d[\d ()/-]{6,}\d")
_URL_ME = re.compile(r"\b(linkedin\.com|github\.com|x\.com|twitter\.com)/\S+", re.I)
_CAP = r"([A-Z][\w'&-]*(?:\s+(?:[A-Z][\w'&-]*|of|de|van|von))*)"  # a full stop ends it: "Acme. Can graphs..." is not one name
_SELF_NAME = re.compile(r"\b(?i:my name is|my name's|call me|this is|i'm|i am)\s+" + _CAP)
_SELF_ORG = re.compile(r"\b(?i:i work (?:at|for|with)|i'm (?:at|with|from)|i am (?:at|with|from)|we are|we're|our company is|"
                       r"i'm working (?:at|for)|i am working (?:at|for)|work at|works at)\s+" + _CAP)
_NOT_NAMES = {"a", "an", "the", "just", "not", "very", "really", "so", "also", "here", "looking", "working", "interested", "new",
              "curious", "trying", "building", "fine", "good", "great", "ok", "okay", "sure", "sorry", "neo4j", "i", "we"}


def identifying_words(visitor_lines: list[str]) -> set[str]:
    """Names and organisations the visitor said about THEMSELVES ("my name is Ana Berg", "I work at Acme"), as lower-case phrases."""
    out: set[str] = set()
    for line in visitor_lines:
        for rx in (_SELF_NAME, _SELF_ORG):
            for m in rx.finditer(line or ""):
                phrase = m.group(1).strip(" .,'")
                if phrase and phrase.split()[0].lower() not in _NOT_NAMES:
                    out.add(phrase.lower())
                    out |= {w.lower() for w in phrase.split() if len(w) > 2 and w.lower() not in _NOT_NAMES}
    return out


def scrub(text: str, identifying: set[str]) -> str:
    """Drop emails, phone numbers, personal profile links and the visitor's own name / organisation from a kept note."""
    t = _URL_ME.sub("", _PHONE.sub("", _EMAIL.sub("", text or "")))
    for w in sorted(identifying, key=len, reverse=True):  # "at Acme" goes as a whole, not leaving "wants ... at with a graph"
        t = re.sub(rf"(?:\b(?:at|from|with|for|of)\s+)?\b{re.escape(w)}\b", "", t, flags=re.I)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+([,.;:!?])", r"\1", t)).strip(" ,;:-")


def _within(short: str, long: str) -> bool:
    """A topic of two or more words inside a longer one ("graph algorithms" in "key graph algorithms"). One word is too general:
    "graphs" is not "knowledge graphs"."""
    return len(short.split()) >= 2 and short != long and bool(re.search(r"\b" + re.escape(short) + r"\b", long))


def clean_topics(items: Any, identifying: set[str] = frozenset(), absorb: bool = False, general: bool = False) -> list[str]:
    """Lower-cased, de-duplicated topics, none naming the visitor. With `absorb`, a phrase inside a longer one is the same thing said
    twice ("interesting keynotes" / "interesting keynotes on agentic ai"): only the longest is kept, where it first appeared. With
    `general` the other way round, for topics that should meet across conversations: "key graph algorithms" is "graph algorithms"."""
    out: list[str] = []
    for t in items or []:
        t = " ".join(str(t).lower().split())[:60]
        if not t or any(t.rstrip("s") == o.rstrip("s") for o in out) or _EMAIL.search(t) \
                or any(re.search(rf"\b{re.escape(w)}\b", t) for w in identifying):  # "knowledge graph" = "knowledge graphs"
            continue
        if general:
            if any(_within(o, t) for o in out):
                continue
            out = [o for o in out if not _within(t, o)]
        elif absorb:
            if any(t in o for o in out):
                continue
            longer = [i for i, o in enumerate(out) if o in t]
            if longer:
                out[longer[0]] = t
                out = [o for i, o in enumerate(out) if i not in longer[1:]]
                continue
        out.append(t)
    return out[:20]


_FILLER = {"what", "with", "that", "this", "they", "them", "their", "there", "about", "would", "could", "should", "want", "like",
           "have", "from", "into", "your", "some", "more", "most", "very", "just", "much", "tell", "know", "learn", "using", "uses",
           "make", "help", "need", "visitor", "booth", "neo4j"}


def _content_words(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 4 and w not in _FILLER}


def grounded(item: str, said: set[str], need: int = 1) -> bool:
    """The note's words were actually said by the visitor: at least `need` of its content words appear in their own lines. Seen
    on the robot: from "Hey PuppyPi" and "What is a graph?" the extraction wrote the need "build a knowledge graph for agent
    memory" and the topic "agent memory". A topic with no content word at all (a short acronym like "rag") is kept."""
    words = _content_words(item)
    return not words or len(words & said) >= min(need, len(words))


def greeting(robot: str = "PuppyPi", wake: str = "") -> str:
    """What the robot says when someone is close and turned toward it."""
    ask = f"Say \"{wake}\" and ask me anything about Neo4j, GenAI or GraphRAG." if wake else "Ask me anything about Neo4j, GenAI or GraphRAG."
    return f"Hi there! I'm {robot}, the Graph Wizard. {ask}"


def group_greeting(robot: str = "PuppyPi", wake: str = "") -> str:
    start = f"Say \"{wake}\" and ask" if wake else "Ask"
    return f"Hello everyone! I'm {robot}, the Graph Wizard. {start} me anything about Neo4j, GenAI or GraphRAG. One at a time, please!"


def choose_greeting(people: list[dict], greeted_recently: bool, group_min: int = 3) -> str | None:
    """Pure policy (unit-tested): only people close AND facing the robot count, so passers-by in a busy hall are ignored; a group
    (group_min or more) gets ONE group hello. Returns "group", "single" or None."""
    engaged = [p for p in people if p.get("facing_robot") and p.get("close")]
    if not engaged or greeted_recently:
        return None
    return "group" if len(engaged) >= group_min else "single"


# ------------------------------------------------------------------ conversation

@dataclass
class Encounter:
    id: str
    started: float = field(default_factory=time.monotonic)
    last: float = field(default_factory=time.monotonic)
    greeted: str = ""
    place: str | None = None
    messages: list[dict] = field(default_factory=list)  # RAM only, while the conversation is open: never stored
    notes: dict = field(default_factory=dict)  # the anonymous note kept for this conversation (topics, need, ...)
    asked_about: list[str] = field(default_factory=list)  # never the same question twice
    turns: int = 0          # answers in this conversation
    closing: bool = False   # they said goodbye: no more questions
    ended: bool = False     # the goodbye was answered: the conversation is over (the next visitor says the wake word)
    unclear_at: float = 0.0  # when the robot last said it did not catch something (at most every UNCLEAR_EVERY_S)
    lock: threading.Lock = field(default_factory=threading.Lock)
    notes_lock: threading.Lock = field(default_factory=threading.Lock)  # the learner may run inside a turn (which holds `lock`)


class PersonView(BaseModel):
    zone: Literal["left", "center", "right"] = Field(description="third of the frame the person is in")
    facing_robot: bool = Field(description="turned toward the robot: face, body or feet point at the camera, or they look down at it")
    close: bool = Field(description="within about two metres of the robot; from floor level their legs and feet look large")


class Crowd(BaseModel):
    people: list[PersonView]


class Notes(BaseModel):
    """What a booth conversation was about, with nothing that identifies the visitor."""
    topics: list[str] = Field(description="technologies or themes the visitor asked about or wants to work with (GraphRAG, agent "
                                          "memory, knowledge graphs...), short lower-case phrases")
    need: str = Field(description="what the Visitor SAID they are trying to build or solve, in at most 15 neutral words with NO "
                                  "person, company or product name of theirs (e.g. 'process thousands of contracts with a knowledge "
                                  "graph'). Empty if they only asked questions: never infer it from a question or the Robot's answer")
    familiarity: Literal["", "none", "curious", "user", "expert"] = Field(
        description="how well they know Neo4j, only if they SAID it ('first time', 'I use it at work'); else empty")
    enjoyed: list[str] = Field(description="what they enjoyed at the event (a keynote topic, the robot demo), short phrases, no names")
    questions: list[str] = Field(description="each question they asked the robot, rephrased neutrally in a few words, no names")
    summary: str = Field(description="one neutral sentence about what the conversation was about, never who the visitor is")


EXTRACT_PROMPT = """Summarise an anonymous conversation between a conference visitor and a robot at the Neo4j booth (speech-to-text, may
be imperfect). Keep ONLY what the booth team may know without knowing who the visitor is: the topics, what they want to build or
solve, how well they know Neo4j, what they enjoyed at the event, and the questions they asked. NEVER include anyone's name, the
visitor's company or employer, job title, email, phone number, location or anything else that could identify them, even if they
said it: write "a visitor" or leave it out. Only what the Visitor said or clearly implied; never take facts from the Robot's lines
and never invent: questions are only the ones the Visitor actually asked, each once; enjoyed is empty unless the Visitor said what
they enjoyed. Fix obvious speech-to-text mishearings of Neo4j terms (GraphAcademy, agent memory, GraphRAG, Cypher, Aura) and
leave out anything you cannot make sense of."""


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
    {"type": "function", "function": {"name": "do_trick",
     "description": "Perform a trick when a visitor asks for one (boxing = stands on its hind legs and pumps its front paws; also wave, "
                    "shake_hands, bow, nod, sit, stand, lie_down, stretch, shake_head).",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": list(TRICKS)}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "new_visitor",
     "description": "A DIFFERENT person has started talking to you (they say it is their turn, or ignore the conversation so far). Call this "
                    "FIRST, so the previous visitor's conversation is not mixed up with theirs.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "ask_world",
     "description": "Ask your own memory about the room: what you can see, where things are, what changed. For any question about your surroundings.",
     "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
]


class BoothHost:
    def __init__(self, s: Settings, mem, *, client=None, agent=None, bus=None, look_fn: Callable[[bytes], list[dict]] | None = None,
                 gesture_fn: Callable[[str], dict] | None = None, extract_fn: Callable[[str], dict] | None = None):
        self.s, self.mem, self.agent, self.bus, self.client = s, mem, agent, bus, client
        self.outbox = Outbox()
        self.encounters: dict[str, Encounter] = {}
        self._lock = threading.Lock()
        self._greeting = threading.Lock()
        self._recent_greeting: tuple[float, str] | None = None  # (when, text) for the next conversation to pick up
        self._greeted_at = -1e9
        self._greet_times: list[float] = []
        self._skip_note = -1e9  # when the UI was last told why a person in view was not greeted
        self.active_fn: Callable[[], bool] = lambda: True  # set by Brain: greet only while someone has the Booth tab open
        self.look_fn = look_fn if look_fn is not None else (self._look if client is not None else None)
        self.gesture_fn = gesture_fn  # set by Brain: plays a trick (booth mode only; rate-limited in drive.py)
        self.nams = None  # set by Brain: the anonymous conversation notes are mirrored to NAMS after each save
        # after each answer, what the conversation is about is noted (in the background, beside the reply being prepared)
        self.extract_fn = extract_fn if extract_fn is not None else (self._extract if client is not None else None)
        self.learn_in_background = True  # tests set False to learn synchronously
        # Saves go to the graph in the background, in order: the visitor never waits for the database (~1 s per Aura round trip)
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
                del self.encounters[k]  # the conversation text goes with it: only the anonymous note was kept
            enc = self.encounters.get(encounter_id or "")
            if enc is None:
                enc = Encounter(id=encounter_id or uuid.uuid4().hex[:10], place=place)
                self.encounters[enc.id] = enc
                if self._recent_greeting and now - self._recent_greeting[0] < 120:  # they were just greeted out loud: continue from that
                    enc.greeted = self._recent_greeting[1]
            enc.last = now
            if place:
                enc.place = place
            return enc

    def _state(self, enc: Encounter) -> str:
        lines = ["", "STATE OF THIS CONVERSATION"]
        if enc.closing:
            lines.append("- They are leaving: thank them warmly and say goodbye, no questions.")
        else:
            lines.append("- Do NOT end with a question of your own about them: one question about their interests is added after your answer.")
        if enc.greeted:
            lines.append(f"- You already greeted them out loud with: \"{enc.greeted}\" Do not greet again; carry on from there.")
        return "\n".join(lines)

    # ---- the anonymous note: kept in memory, written behind
    def _save(self, enc: Encounter) -> None:
        n = enc.notes
        if not (n.get("topics") or n.get("need") or n.get("questions") or n.get("enjoyed")):
            return  # nothing worth keeping yet (a hello)
        note = {"id": enc.id, "event": self.s.event_name, "turns": enc.turns, **n}
        job = (note, enc.place, datetime.now(timezone.utc).isoformat())
        if not self.write_in_background:
            self._write_one(*job)
            return
        self._writes.put(job)
        if self._writer is None or not self._writer.is_alive():
            self._writer = threading.Thread(target=self._write_loop, daemon=True, name="booth-writer")
            self._writer.start()

    def _write_loop(self) -> None:
        while True:
            self._write_one(*self._writes.get())

    def _write_one(self, note: dict, place: str | None, ts: str) -> None:
        try:
            self.mem.save_conversation(note, place=place, ts=ts)
            if self.nams is not None:
                self.nams.submit_conversation(note, place, ts, self.s.robot_name)
            self._publish("booth", {"event": "saved", "id": note["id"], "topics": note.get("topics", [])})
        except Exception as e:
            self._publish("error", {"error": f"could not save the booth conversation: {e}"[:200]})

    def flush(self, timeout: float = 10.0) -> None:
        """Wait until queued saves reached the graph (tests, shutdown)."""
        end = time.monotonic() + timeout
        while not self._writes.empty() and time.monotonic() < end:
            time.sleep(0.05)

    # ---- tools
    def _tool(self, enc: Encounter, name: str, a: dict) -> Any:
        if name == "do_trick":
            if not self.gesture_fn:
                return {"done": False, "reason": "tricks are not available"}
            return self.gesture_fn(a.get("name", ""))
        if name == "new_visitor":
            with self._lock:
                self.encounters.pop(enc.id, None)
                enc.id = uuid.uuid4().hex[:10]  # the previous visitor's note stays as it was saved; this one starts empty
                self.encounters[enc.id] = enc
            enc.greeted, enc.notes, enc.asked_about, enc.turns, enc.closing = "", {}, [], 1, False
            enc.messages.clear()
            return {"new_conversation": True, "note": "Treat them as someone new."}
        if name == "ask_world":
            if not self.agent:
                return {"error": "I cannot look around right now."}
            r = self.agent.ask(a["question"], session_id=f"booth-{enc.id}")
            return {"answer": r["answer"]}
        return {"error": f"unknown tool {name}"}

    UNCLEAR_EVERY_S = 12.0

    def unclear_reply(self, encounter_id: str | None) -> str | None:
        """Something in an OPEN conversation was not understood: say so (at most every UNCLEAR_EVERY_S) and keep the conversation
        open, instead of going silent. Outside a conversation, hall noise stays unanswered (None)."""
        enc = self.encounters.get(encounter_id or "")
        now = time.monotonic()
        if not enc or not enc.messages or enc.ended or now - enc.last >= self.s.wake_window_s or now - enc.unclear_at < self.UNCLEAR_EVERY_S:
            return None
        enc.unclear_at = enc.last = now
        return "Sorry, I didn't catch that. Could you say it again?"

    # ---- wake word (robot microphone only: Hold to talk and typing are addressed to the robot by definition)
    def end_conversations(self) -> int:
        """The Booth tab's "End conversation": every open conversation is over (the next visitor says the wake word)."""
        n = 0
        for e in list(self.encounters.values()):
            if e.messages and not e.ended:
                e.ended, e.closing, n = True, True, n + 1
        return n

    def conversation_active(self) -> bool:
        """Someone talked with the robot less than WAKE_WINDOW_S ago."""
        now = time.monotonic()
        return any(now - e.last < self.s.wake_window_s for e in list(self.encounters.values()) if e.messages and not e.ended)

    def addressed(self, text: str, encounter_id: str | None) -> tuple[bool, str | None]:
        """Should the robot answer what its microphone heard? Yes if it contains the wake word, or a conversation is open (the
        last exchange or a greeting less than WAKE_WINDOW_S ago). Returns (answer?, encounter id to use): after a closed
        conversation, a wake word starts a fresh one, since it is most likely someone new."""
        now = time.monotonic()
        enc = self.encounters.get(encounter_id or "")
        if enc and now - enc.last < self.s.wake_window_s and not enc.ended:
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
        text = fix_terms(text)  # typed turns too ("graph rag", "cipher")
        enc = self._encounter(encounter_id, place)
        with enc.lock:
            enc.turns += 1
            if says_goodbye(text):
                enc.closing = True
            start_id, learner = enc.id, None
            if self.extract_fn and self.learn_in_background:  # note what this conversation is about while the reply is prepared
                learner = threading.Thread(target=self._learn, args=(enc, start_id, *self._transcript(enc, text)), daemon=True)
                learner.start()
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
                        res = self._tool(enc, tc.function.name, args)
                    except Exception as e:
                        res = {"error": str(e)[:300]}
                    trace.append({"tool": tc.function.name, "args": args, "result": res})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(res, default=str)[:6000]})
                    if tc.function.name == "new_visitor":  # forget the previous visitor's conversation, in this turn too
                        msgs = [{"role": "system", "content": system_prompt(self.s.robot_name, self.s.event_name, self._state(enc),
                                                                            self.s.voice_language, self.s.booth_info)},
                                *msgs[1 + len(history):]]
                        history = []
            else:
                reply = "Sorry, my circuits got tangled on that one. Could you ask me again?"
            reply = drop_own_ask(reply)
            if self.extract_fn and (learner is None or enc.id != start_id):  # (again) for a new visitor: only their own words
                after = (enc, enc.id, *self._transcript(enc, text))
                learner = threading.Thread(target=self._learn, args=after, daemon=True) if self.learn_in_background else None
                if learner is None:
                    self._learn(*after)
                else:
                    learner.start()
            enc.messages += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            enc.last = time.monotonic()
        if learner is not None:  # started beside the reply: wait for it, so the question below never asks what they just said
            learner.join(timeout=2)  # not longer: on a slow network the note may take 10 s; it is saved when it is done
        if enc.closing:
            enc.ended = True  # seen at the booth: after "Goodbye" it stayed "in a conversation" for another 90 s
        else:
            question = self._next_question(enc, text)
            if question:
                reply = end_with_question(reply, question)
                enc.messages[-1]["content"] = reply
        out = {"encounter_id": enc.id, "reply": reply, "notes": dict(enc.notes), "evidence": trace}
        self._publish("converse", {"encounter_id": enc.id, "heard": text, "reply": reply, "tools": [t["tool"] for t in trace]})
        return out

    def _next_question(self, enc: Encounter, text: str) -> str | None:
        """The one question that ends this reply: the first interest still unknown, never one already asked, nothing on a goodbye."""
        for key in ASK_ORDER:
            if key == "enjoyed" and (enc.turns < ENJOYED_AFTER_TURNS or asks_something(text)):
                continue
            if not enc.notes.get(NOTE_KEY.get(key, key)) and key not in enc.asked_about:
                enc.asked_about.append(key)
                return QUESTIONS[key].format(event=self.s.event_name)
        return None

    @staticmethod
    def _transcript(enc: Encounter, text: str) -> tuple[str, list[str]]:
        lines = [*enc.messages[-15:], {"role": "user", "content": text}]
        return ("\n".join(("Visitor: " if m["role"] == "user" else "Robot: ") + m["content"] for m in lines),
                [m["content"] for m in lines if m["role"] == "user"])

    def _learn(self, enc: Encounter, conv_id: str, transcript: str, visitor_lines: list[str]) -> None:
        """Note what the conversation is about, scrubbed of anything that identifies the visitor, and save it. Never raises: a
        missed note is not an error. Dropped if a new visitor took over meanwhile (their words are noted on their own)."""
        try:
            f = (self.extract_fn(transcript) if self.extract_fn else {}) or {}
            f = {k: ([fix_terms(str(x)) for x in v] if isinstance(v, list) else fix_terms(v) if isinstance(v, str) else v) for k, v in f.items()}
            idw = identifying_words(visitor_lines)
            said = _content_words(" ".join(visitor_lines))
            notes = {
                "topics": clean_topics([t for t in f.get("topics") or [] if grounded(t, said)], idw, general=True),
                "enjoyed": clean_topics([t for t in f.get("enjoyed") or [] if grounded(t, said)], idw, absorb=True),
                "questions": [q for q in (scrub(q, idw)[:120] for q in f.get("questions") or [] if grounded(q, said)) if len(q) > 3][:20],
                "need": scrub(f.get("need") or "", idw)[:160] if grounded(f.get("need") or "", said, 2) else "",
                "familiarity": f.get("familiarity") if f.get("familiarity") in ("none", "curious", "user", "expert") else "",
                "summary": scrub(f.get("summary") or "", idw)[:300],
            }
            if enc.id != conv_id:
                return
            with enc.notes_lock:
                # While the extraction still sees the whole conversation, its lists replace the earlier ones (adding them up kept
                # every guess: a replay collected questions nobody asked); after that, the older part is only in the earlier lists.
                whole, old = enc.turns <= 8, enc.notes  # the transcript is the last 15 messages and this one
                lists = {k: (notes[k] or old.get(k, [])) if whole else [*old.get(k, []), *notes[k]] for k in ("topics", "enjoyed", "questions")}
                enc.notes = {
                    "topics": clean_topics(lists["topics"], general=True),
                    "enjoyed": clean_topics(lists["enjoyed"], absorb=True),
                    "questions": list(dict.fromkeys(lists["questions"]))[:20],
                    **{k: notes[k] or old.get(k, "") for k in ("need", "familiarity", "summary")},
                }
            self._save(enc)
        except Exception as e:
            self._publish("error", {"error": f"could not note the conversation: {e}"[:200]})

    def _extract(self, transcript: str) -> dict:
        r = self.client.chat.completions.parse(
            model=self.s.chat_model, temperature=0, response_format=Notes,
            messages=[{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": transcript}])
        f = r.choices[0].message.parsed
        return f.model_dump() if f else {}

    # ---- greeting when the camera sees someone
    def on_view(self, scene, jpeg: bytes | None) -> bool:
        """Called after each analysed frame. Cheap checks first (a person in view, rate limits, nobody already talking to the robot);
        only then one vision call to see whether someone is close and turned toward the robot. Runs in the background. Returns True
        if a greeting attempt was started."""
        s = self.s
        if not s.people_greet or not self.active_fn() or not any(o.label == "person" and o.confidence >= s.min_confidence for o in scene.objects):
            return False
        now = time.monotonic()
        self._greet_times = [t for t in self._greet_times if now - t < 3600]
        wait = s.greet_cooldown_s - (now - self._greeted_at)
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

    def _greet(self, jpeg: bytes | None) -> None:
        try:
            people = [{"zone": "center", "facing_robot": True, "close": True}]  # no eyes: assume one
            if jpeg and self.look_fn:
                try:
                    people = self.look_fn(jpeg)
                except Exception as e:
                    self._publish("error", {"error": f"looking at people failed: {e}"[:200]})
                    return
            recently = time.monotonic() - self._greeted_at < self.s.person_regreet_s
            kind = choose_greeting(people, recently, self.s.group_min)
            if kind is None:
                engaged = sum(bool(p.get("facing_robot") and p.get("close")) for p in people)
                self._publish("greeting", {"skipped": True, "people": len(people), "engaged": engaged,
                                           "reason": "nobody in view" if not people else "greeted recently" if engaged else
                                           "nobody close to the robot and turned toward it"})
                return
            text = group_greeting(self.s.robot_name, self.s.wake_word) if kind == "group" else greeting(self.s.robot_name, self.s.wake_word)
            now = time.monotonic()
            self._greeted_at = now
            self._greet_times.append(now)
            self._recent_greeting = (now, text)
            self.outbox.say(text)
            trick = self.gesture_fn(self.s.greet_action) if (self.gesture_fn and self.s.greet_action) else None
            self._publish("greeting", {"text": text, "kind": kind, "people": len(people), "trick": trick})
        finally:
            self._greeting.release()

    def _look(self, jpeg: bytes) -> list[dict]:
        """One vision call: is anyone close and turned toward the robot (actually at the booth, not walking past)? Never reads badges
        and never identifies anyone."""
        from .perception import _prepare

        r = self.client.chat.completions.parse(
            model=self.s.vision_model, temperature=0, response_format=Crowd,
            messages=[{"role": "system", "content": LOOK_PROMPT},
                      {"role": "user", "content": [{"type": "text", "text": "Who is here?"},
                                                   {"type": "image_url", "image_url": {"url": _prepare(jpeg), "detail": "low"}}]}])
        c = r.choices[0].message.parsed
        return [v.model_dump() for v in (c.people if c else [])][:8]


LOOK_PROMPT = (
    "You are the eyes of a small robot dog. Its camera is at FLOOR LEVEL, about 15 cm high and tilted slightly up, so a person "
    "near it often shows only as legs and feet, maybe a lower body, and their face is usually out of frame. List each person "
    "visible, even if only their legs or feet are, at most 8, nearest first. Sitting people count. "
    "facing_robot: they are turned toward the robot, i.e. their toes, knees or body point at the camera, their face is turned to "
    "it, or they are looking down at it. Someone seen from the side or back, or walking across the view, is not facing it. "
    "close: within about two metres; from the floor their legs and feet look large, and someone in the background looks small. "
    "Never read name badges and never identify or describe anyone."
)
