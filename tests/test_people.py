"""People at the booth: consent, memory, lead scoring, greeting. No network: a scripted fake chat model drives the tools."""
import dataclasses
import json
import time
from types import SimpleNamespace as NS

from puppy_mind.config import get_settings
from puppy_mind.devfake import FakeMemory
from puppy_mind.models import Detection, Scene
from puppy_mind.people import (PeopleService, choose_greeting, consent_from, greeting, lead_score, match_person, merge_person, name_key,
                               safe_view)

S = dataclasses.replace(get_settings(), nams_api_key="", people_consent=True, greet_cooldown_s=0.2, people_greet=True,
                        max_greetings_per_hour=40, person_regreet_s=1800, group_min=3, greet_action="boxing")


def call(tool, /, **args):
    return NS(id=f"c-{tool}", function=NS(name=tool, arguments=json.dumps(args)))


class Msg(NS):
    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in (self.tool_calls or [])]}


class FakeChat:
    """Each turn() consumes one scripted list: tool calls (one per model step) then a final text. Records the system prompts shown."""

    def __init__(self, script):
        self.script, self.prompts, self.step = list(script), [], None
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, tools, temperature):
        self.prompts.append(messages[0]["content"])
        if not self.step:
            self.step = list(self.script.pop(0))
        item = self.step.pop(0)
        if isinstance(item, str):
            self.step = None
            return NS(choices=[NS(message=Msg(content=item, tool_calls=None))])
        return NS(choices=[NS(message=Msg(content=None, tool_calls=[item]))])


def svc(script, **kw):
    mem = FakeMemory(S)
    return PeopleService(S, mem, client=FakeChat(script), **kw), mem


def test_consent_detection():
    assert consent_from("Yes, sure") is True and consent_from("no thanks") is False
    assert consent_from("what is graphrag") is None and consent_from("yes but no") is None and consent_from("") is None


def test_name_key_and_matching():
    assert name_key("Zoë  O'Neil") == name_key("zoe oneil")
    a, b = {"person_id": "1", "company": "Acme Inc."}, {"person_id": "2", "company": "Globex"}
    assert match_person([a, b], "acme")["person_id"] == "1"
    assert match_person([a, b], "Initech") is None  # two different John Smiths stay two people
    assert match_person([a], None)["person_id"] == "1" and match_person([], "x") is None


def test_lead_score_stages_and_explainability():
    cold = merge_person(None, {"interests": ["coffee"]}, "Sam")
    assert cold["lead_stage"] == "cold"
    warm = merge_person(None, {"role": "Senior Data Engineer", "company": "Acme", "interests": ["graphrag"], "use_case": "support chatbot"}, "Ana")
    assert warm["lead_stage"] == "warm"
    hot = merge_person(warm, {"wants_follow_up": True, "email": "ana@acme.com"}, "Ana")
    assert hot["lead_stage"] == "hot" and hot["lead_score"] <= 100 and hot["person_id"] == warm["person_id"]
    assert lead_score({"role": "CTO", "wants_demo": True, "wants_follow_up": True, "email": "x", "use_case": "y", "company": "z"})[0] == 100


def test_nothing_is_saved_without_a_spoken_yes():
    s, mem = svc([[call("save_person_facts", name="Ana", company="Acme"), "I cannot save yet."],
                  [call("record_consent", granted=True), "ok"]])  # the visitor never said yes: the model is wrong
    r = s.turn("Hi, I'm Ana from Acme")
    assert r["evidence"][0]["result"]["error"].startswith("Not saved") and mem.list_people() == []
    r = s.turn("tell me about graphs", r["encounter_id"])
    assert "has not clearly said" in r["evidence"][0]["result"]["error"] and r["consent"] is None
    assert mem.list_people() == []


def test_consent_then_save_merge_and_returning_visitor():
    s, mem = svc([
        [call("record_consent", granted=True), call("save_person_facts", name="Ana Berg", role="Staff Engineer", company="Acme",
                                                      interests=["GraphRAG"], enjoyed=["the demo"], encounter_summary="Ana asked about GraphRAG."), "Thanks Ana!"],
        [call("save_person_facts", wants_follow_up=True, email="ana@acme.com"), "Noted."],
    ])
    r = s.turn("Yes please, go ahead")
    assert r["consent"] is True and r["person"]["name"] == "Ana Berg" and r["person"]["interests"] == ["graphrag"]
    r2 = s.turn("Yes follow up with me", r["encounter_id"])
    p = r2["person"]
    assert p["lead_stage"] == "hot" and p["meet_count"] == 1 and p["company"] == "Acme" and p["email"] == "ana@acme.com"
    assert len(mem.list_people()) == 1 and r2["evidence"][0]["args"]["email"] == "***"  # not echoed to the UI trace
    # a later visit: identify_person welcomes back with the safe view only, and consent carries over
    s2 = PeopleService(S, mem, client=FakeChat([[call("identify_person", name="ana berg"), call("save_person_facts", interests=["aura"]), "Welcome back!"]]))
    r3 = s2.turn("Hi, I'm Ana Berg, I was here earlier")
    known = r3["evidence"][0]["result"]
    assert known["known"] and "email" not in known and "lead_stage" not in known and known["company"] == "Acme"
    assert r3["person"]["meet_count"] == 2 and r3["person"]["interests"] == ["graphrag", "aura"]


def test_declined_saves_nothing_and_robot_is_told():
    s, mem = svc([[call("record_consent", granted=False), "No problem, ask me anything."], [call("save_person_facts", name="X"), "ok"]])
    r = s.turn("No thanks")
    assert r["consent"] is False
    s.turn("I'm X from Y", r["encounter_id"])
    assert mem.list_people() == [] and "NO" in s.client.prompts[-1]


def test_forget_me_erases():
    s, mem = svc([[call("record_consent", granted=True), call("save_person_facts", name="Ana"), "ok"], [call("forget_me"), "Done, I forgot you."]])
    r = s.turn("yes")
    assert len(mem.list_people()) == 1
    r = s.turn("please forget me", r["encounter_id"])
    assert r["evidence"][0]["result"]["forgotten"] is True and mem.list_people() == [] and r["person"] is None


def test_safe_view_hides_private_fields():
    v = safe_view({"name": "Ana Berg", "email": "a@b.c", "notes": "n", "lead_score": 90, "company": "Acme", "last_met": "2026-10-31T10:00"})
    assert set(v) == {"first_name", "role", "company", "interests", "enjoyed", "times_met", "last_met"} and v["first_name"] == "Ana"


def person_scene(conf=0.9):
    return Scene(summary="a person", room_type="unknown", room_confidence=0.0, relations=[],
                 objects=[Detection(idx=0, label="person", color="unknown", description="", zone="center", confidence=conf)])


def test_greeting_text():
    assert "Graph Wizard" in greeting(None)
    assert greeting(None, "ana berg").startswith("Hello Ana!")
    back = greeting({"name": "Ana Berg", "company": "Acme", "interests": ["graphrag"]})
    assert back.startswith("Welcome back, Ana!") and "Acme" in back and "graphrag" in back


def wait(pred, t=2.0):
    end = time.time() + t
    while time.time() < end and not pred():
        time.sleep(0.01)
    return pred()




def V(zone="center", facing=True, close=True, name="", company=""):
    return {"zone": zone, "facing_robot": facing, "close": close, "badge_name": name, "badge_company": company}


def look(*people):
    return lambda jpeg: [dict(p) for p in people]


def test_crowd_policy_ignores_passers_by_and_groups_a_crowd():
    nobody = lambda k, c: None
    never = lambda k: False
    assert choose_greeting([V(facing=False), V(close=False), V(zone="left", facing=False, close=False)], nobody, never) is None
    assert choose_greeting([V(), V(zone="left"), V(zone="right"), V(facing=False)], nobody, never)["kind"] == "group"
    assert choose_greeting([V(), V(zone="left")], nobody, never, group_min=3)["kind"] == "anon"


def test_crowd_policy_prefers_the_person_in_front_then_a_returning_visitor_and_skips_recent_ones():
    known = {"ana berg": {"person_id": "p1", "name": "Ana Berg"}}
    lookup = lambda k, c: known.get(k)
    pick = choose_greeting([V(zone="left", name="Ana Berg"), V(zone="center", name="Bob Lee")], lookup, lambda k: False)
    assert pick["name"] == "Bob Lee" and pick["kind"] == "named"  # the one in front first
    pick = choose_greeting([V(zone="left", name="Ana Berg"), V(zone="right", name="Bob Lee")], lookup, lambda k: False)
    assert pick["kind"] == "returning" and pick["person"]["person_id"] == "p1"  # nobody in front: a returning visitor first
    pick = choose_greeting([V(name="Ana Berg"), V(zone="left", name="Bob Lee")], lookup, lambda k: k == "ana berg")
    assert pick["name"] == "Bob Lee"  # Ana was greeted recently
    assert choose_greeting([V(name="Ana Berg")], lookup, lambda k: True) is None


def test_person_in_view_greets_once_with_a_trick_then_cooldown():
    mem = FakeMemory(S)
    p = merge_person(None, {"company": "Acme", "interests": ["graphrag"]}, "Ana Berg")
    mem.save_person(p, encounter_id="e1", place=None, summary="", ts="2026-10-31T10:00:00+00:00")
    tricks = []
    s = PeopleService(S, mem, client=None, look_fn=look(V(name="Ana Berg", company="Acme")),
                      gesture_fn=lambda n: tricks.append(n) or {"ok": True, "action": n})
    assert s.on_view(person_scene(), b"x") is True
    assert wait(lambda: s.outbox._items)
    assert s.outbox.next()["text"].startswith("Welcome back, Ana!") and tricks == ["boxing"]
    assert s.on_view(person_scene(), b"x") is False  # cooldown
    assert s.on_view(person_scene(conf=0.1), b"x") is False  # not confident it is a person
    assert s.on_view(Scene(summary="", room_type="unknown", room_confidence=0, objects=[], relations=[]), None) is False


def test_passers_by_are_not_greeted_and_the_hourly_cap_holds():
    s = PeopleService(S, FakeMemory(S), client=None, look_fn=look(V(facing=False), V(close=False)))
    s.on_view(person_scene(), b"x")
    assert wait(lambda: s._greeting.acquire(blocking=False) and (s._greeting.release() or True))
    assert s.outbox.next() is None
    capped = PeopleService(dataclasses.replace(S, max_greetings_per_hour=1, greet_cooldown_s=0), FakeMemory(S), client=None, look_fn=look(V()))
    capped.on_view(person_scene(), b"x")
    assert wait(lambda: capped.outbox._items)
    assert capped.on_view(person_scene(), b"x") is False


def test_a_crowd_gets_one_group_hello():
    s = PeopleService(S, FakeMemory(S), client=None, look_fn=look(V(), V(zone="left"), V(zone="right"), V(facing=False)))
    s.on_view(person_scene(), b"x")
    assert wait(lambda: s.outbox._items)
    assert s.outbox.next()["text"].startswith("Hello everyone!") and s.outbox.next() is None


def test_badges_are_ignored_without_people_consent_flag():
    off = dataclasses.replace(S, people_consent=False)
    s = PeopleService(off, FakeMemory(off), client=None, look_fn=look(V(name="Ana Berg")))
    s.on_view(person_scene(), b"x")
    assert wait(lambda: s.outbox._items)
    assert "Hi there" in s.outbox.next()["text"] and s._recent_greeting[2] is None


def test_no_greeting_while_someone_is_talking():
    s, _ = svc([["Hello!"]])
    s.turn("hi")
    assert s.on_view(person_scene(), b"x") is False


def test_greeting_is_carried_into_the_next_conversation():
    s, _ = svc([["Hello!"]], look_fn=look(V(name="Ana Berg", company="Acme")))
    s.on_view(person_scene(), b"x")
    assert wait(lambda: s.outbox._items)
    s.turn("hello")
    assert "already greeted" in s.client.prompts[-1] and "Ana Berg" in s.client.prompts[-1] and "unverified" in s.client.prompts[-1]


def test_new_visitor_starts_a_clean_conversation_so_facts_do_not_mix():
    s, mem = svc([
        [call("record_consent", granted=True), call("save_person_facts", name="Ana", company="Acme"), "ok"],
        [call("new_visitor"), call("save_person_facts", name="Bob", company="Globex"), "Hi Bob! May I remember you?"],
    ])
    r = s.turn("yes")
    r2 = s.turn("Hi I'm Bob from Globex, my turn!", r["encounter_id"])
    assert r2["encounter_id"] != r["encounter_id"] and r2["consent"] is None and r2["person"] is None
    assert "Not saved" in r2["evidence"][1]["result"]["error"]  # Bob has not said yes; Ana's consent does not carry over
    assert [p["name"] for p in mem.list_people()] == ["Ana"] and "Their name: Ana" not in s.client.prompts[-1]


def test_do_trick_tool_uses_the_rate_limited_gesture():
    s, _ = svc([[call("do_trick", name="boxing"), "Ta-da!"]], gesture_fn=lambda n: {"ok": False, "reason": "tricks are rate-limited"})
    r = s.turn("can you do boxing?")
    assert r["evidence"][0]["result"]["reason"] == "tricks are rate-limited"


def test_a_person_in_view_who_is_not_greeted_is_logged_with_the_reason():
    events = []
    bus = NS(publish=lambda kind, d: events.append((kind, d)))
    s, _ = svc([["Hello!"]], bus=bus, look_fn=look(V(facing=False)))
    s.on_view(person_scene(), b"x")  # seen from the side: looked at, not greeted
    assert wait(lambda: any(d.get("skipped") for k, d in events if k == "greeting"))
    assert [d["reason"] for k, d in events if k == "greeting"] == ["nobody close to the robot and turned toward it"]
    s.turn("hi")
    s._skip_note = -1e9
    s.on_view(person_scene(), b"x")
    assert events[-1][1]["reason"] == "already talking to someone"
    s.on_view(person_scene(), b"x")  # the same reason again within 20 s is not repeated
    assert sum(1 for k, d in events if d.get("reason") == "already talking to someone") == 1


def test_the_look_prompt_knows_the_camera_is_at_floor_level():
    from puppy_mind.people import LOOK_PROMPT, PersonView

    assert "FLOOR LEVEL" in LOOK_PROMPT and "feet" in LOOK_PROMPT and "Sitting people count" in LOOK_PROMPT
    assert "feet" in PersonView.model_fields["facing_robot"].description
