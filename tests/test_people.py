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
    kw.setdefault("extract_fn", lambda transcript: {})  # no automatic fact extraction unless a test scripts it
    s = PeopleService(S, mem, client=FakeChat(script), **kw)
    s.learn_in_background = s.write_in_background = False
    return s, mem


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
        [call("save_person_facts", wants_follow_up=True, email="ana@acme.com"), "Let me read it back: a-n-a at acme dot com?"],
        ["Saved, thanks!"],
    ])
    r = s.turn("Yes please, go ahead")
    assert r["consent"] is True and r["person"]["name"] == "Ana Berg" and r["person"]["interests"] == ["graphrag"]
    r2 = s.turn("Yes follow up with me, my email is ana at acme.com", r["encounter_id"])
    assert r2["person"]["wants_follow_up"] is True and not r2["person"].get("email")  # held back until she confirms it
    confirmed = s.turn("exactly", r["encounter_id"])
    p = mem.get_person(confirmed["person"]["person_id"])
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


def test_permission_is_asked_at_the_end_once_and_never_by_the_model():
    from puppy_mind.people import asks_consent

    assert asks_consent("GraphRAG is great. May I remember your name and company? Just say yes or no.")
    assert asks_consent("Can I save your details for a follow-up?") and not asks_consent("I remember GraphRAG well.")
    s, _ = svc([["Hi Zaid! GraphRAG links facts."], ["Vector search finds similar text."], ["Bye, enjoy!"], ["Sure."]])
    r = s.turn("Hi, I'm Zaid from Neo4j. What is GraphRAG?")
    assert "remember" not in r["reply"].lower() and "Do NOT ask for permission" in s.client.prompts[-1]
    r = s.turn("And vector search?", r["encounter_id"])
    assert "remember" not in r["reply"].lower()
    r = s.turn("Thanks, I need to run now.", r["encounter_id"])  # the end: now the permission question, with the name check
    assert "Before you go: may I remember" in r["reply"] and "send you material" in r["reply"]
    r = s.turn("Hmm, what about Aura?", r["encounter_id"])  # not an answer: not asked twice
    assert "may I remember" not in r["reply"]


def test_the_wizard_is_told_to_keep_spoken_answers_short_and_not_to_ask_permission_itself():
    from puppy_mind.wizard import system_prompt

    p = system_prompt("PuppyPi", "the conference", "")
    assert "at most 2 short sentences" in p and "asked automatically at the END of the conversation" in p
    assert "AT MOST ONCE per conversation" in p and "not mishearings" in p


def test_the_booth_language_is_fixed_for_hearing_and_answering():
    from types import SimpleNamespace

    from puppy_mind.voice import Voice
    from puppy_mind.wizard import system_prompt

    calls = []
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(
        create=lambda **kw: calls.append(kw) or SimpleNamespace(text="hello"))))
    Voice(dataclasses.replace(S, voice_language="en"), client=client).transcribe(b"x", "s.webm")
    assert calls[0]["language"] == "en"  # the speech model never guesses the language from a noisy clip
    assert "Always answer in English" in system_prompt("PuppyPi", "the event", "")
    assert "Always answer in German" in system_prompt("PuppyPi", "the event", "", "de")
    s, _ = svc([["Hello!"]])
    s.turn("hi")
    assert "Always answer in English" in s.client.prompts[-1]


def test_permission_comes_with_a_goodbye_a_request_for_material_or_after_six_answers():
    from puppy_mind.people import CONSENT_AFTER_TURNS, introduces_self

    assert introduces_self("Sorry. My name is Zaid, I work at Neo4j.") and introduces_self("Hi, I'm Ana")
    assert introduces_self("I am a data engineer") and not introduces_self("What is GraphRAG?") and not introduces_self("I'm fine thanks")
    s, _ = svc([["Sure, happy to."]])
    r = s.turn("Please send me the slides on GraphRAG")
    assert "By the way: may I remember" in r["reply"]  # a request for material: ask right away (not leaving)
    s, _ = svc([["Answer."]] * CONSENT_AFTER_TURNS)
    r = None
    for i in range(CONSENT_AFTER_TURNS):
        r = s.turn(f"question {i}", r["encounter_id"] if r else None)
        if i < CONSENT_AFTER_TURNS - 1:
            assert "may I remember" not in r["reply"]
    assert "may I remember" in r["reply"]  # no goodbye: asked at the latest after six answers


def test_what_a_visitor_says_after_yes_is_saved_and_earlier_facts_at_the_yes():
    # seen on the robot: "we're building a physical AI system" came after consent and never reached the record
    seen = []

    def extract(transcript):
        seen.append(transcript)
        if "physical AI" in transcript:  # the extractor sees the whole conversation: check the later fact first
            return {"use_case": "a physical AI system", "interests": ["memory types"], "wants_follow_up": "", "corrected_name": ""}
        if "Rockstar" in transcript:
            return {"company": "Rockstar Games", "interests": ["agent memory"]}
        return {}

    s, mem = svc([["Hello Rosenberg! May I remember you as Rosenberg from Rockstar Games? Just say yes or no."],
                  [call("record_consent", granted=True), call("save_person_facts", name="Rosenberg"), "Great! What are you building?"],
                  ["Neo4j fits well as the memory of such a system."]], extract_fn=extract)
    r = s.turn("Hi, my name is Rosenberg. I work at Rockstar Games and I'd like to learn about agent memory.")
    r = s.turn("Of course you can remember my name.", r["encounter_id"])
    assert "Rockstar Games" in seen[-1] and "Visitor:" in seen[-1]  # at the yes: everything said so far
    p = mem.list_people()[0]
    assert p["company"] == "Rockstar Games" and "agent memory" in p["interests"]
    r = s.turn("We're building a physical AI system and wonder how Neo4j can fit.", r["encounter_id"])
    p = mem.get_person(p["person_id"])
    assert p["use_case"] == "a physical AI system" and "memory types" in p["interests"] and p["lead_stage"] in ("warm", "hot")
    assert "Robot: Great! And what is your role there?\nVisitor: We're building" in seen[-1]  # whole conversation, robot's question included


def test_before_the_yes_facts_stay_in_memory_and_a_no_discards_them():
    s, mem = svc([["Nice to meet you, Ana."], ["No problem, enjoy!"]],
                 extract_fn=lambda t: {"name": "Ana Berg", "company": "Acme", "role": "CTO"})
    r = s.turn("I'm Ana Berg, CTO at Acme. I have to go now.")
    assert mem.list_people() == [] and "may I remember you as Ana Berg" in r["reply"]  # drafted in memory, asked with her name
    r = s.turn("No thanks", r["encounter_id"])
    enc = s.encounters[r["encounter_id"]]
    assert r["consent"] is False and enc.record is None and mem.list_people() == []  # nothing kept anywhere


def test_a_spelled_name_corrects_the_record():
    s, mem = svc([[call("record_consent", granted=True), call("save_person_facts", name="Zeid Yaim", company="Neo4j"), "Thanks!"],
                  ["Got it, Zaid Zaim!"]],
                 extract_fn=lambda t: {"corrected_name": "Zaid Zaim"} if "Z-A-I-D" in t else {})
    r = s.turn("Yes")
    r = s.turn("It's Zaid, Z-A-I-D, Zaim with a Z.", r["encounter_id"])
    assert [p["name"] for p in mem.list_people()] == ["Zaid Zaim"] and mem.list_people()[0]["company"] == "Neo4j"


def test_the_robot_ends_with_the_next_unknown_fact_as_a_fixed_question():
    from puppy_mind.people import CONSENT_QUESTION, QUESTIONS, unknown_fields

    assert "spell it" in CONSENT_QUESTION
    assert unknown_fields({"company": "Neo4j", "role": "DevRel"})[0] == "what they are building or working on"
    s, _ = svc([[call("record_consent", granted=True), call("save_person_facts", name="Zaid", company="Neo4j"), "Thanks!"],
                ["Graphs connect things. How are you liking it?"]])
    r = s.turn("Yes")
    assert r["reply"].endswith(QUESTIONS["role"])  # company known: the next unknown is the role
    r = s.turn("Tell me about graphs", r["encounter_id"])
    assert r["reply"] == "Graphs connect things. " + QUESTIONS["use_case"]  # the model's own question is replaced
    assert "one question about them is added" in s.client.prompts[-1]


def test_an_unanswered_question_is_not_asked_again_and_a_goodbye_gets_none():
    from puppy_mind.people import QUESTIONS

    s, _ = svc([[call("record_consent", granted=True), call("save_person_facts", name="Christian"), "Thanks!"],
                ["Contracts fit graphs well."], ["Have a great day!"]])
    r = s.turn("Yes, of course")
    assert r["reply"].endswith(QUESTIONS["company"])  # seen at the booth: company was never asked
    r = s.turn("I want to process thousands of contracts", r["encounter_id"])
    assert r["reply"].endswith(QUESTIONS["role"])  # he did not answer the company question: move on, do not repeat it
    r = s.turn("I need to run now to the next one. Thank you very much, PuppyPi.", r["encounter_id"])
    assert r["reply"] == "Have a great day!"  # no question on a goodbye


def test_demo_and_follow_up_only_from_the_visitors_own_words():
    from puppy_mind.people import explicit_wishes

    assert explicit_wishes("I'd love a demo for my team") == {"wants_demo": True}
    assert explicit_wishes("No demo needed, thanks") == {"wants_demo": False}
    assert explicit_wishes("Please follow up with me") == {"wants_follow_up": True}
    assert explicit_wishes("What kind of demos are at the booth?") == {}  # asking what exists is not asking for one
    assert explicit_wishes("That's it, let's close it here") == {}  # a goodbye is not a no
    # the model's or the extractor's guess never sets them (seen at the booth: a demo wish nobody voiced)
    s, mem = svc([[call("record_consent", granted=True), call("save_person_facts", name="Andreas", wants_demo=True), "Thanks!"],
                  ["Sure."]], extract_fn=lambda t: {"wants_demo": "yes"})
    r = s.turn("Yes")
    assert mem.list_people()[0]["wants_demo"] is False
    s.turn("Actually yes, I'd love a demo", r["encounter_id"])
    assert mem.list_people()[0]["wants_demo"] is True


def test_an_email_is_saved_only_after_the_visitor_confirms_it():
    s, mem = svc([[call("record_consent", granted=True), call("save_person_facts", name="Andreas", email="andreas@maxbaum.com"),
                   "Let's confirm: a-n-d-r-e-a-s at maxbaum dot com?"],
                  ["Great, saved."]])
    r = s.turn("Yes, it's andreas at maxbaum.com")
    assert r["evidence"][1]["result"]["note"].startswith("email held back") and not mem.list_people()[0].get("email")
    s.turn("exactly", r["encounter_id"])  # the model did not call the tool again: the code saves it on the confirmation
    assert mem.list_people()[0]["email"] == "andreas@maxbaum.com"


def test_the_summary_covers_the_whole_conversation():
    s, mem = svc([[call("record_consent", granted=True), call("save_person_facts", name="Andreas", encounter_summary="Met Andreas."),
                   "Thanks!"], ["Graphs help with traceability."]],
                 extract_fn=lambda t: {"conversation_summary": "Andreas evaluates knowledge graphs for aircraft manufacturing."}
                 if "aircraft" in t else {})
    r = s.turn("Yes")
    s.turn("We are figuring out why to use knowledge graphs in aircraft manufacturing.", r["encounter_id"])
    enc = mem.get_person(mem.list_people()[0]["person_id"])["encounters"][0]
    assert enc["summary"] == "Andreas evaluates knowledge graphs for aircraft manufacturing."


def test_booth_facts_come_only_from_booth_info():
    from puppy_mind.wizard import system_prompt

    assert "refer them to the Neo4j staff" in system_prompt("PuppyPi", "the event", "")
    p = system_prompt("PuppyPi", "the event", "", booth_info="Demos: GraphRAG chatbot at table 1.")
    assert "BOOTH INFO: Demos: GraphRAG chatbot at table 1." in p and "never make up demos" in p


def test_an_email_heard_in_the_visitors_words_is_held_then_saved_on_the_next_confirmation():
    # replay of the booth: the model did not register the email itself; "exactly" must save it, "that's it" (a goodbye) must not
    from puppy_mind.people import confirms

    assert confirms("exactly") and confirms("yes, correct") and not confirms("That's it. Thank you so much.")
    s, mem = svc([[call("record_consent", granted=True), call("save_person_facts", name="Andreas"), "Thanks!"],
                  ["Let me repeat: a-n-d-r-e-a-s dot b-a-u-m at maxbaum dot com. Correct?"], ["Saved!"]],
                 extract_fn=lambda t: {"email": "andreas.baum@maxbaum.com"} if "maxbaum" in t else {})
    r = s.turn("Yes")
    s.turn("So it's andreas.baum at maxbaum.com", r["encounter_id"])
    assert not mem.list_people()[0].get("email")
    s.turn("exactly", r["encounter_id"])
    assert mem.list_people()[0]["email"] == "andreas.baum@maxbaum.com"


def test_each_lead_gets_a_next_step_the_team_can_take_over():
    s, mem = svc([[call("record_consent", granted=True), call("save_person_facts", name="Andreas", company="Maxbaum"), "Thanks!"],
                  ["Graphs help with traceability."], ["Sure."]],
                 extract_fn=lambda t: {"next_step": "Send the aerospace case study" if "aircraft" in t else "Share Graph Academy links"})
    r = s.turn("Yes")
    s.turn("We evaluate knowledge graphs for aircraft manufacturing.", r["encounter_id"])
    p = mem.list_people()[0]
    assert p["next_step"] == "Send the aerospace case study" and p["next_step_by"] == "ai"
    assert s.set_next_step(p["person_id"], "Call him Tuesday, Anna owns it")  # edited in the People tab mid-conversation
    s.turn("Anything else?", r["encounter_id"])  # the AI keeps suggesting, but the team's words win
    p = mem.get_person(p["person_id"])
    assert p["next_step"] == "Call him Tuesday, Anna owns it" and p["next_step_by"] == "team"



def test_forgetting_someone_mid_conversation_is_not_undone_by_a_later_save():
    s, mem = svc([[call("record_consent", granted=True), call("save_person_facts", name="Ana", company="Acme"), "Thanks!"],
                  ["Sure."]], extract_fn=lambda t: {"role": "CTO"})
    r = s.turn("Yes")
    pid = mem.list_people()[0]["person_id"]
    assert s.forget(pid)  # "Forget this person" clicked while she is still at the robot
    s.turn("I'm the CTO, by the way", r["encounter_id"])
    assert mem.list_people() == []



def test_yes_at_the_end_saves_everything_said_before_and_ends_politely():
    s, mem = svc([["Contract graphs are a great fit."], ["Great talking to you!"], ["Thank you, Andreas, have a great day!"]],
                 extract_fn=lambda t: {"name": "Andreas Rosenberg", "company": "Quadrant",
                                       "use_case": "process electronics documents" if "documents" in t else ""})
    r = s.turn("I'm Andreas Rosenberg from Quadrant. We process a lot of electronics documents.")
    assert mem.list_people() == []
    r = s.turn("That's it, thank you so much.", r["encounter_id"])
    assert "may I remember you as Andreas Rosenberg" in r["reply"]
    r = s.turn("Yes, thank you very much.", r["encounter_id"])  # seen at the booth: a 'thank you very much' ended the talk at the yes
    p = mem.list_people()[0]
    assert r["consent"] is True and p["name"] == "Andreas Rosenberg" and p["company"] == "Quadrant"
    assert p["use_case"] == "process electronics documents" and p["wants_follow_up"] is True
    assert r["reply"] == "Thank you, Andreas, have a great day!"  # leaving: no more questions



def test_asked_mid_way_it_says_by_the_way_and_asks_once_more_at_the_goodbye():
    from puppy_mind.people import CONSENT_AFTER_TURNS

    s, _ = svc([["Answer."]] * (CONSENT_AFTER_TURNS + 3))
    r = None
    for i in range(CONSENT_AFTER_TURNS):
        r = s.turn(f"question {i}", r["encounter_id"] if r else None)
    assert "By the way: may I remember" in r["reply"]  # not "before you go" while they are still asking things
    r = s.turn("one more question about ontologies", r["encounter_id"])
    assert "may I remember" not in r["reply"]
    r = s.turn("That is it for now, see you", r["encounter_id"])
    assert "Before you go: may I remember" in r["reply"]  # once more, now that they leave
    r = s.turn("one last thing", r["encounter_id"])
    assert "may I remember" not in r["reply"]
