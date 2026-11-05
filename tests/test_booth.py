"""The booth host: anonymous notes, greeting, question timing. No network: a scripted fake chat model drives the tools.

The visitors in these tests are made up. Real booth sentences were replayed during development but are not kept here."""
import dataclasses
import json
import time
from types import SimpleNamespace as NS

from puppy_mind.booth import (QUESTIONS, BoothHost, asks_something, choose_greeting, clean_topics, drop_own_ask, greeting,
                              identifying_words, scrub)
from puppy_mind.config import get_settings
from puppy_mind.devfake import FakeMemory
from puppy_mind.models import Detection, Scene

S = dataclasses.replace(get_settings(), nams_api_key="", greet_cooldown_s=0.2, people_greet=True,
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


def host(script, **kw):
    mem = FakeMemory(S)
    kw.setdefault("extract_fn", lambda transcript: {})  # no automatic notes unless a test scripts them
    h = BoothHost(S, mem, client=FakeChat(script), **kw)
    h.learn_in_background = h.write_in_background = False
    return h, mem


def person_scene(conf=0.9):
    return Scene(summary="a person", room_type="unknown", room_confidence=0.0, relations=[],
                 objects=[Detection(idx=0, label="person", color="unknown", description="", zone="center", confidence=conf)])


def wait(pred, t=2.0):
    end = time.time() + t
    while time.time() < end and not pred():
        time.sleep(0.01)
    return pred()


def V(zone="center", facing=True, close=True):
    return {"zone": zone, "facing_robot": facing, "close": close}


def look(*people):
    return lambda jpeg: [dict(p) for p in people]


# ---------------------------------------------------------------- nothing that identifies a visitor is kept

def test_the_scrubber_removes_what_the_visitor_said_about_who_they_are():
    said = ["Hi, my name is Lena Hart and I work at Brightwave Analytics.", "Mail me at lena.hart@brightwave.io or +49 170 1234567."]
    idw = identifying_words(said)
    assert {"lena hart", "brightwave analytics"} <= idw and "lena" in idw and "brightwave" in idw
    assert scrub("Lena wants to process contracts at Brightwave with a knowledge graph", idw) == \
        "wants to process contracts with a knowledge graph"
    assert scrub("contact lena.hart@brightwave.io or +49 170 1234567, linkedin.com/in/lenahart", set()) == "contact or"
    assert clean_topics(["GraphRAG", "brightwave analytics", "lena@x.io", "knowledge graphs", "graphrag", "knowledge graph"], idw) == \
        ["graphrag", "knowledge graphs"]
    assert identifying_words(["I'm curious about GraphRAG", "I am just looking", "We are building a chatbot"]) == set()
    # seen at the booth: "key graph algorithms" next to "graph algorithms"; one word ("graphs") is too general to merge into
    assert clean_topics(["key graph algorithms", "graph algorithms", "graphs", "knowledge graphs", "graph algorithms in pharma"],
                        general=True) == ["graph algorithms", "graphs", "knowledge graphs"]


def test_a_conversation_is_kept_as_an_anonymous_note_scrubbed_in_code():
    def extract(transcript):  # a careless extraction that copied what the visitor said about themselves
        return {"topics": ["Contract management", "knowledge graphs", "Brightwave"], "need": "Lena at Brightwave wants to process "
                "thousands of contracts", "familiarity": "curious", "enjoyed": [], "questions": ["What is GraphRAG, Lena asked"],
                "summary": "Lena Hart from Brightwave asked about contracts. Mail lena.hart@brightwave.io"}

    h, mem = host([["Contracts fit graphs well."]], extract_fn=extract)
    h.turn("Hey PuppyPi, my name is Lena Hart and I work at Brightwave. Can graphs handle thousands of contracts?")
    [note] = mem.list_conversations()
    kept = json.dumps(note).lower()
    assert "lena" not in kept and "brightwave" not in kept and "@" not in kept and "hart" not in kept
    assert note["topics"] == ["contract management", "knowledge graphs"] and note["familiarity"] == "curious"
    assert note["need"] == "wants to process thousands of contracts"
    assert not hasattr(mem, "list_people") and not hasattr(mem, "save_person")  # there is nowhere to put a person


def test_the_robot_never_asks_who_someone_is_and_drops_the_models_own_asks():
    for own in ("May I remember you so the Neo4j team can follow up?", "What's your name, by the way?", "Which company are you with?",
                "And what is your role there?", "Could you give me your email?", "What have you enjoyed most at the conference so far?"):
        assert drop_own_ask("Graphs connect facts. " + own) == "Graphs connect facts.", own
    assert drop_own_ask("Graphs connect facts. Want to hear about GraphRAG?") == "Graphs connect facts. Want to hear about GraphRAG?"
    assert "remember" not in drop_own_ask("May I remember you?")  # the whole reply
    assert not any(w in " ".join(QUESTIONS.values()).lower() for w in ("name", "company", "role", "email"))
    h, _ = host([["Hello! What's your name?"]])
    r = h.turn("Hey PuppyPi")
    assert r["reply"] == "Hello! " + QUESTIONS["use_case"]


def test_the_wizard_is_told_it_collects_no_personal_data():
    from puppy_mind.wizard import system_prompt

    p = system_prompt("PuppyPi", "the conference", "")
    assert "never collect personal data" in p and "Never ask for anyone's name" in p and "do not remember people" in p
    assert "at most 2 short sentences" in p and "AT MOST ONCE per conversation" in p and "not mishearings" in p
    assert "remember about visitors" not in p and "PERMISSION" not in p


def test_new_visitor_starts_a_clean_note_so_conversations_do_not_mix():
    notes = iter([{"topics": ["graphrag"]}, {"topics": ["cypher"]}])
    h, mem = host([["GraphRAG grounds answers."], [call("new_visitor"), "Hi! Cypher is the query language."]],
                  extract_fn=lambda t: next(notes))
    r = h.turn("What is GraphRAG?")
    r2 = h.turn("My turn! What is Cypher?", r["encounter_id"])
    assert r2["encounter_id"] != r["encounter_id"]
    assert sorted(c["topics"][0] for c in mem.list_conversations()) == ["cypher", "graphrag"]


def test_the_latest_notes_win_while_they_cover_the_whole_conversation_and_a_hello_alone_keeps_nothing():
    # each extraction reads the whole conversation so far; adding them up kept every guess (a replay collected questions nobody
    # asked), so the latest one replaces the lists, and an empty answer never erases what was found
    notes = iter([{}, {"topics": ["graphrag"], "questions": ["what is graphrag", "a question nobody asked"]},
                  {"topics": ["graphrag", "agent memory"], "questions": ["what is graphrag", "what is agent memory"],
                   "need": "a memory for support agents"}, {}])
    h, mem = host([["Hi!"], ["It grounds answers."], ["It keeps context."], ["Bye!"]], extract_fn=lambda t: next(notes))
    r = h.turn("Hey PuppyPi")
    assert mem.list_conversations() == []  # nothing worth keeping yet
    for t in ("What is GraphRAG?", "And agent memory?", "Thanks, bye!"):
        r = h.turn(t, r["encounter_id"])
    [c] = mem.list_conversations()
    assert c["topics"] == ["graphrag", "agent memory"] and c["questions"] == ["what is GraphRAG", "what is agent memory"]
    assert c["need"] == "a memory for support agents" and c["turns"] == 4


def test_booth_insights_count_topics_needs_and_familiarity():
    mem = FakeMemory(S)
    mem.save_conversation({"id": "a", "topics": ["graphrag", "vector search"], "need": "chat over manuals", "familiarity": "curious",
                           "questions": ["q1"], "enjoyed": ["robot demo"]}, place=None, ts="2026-11-02T10:00:00Z")
    mem.save_conversation({"id": "b", "topics": ["graphrag"], "familiarity": "user", "questions": ["q2", "q3"]}, place=None, ts="2026-11-02T11:00:00Z")
    i = mem.booth_insights()
    assert (i["conversations"], i["questions"], i["topics"]) == (2, 3, 2)
    assert i["top_topics"][0] == {"name": "graphrag", "count": 2} and i["familiarity"]["curious"] == 1 and i["familiarity"]["user"] == 1
    assert [n["need"] for n in i["needs"]] == ["chat over manuals"] and i["top_enjoyed"] == [{"name": "robot demo", "count": 1}]
    g = mem.booth_graph()  # seen at the booth: topics alone, linked to each other, "missed the relationships"
    kinds = {n["id"]: n["kind"] for n in g["nodes"]}
    assert kinds == {"robot": "Robot", "conv:a": "Conversation", "conv:b": "Conversation", "topic:graphrag": "Topic",
                     "topic:vector search": "Topic", "topic:robot demo": "Topic"}
    assert {n["id"]: n["count"] for n in g["nodes"] if n["kind"] == "Topic"} == {"topic:graphrag": 2, "topic:vector search": 1, "topic:robot demo": 1}
    assert {(e["source"], e["type"], e["target"]) for e in g["edges"]} == {
        ("robot", "HOSTED", "conv:a"), ("robot", "HOSTED", "conv:b"), ("conv:a", "DISCUSSED", "topic:graphrag"),
        ("conv:a", "DISCUSSED", "topic:vector search"), ("conv:a", "ENJOYED", "topic:robot demo"), ("conv:b", "DISCUSSED", "topic:graphrag")}
    assert next(n for n in g["nodes"] if n["id"] == "conv:a")["label"] == "chat over manuals"
    assert mem.delete_conversation("a") and mem.booth_insights()["conversations"] == 1


# ---------------------------------------------------------------- questions about their interests, at the right moment

def test_the_robot_ends_with_one_interest_question_and_never_repeats_it():
    h, _ = host([["Hello!"], ["Graphs connect things. How are you liking it?"], ["Contracts fit graphs well."], ["Have a great day!"]])
    r = h.turn("Hey PuppyPi")
    assert r["reply"] == "Hello! " + QUESTIONS["use_case"]
    r = h.turn("Tell me about graphs", r["encounter_id"])
    assert r["reply"] == "Graphs connect things. " + QUESTIONS["familiarity"]  # the model's own question is replaced
    assert "one question about their interests is added" in h.client.prompts[-1]
    r = h.turn("I want to process thousands of contracts", r["encounter_id"])
    assert r["reply"] == "Contracts fit graphs well."  # nothing left to ask before the wind-down
    r = h.turn("I need to run now to the next one. Thank you very much, PuppyPi.", r["encounter_id"])
    assert r["reply"] == "Have a great day!" and "They are leaving" in h.client.prompts[-1]
    # seen at the booth: after the goodbye it stayed "in a conversation"; now it is over, and only the wake word starts the next one
    assert not h.conversation_active() and h.unclear_reply(r["encounter_id"]) is None
    assert h.addressed("what about vector search", r["encounter_id"]) == (False, None)
    assert h.addressed("Hey PuppyPi, one more thing", r["encounter_id"]) == (True, None)


def test_a_conversation_ends_when_they_leave_and_what_they_enjoyed_is_asked_only_near_the_end():
    said = ["Hey PuppyPi!", "Tell me more about artificial intelligence.", "We want to see how two partners can build on graphs.",
            "Just getting started, I created an Aura instance already.", "There were very interesting keynotes on agentic AI.",
            "Tell me more about knowledge graphs. Why should we use them?", "What are common use cases?",
            "Why should I use Neo4j compared to SQL?", "Okay, interesting.", "Okay, thank you so much. I need to run to the next one."]
    assert all(asks_something(t) for t in said[5:8]) and not asks_something(said[3])
    h, _ = host([["Answer."]] * len(said))
    r, replies = None, []
    for t in said:
        r = h.turn(t, r["encounter_id"] if r else None)
        replies.append(r["reply"])
    enjoyed = QUESTIONS["enjoyed"].format(event=S.event_name)
    assert [i for i, x in enumerate(replies) if enjoyed in x] == [8]  # the 9th answer: not mid-questions, not on the goodbye
    assert replies[-1] == "Answer."  # leaving: no question at all
    assert not any("remember" in x.lower() for x in replies)


def test_do_trick_tool_uses_the_rate_limited_gesture():
    h, _ = host([[call("do_trick", name="boxing"), "Ta-da!"]], gesture_fn=lambda n, **kw: {"ok": False, "reason": "tricks are rate-limited"})
    r = h.turn("can you do boxing?")
    assert r["evidence"][0]["result"]["reason"] == "tricks are rate-limited"


def test_booth_facts_come_only_from_booth_info():
    from puppy_mind.wizard import system_prompt

    assert "refer them to the Neo4j staff" in system_prompt("PuppyPi", "the event", "")
    p = system_prompt("PuppyPi", "the event", "", booth_info="Demos: GraphRAG chatbot at table 1.")
    assert "BOOTH INFO: Demos: GraphRAG chatbot at table 1." in p and "never make up demos" in p


def test_the_booth_language_is_fixed_for_hearing_and_answering():
    from puppy_mind.voice import Voice
    from puppy_mind.wizard import system_prompt

    calls = []
    client = NS(audio=NS(transcriptions=NS(create=lambda **kw: calls.append(kw) or NS(text="hello"))))
    Voice(dataclasses.replace(S, voice_language="en"), client=client).transcribe(b"x", "s.wav")
    assert calls[0]["language"] == "en"  # the speech model never guesses the language from a noisy clip
    assert "Always answer in English" in system_prompt("PuppyPi", "the event", "")
    assert "Always answer in German" in system_prompt("PuppyPi", "the event", "", "de")
    h, _ = host([["Hello!"]])
    h.turn("hi")
    assert "Always answer in English" in h.client.prompts[-1]


# ---------------------------------------------------------------- greeting people the camera sees (never who they are)

def test_greeting_text():
    assert "Graph Wizard" in greeting() and 'Say "Hey PuppyPi"' in greeting("PuppyPi", "Hey PuppyPi")


def test_crowd_policy_ignores_passers_by_and_groups_a_crowd():
    assert choose_greeting([V(facing=False), V(close=False), V(zone="left", facing=False, close=False)], False) is None
    assert choose_greeting([V(), V(zone="left"), V(zone="right"), V(facing=False)], False) == "group"
    assert choose_greeting([V(), V(zone="left")], False, group_min=3) == "single"
    assert choose_greeting([V()], True) is None  # greeted recently


def test_person_in_view_greets_once_with_a_trick_then_cooldown():
    tricks = []
    h = BoothHost(S, FakeMemory(S), client=None, look_fn=look(V()), gesture_fn=lambda n: tricks.append(n) or {"ok": True, "action": n})
    assert h.on_view(person_scene(), b"x") is True
    assert wait(lambda: h.outbox._items)
    assert h.outbox.next()["text"].startswith("Hi there!") and tricks == ["boxing"]
    assert h.on_view(person_scene(), b"x") is False  # cooldown
    assert h.on_view(person_scene(conf=0.1), b"x") is False  # not confident it is a person
    assert h.on_view(Scene(summary="", room_type="unknown", room_confidence=0, objects=[], relations=[]), None) is False


def test_passers_by_are_not_greeted_and_the_hourly_cap_holds():
    h = BoothHost(S, FakeMemory(S), client=None, look_fn=look(V(facing=False), V(close=False)))
    h.on_view(person_scene(), b"x")
    assert wait(lambda: h._greeting.acquire(blocking=False) and (h._greeting.release() or True))
    assert h.outbox.next() is None
    capped = BoothHost(dataclasses.replace(S, max_greetings_per_hour=1, greet_cooldown_s=0, person_regreet_s=0), FakeMemory(S),
                       client=None, look_fn=look(V()))
    capped.on_view(person_scene(), b"x")
    assert wait(lambda: capped.outbox._items)
    assert capped.on_view(person_scene(), b"x") is False


def test_a_crowd_gets_one_group_hello():
    h = BoothHost(S, FakeMemory(S), client=None, look_fn=look(V(), V(zone="left"), V(zone="right"), V(facing=False)))
    h.on_view(person_scene(), b"x")
    assert wait(lambda: h.outbox._items)
    assert h.outbox.next()["text"].startswith("Hello everyone!") and h.outbox.next() is None


def test_no_greeting_while_someone_is_talking_and_a_greeting_carries_into_the_conversation():
    h, _ = host([["Hello!"]], look_fn=look(V()))
    h.on_view(person_scene(), b"x")
    assert wait(lambda: h.outbox._items)
    h.turn("hello")
    assert "already greeted" in h.client.prompts[-1]
    assert h.on_view(person_scene(), b"x") is False


def test_a_person_in_view_who_is_not_greeted_is_logged_with_the_reason():
    events = []
    bus = NS(publish=lambda kind, d: events.append((kind, d)))
    h, _ = host([["Hello!"]], bus=bus, look_fn=look(V(facing=False)))
    h.on_view(person_scene(), b"x")  # seen from the side: looked at, not greeted
    assert wait(lambda: any(d.get("skipped") for k, d in events if k == "greeting"))
    assert [d["reason"] for k, d in events if k == "greeting"] == ["nobody close to the robot and turned toward it"]
    h.turn("hi")
    h._skip_note = -1e9
    h.on_view(person_scene(), b"x")
    assert events[-1][1]["reason"] == "already talking to someone"
    h.on_view(person_scene(), b"x")  # the same reason again within 20 s is not repeated
    assert sum(1 for k, d in events if d.get("reason") == "already talking to someone") == 1


def test_the_look_prompt_knows_the_camera_is_at_floor_level_and_never_reads_badges():
    from puppy_mind.booth import LOOK_PROMPT, PersonView

    assert "FLOOR LEVEL" in LOOK_PROMPT and "feet" in LOOK_PROMPT and "Sitting people count" in LOOK_PROMPT
    assert "Never read name badges" in LOOK_PROMPT and set(PersonView.model_fields) == {"zone", "facing_robot", "close"}


def test_a_visitor_can_end_the_conversation_in_more_words_and_the_tab_can_end_it_too():
    from puppy_mind.booth import says_goodbye

    for done in ("Thank you and goodbye.", "Thanks!", "Okay, thank you so much.", "Thank you, PuppyPi.", "Stop.", "End the conversation.",
                 "That's all for now.", "I need to run to the next one."):
        assert says_goodbye(done), done
    for going_on in ("Thanks, and what about vector search?", "How do I stop a running Cypher query in Aura?", "Thank you for the "
                     "explanation of GraphRAG, can you also explain agent memory?", "Tell me about Neo4j context graph."):
        assert not says_goodbye(going_on), going_on
    h, _ = host([["Hello!"]])
    r = h.turn("Hey PuppyPi")
    assert h.conversation_active() and h.end_conversations() == 1 and not h.conversation_active()
    assert h.addressed("what about vector search", r["encounter_id"]) == (False, None)


def test_the_speech_models_own_word_list_is_not_a_visitor():
    from puppy_mind.voice import VOCAB_HINT, heard_nothing

    # seen on the robot: noise came back as the prompt itself, which starts with "Hey PuppyPi", and opened a conversation
    assert heard_nothing(VOCAB_HINT) and heard_nothing("Hey PuppyPi, Neo4j, Cypher, GraphRAG, GraphAcademy, Aura, AuraDB, GenAI.")
    assert not heard_nothing("Hey PuppyPi, what is GraphRAG?") and not heard_nothing("How do Cypher and GraphRAG work with Aura?")


def test_misheard_neo4j_terms_are_fixed_before_the_wizard_and_the_note_see_them():
    from puppy_mind.voice import fix_terms

    # seen on the robot: "agent memory" came out as "Asian memory", GraphAcademy as "graph economy"
    assert fix_terms("I'd like to learn more about Asian memory") == "I'd like to learn more about agent memory"
    assert fix_terms("Do you recommend any courses on Graph economy?") == "Do you recommend any courses on GraphAcademy?"
    assert fix_terms("graph rag with neo for j and cipher") == "GraphRAG with Neo4j and Cypher"
    assert fix_terms("I enjoyed the keynotes on graphs") == "I enjoyed the keynotes on graphs"  # nothing else changes
    notes = iter([{"topics": ["Asian memory"], "questions": ["courses on graph economy"], "need": ""}])
    h, mem = host([["Agent memory keeps context."]], extract_fn=lambda t: next(notes))
    h.turn("Tell me about Asian memory. Which courses on graph economy?")
    [c] = mem.list_conversations()
    assert c["topics"] == ["agent memory"] and c["questions"] == ["courses on GraphAcademy"]
    assert h.encounters[c["id"]].messages[0]["content"] == "Tell me about agent memory. Which courses on GraphAcademy?"


def test_a_note_keeps_only_what_the_visitor_actually_said():
    from puppy_mind.booth import _content_words, grounded

    # seen on the robot: "Hey PuppyPi" + "What is a graph?" became the need "build a knowledge graph for agent memory"
    said = _content_words("Hey PuppyPi! What is a graph?")
    assert grounded("graphs", said) and grounded("what is a graph", said)
    assert not grounded("agent memory", said) and not grounded("build a knowledge graph for agent memory", said, 2)
    notes = iter([{"topics": ["knowledge graphs", "agent memory", "graphs"], "need": "build a knowledge graph for agent memory",
                   "questions": ["what is a graph", "how to build agent memory"]}])
    h, mem = host([["A graph is nodes and relationships."]], extract_fn=lambda t: next(notes))
    h.turn("What is a graph?")
    [c] = mem.list_conversations()
    assert c["topics"] == ["knowledge graphs", "graphs"] and c["need"] is None and c["questions"] == ["what is a graph"]


def test_the_robot_voice_is_made_louder_without_clipping():
    # the WonderEcho Pro's speaker is already at 100% and too quiet in a conference hall
    import io
    import math
    import wave

    from puppy_mind.voice import louder

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000)
        w.writeframes(b"".join(int(6000 * math.sin(i / 7) * (0.2 if i < 12000 else 1)).to_bytes(2, "little", signed=True)
                               for i in range(24000)))
    quiet = buf.getvalue()

    def rms_peak(wav):
        with wave.open(io.BytesIO(wav)) as w:
            s = [int.from_bytes(w.readframes(1), "little", signed=True) for _ in range(w.getnframes())]
        return math.sqrt(sum(v * v for v in s) / len(s)), max(abs(v) for v in s)

    loud = louder(quiet, 2.5)
    (r0, p0), (r1, p1) = rms_peak(quiet), rms_peak(loud)
    assert r1 > 3 * r0 and p1 <= 32767 * 0.97 + 1  # much louder, never clipped
    assert louder(quiet, 1.0) == quiet and louder(b"not a wav", 2.5) == b"not a wav"


# ---- the claw machine cheerleader
CLAW = dataclasses.replace(S, claw_machine=True, claw_prize="a Graph Wizard", cheer_every_s=100)


def test_cheer_says_a_line_with_the_prize_never_the_same_twice_in_a_row_and_plays_no_trick():
    from puppy_mind.booth import CHEERS
    tricks = []
    h, _ = host([], gesture_fn=lambda n, manual=False: tricks.append((n, manual)) or {"ok": True, "action": n})
    h.s = CLAW
    said = []
    for _ in range(8):
        r = h.cheer("invite")
        said.append(h.outbox.next())
        assert r["ok"] and said[-1]["kind"] == "cheer" and said[-1]["text"] == r["text"]
    assert all(a["text"] != b["text"] for a, b in zip(said, said[1:]))
    assert any("Graph Wizard" in s["text"] for s in said) and "{prize}" not in "".join(s["text"] for s in said)
    for kind in ("playing", "won", "almost"):
        assert "trick" not in h.cheer(kind)
    assert tricks == []  # cheering is words only: no animation
    assert not h.cheer("dance")["ok"] and set(CHEERS) == {"invite", "playing", "won", "almost"}


def test_the_robot_calls_people_over_only_when_nobody_talks_to_it_for_a_while():
    h, _ = host([["Hi! Welcome to the booth."]])
    h.s = CLAW
    assert not h.maybe_invite(listening=True)  # it just started: waits CHEER_EVERY_S first
    h._last_said -= 101
    assert not h.maybe_invite(listening=False)  # not listening (no tab, LISTEN_ALWAYS off): stays quiet
    assert h.maybe_invite(listening=True) and h.outbox._items[0]["kind"] == "cheer"
    assert not h.maybe_invite(listening=True)  # the invite is still waiting to be played
    h.outbox.next()
    h._last_said -= 101
    h.turn("Hey PuppyPi, hello")  # someone talks to it: no invite in the middle of a conversation
    h.outbox._items.clear()
    h._last_said -= 101
    assert not h.maybe_invite(listening=True)
    h.s = dataclasses.replace(CLAW, claw_machine=False)
    for e in h.encounters.values():
        e.last -= 1000
    assert not h.maybe_invite(listening=True)  # switched off
    h.s = dataclasses.replace(CLAW, cheer_every_s=0)
    assert not h.maybe_invite(listening=True)  # 0 = only by button


def test_the_wizard_knows_the_claw_machine_and_mentions_it_at_goodbye():
    h, _ = host([["Hi!"], ["Thanks for stopping by!"]])
    h.s = CLAW
    h.turn("Hey PuppyPi, what can I do here?")
    assert "claw machine" in h.client.prompts[-1] and "Graph Wizard" in h.client.prompts[-1]
    h.turn("Thanks, bye!")
    assert "invite them to try the claw machine" in h.client.prompts[-1]


def test_a_short_thanks_ends_the_conversation_but_not_one_with_a_question():
    from puppy_mind.booth import robot_says_goodbye, says_goodbye
    # seen at the booth: both left the call open until the Booth tab ended it
    assert says_goodbye("Thank you, thank you, thank you") and says_goodbye("Thank you, thank you, thank you I'll try")
    assert not says_goodbye("Thanks, how does GraphRAG work?") and not says_goodbye("Thanks! Can you wave?")
    assert robot_says_goodbye("Bye! And good luck with that claw machine.")
    assert not robot_says_goodbye("Good luck! What are you building?") and not robot_says_goodbye("GraphRAG uses graphs.")
