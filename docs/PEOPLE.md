# People, the Graph Wizard and booth leads

At a conference the PuppyPi becomes the **Graph Wizard**: it greets people it sees, talks with them through the WonderEcho Pro,
answers questions about Neo4j, GenAI and GraphRAG, and, **only when a visitor says yes out loud**, remembers who they are in the
knowledge graph. Everything remembered shows up in the **People** tab as a lead list.

## What happens at the booth

1. **Greeting.** A person is in view (the vision model already reports `person`). The robot says hello once per `GREET_COOLDOWN_S`
   and never while someone is already talking to it. It does not ask the camera who the person is.
2. **Recognition (no faces).** A returning visitor is recognised by what they *say* ("Hi, I'm Ana Berg, I was here earlier") or, if
   you set `PEOPLE_CONSENT=true` (signage says badges may be read), by the **name badge they wear**, read from the frame. Then:
   "Welcome back, Ana! Great to see you again from Acme. Last time we talked about graphrag."
3. **Conversation.** The Graph Wizard answers Neo4j / GenAI / GraphRAG questions in short spoken sentences. For questions about the
   room ("what's on the table?") it asks its own spatial memory (the existing grounded agent), so it never guesses.
4. **Consent, then memory.** Over a few turns it learns name, role, function, company, what they are building, how well they know
   Neo4j and what they enjoyed most at the event. It asks first: *"May I remember your name, company and interests so the Neo4j team
   can follow up?"* Only a clear yes in the visitor's own words unlocks saving (see below). No: it carries on and saves nothing.
5. **Lead.** Each saved person gets a transparent score and stage (cold / warm / hot). See `lead_score` in
   [`people.py`](../src/puppy_mind/people.py): role seniority, a stated use case, interest in graph / GraphRAG / GenAI topics, a
   confirmed email, and above all asking for a follow-up or demo.

## Booth mode: looking up and a trick hello

Switch it on in the People tab (or `BOOTH_MODE=true`). Two things change:

- **It looks up.** The PuppyPi's camera is fixed to its body, so the robot tilts its whole body nose-up by `LOOK_UP_DEG` (default
  10°, capped at 20°). That puts faces and chest-height badges in frame instead of knees. It uses Hiwonder's own pose topic: pitch
  is in radians, nose-up is positive (their stock "LookDown" pose is −15°), and their node rejects anything beyond ±31°. Object
  memory still works, but low objects (on the floor, under tables) are seen less: switch booth mode off for room tours.
- **It greets with a trick.** `GREET_ACTION=boxing` sits the dog back on its hind legs and pumps both front paws. That's Hiwonder's
  stock `boxing` action group, run through `/puppy_control/runActionGroup` as `boxing.d6ac`. Others: `wave`, `shake_hands`, `bow`,
  `nod`, or empty for none. Visitors can also ask for one ("can you box?"): the Wizard has a `do_trick` tool.

Tricks are physical, so `drive.py` guards them: they work only in booth mode and never while the robot walks or turns. At most one
runs per `GESTURE_COOLDOWN_S` (60 s) and `MAX_GESTURES_PER_HOUR` (30) per hour, which limits servo heat and battery. While a trick
runs, every velocity command is ignored, so a key press can't make a dog that's up on its hind legs walk. Afterwards the robot stands
back in its pose. A greeting that falls inside the cooldown is spoken without a trick.

> Boxing shifts all the weight onto the back legs: use it on a flat, non-slippery floor with a clear space around the robot. The
> action names come from Hiwonder's `ros2` branch (`large_models/action.py`). The ROS1 image's service type is assumed to match and
> is unverified.

## A busy booth: whom to greet

With many people around, the robot must not greet everyone or talk over itself. When a person is in view and a greeting is allowed,
**one** vision call lists who is there, and a fixed policy (`choose_greeting`, unit-tested) decides:

| Situation | What it does |
|---|---|
| People walking past, or standing far away | Nothing: only people **close (≈2 m) and facing the robot** count |
| One or two at the booth | Greets the person **in front** (centre), otherwise a **returning visitor** first |
| `GROUP_MIN` (3) or more at the booth | **One group hello**: "Hello everyone! … One at a time, please!" |
| Already talking to someone | No greetings at all until the conversation has been quiet for 60 s |
| Same badge name seen again | Not re-greeted within `PERSON_REGREET_S` (30 min) |
| Always | At most one greeting per `GREET_COOLDOWN_S` (60 s) and `MAX_GREETINGS_PER_HOUR` (40) per hour |

The vision call runs only after the cheap checks pass, so a packed hall doesn't mean a vision call per frame.

**Voices in a crowd.** The microphone hears whoever speaks. If a different person takes over ("Hi, I'm Bob, my turn"), the Wizard
calls `new_visitor`, which starts a clean conversation: the previous visitor's facts and consent don't carry over, and Bob is asked
for consent himself. This is what keeps one person's details out of another's lead record. A close-talking position (people speak
to the dog from about 1 m) and a higher `--threshold` in `puppy_voice.py` help the robot ignore the hall.

## The graph

```
(Person {personId,name,role,function,company,email,useCase,familiarity,wantsFollowUp,wantsDemo,followedUp,
         leadScore,leadStage,consentAt,firstMet,lastMet,meetCount})
(Person)-[:WORKS_AT]->(Company)      (Person)-[:INTERESTED_IN]->(Topic)      (Person)-[:ENJOYED]->(Topic)
(Person)-[:HAD]->(Encounter {id,ts,summary})-[:AT]->(Place)
```

Topics and companies are shared nodes, so you can ask the graph things a CRM cannot: *who else is interested in GraphRAG at fintechs,
and who did the robot meet in the same room as Ana?* The Place link ties people to the spatial memory the robot already keeps.

```cypher
// the hot leads and what they care about
MATCH (p:Person {leadStage:'hot'}) OPTIONAL MATCH (p)-[:WORKS_AT]->(c) OPTIONAL MATCH (p)-[:INTERESTED_IN]->(t)
RETURN p.name, c.name, collect(t.name) AS interests ORDER BY p.leadScore DESC

// people who share an interest (introductions, follow-up clusters)
MATCH (a:Person)-[:INTERESTED_IN]->(t:Topic)<-[:INTERESTED_IN]-(b:Person) WHERE a.personId < b.personId
RETURN t.name, a.name, b.name
```

## The People tab

Table of leads (filter by stage, search name / company / topic), a mini graph of people, companies and topics, a detail panel
(everything remembered, conversation summaries), **Mark followed up**, **Forget this person**, and **Export CSV** for your CRM. Below
it, a typed chat box that talks to the same brain as the voice, so you can try the whole flow without a robot.

## Privacy, enforced in code

| Rule | Where |
|---|---|
| Nothing is saved without a clear yes from the visitor's *own last words*; the model cannot grant consent itself | `consent_from`, `record_consent`, `save_person_facts` |
| Raw conversation is never stored: only extracted facts and a one-sentence summary per encounter | `PeopleService.turn` keeps the transcript in RAM only, expiring after `ENCOUNTER_IDLE_S` |
| No face recognition. Badge reading is opt-in (`PEOPLE_CONSENT=true`), and a badge name is "unverified" until the visitor confirms | `_greet`, `Badge` |
| The robot never reads out an email, notes or lead score; a returning visitor's record is limited to role, company, topics | `safe_view` |
| "Forget me" erases the person and their conversations (voice or the button); orphaned topics and companies are cleaned up | `forget_person` |
| People endpoints answer only this machine unless `BRAIN_TOKEN` is set and passed (`X-Brain-Token` or `?token=`) | `_pii_auth` |
| The CSV neutralises spreadsheet formulas in spoken text | `people_export` |

Before using it for real: put up a sign saying the robot talks to visitors and may remember them if they agree, and have your legal
team confirm what GDPR/CCPA notice and retention you need. The score is a hint for sales, not a decision about a person.
One more thing to know: speech is sent to OpenAI for transcription and the conversation text for the reply, like camera frames are.

## Setting up the voice

The WonderEcho Pro appears to the Pi as a USB audio device. `robot/puppy_voice.py` records with `arecord`, plays with `aplay`, and
sends audio to the laptop; the Pi does no recognition or synthesis.

```bash
# on the robot (inside or outside the container, wherever arecord/aplay see the USB card)
arecord -l ; aplay -l                                   # note the WonderEcho Pro card number
wget http://<laptop-ip>:8000/robot/puppy_voice.py
python3 puppy_voice.py --brain http://<laptop-ip>:8000 --test-mic --mic plughw:1,0      # watch the loudness bars, pick --threshold
python3 puppy_voice.py --brain http://<laptop-ip>:8000 --mic plughw:1,0 --speaker plughw:1,0 --token $BRAIN_TOKEN
```

Laptop side: nothing extra, `.env` needs `OPENAI_API_KEY`. Optional: `EVENT_NAME`, `OPENAI_TTS_VOICE`, `GREET_COOLDOWN_S`, `PEOPLE_GREET=false`
(never greet unprompted). The greeting itself is spoken on the next poll after the camera sees someone, so start the camera too.

> **Not yet verified on the robot.** The code is unit-tested against a scripted model and a fake memory (`tests/test_people*.py`),
> and the Cypher follows the repo's patterns, but it has not been run against Aura or with the WonderEcho Pro attached. Expect to
> tune `--threshold` / `--silence`, and run `uv run puppy-mind init` once so Aura gets the new constraints.
> If the WonderEcho Pro's own wake-word firmware is active it may take the microphone: use it in plain USB-audio mode.

## Not built (on purpose)

- **Face recognition.** It would make recognition seamless but it is biometric data; if you want it, it needs its own explicit opt-in
  and a legal review, not a quiet default. The badge and "say your name" paths cover the booth case.
- **CRM push.** Export CSV is the integration point; a Salesforce/HubSpot connector is a small step from `LEAD_COLUMNS` in `server.py`.
- **Retrieval-grounded Wizard answers.** The Wizard answers from the model plus a careful primer (`wizard.py`) and is told to admit
  uncertainty. For doc-grounded answers, index the Neo4j docs with the GraphRAG pipeline and add a `search_docs` tool.
