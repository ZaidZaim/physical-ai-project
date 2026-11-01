# People, the Graph Wizard and booth leads

At a conference the PuppyPi becomes the **Graph Wizard**: it greets people it sees, talks with them through the WonderEcho Pro,
answers questions about Neo4j, GenAI and GraphRAG, and, **only when a visitor says yes out loud**, remembers who they are in the
knowledge graph. Everything remembered shows up in the **People** tab as a lead list.

## What happens at the booth

1. **Greeting.** A person is in view (the vision model already reports `person`). The robot says hello once per `GREET_COOLDOWN_S`,
   by voice only, and tells them how to start: *"Say 'Hey PuppyPi' and ask me anything…"*. It never greets while someone is already
   talking to it, and it does not ask the camera who the person is.
1. **Wake word.** The robot answers only when addressed: a conversation starts with **"Hey PuppyPi"** (or as a direct reply to its
   greeting), and follow-ups need no wake word until it has been quiet for `WAKE_WINDOW_S` (45 s). After that, the next "Hey PuppyPi"
   starts a fresh conversation, since it is most likely someone new. Everything else the microphone hears in the hall is ignored and
   not echoed anywhere. (It is still transcribed on the laptop to look for the wake word; an on-robot wake-word detector would avoid
   that, see "Not built".) `WAKE_WORD=` (empty) switches it off.
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
- **Tricks.** By default it greets by voice only. `GREET_ACTION=boxing` adds a trick to every greeting: boxing sits the dog back on
  its hind legs and pumps both front paws (Hiwonder's stock action group, run through `/puppy_control/runActionGroup` as `boxing.d6ac`;
  others: `wave`, `shake_hands`, `bow`, `nod`). Visitors can always ask for one ("can you box?"): the Wizard has a `do_trick` tool,
  and the People tab has **Try trick**.

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
(Person)-[:WORKS_AT]->(Company)      (Person)-[:INTERESTED_IN]->(Theme)      (Person)-[:ENJOYED]->(Theme)
(Person)-[:PARTICIPATED_IN]->(Encounter {id,ts,summary})-[:HELD_AT]->(Place)
```

**Same vocabulary as NAMS.** The ontology file ([`ontology/puppy_pi.json`](../ontology/puppy_pi.json)) names these with NAMS's POLE+O
types, and `puppy-mind nams-setup` sends that file to NAMS:

| Ontology / NAMS type | POLE+O | Aura label | Why a different label |
|---|---|---|---|
| Person | PERSON | `Person` | same label; the app's people carry a `personId`, NAMS's never do, and every app query checks it |
| Organization | ORGANIZATION | `Company` | NAMS writes an entity's type as a label in the same database; sharing a label would mix the two |
| Topic | OBJECT | `Theme` | as above, and our unique `name` would reject NAMS's `Topic` writes |
| Event | EVENT | `Encounter` | as above |

That's the same rule the rest of the model follows (Location is `Place`, Object `Item`, Agent `Robot`). The ontology test checks that
every label and relationship `people_graph.py` writes is declared, and that no Aura label equals a NAMS type name.

To give NAMS the new types, run `uv run puppy-mind nams-setup`. That creates a new ontology version and doesn't switch your workspace
to it; add `--activate` once you're happy, since the active ontology decides how NAMS extracts entities from every conversation.
People themselves are **not** copied into NAMS: they stay in one place, which `forget_me` erases completely.

Themes and companies are shared nodes, so you can ask the graph things a CRM can't: *who else is interested in GraphRAG at fintechs,
and who did the robot meet in the same room as Ana?* The Place link ties people to the spatial memory the robot already keeps.

```cypher
// the hot leads and what they care about
MATCH (p:Person {leadStage:'hot'}) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) OPTIONAL MATCH (p)-[:INTERESTED_IN]->(t:Theme)
RETURN p.name, c.name, collect(t.name) AS interests ORDER BY p.leadScore DESC

// people who share an interest (introductions, follow-up clusters)
MATCH (a:Person)-[:INTERESTED_IN]->(t:Theme)<-[:INTERESTED_IN]-(b:Person) WHERE a.personId < b.personId
RETURN t.name, a.name, b.name
```

## The People tab

Graph first. On top, a **lead counter** that counts up (with a "New lead: name · company" banner) whenever a visitor says yes, plus
hot / warm / cold and how many follow-ups are still open. The main area is the **graph**: each lead sized by lead score and coloured
by stage, linked to their company and the topics they care about, so shared interests are visible at a glance.

On the right, **Next steps for the Neo4j team**: one card per lead, highest score first, with the follow-up or call to action. The AI
drafts it from the conversation (e.g. *"Send the aerospace knowledge-graph case study and offer a 30-min call on traceability"*) and
keeps it current; once someone on the team edits it, it is marked "set by team" and the AI never overwrites it. Click a card or a
node for everything remembered, **Mark done**, and **Forget this person**. **Export CSV** includes the next step. Search filters both
the graph and the cards.

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

- **On-robot wake word.** "Hey PuppyPi" is checked on the laptop after transcription, so hall audio is still sent for transcription
  (and dropped). A small on-device detector on the Pi (for example openWakeWord with a custom "Hey PuppyPi" model) would send nothing
  until the wake word, which is better for privacy and cost at a busy booth.

- **Face recognition.** It would make recognition seamless but it is biometric data; if you want it, it needs its own explicit opt-in
  and a legal review, not a quiet default. The badge and "say your name" paths cover the booth case.
- **CRM push.** Export CSV is the integration point; a Salesforce/HubSpot connector is a small step from `LEAD_COLUMNS` in `server.py`.
- **Retrieval-grounded Wizard answers.** The Wizard answers from the model plus a careful primer (`wizard.py`) and is told to admit
  uncertainty. For doc-grounded answers, index the Neo4j docs with the GraphRAG pipeline and add a `search_docs` tool.
