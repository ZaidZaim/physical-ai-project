# The booth host: the Graph Wizard, anonymous by design

At a conference the PuppyPi is the booth's attraction, the **Graph Wizard**. "Hey PuppyPi" makes it do its boxing trick, look up
at the visitor and answer questions about Neo4j, GenAI and GraphRAG through the WonderEcho Pro. It keeps **what the booth talked
about** (topics, needs, questions, how well people know Neo4j) and never **who** it talked to. Everything shows up in the **Booth**
tab.

> **Lead generation is on hold.** Saving visitors' personal data (names, companies, roles, emails, lead scores, follow-ups) is not
> allowed for now: a compliance matter. That version, with spoken consent and a People tab of leads, is parked on branch
> `parked/booth-lead-generation` to pick up later.

## What happens at the booth

1. **Greeting.** A person is in view (the vision model already reports `person`). If they are close and turned toward the robot,
   it says hello once per `GREET_COOLDOWN_S`, by voice: *"Hi there! I'm PuppyPi, the Graph Wizard. Say 'Hey PuppyPi' and ask me
   anything…"*. It never greets while someone is talking to it, and the camera never reads badges or identifies anyone.
2. **Wake word.** The robot answers only when addressed: a conversation starts with **"Hey PuppyPi"** (or as a direct reply to its
   greeting), and follow-ups need no wake word until it has been quiet for `WAKE_WINDOW_S`. After that, the next "Hey PuppyPi"
   starts a fresh conversation. It listens only while a browser has the Booth tab open.
3. **Look up.** On the wake word it tilts nose-up by `ENGAGED_LOOK_UP_DEG` to look at the visitor, and levels again once the
   conversation goes quiet. `WAKE_ACTION` can add a trick first (none by default: boxing on the wake word tipped the robot over).
4. **Conversation.** Short spoken answers. Questions about the room go to the robot's own spatial memory, so it never guesses.
5. **One question about their interests per answer, at the right moment** (chosen in code, not by the model): what they are working
   on, how well they know Neo4j, and near the end (from the 8th answer, never right after a question of theirs) what they enjoyed at
   the event. While they keep asking, the robot just answers. When they say goodbye it says goodbye, nothing else.
6. **Material or a follow-up?** The robot points them to the Neo4j staff at the booth and to GraphAcademy. It takes no emails.

## What is kept, and what never is

After each answer a second model call writes an anonymous note of the conversation, which code then scrubs:

| Kept per conversation (`Encounter`) | Never kept |
|---|---|
| topics raised (`DISCUSSED` → `Theme`), what they enjoyed (`ENJOYED` → `Theme`) | names, companies, job titles |
| their need in a few neutral words ("process thousands of contracts with a knowledge graph") | emails, phone numbers, profile links |
| how well they know Neo4j (none / curious / user / expert) | the conversation text or audio |
| the questions they asked, rephrased; a one-sentence neutral summary; time, event, place | anything from the camera about a person |

Enforced in code, not only in the prompt ([`booth.py`](../src/puppy_mind/booth.py)):

| Rule | Where |
|---|---|
| The robot's questions never ask who someone is; a closing question of that kind from the model (name, company, role, email, "may I remember you") is removed | `QUESTIONS`, `drop_own_ask` |
| What the visitor said about themselves ("my name is…", "I work at…") is cut out of every kept field, with emails and phone numbers | `identifying_words`, `scrub`, `clean_topics` |
| The conversation text lives in RAM only while the conversation is open, and expires after `ENCOUNTER_IDLE_S` | `Encounter.messages` |
| There is no Person in the graph: nothing to look up, export or forget | `booth_graph.py` |
| Booth data answers only this machine unless `BRAIN_TOKEN` is set and passed | `_local_auth` |

Worth raising with compliance even so: speech is sent to OpenAI to be transcribed and the conversation text to write the reply and the
note (nothing is stored there by this app), and with frame analysis on, camera frames go to OpenAI every few seconds and people at the
booth are in them. For the booth, keep the camera on the live view only.

## The graph

```
(Encounter {id, ts, event, need, familiarity, questions, summary, turns})
(Encounter)-[:DISCUSSED]->(Theme {name})   (Encounter)-[:ENJOYED]->(Theme)   (Encounter)-[:HELD_AT]->(Place)
```

```cypher
// what the booth talked about today
MATCH (e:Encounter)-[:DISCUSSED]->(t:Theme) WHERE e.ts > datetime() - duration('P1D')
RETURN t.name, count(e) AS conversations ORDER BY conversations DESC

// topics that come up together
MATCH (a:Theme)<-[:DISCUSSED]-(e:Encounter)-[:DISCUSSED]->(b:Theme) WHERE a.name < b.name
RETURN a.name, b.name, count(e) AS together ORDER BY together DESC LIMIT 20
```

**NAMS.** In the ontology ([`ontology/puppy_pi.json`](../ontology/puppy_pi.json), revision 3) a booth conversation is an anonymous
`Event` that `DISCUSSED` / `ENJOYED` `Topic`s and that the robot (`Agent`) `HOSTED`; no relationship reaches a `Person`. Once that
revision is active (`uv run puppy-mind nams-setup --activate`), each note is mirrored to NAMS after it is saved.

## The Booth tab

Graph first: every topic visitors raised, sized by how many conversations raised it and linked when raised together. On top a
conversation counter that counts up ("New conversation: …"), questions and topics counted, and how well visitors know Neo4j. On the
right, **What visitors need**, the **Questions asked** and what people **enjoyed at the event**. A note can be removed (×).

## Booth mode and tricks

The PuppyPi's camera is fixed to its body, so to look up the robot tilts its whole body nose-up (Hiwonder's pose topic, radians,
nose-up positive, rejected beyond ±31°). Tricks (`boxing`, `wave`, `shake_hands`, `bow`, `nod`) are Hiwonder action groups run
through `/puppy_control/runActionGroup` as `<name>.d6ac`; visitors can also ask for one ("can you box?"). `drive.py` guards them:
only in booth mode, never while walking or turning, at most one per `GESTURE_COOLDOWN_S` (60 s) and `MAX_GESTURES_PER_HOUR`. The
service replies at once (Hiwonder runs the action group in a thread), so a reply does not prove the robot moved: supervised on the
robot, boxing via `/api/gesture` with booth mode on did box.

> Boxing shifts all the weight onto the back legs: use it on a flat, non-slippery floor with a clear space around the robot.

## The claw machine cheerleader

With `CLAW_MACHINE=true` the robot cheers for the booth's claw machine (prize: `CLAW_PRIZE`, default "a Graph Wizard"):

- **By itself (off by default):** with `CHEER_EVERY_S` set (e.g. 150), when nobody has talked to it that long and it is listening, it calls people
  over ("Feeling lucky? Try the claw machine and win a Graph Wizard!"). It never does this during or just after a conversation.
- **Booth tab buttons** (the "Claw machine" row): *Call people over*, *Someone is playing*, *They won!*, *So close* (try again).
  Each has several lines, and the same line never plays twice in a row. Cheering is words only: the robot does no trick.
- **The Wizard** knows about the claw machine: it mentions it when visitors ask what to do, and invites them to play at goodbye.

The lines are in `CHEERS` in `booth.py`. Nothing about the players is stored.

## Setting up the voice

The WonderEcho Pro appears to the Pi as a USB audio device. `robot/puppy_voice.py` records with `arecord`, plays with `aplay`, and
sends audio to the laptop; the Pi does no recognition or synthesis.

The voice client starts by itself at boot: `puppy-voice.service` runs `robot/start_voice_auto.sh`, which finds the laptop running
the app on the robot's current Wi-Fi (the address is remembered in `~/brain_url` and searched for again when it changes), fetches
the latest `puppy_voice.py` from it, and keeps it running (retrying every 10 s until the app is up). The log `~/puppy_voice.log`
starts empty at every boot, so what visitors said is kept no longer than the current day.

```bash
# install once, on the robot
wget -q http://<laptop-ip>:8000/robot/start_voice_auto.sh http://<laptop-ip>:8000/robot/puppy-voice.service
sudo cp puppy-voice.service /etc/systemd/system/ && sudo systemctl enable --now puppy-voice
sudo systemctl restart puppy-voice     # by hand, e.g. after updating the app
```

**The place.** At a booth the place is fixed: set `DEFAULT_LOCATION=Conference hall` and `AUTO_PLACE=false` in `.env`, so the camera
never guesses rooms ("Unnamed area 1", or the rooms of home) and every conversation is held at the booth.

**Wi-Fi on the robot.** Hiwonder's own `wifi.service` deletes every saved Wi-Fi at boot and knows only one network (its
`wifi_conf.py`), falling back to the robot's own Wi-Fi `HW-<serial>` (192.168.149.1). It is disabled on this robot so
NetworkManager keeps several networks (the phone hotspot first, the robot's own Wi-Fi as the last resort). Undo:
`sudo systemctl enable wifi.service hw_wifi.service`. Switch the phone hotspot on before the robot.

Laptop side: `.env` needs `OPENAI_API_KEY`. Optional: `EVENT_NAME`, `BOOTH_INFO`, `OPENAI_TTS_VOICE`, `GREET_COOLDOWN_S`,
`PEOPLE_GREET=false` (never greet unprompted). The robot's log `~/puppy_voice.log` prints what it heard: clear it after a booth day.

## GPT-Live: faster conversations

`LIVE_VOICE=true` in `.env` makes the conversations GPT-Live calls (OpenAI `gpt-live-1`, a full-duplex voice model) instead of
turn by turn. Measured: the classic turn needs about 5 to 7 s from the end of a question to the first sound (1.5 s of silence on
the robot, then transcription, the chat model, and the whole reply synthesised before it plays); GPT-Live starts answering about
1 to 1.5 s after the visitor stops, the first answer of a call included.

- The robot asks the laptop at start (`/api/voice/mode`) and then keeps one WebSocket (`/api/voice/live`). Outside a call it still
  sends only what someone says, but streamed from the start of the sentence (GPT-Live listens at real-time speed: a sentence sent
  only once it ended cost its whole length again). Each sentence opens a GPT-Live session at once; nothing it says is played and
  nothing is kept until its own transcript shows "Hey PuppyPi" (or an answer to a greeting). Without the wake word the session
  closes about 2 s after the sentence (a fraction of a cent). With it, the call goes on: the robot streams its microphone and plays
  the answer as it arrives. Measured: the first answer starts about 1.3 s after the visitor stops (2.4 to 4.5 s when the wake word
  was transcribed separately first). Switching `LIVE_VOICE` off needs only an app restart: the robot falls back to the classic
  turns. `--classic` on the robot forces them.
- Tricks a visitor asks for wait only for the previous trick (2 s), not the 60 s cooldown of unprompted tricks (that refused the
  second trick at the booth). Push-ups and moonwalk stay Booth tab buttons only: they need space.
- The call closes on goodbye, the Booth tab's End conversation, or `WAKE_WINDOW_S` of quiet. GPT-Live costs $0.05 per minute of
  call (billed per second), plus the backend model's tokens when it runs a tool.
- Tricks, questions about the room and a new visitor go to GPT-Live's delegated backend (`LIVE_BACKEND_MODEL`, default `gpt-6-luna`)
  as the same tools (`do_trick`, `ask_world`, `new_visitor`). The instructions say when to delegate with GPT-Live's
  `<|SpawnThinking|>` token; without that it answered "I can only wave in spirit".
- The booth rules and the anonymous notes are the same: the live transcript (ordered by its timestamps, since the text arrives in
  bursts) fills the conversation, and the notes are extracted from it as before.
- The robot's microphone hears its own speaker, so the laptop sends silence while the robot talks: visitors cannot interrupt it
  yet. Settings: `LIVE_MODEL`, `LIVE_VOICE_NAME` (default `marin`), `LIVE_BACKEND_MODEL`. Code: `live.py`.

## Not built (on purpose)

- **On-robot wake word.** "Hey PuppyPi" is checked on the laptop after transcription, so hall audio is still sent for transcription
  (and dropped). An on-device detector on the Pi (for example openWakeWord) would send nothing until the wake word: better for
  privacy and cost at a busy booth.
- **Retrieval-grounded Wizard answers.** The Wizard answers from the model plus a careful primer (`wizard.py`) and is told to admit
  uncertainty. For doc-grounded answers, index the Neo4j docs with the GraphRAG pipeline and add a `search_docs` tool.
