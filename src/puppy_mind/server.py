"""FastAPI app: robot ingest endpoint, Q&A, and the browser UI."""
from __future__ import annotations

import json
import os
import queue
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from urllib.parse import quote

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .brain import Brain
from .drive import TRICKS
from .scenario import _scene
from .sources import open_source

STATIC = Path(__file__).parent / "static"
brain: Brain
recent: list[dict] = []  # last few watch/observe results for the UI


@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain
    brain = Brain()
    brain.mem.init_schema()
    url = os.getenv("ROSBRIDGE_URL")
    if url:  # drive through the robot's own rosbridge: nothing to run on the robot
        from .rosbridge import RosbridgeDriver

        brain.rosbridge = RosbridgeDriver(url, brain.drive, on_imu=lambda wz: brain.heading.feed(wz, still=not brain.drive.moving)).start()
    yield
    if getattr(brain, "rosbridge", None):
        brain.rosbridge.close()
    brain.close()


app = FastAPI(title="puppy-mind", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")  # served locally: the robot's Wi-Fi has no internet


def _push(r: dict) -> None:
    recent.insert(0, r)
    del recent[20:]


class Ask(BaseModel):
    question: str


class Loc(BaseModel):
    name: str


class PlaceBody(BaseModel):
    mode: str  # "auto" or "manual"
    name: str | None = None


class Rename(BaseModel):
    old: str
    new: str


class Merge(BaseModel):
    src: str
    dst: str


class GoTo(BaseModel):
    label: str
    color: str | None = None
    execute: bool = True


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.post("/api/observe")
def observe(
    file: UploadFile = File(...),
    location: str | None = Form(None),
    source: str = Form("puppypi-push"),
    x_brain_token: str | None = Header(None),
):
    """Robot pushes a JPEG here (sync handler: runs in the threadpool, so blocking calls are fine)."""
    if brain.s.brain_token and not secrets.compare_digest(x_brain_token or "", brain.s.brain_token):
        raise HTTPException(401, "bad token")
    res = brain.observe_frame(file.file.read(), source=source, location=location or None)
    _push(res)
    return res


@app.post("/api/frame")
def frame(
    file: UploadFile = File(...),
    location: str | None = Form(None),
    source: str = Form("puppypi-push"),
    x_brain_token: str | None = Header(None),
):
    """Robot pushes live frames (several per second). Shown immediately; perceived at most every WATCH_INTERVAL_S."""
    if brain.s.brain_token and not secrets.compare_digest(x_brain_token or "", brain.s.brain_token):
        raise HTTPException(401, "bad token")
    return {"perceiving": brain.submit_frame(file.file.read(), source=source, location=location or None)}


@app.get("/api/stream")
def stream():
    """MJPEG of the latest frame: drop into an <img>."""

    def gen():
        last, sent = -1, 0.0
        while True:
            f = brain.frame
            if f and (brain.frame_id != last or time.monotonic() - sent > 3):
                last, sent = brain.frame_id, time.monotonic()
                head = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(f)
                yield head + f + b"\r\n"
            time.sleep(0.04)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/events")
def events():
    """Server-Sent Events: observation / query / status / error."""
    q = brain.bus.subscribe()

    def gen():
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    t, d = q.get(timeout=15)
                    yield f"event: {t}\ndata: {json.dumps(d, default=str)}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            brain.bus.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/demo/replay")
def demo_replay(delay: float = 4.0, scenario: str = "home"):
    if brain.memory_kind != "fake":
        raise HTTPException(403, "Demo replay only runs on throwaway in-RAM memory (it would write fake data into your real graph).")
    """Replay the scripted booth scenario live (timestamps = now) to see the graph animate without a robot."""
    if scenario not in ("home", "booth"):
        raise HTTPException(400, "scenario must be home or booth")
    data = json.loads((Path(__file__).parents[2] / "eval" / "scenarios" / f"{scenario}.json").read_text(encoding="utf-8"))

    def run():
        for st in data["steps"]:
            brain.observe_scene(_scene(st), source="demo", location=data.get("location"))  # no pinned place: the robot works it out
            time.sleep(delay)

    threading.Thread(target=run, daemon=True).start()
    return {"steps": len(data["steps"]), "delay": delay}


@app.post("/api/ask")
def ask(body: Ask):
    if not brain.agent:
        raise HTTPException(503, "OPENAI_API_KEY is not set")
    return brain.ask(body.question)


ROBOT_FILES = {"puppy_drive.py", "puppy_streamer.py", "puppy_voice.py"}


@app.get("/robot/{name}")
def robot_script(name: str):
    """So the robot can `wget http://<laptop>:8000/robot/puppy_drive.py` (whitelisted files only)."""
    p = Path(__file__).parents[2] / "robot" / name
    if name not in ROBOT_FILES or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="text/x-python")


@app.get("/api/state")
def state():
    return {
        "session": brain.session,
        "location": brain.location,
        "watching": brain.watching,
        "has_frame": brain.frame is not None,
        "fps": brain.fps,
        "camera_name": getattr(brain._src, "name", "") if brain._src else "",
        "camera_error": getattr(brain._src, "error", "") if brain._src else "",
        "nams": brain.nams.enabled,
        "memory": brain.memory_kind,
        "place": brain.place_info(),
        "nams_status": brain.nams.status(),
        "stats": brain.mem.stats(),
        "current": brain.mem.current_view(),
        "recent": recent,
        "timeline": brain.mem.timeline(10),
    }


@app.post("/api/location")
def set_location(body: Loc):
    """Pin the place by hand (manual mode). POST /api/place with mode=auto hands control back to the robot."""
    return {"location": brain.set_place_manual(body.name), "place": brain.place_info()}


@app.get("/api/place")
def get_place():
    return brain.place_info()


@app.post("/api/place")
def set_place(body: PlaceBody):
    if body.mode == "manual":
        if not (body.name or "").strip():
            raise HTTPException(400, "manual mode needs a place name")
        brain.set_place_manual(body.name)
    elif body.mode == "auto":
        brain.set_place_auto()
    else:
        raise HTTPException(400, "mode must be auto or manual")
    return brain.place_info()


@app.get("/api/places")
def places():
    return {"current": brain.place_info(), "places": brain.mem.places_overview()}


@app.post("/api/places/rename")
def places_rename(body: Rename):
    if not brain.rename_place(body.old, body.new):
        raise HTTPException(409, "rename failed: unknown place, empty name, or the name is already taken")
    return {"ok": True}


@app.post("/api/places/merge")
def places_merge(body: Merge):
    if not brain.merge_places(body.src, body.dst):
        raise HTTPException(409, "merge failed: both places must exist and differ")
    return {"ok": True}


@app.post("/api/watch/{action}")
def watch(action: str):
    if action == "start" and not brain.watching:
        spec = os.environ.get("CAMERA_SOURCE", brain.s.camera_source)
        if not spec:
            raise HTTPException(400, "No camera configured. The robot camera was not found: check the Camera line printed by robot-test, or pass --camera <url>.")
        brain.watch(open_source(spec), brain.s.watch_interval_s, _push)
    elif action == "stop":
        brain.stop_watch()
    return {"watching": brain.watching}


@app.get("/api/timeline")
def timeline(limit: int = 60):
    return brain.mem.timeline(max(1, min(limit, 200)))


@app.get("/api/changes")
def changes(since: str):
    return brain.mem.changes_since(since)


@app.get("/api/graph")
def graph(lens: str = "long", nams: bool = False):
    if lens not in ("long", "short", "reasoning", "all"):
        raise HTTPException(400, "lens must be long, short, reasoning or all")
    return brain.mem.graph_snapshot(lens=lens, nams_entities=nams)


@app.get("/api/memory/summary")
def memory_summary():
    return brain.mem.memory_summary()


@app.get("/api/ontology")
def ontology_view():
    from . import ontology

    return ontology.build(brain.mem.label_counts(), brain.mem.rel_counts(), brain.nams.status())


# ---------------------------------------------------------------- keyboard drive
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _drive_event() -> None:
    brain.bus.publish("drive", brain.drive.status())


@app.websocket("/api/drive/ws")
async def drive_ws(ws: WebSocket, token: str = ""):
    """Browser keyboard -> drive state. Local-only unless BRAIN_TOKEN is set and supplied."""
    host = ws.client.host if ws.client else ""
    ok = host in LOOPBACK or (brain.s.brain_token and secrets.compare_digest(token, brain.s.brain_token))
    await ws.accept()
    if not ok:
        await ws.send_json({"error": "Driving from another machine needs BRAIN_TOKEN (set it in .env and pass ?token=)."})
        await ws.close(code=4403)
        return
    d = brain.drive
    try:
        while True:
            m = await ws.receive_json()
            if "arm" in m:
                d.arm(bool(m["arm"]))
                _drive_event()
            if "speed" in m:
                d.set_speed(int(m["speed"]))
                _drive_event()
            if m.get("stop"):
                d.stop()
                _drive_event()
            if m.get("home"):
                d.go_home()
                brain._moved = True
            c = m.get("cmd")
            if c:
                was = d.moving
                d.command(float(c.get("x", 0)), float(c.get("y", 0)), float(c.get("yaw", 0)))
                if d.moving:
                    brain._moved = True
                if d.moving != was:
                    _drive_event()
    except WebSocketDisconnect:
        pass
    finally:  # controller gone => stop and disarm, always
        d.arm(False)
        _drive_event()


@app.get("/api/drive/cmd")
def drive_cmd(ax: float | None = None, ay: float | None = None, ayaw: float | None = None,
              x_brain_token: str | None = Header(None)):
    """Polled by robot/puppy_drive.py (the robot pulls; the brain never connects in)."""
    if brain.s.brain_token and not secrets.compare_digest(x_brain_token or "", brain.s.brain_token):
        raise HTTPException(401, "bad token")
    if ax is not None:
        brain.drive.robot_report({"x": ax, "y": ay or 0.0, "yaw": ayaw or 0.0})
    return brain.drive.current()


@app.get("/api/drive/state")
def drive_state():
    st = brain.drive.status()
    rb = getattr(brain, "rosbridge", None)
    st["rosbridge"] = rb.status() if rb else None
    return st


# ---------------------------------------------------------------- phase 5: go where you last saw it
@app.post("/api/nav/go")
def nav_go(body: GoTo):
    """Plan (and, only if driving is armed, start) a turn toward a remembered object. Never walks; see navigate.py."""
    return brain.go_to(body.label, body.color, execute=body.execute)


@app.post("/api/nav/cancel")
def nav_cancel():
    brain.nav.cancel("cancelled")
    return brain.nav.state


@app.get("/api/nav")
def nav_state():
    return brain.nav.state


# ---------------------------------------------------------------- people: talk, greet, remember (docs/PEOPLE.md)
class Say(BaseModel):
    text: str
    encounter_id: str | None = None


class FollowUp(BaseModel):
    done: bool = True


def _robot_auth(token: str | None) -> None:
    if brain.s.brain_token and not secrets.compare_digest(token or "", brain.s.brain_token):
        raise HTTPException(401, "bad token")


def _local_auth(request: Request, header: str | None, query: str | None) -> None:
    """People records are personal data and booth mode moves the robot: this machine only, unless BRAIN_TOKEN is set and supplied (header or ?token=)."""
    host = request.client.host if request.client else ""
    tok = brain.s.brain_token
    if host in LOOPBACK or (tok and (secrets.compare_digest(header or "", tok) or secrets.compare_digest(query or "", tok))):
        return
    raise HTTPException(403, "People data and booth controls are only for this machine. Set BRAIN_TOKEN in .env and pass ?token= to open it from another one.")


def _need_voice() -> None:
    if not brain.people.client:
        raise HTTPException(503, "OPENAI_API_KEY is not set")


@app.post("/api/converse")
def converse(body: Say, request: Request, x_brain_token: str | None = Header(None), token: str | None = None):
    """One typed turn of the booth conversation (the same brain the voice endpoint uses)."""
    _local_auth(request, x_brain_token, token)
    _need_voice()
    return brain.people.turn(body.text, body.encounter_id, brain.place_info().get("name"))


@app.post("/api/voice/turn")
def voice_turn(file: UploadFile = File(...), encounter_id: str | None = Form(None), x_brain_token: str | None = Header(None)):
    """The robot (WAV) or the UI's Hold to talk (WebM) posts what its microphone heard; the reply comes back as WAV to play.
    204 when nothing intelligible was said."""
    _robot_auth(x_brain_token)
    _need_voice()
    text = brain.voice.transcribe(file.file.read(), file.filename or "speech.wav")  # the extension tells the model the format
    if len(text) < 2:
        return Response(status_code=204)
    r = brain.people.turn(text, encounter_id, brain.place_info().get("name"))
    return Response(brain.voice.speak(r["reply"]), media_type="audio/wav",
                    headers={"X-Encounter": r["encounter_id"], "X-Heard": quote(text[:300]), "X-Reply": quote(r["reply"][:600])})


@app.get("/api/voice/next")
def voice_next(x_brain_token: str | None = Header(None)):
    """Polled by the robot: a greeting it should say (WAV), or 204."""
    _robot_auth(x_brain_token)
    it = brain.people.outbox.next()
    if not it:
        return Response(status_code=204)
    if not brain.voice:
        return Response(status_code=204)
    return Response(brain.voice.speak(it["text"]), media_type="audio/wav", headers={"X-Reply": quote(it["text"][:600])})


@app.get("/api/people")
def people_list(request: Request, stage: str | None = None, q: str | None = None, x_brain_token: str | None = Header(None), token: str | None = None):
    _local_auth(request, x_brain_token, token)
    return {"people": brain.mem.list_people(stage, q), "counts": brain.mem.people_counts(), "consent_required": True}


@app.get("/api/people/graph")
def people_graph(request: Request, x_brain_token: str | None = Header(None), token: str | None = None):
    _local_auth(request, x_brain_token, token)
    return brain.mem.people_graph()


LEAD_COLUMNS = ["name", "company", "role", "function", "email", "lead_stage", "lead_score", "use_case", "familiarity", "interests", "enjoyed",
                "wants_follow_up", "wants_demo", "followed_up", "notes", "first_met", "last_met", "meet_count"]


@app.get("/api/people/export.csv")
def people_export(request: Request, x_brain_token: str | None = Header(None), token: str | None = None):
    """Leads for the CRM. Everyone in the graph said yes to being remembered, so everyone is exportable."""
    _local_auth(request, x_brain_token, token)
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(LEAD_COLUMNS)
    def cell(v) -> str:  # spoken text ends up in a spreadsheet: neutralise formulas such as =HYPERLINK(...)
        v = "; ".join(v) if isinstance(v, list) else ("" if v is None else str(v))
        return "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r") else v

    for p in brain.mem.list_people():
        w.writerow([cell(p.get(c)) for c in LEAD_COLUMNS])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=booth-leads.csv"})


@app.get("/api/people/{person_id}")
def people_get(person_id: str, request: Request, x_brain_token: str | None = Header(None), token: str | None = None):
    _local_auth(request, x_brain_token, token)
    p = brain.mem.get_person(person_id)
    if not p:
        raise HTTPException(404, "unknown person")
    return p


@app.post("/api/people/{person_id}/followup")
def people_followup(person_id: str, body: FollowUp, request: Request, x_brain_token: str | None = Header(None), token: str | None = None):
    _local_auth(request, x_brain_token, token)
    if not brain.mem.set_followed_up(person_id, body.done):
        raise HTTPException(404, "unknown person")
    return {"ok": True}


@app.delete("/api/people/{person_id}")
def people_forget(person_id: str, request: Request, x_brain_token: str | None = Header(None), token: str | None = None):
    """Erase a person and their encounters."""
    _local_auth(request, x_brain_token, token)
    if not brain.mem.forget_person(person_id):
        raise HTTPException(404, "unknown person")
    brain.bus.publish("people", {"event": "forgotten", "person_id": person_id})
    return {"ok": True}


# ---------------------------------------------------------------- booth mode: look up at people, greet with a trick
class Booth(BaseModel):
    on: bool
    look_up_deg: float | None = None


class Trick(BaseModel):
    name: str


@app.get("/api/booth")
def booth_get():
    st = brain.drive.status()
    return {"on": st["booth"], "look_up_deg": st["look_up_deg"], "in_action": st["in_action"], "greet_action": brain.s.greet_action,
            "tricks": list(TRICKS)}


@app.post("/api/booth")
def booth_set(body: Booth, request: Request, x_brain_token: str | None = Header(None), token: str | None = None):
    """Booth mode moves the robot (pitch, tricks): this machine only, or with BRAIN_TOKEN, like driving."""
    _local_auth(request, x_brain_token, token)
    brain.drive.set_booth(body.on, body.look_up_deg)
    _drive_event()
    return booth_get()


@app.post("/api/gesture")
def gesture(body: Trick, request: Request, x_brain_token: str | None = Header(None), token: str | None = None):
    _local_auth(request, x_brain_token, token)
    return brain.gesture(body.name)


@app.post("/api/voice/say")
def voice_say(body: Say, request: Request, x_brain_token: str | None = Header(None), token: str | None = None):
    """Text -> WAV in the robot's voice, for the UI's 'Speak here' (testing without the robot's speaker). This machine only."""
    _local_auth(request, x_brain_token, token)
    if not brain.voice:
        raise HTTPException(503, "OPENAI_API_KEY is not set")
    return Response(brain.voice.speak(body.text[:1500]), media_type="audio/wav")
