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

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .brain import Brain
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

        brain.rosbridge = RosbridgeDriver(url, brain.drive).start()
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
def demo_replay(delay: float = 4.0):
    """Replay the scripted booth scenario live (timestamps = now) to see the graph animate without a robot."""
    data = json.loads((Path(__file__).parents[2] / "eval" / "scenarios" / "booth.json").read_text(encoding="utf-8"))

    def run():
        for st in data["steps"]:
            brain.observe_scene(_scene(st), source="demo", location=data["location"])
            time.sleep(delay)

    threading.Thread(target=run, daemon=True).start()
    return {"steps": len(data["steps"]), "delay": delay}


@app.post("/api/ask")
def ask(body: Ask):
    if not brain.agent:
        raise HTTPException(503, "OPENAI_API_KEY is not set")
    return brain.ask(body.question)


ROBOT_FILES = {"puppy_drive.py", "puppy_streamer.py"}


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
        "nams_status": brain.nams.status(),
        "stats": brain.mem.stats(),
        "current": brain.mem.current_view(),
        "recent": recent,
        "timeline": brain.mem.timeline(10),
    }


@app.post("/api/location")
def set_location(body: Loc):
    brain.location = body.name.strip() or brain.location
    return {"location": brain.location}


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


@app.get("/api/changes")
def changes(since: str):
    return brain.mem.changes_since(since)


@app.get("/api/graph")
def graph():
    return brain.mem.graph_snapshot()


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
