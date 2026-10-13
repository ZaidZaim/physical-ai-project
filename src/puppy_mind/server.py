"""FastAPI app: robot ingest endpoint, Q&A, and the browser UI."""
from __future__ import annotations

import json
import queue
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
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
    yield
    brain.close()


app = FastAPI(title="puppy-mind", lifespan=lifespan)


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


@app.get("/api/state")
def state():
    return {
        "session": brain.session,
        "location": brain.location,
        "watching": brain.watching,
        "has_frame": brain.frame is not None,
        "nams": brain.nams.enabled,
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
        brain.watch(open_source(brain.s.camera_source), brain.s.watch_interval_s, _push)
    elif action == "stop":
        brain.stop_watch()
    return {"watching": brain.watching}


@app.get("/api/changes")
def changes(since: str):
    return brain.mem.changes_since(since)


@app.get("/api/graph")
def graph():
    return brain.mem.graph_snapshot()
