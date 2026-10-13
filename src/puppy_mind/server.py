"""FastAPI app: robot ingest endpoint, Q&A, and the browser UI."""
from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .brain import Brain
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
    return FileResponse(STATIC / "index.html")


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


@app.post("/api/ask")
def ask(body: Ask):
    return brain.ask(body.question)


@app.get("/api/state")
def state():
    return {
        "session": brain.session,
        "location": brain.location,
        "watching": brain.watching,
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
