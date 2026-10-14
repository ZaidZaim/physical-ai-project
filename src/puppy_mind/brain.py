"""Wires perception -> memory -> agent. One instance per process (server or CLI).

Live view: every incoming frame updates `frame` (served as MJPEG); perception runs on at most one
frame per WATCH_INTERVAL_S in a background thread; each completed observation, query and status
change is published on `bus` (served as Server-Sent Events) so the UI can animate the graph.
"""
from __future__ import annotations

import os
import collections
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from .agent import Agent
from .config import Settings, get_settings
from .drive import DriveState
from .graph import Memory
from .models import Scene
from .nams import Nams
from .sources import FolderSource, FrameSource, ahash, hamming

_OBJ_ID = re.compile(r"obj-[0-9a-f]{8}")


class Bus:
    """Tiny thread-safe pub/sub; slow subscribers drop events rather than block the robot."""

    def __init__(self) -> None:
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, type_: str, data: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait((type_, data))
            except queue.Full:
                pass


class Brain:
    def __init__(self, s: Settings | None = None, *, need_openai: bool = True):
        self.s = s or get_settings()
        if os.getenv("PUPPY_FAKE_MEMORY") == "1":  # dev only: no Aura, nothing persists
            from .devfake import FakeMemory

            self.mem = FakeMemory(self.s)
        else:
            self.mem = Memory(self.s)
        self.nams = Nams(self.s)
        self.bus = Bus()
        self.drive = DriveState(allow_combine=os.getenv("DRIVE_COMBINE") == "1")
        self._moved = False  # robot moved since the last observation
        self.session = self.mem.start_session()
        self.location = self.s.default_location
        self.frame: bytes | None = None
        self.frame_id = 0
        self._ftimes: collections.deque = collections.deque(maxlen=120)
        self._src: FrameSource | None = None
        self._last_hash: dict[str, int] = {}
        self._last_perceive = 0.0
        self._busy = threading.Lock()
        self._watch: threading.Thread | None = None
        self._stop = threading.Event()
        self.perception = None
        self.agent = None
        if need_openai and self.s.openai_api_key:  # without a key: live view + replay only
            from .perception import Perception  # lazy: seed/tests need no OpenAI

            # Camera frames are sent to OpenAI ONLY when perception is on. PERCEPTION=off keeps the question-answering
            # agent but never uploads frames (used by plain driving tests).
            if os.getenv("PERCEPTION", "on").lower() != "off":
                self.perception = Perception(self.s)
            self.agent = Agent(self.s, self.mem, self.nams)

    @property
    def fps(self) -> float:
        now = time.monotonic()
        return round(sum(1 for t in tuple(self._ftimes) if now - t <= 2.0) / 2.0, 1)

    # ---- live frames
    def submit_frame(self, jpeg: bytes, *, source: str = "camera", location: str | None = None) -> bool:
        """Show the frame immediately; start perception if due and idle. Returns True if perception started."""
        self.frame, self.frame_id = jpeg, self.frame_id + 1
        self._ftimes.append(time.monotonic())
        if self.drive.moving:  # blurry + viewpoint changing: don't analyse, don't learn from it
            self._moved = True
            return False
        if (
            self.perception
            and time.monotonic() - self._last_perceive >= self.s.watch_interval_s
            and self._busy.acquire(blocking=False)
        ):
            self._last_perceive = time.monotonic()
            threading.Thread(target=self._perceive, args=(jpeg, source, location), daemon=True).start()
            return True
        return False

    def _perceive(self, jpeg: bytes, source: str, location: str | None) -> None:
        try:
            self.bus.publish("status", {"state": "perceiving"})
            res = self.observe_frame(jpeg, source=source, location=location)
            if "skipped" in res:
                self.bus.publish("status", {"state": "idle", "note": res["skipped"]})
        except Exception as e:  # keep the loop alive through API/network errors
            self.bus.publish("error", {"error": str(e)[:300]})
        finally:
            self._busy.release()

    # ---- observation
    def observe_frame(self, jpeg: bytes, *, source: str = "camera", location: str | None = None,
                      skip_if_similar: bool = True) -> dict[str, Any]:
        assert self.perception, "perception disabled"
        self.frame, self.frame_id = jpeg, self.frame_id + 1
        h = ahash(jpeg)
        prev = self._last_hash.get(source)
        if skip_if_similar and prev is not None and hamming(prev, h) <= 3:
            return {"skipped": "frame nearly identical to previous"}
        self._last_hash[source] = h
        scene = self.perception.describe(jpeg)
        return self.observe_scene(scene, source=source, location=location, jpeg=jpeg)

    def observe_scene(self, scene: Scene, *, source: str = "camera", location: str | None = None,
                      jpeg: bytes | None = None, ts: str | None = None) -> dict[str, Any]:
        image_ref = None
        if jpeg and self.s.store_images:
            d = Path("frames")
            d.mkdir(exist_ok=True)
            image_ref = str(d / f"{int(time.time() * 1000)}.jpg")
            Path(image_ref).write_bytes(jpeg)
        loc = location or self.location
        res = self.mem.ingest(scene, session_id=self.session, location=loc, source=source,
                              ts=ts, image_ref=image_ref, count_misses=not self._moved)
        res["misses_counted"] = not self._moved
        self._moved = False
        res["summary"] = scene.summary
        res["detected"] = [f"{o.label} ({o.color}, {o.confidence:.2f})" for o in scene.objects]
        self.bus.publish("observation", {
            "id": res["observation_id"], "ts": res["ts"], "location": loc, "summary": scene.summary,
            "objects": res["objects"], "relations": res["relation_list"], "missed": res["missed"],
        })
        for d in res["objects"]:  # mirror new objects into NAMS long-term memory (best effort)
            if d["new"]:
                self.nams.remember_object(d["label"], d["color"], loc, d["id"])
        return res

    # ---- continuous watching
    def watch(self, source: FrameSource, interval: float, on_result=None, fps: float = 40.0) -> None:
        self._src = source
        folder = isinstance(source, FolderSource)  # replay: perceive every frame, no live-video pacing

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    frame = source.read()
                    if frame and folder:
                        r = self.observe_frame(frame, source=source.name)
                        on_result and on_result(r)
                    elif frame:
                        self.submit_frame(frame, source=source.name)
                except Exception as e:
                    self.bus.publish("error", {"error": str(e)[:300]})
                    on_result and on_result({"error": str(e)})
                self._stop.wait(interval if folder else 1.0 / fps)

        self._stop.clear()
        self._watch = threading.Thread(target=loop, daemon=True)
        self._watch.start()

    def stop_watch(self) -> None:
        self._stop.set()
        if self._watch:
            self._watch.join(timeout=5)
        if self._src:
            self._src.close()
            self._src = None

    @property
    def watching(self) -> bool:
        return bool(self._watch and self._watch.is_alive() and not self._stop.is_set())

    def ask(self, question: str) -> dict[str, Any]:
        assert self.agent, "agent disabled"
        r = self.agent.ask(question, self.session)
        ids = sorted(set(_OBJ_ID.findall(str(r["evidence"]))))
        self.bus.publish("query", {"question": question, "tools": [t["tool"] for t in r["evidence"]], "ids": ids})
        return r

    def close(self) -> None:
        self.stop_watch()
        self.mem.close()
