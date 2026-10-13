"""Wires perception -> memory -> agent. One instance per process (server or CLI)."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .agent import Agent
from .config import Settings, get_settings
from .graph import Memory
from .models import Scene
from .nams import Nams
from .sources import FrameSource, ahash, hamming


class Brain:
    def __init__(self, s: Settings | None = None, *, need_openai: bool = True):
        self.s = s or get_settings()
        self.mem = Memory(self.s)
        self.nams = Nams(self.s)
        self.session = self.mem.start_session()
        self.location = self.s.default_location
        self._last_hash: dict[str, int] = {}
        self._watch: threading.Thread | None = None
        self._stop = threading.Event()
        self.perception = None
        self.agent = None
        if need_openai:
            from .perception import Perception  # imported lazily: tests/CLI seed need no OpenAI

            self.perception = Perception(self.s)
            self.agent = Agent(self.s, self.mem, self.nams)

    # ---- observation
    def observe_frame(self, jpeg: bytes, *, source: str = "camera", location: str | None = None,
                      skip_if_similar: bool = True) -> dict[str, Any]:
        assert self.perception, "perception disabled"
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
            image_ref = str(d / f"{int(time.time()*1000)}.jpg")
            Path(image_ref).write_bytes(jpeg)
        loc = location or self.location
        res = self.mem.ingest(scene, session_id=self.session, location=loc, source=source,
                              ts=ts, image_ref=image_ref)
        res["summary"] = scene.summary
        res["detected"] = [f"{o.label} ({o.color}, {o.confidence:.2f})" for o in scene.objects]
        for o in res["new_objects"]:  # mirror new objects into NAMS long-term memory (best effort)
            label = o.split(" (")[0]
            self.nams.remember_object(label, "", loc, o.split("(")[-1].rstrip(")"))
        return res

    # ---- continuous watching
    def watch(self, source: FrameSource, interval: float, on_result=None) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                try:
                    frame = source.read()
                    if frame:
                        r = self.observe_frame(frame, source=source.name)
                        if on_result:
                            on_result(r)
                except Exception as e:  # keep watching through transient camera/API errors
                    if on_result:
                        on_result({"error": str(e)})
                self._stop.wait(interval)

        self._stop.clear()
        self._watch = threading.Thread(target=loop, daemon=True)
        self._watch.start()

    def stop_watch(self) -> None:
        self._stop.set()
        if self._watch:
            self._watch.join(timeout=5)

    @property
    def watching(self) -> bool:
        return bool(self._watch and self._watch.is_alive() and not self._stop.is_set())

    def ask(self, question: str) -> dict[str, Any]:
        assert self.agent, "agent disabled"
        return self.agent.ask(question, self.session)

    def close(self) -> None:
        self.stop_watch()
        self.mem.close()
