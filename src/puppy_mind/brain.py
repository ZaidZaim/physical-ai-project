"""Wires perception -> memory -> agent. One instance per process (server or CLI).

Live view: every incoming frame updates `frame` (served as MJPEG); perception runs on at most one
frame per WATCH_INTERVAL_S in a background thread; each completed observation, query and status
change is published on `bus` (served as Server-Sent Events) so the UI can animate the graph.
"""
from __future__ import annotations

import os
import collections
import dataclasses
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from .agent import Agent
from .config import Settings, get_settings
from .drive import DriveState
from .graph import Memory, now_iso
from .models import Scene
from .pose import HeadingTracker
from .spatial import PlaceInfo, PlaceResolver, PlaceTracker
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
        # Throwaway in-RAM memory must never reach a real NAMS workspace: a dev demo once leaked 12 fake entities into it.
        nams_settings = dataclasses.replace(self.s, nams_api_key="") if os.getenv("PUPPY_FAKE_MEMORY") == "1" else self.s
        self.nams = Nams(nams_settings, memory=self.mem)
        self.bus = Bus()
        self.drive = DriveState(allow_combine=os.getenv("DRIVE_COMBINE") == "1")
        self._moved = False  # robot moved since the last observation
        self.session = self.mem.start_session()
        self.location = self.s.default_location
        self.tracker = PlaceTracker(PlaceResolver(), self.s.place_confirm_frames)
        self.place_mode = "auto" if self.s.auto_place else "manual"
        self.place_state: dict[str, Any] = {"mode": self.place_mode, "name": self.location, "confidence": None, "reason": "not yet observed", "room_type": "unknown", "uncertain": False}
        self._place_ids: dict[str, str] = {}
        self._moved_view = False  # the first frame after a place change is a new view
        self._just_changed = False
        self.heading = HeadingTracker()  # orientation relative to where we entered the current place (IMU gyro; None without data)
        self._deferred: list[tuple] = []  # frames held back while a change of place is unconfirmed
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
            self.agent.place_fn = self.place_info

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
        uncertain, place_conf, changed = False, None, False
        if location:  # scripted callers (tests, replay) pin the place
            loc = location
        elif self.place_mode == "auto":
            loc, uncertain, place_conf = self._resolve_place(scene)
            changed = self._just_changed
        else:
            loc = self.location
        if uncertain:
            # The robot may have just moved. Do NOT file this frame under the current place: it would teach that place about things
            # that live somewhere else (and then it could never leave). Hold it; it is filed under the new place if the move is
            # confirmed, and dropped if the next frame shows we never left (a blur, a passer-by).
            self._deferred.append((scene, source, jpeg, ts or now_iso()))
            del self._deferred[:-max(1, self.s.place_confirm_frames)]
            return {"skipped": "possible change of place: waiting to confirm", "deferred": True, "location": loc, "reason": self.place_state["reason"]}
        held, self._deferred = (self._deferred if changed else []), []
        for d_scene, d_src, d_jpeg, d_ts in held:  # frames that were waiting for this very decision
            self._ingest(d_scene, loc, d_src, d_jpeg, d_ts, count_misses=False, place_conf=place_conf, heading=None)
        moved = self._moved or self._moved_view
        res = self._ingest(scene, loc, source, jpeg, ts, count_misses=not moved, place_conf=place_conf, heading=self.heading.value())
        self._moved = self._moved_view = False
        return res

    def _ingest(self, scene: Scene, loc: str, source: str, jpeg: bytes | None, ts: str | None, *, count_misses: bool,
                place_conf: float | None, heading: float | None = None) -> dict[str, Any]:
        image_ref = None
        if jpeg and self.s.store_images:
            d = Path("frames")
            d.mkdir(exist_ok=True)
            image_ref = str(d / f"{int(time.time() * 1000)}.jpg")
            Path(image_ref).write_bytes(jpeg)
        res = self.mem.ingest(scene, session_id=self.session, location=loc, source=source, ts=ts, image_ref=image_ref,
                              count_misses=count_misses, room_type=scene.room_type, place_confidence=place_conf, heading=heading)
        res["summary"] = scene.summary
        res["detected"] = [f"{o.label} ({o.color}, {o.confidence:.2f})" for o in scene.objects]
        self.bus.publish("observation", {
            "id": res["observation_id"], "ts": res["ts"], "location": loc, "summary": scene.summary,
            "objects": res["objects"], "relations": res["relation_list"], "missed": res["missed"],
            "overlap": res.get("overlap"), "misses_counted": res.get("misses_counted"), "room_type": scene.room_type,
            "moved_hypotheses": res.get("moved_hypotheses", []), "place": self.place_info(),
        })
        self.nams.submit_observation({  # background: never blocks the robot
            "session": self.session, "ts": res["ts"], "location": loc, "place_id": self._place_id(loc, scene.room_type), "summary": scene.summary,
            "robot": self.s.robot_name, "observation_id": res["observation_id"],
            "objects": res["objects"], "relations": res["relation_list"], "missed": [dict(m) for m in res["missed"]],
        })
        return res

    # ---- spatial memory: where am I?
    def _places(self) -> list[PlaceInfo]:
        return [PlaceInfo(p["name"], p["room_type"], p["provisional"], p["visits"], p["objects"]) for p in self.mem.list_places()]

    def _place_id(self, name: str, room: str = "unknown") -> str:
        if name not in self._place_ids:
            self._place_ids[name] = self.mem.ensure_place(name, room, False)
        return self._place_ids[name]

    def _resolve_place(self, scene: Scene) -> tuple[str, bool, float | None]:
        """Decide which place this frame belongs to. Returns (place, uncertain, confidence)."""
        view = [{"label": o.label.lower(), "color": o.color.lower()} for o in scene.objects if o.confidence >= self.s.min_confidence]
        prev = self.tracker.current  # None until a place has actually been recognised: the default name is not somewhere we came from
        r = self.tracker.update(view, scene.room_type, scene.room_confidence, self._places())
        self._just_changed = r.changed
        if r.changed:
            if r.is_new:
                name = r.suggested_name or "Unnamed area 1"
                self.mem.ensure_place(name, scene.room_type, provisional=True)
                self.tracker.confirm_new(name)
            else:
                name = r.place  # type: ignore[assignment]
            self.mem.touch_place(name, scene.room_type)
            self.location = name
            self.heading.reset()  # new place, new origin: 0 degrees = the way we face on arrival
            if prev and prev != name:
                self.mem.note_transition(prev, name)
                self.nams.submit_transition(prev, self._place_id(prev), name, self._place_id(name, scene.room_type))
                self._moved_view = True
            self.bus.publish("place", {"name": name, "previous": prev, "new": r.is_new, "confidence": round(r.confidence, 2),
                                       "reason": r.reason, "room_type": scene.room_type})
        elif r.place and r.place != self.location:
            self.location = r.place
        elif r.uncertain and r.reason != self.place_state.get("reason"):
            self.bus.publish("place", {"name": self.location, "uncertain": True, "reason": r.reason})
        self.place_state = {"mode": "auto", "name": self.location, "confidence": round(r.confidence, 2), "reason": r.reason,
                            "room_type": scene.room_type, "uncertain": r.uncertain}
        return self.location, r.uncertain, r.confidence

    def place_info(self) -> dict[str, Any]:
        return {**self.place_state, "mode": self.place_mode, "name": self.location, "heading_deg": self.heading.value(),
                "heading_source": "imu gyro (relative to entry, drifts)" if self.heading.value() is not None else None}

    def set_place_manual(self, name: str) -> str:
        name = name.strip()
        if not name:
            return self.location
        self.mem.ensure_place(name, "unknown", False)
        self.location, self.place_mode = name, "manual"
        self.heading.reset()
        self.tracker.set_current(name)
        self.place_state = {"mode": "manual", "name": name, "confidence": None, "reason": "set by you", "room_type": "unknown", "uncertain": False}
        self._moved = True  # the view is about to change
        return name

    def set_place_auto(self) -> None:
        self.place_mode = "auto"
        self.tracker.set_current(self.location)
        self.place_state = {**self.place_state, "mode": "auto", "reason": "back to automatic"}

    def rename_place(self, old: str, new: str) -> bool:
        new = new.strip()
        ok = bool(new) and self.mem.rename_place(old, new)
        if ok:
            if self.location == old:
                self.location = new
            if self.tracker.current == old:
                self.tracker.current = new
            self._place_ids[new] = self._place_ids.pop(old, None) or self._place_id(new)
            self.nams.submit_rename(old, new, self._place_ids[new])
        return ok

    def merge_places(self, src: str, dst: str) -> bool:
        ok = self.mem.merge_places(src, dst)
        if ok:
            if self.location == src:
                self.location = dst
            if self.tracker.current == src:
                self.tracker.current = dst
            self._place_ids.pop(src, None)
        return ok

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
        threading.Thread(target=self._remember_exchange, args=(question, r, ids), daemon=True).start()  # off the answer path
        self.bus.publish("query", {"question": question, "tools": [t["tool"] for t in r["evidence"]], "ids": ids})
        return r

    def _remember_exchange(self, question: str, r: dict, ids: list[str]) -> None:
        """Short-term (Dialogue/Utterance) + reasoning (ReasoningStep/Lookup) memory in the graph; then tell the UI."""
        try:
            if self.nams.auth == "ok":  # NAMS stores this exchange in the shared database; avoid a duplicate local copy
                self.nams.flush(20)
            else:
                self.mem.record_exchange(self.session, question, r["answer"], r["evidence"], ids)
            self.bus.publish("memory", {"kind": "exchange"})
        except Exception as e:
            self.bus.publish("error", {"error": f"could not save the conversation to memory: {e}"[:300]})

    @property
    def memory_kind(self) -> str:
        return "fake" if os.getenv("PUPPY_FAKE_MEMORY") == "1" else "aura"

    def close(self) -> None:
        self.stop_watch()
        self.nams.close()
        self.mem.close()
