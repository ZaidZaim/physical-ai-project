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
from .drive import SETTLE_S, DriveState
from .graph import Memory, now_iso
from .models import Scene
from .navigate import Navigator, Plan, pick_target, plan_go_to
from .pose import HeadingTracker
from .spatial import TRANSIENT, PlaceInfo, PlaceResolver, PlaceTracker, TrackResult
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
        self._place_frames: dict[str, list[tuple[int, bytes]]] = {}  # a few recent views per place, RAM only: used to tell two similar rooms apart
        self._room_checks: dict[str, int] = {}
        self.heading = HeadingTracker()  # orientation relative to where we entered the current place (IMU gyro; None without data)
        self.heading_epoch = now_iso()  # when the current heading frame began: bearings recorded before it are in another frame
        self.obs_seq = 0  # camera views checked so far (stored, near-duplicate or empty); the look-around scan waits for it to move
        self.last_objects: list[dict] = []  # what the latest checked view showed (label, color, zone; id once stored)
        self.nav = Navigator(self.drive, lambda: self.heading.value(), publish=self.bus.publish, obs_seq_fn=lambda: self.obs_seq,
                             seen_fn=self._seen_now, fov_deg=self.s.camera_fov_deg, settle_s=SETTLE_S,
                             obs_wait_s=self.s.watch_interval_s * 2 + 5)
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
            self.agent.go_fn = self.go_to

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
            self.obs_seq += 1  # a near-duplicate keeps last_objects: same view, same things
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
        self.last_objects = [{"label": o.label.lower(), "color": o.color.lower(), "zone": o.zone} for o in scene.objects]
        if not scene.objects:  # blur, darkness, a turn in progress: says nothing about the world, so it is not stored (and cannot cause misses)
            return {"skipped": "nothing recognised in this frame", "location": self.location}
        uncertain, place_conf, changed = False, None, False
        if location:  # scripted callers (tests, replay) pin the place
            loc = location
        elif self.place_mode == "auto":
            loc, uncertain, place_conf = self._resolve_place(scene, jpeg)
            changed = self._just_changed
            if self.tracker.current is None:  # no place recognised yet: do not file anything under the default name
                return {"skipped": "no place recognised yet: waiting for a frame that shows enough", "location": loc}
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
        learned = self.mem.learn_room_type(loc) if hasattr(self.mem, "learn_room_type") else None
        if learned:
            self.bus.publish("place", {"name": loc, "learned_room_type": learned})
            if loc.startswith("Unnamed area"):  # now we know what kind of room it is: say so
                new = PlaceResolver.suggest_name(learned, 1.0, self._places())
                if self.rename_place(loc, new):
                    loc = new
        if hasattr(self.mem, "set_robot_place"):
            self.mem.set_robot_place(loc, heading)
        self._remember_view(loc, jpeg)
        self.last_objects = res["objects"]
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

    def _resolve_place(self, scene: Scene, jpeg: bytes | None = None) -> tuple[str, bool, float | None]:
        """Decide which place this frame belongs to. Returns (place, uncertain, confidence)."""
        view = [{"label": o.label.lower(), "color": o.color.lower()} for o in scene.objects if o.confidence >= self.s.min_confidence and o.label.lower() not in TRANSIENT]
        prev = self.tracker.current  # None until a place has actually been recognised: the default name is not somewhere we came from
        r = self.tracker.update(view, scene.room_type, scene.room_confidence, self._places())
        r = self._verify_room(r, scene, jpeg, view)
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
            self._room_checks.clear()
            self._new_heading_frame()  # new place, new origin: 0 degrees = the way we face on arrival
            if prev and prev != name and hasattr(self.mem, "consolidate_duplicates"):
                try:  # leaving a place: tidy look-alikes of one thing seen from two angles
                    self.mem.consolidate_duplicates(prev)
                except Exception:
                    pass
            if prev and prev != name:
                if not (r.is_new and self._room_of(prev) == scene.room_type):  # a 'jump' to a second room of the same kind is not adjacency
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
                            "room_type": scene.room_type, "uncertain": r.uncertain, "hint": r.hint}
        return self.location, r.uncertain, r.confidence

    def _verify_room(self, r: TrackResult, scene: Scene, jpeg: bytes | None, view: list[dict]) -> TrackResult:
        """The resolver doubts it: few known objects, but this kind of room is known. Look at the pictures instead of guessing:
        ask the vision model whether this is the SAME room as the views we hold of it. Another bedroom becomes 'Bedroom 2' by itself."""
        cand = r.place
        refs = [j for _, j in self._place_frames.get(cand or "", [])]
        if not (r.hint and jpeg and self.perception and cand and refs and not r.is_new and not r.uncertain) or self._room_checks.get(cand, 0) >= 3:
            return r
        self._room_checks[cand] = self._room_checks.get(cand, 0) + 1
        try:
            m = self.perception.same_room(jpeg, refs)
        except Exception as e:  # never let a failed check stop the robot: keep the resolver's answer
            self.bus.publish("error", {"error": "room check failed: " + str(e)[:200]})
            return r
        self.bus.publish("place", {"name": cand, "reason": f"room check: {'same room' if m.same_room else 'a different room'} ({m.confidence:.0%}) {m.reason}"})
        if m.same_room or m.confidence < 0.6:
            return r
        room = scene.room_type if scene.room_type not in ("unknown", "other", "") else (self._room_of(cand) or "room")
        name = PlaceResolver.suggest_name(room, 1.0, self._places())
        return TrackResult(None, True, True, False, m.confidence, f"a different {room} from '{cand}' ({m.reason})", suggested_name=name)

    def _room_of(self, name: str) -> str | None:
        return next((p.room_type for p in self._places() if p.name == name), None)

    def _remember_view(self, loc: str, jpeg: bytes | None) -> None:
        if not jpeg:
            return
        try:
            h = ahash(jpeg)
        except Exception:  # not a decodable picture: nothing to remember
            return
        frames = self._place_frames.setdefault(loc, [])
        if all(hamming(h, fh) > 8 for fh, _ in frames):
            frames.append((h, jpeg))
            del frames[:-3]

    def place_info(self) -> dict[str, Any]:
        return {**self.place_state, "mode": self.place_mode, "name": self.location, "recognised": self.tracker.current is not None or self.place_mode == "manual", "heading_deg": self.heading.value(),
                "heading_source": "imu gyro (relative to entry, drifts)" if self.heading.value() is not None else None}

    def _new_heading_frame(self) -> None:
        if self.nav.state.get("active"):  # the frame the turn was planned in is gone
            self.nav.cancel("I changed place, so the direction I was turning to no longer applies")
        self.heading.reset()
        self.heading_epoch = now_iso()

    # ---- phase 5: go where you last saw it (turn in place / look around; never walks on its own)
    def go_to(self, label: str, color: str | None = None, execute: bool = True) -> dict[str, Any]:
        here, heading = self.location, self.heading.value()
        target = pick_target(self.mem.find_object(label, color), here)
        if target and here not in {p.name for p in self._places()}:  # e.g. still on the default name before a room is recognised
            where = f" It was last seen in {target['location']}." if target.get("location") else ""
            return {"plan": Plan("no_place", f"I don't know which room I'm in yet.{where} Start the camera so I can recognise "
                                             "the room, then ask again.", target).as_dict()}
        route = bearing = None
        if target and target.get("location") and target["location"] != here:
            route = self.mem.route_between(here, target["location"])
        elif target and heading is not None:
            bearing = self.mem.bearing_since(target["id"], here, self.heading_epoch)
        plan = plan_go_to(target, here, heading, bearing, route)
        out: dict[str, Any] = {"plan": plan.as_dict()}
        if plan.kind == "scan" and not (self.perception and self.fps > 0):
            out["run"] = {"started": False, "reason": "To look around I need the camera running with frame analysis on. Start the camera and ask again."}
        elif execute and plan.kind in ("turn", "scan"):
            out["run"] = self.nav.start(plan)
        return out

    def _seen_now(self, target: dict, seq: int) -> dict | None:
        """Did the latest stored observation detect the target? Same label; colour must agree when both are known."""
        want_c = (target.get("color") or "unknown").lower()
        for o in self.last_objects:
            if o["label"].lower() != str(target.get("label", "")).lower():
                continue
            c = (o.get("color") or "unknown").lower()
            if o.get("id") == target.get("id") or want_c == "unknown" or c in ("unknown", want_c):
                return {"zone": o.get("zone", "center"), "id": o.get("id")}
        return None

    def set_place_manual(self, name: str) -> str:
        name = name.strip()
        if not name:
            return self.location
        self.mem.ensure_place(name, self.place_state.get("room_type") or "unknown", False)  # a room you name by hand is the kind of room being seen now
        self.location, self.place_mode = name, "manual"
        self._new_heading_frame()
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
