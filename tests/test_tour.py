"""Real-scene tours: record -> load -> replay, on in-RAM memory (no Aura, no OpenAI)."""
import dataclasses

from puppy_mind import tour
from puppy_mind.config import get_settings
from puppy_mind.models import Detection, Scene


def scene(room, *objs):
    return Scene(summary=room, room_type=room, room_confidence=0.9, relations=[],
                 objects=[Detection(idx=i, label=l, color=c, description="", zone="center", confidence=0.9) for i, (l, c) in enumerate(objs)])


KITCHEN = scene("kitchen", ("fridge", "white"), ("stove", "black"), ("table", "brown"))
HALL = scene("hallway", ("door", "brown"), ("stairs", "brown"), ("shoe", "black"))
STAIRS_AS_GARAGE = scene("garage", ("stairs", "brown"))


def test_a_recorded_tour_loads_back_frame_for_frame(tmp_path):
    rec = tour.TourRecorder(tmp_path / "t")
    rec.add(KITCHEN, b"\xff\xd8jpeg", ts="2026-10-31T10:00:00+00:00", session="s1", place="Kitchen")
    rec.add(HALL, None, ts="2026-10-31T10:00:05+00:00", session="s1", place="Hallway", note="")
    frames = tour.load(tmp_path / "t")
    assert [f["place"] for f in frames] == ["Kitchen", "Hallway"]
    assert frames[0]["image"] == "frames/00001.jpg" and (tmp_path / "t" / frames[0]["image"]).read_bytes() == b"\xff\xd8jpeg"
    assert Scene.model_validate(frames[1]["scene"]) == HALL


def test_replay_reports_rooms_and_agreement_and_never_needs_aura(tmp_path):
    frames = [{"ts": f"2026-10-31T10:00:{i:02d}+00:00", "session": "s1", "place": p, "scene": sc.model_dump()}
              for i, (p, sc) in enumerate([("Kitchen", KITCHEN), ("Kitchen", KITCHEN), ("Hallway", HALL), ("Hallway", HALL)])]
    # a restart, then the stairs mistaken for a garage before the hallway is recognised
    frames += [{"ts": f"2026-10-31T11:00:{i:02d}+00:00", "session": "s2", "place": "Hallway", "scene": sc.model_dump()}
               for i, sc in enumerate([STAIRS_AS_GARAGE, HALL])]
    s = dataclasses.replace(get_settings(), neo4j_uri="bolt://nowhere.invalid:1", place_confirm_frames=2, new_place_confirm_frames=2)
    res = tour.replay(frames, s)
    assert res["rooms"] == ["Hallway", "Kitchen"]  # no 'Garage'
    assert res["agreement"] == 1.0 and res["frames"] == 6
    assert res["rows"][4]["now"] is None and res["rows"][5]["now"] == "Hallway"


def test_replay_leaves_the_environment_as_it_found_it(monkeypatch):
    import os

    monkeypatch.delenv("PUPPY_FAKE_MEMORY", raising=False)
    tour.replay([], dataclasses.replace(get_settings(), neo4j_uri="bolt://nowhere.invalid:1"))
    assert "PUPPY_FAKE_MEMORY" not in os.environ


def test_record_flag_is_accepted_by_the_commands_that_take_it(monkeypatch, tmp_path):
    """--record crashed 'serve' on start (an 'import os' further down main() made os a local name)."""
    import os

    from puppy_mind import cli

    monkeypatch.delenv("RECORD_TOUR", raising=False)
    started = {}
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: started.setdefault("ok", True))
    monkeypatch.setenv("PUPPY_FAKE_MEMORY", "1")
    cli.main(["serve", "--record", str(tmp_path / "t")])
    assert started.get("ok") and os.environ.get("RECORD_TOUR") == str(tmp_path / "t")
