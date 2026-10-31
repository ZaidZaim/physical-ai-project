"""robot/puppy_voice.py runs on the Pi (stdlib only): it must find the WonderEcho Pro and set its own speech threshold."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("puppy_voice", Path(__file__).parents[1] / "robot" / "puppy_voice.py")
pv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv)

CAPTURE = """**** List of CAPTURE Hardware Devices ****
card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""
PLAYBACK = """**** List of PLAYBACK Hardware Devices ****
card 0: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]
card 1: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
card 2: Device [USB Audio Device], device 0: USB Audio [USB Audio]
"""


def test_finds_the_usb_sound_card_and_skips_the_pis_own_outputs():
    assert pv.pick_card(CAPTURE) == "plughw:1,0"
    assert pv.pick_card(PLAYBACK) == "plughw:2,0"
    assert pv.pick_card("card 3: WonderEcho [WonderEcho Pro], device 0: X [X]") == "plughw:3,0"
    assert pv.pick_card("") is None and pv.pick_card("card 0: Headphones [bcm2835 Headphones], device 0: a [a]") is None


def test_threshold_sits_well_above_room_noise_but_never_too_low():
    assert pv.auto_threshold([50, 60, 80, 70]) == 300.0
    assert pv.auto_threshold([200, 300, 400, 350, 250]) == 1200.0
    assert pv.auto_threshold([]) == 300.0


def test_a_turn_keeps_the_conversation_and_listening_resumes_only_after_the_reply(monkeypatch):
    from types import SimpleNamespace as NS

    sent, played = [], []

    class Resp:
        status, headers = 200, {"X-Encounter": "enc-1", "X-Heard": "hi", "X-Reply": "hello"}

        def read(self):
            return b"RIFF"

    monkeypatch.setattr(pv, "post_audio", lambda brain, wav, token, enc: sent.append(enc) or Resp())
    monkeypatch.setattr(pv.subprocess, "run", lambda *a, **k: played.append(1) or NS(returncode=0, stderr=b""))
    monkeypatch.setattr(pv.time, "sleep", lambda s: None)
    v = pv.Voice(NS(brain="http://x", token="", speaker="plughw:2,0", forget_after=120.0))
    v.talking.set()  # set by listen() when the sentence ended
    v.turn(b"\0" * 32000)
    assert not v.talking.is_set() and played == [1]  # listening again only after the reply was played
    v.talking.set()
    v.turn(b"\0" * 32000)
    assert sent == [None, "enc-1"]  # the second sentence continues the same conversation


def test_a_loud_start_up_measurement_never_makes_the_robot_deaf():
    # seen on the robot: someone talked during the 2 s measurement -> threshold 88986, above any possible level
    assert pv.auto_threshold([30000] * 60) == pv.DEFAULT_THRESHOLD
    assert pv.auto_threshold([1500] * 60) == 4000.0  # capped: a speaking voice must always get through
