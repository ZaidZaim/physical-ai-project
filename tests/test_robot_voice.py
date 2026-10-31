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
