"""`puppy-mind booth`: the booth's start command logs everything it prints and starts the robot app with Aura memory."""
import sys

from puppy_mind import cli


def test_booth_starts_the_robot_app_with_aura_and_logs_what_it_prints(tmp_path, monkeypatch):
    started = {}

    def fake_robot_test(robot_ip, port, light=True, combine=False, real_memory=False):
        started.update(ip=robot_ip, port=port, real_memory=real_memory)
        print("live: woken by 'Hey PuppyPi'")
        print("an error line", file=sys.stderr)
        return 0

    monkeypatch.setattr(cli, "_robot_test", fake_robot_test)
    out, err = sys.stdout, sys.stderr
    try:
        assert cli.main(["booth", "--robot-ip", "10.0.0.7", "--log", str(tmp_path / "booth.log")]) == 0
    finally:
        sys.stdout, sys.stderr = out, err
    assert started == {"ip": "10.0.0.7", "port": 8000, "real_memory": True}
    log = (tmp_path / "booth.log").read_text(encoding="utf-8")
    assert "booth started" in log and "woken by 'Hey PuppyPi'" in log and "an error line" in log
