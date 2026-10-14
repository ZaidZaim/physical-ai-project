"""puppy-mind CLI.  `puppy-mind --help`"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .brain import Brain
from .config import get_settings


def _p(x) -> None:
    print(json.dumps(x, indent=2, default=str, ensure_ascii=False))


def _aura_reachable(s) -> str:
    """'' if the Aura login works, otherwise a short reason. Never raises."""
    if not (s.neo4j_uri and s.neo4j_password):
        return "NEO4J_URI / NEO4J_PASSWORD are not set in .env"
    try:
        from neo4j import GraphDatabase

        d = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password),
                                 connection_timeout=8, max_transaction_retry_time=2)
        try:
            d.verify_connectivity()
        finally:
            d.close()
        return ""
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:120]}"


def _robot_test(robot_ip: str, port: int, light: bool = True, combine: bool = False,
                real_memory: bool = False) -> int:
    import os
    import socket
    import threading
    import webbrowser

    import uvicorn

    from .robotcheck import _open, camera_url, check, find_robot, local_ips, pi_like_hosts, local_prefix

    ips = local_ips()
    if robot_ip == "auto":
        if any(i.startswith("192.168.149.") for i in ips):
            robot_ip = "192.168.149.1"  # on the robot's own hotspot
            print("Laptop is on the robot's own hotspot.")
        else:
            pre = local_prefix()
            print(f"Looking for the PuppyPi on your network ({pre}.x) ...")
            found = find_robot() if pre else []
            if len(found) == 1:
                robot_ip = found[0]
                print(f"Found it: {robot_ip}\n")
            elif len(found) > 1:
                print("Several robots/cameras answered: " + ", ".join(found))
                print("Re-run with --robot-ip <one of them>.")
                return 1
            else:
                print("\nNo PuppyPi found on this network.")
                pis = pi_like_hosts(pre) if pre else []
                if pis:
                    print("Devices here with a Raspberry Pi network chip (one may be the robot, e.g. with its camera/ROS services stopped):")
                    for ip, mac in pis:
                        print(f"   {ip}  {mac}")
                    print("Try: uv run puppy-mind robot-check <that ip>")
                else:
                    print("None of the devices here look like a Raspberry Pi, so the robot has not joined this Wi-Fi.")
                print("""
Put the robot on your home Wi-Fi ("LAN mode"). Easiest: WonderPi phone app -> choose PuppyPi -> "+" -> LAN Mode ->
home Wi-Fi password -> "Connect to Device Hotspot" (join HW..., password hiwonder) -> return. Success: robot icon in the app
and the expansion-board LED stays on. Hold the robot icon in the app to read its IP, then:
    uv run puppy-mind robot-test --robot-ip <that ip>
Or join the robot's own hotspot (HW...) and run this command again.""")
                return 1
    else:
        try:  # allow a hostname such as raspberrypi.fritz.box
            robot_ip = socket.gethostbyname(robot_ip)
        except OSError:
            print(f"Cannot resolve {robot_ip}")
            return 1

    prefix = robot_ip.rsplit(".", 1)[0] + "."
    mine = [i for i in ips if i.startswith(prefix)]
    if not mine:
        print(f"Note: this laptop has no address on {prefix}x (laptop: {', '.join(i for i in ips if not i.startswith(('172.', '169.254.'))) or 'unknown'}).")
        print("The robot is on a different network segment (guest Wi-Fi, repeater or other Wi-Fi?). Trying anyway.")
        laptop = next((i for i in ips if i.startswith(("192.168.", "10."))), "<laptop-ip>")
    else:
        laptop = mine[0]
    print(f"Robot: {robot_ip}   Laptop: {laptop}\n")

    code = check(robot_ip)
    cam = camera_url(robot_ip, light=light) if code == 0 else None
    if combine:
        os.environ["DRIVE_COMBINE"] = "1"

    from .config import get_settings

    s = get_settings()
    if real_memory:
        why = _aura_reachable(s)
        if why:
            print(f"\nAura is NOT reachable ({why}). Using in-RAM memory for now: nothing will be saved.")
            os.environ["PUPPY_FAKE_MEMORY"] = "1"
        else:
            print(f"\nAura connected ({s.neo4j_uri}): observations will be saved to your graph.")
            if not s.openai_api_key:
                print("OPENAI_API_KEY is not set: camera frames will NOT be analysed yet (video and driving still work).")
    else:
        os.environ["PUPPY_FAKE_MEMORY"] = "1"
    if not real_memory or os.environ.get("PUPPY_FAKE_MEMORY") == "1":
        os.environ["PERCEPTION"] = "off"  # driving test only: camera frames are NOT sent to OpenAI
        print("Frame analysis is OFF (frames stay on this computer). Use --real-memory with Aura reachable to turn it on.")
    elif s.openai_api_key:
        print(f"Frame analysis is ON: about one camera frame every {s.watch_interval_s:g}s is sent to OpenAI ({s.vision_model}) while the camera runs.")
    bridge = _open(robot_ip, 9090, 2.0)
    if bridge:
        os.environ["ROSBRIDGE_URL"] = f"ws://{robot_ip}:9090"
    os.environ["CAMERA_SOURCE"] = cam or ""  # never fall back to a laptop webcam while testing the robot
    base = f"http://{laptop}:{port}"
    if bridge:
        print(f"""
 DRIVING: the robot's rosbridge is reachable (ws://{robot_ip}:9090), so NOTHING needs to run on the robot.
 In the page: [Start camera], [Enable keyboard drive], then W/S forward-back, A/D turn, Space=stop, 1/2/3=speed.
 Start on speed 1 with the dog held off the ground. Keep a hand on it / the power switch: if this laptop
 crashes or Wi-Fi drops mid-walk the robot is NOT told to stop (the dead-man runs on the laptop).
 Camera: {cam or 'NOT FOUND'}
""")
    else:
        print(f"""
 The robot's rosbridge (port 9090) is NOT reachable, so keyboard driving needs a small script ON the robot:
   open VNC/SSH, enter the docker container, then run puppy_drive.py (see docs/ROBOT_SETUP.md section 4a), e.g.
     python3 -c "import urllib.request as u; u.urlretrieve('{base}/robot/puppy_drive.py','puppy_drive.py')"
     python3 puppy_drive.py --brain {base} --dry-run
 Without that you still get the live camera. Hiwonder's phone app / handle can move the robot meanwhile.
 Camera: {cam or 'NOT FOUND'}
""")
    threading.Timer(2.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run("puppy_mind.server:app", host="0.0.0.0", port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="puppy-mind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create constraints/indexes in Aura and verify connectivity")
    o = sub.add_parser("observe", help="describe one image file and write it to memory")
    o.add_argument("image")
    o.add_argument("--location")
    w = sub.add_parser("watch", help="continuously observe CAMERA_SOURCE")
    w.add_argument("--source")
    w.add_argument("--interval", type=float)
    a = sub.add_parser("ask", help="ask a question (omit for interactive chat)")
    a.add_argument("question", nargs="?")
    a.add_argument("--evidence", action="store_true")
    sd = sub.add_parser("seed", help="replay a scripted scenario (no camera/OpenAI needed)")
    sd.add_argument("scenario", nargs="?", default="eval/scenarios/booth.json")
    sub.add_parser("stats")
    sub.add_parser("reset").add_argument("--yes", action="store_true")
    e = sub.add_parser("eval", help="run the question set against current memory")
    e.add_argument("questions", nargs="?", default="eval/questions.json")
    sv = sub.add_parser("serve", help="web UI + robot ingest API")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--robot-test", action="store_true",
                    help="no Aura/OpenAI needed: live camera + keyboard drive only (memory is in-RAM)")
    sv.add_argument("--rosbridge", help="drive via the robot's rosbridge, e.g. ws://192.168.149.1:9090 (no robot terminal needed)")
    sv.add_argument("--camera", help="camera source for the Start camera button (overrides CAMERA_SOURCE)")
    rt = sub.add_parser("robot-test", help="one command: find robot on its hotspot, start server+camera, print robot commands")
    rt.add_argument("--robot-ip", default="auto", help="robot IP or hostname; auto = its hotspot, else scan your home network")
    rt.add_argument("--port", type=int, default=8000)
    rt.add_argument("--full-camera", action="store_true", help="full-size camera stream instead of the lighter 320x240")
    rt.add_argument("--real-memory", action="store_true", help="save observations to Aura (and analyse frames with OpenAI) instead of in-RAM memory")
    rt.add_argument("--combine", action="store_true", help="allow forward+turn together (risky on this robot)")
    rc = sub.add_parser("robot-check", help="probe the PuppyPi at an IP and find the working camera URL")
    rc.add_argument("ip", help="robot IP, or 'auto' to scan the local network")
    args = ap.parse_args(argv)

    if args.cmd == "robot-test":
        return _robot_test(args.robot_ip, args.port, light=not args.full_camera, combine=args.combine, real_memory=args.real_memory)

    if args.cmd == "robot-check":
        from .robotcheck import check, scan

        return scan(args.ip.split(":", 1)[1] if args.ip.startswith("auto:") else None) if args.ip.startswith("auto") else check(args.ip)

    if args.cmd == "serve":
        import os

        import uvicorn

        if args.robot_test:
            os.environ["PUPPY_FAKE_MEMORY"] = "1"
            print("ROBOT TEST MODE: memory is in-RAM and nothing is saved to Aura.")
        if args.camera:
            os.environ["CAMERA_SOURCE"] = args.camera
        if args.rosbridge:
            os.environ["ROSBRIDGE_URL"] = args.rosbridge

        uvicorn.run("puppy_mind.server:app", host=args.host, port=args.port)
        return 0

    s = get_settings()
    need_ai = args.cmd in {"observe", "watch", "ask", "eval"}
    if need_ai and not s.openai_api_key:
        print("OPENAI_API_KEY is not set (put it in .env)", file=sys.stderr)
        return 2
    brain = Brain(s, need_openai=need_ai)
    try:
        if args.cmd == "init":
            brain.mem.init_schema()
            _p({"connected": True, "database": s.neo4j_uri, "stats": brain.mem.stats(), "nams": brain.nams.enabled})
        elif args.cmd == "stats":
            _p(brain.mem.stats())
        elif args.cmd == "observe":
            _p(brain.observe_frame(Path(args.image).read_bytes(), source=args.image,
                                   location=args.location, skip_if_similar=False))
        elif args.cmd == "watch":
            from .sources import open_source

            spec = args.source or s.camera_source
            if not spec:
                print("No camera set. Use --source or CAMERA_SOURCE (webcam:0, http://<robot>:8080/stream?topic=..., folder:./frames).")
                return 2
            src = open_source(spec)
            interval = args.interval or s.watch_interval_s
            print(f"watching {src.name} every {interval}s - Ctrl+C to stop")
            brain.watch(src, interval, lambda r: print(json.dumps(r, default=str)[:400]))
            brain._watch.join()  # type: ignore[union-attr]
        elif args.cmd == "seed":
            from .scenario import replay

            brain.mem.init_schema()
            for r in replay(brain, args.scenario):
                print(r["ts"][:19], "new:", [x.split(" (")[0] for x in r["new_objects"]],
                      "missed:", [m["label"] for m in r["missed"]])
        elif args.cmd == "ask":
            if args.question:
                r = brain.ask(args.question)
                print(r["answer"])
                if args.evidence:
                    _p(r["evidence"])
            else:
                print("Ask about your environment (empty line to quit)")
                while (q := input("you> ").strip()):
                    print("puppy>", brain.ask(q)["answer"])
        elif args.cmd == "reset":
            if not args.yes:
                print("refusing: this deletes ALL robot memory. Re-run with --yes")
                return 1
            brain.mem.reset()
            print("memory cleared")
        elif args.cmd == "eval":
            from .evaluate import run

            return run(brain, args.questions)
    except KeyboardInterrupt:
        pass
    finally:
        brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
