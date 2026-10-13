"""`puppy-mind robot-check <ip>`: find out what the PuppyPi is serving and which camera URL works.

Probes (read-only, short timeouts): SSH 22, rosbridge 9090, web_video_server 8080, VNC 5900. Then reads the
web_video_server index for camera topics and confirms a real JPEG frame arrives on the best candidate.
"""
from __future__ import annotations

import re
import socket
import urllib.request

PORTS = {22: "SSH", 5900: "VNC", 8080: "web_video_server (camera)", 9090: "rosbridge"}
PREFER = ("usb_cam/image_raw", "image_raw", "image_rect_color", "rgb/image_raw", "camera")


def _open(ip: str, port: int, t: float = 1.5) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=t):
            return True
    except OSError:
        return False


def _first_jpeg(url: str, t: float = 6.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=t) as r:
            blob = r.read(300_000)
        return b"\xff\xd8" in blob
    except Exception:
        return False


def local_ips() -> list[str]:
    try:
        return sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    except OSError:
        return []


def camera_url(ip: str) -> str | None:
    """Best working MJPEG URL on the robot's web_video_server, or None."""
    topics: list[str] = []
    try:
        with urllib.request.urlopen(f"http://{ip}:8080/", timeout=4) as r:
            html = r.read().decode("utf-8", "replace")
        topics = list(dict.fromkeys(re.findall(r"topic=([^&\"'<> ]+)", html)))
    except Exception:
        pass
    cands = sorted(topics, key=lambda t: next((i for i, p in enumerate(PREFER) if p in t), 99)) or ["/usb_cam/image_raw"]
    for t in cands[:4]:
        url = f"http://{ip}:8080/stream?topic={t}"
        if _first_jpeg(url):
            return url
    return None


def check(ip: str) -> int:
    print(f"Probing {ip} ...")
    up = {p: _open(ip, p) for p in PORTS}
    for p, name in PORTS.items():
        print(f"  {'OPEN  ' if up[p] else 'closed'} {p:5d}  {name}")
    if not any(up.values()):
        print("\nNothing answered. Check: laptop is on the robot's Wi-Fi/LAN, the IP is right, the robot is powered and booted"
              "\n(~1 min). Robot's own hotspot mode is usually 192.168.149.1; in LAN mode read the IP from the Hiwonder app/router.")
        return 1
    if not up[8080]:
        print("\nPort 8080 is closed, so the camera server isn't running. Over SSH/VNC start the camera app or use robot/puppy_streamer.py"
              "\n(push mode). See docs/ROBOT_SETUP.md.")
        return 2
    topics: list[str] = []
    try:
        with urllib.request.urlopen(f"http://{ip}:8080/", timeout=4) as r:
            html = r.read().decode("utf-8", "replace")
        topics = list(dict.fromkeys(re.findall(r"topic=([^&\"'<> ]+)", html)))
    except Exception as e:
        print(f"\nCould not read http://{ip}:8080/ ({e}).")
    cands = sorted(topics, key=lambda t: next((i for i, p in enumerate(PREFER) if p in t), 99)) or ["/usb_cam/image_raw"]
    print("\nCamera topics found:" if topics else "\nNo topics listed; guessing:")
    for t in cands:
        print("  ", t)
    for t in cands[:4]:
        url = f"http://{ip}:8080/stream?topic={t}"
        ok = _first_jpeg(url)
        print(f"  {'WORKS' if ok else 'no frame'}  {url}")
        if ok:
            print(f'\nUse:  CAMERA_SOURCE="{url}"   (set in .env, then click "Start camera")')
            return 0
    print("\nServer is up but no topic produced a frame. Is the camera node running (is the stock app started)?")
    return 3


def local_prefix() -> str | None:
    """First three octets of a real home/office LAN address. Never a VPN/overlay (WARP/Tailscale use 100.64/10),
    never Hyper-V/WSL virtual switches (172.x): scanning those would hit networks that are not yours to probe."""
    try:
        addrs = {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        return None
    for want in ("192.168.", "10."):
        for ip in sorted(addrs):
            if ip.startswith(want):
                return ".".join(ip.split(".")[:3])
    return None


def scan(prefix: str | None = None) -> int:
    """Look for a PuppyPi on the local /24: hosts with the camera server (8080) or rosbridge (9090) open."""
    from concurrent.futures import ThreadPoolExecutor

    pre = prefix or local_prefix()
    if not pre:
        print("Could not find a home/office LAN address (192.168.x / 10.x). Pass it: robot-check auto:192.168.1")
        return 1
    print(f"Scanning {pre}.0/24 for ports 8080/9090/22 (own network, read-only connects)...")

    def probe(i: int):
        ip = f"{pre}.{i}"
        r = {p: _open(ip, p, 0.6) for p in (8080, 9090, 22)}
        return ip, r

    with ThreadPoolExecutor(64) as ex:
        hits = [(ip, r) for ip, r in ex.map(probe, range(1, 255)) if any(r.values())]
    if not hits:
        print("No hosts with those ports open. Is the robot on THIS Wi-Fi (LAN mode)? Is the laptop VPN/WARP blocking local traffic?")
        return 1
    for ip, r in hits:
        likely = r[8080] or r[9090]
        print(f"  {ip:15s} " + " ".join(f"{p}:{'open' if o else '-'}" for p, o in r.items()) + ("   <-- likely PuppyPi" if likely else ""))
    return 0
