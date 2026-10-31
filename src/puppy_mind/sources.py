"""Frame sources. The robot is "just a camera" for Phases 1-4.

  webcam:0                      laptop webcam (develop before the robot is wired up)
  http://<ip>:8080/stream?...   PuppyPi MJPEG stream (ROS web_video_server) pulled by the laptop
  folder:./frames               replay a folder of jpg/png (repeatable evals)
  (push)                        robot/puppy_streamer.py POSTs frames to /api/observe
"""
from __future__ import annotations

import io
import threading
import urllib.request
from pathlib import Path

import cv2
from PIL import Image


def ahash(jpeg: bytes) -> int:
    """8x8 average hash: cheap near-duplicate detection to avoid paying for identical frames."""
    img = Image.open(io.BytesIO(jpeg)).convert("L").resize((8, 8))
    px = list(img.tobytes())  # an 8-bit greyscale image: one byte per pixel (getdata is deprecated)
    avg = sum(px) / 64
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class FrameSource:
    name = "camera"

    def read(self) -> bytes | None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class CvSource(FrameSource):
    def __init__(self, target: str | int, name: str):
        self.target, self.name = target, name
        self.cap = cv2.VideoCapture(target)

    def read(self) -> bytes | None:
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()  # stream hiccup: reopen next tick
            self.cap = cv2.VideoCapture(self.target)
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else None


class MjpegSource(FrameSource):
    """Pulls an MJPEG-over-HTTP stream on a background thread and keeps ONLY the newest JPEG.

    Why not cv2.VideoCapture: it hands out frames in order, so if we read slower than the robot sends,
    frames queue up and the picture lags further behind every second. Here the reader never falls behind,
    old frames are dropped, and the JPEG bytes pass through untouched (no decode/re-encode, low CPU).
    """

    SOI, EOI = b"\xff\xd8", b"\xff\xd9"

    def __init__(self, url: str, name: str = "puppypi-mjpeg"):
        self.url, self.name = url, name
        self._latest: bytes | None = None
        self._seq = self._seen = 0
        self.error = ""
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="mjpeg-reader")
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=3) as r:
                    buf = b""
                    while not self._stop.is_set():
                        chunk = r.read1(65536)  # whatever has arrived; read(n) would wait to fill n bytes and batch frames
                        if not chunk:
                            break
                        buf += chunk
                        while True:
                            a = buf.find(self.SOI)
                            if a < 0:
                                buf = b""
                                break
                            b = buf.find(self.EOI, a + 2)
                            if b < 0:
                                buf = buf[a:]
                                break
                            self._latest = buf[a : b + 2]
                            self._seq += 1
                            self.error = ""
                            buf = buf[b + 2 :]
                        if len(buf) > 4_000_000:  # never grow unbounded on garbage
                            buf = b""
                    self.error = "stream ended"
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"[:160]
            self._stop.wait(1.0)  # reconnect

    def read(self) -> bytes | None:
        """Newest frame if it is new since the last call, else None."""
        if self._seq == self._seen:
            return None
        self._seen = self._seq
        return self._latest

    def close(self) -> None:
        self._stop.set()


class FolderSource(FrameSource):
    def __init__(self, path: str):
        self.files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.i = 0
        self.name = f"folder:{path}"

    def read(self) -> bytes | None:
        if self.i >= len(self.files):
            return None
        data = self.files[self.i].read_bytes()
        self.i += 1
        return data


def open_source(spec: str) -> FrameSource:
    if spec.startswith("webcam:"):
        return CvSource(int(spec.split(":", 1)[1]), spec)
    if spec.startswith("folder:"):
        return FolderSource(spec.split(":", 1)[1])
    if spec.startswith("http"):
        return MjpegSource(spec)
    raise ValueError(f"unknown camera source: {spec}")
