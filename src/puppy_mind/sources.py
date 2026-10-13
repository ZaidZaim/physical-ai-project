"""Frame sources. The robot is "just a camera" for Phases 1-4.

  webcam:0                      laptop webcam (develop before the robot is wired up)
  http://<ip>:8080/stream?...   PuppyPi MJPEG stream (ROS web_video_server) pulled by the laptop
  folder:./frames               replay a folder of jpg/png (repeatable evals)
  (push)                        robot/puppy_streamer.py POSTs frames to /api/observe
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
from PIL import Image


def ahash(jpeg: bytes) -> int:
    """8x8 average hash: cheap near-duplicate detection to avoid paying for identical frames."""
    img = Image.open(io.BytesIO(jpeg)).convert("L").resize((8, 8))
    px = list(img.getdata())
    avg = sum(px) / 64
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class FrameSource:
    name = "camera"

    def read(self) -> bytes | None:  # pragma: no cover - interface
        raise NotImplementedError


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
        return CvSource(spec, "puppypi-mjpeg")
    raise ValueError(f"unknown camera source: {spec}")
