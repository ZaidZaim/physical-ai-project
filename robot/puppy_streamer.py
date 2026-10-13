#!/usr/bin/env python3
"""Runs ON the PuppyPi (Raspberry Pi). Grabs camera frames and POSTs them to the brain.

Stdlib + OpenCV only (OpenCV ships on the PuppyPi image). The robot does no inference: the
laptop runs perception, memory and Q&A.

Usage on the robot:
  python3 puppy_streamer.py --brain http://<laptop-ip>:8000 --source mjpeg --interval 5
  python3 puppy_streamer.py --brain http://<laptop-ip>:8000 --source opencv:0

Sources:
  opencv:N   cv2.VideoCapture(N). Fails if Hiwonder's own ROS camera node already holds the camera.
  mjpeg      reads the robot's local ROS web_video_server stream (default URL below). Use this
             when the stock services are running (recommended; it coexists with them).
  ros        ROS1 subscriber on /usb_cam/image_raw (needs rospy + cv_bridge, present on the image).
Verify the mjpeg URL / topic name on your unit (see docs/ROBOT_SETUP.md); they differ by image version.
"""
import argparse
import time
import urllib.request
import uuid

import cv2

DEFAULT_MJPEG = "http://127.0.0.1:8080/stream?topic=/usb_cam/image_raw"


def open_frames(source):
    if source == "mjpeg":
        cap = cv2.VideoCapture(DEFAULT_MJPEG)
        return lambda: cap.read()[1]
    if source.startswith("opencv:"):
        cap = cv2.VideoCapture(int(source.split(":")[1]))
        return lambda: cap.read()[1]
    if source == "ros":
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        rospy.init_node("puppy_streamer", anonymous=True, disable_signals=True)
        bridge, latest = CvBridge(), {}
        rospy.Subscriber("/usb_cam/image_raw", Image, lambda m: latest.update(f=bridge.imgmsg_to_cv2(m, "bgr8")))
        return lambda: latest.get("f")
    raise SystemExit("unknown --source")


def post(url, jpeg, token, location, source):
    b = uuid.uuid4().hex
    parts = []
    for k, v in (("location", location), ("source", source)):
        if v:
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="f.jpg"\r\n'
                 f"Content-Type: image/jpeg\r\n\r\n".encode() + jpeg + f"\r\n--{b}--\r\n".encode())
    req = urllib.request.Request(url + "/api/observe", data=b"".join(parts), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={b}")
    if token:
        req.add_header("X-Brain-Token", token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--source", default="mjpeg")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--token", default="")
    ap.add_argument("--location", default="")
    a = ap.parse_args()
    grab = open_frames(a.source)
    while True:
        try:
            frame = grab()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    print(post(a.brain, buf.tobytes(), a.token, a.location, "puppypi-push"))
        except Exception as e:  # keep streaming through wifi blips
            print("error:", e)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
