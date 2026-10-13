# Connecting the PuppyPi

Goal for Phases 1-4: the robot is *just a camera on the network*. No movement code is needed.

## 1. Network
Put the PuppyPi and the laptop on the same Wi-Fi (Hiwonder docs: network configuration + LAN mode, sections 3.5/3.6).
Find the robot IP (app, router, or `hostname -I` over VNC/SSH). Find the laptop IP (`ipconfig`).

## 2. Choose a camera path (try in this order)
**A. Pull (no code on the robot).** The Hiwonder image runs ROS with a web video server. In a browser open
`http://<puppy-ip>:8080/` and look for the camera stream/topic (commonly `/usb_cam/image_raw`). Then on the laptop:
```
CAMERA_SOURCE="http://<puppy-ip>:8080/stream?topic=/usb_cam/image_raw"
uv run puppy-mind watch
```
**B. Push (small script on the robot).** Copy `robot/puppy_streamer.py` to the Pi and run
```
python3 puppy_streamer.py --brain http://<laptop-ip>:8000 --source mjpeg --interval 5 --token $BRAIN_TOKEN
```
(`--source opencv:0` if nothing else holds the camera; `--source ros` for a ROS1 subscriber.)
Set `BRAIN_TOKEN` in the laptop's `.env` and pass the same token. Allow inbound TCP 8000 in the Windows firewall.

> Unverified on your unit: exact web_video_server port/topic and ROS version (docs list ROS1 and ROS2/Humble images).
> Confirm with step A in a browser before relying on defaults.

## 3. Name where the robot is
Memory is keyed by location name. Set it in the UI header (or `DEFAULT_LOCATION`) whenever you move the robot to a
new area, e.g. "Booth Table". Misses are only counted against the *same* location, so changing areas does not make
objects "disappear".

## 4. Good capture practice
Keep the robot still while it observes (stand/sit pose), steady lighting, camera at the area you want remembered.
`WATCH_INTERVAL_S=5` with near-duplicate frame skipping keeps OpenAI cost low.

## 5. Later (Phase 5, not implemented)
The Hiwonder repo (`ros2` branch: `puppy_control`, `puppy_navigation`, `puppy_slam`) exposes motion/navigation. Safe
"go to where you last saw X" needs: LiDAR SLAM pose per observation (store as `Observation.x,y,theta`), reachability
checks, obstacle avoidance, a supervised kill-switch, and an allow-list of named waypoints written from the graph.
