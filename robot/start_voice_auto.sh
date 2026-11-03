#!/bin/bash
# Runs ON the PuppyPi, started by puppy-voice.service at boot (and again whenever the voice client stops).
# Finds the laptop running Puppy Mind on the robot's current Wi-Fi (home, phone hotspot, ...), fetches the latest puppy_voice.py
# from it and runs it. The laptop's address is remembered in ~/brain_url, and searched for again when it no longer answers.
# The log starts empty at every boot: what visitors said is kept no longer than the current day.
cd ~ || exit 1
LOG=~/puppy_voice.log
if [ ! -f /tmp/puppy_voice_booted ]; then : > "$LOG"; touch /tmp/puppy_voice_booted; fi

answers() { [ "$(timeout 3 curl -s -o /dev/null -w '%{http_code}' "$1/robot/puppy_voice.py" 2>/dev/null)" = "200" ]; }

BRAIN=$(cat ~/brain_url 2>/dev/null)
if [ -z "$BRAIN" ] || ! answers "$BRAIN"; then
  NET=$(ip -4 -o addr show wlan0 2>/dev/null | awk '{print $4}' | head -1)   # e.g. 10.69.54.176/24
  if [ -z "$NET" ]; then echo "$(date +%T) no Wi-Fi address yet: trying again soon"; sleep 5; exit 1; fi
  ME=${NET%/*}; BASE=${ME%.*}
  echo "$(date +%T) looking for the laptop on $BASE.0/24 ..."
  FOUND=$(mktemp)
  for i in $(seq 1 254); do
    ip="$BASE.$i"; [ "$ip" = "$ME" ] && continue
    ( answers "http://$ip:8000" && echo "http://$ip:8000" >> "$FOUND" ) &
  done
  wait
  BRAIN=$(head -1 "$FOUND"); rm -f "$FOUND"
  if [ -z "$BRAIN" ]; then echo "$(date +%T) laptop app not found on this Wi-Fi: trying again soon"; sleep 10; exit 1; fi
  echo "$BRAIN" > ~/brain_url
fi

if wget -qO ~/puppy_voice.py.new "$BRAIN/robot/puppy_voice.py" && [ -s ~/puppy_voice.py.new ]; then
  mv ~/puppy_voice.py.new ~/puppy_voice.py
else
  rm -f ~/puppy_voice.py.new   # keep the copy we have
fi
echo "$(date +%T) starting the voice client (brain $BRAIN)"
exec python3 -u ~/puppy_voice.py --brain "$BRAIN"
