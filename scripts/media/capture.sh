#!/bin/sh
# Records the LLM Trainer window (that one window, never the screen) while scripts/dev/readme_tour.js drives it.
#
#   scripts/media/capture.sh OUT_DIR
#
# Writes OUT_DIR/frames/f00001.jpg ... about one a second, and OUT_DIR/stills/<name>.png whenever the tour announces
# `SHOT <name>` on the dev-report log. Stops when the tour announces DONE. Encode the frames afterwards with
# scripts/media/encode.swift. See the README section "Screenshots and video" for the whole procedure.
set -eu
OUT="${1:?usage: capture.sh OUT_DIR}"
LOG=/tmp/minagi-dev-report.log
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/frames" "$OUT/stills"

# wait for the window to exist
WID=""
for _ in $(seq 1 240); do
  WID="$(sh "$HERE/../winid.sh")"
  [ -n "$WID" ] && break
  sleep 0.5
done
[ -n "$WID" ] || { echo "no LLM Trainer window found" >&2; exit 1; }
echo "capturing window $WID into $OUT"

# keep the display awake for the length of the recording (a sleeping display produces blank captures)
caffeinate -u -d -t 2400 >/dev/null 2>&1 &
KEEP=$!
trap 'kill $KEEP 2>/dev/null || true' EXIT

n=0
seen=0
done_at=0
while :; do
  n=$((n + 1))
  screencapture -l"$WID" -o -x -t jpg "$OUT/frames/$(printf 'f%05d' "$n").jpg" 2>/dev/null || true

  if [ -f "$LOG" ]; then
    total="$(wc -l < "$LOG" | tr -d ' ')"
    if [ "$total" -gt "$seen" ]; then
      tail -n +"$((seen + 1))" "$LOG" | while IFS= read -r line; do
        case "$line" in
          *SHOT\ *)
            name="$(printf '%s' "$line" | sed -E 's/.*SHOT ([A-Za-z0-9_-]+).*/\1/')"
            screencapture -l"$WID" -o -x "$OUT/stills/$name.png" 2>/dev/null || true
            echo "still: $name (frame $n)"
            ;;
          *NOTE*) echo "$line" ;;
        esac
      done
      seen="$total"
      tail -n +1 "$LOG" | grep -q '"DONE"' && done_at=$((done_at == 0 ? n : done_at))
    fi
  fi
  # a few frames after DONE, then stop
  if [ "$done_at" -gt 0 ] && [ "$n" -ge $((done_at + 4)) ]; then break; fi
  sleep 0.7
done
echo "captured $n frames"
