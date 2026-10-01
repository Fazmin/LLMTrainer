#!/bin/sh
# Print the window id of the LLM Trainer app window, and nothing else, so a capture can target that one window:
#   screencapture -l "$(scripts/winid.sh)" -o out.png
osascript -l JavaScript -e '
ObjC.import("CoreGraphics");
const list = ObjC.deepUnwrap(ObjC.castRefToObject($.CGWindowListCopyWindowInfo($.kCGWindowListOptionAll, 0)));
const mine = list.filter(w => w.kCGWindowOwnerName === "llm-trainer-app" && w.kCGWindowName === "LLM Trainer");
mine.length ? String(mine[0].kCGWindowNumber) : "";
'
