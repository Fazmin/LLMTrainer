#!/usr/bin/env python3
"""Generate the 1024x1024 app icon source (a falling learning curve on a rounded square). Pure standard library."""
import math, struct, sys, zlib

N = 1024
out = sys.argv[1] if len(sys.argv) > 1 else "src-tauri/icons/source.png"

def rounded_rect_cov(x, y, half, radius):
    # signed distance to a rounded square centred at (N/2, N/2)
    dx, dy = abs(x - N / 2) - (half - radius), abs(y - N / 2) - (half - radius)
    d = math.hypot(max(dx, 0), max(dy, 0)) + min(max(dx, dy), 0) - radius
    return max(0.0, min(1.0, 0.5 - d))

# learning curve: starts high on the left, drops fast, then flattens
pts = []
for i in range(0, 361):
    t = i / 360
    x = 200 + t * 624
    y = 280 + 430 * (1 - math.exp(-5.0 * t)) / (1 - math.exp(-5.0)) * 0.92 + 40 * t
    pts.append((x, y))

line = [[0.0] * N for _ in range(N)]
R = 38
for (cx, cy) in pts:
    for yy in range(int(cy - R - 2), int(cy + R + 3)):
        if 0 <= yy < N:
            row = line[yy]
            for xx in range(int(cx - R - 2), int(cx + R + 3)):
                if 0 <= xx < N:
                    d = math.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
                    c = max(0.0, min(1.0, R - d + 0.5))
                    if c > row[xx]:
                        row[xx] = c

# end-of-curve marker: a ring + dot
ex, ey = pts[-1]
def ring(x, y):
    d = math.hypot(x - ex, y - ey)
    return max(0.0, min(1.0, 0.5 - abs(d - 78) + 14))  # ring of half-width 14

raw = bytearray()
top, bot = (79, 70, 229), (37, 99, 235)  # indigo -> blue
for y in range(N):
    raw.append(0)
    t = y / (N - 1)
    base = tuple(top[k] + (bot[k] - top[k]) * t for k in range(3))
    for x in range(N):
        a = rounded_rect_cov(x + 0.5, y + 0.5, 480, 200)
        if a <= 0:
            raw += bytes((0, 0, 0, 0))
            continue
        w = max(line[y][x], 0.0)
        if x > ex - 100 and y > ey - 100 and x < ex + 100 and y < ey + 100:
            w = max(w, ring(x + 0.5, y + 0.5))
        r = base[0] + (255 - base[0]) * w
        g = base[1] + (255 - base[1]) * w
        b = base[2] + (255 - base[2]) * w
        raw += bytes((int(r), int(g), int(b), int(a * 255)))

def chunk(tag, data):
    c = struct.pack(">I", len(data)) + tag + data
    return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", N, N, 8, 6, 0, 0, 0)) \
    + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
open(out, "wb").write(png)
print("wrote", out, len(png), "bytes")
