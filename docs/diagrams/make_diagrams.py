"""Draw the README diagrams (architecture, graph model) as hand-laid-out SVGs, in a light and a dark version.

    uv run python docs/diagrams/make_diagrams.py

The README shows them with <picture>, so GitHub picks the version that matches the reader's theme.
"""
from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).parent
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", mute="#59636e", line="#8c959f", frame="#d1d9e0", panel="#f6f8fa",
                  robot="#2a7f9e", long="#b9691a", short="#2f8f55", nams="#7050cc", on_fill="#ffffff", label_bg="#ffffff"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", mute="#9198a1", line="#6e7681", frame="#3d444d", panel="#151b23",
                 robot="#4db3d6", long="#e3a047", short="#4cc27a", nams="#a98cf5", on_fill="#0d1117", label_bg="#0d1117"),
}


class Svg:
    def __init__(self, w: int, h: int, t: dict, title: str):
        self.w, self.h, self.t, self.parts = w, h, t, []
        self.title = title

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=14, weight=400, fill=None, anchor="middle", italic=False):
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill or self.t["ink"]}" '
                 f'text-anchor="{anchor}"{" font-style=\"italic\"" if italic else ""}>{s}</text>')

    def label(self, x, y, s, size=12):
        """Edge label on a background chip so it stays readable over lines."""
        w = 7.0 * len(s) * size / 12 + 12
        self.add(f'<rect x="{x - w / 2:.1f}" y="{y - size / 2 - 5:.1f}" width="{w:.1f}" height="{size + 10}" rx="4" fill="{self.t["label_bg"]}"/>')
        self.text(x, y + size / 2 - 1.5, s, size=size, fill=self.t["mute"])

    def arrow(self, d: str, color=None, dashed=False, end=True, start=False, width=1.6):
        c = color or self.t["line"]
        mid = c.lstrip("#")
        self.markers.add(mid)
        self.add(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{width}"{" stroke-dasharray=\"5 4\"" if dashed else ""}'
                 f'{f" marker-end=\"url(#a{mid})\"" if end else ""}{f" marker-start=\"url(#s{mid})\"" if start else ""}/>')

    def render(self) -> str:
        defs = "".join(
            f'<marker id="a{m}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M0 0L10 5L0 10z" fill="#{m}"/></marker>'
            f'<marker id="s{m}" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M0 0L10 5L0 10z" fill="#{m}"/></marker>' for m in sorted(self.markers))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'font-family="{FONT}" role="img" aria-label="{self.title}"><title>{self.title}</title>'
                f'<defs>{defs}</defs><rect width="100%" height="100%" fill="{self.t["bg"]}" rx="8"/>' + "".join(self.parts) + "</svg>\n")

    markers: set


def new(w, h, t, title) -> Svg:
    s = Svg(w, h, t, title)
    s.markers = set()
    return s


# ------------------------------------------------------------------ architecture
def box(s: Svg, x, y, w, h, title, sub, fill=None, stroke=None, title_fill=None, sub_fill=None):
    t = s.t
    s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill or t["panel"]}" stroke="{stroke or t["frame"]}" stroke-width="1.2"/>')
    s.text(x + w / 2, y + h / 2 - 4, title, size=15, weight=600, fill=title_fill)
    s.text(x + w / 2, y + h / 2 + 16, sub, size=12, fill=sub_fill or t["mute"])


def cylinder(s: Svg, cx, cy, w, h, title, sub, color):
    t = s.t
    rx, ry = w / 2, 9
    top, bot = cy - h / 2, cy + h / 2
    s.add(f'<path d="M{cx - rx} {top} v{h} a{rx} {ry} 0 0 0 {w} 0 v{-h} z" fill="{color}"/>')
    s.add(f'<ellipse cx="{cx}" cy="{top}" rx="{rx}" ry="{ry}" fill="{color}" stroke="{t["bg"]}" stroke-opacity=".35" stroke-width="1.5"/>')
    s.text(cx, cy + 2, title, size=15, weight=700, fill=t["on_fill"])
    s.text(cx, cy + 20, sub, size=12, fill=t["on_fill"])


def architecture(theme: str) -> str:
    t = THEMES[theme]
    s = new(900, 410, t, "puppy-mind architecture: perceive, remember, answer, act")
    # frames
    s.add(f'<rect x="12" y="34" width="196" height="360" rx="12" fill="none" stroke="{t["frame"]}" stroke-dasharray="6 5"/>')
    s.text(110, 24, "PuppyPi robot", size=13, weight=600, fill=t["mute"])
    s.add(f'<rect x="234" y="34" width="490" height="360" rx="12" fill="none" stroke="{t["frame"]}" stroke-dasharray="6 5"/>')
    s.text(479, 24, "Laptop: the brain (this repo)", size=13, weight=600, fill=t["mute"])
    s.text(820, 24, "Cloud", size=13, weight=600, fill=t["mute"])
    # robot
    rf = dict(fill=t["robot"], stroke=t["robot"], title_fill=t["on_fill"], sub_fill=t["on_fill"])
    box(s, 28, 70, 164, 76, "Sensors", "camera · gyro", **rf)
    box(s, 28, 282, 164, 76, "Legs", "rosbridge commands", **rf)
    # steps
    box(s, 252, 70, 210, 76, "1 · Perceive", "OpenAI vision → objects")
    box(s, 496, 70, 210, 76, "2 · Remember", "match objects · know the room")
    box(s, 496, 282, 210, 76, "3 · Answer", "agent with graph tools · UI")
    box(s, 252, 282, 210, 76, "4 · Act", "turn to an object · drive")
    # stores
    cylinder(s, 818, 206, 120, 62, "Neo4j Aura", "world model", t["long"])
    cylinder(s, 818, 338, 120, 50, "NAMS", "agent memory", t["nams"])
    # flow
    s.arrow("M192 108 H250")
    s.label(221, 92, "frames")
    s.arrow("M462 108 H494")
    s.arrow("M706 108 C780 108 818 120 818 166", color=t["long"])
    s.label(770, 116, "writes")
    s.arrow("M758 220 C730 232 706 250 640 280", color=t["long"])
    s.label(735, 254, "reads")
    s.arrow("M706 336 H756", color=t["nams"])
    s.label(731, 320, "chat")
    s.arrow("M818 238 V312", color=t["line"], dashed=True, end=False)
    s.label(862, 276, "same DB", size=11)
    s.arrow("M494 320 H464")
    s.arrow("M252 320 H194")
    s.label(222, 304, "turn")
    # captions under the robot boxes explain the loop
    s.text(110, 182, "sees the room", size=12, fill=t["mute"], italic=True)
    s.text(110, 262, "moves when you ask", size=12, fill=t["mute"], italic=True)
    s.arrow("M110 192 V252", color=t["frame"], dashed=True, end=False)
    return s.render()


# ------------------------------------------------------------------ graph model
def node(s: Svg, cx, cy, r, name, sub, color):
    t = s.t
    s.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
    s.text(cx, cy + (0 if sub else 5), name, size=15, weight=700, fill=t["on_fill"])
    if sub:
        s.text(cx, cy + 18, sub, size=11, fill=t["on_fill"])


def edge(s: Svg, a, b, ra, rb, name, color=None, dashed=False, bend=0.0, lpos=0.5, loff=(0, 0)):
    """Straight or bent edge between circle centres a and b, trimmed to the circles, with a label."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    mx, my = (ax + bx) / 2 + nx * bend, (ay + by) / 2 + ny * bend
    def trim(px, py, qx, qy, r):
        vx, vy = qx - px, qy - py
        d = math.hypot(vx, vy)
        return px + vx / d * r, py + vy / d * r
    sx, sy = trim(ax, ay, mx, my, ra + 2)
    ex, ey = trim(bx, by, mx, my, rb + 4)
    s.arrow(f"M{sx:.1f} {sy:.1f} Q{mx:.1f} {my:.1f} {ex:.1f} {ey:.1f}", color=color, dashed=dashed)
    t_ = lpos
    lx = (1 - t_) ** 2 * sx + 2 * (1 - t_) * t_ * mx + t_ ** 2 * ex + loff[0]
    ly = (1 - t_) ** 2 * sy + 2 * (1 - t_) * t_ * my + t_ ** 2 * ey + loff[1]
    s.label(lx, ly, name)


def loop(s: Svg, cx, cy, r, side, name, lines=None):
    """Self-relationship drawn as a loop on one side of the node."""
    t = s.t
    if side == "right":
        a1, a2 = (cx + r * math.cos(-0.5), cy + r * math.sin(-0.5)), (cx + r * math.cos(0.5), cy + r * math.sin(0.5))
        c1, c2 = (cx + r + 70, cy - 52), (cx + r + 70, cy + 52)
        lx, ly = cx + r + 70, cy
    else:  # below
        a1, a2 = (cx + r * math.cos(2.0), cy + r * math.sin(2.0)), (cx + r * math.cos(1.14), cy + r * math.sin(1.14))
        c1, c2 = (cx - 60, cy + r + 72), (cx + 60, cy + r + 72)
        lx, ly = cx, cy + r + 66
    s.arrow(f"M{a1[0]:.1f} {a1[1]:.1f} C{c1[0]} {c1[1]} {c2[0]} {c2[1]} {a2[0]:.1f} {a2[1]:.1f}")
    for i, ln in enumerate(lines or [name]):
        s.label(lx, ly + i * 20, ln)


def graph_model(theme: str) -> str:
    t = THEMES[theme]
    s = new(900, 520, t, "puppy-mind graph model")
    R = {"Robot": (120, 112), "Session": (430, 112), "Place": (740, 150), "Sighting": (300, 320), "Item": (610, 320)}
    r = 46
    # edges first, nodes on top
    edge(s, R["Robot"], R["Session"], r, r, "RAN")
    edge(s, R["Robot"], R["Place"], r, r + 4, "CURRENTLY_IN", bend=-150, lpos=0.5)
    edge(s, R["Sighting"], R["Session"], r, r, "IN_SESSION", lpos=0.45)
    edge(s, R["Sighting"], R["Place"], r, r + 4, "TAKEN_AT", bend=-30, lpos=0.38)
    edge(s, R["Sighting"], R["Item"], r, r, "DETECTED", color=t["short"], bend=-26, lpos=0.5)
    edge(s, R["Sighting"], R["Item"], r, r, "MISSED", dashed=True, bend=26, lpos=0.5)
    edge(s, R["Item"], R["Place"], r, r + 4, "LOCATED_AT", color=t["long"], lpos=0.55, loff=(44, 0))
    loop(s, *R["Place"], r + 4, "right", "ADJACENT_TO")
    loop(s, *R["Item"], r, "below", "", lines=["ON · INSIDE · NEXT_TO", "NEAR · IN_FRONT_OF · BEHIND"])
    node(s, *R["Robot"], r, "Robot", "", t["robot"])
    node(s, *R["Session"], r, "Session", "one run", t["short"])
    node(s, *R["Sighting"], r, "Sighting", "one frame", t["short"])
    node(s, *R["Item"], r, "Item", "a thing", t["long"])
    node(s, *R["Place"], r + 4, "Place", "a room", t["long"])
    # legend
    y = 502
    for i, (c, txt, dash) in enumerate([(t["long"], "long-term: the world", False), (t["short"], "short-term: what was seen, when", False),
                                        (t["line"], "dashed: expected but not seen", True)]):
        x = 24 + i * 250
        if dash:
            s.add(f'<path d="M{x} {y - 4} h22" stroke="{c}" stroke-width="2" stroke-dasharray="5 4"/>')
        else:
            s.add(f'<circle cx="{x + 8}" cy="{y - 4}" r="7" fill="{c}"/>')
        s.text(x + 30, y, txt, size=12, fill=t["mute"], anchor="start")
    return s.render()


if __name__ == "__main__":
    for theme in THEMES:
        (OUT / f"architecture-{theme}.svg").write_text(architecture(theme), encoding="utf-8")
        (OUT / f"graph-model-{theme}.svg").write_text(graph_model(theme), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
