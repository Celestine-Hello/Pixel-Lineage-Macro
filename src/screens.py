from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import cv2
import numpy as np

from ocr import Line

REF_W = 1920.0
REF_H = 1008.0

DEFAULT_AREAS = {
    "mg_area": {"x": 0.18, "y": 0.05, "w": 0.64, "h": 0.42},
    "en_area": {"x": 0.01, "y": 0.703, "w": 0.163, "h": 0.044},
    "skills_area": {"x": 0.18, "y": 0.385, "w": 0.276, "h": 0.28},
    "path_area": {"x": 0.31, "y": 0.34, "w": 0.52, "h": 0.52},
    "event_area": {"x": 0.33, "y": 0.10, "w": 0.48, "h": 0.65},
    "continue_area": {"x": 0.25, "y": 0.56, "w": 0.50, "h": 0.16},
    "skip_area": {"x": 0.20, "y": 0.08, "w": 0.65, "h": 0.84},
    "follow_light_area": {"x": 0.20, "y": 0.08, "w": 0.65, "h": 0.84},
}

BANNER_BAND = (0.30, 0.08, 0.45, 0.18)
VICTORY_BAND = (0.12, 0.06, 0.76, 0.26)
CHEST_BAND = {"x": 0.28, "y": 0.12, "w": 0.50, "h": 0.45}
TURN_BAND = (0.150, 0.71, 0.055, 0.25)
BOX_BAND = (0.002, 0.71, 0.030, 0.25)
MENU_BAND = {"x": 0.20, "y": 0.08, "w": 0.65, "h": 0.84}
BACK_BAND = (0.10, 0.70, 0.07, 0.045)
ROW_TOP = 0.7698
ROW_STEP = 0.0362
GUARD_ROW = 3
ROW_X = (0.015, 0.17)
USUAL_CARDS = ({"x": 0.573, "y": 0.434}, {"x": 0.573, "y": 0.597}, {"x": 0.573, "y": 0.762})
PATH_CONFIRM = ({"x": 0.5, "y": 0.62}, {"x": 0.5, "y": 0.69}, {"x": 0.5, "y": 0.55})

_EN = re.compile(r"E\s*N\W{0,2}([0-9OoIl|]{1,3})\s*/\s*([0-9OoIl|]{1,3})")
_COST = re.compile(r"\s\d+\s*EN\b.*$", re.IGNORECASE)


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


@dataclass
class Button:
    box: Box
    active: bool


def ui_ref(w: int, h: int) -> float:
    return max(320.0, min(float(w), h * (REF_W / REF_H)))


def area_px(area: dict, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(w - 1, int(round(area["x"] * w))))
    y0 = max(0, min(h - 1, int(round(area["y"] * h))))
    x1 = max(x0 + 1, min(w, int(round((area["x"] + area["w"]) * w))))
    y1 = max(y0 + 1, min(h, int(round((area["y"] + area["h"]) * h))))
    return x0, y0, x1, y1


def crop(frame: np.ndarray | None, area: dict) -> tuple[np.ndarray | None, int, int]:
    if frame is None:
        return None, 0, 0
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = area_px(area, w, h)
    return frame[y0:y1, x0:x1], x0, y0


def _channels(img: np.ndarray):
    px = img.astype(np.int16)
    return px[..., 0], px[..., 1], px[..., 2]


def find_button(img: np.ndarray | None, client_w: int, client_h: int) -> Button | None:
    if img is None or img.size == 0:
        return None
    b, g, r = _channels(img)
    gold = (np.abs(r - 231) <= 22) & (np.abs(g - 196) <= 22) & (np.abs(b - 120) <= 28)
    grey = (np.abs(r - 127) <= 12) & (np.abs(g - 108) <= 12) & (np.abs(b - 66) <= 12)
    min_w = max(40, int(client_w * 0.05))
    min_h = max(12, int(client_h * 0.02))
    best: Button | None = None
    for mask, active in ((gold, True), (grey, False)):
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if w < min_w or h < min_h or w < 2.5 * h or area < 0.7 * w * h:
                continue
            if best is None or w * h > best.box.w * best.box.h:
                best = Button(Box(x, y, w, h), active)
    return best


def path_banner(frame: np.ndarray | None) -> bool:
    if frame is None:
        return False
    h, w = frame.shape[:2]
    x, y, bw, bh = BANNER_BAND
    band = frame[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)]
    if band.size == 0:
        return False
    b, g, r = _channels(band)
    brown = (r >= 105) & (r <= 165) & (g >= 70) & (g <= 120) & (b >= 35) & (b <= 85) & (r - b >= 50)
    return float(brown.mean()) >= 0.06


def _gold_run(row: np.ndarray, lo: int, hi: int) -> tuple[int, int] | None:
    idx = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
    best = None
    for s, e in zip(idx[::2].tolist(), idx[1::2].tolist()):
        if lo <= e - s <= hi and (best is None or e - s > best[1] - best[0]):
            best = (s, e)
    return best


def victory_banner(frame: np.ndarray | None) -> bool:
    if frame is None:
        return False
    h, w = frame.shape[:2]
    x, y, bw, bh = VICTORY_BAND
    x0, y0 = int(x * w), int(y * h)
    band = frame[y0:int((y + bh) * h), x0:int((x + bw) * w)]
    if band.size == 0:
        return False
    b, g, r = _channels(band)
    gold = (np.abs(r - 231) <= 45) & (np.abs(g - 196) <= 45) & (np.abs(b - 120) <= 60)
    ref = ui_ref(w, h)
    ref_h = ref / (REF_W / REF_H)
    lo, hi = int(ref * 0.27), int(ref * 0.50)
    lines = []
    for i in np.flatnonzero(gold.sum(axis=1) >= lo).tolist():
        run = _gold_run(gold[i], lo, hi)
        if run is not None and abs(x0 + (run[0] + run[1]) / 2.0 - w / 2.0) <= w * 0.05:
            lines.append((i, run[0], run[1]))
    for k, (ya, sa, ea) in enumerate(lines):
        for yb, sb, eb in lines[k + 1:]:
            if 0.055 * ref_h <= yb - ya <= 0.10 * ref_h and abs(sa - sb) <= ref * 0.012 and abs(ea - eb) <= ref * 0.012:
                return True
    return False


def _border_column(frame: np.ndarray | None, band: tuple[float, float, float, float]) -> bool:
    if frame is None:
        return False
    h, w = frame.shape[:2]
    x, y, bw, bh = band
    reg = frame[int(y * h):int((y + bh) * h), int(x * w):max(int(x * w) + 1, int((x + bw) * w))]
    if reg.size == 0:
        return False
    b, g, r = _channels(reg)
    edge = (np.abs(r - 80) <= 28) & (np.abs(g - 110) <= 28) & (np.abs(b - 143) <= 30)
    return float(edge.mean(axis=0).max()) >= 0.5


def your_turn(frame: np.ndarray | None) -> bool:
    return _border_column(frame, TURN_BAND)


def in_battle(frame: np.ndarray | None) -> bool:
    return _border_column(frame, BOX_BAND)


def is_retry(text: str) -> bool:
    t = norm(text)
    return "AGAIN" in t or "RETRY" in t or "PLAY AG" in t


def is_chest(text: str) -> bool:
    t = norm(text)
    return "CLICK TO" in t or "TO OPEN" in t


def button_kind(text: str) -> str | None:
    t = norm(text)
    if not t:
        return None
    if "BUY" in t:
        return "buy"
    if is_retry(t):
        return "retry"
    if "CONTINUE" in t or "CONTIN" in t:
        return "continue"
    if "FOLLOW THE LIGHT" in t or "FOLLOW LIGHT" in t:
        return "follow_light"
    if "SKIP" in t:
        return "skip"
    if "BATTLE" in t or "DONE" in t:
        return "battle"
    return None


def back_header(frame: np.ndarray | None) -> bool:
    if frame is None:
        return False
    h, w = frame.shape[:2]
    x, y, bw, bh = BACK_BAND
    reg = frame[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)]
    if reg.size == 0:
        return False
    b, g, r = _channels(reg)
    gold = (np.abs(r - 231) <= 35) & (np.abs(g - 196) <= 35) & (np.abs(b - 120) <= 45)
    return float(gold.mean()) >= 0.02


def row_point(k: int) -> dict:
    return {"x": 0.09, "y": ROW_TOP + k * ROW_STEP}


def row_band(frame: np.ndarray, pt: dict) -> np.ndarray | None:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    half = max(4, int(round(h * ROW_STEP * 0.45)))
    cy = int(round(pt["y"] * h))
    return frame[max(0, cy - half):min(h, cy + half), int(ROW_X[0] * w):int(ROW_X[1] * w)]


def guard_alert(frame: np.ndarray | None, pt: dict) -> bool:
    band = row_band(frame, pt)
    if band is None or band.size == 0:
        return False
    b, g, r = _channels(band)
    yellow = (r >= 170) & (g >= 120) & (b <= 140) & (r - b >= 70)
    if float(yellow.mean()) < 0.008:
        return False
    cols = np.flatnonzero(yellow.any(axis=0))
    return len(cols) > 0 and cols[-1] - cols[0] >= band.shape[1] * 0.12


def _panel(img: np.ndarray) -> np.ndarray:
    b, g, r = _channels(img)
    return (b >= r + 18) & (b >= g + 10) & (r <= 80) & (b <= 120) & (b >= 25)


def find_cards(img: np.ndarray | None, client_h: int) -> list[Box]:
    if img is None or img.size == 0:
        return []
    mask = _panel(img)
    rows = mask.mean(axis=1) >= 0.30
    cards = []
    min_h = max(30, int(client_h * 0.06))
    d = np.diff(np.concatenate(([0], rows.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s < min_h:
            continue
        cols = np.flatnonzero(mask[s:e].mean(axis=0) >= 0.3)
        if len(cols) == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1])
        if x0 <= 2 and x1 >= img.shape[1] - 3:
            body = img[s:e, x0:x1 + 1][mask[s:e, x0:x1 + 1]]
            if len(body) == 0 or float(np.median(body.max(axis=1))) > 100:
                continue
        cards.append(Box(x0, int(s), x1 - x0 + 1, int(e - s)))
    return merge_cards(cards, client_h)


def merge_cards(cards: list[Box], client_h: int) -> list[Box]:
    gap = max(3, int(client_h * 0.014))
    out: list[Box] = []
    for c in sorted(cards, key=lambda b: b.y):
        if out and c.y - (out[-1].y + out[-1].h) <= gap and abs(c.cx - out[-1].cx) <= max(c.w, out[-1].w) * 0.1:
            p = out[-1]
            x0, y0 = min(p.x, c.x), p.y
            x1, y1 = max(p.x + p.w, c.x + c.w), c.y + c.h
            out[-1] = Box(x0, y0, x1 - x0, y1 - y0)
        else:
            out.append(c)
    return out


def find_options(img: np.ndarray | None, client_h: int) -> list[Box]:
    if img is None or img.size == 0:
        return []
    h, w = img.shape[:2]
    px = img.astype(np.int16)
    hi = np.maximum(np.maximum(px[..., 0], px[..., 1]), px[..., 2])
    lo = np.minimum(np.minimum(px[..., 0], px[..., 1]), px[..., 2])
    sat = (hi - lo >= 70) & (hi >= 140)
    dark = hi <= 60
    limit = max(16, int(w * 0.15))
    near = max(3, int(round(w * 0.0087)))
    far = max(near + 2, int(round(w * 0.013)))
    if w < limit + far + 2:
        return []
    hit = sat[:, :limit] & (dark[:, near:limit + near] | dark[:, far:limit + far])
    rows = hit.any(axis=1)
    min_h = max(14, int(client_h * 0.028))
    out = []
    d = np.diff(np.concatenate(([0], rows.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s < min_h:
            continue
        cols = np.flatnonzero(hit[s:e].sum(axis=0) >= (e - s) * 0.6)
        x = int(cols[0]) if len(cols) else 0
        out.append(Box(x, int(s), w - x, int(e - s)))
    return out


def parse_header(text: str) -> tuple[str, int, int] | None:
    up = str(text or "").upper()
    m = _EN.search(up)
    if m:
        fix = str.maketrans({"O": "0", "I": "1", "L": "1", "|": "1"})
        try:
            return "menu", int(m.group(1).translate(fix)), int(m.group(2).translate(fix))
        except ValueError:
            pass
    if "CHOOSE" in up or "FOE" in up:
        return "foe", 0, 0
    if "BACK" in up:
        return "back", 0, 0
    return None


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", str(text or "").upper().replace("'", "")).split())


def matches(keyword: str, text: str) -> bool:
    k, t = norm(keyword), norm(text)
    if not k or not t:
        return False
    if k in t:
        return True
    words = t.split()
    for kw in k.split():
        if len(kw) < 3:
            if kw not in words:
                return False
            continue
        if not any(difflib.SequenceMatcher(None, kw, tw).ratio() >= 0.8 for tw in words):
            return False
    return True


def lines_in(lines: list[Line], box: Box) -> list[Line]:
    return [ln for ln in lines if box.y <= ln.cy <= box.y + box.h and ln.cx >= box.x - 4]


def rank(text: str, prefer: list[str], avoid: list[str]) -> tuple[int, int]:
    avoided = any(matches(a, text) for a in avoid)
    pos = next((i for i, p in enumerate(prefer) if matches(p, text)), len(prefer))
    return (1 if avoided else 0, pos)


def letters_matched(want: str, label: str) -> int:
    a, b = want.replace(" ", ""), label.replace(" ", "")
    blocks = difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    return sum(m.size for m in blocks if m.size >= 2)


def skill_line(lines: list[Line], name: str, letters: int = 0) -> Line | None:
    want = norm(name)
    if not want:
        return None
    need = min(int(letters), len(want.replace(" ", ""))) if letters and letters > 0 else 0
    best, score = None, (0, 0.0)
    for ln in lines:
        label = norm(_COST.sub("", " " + ln.text))
        if not label or label.startswith("SKILLS"):
            continue
        s = 1.0 if want == label else difflib.SequenceMatcher(None, want, label).ratio()
        if want in label:
            s = max(s, 0.9)
        got = letters_matched(want, label)
        ok = got >= need if need else s >= 0.72
        if ok and (got, s) > score:
            best, score = ln, (got, s)
    return best
