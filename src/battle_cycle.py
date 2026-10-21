from __future__ import annotations

import time
from typing import Callable

import cv2
import mss
import numpy as np

from advanced_settings import DEFAULT_ADVANCED
from games import BAR, DODGE, LANE, LOCK, PanelPlayer, colour_game, find_meter, find_panel, find_ring, lock_box, ui_scale
from input_actions import release_inputs, sleep_interruptible, tap_key
from mouse_control import glide_click_at, gui_click_at, hover_at
from ocr import Line, Reader
from platform_keys import key_press, key_release
from roblox_geo import roblox_client_rect
from screens import (
    CHEST_BAND,
    DEFAULT_AREAS,
    GUARD_ROW,
    MENU_BAND,
    PATH_CONFIRM,
    USUAL_CARDS,
    Box,
    area_px,
    back_header,
    button_kind,
    crop,
    find_button,
    find_cards,
    find_options,
    guard_alert,
    in_battle,
    is_chest,
    lines_in,
    parse_header,
    path_banner,
    rank,
    row_band,
    row_point,
    skill_line,
    victory_banner,
    your_turn,
)
from timing import ATTACK, DODGE as BAR_DODGE, Player, find_bar, has_block, ui_ref
from win_focus import focus_roblox

SPACE = "space"
DEFAULT_POINTS = {
    "attack_btn": row_point(0),
    "skills_btn": row_point(1),
    "back_btn": {"x": 0.1344, "y": 0.7242},
    "guard_btn": row_point(GUARD_ROW),
    "replay_btn": {"x": 0.554, "y": 0.6825},
}

PLAYED = "played"
SEEN = "seen"
CLEAR = ""

HIT = "hit"
OPEN = "open"
PICK = "pick"
BAR_WAIT_SEC = 0.5

cv2.setNumThreads(1)


class Grabber:
    def __init__(self) -> None:
        self._sct = (getattr(mss, "MSS", None) or mss.mss)()

    def grab(self, x: int, y: int, w: int, h: int) -> np.ndarray | None:
        if w < 1 or h < 1:
            return None
        try:
            shot = self._sct.grab({"left": int(x), "top": int(y), "width": int(w), "height": int(h)})
        except Exception:
            return None
        return np.ascontiguousarray(np.asarray(shot, dtype=np.uint8)[:, :, :3])

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


def client():
    rect = roblox_client_rect()
    if rect is None or rect.w < 64 or rect.h < 64:
        return None
    return rect


def area_of(cfg: dict, key: str) -> dict:
    return cfg.get(key) or DEFAULT_AREAS[key]


def point_of(cfg: dict, key: str) -> dict:
    return (cfg.get("points") or {}).get(key) or DEFAULT_POINTS[key]


def _union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def search_box(cfg: dict, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = area_px(area_of(cfg, "mg_area"), w, h)
    pad = max(4, int(round(min(w, h) * 0.02)))
    side = pad + int(round((x1 - x0) * 0.02))
    up = pad + int(round((y1 - y0) * 0.3))
    return max(0, x0 - side), max(0, y0 - up), min(w, x1 + side), min(h, y1 + pad)


def menu_box(cfg: dict, w: int, h: int) -> tuple[int, int, int, int]:
    return _union(area_px(area_of(cfg, "event_area"), w, h), area_px(MENU_BAND, w, h))


def _panel_report(cfg: dict, frame: np.ndarray, rect) -> str | None:
    x0, y0, x1, y1 = search_box(cfg, rect.w, rect.h)
    img = frame[y0:y1, x0:x1]
    panel = find_panel(img, rect.w, rect.h)
    if panel is None:
        return None
    found = colour_game(panel.color, str(cfg.get("class") or ""))
    if found is None:
        return "There's a box up there but its top line isn't a minigame colour, so it keeps pressing Space."
    game, name = found
    if game in (BAR, DODGE):
        return "%s box is up (box %d px wide), it plays the bar as soon as the bar shows up." % (name, panel.right - panel.left)
    scale = panel.scale
    body = img[panel.top + max(12, int(round(60 * scale))):, panel.left:panel.right]
    if game == LOCK:
        seen = "the frame is gold right now" if lock_box(body, scale) is not None else "waiting for the frame to turn gold"
    elif game == LANE:
        seen = "found the ring" if find_ring(body, scale) is not None else "can't see the ring yet"
    else:
        seen = "found the power meter" if find_meter(body, scale) is not None else "can't see the power meter yet"
    return "%s minigame box is up, %s." % (name, seen)


def read_label(reader: Reader, img: np.ndarray, box: Box) -> str:
    part = img[box.y:box.y + box.h, box.x:box.x + box.w]
    f = max(1.0, min(5.0, 60.0 / max(1, box.h)))
    if f > 1.05:
        part = cv2.resize(part, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    return reader.text(part)


def _title(lines: list[Line]) -> str:
    return lines[0].text if lines else ""


def choose(cards: list[Box], lines: list[Line], cfg: dict) -> int | None:
    if not cards:
        return None
    prefer = list(cfg.get("path_prefer") or [])
    avoid = list(cfg.get("path_avoid") or [])
    scored = []
    for i, card in enumerate(cards):
        text = " ".join(ln.text for ln in lines_in(lines, card))
        scored.append((rank(text, prefer, avoid), i))
    return min(scored)[1]


def configured_skills(cfg: dict) -> list[dict]:
    slots = cfg.get("skill_slots")
    if isinstance(slots, list):
        return (slots + [{"enabled": False, "name": "", "en": 3, "cooldown": 0}] * 3)[:3]
    return [
        {"enabled": bool(cfg.get("skill_on")), "name": cfg.get("skill_name", ""),
         "en": cfg.get("skill_en", 3), "cooldown": 0},
        {"enabled": False, "name": "", "en": 3, "cooldown": 0},
        {"enabled": False, "name": "", "en": 3, "cooldown": 0},
    ]


def read_card_lines(reader: Reader, img: np.ndarray, cards: list[Box]) -> list[Line]:
    """OCR the detected cards instead of the full configurable path-screen area."""
    if img is None or img.size == 0 or not cards:
        return []
    pad_x = max(4, int(round(img.shape[1] * 0.01)))
    pad_y = max(4, int(round(img.shape[0] * 0.01)))
    x0 = max(0, min(card.x for card in cards) - pad_x)
    y0 = max(0, min(card.y for card in cards) - pad_y)
    x1 = min(img.shape[1], max(card.x + card.w for card in cards) + pad_x)
    y1 = min(img.shape[0], max(card.y + card.h for card in cards) + pad_y)
    lines = reader.read(img[y0:y1, x0:x1])
    # Keep coordinates relative to the original path-area image so lines_in()
    # and choose() continue to use the same card boxes as before.
    return [Line(line.text, line.x + x0, line.y + y0, line.w, line.h) for line in lines]


def describe(cfg: dict, reader: Reader, grabber: Grabber) -> str:
    rect = client()
    if rect is None:
        return "Roblox not found, open it first."
    frame = grabber.grab(rect.x, rect.y, rect.w, rect.h)
    if frame is None:
        return "Couldn't capture Roblox."
    sx0, sy0, sx1, sy1 = search_box(cfg, rect.w, rect.h)
    panel = _panel_report(cfg, frame, rect)
    if panel is not None:
        return panel
    ref = ui_ref(rect.w, rect.h)
    img, _x, _y = crop(frame, area_of(cfg, "mg_area"))
    bar = find_bar(img, ref) or find_bar(frame[sy0:sy1, sx0:sx1], ref)
    if bar is not None:
        return "A %s bar is on screen but there's no minigame box above it to read, so it leaves it alone and keeps pressing Space." % ("dodge" if bar.kind == BAR_DODGE else "attack")
    if in_battle(frame):
        manual = bool(cfg.get("skill_manual"))
        slots = configured_skills(cfg)
        skill = any(bool(slot.get("enabled")) and (bool(str(slot.get("name") or "").strip()) or (i == 0 and manual))
                    for i, slot in enumerate(slots))
        gap = float(cfg["advanced"]["skill_space_sec" if skill else "battle_space_sec"])
        if back_header(frame):
            return "Battle, CHOOSE A FOE or the skill list is up. Space every %.2f s, and it clicks BACK if it sticks for %.0f s." % (gap, float(cfg["advanced"]["stuck_fix_sec"]))
        if your_turn(frame) and guard_alert(frame, point_of(cfg, "guard_btn")):
            return "Battle, your turn, GUARD ! is lit up, it guards this turn instead of attacking."
        if not your_turn(frame):
            return "Battle, enemy's turn. It stays quiet so it can't ruin the dodge (one Space if nothing happens for %.1f s), and it plays a minigame the moment one shows up." % float(cfg["advanced"]["enemy_space_sec"])
        if not skill:
            return "Battle, your turn. Mouse on ATTACK, Space every %.2f s." % gap
        img, _x, _y = crop(frame, area_of(cfg, "en_area"))
        text = reader.text(img)
        head = parse_header(text)
        if head is None or head[0] != "menu":
            return "Battle, your turn, but the EN area read '%s' instead of EN 2/10, so it just attacks. Fix it with Select on EN area." % (text or "nothing")
        ready = [(i, slot) for i, slot in enumerate(slots)
                 if slot.get("enabled") and (str(slot.get("name") or "").strip() or (i == 0 and manual))
                 and head[1] >= int(slot.get("en") or 0)]
        if ready:
            i, slot = ready[0]
            label = "Manual Skill 1" if i == 0 and manual else str(slot.get("name") or "Skill %d" % (i + 1))
            return "Battle, your turn. EN %d/%d, it would use Skill %d (%s)." % (head[1], head[2], i + 1, label)
        next_en = min((int(slot.get("en") or 0) for i, slot in enumerate(slots)
                       if slot.get("enabled") and (str(slot.get("name") or "").strip() or (i == 0 and manual))), default=0)
        return "Battle, your turn. EN %d/%d, it attacks until an enabled skill's EN minimum (%d) is met." % (head[1], head[2], next_en)
    cx0, cy0, cx1, cy1 = area_px(area_of(cfg, "continue_area"), rect.w, rect.h)
    continue_img = frame[cy0:cy1, cx0:cx1]
    continue_btn = find_button(continue_img, rect.w, rect.h)
    if continue_btn is not None:
        label = read_label(reader, continue_img, continue_btn.box)
        if button_kind(label) == "continue" and continue_btn.active:
            return "CONTINUE button in Continue area, it clicks it."
    for area_key, expected, title in (("skip_area", "skip", "SKIP"), ("follow_light_area", "follow_light", "FOLLOW THE LIGHT")):
        ax0, ay0, ax1, ay1 = area_px(area_of(cfg, area_key), rect.w, rect.h)
        area_img = frame[ay0:ay1, ax0:ax1]
        area_lines = reader.read(area_img)
        area_label = " ".join(ln.text for ln in area_lines)
        if button_kind(area_label) == expected:
            return "%s text in its selected area, it clicks the text." % title
    x0, y0, x1, y1 = menu_box(cfg, rect.w, rect.h)
    img = frame[y0:y1, x0:x1]
    btn = find_button(img, rect.w, rect.h)
    if btn is not None:
        label = read_label(reader, img, btn.box)
        kind = button_kind(label)
        if kind == "continue":
            return "CONTINUE button is outside the Continue area, so it is ignored. Adjust Continue button with Select."
        if kind in ("skip", "follow_light"):
            return "%s button is outside its selected area, so it is ignored." % ("SKIP" if kind == "skip" else "FOLLOW THE LIGHT")
        if kind == "buy":
            return "Shop, it leaves the BUY buttons alone and clicks DONE - TO BATTLE!."
        if kind == "retry":
            return "Run finished, PLAY AGAIN is %s." % ("ready, it clicks it" if btn.active else "not lit up yet")
        if kind == "continue":
            return "CONTINUE button, it presses Space (and clicks it if Space doesn't work)."
        if kind in ("skip", "follow_light") and btn.active:
            return "%s button, it clicks it." % ("SKIP" if kind == "skip" else "FOLLOW THE LIGHT")
        if btn.active and kind == "battle":
            return "%s is ready, it clicks it." % label
        if not btn.active:
            ev, _ex, _ey = crop(frame, area_of(cfg, "event_area"))
            opts = find_options(ev, rect.h)
            lines = reader.read(ev)
            names = [_title(lines_in(lines, o)) or "?" for o in opts]
            return "Event screen, TO BATTLE! is greyed out. Choices: %s" % (", ".join(names) or "none found")
        return "Gold button reading '%s', it presses Space on it." % (label or "nothing")
    if path_banner(frame):
        img, _x, _y = crop(frame, area_of(cfg, "path_area"))
        cards = find_cards(img, rect.h)
        lines = read_card_lines(reader, img, cards)
        if not cards:
            return "Choose Your Path screen, but no path cards were found in the path area, it clicks where the cards usually are."
        names = [_title(lines_in(lines, c)) or "?" for c in cards]
        pick = choose(cards, lines, cfg)
        return "Choose Your Path: %s. It would pick %s." % (", ".join(names), names[pick] if pick is not None else "nothing")
    if victory_banner(frame):
        return "VICTORY, it presses Space until the Choose Your Path screen shows up."
    img, _x, _y = crop(frame, CHEST_BAND)
    if is_chest(reader.text(img)):
        return "Chest, it presses Space to open it."
    return "Not in a fight and nothing to click, it presses Space every %.2f s." % float(cfg["advanced"]["wait_space_sec"])


class Runner:
    def __init__(
        self,
        cfg: dict,
        reader: Reader,
        player: Player,
        aborted: Callable[[], bool],
        status: Callable[[str], None],
        note: Callable[[str, str], None],
    ) -> None:
        self.cfg = cfg
        self.reader = reader
        self.player = player
        self.panels = PanelPlayer()
        self.aborted = aborted
        self.status = status
        self.note = note
        self.grabber = Grabber()
        self._last_screen = ""
        self._event_clicks: dict[int, int] = {}
        self._chest_tries = 0
        self._continue_tries = 0
        self._continue_retry_at = 0.0
        self._continue_path_watch_until = 0.0
        self._retry_at = 0.0
        self._retry_tries = 0
        self._next_chest = 0.0
        self._label = ("", 0, 0, 0.0)
        self._next_space = 0.0
        self._hover = ""
        self._mode = HIT
        self._turn = False
        self._menu_at = 0.0
        self._nudges = 0
        self._en: tuple[int, int] | None = None
        self._en_read = False
        self._skill_off: set[int] = set()
        self._skill_block = [0.0, 0.0, 0.0]
        self._skill_candidate: int | None = None
        self._turn_actions = 0
        self._skill_ready_after = [0, 0, 0]
        self._skill_at = 0.0
        self._skill_tries = 0
        self._mg_block = 0.0
        self._mg_game = ""
        self._bar_at = 0.0
        self._seen: tuple | None = None
        self._win = 0.0
        self._scrolls = 0
        self._guard_tries = 0
        self._guard_read = False
        self._guard_seen = False
        self._path_at = 0.0
        self._path_tries = 0
        self._path_cards = 0
        self._path_pick = 0
        self._fight_at = time.perf_counter()
        self._replay_at = 0.0
        self._moved = time.perf_counter()

    @property
    def adv(self) -> dict:
        return self.cfg["advanced"]

    def _num(self, key: str, low: float) -> float:
        try:
            return max(low, float(self.adv[key]))
        except (KeyError, TypeError, ValueError):
            return max(low, float(DEFAULT_ADVANCED.get(key, low)))

    def close(self) -> None:
        self.grabber.close()
        release_inputs()

    def _sleep(self, sec: float) -> bool:
        return sleep_interruptible(max(0.0, float(sec)), self.aborted)

    def _progress(self) -> None:
        self._moved = time.perf_counter()

    def _screen(self, name: str) -> None:
        if name != self._last_screen:
            self._last_screen = name
            self.note("screen", name)
        if name != "event":
            self._event_clicks.clear()
        if name != "chest":
            self._chest_tries = 0
        if name != "victory":
            self._win = 0.0
        if name != "continue":
            self._continue_tries = 0
            self._continue_retry_at = 0.0
        if name != "end":
            self._retry_at = 0.0
            self._retry_tries = 0
        if name != "path":
            self._path_at = 0.0
            self._path_tries = 0
            self._path_cards = 0

    def _abs(self, rect, pt: dict) -> tuple[int, int]:
        return rect.x + int(round(pt["x"] * rect.w)), rect.y + int(round(pt["y"] * rect.h))

    def _click(self, rect, x: int, y: int) -> None:
        gui_click_at(rect.x + int(x), rect.y + int(y))
        self._progress()
        self.note("action", "")

    def _click_point(self, rect, key: str) -> None:
        gui_click_at(*self._abs(rect, point_of(self.cfg, key)))
        self._progress()
        self.note("action", "")

    def _press(self, hold: float) -> None:
        try:
            key_press(SPACE)
            time.sleep(max(0.0, hold))
        finally:
            key_release(SPACE)

    def _focus(self) -> None:
        if self.cfg.get("auto_focus_roblox"):
            focus_roblox()

    def _tap(self, action: bool = True, key: str = SPACE) -> None:
        self._focus()
        tap_key(key, self._num("key_hold_sec", 0.01), self.aborted)
        self._progress()
        if action:
            self.note("action", "")

    def _frame(self, rect) -> np.ndarray | None:
        frame = self.grabber.grab(rect.x, rect.y, rect.w, rect.h)
        if frame is None or frame.shape[0] != rect.h or frame.shape[1] != rect.w:
            return None
        return frame

    def loop(self) -> None:
        while not self.aborted():
            rect = client()
            if rect is None:
                self.status("Roblox not found, open it.")
                if self._sleep(1.0):
                    return
                continue
            frame = self._frame(rect)
            if frame is None:
                if self._sleep(0.2):
                    return
                continue
            battle = in_battle(frame)
            state = self._minigame(rect, frame, battle)
            now = time.perf_counter()
            if battle or state != CLEAR:
                self._fight_at = now
            if state == PLAYED:
                continue
            busy = state == SEEN
            if battle:
                self._fight(rect, frame, now, busy)
            else:
                self._outside(rect, frame, now, busy)
            if self._sleep(self._num("poll_sec", 0.005)):
                return

    def _minigame(self, rect, frame: np.ndarray, battle: bool) -> str:
        sx0, sy0, sx1, sy1 = search_box(self.cfg, rect.w, rect.h)
        area = frame[sy0:sy1, sx0:sx1]
        dodge = find_bar(area, ui_ref(rect.w, rect.h))
        if dodge is not None and dodge.kind == BAR_DODGE and has_block(area, dodge):
            self._bar_at = 0.0
            self._seen = None
            return self._play_bar(rect, dodge, sx0, sy0, sx1)
        panel = find_panel(area, rect.w, rect.h)
        found = colour_game(panel.color, str(self.cfg.get("class") or "")) if panel is not None else None
        if found is None:
            self._bar_at = 0.0
            self._seen = None
            return CLEAR
        now = time.perf_counter()
        game, name = found
        if now < self._mg_block and game == self._mg_game:
            return CLEAR
        if game in (BAR, DODGE):
            self._seen = None
            bar, bx0, by0, bx1 = self._bar_in_box(rect, frame, panel, sx0, sy0)
            if bar is not None:
                self._bar_at = 0.0
                return self._play_bar(rect, bar, bx0, by0, bx1)
            if not self._bar_at:
                self._bar_at = now
            if now - self._bar_at > BAR_WAIT_SEC:
                return CLEAR
            self.status("%s box is up, waiting for the bar." % name)
            return SEEN
        self._bar_at = 0.0
        if not battle:
            self._seen = None
            return CLEAR
        key = (game, panel.left // 8, panel.top // 8, (panel.right - panel.left) // 8)
        if self._seen != key:
            self._seen = key
            return SEEN
        self._seen = None
        return self._play_panel(rect, panel, game, name)

    def _bar_in_box(self, rect, frame: np.ndarray, panel, sx0: int, sy0: int):
        bw = panel.right - panel.left
        x0 = sx0 + panel.left
        x1 = sx0 + panel.right
        y0 = sy0 + panel.top + int(round(bw * 0.09))
        y1 = min(rect.h, sy0 + panel.top + int(round(bw * 0.45)))
        if y1 - y0 < 12 or x1 - x0 < 60:
            return None, 0, 0, 0
        bar = find_bar(frame[y0:y1, x0:x1], bw / 0.3938)
        if bar is None:
            return None, 0, 0, 0
        return bar, x0, y0, x1

    def _play_bar(self, rect, bar, x0: int, y0: int, x1: int) -> str:
        dodge = bar.kind == BAR_DODGE
        if dodge and not self.cfg["play_dodge"]:
            self.status("Dodge minigame is up, you play it (Play dodge minigame is off).")
            return SEEN
        if not dodge and not self.cfg["play_attack"]:
            self.status("Attack minigame is up, you play it (Play attack minigame is off).")
            return SEEN
        early = bool(self.cfg.get("early_mode")) and not dodge
        self._focus()
        self._screen("minigame")
        if dodge:
            self.status("Dodging...")
        else:
            self.status("Attacking (Blade Focus, Early Mode)..." if early else "Attacking (Blade Focus)...")
        sx, sy = rect.x + x0, rect.y + y0 + bar.above
        sh = bar.mid - bar.above + 1
        sw = x1 - x0
        want = {ATTACK: bool(self.cfg["play_attack"]), BAR_DODGE: bool(self.cfg["play_dodge"])}

        def grab():
            strip = self.grabber.grab(sx, sy, sw, sh)
            return strip, time.perf_counter()

        result = self.player.play(grab, bar, self._press, self.aborted, want, early)
        self.note(result["kind"], str(result["presses"]))
        self._after_play(self.player.timed_out, DODGE if dodge else BAR)
        return PLAYED

    def _play_panel(self, rect, panel, game: str, name: str) -> str:
        if not self.cfg["play_attack"]:
            self.status("%s is up, you play it (Play attack minigame is off)." % name)
            return SEEN
        x0, y0, x1, y1 = search_box(self.cfg, rect.w, rect.h)
        early = bool(self.cfg.get("early_mode"))
        scale = panel.scale
        self.panels.configure(self.adv)
        play = None if early else self.panels.make(game, scale, panel.who)
        self._focus()
        self._screen("minigame")
        self.status("Attacking (%s, Early Mode)..." % name if early else "Attacking (%s)..." % name)
        margin = max(8, int(round(20 * scale)))
        gx = rect.x + x0 + panel.left
        gy = rect.y + max(0, y0 + panel.top - margin)
        gw = panel.right - panel.left
        gh = min(rect.h - (gy - rect.y), int(round(363 * scale)))

        def grab():
            img = self.grabber.grab(gx, gy, gw, gh)
            return img, time.perf_counter()

        presses = self.panels.play(play, grab, scale, self._press, self.aborted, margin * 2 + 4)
        self.note("attack", str(presses))
        self._after_play(self.panels.timed_out, game)
        return PLAYED

    def _after_play(self, timed_out: bool, game: str) -> None:
        now = time.perf_counter()
        if timed_out and game == DODGE:
            self.status("That dodge never ended, back to Space.")
        elif timed_out:
            block = self._num("minigame_block_sec", 0.0)
            self._mg_block = now + block
            self._mg_game = game
            self.status("That minigame never ended, ignoring it for %.1f s." % block)
        self._next_space = now + self._num("battle_space_sec", 0.02)
        self._progress()

    def _fight(self, rect, frame: np.ndarray, now: float, busy: bool) -> None:
        self._screen("battle")
        if busy:
            self._menu_at = 0.0
            return
        turn = your_turn(frame)
        back = back_header(frame)
        if turn and not self._turn:
            self._en = None
            self._en_read = False
            self._guard_read = False
            self._guard_seen = False
            self._guard_tries = 0
        self._turn = turn
        if not (turn or back):
            self._menu_at = 0.0
            self._nudges = 0
        elif not self._menu_at:
            self._menu_at = now
        if self._mode == PICK:
            self._pick_skill(rect, now, back)
            return
        if self._mode == OPEN:
            self._open_skills(rect, now)
            return
        if turn and not back and self._guard(rect, frame, now):
            return
        if self._menu_at and now - self._menu_at > self._num("stuck_fix_sec", 1.0):
            self._unstick(rect, now, back)
            return
        if turn and self._skill_due(rect, frame):
            return
        if not (turn or back) and now - self._moved < self._num("enemy_space_sec", 0.5):
            self.status("Enemy's turn, waiting for the dodge.")
            return
        if now < self._next_space:
            return
        self._face(rect, "attack_btn")
        if turn and self._en:
            self.status("Your turn (EN %d/%d), ATTACK." % self._en)
        elif turn:
            self.status("Your turn, ATTACK.")
        else:
            self.status("In a fight, Space.")
        self._tap(False)
        if turn:
            self._record_turn_action()
        self._next_space = time.perf_counter() + self._pace(turn)

    def _guard_up(self, frame: np.ndarray) -> bool:
        pt = point_of(self.cfg, "guard_btn")
        if guard_alert(frame, pt):
            return True
        if not self._guard_read:
            self._guard_read = True
            band = row_band(frame, pt)
            if band is not None and band.size:
                text = self.reader.text(band)
                self._guard_seen = "GUARD" in text.upper().replace(" ", "") and "!" in text
        return self._guard_seen

    def _guard(self, rect, frame: np.ndarray, now: float) -> bool:
        if self._guard_tries >= 5 or not self._guard_up(frame):
            return False
        if now < self._next_space:
            return True
        self._menu_at = now
        self._guard_tries += 1
        if self._guard_tries == 4:
            self.status("GUARD ! is up and Space didn't take, clicking GUARD.")
            self._click_point(rect, "guard_btn")
            self._hover = "guard_btn"
            self._record_turn_action()
        else:
            self.status("GUARD ! is up, guarding.")
            self._face(rect, "guard_btn")
            self._tap()
            self._record_turn_action()
        if self._guard_tries == 1:
            self.note("guard", "")
        self._next_space = time.perf_counter() + max(0.35, self._pace())
        return True

    def _pace(self, turn: bool = True) -> float:
        if turn and self._skill_set():
            return self._num("skill_space_sec", 0.05)
        return self._num("battle_space_sec", 0.02)

    def _face(self, rect, key: str) -> None:
        if self._hover == key:
            return
        self._hover = key
        hover_at(*self._abs(rect, point_of(self.cfg, key)))

    def _skill_set(self) -> bool:
        manual = bool(self.cfg.get("skill_manual"))
        return any(bool(slot.get("enabled"))
                   and (bool(str(slot.get("name") or "").strip()) or (i == 0 and manual))
                   and i not in self._skill_off
                   for i, slot in enumerate(configured_skills(self.cfg)))

    def _record_turn_action(self, skill_index: int | None = None) -> None:
        self._turn_actions += 1
        if skill_index is not None:
            slot = configured_skills(self.cfg)[skill_index]
            cooldown = max(0, int(slot.get("cooldown") or 0))
            self._skill_ready_after[skill_index] = self._turn_actions + cooldown

    def _reset_encounter_skill_cooldowns(self) -> None:
        self._turn_actions = 0
        self._skill_ready_after = [0, 0, 0]
        self._skill_block = [0.0, 0.0, 0.0]

    def _skill_due(self, rect, frame: np.ndarray) -> bool:
        if self._en_read or not self._skill_set():
            return False
        self._en_read = True
        self._en = self._read_en(frame)
        if self._en is None:
            self.status("Your turn, couldn't read your EN, attacking.")
            return False
        now = time.perf_counter()
        manual = bool(self.cfg.get("skill_manual"))
        for i, slot in enumerate(configured_skills(self.cfg)):
            name = str(slot.get("name") or "").strip()
            if not slot.get("enabled") or i in self._skill_off:
                continue
            if not name and not (i == 0 and manual):
                continue
            if self._turn_actions < self._skill_ready_after[i] or now < self._skill_block[i]:
                continue
            if self._en[0] < max(0, int(slot.get("en") or 0)):
                continue
            self._skill_candidate = i
            self._mode = OPEN
            self._skill_at = 0.0
            self._skill_tries = 0
            return True
        return False

    def _read_en(self, frame: np.ndarray) -> tuple[int, int] | None:
        img, _x, _y = crop(frame, area_of(self.cfg, "en_area"))
        head = parse_header(self.reader.text(img))
        if head is not None and head[0] == "menu":
            return head[1], head[2]
        return None

    def _open_skills(self, rect, now: float) -> None:
        if self._skill_at and now - self._skill_at < self._num("skill_wait_sec", 0.05):
            return
        if self._skill_tries >= 3:
            self._drop_skill(rect, "SKILLS wouldn't open, back to attacking.", False)
            return
        self._skill_tries += 1
        self.status("EN %d/%d, opening SKILLS." % self._en if self._en else "Opening SKILLS.")
        self._face(rect, "skills_btn")
        self._tap()
        self._skill_at = time.perf_counter()
        self._scrolls = 0
        self._mode = PICK

    def _pick_skill(self, rect, now: float, back: bool) -> None:
        wait = self._num("skill_wait_sec", 0.05)
        if now - self._skill_at < wait:
            return
        if bool(self.cfg.get("skill_manual")) and self._skill_candidate == 0:
            presses = min(50, max(0, int(self.cfg.get("skill_manual_presses") or 0)))
            self.status("Manual select: pressing S %d times, then Space." % presses)
            for _ in range(presses):
                self._tap(False, "s")
                if self.aborted():
                    return
            self._tap()
            self._record_turn_action(0)
            self.note("skill", "Manual select (%d S presses)" % presses)
            self._skill_block[0] = time.perf_counter() + self._num("skill_retry_sec", 0.0)
            self._back_to_attack(rect)
            return
        idx = self._skill_candidate
        if idx is None:
            self._drop_skill(rect, "No active skill was selected, back to attacking.", False)
            return
        slot = configured_skills(self.cfg)[idx]
        name = str(slot.get("name") or "").strip()
        if not name:
            self._skill_off.add(idx)
            self._drop_skill(rect, "Skill %d has no name, skipping it." % (idx + 1), False)
            return
        frame = self._frame(rect)
        if frame is None:
            if now - self._skill_at > wait * 4:
                self._drop_skill(rect, "Lost the screen while the skill list was open.", False)
            return
        back = back_header(frame)
        img, ax, ay = crop(frame, area_of(self.cfg, "skills_area"))
        line = skill_line(self.reader.read(img), name, int(self.cfg.get("skill_letters") or 0))
        if line is None:
            if not back:
                self._mode = OPEN
                self._skill_at = 0.0
                return
            most = int(self._num("skill_scroll_max", 0.0))
            if self._scrolls < most:
                self._scrolls += 1
                self.status("%s isn't showing yet, S to scroll the list down (%d/%d)." % (name, self._scrolls, most))
                self._tap(False, "s")
                self._skill_at = time.perf_counter()
                return
            self.status("%s isn't in the skill list, skipping Skill %d for this run." % (name, idx + 1))
            self._skill_off.add(idx)
            self._drop_skill(rect, "", back)
            return
        self.status("Using %s." % name)
        hover_at(rect.x + ax + line.x + min(line.w // 3, 60), rect.y + ay + line.cy)
        time.sleep(0.08)
        self._tap()
        self._record_turn_action(idx)
        self.note("skill", name)
        self._skill_block[idx] = time.perf_counter() + self._num("skill_retry_sec", 0.0)
        self._back_to_attack(rect)

    def _drop_skill(self, rect, msg: str, back: bool) -> None:
        if msg:
            self.status(msg)
        if back:
            self._click_point(rect, "back_btn")
        if self._skill_candidate is not None:
            self._skill_block[self._skill_candidate] = time.perf_counter() + self._num("skill_retry_sec", 0.0)
        self._back_to_attack(rect)

    def _back_to_attack(self, rect) -> None:
        self._mode = HIT
        self._scrolls = 0
        self._skill_at = 0.0
        self._skill_tries = 0
        self._skill_candidate = None
        self._en = None
        self._en_read = True
        self._menu_at = 0.0
        self._nudges = 0
        self._hover = ""
        self._face(rect, "attack_btn")
        self._next_space = time.perf_counter() + self._pace()

    def _unstick(self, rect, now: float, back: bool) -> None:
        self._menu_at = now
        self._nudges += 1
        self._hover = ""
        if self._nudges == 1:
            self.status("The menu hasn't moved, putting the mouse back on ATTACK.")
            self._face(rect, "attack_btn")
            self._tap()
            return
        if self._nudges == 2:
            self.status("The menu still hasn't moved, clicking ATTACK.")
            self._click_point(rect, "attack_btn")
            self._hover = "attack_btn"
            return
        self.status("The menu is stuck, clicking BACK.")
        self._click_point(rect, "back_btn")
        self._nudges = 0
        self._next_space = time.perf_counter() + self._pace()

    def _outside(self, rect, frame: np.ndarray, now: float, busy: bool) -> None:
        self._hover = ""
        self._mode = HIT
        self._menu_at = 0.0
        self._en_read = False
        handled = self._menus(rect, frame, now)
        if (not handled or self._last_screen == "end") and not busy and self._replay(rect, now):
            return
        if handled or busy:
            return
        gap = self._num("wait_space_sec", 0.1)
        if now - self._moved < gap:
            return
        self.status("Nothing to click, Space.")
        self._tap(False)

    def _menus(self, rect, frame: np.ndarray, now: float) -> bool:
        # After Continue, watch for the path screen on each normal poll. The
        # regular path detection below remains active even after this window.
        if now < self._continue_path_watch_until and path_banner(frame):
            self._continue_path_watch_until = 0.0
            self._screen("path")
            self._path(rect, fast=True)
            return True

        # Mini-boss Continue gets its own adjustable detection region.
        cx0, cy0, cx1, cy1 = area_px(area_of(self.cfg, "continue_area"), rect.w, rect.h)
        continue_img = frame[cy0:cy1, cx0:cx1]
        continue_btn = find_button(continue_img, rect.w, rect.h)
        if continue_btn is not None:
            label = "" if self._last_screen == "continue" else " ".join(ln.text for ln in self.reader.read(continue_img))
            # OCR can miss the label on later frames after the first click.
            # Once confirmed, keep retrying while the active button remains
            # inside the dedicated area.
            upper_label = label.upper()
            confirmed = "CONTIN" in upper_label or self._last_screen == "continue"
            if confirmed and continue_btn.active:
                self._screen("continue")
                now = time.perf_counter()
                if self._continue_retry_at and now - self._continue_retry_at < 4.0:
                    self.status("CONTINUE is still on screen, watching for the path screen.")
                    self._sleep(min(0.10, 4.0 - (now - self._continue_retry_at)))
                    return True
                self._continue_tries += 1
                cx, cy = rect.x + cx0 + continue_btn.box.cx, rect.y + cy0 + continue_btn.box.cy
                if self._continue_tries == 1:
                    self.status("CONTINUE detected, clicking it.")
                    gui_click_at(cx, cy)
                else:
                    self.status("CONTINUE is still there, gliding right and clicking again.")
                    glide_px = min(6 * (self._continue_tries - 1), max(6, continue_btn.box.w // 2 - 3))
                    glide_click_at(cx, cy, glide_px)
                self._progress()
                self.note("action", "")
                self._continue_retry_at = time.perf_counter()
                self._continue_path_watch_until = self._continue_retry_at + 12.0
                self._sleep(0.10)
                return True
        for area_key, expected, title in (("skip_area", "skip", "SKIP"), ("follow_light_area", "follow_light", "FOLLOW THE LIGHT")):
            ax0, ay0, ax1, ay1 = area_px(area_of(self.cfg, area_key), rect.w, rect.h)
            area_img = frame[ay0:ay1, ax0:ax1]
            area_lines = self.reader.read(area_img)
            label = " ".join(ln.text for ln in area_lines)
            if button_kind(label) != expected:
                continue
            if expected == "skip":
                text_line = next((ln for ln in area_lines if "SKIP" in ln.text.upper()), None)
            else:
                text_line = next((ln for ln in area_lines if "FOLLOW" in ln.text.upper()), None)
                if text_line is None:
                    text_line = next((ln for ln in area_lines if "LIGHT" in ln.text.upper()), None)
            if text_line is None:
                continue
            self._screen("event")
            self.status("%s button, clicking it." % title)
            self._click(rect, ax0 + text_line.cx, ay0 + text_line.cy)
            self._sleep(self._num("after_pick_sec", 0.0))
            return True
        x0, y0, x1, y1 = menu_box(self.cfg, rect.w, rect.h)
        img = frame[y0:y1, x0:x1]
        btn = find_button(img, rect.w, rect.h)
        if btn is not None:
            # Continue is handled only from continue_area so its calibration
            # controls where this prompt can trigger.
            generic_kind = button_kind(self._read_button(img, btn.box))
            if generic_kind not in ("continue", "skip", "follow_light") and self._button(rect, frame, img, x0, y0, btn):
                return True
        if path_banner(frame):
            self._screen("path")
            self._path(rect)
            return True
        if now >= self._next_chest:
            self._next_chest = now + self._num("chest_check_sec", 0.3)
            if self._chest(rect, frame):
                return True
        if self._victory(rect, frame, now):
            return True
        if self._last_screen in ("battle", "minigame"):
            self.status("Fight over, waiting for the next screen...")
        else:
            self.status("Waiting...")
        return False

    def _victory(self, rect, frame: np.ndarray, now: float) -> bool:
        if victory_banner(frame):
            if not self._win:
                self._win = now
            self._screen("victory")
        elif not self._win or now - self._win > self._num("victory_max_sec", 1.0):
            return False
        if now - self._moved >= self._num("victory_gap_sec", 0.1):
            self.status("VICTORY, Space until Choose Your Path.")
            self._tap()
        return True

    def _replay(self, rect, now: float) -> bool:
        if not self.cfg.get("replay_check"):
            return False
        every = max(3.0, float(self.cfg.get("replay_sec") or 10.0))
        if now - self._fight_at < every or now - self._replay_at < every:
            return False
        self._replay_at = now
        x, y = self._abs(rect, point_of(self.cfg, "replay_btn"))
        self.status("No fight for %.0f s, clicking the replay spot." % every)
        self._focus()
        self._hover = ""
        hover_at(x, y)
        time.sleep(0.15)
        gui_click_at(x, y)
        self._progress()
        self.note("action", "")
        return True

    def _read_button(self, img: np.ndarray, box: Box) -> str:
        now = time.perf_counter()
        text, bx, by, at = self._label
        if text and abs(bx - box.x) <= 4 and abs(by - box.y) <= 4 and now - at < self._num("label_read_sec", 0.0):
            return text
        text = read_label(self.reader, img, box)
        self._label = (text, box.x, box.y, now)
        return text

    def _button(self, rect, frame: np.ndarray, img: np.ndarray, x0: int, y0: int, btn) -> bool:
        b = btn.box
        label = self._read_button(img, b)
        kind = button_kind(label)
        cx, cy = x0 + b.cx, y0 + b.cy
        if kind == "buy":
            return False
        if kind == "retry":
            self._screen("end")
            now = time.perf_counter()
            if not btn.active:
                self._retry_at = 0.0
                self.status("Run finished, waiting for PLAY AGAIN to light up.")
                self._sleep(0.3)
                return True
            if not self._retry_at:
                self._retry_at = now
            if now - self._retry_at < 0.6:
                self.status("Run finished, letting PLAY AGAIN finish fading in.")
                return True
            self._retry_tries += 1
            if self._retry_tries == 1:
                self.note("run", "")
            x, y = rect.x + cx, rect.y + cy
            self._hover = ""
            hover_at(x, y)
            time.sleep(0.15)
            if self._retry_tries % 3 == 0:
                self.status("PLAY AGAIN didn't take the click, trying Space.")
                self._tap()
            else:
                self.status("Run finished, clicking PLAY AGAIN.")
                gui_click_at(x, y)
                self._progress()
                self.note("action", "")
            self._sleep(1.0)
            return True
        if kind == "continue" and btn.active:
            self._screen("continue")
            self._continue_tries += 1
            if self._continue_tries <= 2:
                self.status("CONTINUE, Space.")
                self._tap()
            else:
                self.status("CONTINUE didn't take Space, clicking it.")
                self._click(rect, cx, cy)
                self._continue_tries = 0
            self._sleep(self._num("after_pick_sec", 0.0))
            return True
        if kind in ("skip", "follow_light") and btn.active:
            self._screen("event")
            title = "SKIP" if kind == "skip" else "FOLLOW THE LIGHT"
            self.status("%s button, clicking it." % title)
            self._click(rect, cx, cy)
            self._sleep(self._num("after_pick_sec", 0.0))
            return True
        if kind == "battle" and btn.active:
            self._screen("event")
            self.status("%s" % (label.strip() or "TO BATTLE!"))
            self._click(rect, cx, cy)
            self.note("event", "")
            self._sleep(self._num("after_pick_sec", 0.0))
            return True
        if not btn.active and kind in ("battle", None):
            ev, ex, ey = crop(frame, area_of(self.cfg, "event_area"))
            opts = find_options(ev, rect.h)
            if not opts:
                return False
            self._screen("event")
            self._event(rect, ev, ex, ey, opts)
            return True
        if btn.active and kind is None:
            self._screen("continue")
            self.status("Gold button (%s), Space." % (label.strip() or "can't read it"))
            self._tap()
            self._sleep(self._num("after_pick_sec", 0.0))
            return True
        return False

    def _event(self, rect, img: np.ndarray, ex: int, ey: int, opts: list[Box]) -> None:
        lines = self.reader.read(img)
        names = [_title(lines_in(lines, o)) for o in opts]
        prefer = list(self.cfg.get("event_prefer") or [])
        order = sorted(range(len(opts)), key=lambda i: (rank(names[i], prefer, [])[1], i))
        pick = None
        for i in order:
            if self._event_clicks.get(i, 0) < 2:
                pick = i
                break
        if pick is None:
            self._event_clicks.clear()
            pick = order[0]
        self._event_clicks[pick] = self._event_clicks.get(pick, 0) + 1
        opt = opts[pick]
        self.status("Event: picking %s." % (names[pick] or "choice %d" % (pick + 1)))
        self._click(rect, ex + opt.x + max(20, opt.w // 4), ey + opt.cy)
        self._sleep(self._num("option_wait_sec", 0.0))

    def _chest(self, rect, frame: np.ndarray) -> bool:
        img, cx0, cy0 = crop(frame, CHEST_BAND)
        label = next((ln for ln in self.reader.read(img) if is_chest(ln.text)), None)
        if label is None:
            return False
        if self._last_screen != "chest":
            self.note("chest", "")
        self._screen("chest")
        self._chest_tries += 1
        if self._chest_tries <= 2:
            self.status("Chest, opening it with Space.")
            self._tap()
        else:
            self.status("The chest didn't open with Space, clicking it.")
            self._click(rect, cx0 + label.cx, cy0 + label.y + label.h + int(round(rect.h * 0.07)))
            self._chest_tries = 0
        self._sleep(1.0)
        return True

    def _path(self, rect, fast: bool = False) -> None:
        now = time.perf_counter()
        if self._path_at and now - self._path_at < self._num("path_retry_sec", 0.5):
            self.status("Choose Your Path, waiting for the pick to go through...")
            return
        self.status("Choose Your Path, reading the paths...")
        settle = self._num("path_settle_sec", 0.0)
        if fast:
            settle = min(settle, 0.12)
        if self._sleep(settle):
            return
        frame = self._frame(rect)
        if frame is None or not path_banner(frame):
            return
        img, px, py = crop(frame, area_of(self.cfg, "path_area"))
        cards = find_cards(img, rect.h)
        self._path_tries += 1
        tries = self._path_tries
        if tries > 1 and self._path_cards and not cards:
            pt = PATH_CONFIRM[(tries - 2) % len(PATH_CONFIRM)]
            self.status("Something is covering the path cards, clicking where its confirm button would be.")
            self._click_point_at(rect, pt)
            if tries % 2 == 0:
                self._tap(False)
            self._path_at = time.perf_counter()
            return
        if not cards:
            spot = (1, 0, 2)[(tries - 1) % 3]
            self.status("Couldn't find the path cards, clicking where they usually are.")
            self._click_point_at(rect, USUAL_CARDS[spot])
            if tries == 1:
                self.note("path", "path %d" % (spot + 1))
                self._reset_encounter_skill_cooldowns()
            self._path_at = time.perf_counter()
            self._sleep(self._num("after_pick_sec", 0.0))
            return
        self._path_cards = max(self._path_cards, len(cards))
        lines = read_card_lines(self.reader, img, cards)
        pick = choose(cards, lines, self.cfg)
        self._path_pick = pick
        card = cards[pick]
        title = _title(lines_in(lines, card)) or "path %d" % (pick + 1)
        if tries > 1 and tries % 2 == 0 and len(cards) == len(USUAL_CARDS):
            self.status("%s didn't go through, clicking where that card usually is." % title)
            self._click_point_at(rect, USUAL_CARDS[pick])
        else:
            self.status("Picked %s." % title)
            self._click(rect, px + card.cx, py + card.cy)
        if tries == 1:
            self.note("path", title)
        self._reset_encounter_skill_cooldowns()
        self._path_at = time.perf_counter()
        self._sleep(self._num("after_pick_sec", 0.0))

    def _click_point_at(self, rect, pt: dict) -> None:
        gui_click_at(*self._abs(rect, pt))
        self._progress()
        self.note("action", "")
