"""The NBA's official injury report (a PDF published every 15 minutes on game days).

The PDF is a table: Game Date, Game Time, Matchup, Team, Player Name,
Current Status, Reason. Text is read word by word with its position on the
page, and each word is placed in a column by its x position. Column edges
come from the header row on each page; if the header cannot be found, the
measured edges of the 2023-26 layout are used.

Two layout quirks are handled:
  * A cell that wraps onto two lines makes the other cells sit between
    them, so the first half of a long reason appears above the player's
    line. Reason-only lines are attached to the nearest player line.
  * Date, time, matchup and team are printed once per group, so they carry
    down to the rows below.

Anything that does not fit (an unknown status, a row with no team) is
counted. If too much does not fit, the report is stored with parse_ok = 0
and the dashboard labels it unverified instead of trusting it.
"""
import io
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
HEADERS = ["Game Date", "Game Time", "Matchup", "Team", "Player Name", "Current Status", "Reason"]
# Right edges of the first six columns in the 2023-26 layout (points), measured from real reports.
FALLBACK_EDGES = [108.8, 183.5, 255.1, 371.9, 543.5, 655.0]
STATUSES = {"out": "out", "doubtful": "doubtful", "questionable": "questionable",
            "probable": "probable", "available": "available"}
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*\(ET\)$")
MATCH_RE = re.compile(r"^[A-Z]{2,4}@[A-Z]{2,4}$")
TITLE_RE = re.compile(r"Injury Report:\s*(\d{2}/\d{2}/\d{2,4})\s+(\d{1,2}:\d{2})\s*(AM|PM)", re.I)


@dataclass
class Entry:
    game_date: Optional[str]      # YYYY-MM-DD
    game_time: Optional[str]
    matchup: Optional[str]        # AWAY@HOME
    team: Optional[str]           # as printed
    player: Optional[str]         # "Last, First" as printed; None for NOT YET SUBMITTED
    status: str                   # out | doubtful | questionable | probable | available | not_yet_submitted
    reason: str = ""
    top: float = 0.0
    page: int = 0


@dataclass
class ParsedReport:
    report_time_et: Optional[str]          # "YYYY-MM-DDTHH:MM"
    entries: list = field(default_factory=list)
    unparsed_lines: int = 0
    pages: int = 0
    used_fallback_edges: bool = False

    @property
    def ok(self) -> bool:
        good = len(self.entries)
        return good > 0 and self.unparsed_lines <= max(3, 0.05 * good)


# -- URLs ---------------------------------------------------------------------

def report_urls_before(when_et: datetime, steps: int = 12):
    """Candidate report URLs, newest first, at or before `when_et` (Eastern time).

    Since 2026-27 reports are named every 15 minutes (..._05_30PM). Earlier
    seasons used hourly names (..._05PM). Both are tried.
    """
    t = when_et.replace(second=0, microsecond=0)
    t = t - timedelta(minutes=t.minute % 15)
    out = []
    for i in range(steps):
        x = t - timedelta(minutes=15 * i)
        out.append((x, BASE.format(stamp=x.strftime("%Y-%m-%d_%I_%M%p"))))
    hourly = when_et.replace(minute=0, second=0, microsecond=0)
    for i in range(max(1, steps // 4)):
        x = hourly - timedelta(hours=i)
        out.append((x.replace(minute=30), BASE.format(stamp=x.strftime("%Y-%m-%d_%I%p"))))
    return out


# -- parsing ------------------------------------------------------------------

def _clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _edges_from_header(words):
    """Find the header row and return the x where each of columns 2-7 starts."""
    by_line = {}
    for w in words:
        by_line.setdefault(round(w["top"] / 3), []).append(w)
    for _, ws in sorted(by_line.items()):
        ws = sorted(ws, key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ws)
        if "Matchup" in text and "Current" in text and "Reason" in text:
            starts = {}
            games = [w for w in ws if w["text"] == "Game"]
            if len(games) >= 2:
                starts["Game Time"] = games[1]["x0"]
            for w in ws:
                if w["text"] == "Matchup":
                    starts["Matchup"] = w["x0"]
                elif w["text"] == "Team":
                    starts["Team"] = w["x0"]
                elif w["text"] == "Player":
                    starts["Player Name"] = w["x0"]
                elif w["text"] == "Current":
                    starts["Current Status"] = w["x0"]
                elif w["text"] == "Reason":
                    starts["Reason"] = w["x0"]
            if len(starts) == 6:
                header_bottom = max(w["bottom"] for w in ws)
                return [starts[h] - 2 for h in HEADERS[1:]], header_bottom
    return None, None


def parse_pdf(data: bytes) -> ParsedReport:
    import pdfplumber

    rep = ParsedReport(report_time_et=None)
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        rep.pages = len(pdf.pages)
        carry = {"game_date": None, "game_time": None, "matchup": None, "team": None}
        for pn, page in enumerate(pdf.pages):
            words = page.extract_words(x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False)
            text = page.extract_text() or ""
            if rep.report_time_et is None:
                m = TITLE_RE.search(text)
                if m:
                    d, hm, ampm = m.groups()
                    fmt = "%m/%d/%y" if len(d.split("/")[-1]) == 2 else "%m/%d/%Y"
                    dt = datetime.strptime(f"{d} {hm} {ampm.upper()}", f"{fmt} %I:%M %p")
                    rep.report_time_et = dt.strftime("%Y-%m-%dT%H:%M")
            edges, header_bottom = _edges_from_header(words)
            if edges is None:
                edges = FALLBACK_EDGES
                header_bottom = 0
                rep.used_fallback_edges = True
            lines = _group_lines([w for w in words if w["top"] > (header_bottom or 0) + 1])
            main, extra = [], []
            for top, ws in lines:
                cells = _cells(ws, edges)
                joined = " ".join(c for c in cells if c)
                if not joined or re.fullmatch(r"Page \d+ of \d+", joined) or joined.startswith("Injury Report:"):
                    continue
                if all(h in joined for h in ("Matchup", "Reason")):
                    continue
                status = STATUSES.get(cells[5].lower()) if cells[5] else None
                if "NOT YET SUBMITTED" in joined.upper():
                    main.append((top, cells, "not_yet_submitted"))
                elif status and cells[4]:
                    main.append((top, cells, status))
                elif any(cells[:6]) and not status:
                    # a team or matchup line on its own, or something unexpected
                    if cells[3] and not any(cells[4:6]):
                        main.append((top, cells, None))
                    else:
                        rep.unparsed_lines += 1
                        extra.append((top, cells))
                else:
                    extra.append((top, cells))     # reason-only line
            entries = []
            for top, cells, status in main:
                for i, key in enumerate(("game_date", "game_time", "matchup", "team")):
                    v = cells[i]
                    if key == "game_date" and v and DATE_RE.match(v):
                        carry[key] = datetime.strptime(v, "%m/%d/%Y").strftime("%Y-%m-%d")
                        carry.update(game_time=None, matchup=None, team=None)
                    elif key == "game_time" and v and TIME_RE.match(v):
                        carry[key] = v
                        carry.update(matchup=None, team=None)
                    elif key == "matchup" and v and MATCH_RE.match(v):
                        carry[key] = v
                        carry["team"] = None
                    elif key == "team" and v:
                        carry[key] = v
                if status is None:
                    continue
                player = None if status == "not_yet_submitted" else cells[4]
                entries.append(Entry(carry["game_date"], carry["game_time"], carry["matchup"], carry["team"],
                                     player, status, cells[6], top, pn))
            # attach reason-only lines to the nearest player line on this page
            for top, cells in extra:
                if not cells[6] or not entries:
                    continue
                near = min(entries, key=lambda e: abs(e.top - top))
                if abs(near.top - top) > 30:
                    rep.unparsed_lines += 1
                    continue
                near.reason = (f"{cells[6]} {near.reason}" if top < near.top else f"{near.reason} {cells[6]}").strip()
            rep.entries.extend(entries)
    for e in rep.entries:
        e.reason = _clean(e.reason)
        if e.team is None or e.matchup is None:
            rep.unparsed_lines += 1
    return rep


def _group_lines(words, tol=2.5):
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if lines and abs(lines[-1][0] - w["top"]) <= tol:
            lines[-1][1].append(w)
        else:
            lines.append([w["top"], [w]])
    return [(t, sorted(ws, key=lambda w: w["x0"])) for t, ws in lines]


def _cells(ws, edges):
    cells = [[] for _ in range(7)]
    for w in ws:
        mid = (w["x0"] + w["x1"]) / 2 if w["x1"] - w["x0"] < 60 else w["x0"] + 5
        col = sum(1 for e in edges if mid >= e)
        cells[col].append(w["text"])
    return [_clean(" ".join(c)) for c in cells]


# -- matching names to players -------------------------------------------------

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z ,\-']", " ", s).lower().replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


def split_report_name(name: str):
    """'Porter Jr., Kevin' -> ('porter', 'kevin')."""
    if "," in name:
        last, first = name.split(",", 1)
    else:
        parts = name.split()
        last, first = parts[-1], " ".join(parts[:-1])
    last = " ".join(t for t in norm(last).replace(",", " ").split() if t not in SUFFIXES)
    first = " ".join(t for t in norm(first).replace(",", " ").split() if t not in SUFFIXES)
    return last, first


def split_db_name(full: str):
    """'Kevin Porter Jr.' or 'K. Porter Jr.' -> ('porter', 'kevin' or 'k')."""
    toks = [t for t in norm(full).replace(",", " ").split() if t not in SUFFIXES]
    if not toks:
        return "", ""
    return " ".join(toks[1:]) if len(toks) > 1 else toks[0], toks[0] if len(toks) > 1 else ""


def match_player(report_name: str, candidates: dict):
    """candidates: player_id -> full name (players on that team). Returns (player_id, method) or (None, reason)."""
    last, first = split_report_name(report_name)
    scored = []
    for pid, full in candidates.items():
        cl, cf = split_db_name(full)
        if cl != last and not cl.endswith(" " + last) and not last.endswith(" " + cl):
            continue
        if cf == first:
            scored.append((3, pid))
        elif cf and first and (cf.startswith(first[:1]) and (len(cf) == 1 or first.startswith(cf))):
            scored.append((2, pid))
        elif cf and first and cf[:1] == first[:1]:
            scored.append((1, pid))
    if not scored:
        return None, "no player with that name on the team"
    best = max(s for s, _ in scored)
    top = [p for s, p in scored if s == best]
    if len(top) > 1:
        return None, "more than one possible match"
    return top[0], {3: "exact", 2: "initial", 1: "first letter"}[best]
