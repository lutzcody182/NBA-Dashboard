"""Phone notifications for newly locked bets, through ntfy (ntfy.sh).

The topic name comes from the NTFY_TOPIC environment variable (a GitHub
secret) and is never written to the site. Every bet on the ledger with a
higher bet_id than the last one notified is sent, as long as its game has
not started, so a failed send is retried on the next run and nothing is
sent twice. One notification per batch.
"""
import json
import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger(__name__)
SERVER = "https://ntfy.sh"
SITE = "https://lutzcody182.github.io/NBA-Dashboard/#bets"
STATE_KEY = "ntfy:last_bet_id"
ET = ZoneInfo("America/New_York")
STAT_WORD = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "pra": "pts+reb+ast",
             "pr": "pts+reb", "pa": "pts+ast", "ra": "reb+ast"}
BOOKS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
         "caesars": "Caesars", "espnbet": "ESPN BET", "betrivers": "BetRivers", "fanatics": "Fanatics",
         "hardrockbet": "Hard Rock", "ballybet": "Bally Bet", "betparx": "betPARX", "pointsbetus": "PointsBet",
         "unibet_us": "Unibet", "superbook": "SuperBook", "wynnbet": "WynnBET", "windcreek": "Wind Creek",
         "betway": "Betway", "tipico": "Tipico", "sisportsbook": "SI Sportsbook", "circasports": "Circa"}


def _odds(p) -> str:
    p = int(round(float(p)))
    return f"+{p}" if p > 0 else str(p)


def _num(x) -> str:
    x = float(x)
    return str(int(x)) if x == int(x) else f"{x:g}"


def bet_line(b: dict, abbr: dict, names: dict) -> str:
    home, away = abbr.get(b["home_team_id"], "?"), abbr.get(b["away_team_id"], "?")
    if b["kind"] == "prop":
        what = f"{names.get(b['player_id'], b['player_id'])} {b['side']} {_num(b['line'])} {STAT_WORD.get(b['stat'], b['stat'])}"
    elif b["kind"] == "total":
        what = f"{away} @ {home} {b['side']} {_num(b['line'])}"
    else:
        team = home if b["side"] == "home" else away
        what = f"{team} moneyline" if b["kind"] == "moneyline" else \
            f"{team} {'+' if b['line'] > 0 else ''}{_num(b['line'])}"
    units = f"{float(b['units']):g}u"
    return f"{what} {_odds(b['price'])} · {units} · {BOOKS.get(b['book'], b['book'])}"


def _tip_et(tip: str) -> str:
    t = datetime.strptime(tip, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(ET)
    return t.strftime("%-I:%M %p").replace(":00 ", " ") + " ET"


def pending(conn, now_iso: str) -> list:
    row = conn.execute("SELECT payload FROM model_calibration WHERE key = ?", (STATE_KEY,)).fetchone()
    last = json.loads(row[0]).get("bet_id", 0) if row else None
    if last is None:
        # first run: start from the newest bet so the old ledger is not sent all at once
        top = conn.execute("SELECT COALESCE(MAX(bet_id), 0) FROM bets").fetchone()[0]
        _save(conn, top)
        return []
    cur = conn.execute(
        """SELECT b.bet_id, b.kind, b.player_id, b.stat, b.side, b.line, b.price, b.book, b.units, b.ev,
                  g.home_team_id, g.away_team_id, g.tip_utc
           FROM bets b JOIN games g ON g.game_id = b.game_id
           WHERE b.bet_id > ? ORDER BY g.tip_utc, b.bet_id""", (last,))
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _save(conn, bet_id: int) -> None:
    conn.execute("INSERT OR REPLACE INTO model_calibration (key, payload, created_at) VALUES (?, ?, ?)",
                 (STATE_KEY, json.dumps({"bet_id": int(bet_id)}),
                  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
    conn.commit()


def message(bets: list, abbr: dict, names: dict) -> tuple:
    units = sum(float(b["units"]) for b in bets)
    title = f"{len(bets)} new pick{'s' if len(bets) != 1 else ''} · {units:g} units"
    lines, last_tip = [], None
    for b in bets:
        if b["tip_utc"] != last_tip:
            lines.append(f"{abbr.get(b['away_team_id'], '?')} @ {abbr.get(b['home_team_id'], '?')}, {_tip_et(b['tip_utc'])}")
            last_tip = b["tip_utc"]
        lines.append("• " + bet_line(b, abbr, names))
    lines.append("Not betting advice. Bet only what you can afford to lose.")
    return title, "\n".join(lines)


def send(topic: str, title: str, body: str, session=None) -> None:
    s = session or requests
    r = s.post(f"{SERVER}/{topic}", data=body.encode("utf-8"), timeout=20,
               headers={"Title": title.encode("utf-8"), "Click": SITE, "Tags": "basketball", "Priority": "default"})
    r.raise_for_status()


def notify_new_bets(conn, topic=None, now_iso=None, session=None) -> dict:
    topic = topic if topic is not None else os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return {"status": "skipped", "reason": "no NTFY_TOPIC set"}
    now_iso = now_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        bets = pending(conn, now_iso)
    except Exception as exc:          # no bets table yet
        return {"status": "skipped", "reason": str(exc)}
    if not bets:
        return {"status": "nothing new"}
    top = max(b["bet_id"] for b in bets)
    live = [b for b in bets if b["tip_utc"] and b["tip_utc"] > now_iso]
    if not live:                       # every new one has already started: just move past them
        _save(conn, top)
        return {"status": "nothing new", "skipped_started": len(bets)}
    abbr = dict(conn.execute("SELECT team_id, abbr FROM teams").fetchall())
    names = dict(conn.execute("SELECT player_id, full_name FROM players").fetchall())
    title, body = message(live, abbr, names)
    try:
        send(topic, title, body, session)
    except Exception as exc:
        log.warning("ntfy send failed: %s", exc)
        return {"status": "error", "error": str(exc)}     # not saved: retried next run
    _save(conn, top)
    return {"status": "sent", "bets": len(live)}


def send_test(topic=None) -> dict:
    topic = topic if topic is not None else os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return {"status": "skipped", "reason": "no NTFY_TOPIC set"}
    send(topic, "NBA Court Lab is connected",
         "You'll get a notification here whenever new bet picks lock. Tap to open the Bets tab.")
    return {"status": "sent"}
