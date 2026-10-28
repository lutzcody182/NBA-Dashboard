"""Player prop odds from The Odds API (the-odds-api.com, v4).

The key comes from the ODDS_API_KEY environment variable (a GitHub secret);
it is never written to the database or the site.

Cost: listing events is free; each event's odds cost one credit per market
returned (one region). To stay inside a monthly plan, a pull happens at most
MAX_PULLS_PER_DAY times per Eastern day, at least MIN_HOURS_BETWEEN apart,
only for games that have not started, and never when fewer than
RESERVE_CREDITS remain.
"""
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone

import requests

BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
MARKETS = {
    "player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m",
    "player_points_rebounds_assists": "pra", "player_points_rebounds": "pr", "player_points_assists": "pa",
    "player_rebounds_assists": "ra",
}
REGION = "us"
MAX_PULLS_PER_DAY = 3
MIN_HOURS_BETWEEN = 3.0
LOOKAHEAD_HOURS = 8
RESERVE_CREDITS = 1500
SOURCE = "odds_api"


class OddsError(Exception):
    pass


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’`-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _get(path: str, key: str, params: dict = None):
    r = requests.get(f"{BASE}{path}", params={"apiKey": key, **(params or {})}, timeout=30)
    if r.status_code == 401:
        raise OddsError("the API key was rejected (401)")
    if r.status_code == 422:
        raise OddsError(f"request not accepted (422): {r.text[:200]}")
    if r.status_code == 429:
        raise OddsError("rate limited or out of credits (429)")
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining")
    return r.json(), (float(remaining) if remaining not in (None, "") else None)


def _state(conn) -> dict:
    row = conn.execute("SELECT payload FROM model_calibration WHERE key = 'odds_api:state'").fetchone()
    import json
    return json.loads(row[0]) if row else {"pulls": [], "remaining": None}


def _save_state(conn, s: dict, now_iso: str):
    import json
    conn.execute("INSERT OR REPLACE INTO model_calibration (key, payload, created_at) VALUES ('odds_api:state', ?, ?)",
                 (json.dumps(s), now_iso))
    conn.commit()


def should_pull(state: dict, now: datetime, upcoming: list, et_day: str):
    """(yes/no, reason). `upcoming`: tip times (UTC datetimes) of games not yet started."""
    if not upcoming:
        return False, "no games in the next hours"
    soon = [t for t in upcoming if now < t <= now + timedelta(hours=LOOKAHEAD_HOURS)]
    if not soon:
        return False, "no games in the next hours"
    today = [p for p in state.get("pulls", []) if p.get("et_day") == et_day]
    if len(today) >= MAX_PULLS_PER_DAY:
        return False, f"already pulled {len(today)} times today"
    if today:
        last = max(_parse(p["at"]) for p in today)
        if now - last < timedelta(hours=MIN_HOURS_BETWEEN):
            return False, f"last pull {((now - last).total_seconds() / 3600):.1f} hours ago"
    rem = state.get("remaining")
    if rem is not None and rem < RESERVE_CREDITS:
        return False, f"only {int(rem)} credits left this month"
    return True, "ok"


def match_event(ev: dict, games: list, teams: dict):
    """Our game for an Odds API event: same two teams, tip within 18 hours."""
    def team_id(name):
        n = norm_name(name)
        for tid, (city, nick) in teams.items():
            nk = norm_name(nick)
            if n == f"{norm_name(city)} {nk}" or n.endswith(" " + nk) or n == nk:
                return tid
        return None
    h, a = team_id(ev.get("home_team")), team_id(ev.get("away_team"))
    if h is None or a is None:
        return None
    t = _parse(ev["commence_time"])
    for g in games:
        if g["home_team_id"] == h and g["away_team_id"] == a and g.get("tip_utc"):
            if abs((_parse(g["tip_utc"]) - t).total_seconds()) < 18 * 3600:
                return g
    return None


def parse_props(payload: dict, roster: dict):
    """Rows from one event's odds. roster: normalized name -> player_id for the two teams."""
    rows, unmatched = [], set()
    for bk in payload.get("bookmakers", []):
        for mk in bk.get("markets", []):
            stat = MARKETS.get(mk.get("key"))
            if not stat:
                continue
            for o in mk.get("outcomes", []):
                side = (o.get("name") or "").lower()
                if side not in ("over", "under") or o.get("point") is None or o.get("price") is None:
                    continue
                pid = roster.get(norm_name(o.get("description")))
                if pid is None:
                    unmatched.add(o.get("description"))
                    continue
                rows.append({"book": bk.get("key"), "book_title": bk.get("title"), "market": mk["key"],
                             "player_id": pid, "side": side, "line": float(o["point"]),
                             "price": int(round(float(o["price"])))})
    return rows, sorted(n for n in unmatched if n)


def pull(conn, games: list, rosters: dict, now: datetime = None, et_day: str = None, force: bool = False) -> dict:
    """Fetch prop odds for upcoming games and store them in odds_snapshots.

    games: rows of the games table (dicts) for today/tomorrow.
    rosters: game_id -> {normalized name: player_id}.
    """
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        return {"status": "skipped", "reason": "no ODDS_API_KEY set"}
    now = now or datetime.now(timezone.utc)
    upcoming = [g for g in games if g.get("tip_utc") and _parse(g["tip_utc"]) > now and g.get("status") == "scheduled"]
    state = _state(conn)
    ok, why = should_pull(state, now, [_parse(g["tip_utc"]) for g in upcoming], et_day)
    if not ok and not force:
        out = {"status": "skipped", "reason": why, "remaining": state.get("remaining")}
        # once a day, confirm the key works with the free events call (costs no credits)
        if state.get("key_checked") != et_day:
            try:
                evs, rem = _get("/events", key, {"dateFormat": "iso"})
                state.update({"key_checked": et_day, "key_ok": True, "events_listed": len(evs)})
                if rem is not None:
                    state["remaining"] = rem
                out.update({"key_ok": True, "remaining": state["remaining"], "events_listed": len(evs)})
            except Exception as exc:
                state.update({"key_checked": et_day, "key_ok": False})
                out.update({"key_ok": False, "status": "error", "errors": [f"key check: {exc}"]})
            _save_state(conn, state, now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        return out
    teams = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT team_id, city, name FROM teams")}
    events, remaining = _get("/events", key, {"dateFormat": "iso"})
    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    stored, matched, unmatched_names, unmatched_events, errors = 0, 0, set(), [], []
    for ev in events:
        g = match_event(ev, upcoming, teams)
        if g is None:
            if _parse(ev["commence_time"]) <= now + timedelta(hours=LOOKAHEAD_HOURS) and _parse(ev["commence_time"]) > now:
                unmatched_events.append(f"{ev.get('away_team')} at {ev.get('home_team')}")
            continue
        if _parse(g["tip_utc"]) > now + timedelta(hours=LOOKAHEAD_HOURS):
            continue
        try:
            payload, rem = _get(f"/events/{ev['id']}/odds", key, {
                "regions": REGION, "markets": ",".join(MARKETS), "oddsFormat": "american", "dateFormat": "iso"})
        except OddsError as exc:
            errors.append(str(exc))
            break
        except Exception as exc:
            errors.append(f"{ev.get('away_team')} at {ev.get('home_team')}: {exc}")
            continue
        if rem is not None:
            remaining = rem
        matched += 1
        rows, un = parse_props(payload, rosters.get(g["game_id"], {}))
        unmatched_names.update(un)
        conn.executemany(
            """INSERT OR REPLACE INTO odds_snapshots
               (game_id, fetched_at, book, market, player_id, side, line, price_american, is_closing, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            [(g["game_id"], fetched_at, r["book"], r["market"], r["player_id"], r["side"], r["line"], r["price"], SOURCE)
             for r in rows])
        stored += len(rows)
    conn.commit()
    state["pulls"] = state.get("pulls", [])[-30:]
    state["pulls"].append({"at": fetched_at, "et_day": et_day, "games": matched, "rows": stored})
    state["remaining"] = remaining
    _save_state(conn, state, fetched_at)
    return {"status": "error" if errors and not stored else "stored", "fetched_at": fetched_at, "games": matched,
            "rows": stored, "remaining": remaining, "unmatched_players": sorted(unmatched_names)[:40],
            "unmatched_events": unmatched_events, "errors": errors[:5]}
