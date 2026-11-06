"""Player prop odds from The Odds API (the-odds-api.com, v4).

The key comes from the ODDS_API_KEY environment variable (a GitHub secret);
it is never written to the database or the site.

Cost: listing events is free; each event's odds cost one credit per market
returned (one region), and the game lines one credit per market for the
whole slate. Each game is pulled once on its game day (from 8 AM Eastern),
then about hourly in the last NEAR_HOURS before tip-off (plan_pull); only the
games that are due are pulled. NBA pulls stay within DAILY_CREDIT_CAP a day
and MONTHLY_CREDIT_CAP a month, and stop when fewer than RESERVE_CREDITS
remain on the shared key.
"""
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
MARKETS = {
    "player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m",
    "player_points_rebounds_assists": "pra", "player_points_rebounds": "pr", "player_points_assists": "pa",
    "player_rebounds_assists": "ra",
}
GAME_MARKETS = ("h2h", "spreads", "totals")
REGION = "us"
# State-licensed US sportsbooks. The "us" region also carries offshore books
# (Bovada, BetOnline, MyBookie, LowVig, BetUS); those are stored but never used.
LEGAL_BOOKS = {
    "draftkings", "fanduel", "betmgm", "williamhill_us", "caesars", "espnbet", "betrivers", "fanatics",
    "hardrockbet", "ballybet", "betparx", "pointsbetus", "unibet_us", "superbook", "wynnbet", "windcreek",
    "betway", "tipico", "sisportsbook", "circasports", "twinspires", "foxbet", "barstool", "thescore",
}
# Where picks may be placed: the best price shown and every bet comes only from these books (FanDuel,
# theScore Bet / ESPN BET, Caesars, Fanatics). The other legal books still count toward the market's fair
# chance and line, because more books give a sharper read of the market, but are never offered as a place to bet.
BET_BOOKS = {"fanduel", "espnbet", "thescore", "williamhill_us", "caesars", "fanatics"}
# When odds are pulled (since Nov 6, 2026): each game gets one pull on its game day once it is within
# LOOKAHEAD_HOURS (the morning pull), then one every REPULL_MINUTES during the last NEAR_HOURS before tip-off,
# when injury news and starting lineups land and lines move most. Only the games that are due are pulled,
# so a pull costs about 8 credits per game plus 3 for the game lines.
NEAR_HOURS = 4.0
REPULL_MINUTES = 55               # "hourly": the site runs every 15 minutes, so this lands about once an hour
MIN_TIP_MINUTES = 5               # no pull for a game about to start
DAILY_CREDIT_CAP = 600            # NBA's share of the key that day; when short, the games tipping soonest go first
MONTHLY_CREDIT_CAP = 10500        # NBA's share per calendar month; the 20,000-credit key is shared with NHL and NFL
LOOKAHEAD_HOURS = 15               # from ~8 AM ET this covers the whole evening slate
RESERVE_CREDITS = 1500
EARLIEST_ET_HOUR = 8               # no pulls before 8 AM Eastern: overnight lines are thin and it wastes a pull
ET = ZoneInfo("America/New_York")
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


def pull_cost(n_games: int) -> int:
    return n_games * len(MARKETS) + len(GAME_MARKETS)


def spent(state: dict, et_day: str = None, month: str = None) -> float:
    """Credits NBA pulls used on one Eastern day, or in one calendar month ("YYYY-MM")."""
    ps = state.get("pulls", [])
    if et_day is not None:
        ps = [p for p in ps if p.get("et_day") == et_day]
    if month is not None:
        ps = [p for p in ps if (p.get("et_day") or "")[:7] == month]
    return float(sum(p.get("cost", pull_cost(p.get("games", 0))) for p in ps))


def plan_pull(state: dict, now: datetime, upcoming: dict, last_pulled: dict, et_day: str):
    """Which games to pull now: (game_ids, reason).

    upcoming: game_id -> tip time (UTC datetime) for games not yet started.
    last_pulled: game_id -> when its props were last pulled (UTC datetime), if ever.
    """
    if not upcoming:
        return [], "no games in the next hours"
    if now.astimezone(ET).hour < EARLIEST_ET_HOUR:
        return [], f"odds are pulled from {EARLIEST_ET_HOUR} AM Eastern"
    rem = state.get("remaining")
    if rem is not None and rem < RESERVE_CREDITS:
        return [], f"only {int(rem)} credits left this month"
    day_start = now.astimezone(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    due, waits = [], []
    for gid, tip in upcoming.items():
        to_tip = tip - now
        if to_tip <= timedelta(minutes=MIN_TIP_MINUTES) or to_tip > timedelta(hours=LOOKAHEAD_HOURS):
            continue
        last = last_pulled.get(gid)
        if last is None or last < day_start:
            due.append((tip, gid))                                    # first pull of its game day
        elif to_tip <= timedelta(hours=NEAR_HOURS):
            ready = last + timedelta(minutes=REPULL_MINUTES)
            (due.append((tip, gid)) if now >= ready else waits.append(ready))
        else:
            waits.append(max(last + timedelta(minutes=REPULL_MINUTES), tip - timedelta(hours=NEAR_HOURS)))
    if not due:
        nxt = min(waits) if waits else None
        return [], ("next pull in about %d minutes" % max(1, (nxt - now).total_seconds() // 60)) if nxt else \
            "no games in the next hours"
    due.sort()
    left_day = DAILY_CREDIT_CAP - spent(state, et_day=et_day)
    left_month = MONTHLY_CREDIT_CAP - spent(state, month=et_day[:7])
    left = min(left_day, left_month)
    n = 0
    while n < len(due) and pull_cost(n + 1) <= left:                  # soonest tips first when short
        n += 1
    if n == 0:
        which = "today's" if left_day <= left_month else "this month's"
        return [], f"{which} NBA odds budget is used"
    return [gid for _, gid in due[:n]], "ok" if n == len(due) else f"budget covers {n} of {len(due)} due games"


def last_pulled(conn) -> dict:
    rows = conn.execute("""SELECT game_id, MAX(fetched_at) FROM odds_snapshots
                           WHERE source = ? AND player_id IS NOT NULL GROUP BY game_id""", (SOURCE,)).fetchall()
    return {g: _parse(t) for g, t in rows if t}


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


def parse_game_lines(events: list, games: list, teams: dict) -> list:
    """Moneyline, spread and total rows from the /odds response, keyed to our games."""
    rows = []
    for ev in events:
        g = match_event(ev, games, teams)
        if g is None:
            continue
        home, away = norm_name(ev.get("home_team")), norm_name(ev.get("away_team"))
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") not in GAME_MARKETS:
                    continue
                for o in mk.get("outcomes", []):
                    nm = norm_name(o.get("name"))
                    if mk["key"] == "totals":
                        side = nm if nm in ("over", "under") else None
                    else:
                        side = "home" if nm == home else "away" if nm == away else None
                    if side is None or o.get("price") is None:
                        continue
                    line = o.get("point")
                    if mk["key"] != "h2h" and line is None:
                        continue
                    rows.append({"game_id": g["game_id"], "book": bk.get("key"), "market": mk["key"], "side": side,
                                 "line": float(line) if line is not None else None,
                                 "price": int(round(float(o["price"])))})
    return rows


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
    if force:
        due_ids = [g["game_id"] for g in upcoming if _parse(g["tip_utc"]) <= now + timedelta(hours=LOOKAHEAD_HOURS)]
        why = "forced"
    else:
        due_ids, why = plan_pull(state, now, {g["game_id"]: _parse(g["tip_utc"]) for g in upcoming},
                                 last_pulled(conn), et_day)
    if not due_ids:
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
    # the free events call reports the balance right before this pull, so the cost below
    # is only what this pull spent (not credits used elsewhere since the last pull)
    before = remaining if remaining is not None else state.get("remaining")
    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    stored, matched, unmatched_names, unmatched_events, errors = 0, 0, set(), [], []
    due_set = set(due_ids)
    for ev in events:
        g = match_event(ev, upcoming, teams)
        if g is None:
            if _parse(ev["commence_time"]) <= now + timedelta(hours=LOOKAHEAD_HOURS) and _parse(ev["commence_time"]) > now:
                unmatched_events.append(f"{ev.get('away_team')} at {ev.get('home_team')}")
            continue
        if g["game_id"] not in due_set:                 # only the games that are due this run
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
    # game lines for the same games: one call, one credit per market
    game_rows = 0
    soon = [g for g in upcoming if g["game_id"] in due_set]
    if soon and not (errors and not stored):
        try:
            evs, rem = _get("/odds", key, {"regions": REGION, "markets": ",".join(GAME_MARKETS),
                                           "oddsFormat": "american", "dateFormat": "iso"})
            if rem is not None:
                remaining = rem
            gl = parse_game_lines(evs, soon, teams)
            # game markets have no player (player_id NULL); the moneyline is stored at line 0
            conn.executemany(
                """INSERT OR REPLACE INTO odds_snapshots
                   (game_id, fetched_at, book, market, player_id, side, line, price_american, is_closing, source)
                   VALUES (?, ?, ?, ?, NULL, ?, ?, ?, 0, ?)""",
                [(r["game_id"], fetched_at, r["book"], r["market"], r["side"],
                  r["line"] if r["line"] is not None else 0.0, r["price"], SOURCE) for r in gl])
            game_rows = len(gl)
        except Exception as exc:
            errors.append(f"game lines: {exc}")
    conn.commit()
    state["pulls"] = state.get("pulls", [])[-600:]      # enough to total a month
    cost = (before - remaining) if before is not None and remaining is not None and before >= remaining \
        else pull_cost(matched)
    state["pulls"].append({"at": fetched_at, "et_day": et_day, "games": matched, "rows": stored, "game_rows": game_rows,
                           "cost": cost})
    state["remaining"] = remaining
    _save_state(conn, state, fetched_at)
    return {"status": "error" if errors and not stored else "stored", "fetched_at": fetched_at, "games": matched,
            "plan": why,
            "rows": stored, "game_rows": game_rows, "remaining": remaining, "unmatched_players": sorted(unmatched_names)[:40],
            "unmatched_events": unmatched_events, "errors": errors[:5]}
