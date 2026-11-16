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

Extra markets (since Nov 16, 2026, tracked before they can be recommended):
steals, blocks, steals+blocks, turnovers, made free throws and field goals,
double-doubles, alternate player lines, team totals, first-half lines and
alternate spreads/totals. They cost about three times the main props, so they
ride along on only two pulls per game: the first of its game day and one in
the last EXT_NEAR_MINUTES before tip-off.

News pulls: when a player's injury status or a confirmed starting lineup
changes for a game within NEWS_HOURS of tip-off, that game's main markets are
pulled again right away (at most NEWS_MAX_PER_GAME a day, NEWS_MIN_GAP apart),
to catch prices the books have not moved yet.
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
# extra markets, two pulls a game a day (see the module notes)
EXTRA_PROP_MARKETS = {
    "player_steals": "stl", "player_blocks": "blk", "player_blocks_steals": "stocks", "player_turnovers": "tov",
    "player_frees_made": "ftm", "player_field_goals": "fgm", "player_double_double": "dd",
}
ALT_MARKETS = {f"{k}_alternate": f"alt_{v}" for k, v in (
    ("player_points", "pts"), ("player_rebounds", "reb"), ("player_assists", "ast"), ("player_threes", "fg3m"),
    ("player_points_rebounds_assists", "pra"), ("player_points_rebounds", "pr"), ("player_points_assists", "pa"),
    ("player_rebounds_assists", "ra"), ("player_blocks", "blk"), ("player_steals", "stl"),
    ("player_turnovers", "tov"))}
EVENT_GAME_MARKETS = ("team_totals", "h2h_h1", "spreads_h1", "totals_h1", "alternate_spreads", "alternate_totals")
EXT_MARKETS = (*EXTRA_PROP_MARKETS, *ALT_MARKETS, *EVENT_GAME_MARKETS)
PROP_MARKETS = {**MARKETS, **EXTRA_PROP_MARKETS, **ALT_MARKETS}     # market key -> our stat code
STAT_MARKET = {v: k for k, v in PROP_MARKETS.items()}
GAME_MARKETS = ("h2h", "spreads", "totals")
ALL_GAME_MARKETS = (*GAME_MARKETS, *EVENT_GAME_MARKETS)
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
DAILY_CREDIT_CAP = 1200           # NBA's share of the key that day; when short, the games tipping soonest go first
MONTHLY_CREDIT_CAP = 24000        # NBA's share per calendar month (raised from 10,500 on Nov 16, 2026 for the extra
                                  # markets); the key (about 100,000 a month) is shared with NHL and NFL
EXT_NEAR_MINUTES = 90             # the second extra-markets pull lands in the last 90 minutes before tip-off
EXT_MIN_GAP_HOURS = 2.0           # and at least this long after the first
NEWS_HOURS = 3.0                  # news pulls only this close to tip-off
NEWS_MIN_GAP = 10                 # minutes since the game's last pull
NEWS_MAX_PER_GAME = 6             # a day
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


def pull_cost(n_games: int, n_ext: int = 0) -> int:
    return n_games * len(MARKETS) + n_ext * len(EXT_MARKETS) + len(GAME_MARKETS)


def ext_due(state: dict, gid: str, now: datetime, tip: datetime, day_start: datetime) -> bool:
    """Should this game's pull also fetch the extra markets? First pull of its game day, and once near tip."""
    done = [_parse(t) for t in state.get("ext", {}).get(gid, []) if _parse(t) >= day_start]
    if not done:
        return True
    near = tip - now <= timedelta(minutes=EXT_NEAR_MINUTES)
    return near and len(done) < 2 and now - max(done) >= timedelta(hours=EXT_MIN_GAP_HOURS)


def news_games(state: dict, fingerprints: dict) -> set:
    """Games whose injury/lineup fingerprint changed since their last pull."""
    old = state.get("fp", {})
    return {g for g, fp in (fingerprints or {}).items() if g in old and old[g] != fp}


def spent(state: dict, et_day: str = None, month: str = None) -> float:
    """Credits NBA pulls used on one Eastern day, or in one calendar month ("YYYY-MM")."""
    ps = state.get("pulls", [])
    if et_day is not None:
        ps = [p for p in ps if p.get("et_day") == et_day]
    if month is not None:
        ps = [p for p in ps if (p.get("et_day") or "")[:7] == month]
    return float(sum(p.get("cost", pull_cost(p.get("games", 0))) for p in ps))


def plan_pull(state: dict, now: datetime, upcoming: dict, last_pulled: dict, et_day: str, news: set = None):
    """Which games to pull now: (game_ids, reason). Also sets state["_plan"] = {"ext": [...], "news": [...]}.

    upcoming: game_id -> tip time (UTC datetime) for games not yet started.
    last_pulled: game_id -> when its props were last pulled (UTC datetime), if ever.
    news: games whose injury or lineup news changed since their last pull.
    """
    state["_plan"] = {"ext": [], "news": []}
    if not upcoming:
        return [], "no games in the next hours"
    if now.astimezone(ET).hour < EARLIEST_ET_HOUR:
        return [], f"odds are pulled from {EARLIEST_ET_HOUR} AM Eastern"
    rem = state.get("remaining")
    if rem is not None and rem < RESERVE_CREDITS:
        return [], f"only {int(rem)} credits left this month"
    day_start = now.astimezone(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    due, waits, by_news = [], [], []
    news_today = state.get("news_pulls", {})
    for gid, tip in upcoming.items():
        to_tip = tip - now
        if to_tip <= timedelta(minutes=MIN_TIP_MINUTES) or to_tip > timedelta(hours=LOOKAHEAD_HOURS):
            continue
        last = last_pulled.get(gid)
        if last is None or last < day_start:
            due.append((tip, gid))                                    # first pull of its game day
        elif to_tip <= timedelta(hours=NEAR_HOURS):
            ready = last + timedelta(minutes=REPULL_MINUTES)
            if now >= ready:
                due.append((tip, gid))
            elif gid in (news or ()) and to_tip <= timedelta(hours=NEWS_HOURS) and \
                    now - last >= timedelta(minutes=NEWS_MIN_GAP) and \
                    len([t for t in news_today.get(gid, []) if _parse(t) >= day_start]) < NEWS_MAX_PER_GAME:
                due.append((tip, gid))                                # news since the last pull
                by_news.append(gid)
            else:
                waits.append(ready)
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
    n, ext, cost = 0, [], len(GAME_MARKETS)
    for tip, gid in due:                                              # soonest tips first when short
        c = len(MARKETS)
        if cost + c > left:
            break
        cost += c
        n += 1
        if ext_due(state, gid, now, tip, day_start) and cost + len(EXT_MARKETS) <= left:
            cost += len(EXT_MARKETS)
            ext.append(gid)
    if n == 0:
        which = "today's" if left_day <= left_month else "this month's"
        return [], f"{which} NBA odds budget is used"
    picked = [gid for _, gid in due[:n]]
    state["_plan"] = {"ext": ext, "news": [g for g in by_news if g in picked]}
    return picked, "ok" if n == len(due) else f"budget covers {n} of {len(due)} due games"


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
    """Rows from one event's odds. roster: normalized name -> player_id for the two teams.

    Yes/No markets (double-doubles) are stored as over/under 0.5."""
    rows, unmatched = [], set()
    for bk in payload.get("bookmakers", []):
        for mk in bk.get("markets", []):
            stat = PROP_MARKETS.get(mk.get("key"))
            if not stat:
                continue
            for o in mk.get("outcomes", []):
                side = (o.get("name") or "").lower()
                side = {"yes": "over", "no": "under"}.get(side, side)
                if o.get("point") is None and stat == "dd":
                    o = {**o, "point": 0.5}
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


def parse_event_game_lines(payload: dict, g: dict) -> list:
    """Team totals, first-half and alternate game lines from one event's odds.

    Team totals are stored with side 'home_over', 'away_under' and so on."""
    home, away = norm_name(payload.get("home_team")), norm_name(payload.get("away_team"))
    rows = []
    for bk in payload.get("bookmakers", []):
        for mk in bk.get("markets", []):
            key = mk.get("key")
            if key not in EVENT_GAME_MARKETS:
                continue
            for o in mk.get("outcomes", []):
                nm, price, line = norm_name(o.get("name")), o.get("price"), o.get("point")
                if price is None:
                    continue
                if key == "team_totals":
                    team = norm_name(o.get("description"))
                    t = "home" if team == home else "away" if team == away else None
                    side = f"{t}_{nm}" if t and nm in ("over", "under") else None
                elif key in ("totals_h1", "alternate_totals"):
                    side = nm if nm in ("over", "under") else None
                else:
                    side = "home" if nm == home else "away" if nm == away else None
                if side is None or (key != "h2h_h1" and line is None):
                    continue
                rows.append({"game_id": g["game_id"], "book": bk.get("key"), "market": key, "side": side,
                             "line": float(line) if line is not None else 0.0, "price": int(round(float(price)))})
    return rows


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


def pull(conn, games: list, rosters: dict, now: datetime = None, et_day: str = None, force: bool = False,
         fingerprints: dict = None) -> dict:
    """Fetch prop odds for upcoming games and store them in odds_snapshots.

    games: rows of the games table (dicts) for today/tomorrow.
    rosters: game_id -> {normalized name: player_id}.
    fingerprints: game_id -> a summary of its injury statuses and confirmed lineups, to spot news.
    """
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        return {"status": "skipped", "reason": "no ODDS_API_KEY set"}
    now = now or datetime.now(timezone.utc)
    upcoming = [g for g in games if g.get("tip_utc") and _parse(g["tip_utc"]) > now and g.get("status") == "scheduled"]
    state = _state(conn)
    news = news_games(state, fingerprints)
    for gid, fp in (fingerprints or {}).items():       # first sight of a game: remember it, no news yet
        state.setdefault("fp", {}).setdefault(gid, fp)
    if force:
        due_ids = [g["game_id"] for g in upcoming if _parse(g["tip_utc"]) <= now + timedelta(hours=LOOKAHEAD_HOURS)]
        why = "forced"
        state["_plan"] = {"ext": list(due_ids), "news": []}
    else:
        due_ids, why = plan_pull(state, now, {g["game_id"]: _parse(g["tip_utc"]) for g in upcoming},
                                 last_pulled(conn), et_day, news)
    plan = state.pop("_plan", {"ext": [], "news": []})
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
        _save_state(conn, state, now.strftime("%Y-%m-%dT%H:%M:%SZ"))   # keeps the news fingerprints too
        return out
    teams = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT team_id, city, name FROM teams")}
    events, remaining = _get("/events", key, {"dateFormat": "iso"})
    # the free events call reports the balance right before this pull, so the cost below
    # is only what this pull spent (not credits used elsewhere since the last pull)
    before = remaining if remaining is not None else state.get("remaining")
    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    stored, matched, unmatched_names, unmatched_events, errors = 0, 0, set(), [], []
    ext_rows = 0
    due_set = set(due_ids)
    for ev in events:
        g = match_event(ev, upcoming, teams)
        if g is None:
            if _parse(ev["commence_time"]) <= now + timedelta(hours=LOOKAHEAD_HOURS) and _parse(ev["commence_time"]) > now:
                unmatched_events.append(f"{ev.get('away_team')} at {ev.get('home_team')}")
            continue
        if g["game_id"] not in due_set:                 # only the games that are due this run
            continue
        with_ext = g["game_id"] in plan["ext"]
        try:
            payload, rem = _get(f"/events/{ev['id']}/odds", key, {
                "regions": REGION, "markets": ",".join([*MARKETS, *(EXT_MARKETS if with_ext else ())]),
                "oddsFormat": "american", "dateFormat": "iso"})
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
        if with_ext:
            gl = parse_event_game_lines(payload, g)
            conn.executemany(
                """INSERT OR REPLACE INTO odds_snapshots
                   (game_id, fetched_at, book, market, player_id, side, line, price_american, is_closing, source)
                   VALUES (?, ?, ?, ?, NULL, ?, ?, ?, 0, ?)""",
                [(r["game_id"], fetched_at, r["book"], r["market"], r["side"], r["line"], r["price"], SOURCE)
                 for r in gl])
            ext_rows += len(gl)
            state.setdefault("ext", {}).setdefault(g["game_id"], []).append(fetched_at)
        if g["game_id"] in plan["news"]:
            state.setdefault("news_pulls", {}).setdefault(g["game_id"], []).append(fetched_at)
        if fingerprints and g["game_id"] in fingerprints:
            state.setdefault("fp", {})[g["game_id"]] = fingerprints[g["game_id"]]
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
    keep_from = (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for k in ("ext", "news_pulls"):                      # only recent days are needed
        state[k] = {g: [t for t in ts if t >= keep_from] for g, ts in state.get(k, {}).items()}
        state[k] = {g: ts for g, ts in state[k].items() if ts}
    live = {g["game_id"] for g in upcoming}
    state["fp"] = {g: v for g, v in state.get("fp", {}).items() if g in live}
    cost = (before - remaining) if before is not None and remaining is not None and before >= remaining \
        else pull_cost(matched, len(plan["ext"]))
    state["pulls"].append({"at": fetched_at, "et_day": et_day, "games": matched, "rows": stored, "game_rows": game_rows,
                           "cost": cost, "ext_games": len(plan["ext"]), "news_games": plan["news"]})
    state["remaining"] = remaining
    _save_state(conn, state, fetched_at)
    return {"status": "error" if errors and not stored else "stored", "fetched_at": fetched_at, "games": matched,
            "plan": why, "ext_games": plan["ext"], "news_games": plan["news"], "ext_rows": ext_rows,
            "rows": stored, "game_rows": game_rows, "remaining": remaining, "unmatched_players": sorted(unmatched_names)[:40],
            "unmatched_events": unmatched_events, "errors": errors[:5]}
