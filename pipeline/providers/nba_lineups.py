"""Starting lineups from nba.com's daily lineups file (the JSON nba.com/players/todays-lineups loads).

    https://stats.nba.com/js/data/leaders/00_daily_lineups_YYYYMMDD.json

For each game of the Eastern date it lists five players per team with a
lineupStatus of "Expected" (the team's likely starters) or "Confirmed"
(announced, about 30 minutes before tip-off). Player and team ids are the
NBA's own, the same ones the database uses. The file is free and changes
through the day, so it is read on every update; a failed read leaves the
last stored lineups in place.

Stored in game_lineups, one row per starter, with data_status 'confirmed'
when all five of a team's players are Confirmed, otherwise 'projected'.
A team's lineup is stored again only when it changes.
"""
from datetime import date, datetime, timezone

import requests

URL = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{d}.json"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}
SOURCE = "nba_daily_lineups"


def fetch(day: date, session=None) -> dict:
    s = session or requests
    r = s.get(URL.format(d=day.strftime("%Y%m%d")), headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def parse(payload: dict) -> list:
    """[{home, away, teams: {team_id: {"status": "confirmed"|"projected", "starters": [player_id, ...]}}}]"""
    out = []
    for g in (payload or {}).get("games", []):
        teams = {}
        for side in ("homeTeam", "awayTeam"):
            t = g.get(side) or {}
            players = [p for p in t.get("players") or [] if p.get("personId")]
            if not t.get("teamId") or not players:
                continue
            confirmed = all((p.get("lineupStatus") or "").lower() == "confirmed" for p in players)
            teams[int(t["teamId"])] = {"status": "confirmed" if confirmed else "projected",
                                       "starters": sorted(int(p["personId"]) for p in players)}
        home, away = (g.get("homeTeam") or {}).get("teamId"), (g.get("awayTeam") or {}).get("teamId")
        if home and away and teams:
            out.append({"home": int(home), "away": int(away), "teams": teams})
    return out


def latest(conn, game_ids=None) -> dict:
    """(game_id, team_id) -> {"status", "starters": set, "as_of"} from the newest stored lineup."""
    q = """SELECT l.game_id, l.team_id, l.player_id, l.data_status, l.as_of FROM game_lineups l
           JOIN (SELECT game_id, team_id, MAX(as_of) AS a FROM game_lineups WHERE source = ? GROUP BY game_id, team_id) m
             ON m.game_id = l.game_id AND m.team_id = l.team_id AND m.a = l.as_of
           WHERE l.source = ? AND l.is_starter = 1"""
    out = {}
    for gid, tid, pid, st, asof in conn.execute(q, (SOURCE, SOURCE)).fetchall():
        if game_ids is not None and gid not in game_ids:
            continue
        x = out.setdefault((gid, int(tid)), {"status": st, "starters": set(), "as_of": asof})
        x["starters"].add(int(pid))
    return out


def update(conn, day: date, session=None, now_iso: str = None) -> dict:
    """Read the day's lineups and store any team whose lineup or status changed."""
    now_iso = now_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    games = parse(fetch(day, session))
    ours = {(h, a): gid for gid, h, a in conn.execute(
        "SELECT game_id, home_team_id, away_team_id FROM games WHERE game_date = ?", (day.isoformat(),)).fetchall()}
    have = latest(conn)
    known = {r[0] for r in conn.execute("SELECT player_id FROM players")}
    stored, confirmed, matched = 0, 0, 0
    for g in games:
        gid = ours.get((g["home"], g["away"]))
        if gid is None:
            continue
        matched += 1
        for tid, t in g["teams"].items():
            t = {**t, "starters": [p for p in t["starters"] if p in known]}   # a player new to the database waits a run
            if len(t["starters"]) < 5:
                continue
            confirmed += t["status"] == "confirmed"
            old = have.get((gid, tid))
            if old and old["status"] == t["status"] and old["starters"] == set(t["starters"]):
                continue
            conn.executemany(
                "INSERT OR REPLACE INTO game_lineups (game_id, team_id, player_id, is_starter, data_status, as_of, source) "
                "VALUES (?, ?, ?, 1, ?, ?, ?)", [(gid, tid, pid, t["status"], now_iso, SOURCE) for pid in t["starters"]])
            stored += 1
    conn.commit()
    return {"status": "ok", "games": len(games), "matched": matched, "teams_confirmed": confirmed, "teams_stored": stored}
