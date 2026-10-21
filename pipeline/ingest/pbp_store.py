"""Store play-by-play results: lineup stints, player on-court totals, exact possessions.

For past seasons loaded from the archive there are no box scores, so this
module also builds the game, team and player box-score rows from the
play-by-play itself. Those rows are marked with the archive as their source.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from .. import config
from ..db import _PLAYER_ZONES, _STINT_STATS
from ..pbp.parse import GamePbp
from ..util import season_from_game_id, utc_now_iso
from .store import ensure_season, season_type

ET = ZoneInfo("America/New_York")


def _key(players) -> str:
    return "-".join(str(p) for p in sorted(players))


def store_pbp(conn, g: GamePbp, source: str) -> None:
    """Write stints and on-court totals, and replace estimated possessions with counted ones."""
    now = utc_now_iso()
    conn.execute("DELETE FROM lineup_stints WHERE game_id = ?", (g.game_id,))
    conn.execute("DELETE FROM player_game_pbp WHERE game_id = ?", (g.game_id,))
    teams = (g.home_id, g.away_id)
    rows = []
    for n, s in enumerate(g.stints, start=1):
        for t in teams:
            o = teams[1] if t == teams[0] else teams[0]
            lead = s.margin_start if t == g.home_id else -s.margin_start
            rows.append((
                g.game_id, n, t, o, int(t == g.home_id), s.period, s.start, s.end,
                _key(s.lineups[t]), _key(s.lineups[o]), s.poss[t], s.poss[o], s.pts[t], s.pts[o],
                lead, int(s.ok), *(s.stats[t][c] for c in _STINT_STATS),
            ))
    cols = ("game_id, stint_num, team_id, opp_id, is_home, period, start_sec, end_sec, lineup_key, "
            "opp_lineup_key, poss, opp_poss, pts, opp_pts, margin_start, ok, " + ", ".join(_STINT_STATS))
    conn.executemany(
        f"INSERT INTO lineup_stints ({cols}) VALUES ({', '.join('?' * (16 + len(_STINT_STATS)))})", rows)
    prow = []
    for pid, p in g.players.items():
        if not p.get("team_id"):
            continue
        prow.append((g.game_id, pid, p["team_id"], round(p["seconds"], 1), int(p["starter"]),
                     p["on_court_poss"], p["opp_on_court_poss"], p["and1"],
                     *(p[c] for c in _PLAYER_ZONES)))
    conn.executemany(
        f"""INSERT INTO player_game_pbp (game_id, player_id, team_id, seconds, starter, on_poss, opp_poss,
               and1, {", ".join(_PLAYER_ZONES)}) VALUES ({", ".join("?" * (8 + len(_PLAYER_ZONES)))})""",
        prow)
    for t in teams:
        conn.execute(
            "UPDATE team_game_stats SET possessions = ?, poss_source = 'pbp' WHERE game_id = ? AND team_id = ?",
            (g.team_poss[t], g.game_id, t))
    conn.execute(
        """INSERT OR REPLACE INTO pbp_games (game_id, events, bad_events, quality, stints, bad_stints,
               notes, source, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (g.game_id, g.events, g.bad_events, g.quality, len(g.stints), sum(not s.ok for s in g.stints),
         "; ".join(g.notes)[:1000] or None, source, now))


def _tip_and_date(actions):
    times = sorted(str(a["timeActual"]) for a in actions if a.get("timeActual"))
    if not times:
        return None, None
    tip = times[0].replace(".0Z", "Z")
    dt = datetime.fromisoformat(times[0].replace("Z", "+00:00"))
    return tip[:19] + "Z", dt.astimezone(ET).date().isoformat()


def store_archive_game(conn, g: GamePbp, actions: list, source: str) -> None:
    """Build game, team and player box-score rows from play-by-play, then store the stints."""
    now = utc_now_iso()
    tip, game_date = _tip_and_date(actions)
    sid = season_from_game_id(g.game_id)
    ensure_season(conn, sid, game_date)
    tricodes, names = {}, {}
    for a in actions:
        t, tri = a.get("teamId"), a.get("teamTricode")
        if t and tri:
            tricodes[int(float(t))] = tri
        p = a.get("personId")
        if p and a.get("playerNameI"):
            names[int(float(p))] = a["playerNameI"]
    for t in (g.home_id, g.away_id):
        conf, div = config.TEAM_GROUPS.get(t, (None, None))
        conn.execute(
            """INSERT INTO teams (team_id, abbr, city, name, conference, division) VALUES (?, ?, '', '', ?, ?)
               ON CONFLICT(team_id) DO NOTHING""", (t, tricodes.get(t, str(t)), conf, div))
    hs, as_ = g.final_score
    conn.execute(
        """INSERT INTO games (game_id, season_id, season_type, game_date, tip_utc, home_team_id, away_team_id,
               status, home_score, away_score, periods, source, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'final', ?, ?, ?, ?, ?)
           ON CONFLICT(game_id) DO UPDATE SET status = 'final', home_score = excluded.home_score,
               away_score = excluded.away_score, periods = excluded.periods""",
        (g.game_id, sid, season_type(g.game_id), game_date, tip, g.home_id, g.away_id, hs, as_,
         g.periods, source, now))

    team_tot = {t: {c: 0 for c in _STINT_STATS} for t in (g.home_id, g.away_id)}
    for s in g.stints:
        for t in team_tot:
            for c in _STINT_STATS:
                team_tot[t][c] += s.stats[t][c]
    plus_minus = {}
    for s in g.stints:
        for t in (g.home_id, g.away_id):
            o = g.away_id if t == g.home_id else g.home_id
            for p in s.lineups[t]:
                plus_minus[p] = plus_minus.get(p, 0) + s.pts[t] - s.pts[o]
    for t, is_home in ((g.home_id, 1), (g.away_id, 0)):
        tt = team_tot[t]
        mins = sum(p["seconds"] for p in g.players.values() if p["team_id"] == t) / 60
        conn.execute(
            """INSERT OR REPLACE INTO team_game_stats (game_id, team_id, is_home, minutes, pts, fgm, fga,
                   fg3m, fg3a, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf, pts_paint, pts_second_chance,
                   pts_fast_break, pts_off_tov, possessions, source, fetched_at, poss_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pbp')""",
            (g.game_id, t, is_home, round(mins, 2), hs if is_home else as_, tt["fgm"], tt["fga"], tt["fg3m"],
             tt["fg3a"], tt["ftm"], tt["fta"], tt["oreb"], tt["dreb"], tt["ast"], tt["stl"], tt["blk"],
             tt["tov"], tt["pf"], tt["pts_paint"], tt["pts_2nd"], tt["pts_fb"], tt["pts_off_tov"],
             g.team_poss[t], source, now))
    for pid, p in g.players.items():
        if not p.get("team_id") or p["seconds"] <= 0:
            continue
        conn.execute(
            """INSERT INTO players (player_id, full_name, fetched_at) VALUES (?, ?, ?)
               ON CONFLICT(player_id) DO NOTHING""", (pid, names.get(pid, str(pid)), now))
        conn.execute(
            """INSERT OR REPLACE INTO player_game_stats (game_id, player_id, team_id, started, seconds, pts, fgm,
                   fga, fg3m, fg3a, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf, plus_minus, dnp_reason,
                   source, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
            (g.game_id, pid, p["team_id"], int(p["starter"]), int(round(p["seconds"])), p["pts"], p["fgm"],
             p["fga"], p["fg3m"], p["fg3a"], p["ftm"], p["fta"], p["oreb"], p["dreb"], p["ast"], p["stl"],
             p["blk"], p["tov"], p["pf"], plus_minus.get(pid), source, now))
    store_pbp(conn, g, source)
