"""Write the JSON files the dashboard reads (everything under data/).

Large tables are written in a compact column format:
  {"cols": ["a", "b"], "rows": [[1, 2], [3, 4]]}
"""
import json
import logging
import math
import traceback
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .. import config
from .. import db as dbmod
from ..ingest.update import today_et
from ..stats.formulas import assist_pct_parts, rebound_pct_parts, usage_parts
from ..stats.team_ratings import load_team_games, team_table
from ..util import utc_now_iso


def _clean(obj):
    """Make values JSON-safe: NaN/inf -> None, numpy types -> Python, round floats."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 4)
    return obj


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), separators=(",", ":")))


def _columnar(df: pd.DataFrame) -> dict:
    df = df.astype(object).where(pd.notna(df), None)
    return {"cols": list(df.columns), "rows": df.values.tolist()}


def current_season(conn):
    row = conn.execute(
        "SELECT season_id FROM games WHERE season_type IN ('preseason','regular') "
        "ORDER BY season_id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def previous_season(season_id: str) -> str:
    start = int(season_id[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


# -- player game logs -------------------------------------------------------

PLAYER_LOG_SQL = """
SELECT g.game_id, g.game_date AS date, p.player_id, p.team_id, o.team_id AS opp_id,
       tm.is_home AS home, p.started, p.seconds,
       p.pts, p.oreb, p.dreb, p.ast, p.stl, p.blk, p.tov, p.pf,
       p.fgm, p.fga, p.fg3m, p.fg3a, p.ftm, p.fta, p.plus_minus,
       tm.minutes AS tm_min, tm.fgm AS tm_fgm, tm.fga AS tm_fga, tm.fta AS tm_fta, tm.tov AS tm_tov,
       tm.oreb + tm.dreb AS tm_reb, o.oreb + o.dreb AS opp_reb,
       tm.oreb AS tm_oreb, tm.dreb AS tm_dreb, o.oreb AS opp_oreb, o.dreb AS opp_dreb
FROM player_game_stats p
JOIN games g ON g.game_id = p.game_id
JOIN team_game_stats tm ON tm.game_id = p.game_id AND tm.team_id = p.team_id
JOIN team_game_stats o ON o.game_id = p.game_id AND o.team_id <> p.team_id
WHERE g.status = 'final' AND g.season_id = ? AND g.season_type IN ({types}) AND p.seconds > 0
ORDER BY g.game_date, g.game_id
"""


def player_logs(conn, season_id: str) -> pd.DataFrame:
    types = ",".join(f"'{t}'" for t in config.STAT_SEASON_TYPES)
    df = pd.read_sql_query(PLAYER_LOG_SQL.format(types=types), conn, params=(season_id,))
    if df.empty:
        return df
    mins = df["seconds"] / 60
    df["min"] = mins.round(2)
    df["reb"] = df["oreb"] + df["dreb"]
    usg = [usage_parts(minutes=m, fga=a, fta=b, tov=c, tm_minutes=tmm, tm_fga=d, tm_fta=e, tm_tov=f)
           for m, a, b, c, tmm, d, e, f in zip(mins, df.fga, df.fta, df.tov, df.tm_min, df.tm_fga, df.tm_fta, df.tm_tov)]
    df["usg_n"], df["usg_d"] = zip(*usg)
    df["ast_d"] = [assist_pct_parts(minutes=m, ast=a, fgm=f, tm_minutes=tmm, tm_fgm=tf)[1]
                   for m, a, f, tmm, tf in zip(mins, df.ast, df.fgm, df.tm_min, df.tm_fgm)]
    df["reb_d"] = [rebound_pct_parts(minutes=m, reb=r, tm_minutes=tmm, tm_reb=tr, opp_reb=orr)[1]
                   for m, r, tmm, tr, orr in zip(mins, df.reb, df.tm_min, df.tm_reb, df.opp_reb)]
    df["oreb_d"] = [rebound_pct_parts(minutes=m, reb=0, tm_minutes=tmm, tm_reb=a, opp_reb=b)[1]
                    for m, tmm, a, b in zip(mins, df.tm_min, df.tm_oreb, df.opp_dreb)]
    df["dreb_d"] = [rebound_pct_parts(minutes=m, reb=0, tm_minutes=tmm, tm_reb=a, opp_reb=b)[1]
                    for m, tmm, a, b in zip(mins, df.tm_min, df.tm_dreb, df.opp_oreb)]
    tens = (df[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)
    df["dd"] = (tens >= 2).astype(int)
    df["td"] = (tens >= 3).astype(int)
    keep = ["game_id", "date", "player_id", "team_id", "opp_id", "home", "started", "min",
            "pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
            "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "plus_minus",
            "usg_n", "usg_d", "ast_d", "reb_d", "oreb_d", "dreb_d", "dd", "td"]
    out = df[keep].copy()
    for c in ("usg_n", "usg_d", "ast_d", "reb_d", "oreb_d", "dreb_d"):
        out[c] = out[c].round(3)
    return out


# -- rosters ------------------------------------------------------------------

def rosters(conn, season_id: str) -> dict:
    """Each team's roster from its most recent box score this season.

    Status labels:
      confirmed  - listed in a regular-season box score
      estimated  - only preseason box scores so far (training-camp players
                   may since have been cut)
    """
    out = {}
    teams = conn.execute("SELECT team_id FROM teams").fetchall()
    for (team_id,) in teams:
        if team_id not in config.NBA_TEAM_IDS:
            continue
        g = conn.execute(
            """SELECT g.game_id, g.game_date, g.season_type FROM games g
               JOIN team_game_stats t ON t.game_id = g.game_id AND t.team_id = ?
               WHERE g.season_id = ? AND g.status = 'final'
               ORDER BY (g.season_type = 'preseason'), g.game_date DESC, g.game_id DESC LIMIT 1""",
            (team_id, season_id),
        ).fetchone()
        if not g:
            out[str(team_id)] = {"data_status": "missing", "as_of": None, "players": []}
            continue
        players = conn.execute(
            """SELECT p.player_id, pl.full_name, pl.position, s.jersey,
                      p.seconds, p.started, p.dnp_reason
               FROM player_game_stats p
               JOIN players pl ON pl.player_id = p.player_id
               LEFT JOIN roster_stints s ON s.player_id = p.player_id AND s.team_id = p.team_id
                                         AND s.end_date IS NULL
               WHERE p.game_id = ? AND p.team_id = ?
               ORDER BY p.started DESC, p.seconds DESC""",
            (g["game_id"], team_id),
        ).fetchall()
        out[str(team_id)] = {
            "data_status": "confirmed" if g["season_type"] != "preseason" else "estimated",
            "as_of": g["game_date"],
            "source_game": g["game_id"],
            "players": [
                {"player_id": r["player_id"], "name": r["full_name"], "pos": r["position"],
                 "jersey": r["jersey"], "last_min": round(r["seconds"] / 60, 1),
                 "started": bool(r["started"]), "dnp": r["dnp_reason"],
                 "still_on_team": True}
                for r in players
            ],
        }
        # Flag anyone who has since moved (their open stint is elsewhere).
        for p in out[str(team_id)]["players"]:
            cur = conn.execute(
                "SELECT team_id FROM roster_stints WHERE player_id = ? AND end_date IS NULL",
                (p["player_id"],),
            ).fetchone()
            p["still_on_team"] = bool(cur) and cur[0] == team_id
    return out


# -- schedule -----------------------------------------------------------------

def games_around_today(conn) -> dict:
    today = today_et()
    days = {"yesterday": today - timedelta(days=1), "today": today, "tomorrow": today + timedelta(days=1)}
    out = {}
    for label, d in days.items():
        rows = conn.execute(
            """SELECT g.game_id, g.game_date, g.tip_utc, g.status, g.season_type,
                      g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.periods, g.arena
               FROM games g WHERE g.game_date = ? ORDER BY g.tip_utc, g.game_id""",
            (d.isoformat(),),
        ).fetchall()
        out[label] = {"date": d.isoformat(), "games": [dict(r) for r in rows]}
    return out


def team_records(conn, season_id: str) -> dict:
    rows = conn.execute(
        """SELECT t.team_id,
                  SUM(CASE WHEN (g.home_team_id = t.team_id AND g.home_score > g.away_score)
                             OR (g.away_team_id = t.team_id AND g.away_score > g.home_score) THEN 1 ELSE 0 END) AS w,
                  SUM(CASE WHEN (g.home_team_id = t.team_id AND g.home_score < g.away_score)
                             OR (g.away_team_id = t.team_id AND g.away_score < g.home_score) THEN 1 ELSE 0 END) AS l
           FROM teams t JOIN games g ON (g.home_team_id = t.team_id OR g.away_team_id = t.team_id)
           WHERE g.season_id = ? AND g.status = 'final' AND g.season_type = 'regular'
           GROUP BY t.team_id""",
        (season_id,),
    ).fetchall()
    return {str(r["team_id"]): f'{r["w"]}-{r["l"]}' for r in rows}


# -- main ---------------------------------------------------------------------

def export_all(conn, out_dir: Path = config.DATA_DIR) -> dict:
    season = current_season(conn)
    generated = utc_now_iso()
    freshness = [dict(r) for r in conn.execute("SELECT * FROM data_freshness ORDER BY dataset")]
    teams = [dict(r) for r in conn.execute(
        "SELECT team_id, abbr, city, name, conference, division FROM teams ORDER BY abbr")]
    teams = [t for t in teams if t["team_id"] in config.NBA_TEAM_IDS]

    summary = {"season": season, "files": []}
    if season is None:
        _write(out_dir / "meta.json", {"generated_at": generated, "season": None,
                                       "freshness": freshness, "teams": teams})
        return summary
    prev = previous_season(season)

    # Players seen in either season (for names in the leader tables). Current team comes
    # from roster history when known, otherwise from his most recent game.
    players = {
        str(r["player_id"]): {"name": r["full_name"], "pos": r["position"], "team_id": r["team_id"]}
        for r in conn.execute(
            """SELECT pl.player_id, pl.full_name, pl.position,
                      COALESCE(s.team_id, (SELECT p.team_id FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
                                           WHERE p.player_id = pl.player_id ORDER BY g.game_date DESC LIMIT 1)) AS team_id
               FROM players pl LEFT JOIN roster_stints s
                 ON s.player_id = pl.player_id AND s.end_date IS NULL"""
        )
    }

    seasons_out = {}
    team_tables = {}
    prior_lambda = None
    for sid in (prev, season):
        logs = player_logs(conn, sid)
        tg = load_team_games(conn, sid)
        table = team_table(tg, prior_lambda=prior_lambda)
        poss_src = conn.execute(
            """SELECT SUM(CASE WHEN t.poss_source = 'pbp' THEN 1 ELSE 0 END), COUNT(*) FROM team_game_stats t
               JOIN games g ON g.game_id = t.game_id WHERE g.season_id = ? AND g.season_type = 'regular'""",
            (sid,)).fetchone()
        if poss_src and poss_src[1]:
            table["model"]["possessions_method"] = (
                f"counted from play-by-play in {poss_src[0] or 0} of {poss_src[1]} team games; "
                "box-score estimate for the rest")
        team_tables[sid] = table
        if table["model"].get("lambda_source", "").startswith("walk-forward"):
            prior_lambda = table["model"]["lambda_possessions"]
        if not logs.empty:
            _write(out_dir / "players" / f"gamelog_{sid}.json", _columnar(logs))
            summary["files"].append(f"players/gamelog_{sid}.json")
        _write(out_dir / "teams" / f"ratings_{sid}.json", table)
        summary["files"].append(f"teams/ratings_{sid}.json")
        if not tg.empty:
            trend = tg[["game_id", "game_date", "team_id", "opp_id", "is_home", "win", "game_num",
                        "pts", "opp_pts", "poss", "ortg", "drtg", "pace"]].copy()
            for c in ("poss", "ortg", "drtg", "pace"):
                trend[c] = trend[c].round(2)
            _write(out_dir / "teams" / f"gamelog_{sid}.json", _columnar(trend))
            summary["files"].append(f"teams/gamelog_{sid}.json")
        seasons_out[sid] = {
            "player_games": int(len(logs)),
            "team_games": int(len(tg) // 2),
        }

    phase2 = export_phase2(conn, out_dir, season, prev, team_tables)
    summary["files"] += phase2.pop("files")
    summary["phase2"] = phase2

    _write(out_dir / "rosters.json", {"season": season, "teams": rosters(conn, season)})
    _write(out_dir / "players" / "index.json", players)
    _write(out_dir / "schedule.json", {
        "season": season, "records": team_records(conn, season), **games_around_today(conn)})
    _write(out_dir / "meta.json", {
        "generated_at": generated,
        "season": season,
        "previous_season": prev,
        "seasons": seasons_out,
        "freshness": freshness,
        "teams": teams,
        "phase2": phase2,
        "notes": [
            "Possessions are counted from play-by-play when it is available, and estimated from the box score otherwise.",
            "Rosters come from each team's latest game-day box score. Before a team's first regular-season game they are labeled estimated.",
            "Player ratings are regularized plus-minus with a box-score prior, and every setting is chosen by testing on games the model had not seen.",
            "Player tracking stats (touches, potential assists, rebound chances, matchups) are not used.",
        ],
    })
    summary["files"] += ["rosters.json", "players/index.json", "schedule.json", "meta.json"]
    return summary


# -- Phase 2: ratings, lineups, projections -------------------------------------

log = logging.getLogger(__name__)


def export_phase2(conn, out_dir: Path, season: str, prev: str, team_tables: dict) -> dict:
    """Player ratings, lineups and game-day projections. A failure here never blocks Phase 1 files."""
    from ..injuries.ingest import play_rates
    from ..models import lineups as L
    from ..models import ratings as R
    from ..models.features import load_stints
    from ..models.projections import project_games

    out = {"files": [], "errors": []}
    for sid in (prev, season):
        try:
            r = R.compute(conn, sid, team_tables.get(sid))
            if not r["players"]:
                continue
            st = load_stints(conn, [sid])
            oo = L.on_off(st)
            for p in r["players"]:
                p["on_off"] = oo.get(str(p["player_id"]))
            _write(out_dir / "players" / f"ratings_{sid}.json", r)
            net = {p["player_id"]: p["overall"]["value"] for p in r["players"]}
            _write(out_dir / "teams" / f"lineups_{sid}.json", {
                "season": sid, "lineups": L.lineups(st, net), "score_state": L.score_state_pace(st),
                "starting": L.starting_lineups(conn, sid)})
            out["files"] += [f"players/ratings_{sid}.json", f"teams/lineups_{sid}.json"]
            out.setdefault("ratings_seasons", []).append(sid)
        except Exception as exc:
            log.exception("Ratings for %s failed", sid)
            out["errors"].append(f"ratings {sid}: {exc}")
    dbmod.mark_attempt(conn, "ratings", not any(e.startswith("ratings") for e in out["errors"]),
                       ", ".join(out.get("ratings_seasons", [])) or "no seasons rated")

    try:
        today = today_et()
        days = [today.isoformat(), (today + timedelta(days=1)).isoformat()]
        games = [dict(r) for r in conn.execute(
            f"""SELECT game_id, game_date, tip_utc, home_team_id, away_team_id, status, season_type FROM games
                WHERE game_date IN ({",".join("?" * len(days))}) AND status IN ('scheduled', 'live')
                ORDER BY tip_utc""", days)]
        rates = play_rates(conn)
        proj = project_games(conn, season, games, rates)
        proj["generated_at"] = utc_now_iso()
        _write(out_dir / "projections.json", proj)
        out["files"].append("projections.json")
        save_pregame_snapshot(conn, proj)
        dbmod.mark_attempt(conn, "projections", True, f"{len(proj['games'])} games projected")
    except Exception as exc:
        log.exception("Projections failed")
        out["errors"].append(f"projections: {exc}")
        dbmod.mark_attempt(conn, "projections", False, f"{exc}; {traceback.format_exc(limit=1)}")
    return out


def save_pregame_snapshot(conn, proj: dict) -> None:
    """Keep what the model said before each game, for honest backtests later (Phase 6)."""
    if not proj.get("games"):
        return
    now = utc_now_iso()
    rep = proj.get("injury_report") or {}
    cur = conn.execute(
        """INSERT INTO model_runs (started_at, finished_at, run_type, model_version, data_cutoff, status, notes)
           VALUES (?, ?, 'pregame', 'minutes-1', ?, 'ok', ?)""",
        (now, now, now, f"injury report {rep.get('report_time_et')}"))
    run_id = cur.lastrowid
    for g in proj["games"]:
        for team_id, t in g["teams"].items():
            for p in t["players"]:
                conn.execute(
                    """INSERT OR REPLACE INTO player_projections (run_id, game_id, player_id, availability, play_prob,
                           minutes_mean, minutes_p10, minutes_p90) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, g["game_id"], p["player_id"], p["status"],
                     0.0 if p["status"] == "out" else (p["appear"] or {}).get("p", 0.0),
                     p["min"], p["p10"], p["p90"]))
            for sc in t["scenarios"]:
                conn.execute(
                    """INSERT OR REPLACE INTO injury_scenarios (run_id, game_id, team_id, scenario_key, probability,
                           minutes_json) VALUES (?, ?, ?, ?, ?, ?)""",
                    (run_id, g["game_id"], int(team_id), "-".join(str(x) for x in sc["sitting"]) or "none",
                     sc["probability"], json.dumps(sc["minutes"])))
    conn.commit()
