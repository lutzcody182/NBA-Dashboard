"""Pregame simulations for upcoming games, and player projections from them.

For each game:
  * team rates so far this season (team_rates), home court and schedule
    effects (schedule), and who is expected to play: the minutes projection
    against the lineups the team's rates were built on, valued with the
    player ratings
  * N_SIMS_LIVE simulated games -> win chance, score, spread, total, ranges
  * each team's expected shots, free throws, rebounds, assists and turnovers
    from the same rates, split among its players by expected minutes and
    the usage model (usage.py), so absences move shots to teammates
"""
import json
import logging
import zlib

import numpy as np
import pandas as pd

from ..util import utc_now_iso
from . import game_model as GM
from . import schedule as S
from . import sim
from . import team_rates as TR
from . import usage as U
from .ratings import model_season

log = logging.getLogger(__name__)
HIST_EDGES = list(range(-40, 42, 2))


def usage_calibration(conn, season_m: str) -> dict:
    key = f"usage:{season_m}"
    row = conn.execute("SELECT payload FROM model_calibration WHERE key = ?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    cal = U.calibrate(conn, season_m)
    conn.execute("INSERT OR REPLACE INTO model_calibration (key, payload, created_at) VALUES (?, ?, ?)",
                 (key, json.dumps(cal), utc_now_iso()))
    conn.commit()
    return cal


def team_counts(r: dict, shift: float, pace: float) -> dict:
    """Expected per-game counts for one offense from its matchup rates."""
    q = sim.trip_rate(r)
    shares, makes = sim._arrays(r)
    makes = sim._inv(sim._logit(makes) + shift)
    pbar = float(shares @ makes)
    E = (1 - r["tov"]) / (1 - (1 - q) * (1 - pbar) * r["oreb"])
    fga = (1 - q) * E * pace
    three_share = r["share_c3"] + r["share_ab3"]
    return {"fga": fga, "fg3a": fga * three_share, "fta": r["fta"] * pace, "fgm": fga * pbar,
            "oreb": fga * (1 - pbar) * r["oreb"], "tov": r["tov"] * pace, "miss": fga * (1 - pbar)}


def _histogram(values) -> list:
    counts, _ = np.histogram(np.clip(values, HIST_EDGES[0], HIST_EDGES[-1] - 0.01), bins=HIST_EDGES)
    return [round(float(c) / len(values), 4) for c in counts]


def build(conn, season: str, games: list, proj: dict, ratings_payload: dict) -> dict:
    """Simulations and player projections for `games` (rows of the games table)."""
    m = model_season(conn, season)
    if m is None or not games:
        return {"games": [], "note": "needs a finished season of play-by-play"}
    cal = GM.get_calibration(conn, m)
    ucal = usage_calibration(conn, m).get("stats", {})
    prev = GM.prev_season(season)
    cur, prev_tg = TR.team_games(conn, [season]), TR.team_games(conn, [prev])
    teams = sorted({g["home_team_id"] for g in games} | {g["away_team_id"] for g in games})
    est = TR.estimates(cur, prev_tg, cal["team_rate_fits"], teams)
    league_ppp = float(cur.pts.sum() / cur.poss.sum()) if len(cur) >= 60 else float(prev_tg.pts.sum() / prev_tg.poss.sum())
    all_games = pd.read_sql_query(
        "SELECT game_id, season_id, game_date, home_team_id, away_team_id FROM games WHERE season_type = 'regular'",
        conn)
    sched = GM._sched_lookup(S.team_schedule(all_games))
    ctx = GM.Context(est, cal["schedule"], sched, cal["lead_effect"], league_ppp, cal["form_sd"], cal["pace_sd"],
                     cal.get("strength_scale", 1.0))

    ratings = {p["player_id"]: (p["offense"]["value"], p["defense"]["value"], p["poss"] + p.get("poss_prev", 0))
               for p in (ratings_payload or {}).get("players", [])}
    repl = GM.replacement_value(ratings)
    stats = {p["player_id"]: p.get("stats", {}) for p in (ratings_payload or {}).get("players", [])}
    hist = GM.RosterHistory(conn, season)
    st_u, pg_u = U.load(conn, [prev, season])
    weights = {s: U.weights(st_u, pg_u, s, ucal.get(s, {}).get("k", 50)) for s in U.STATS}
    names = dict(conn.execute("SELECT player_id, full_name FROM players").fetchall())
    proj_by_game = {g["game_id"]: g for g in (proj or {}).get("games", [])}
    lg_ts = np.nanmean([v["scoring_eff"][5] for v in stats.values() if v.get("scoring_eff")]) / 100 \
        if stats else 0.57
    lg_3p = np.nanmean([v["three_pt"][5] for v in stats.values() if v.get("three_pt")]) / 100 if stats else 0.36

    out = []
    for g in games:
        if g.get("season_type") not in ("regular", "playoffs", "play_in", "cup_final"):
            continue
        h, a = g["home_team_id"], g["away_team_id"]
        if h not in est["teams"] or a not in est["teams"]:
            continue
        pj = proj_by_game.get(g["game_id"], {}).get("teams", {})
        minutes, roster = {}, {}
        for t in (h, a):
            players = pj.get(str(t), {}).get("players", [])
            mins = {p["player_id"]: (p["expected_min"] if p.get("expected_min") is not None else p["min"])
                    for p in players}
            mins = {p: v for p, v in mins.items() if v and v > 0}
            minutes[t] = mins
            base = hist.base(t, g["game_date"], ratings, repl)
            roster[t] = GM.roster_effect(mins, base, ratings, repl, cal.get("roster_scale", 0.0))
        res = ctx.predict(g["game_id"], h, a, bool(g.get("neutral_site")), roster=roster, keep=True,
                          seed=zlib.crc32(g["game_id"].encode()))
        x = ctx.inputs(g["game_id"], h, a, bool(g.get("neutral_site")), roster)
        counts = {h: team_counts(x["rates"][0], x["shift"][0], x["pace"]),
                  a: team_counts(x["rates"][1], x["shift"][1], x["pace"])}
        pts_mean = {h: res["home_pts_mean"], a: res["away_pts_mean"]}
        sq = TR.shot_quality(est)["teams"]
        team_out = {}
        for t, o in ((h, a), (a, h)):
            c = counts[t]
            opp = counts[o]
            c["dreb"] = opp["miss"] * (1 - x["rates"][1 if t == h else 0]["oreb"])
            c["ast"] = c["fgm"] * est["teams"][t]["off"]["ast"]
            c["pts"] = pts_mean[t]
            mins = minutes[t]
            players = []
            if mins:
                share = {s: U.expected_shares(mins, weights[s], ucal.get(s, {}).get("g", 1.0)) for s in U.STATS}
                raw = {s: {p: c[s] * mins[p] / 48 * share[s][p] for p in mins} for s in U.STATS}
                for s in U.STATS:          # the team's whole total is handed out
                    tot = sum(raw[s].values())
                    if tot > 0:
                        raw[s] = {p: v * c[s] / tot for p, v in raw[s].items()}
                ts = {p: (stats.get(p, {}).get("scoring_eff") or [lg_ts * 100])[0] / 100 for p in mins}
                p3 = {p: (stats.get(p, {}).get("three_pt") or [lg_3p * 100])[0] / 100 for p in mins}
                pts = {p: 2 * ts[p] * (raw["fga"][p] + 0.44 * raw["fta"][p]) for p in mins}
                tot = sum(pts.values())
                pts = {p: v * c["pts"] / tot for p, v in pts.items()} if tot else pts
                for p in sorted(mins, key=lambda p: -mins[p]):
                    if mins[p] < 3:
                        continue
                    players.append({
                        "player_id": int(p), "name": names.get(p, str(p)), "min": round(mins[p], 1),
                        "pts": round(pts[p], 1), "reb": round(raw["oreb"][p] + raw["dreb"][p], 1),
                        "ast": round(raw["ast"][p], 1), "fg3m": round(raw["fg3a"][p] * p3[p], 1),
                        "fga": round(raw["fga"][p], 1), "fg3a": round(raw["fg3a"][p], 1),
                        "fta": round(raw["fta"][p], 1), "tov": round(raw["tov"][p], 1),
                        "fga_share": round(share["fga"][p], 3), "rated": p in ratings,
                    })
            team_out[str(t)] = {
                "pts": round(pts_mean[t], 1),
                "expected": {k: round(v, 1) for k, v in c.items() if k != "miss"},
                "roster_effect": {"offense": round(roster[t][0], 2), "defense": round(roster[t][1], 2)},
                "shot_quality": {k: round(v, 4) for k, v in sq[t]["off"].items()},
                "players": players,
            }
        out.append({
            "game_id": g["game_id"], "game_date": g["game_date"], "tip_utc": g.get("tip_utc"),
            "home_team_id": h, "away_team_id": a,
            "home_win": round(res["home_win"], 4), "margin": round(res["margin_mean"], 1),
            "total": round(res["total_mean"], 1), "home_pts": round(res["home_pts_mean"], 1),
            "away_pts": round(res["away_pts_mean"], 1), "overtime": round(res["overtime"], 3),
            "margin_q": {k: round(v, 1) for k, v in res["margin_q"].items()},
            "total_q": {k: round(v, 1) for k, v in res["total_q"].items()},
            "margin_hist": _histogram(res["_margin"]),
            "inputs": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["inputs"].items()},
            "teams": team_out, "n_sims": res["n_sims"],
        })
    return {
        "generated_at": utc_now_iso(), "season": season, "calibration_season": m,
        "hist_edges": HIST_EDGES, "games": out,
        "model": {
            "report": cal["report"], "form_sd": cal["form_sd"], "pace_sd": cal["pace_sd"],
            "lead_effect": cal["lead_effect"], "lead_slope_ppp": cal["lead_slope_ppp"],
            "roster_scale": cal.get("roster_scale"), "roster_grid": cal.get("roster_grid"),
            "strength_scale": cal.get("strength_scale"),
            "schedule": cal["schedule"], "usage": {"stats": ucal},
            "team_rates": {k: {"k": v["k"], "w": v["w"], "improvement_vs_raw": v["improvement_vs_raw"]}
                           for k, v in cal["team_rate_fits"].items()},
            "league": {k: round(v, 4) for k, v in est["league"].items()},
            "league_ppp": league_ppp,
        },
    }


def team_shot_quality(conn, season: str) -> dict:
    """Shot quality and shot-making for every team, for the Teams tab."""
    m = model_season(conn, season)
    if m is None:
        return {}
    cal = GM.get_calibration(conn, m)
    cur, prev_tg = TR.team_games(conn, [season]), TR.team_games(conn, [GM.prev_season(season)])
    teams = sorted(set(cur.team_id) | set(prev_tg.team_id))
    est = TR.estimates(cur, prev_tg, cal["team_rate_fits"], teams)
    sq = TR.shot_quality(est)
    raw = cur.groupby("team_id")[["fga", "fgm", "fg3m"]].sum()
    out = {}
    for t, r in sq["teams"].items():
        rr = raw.loc[t] if t in raw.index else None
        out[str(t)] = {
            "off": {k: round(v, 4) for k, v in r["off"].items()},
            "def": {k: round(v, 4) for k, v in r["def"].items()},
            "raw_efg": round(float((rr.fgm + 0.5 * rr.fg3m) / rr.fga), 4) if rr is not None and rr.fga else None,
            "pace": round(est["teams"][t]["off"]["pace"], 1),
            "tov": round(est["teams"][t]["off"]["tov"], 4), "opp_tov": round(est["teams"][t]["def"]["tov"], 4),
            "oreb": round(est["teams"][t]["off"]["oreb"], 4), "opp_oreb": round(est["teams"][t]["def"]["oreb"], 4),
        }
    return {"season": season, "league_efg": round(sq["league_efg"], 4), "teams": out,
            "league": {k: round(v, 4) for k, v in est["league"].items()}}
