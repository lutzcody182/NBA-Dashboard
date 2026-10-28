"""Daily leaderboard: every player line from recent game days, with the analytics behind it.

For each of the last DAYS game days of the current season (regular season
and playoffs) every player who got in gets:

  * the box score: minutes, points, rebounds, assists, steals, blocks,
    turnovers, threes, shooting
  * efficiency and role: true shooting %, effective FG %, usage %, game score
  * on-court impact that night: the team's points per 100 possessions minus
    the opponent's while he was on the floor (from lineup stints)
  * what changed: minutes and starts against his previous 10 games
  * against the pregame projection, when one was saved before tip-off
"""
import numpy as np
import pandas as pd

DAYS = 14


def _projections(conn, game_ids) -> pd.DataFrame:
    """The last projection saved before each game's tip-off."""
    if not game_ids:
        return pd.DataFrame()
    q = ",".join("?" * len(game_ids))
    df = pd.read_sql_query(
        f"""SELECT pp.game_id, pp.player_id, pp.minutes_mean AS proj_min, pp.pts_mean AS proj_pts,
                   pp.reb_mean AS proj_reb, pp.ast_mean AS proj_ast, pp.fg3m_mean AS proj_fg3m,
                   r.started_at, g.tip_utc
            FROM player_projections pp JOIN model_runs r ON r.run_id = pp.run_id
            JOIN games g ON g.game_id = pp.game_id
            WHERE pp.game_id IN ({q}) AND (g.tip_utc IS NULL OR r.started_at < g.tip_utc)""",
        conn, params=list(game_ids))
    if df.empty:
        return df
    df = df.sort_values("started_at").groupby(["game_id", "player_id"]).tail(1)
    return df.drop(columns=["started_at", "tip_utc"])


def _on_court(conn, game_ids) -> pd.DataFrame:
    if not game_ids:
        return pd.DataFrame()
    q = ",".join("?" * len(game_ids))
    st = pd.read_sql_query(
        f"""SELECT game_id, lineup_key, poss, opp_poss, pts, opp_pts FROM lineup_stints
            WHERE game_id IN ({q})""", conn, params=list(game_ids))
    if st.empty:
        return st
    st["player_id"] = st["lineup_key"].str.split("-")
    st = st.explode("player_id")
    st["player_id"] = st["player_id"].astype(int)
    g = st.groupby(["game_id", "player_id"])[["poss", "opp_poss", "pts", "opp_pts"]].sum().reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        g["on_net"] = 100 * g["pts"] / g["poss"] - 100 * g["opp_pts"] / g["opp_poss"]
    g["on_poss"] = (g["poss"] + g["opp_poss"]) / 2
    return g[["game_id", "player_id", "on_net", "on_poss"]]


def build(conn, season: str, logs: pd.DataFrame) -> dict:
    """logs: player_logs() output for the season (one row per player-game)."""
    if logs is None or logs.empty:
        return {"season": season, "days": []}
    df = logs.sort_values(["date", "game_id"]).copy()
    # previous 10 games, for what changed
    grp = df.groupby("player_id")
    df["l10_min"] = grp["min"].transform(lambda s: s.shift().rolling(10, min_periods=1).mean())
    df["l10_starts"] = grp["started"].transform(lambda s: s.shift().rolling(10, min_periods=1).sum())
    df["prior_games"] = grp.cumcount()
    days = sorted(df["date"].unique())[-DAYS:]
    d = df[df["date"].isin(days)].copy()
    tsa = d["fga"] + 0.44 * d["fta"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["ts"] = np.where(tsa > 0, d["pts"] / (2 * tsa), np.nan)
        d["efg"] = np.where(d["fga"] > 0, (d["fgm"] + 0.5 * d["fg3m"]) / d["fga"], np.nan)
        d["usg"] = np.where(d["usg_d"] > 0, d["usg_n"] / d["usg_d"], np.nan)
    d["gmsc"] = (d["pts"] + 0.4 * d["fgm"] - 0.7 * d["fga"] - 0.4 * (d["fta"] - d["ftm"]) + 0.7 * d["oreb"]
                 + 0.3 * d["dreb"] + d["stl"] + 0.7 * d["ast"] + 0.7 * d["blk"] - 0.4 * d["pf"] - d["tov"])
    d["pra"] = d["pts"] + d["reb"] + d["ast"]
    gids = sorted(d["game_id"].unique())
    oc = _on_court(conn, gids)
    if not oc.empty:
        d = d.merge(oc, on=["game_id", "player_id"], how="left")
    pr = _projections(conn, gids)
    if not pr.empty:
        d = d.merge(pr, on=["game_id", "player_id"], how="left")
    cols = ["game_id", "player_id", "team_id", "opp_id", "home", "started", "min", "pts", "reb", "oreb", "dreb",
            "ast", "stl", "blk", "tov", "pf", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "plus_minus",
            "ts", "efg", "usg", "gmsc", "pra", "on_net", "on_poss", "l10_min", "l10_starts", "prior_games",
            "proj_min", "proj_pts", "proj_reb", "proj_ast", "proj_fg3m"]
    for c in cols:
        if c not in d:
            d[c] = np.nan
    finals = dict(conn.execute(
        f"SELECT game_id, home_score || '-' || away_score FROM games WHERE game_id IN ({','.join('?' * len(gids))})",
        gids).fetchall()) if gids else {}
    out = []
    for day in reversed(days):
        x = d[d["date"] == day][cols].copy()
        for c in ("ts", "efg", "usg"):
            x[c] = pd.to_numeric(x[c], errors="coerce").round(3)
        for c in ("min", "gmsc", "on_net", "on_poss", "l10_min", "l10_starts", "proj_min", "proj_pts", "proj_reb",
                  "proj_ast", "proj_fg3m"):
            x[c] = pd.to_numeric(x[c], errors="coerce").round(1)
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.astype(object).where(pd.notna(x), None)
        rows = {"cols": list(x.columns), "rows": x.values.tolist()}
        out.append({"date": day, "games": int(x["game_id"].nunique()), "players": int(len(x)),
                    "has_projections": bool(x["proj_min"].notna().any()), "cols": rows,
                    "scores": {g: finals.get(g) for g in x["game_id"].unique()}})
    return {"season": season, "days": out}
