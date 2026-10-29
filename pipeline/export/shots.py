"""Shot maps from saved shot locations (shot_locations, ESPN play-by-play).

The half court is cut into CELL-foot squares. For every team, on offense and
on defense, and for every player with MIN_PLAYER_SHOTS or more attempts:
attempts and makes per square, plus the league's attempts and makes in the
same square so the page can show frequency and accuracy against average.
Shots beyond half court are dropped.
"""
import pandas as pd

CELL = 2.5
X_MAX, Y_MIN, Y_MAX = 50.0, -4.0, 41.75
MIN_PLAYER_SHOTS = 25


def _cells(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["x"] >= 0) & (df["x"] <= X_MAX) & (df["y"] >= Y_MIN) & (df["y"] <= Y_MAX)].copy()
    df["cx"] = (df["x"].clip(upper=X_MAX - 0.01) // CELL).astype(int)
    df["cy"] = ((df["y"] - Y_MIN) // CELL).astype(int)
    return df


def _pack(g: pd.DataFrame) -> dict:
    agg = g.groupby(["cx", "cy"]).agg(a=("made", "size"), m=("made", "sum")).reset_index()
    return {"cells": [[int(r.cx), int(r.cy), int(r.a), int(r.m)] for r in agg.itertuples()],
            "fga": int(len(g)), "fgm": int(g["made"].sum()),
            "zones": {z: [int((g["zone"] == z).sum()), int(g.loc[g["zone"] == z, "made"].sum())]
                      for z in ("rim", "paint", "mid", "c3", "ab3")}}


def build(conn, season: str) -> dict:
    df = pd.read_sql_query(
        """SELECT s.team_id, s.def_team_id, s.player_id, s.x, s.y, s.zone, s.points, s.made
           FROM shot_locations s JOIN games g ON g.game_id = s.game_id
           WHERE g.season_id = ? AND g.season_type IN ('regular', 'playoffs', 'play_in')""", conn, params=[season])
    games = conn.execute(
        """SELECT COUNT(DISTINCT s.game_id) FROM shot_locations s JOIN games g ON g.game_id = s.game_id
           WHERE g.season_id = ?""", (season,)).fetchone()[0]
    if df.empty:
        return {"season": season, "games": 0, "cell": CELL, "y_min": Y_MIN, "teams": {}, "players": {}}
    df = _cells(df)
    out = {"season": season, "games": int(games), "cell": CELL, "y_min": Y_MIN, "y_max": Y_MAX,
           "league": _pack(df), "teams": {}, "players": {}}
    for tid, g in df.groupby("team_id"):
        out["teams"].setdefault(str(int(tid)), {})["offense"] = _pack(g)
    for tid, g in df.groupby("def_team_id"):
        out["teams"].setdefault(str(int(tid)), {})["defense"] = _pack(g)
    for pid, g in df.groupby("player_id"):
        if len(g) >= MIN_PLAYER_SHOTS:
            p = _pack(g)
            p["team_id"] = int(g["team_id"].mode().iloc[0])
            out["players"][str(int(pid))] = p
    return out
