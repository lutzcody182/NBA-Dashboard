"""Assemble per-player-game data for the models.

One row per player per game he played, with:
  * his own box-score counts and shots by zone
  * what his team and the opponent did while he was on the floor
    (from lineup stints), which is what rebound %, assist %, rim
    protection and on/off numbers need
"""
import pandas as pd

TEAM_ON = ["poss", "pts", "fga", "fgm", "fta", "tov", "oreb", "dreb"]
OPP_ON = ["poss", "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "tov", "oreb", "dreb", "fga_rim", "fgm_rim"]


def load_games(conn, seasons) -> pd.DataFrame:
    q = ",".join("?" * len(seasons))
    return pd.read_sql_query(
        f"""SELECT game_id, season_id, game_date, home_team_id, away_team_id, home_score, away_score, periods
            FROM games WHERE status = 'final' AND season_type = 'regular' AND season_id IN ({q})""",
        conn, params=list(seasons))


def load_stints(conn, seasons) -> pd.DataFrame:
    q = ",".join("?" * len(seasons))
    df = pd.read_sql_query(
        f"""SELECT s.*, g.season_id, g.game_date FROM lineup_stints s JOIN games g ON g.game_id = s.game_id
            WHERE s.ok = 1 AND g.season_type = 'regular' AND g.season_id IN ({q})""",
        conn, params=list(seasons))
    return df


def on_court_totals(stints: pd.DataFrame) -> pd.DataFrame:
    """Sum team and opponent stats over the stints each player was on the floor."""
    if stints.empty:
        return pd.DataFrame()
    me = stints[["game_id", "stint_num", "team_id", "lineup_key", "end_sec", "start_sec"] + TEAM_ON]
    opp = stints[["game_id", "stint_num", "team_id"] + OPP_ON].rename(
        columns={c: f"opp_{c}" for c in OPP_ON}).rename(columns={"team_id": "opp_team"})
    m = me.merge(opp, on=["game_id", "stint_num"])
    m = m[m["team_id"] != m["opp_team"]]
    m["secs"] = m["end_sec"] - m["start_sec"]
    m["player_id"] = m["lineup_key"].str.split("-")
    m = m.explode("player_id")
    m["player_id"] = m["player_id"].astype(int)
    cols = ["secs"] + TEAM_ON + [f"opp_{c}" for c in OPP_ON]
    out = m.groupby(["game_id", "player_id"], as_index=False)[cols].sum()
    return out.rename(columns={c: f"on_{c}" for c in cols})


def player_games(conn, seasons) -> pd.DataFrame:
    """Every player-game in the given seasons, ready for the rating and minutes models."""
    q = ",".join("?" * len(seasons))
    box = pd.read_sql_query(
        f"""SELECT p.game_id, p.player_id, p.team_id, p.started, p.seconds, p.pts, p.fgm, p.fga, p.fg3m, p.fg3a,
                   p.ftm, p.fta, p.oreb, p.dreb, p.ast, p.stl, p.blk, p.tov, p.pf, p.plus_minus,
                   g.season_id, g.game_date,
                   CASE WHEN g.home_team_id = p.team_id THEN g.away_team_id ELSE g.home_team_id END AS opp_id,
                   CASE WHEN g.home_team_id = p.team_id THEN 1 ELSE 0 END AS home
            FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
            WHERE g.status = 'final' AND g.season_type = 'regular' AND p.seconds > 0 AND g.season_id IN ({q})""",
        conn, params=list(seasons))
    zones = pd.read_sql_query(
        f"""SELECT z.* FROM player_game_pbp z JOIN games g ON g.game_id = z.game_id
            WHERE g.season_type = 'regular' AND g.season_id IN ({q})""", conn, params=list(seasons))
    zones = zones.drop(columns=["team_id", "seconds", "starter", "on_poss", "opp_poss"])
    df = box.merge(zones, on=["game_id", "player_id"], how="left")
    oc = on_court_totals(load_stints(conn, seasons))
    if not oc.empty:
        df = df.merge(oc, on=["game_id", "player_id"], how="left")
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(0)
    df["min"] = df["seconds"] / 60
    df["reb"] = df["oreb"] + df["dreb"]
    df["tsa"] = df["fga"] + 0.44 * df["fta"]
    df["fg2a"] = df["fga"] - df["fg3a"]
    df["fg2m"] = df["fgm"] - df["fg3m"]
    return df.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def rest_days(df: pd.DataFrame) -> pd.Series:
    """Days since the player's team last played (1 = back-to-back)."""
    team_dates = (df[["team_id", "game_date"]].drop_duplicates().sort_values(["team_id", "game_date"]))
    team_dates["prev"] = team_dates.groupby("team_id")["game_date"].shift()
    d = (pd.to_datetime(team_dates["game_date"]) - pd.to_datetime(team_dates["prev"])).dt.days
    team_dates["rest"] = d
    return df.merge(team_dates[["team_id", "game_date", "rest"]], on=["team_id", "game_date"], how="left")["rest"]
