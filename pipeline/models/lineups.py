"""Lineups and on/off splits from play-by-play stints.

Five-man lineups: a lineup's raw net rating is mostly noise until it has
played a lot together. Its expected net rating is the sum of its five
players' ratings; 'synergy' is how far it beats that, shrunk by how much
lineups genuinely differ from the sum of their parts (estimated from all
lineups in the league, normal-normal empirical Bayes).

On/off: team net rating and pace with each player on the floor and off it.
Raw, with possessions shown, because on/off mixes the player with who he
plays alongside (the overall rating already adjusts for that).
"""
import numpy as np
import pandas as pd

MIN_LINEUP_POSS = 50


def _team_view(st: pd.DataFrame) -> pd.DataFrame:
    """One row per stint per team with offense and defense numbers side by side."""
    opp = st[["game_id", "stint_num", "team_id", "poss", "pts"]].rename(
        columns={"team_id": "opp_team", "poss": "opp_poss_x", "pts": "opp_pts_x"})
    m = st.merge(opp, on=["game_id", "stint_num"])
    m = m[m["team_id"] != m["opp_team"]]
    m["secs"] = m["end_sec"] - m["start_sec"]
    return m


def lineups(st: pd.DataFrame, ratings: dict, top_n: int = 8) -> dict:
    """Most-used five-man lineups per team, with shrunk synergy."""
    if st.empty:
        return {"teams": {}, "tau": None}
    m = _team_view(st)
    g = m.groupby(["team_id", "lineup_key"]).agg(
        poss=("poss", "sum"), opp_poss=("opp_poss", "sum"), pts=("pts", "sum"), opp_pts=("opp_pts", "sum"),
        secs=("secs", "sum")).reset_index()
    g["ortg"] = 100 * g["pts"] / g["poss"].replace(0, np.nan)
    g["drtg"] = 100 * g["opp_pts"] / g["opp_poss"].replace(0, np.nan)
    g["net"] = g["ortg"] - g["drtg"]
    g["pace"] = 48 * 60 * (g["poss"] + g["opp_poss"]) / 2 / g["secs"].replace(0, np.nan)

    def expected(key):
        vals = [ratings.get(int(p)) for p in key.split("-")]
        return float(sum(v for v in vals if v is not None)) if vals else 0.0

    g["expected"] = g["lineup_key"].map(expected)
    g["n"] = (g["poss"] + g["opp_poss"]) / 2
    use = g[g["n"] >= MIN_LINEUP_POSS].dropna(subset=["net"])
    # Noise in a lineup's net rating: a per-100 rating over n possessions has variance
    # about v / n, with v measured from every stint (and doubled for offense minus defense).
    ok = st["poss"] > 0
    y = 100 * st.loc[ok, "pts"] / st.loc[ok, "poss"]
    mu = np.average(y, weights=st.loc[ok, "poss"])
    s2 = 2 * float(np.mean(st.loc[ok, "poss"] * (y - mu) ** 2))
    if len(use) > 20:
        resid = use["net"] - use["expected"]
        noise = (s2 / use["n"]).mean()
        tau2 = max(0.0, float(np.average(resid ** 2, weights=use["n"]) - noise))
    else:
        tau2 = 0.0
    g["synergy_raw"] = g["net"] - g["expected"]
    g["synergy"] = g["synergy_raw"] * tau2 / (tau2 + s2 / g["n"].clip(lower=1)) if tau2 > 0 else 0.0
    g["net_shrunk"] = g["expected"] + g["synergy"]
    out = {}
    for t, tg in g.sort_values("secs", ascending=False).groupby("team_id"):
        rows = []
        for r in tg.head(top_n).itertuples():
            rows.append({"players": [int(x) for x in r.lineup_key.split("-")], "minutes": r.secs / 60,
                         "poss": r.n, "ortg": r.ortg, "drtg": r.drtg, "net": r.net, "pace": r.pace, "expected": r.expected, "synergy": r.synergy,
                         "net_shrunk": r.net_shrunk})
        out[str(int(t))] = rows
    return {"teams": out, "tau": float(np.sqrt(tau2)), "noise_per_poss": s2, "min_poss": MIN_LINEUP_POSS}


def _rate(df, num, den):
    d = df[den].sum()
    return float(100 * df[num].sum() / d) if d else None


def on_off(st: pd.DataFrame) -> dict:
    """Team net rating and pace with each player on and off the floor."""
    if st.empty:
        return {}
    m = _team_view(st)
    m["players"] = m["lineup_key"].str.split("-")
    team_tot = m.groupby("team_id")[["poss", "opp_poss", "pts", "opp_pts", "secs"]].sum()
    e = m.explode("players")
    e["player_id"] = e["players"].astype(int)
    on = e.groupby(["player_id", "team_id"])[["poss", "opp_poss", "pts", "opp_pts", "secs"]].sum()
    out = {}
    for (pid, tid), r in on.iterrows():
        tot = team_tot.loc[tid]
        off = tot - r
        if r["poss"] < 100:
            continue

        def block(x):
            if x["poss"] <= 0 or x["opp_poss"] <= 0 or x["secs"] <= 0:
                return None
            return {"poss": float(x["poss"]), "net": 100 * x["pts"] / x["poss"] - 100 * x["opp_pts"] / x["opp_poss"],
                    "off": 100 * x["pts"] / x["poss"], "def": 100 * x["opp_pts"] / x["opp_poss"],
                    "pace": 48 * 60 * (x["poss"] + x["opp_poss"]) / 2 / x["secs"]}

        out.setdefault(str(int(pid)), {})[str(int(tid))] = {"on": block(r), "off": block(off)}
    return out


def score_state_pace(st: pd.DataFrame) -> dict:
    """Pace and efficiency when leading, trailing and close (score at the start of each stint)."""
    if st.empty:
        return {}
    m = _team_view(st)
    m["state"] = np.select([m["margin_start"] >= 6, m["margin_start"] <= -6], ["leading", "trailing"], "close")
    out = {}
    for (t, s), g in m.groupby(["team_id", "state"]):
        secs = g["secs"].sum()
        out.setdefault(str(int(t)), {})[s] = {
            "minutes": secs / 60,
            "pace": 48 * 60 * (g["poss"].sum() + g["opp_poss"].sum()) / 2 / secs if secs else None,
            "ortg": _rate(g, "pts", "poss"), "drtg": _rate(g, "opp_pts", "opp_poss"),
        }
    return out


def starting_lineups(conn, season: str) -> dict:
    """Record and net rating by starting five."""
    rows = pd.read_sql_query(
        """SELECT z.game_id, z.team_id, z.player_id FROM player_game_pbp z JOIN games g ON g.game_id = z.game_id
           WHERE z.starter = 1 AND g.season_id = ? AND g.season_type = 'regular' AND g.status = 'final'""",
        conn, params=(season,))
    if rows.empty:
        return {}
    keys = rows.groupby(["game_id", "team_id"])["player_id"].apply(lambda s: "-".join(str(x) for x in sorted(s)))
    tg = pd.read_sql_query(
        """SELECT t.game_id, t.team_id, t.pts, t.possessions, o.pts AS opp_pts, o.possessions AS opp_poss
           FROM team_game_stats t JOIN team_game_stats o ON o.game_id = t.game_id AND o.team_id <> t.team_id
           JOIN games g ON g.game_id = t.game_id WHERE g.season_id = ? AND g.season_type = 'regular'""",
        conn, params=(season,)).set_index(["game_id", "team_id"])
    d = tg.join(keys.rename("five"), how="inner").reset_index()
    out = {}
    for (t, five), g in d.groupby(["team_id", "five"]):
        if len(g) < 2:
            continue
        out.setdefault(str(int(t)), []).append({
            "players": [int(x) for x in five.split("-")], "games": int(len(g)),
            "wins": int((g["pts"] > g["opp_pts"]).sum()),
            "net": float(100 * g["pts"].sum() / g["possessions"].sum() - 100 * g["opp_pts"].sum() / g["opp_poss"].sum()),
        })
    for t in out:
        out[t].sort(key=lambda r: -r["games"])
        out[t] = out[t][:5]
    return out
