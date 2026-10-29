"""Every lineup combination a team has used: five-man units, and the two-,
three- and four-man groups inside them.

Built from play-by-play stints (only stints whose five players were
verified). For each group: time together, possessions, offensive and
defensive rating, net, pace, and the four factors on both ends (effective
FG%, turnover rate, offensive rebound rate, free throw rate).

A net rating over few possessions is mostly noise. Each group also gets a
margin: 1.96 times the standard error of a net rating over that many
possessions, measured from the spread of every stint in the league, so a
+15 with a margin of 25 reads as what it is.
"""
from itertools import combinations

import numpy as np
import pandas as pd

STATS = ["poss", "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "tov", "oreb", "dreb"]
MIN_MINUTES = {5: 4.0, 4: 6.0, 3: 8.0, 2: 10.0}
MAX_ROWS = {5: 60, 4: 120, 3: 160, 2: 120}


def _paired(st: pd.DataFrame) -> pd.DataFrame:
    """One row per stint per team, this team's stats and the opponent's side by side."""
    keep = ["game_id", "stint_num", "team_id", "lineup_key", "start_sec", "end_sec"] + STATS
    me = st[[c for c in keep if c in st]].copy()
    opp = st[["game_id", "stint_num", "team_id"] + STATS].rename(
        columns={**{c: f"o_{c}" for c in STATS}, "team_id": "opp_team"})
    m = me.merge(opp, on=["game_id", "stint_num"])
    m = m[m["team_id"] != m["opp_team"]].copy()
    m["secs"] = m["end_sec"] - m["start_sec"]
    return m


def _net_variance(m: pd.DataFrame) -> float:
    """Per-possession variance of offense-minus-defense points, from every stint."""
    ok = m["poss"] > 0
    y = 100 * m.loc[ok, "pts"] / m.loc[ok, "poss"]
    mu = np.average(y, weights=m.loc[ok, "poss"])
    return 2 * float(np.mean(m.loc[ok, "poss"] * (y - mu) ** 2))


def _summarise(g: pd.DataFrame, s2: float) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        g["minutes"] = g["secs"] / 60
        g["ortg"] = 100 * g["pts"] / g["poss"]
        g["drtg"] = 100 * g["o_pts"] / g["o_poss"]
        g["net"] = g["ortg"] - g["drtg"]
        n = (g["poss"] + g["o_poss"]) / 2
        g["n"] = n
        g["margin"] = 1.96 * np.sqrt(s2 / n.clip(lower=1))
        g["pace"] = 48 * 60 * n / g["secs"]
        g["efg"] = (g["fgm"] + 0.5 * g["fg3m"]) / g["fga"]
        g["tov_pct"] = g["tov"] / g["poss"]
        g["orb_pct"] = g["oreb"] / (g["oreb"] + g["o_dreb"])
        g["ft_rate"] = g["ftm"] / g["fga"]
        g["three_rate"] = g["fg3a"] / g["fga"]
        g["o_efg"] = (g["o_fgm"] + 0.5 * g["o_fg3m"]) / g["o_fga"]
        g["o_tov_pct"] = g["o_tov"] / g["o_poss"]
        g["drb_pct"] = g["dreb"] / (g["dreb"] + g["o_oreb"])
        g["o_ft_rate"] = g["o_ftm"] / g["o_fga"]
    return g


OUT_COLS = ["players", "minutes", "n", "pts", "o_pts", "ortg", "drtg", "net", "margin", "pace", "efg", "tov_pct",
            "orb_pct", "ft_rate", "three_rate", "o_efg", "o_tov_pct", "drb_pct", "o_ft_rate"]
ROUND = {"minutes": 1, "n": 1, "ortg": 1, "drtg": 1, "net": 1, "margin": 1, "pace": 1, "efg": 3, "tov_pct": 3,
         "orb_pct": 3, "ft_rate": 3, "three_rate": 3, "o_efg": 3, "o_tov_pct": 3, "drb_pct": 3, "o_ft_rate": 3}


def build(st: pd.DataFrame, season: str) -> dict:
    if st is None or st.empty:
        return {"season": season, "teams": {}}
    m = _paired(st)
    if m.empty:
        return {"season": season, "teams": {}}
    s2 = _net_variance(m)
    agg_cols = ["secs"] + STATS + [f"o_{c}" for c in STATS]
    five = m.groupby(["team_id", "lineup_key"], as_index=False)[agg_cols].sum()
    teams = {}
    for tid, tg in five.groupby("team_id"):
        tot = tg[agg_cols].sum()
        team_row = _summarise(pd.DataFrame([tot]), s2).iloc[0]
        out = {"team": {k: (None if pd.isna(team_row[k]) else round(float(team_row[k]), ROUND.get(k, 2)))
                        for k in OUT_COLS if k != "players"}}
        for size in (5, 4, 3, 2):
            if size == 5:
                g = tg.copy()
                g["key"] = g["lineup_key"]
            else:
                # split every five-man unit's totals into each of its smaller groups
                parts = []
                arr = tg[agg_cols].to_numpy()
                for keys, row in zip(tg["lineup_key"], arr):
                    ids = sorted(keys.split("-"), key=int)
                    for combo in combinations(ids, size):
                        parts.append(("-".join(combo), row))
                if not parts:
                    continue
                g = pd.DataFrame([p[1] for p in parts], columns=agg_cols)
                g["key"] = [p[0] for p in parts]
                g = g.groupby("key", as_index=False)[agg_cols].sum()
            g = g[g["secs"] / 60 >= MIN_MINUTES[size]].copy()
            if g.empty:
                out[str(size)] = {"cols": OUT_COLS, "rows": []}
                continue
            g = _summarise(g, s2).sort_values("secs", ascending=False).head(MAX_ROWS[size])
            g["players"] = g["key"].map(lambda k: [int(x) for x in k.split("-")])
            rows = []
            for r in g.itertuples(index=False):
                d = r._asdict()
                rows.append([d["players"]] + [None if pd.isna(d[c]) or np.isinf(d[c]) else
                                              (round(float(d[c]), ROUND.get(c, 0)) if c in ROUND else int(d[c]))
                                              for c in OUT_COLS[1:]])
            out[str(size)] = {"cols": OUT_COLS, "rows": rows}
        teams[str(int(tid))] = out
    return {"season": season, "teams": teams, "net_variance_per_poss": round(s2, 1),
            "min_minutes": MIN_MINUTES}
