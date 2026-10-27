"""Per-possession team rates, shrunk by amounts learned from data: the simulator's inputs.

For every team, on offense and on defense:
  * turnovers per possession
  * free throws per possession, free-throw %, and-ones per made shot
  * shot mix: share of shots from each zone (rim, paint, mid-range,
    corner 3, above-the-break 3)
  * make % in each zone
  * offensive rebound % (of the team's own misses)
  * pace (possessions per 48 minutes)
  * assists per made shot (for player projections)

Each is estimated like the player stats are (see shrink.py): this
season's numbers plus a learned share of last season's, pulled toward the
league by a learned amount. The amounts are picked by predicting the rest
of a season from its first part, so for example opponents' 3-point
percentage, which is mostly luck, ends up shrunk hard, while rim-attempt
share, which teams control, is trusted quickly.

Shot quality is the expected effective FG% of a team's shot mix if every
shot went in at the league rate for its zone; shot-making is how far the
team's (shrunk) zone percentages sit above or below that.
"""
import numpy as np
import pandas as pd

from .shrink import calibrate as shrink_calibrate

ZONES = ("rim", "paint", "mid", "c3", "ab3")
THREE = {"c3", "ab3"}
TEAM_K_GRID = np.array([0, 20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 1e6])

# name -> (made column, tries column, applies to defense too)
RATES = {
    "tov": ("tov", "poss", True),
    "fta": ("fta", "poss", True),
    "ftp": ("ftm", "fta", False),
    "and1": ("and1", "fgm", True),
    "oreb": ("oreb", "reb_chances", True),
    **{f"share_{z}": (f"fga_{z}", "fga", True) for z in ZONES},
    **{f"fg_{z}": (f"fgm_{z}", f"fga_{z}", True) for z in ZONES},
    "pace": ("poss", "games48", False),
    "ast": ("ast", "fgm", False),
}
COUNTS = ["poss", "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "tov", "ast",
          *(f"fga_{z}" for z in ZONES), *(f"fgm_{z}" for z in ZONES)]


def team_games(conn, seasons, season_type="regular") -> pd.DataFrame:
    """One row per team-game: that team's offense, with the opponent's defensive rebounds."""
    q = ",".join("?" * len(seasons))
    st = pd.read_sql_query(
        f"""SELECT s.game_id, s.team_id, s.opp_id, g.season_id, g.game_date, g.periods,
                   {", ".join(f"SUM(s.{c}) AS {c}" for c in COUNTS)}
            FROM lineup_stints s JOIN games g ON g.game_id = s.game_id
            WHERE g.status = 'final' AND g.season_type = ? AND g.season_id IN ({q})
            GROUP BY s.game_id, s.team_id""", conn, params=[season_type, *seasons])
    a1 = pd.read_sql_query(
        f"""SELECT z.game_id, z.team_id, SUM(z.and1) AS and1 FROM player_game_pbp z
            JOIN games g ON g.game_id = z.game_id WHERE g.season_type = ? AND g.season_id IN ({q})
            GROUP BY 1, 2""", conn, params=[season_type, *seasons])
    st = st.merge(a1, on=["game_id", "team_id"], how="left").fillna({"and1": 0})
    opp = st[["game_id", "team_id", "dreb"]].rename(columns={"team_id": "opp_id", "dreb": "opp_dreb"})
    st = st.merge(opp, on=["game_id", "opp_id"], how="left")
    st["reb_chances"] = st["oreb"] + st["opp_dreb"]
    st["games48"] = (48 + 5 * (st["periods"].fillna(4).clip(lower=4) - 4)) / 48
    return st.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def _side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Offense: grouped by the team. Defense: what opponents did against the team."""
    d = df.copy()
    d["player_id"] = d["team_id"] if side == "off" else d["opp_id"]   # shrink.calibrate groups by player_id
    return d


def calibrate(conn, season: str, prev: str) -> dict:
    """Learn k (shrinkage) and w (last season's weight) for every rate, offense and defense."""
    cur, before = team_games(conn, [season]), team_games(conn, [prev])
    dates = sorted(cur["game_date"].unique())
    cutoffs = [dates[int(len(dates) * f)] for f in (0.08, 0.2, 0.35, 0.5, 0.7)]
    fits = {}
    for name, (made, tries, on_def) in RATES.items():
        for side in (("off", "def") if on_def else ("off",)):
            f = shrink_calibrate(_side(cur, side), _side(before, side), made, tries, cutoffs,
                                 min_test=1.0, k_grid=TEAM_K_GRID)
            fits[f"{side}:{name}"] = f.to_dict()
    return {"season": season, "prev": prev, "fits": fits}


def _league(df: pd.DataFrame) -> dict:
    out = {}
    for name, (made, tries, _) in RATES.items():
        t = df[tries].sum()
        out[name] = float(df[made].sum() / t) if t else np.nan
    return out


def estimates(cur: pd.DataFrame, prev: pd.DataFrame, fits: dict, teams) -> dict:
    """Shrunk rates for every team from this season so far (cur) and last season (prev).

    Returns {"league": {...}, "teams": {team_id: {"off": {...}, "def": {...}}}}.
    """
    league = _league(cur) if len(cur) >= 60 else _league(prev)
    if len(cur) and len(cur) < 60:   # very early: blend toward last season's league numbers
        lc, lp = _league(cur), _league(prev)
        share = len(cur) / 60
        league = {k: share * lc[k] + (1 - share) * lp[k] if not np.isnan(lc[k]) else lp[k] for k in lp}
    out = {}
    for side in ("off", "def"):
        c, p = _side(cur, side), _side(prev, side)
        cg, pg = c.groupby("player_id"), p.groupby("player_id")
        csum, psum = cg[list({*COUNTS, "and1", "reb_chances", "games48"})].sum(), \
            pg[list({*COUNTS, "and1", "reb_chances", "games48"})].sum()
        for name, (made, tries, on_def) in RATES.items():
            if side == "def" and not on_def:
                continue
            f = fits[f"{side}:{name}"]
            k, w, mu = f["k"], f["w"], league[name]
            for t in teams:
                cm = csum[made].get(t, 0.0)
                ct = csum[tries].get(t, 0.0)
                pm = psum[made].get(t, 0.0)
                pt = psum[tries].get(t, 0.0)
                den = ct + w * pt + k
                val = (cm + w * pm + k * mu) / den if den > 0 else mu
                out.setdefault(int(t), {}).setdefault(side, {})[name] = float(val)
    for t in out:   # zone shares must sum to one
        for side in out[t]:
            s = sum(out[t][side][f"share_{z}"] for z in ZONES)
            for z in ZONES:
                out[t][side][f"share_{z}"] /= s
    return {"league": league, "teams": out}


def shot_quality(est: dict) -> dict:
    """Expected eFG% of each team's shot mix at league zone rates, and shot-making above that."""
    lg = est["league"]
    res = {}
    for t, r in est["teams"].items():
        row = {}
        for side in ("off", "def"):
            s = r[side]
            xefg = sum(s[f"share_{z}"] * lg[f"fg_{z}"] * (1.5 if z in THREE else 1.0) for z in ZONES)
            efg = sum(s[f"share_{z}"] * s[f"fg_{z}"] * (1.5 if z in THREE else 1.0) for z in ZONES)
            row[side] = {"shot_quality": xefg, "expected_efg": efg, "shot_making": efg - xefg,
                         **{f"share_{z}": s[f"share_{z}"] for z in ZONES},
                         **{f"fg_{z}": s[f"fg_{z}"] for z in ZONES}}
        res[t] = row
    lg_efg = sum(_lg_share(est, z) * lg[f"fg_{z}"] * (1.5 if z in THREE else 1.0) for z in ZONES)
    return {"league_efg": lg_efg, "teams": res}


def _lg_share(est, z):
    return est["league"][f"share_{z}"]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _inv(x):
    return 1 / (1 + np.exp(-x))


def matchup(est: dict, off_team: int, def_team: int) -> dict:
    """Rates for one team's offense against another's defense.

    Probabilities combine on the log-odds scale (offense edge plus defense
    edge over the league); zone shares combine multiplicatively and are
    renormalised. Free-throw % belongs to the shooters only.
    """
    lg = est["league"]
    o, d = est["teams"][off_team]["off"], est["teams"][def_team]["def"]
    r = {}
    for name in ("tov", "and1", "oreb", *(f"fg_{z}" for z in ZONES)):
        r[name] = float(_inv(_logit(o[name]) + _logit(d[name]) - _logit(lg[name])))
    r["fta"] = float(o["fta"] * d["fta"] / lg["fta"])
    r["ftp"] = o["ftp"]
    sh = {z: o[f"share_{z}"] * d[f"share_{z}"] / lg[f"share_{z}"] for z in ZONES}
    tot = sum(sh.values())
    for z in ZONES:
        r[f"share_{z}"] = sh[z] / tot
    return r
