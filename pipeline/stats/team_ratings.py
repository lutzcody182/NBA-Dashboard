"""Team pace and efficiency: raw season numbers, splits, and opponent-adjusted ratings."""
import numpy as np
import pandas as pd

from .. import config
from .formulas import pythagorean_win_pct

TEAM_GAME_SQL = """
SELECT g.game_id, g.game_date, g.season_id, g.neutral_site,
       t.team_id, o.team_id AS opp_id, t.is_home,
       t.minutes, t.possessions AS poss,
       t.pts, t.fgm, t.fga, t.fg3m, t.fg3a, t.ftm, t.fta, t.oreb, t.dreb, t.ast,
       t.stl, t.blk, t.tov, t.pf, t.pts_paint, t.pts_second_chance, t.pts_fast_break,
       o.pts AS opp_pts, o.fgm AS opp_fgm, o.fga AS opp_fga, o.fg3m AS opp_fg3m,
       o.fg3a AS opp_fg3a, o.ftm AS opp_ftm, o.fta AS opp_fta, o.oreb AS opp_oreb,
       o.dreb AS opp_dreb, o.tov AS opp_tov, o.pf AS opp_pf, o.pts_paint AS opp_pts_paint,
       o.pts_second_chance AS opp_pts_second_chance, o.pts_fast_break AS opp_pts_fast_break
FROM team_game_stats t
JOIN team_game_stats o ON o.game_id = t.game_id AND o.team_id <> t.team_id
JOIN games g ON g.game_id = t.game_id
WHERE g.status = 'final' AND g.season_id = ? AND g.season_type IN ({types})
ORDER BY g.game_date, g.game_id
"""


def load_team_games(conn, season_id: str) -> pd.DataFrame:
    types = ",".join(f"'{t}'" for t in config.STAT_SEASON_TYPES)
    df = pd.read_sql_query(TEAM_GAME_SQL.format(types=types), conn, params=(season_id,))
    if df.empty:
        return df
    df["win"] = (df["pts"] > df["opp_pts"]).astype(int)
    df["ortg"] = 100 * df["pts"] / df["poss"]
    df["drtg"] = 100 * df["opp_pts"] / df["poss"]
    df["pace"] = 48 * df["poss"] / (df["minutes"] / 5)
    df["game_num"] = df.groupby("team_id").cumcount() + 1
    return df


def _sum_ratios(g: pd.DataFrame) -> dict:
    """Season-style numbers from summed counts (never an average of game ratios)."""
    s = g.sum(numeric_only=True)
    poss, n = s["poss"], len(g)

    def r(num, den):
        return float(num / den) if den and num is not None and num == num else None

    for k in ("pts_paint", "pts_second_chance", "pts_fast_break",
              "opp_pts_paint", "opp_pts_second_chance", "opp_pts_fast_break"):
        # Some sources leave these out (ESPN has no second-chance points): report nothing
        # rather than a total that silently skips games.
        if k not in g or g[k].isna().any():
            s[k] = None

    return {
        "games": int(n),
        "wins": int(s["win"]),
        "losses": int(n - s["win"]),
        "poss": float(poss),
        "pace": r(48 * poss, s["minutes"] / 5),
        "ortg": r(100 * s["pts"], poss),
        "drtg": r(100 * s["opp_pts"], poss),
        "net": r(100 * (s["pts"] - s["opp_pts"]), poss),
        # Four factors, offense
        "efg": r(s["fgm"] + 0.5 * s["fg3m"], s["fga"]),
        "tov_pct": r(s["tov"], s["fga"] + 0.44 * s["fta"] + s["tov"]),
        "orb_pct": r(s["oreb"], s["oreb"] + s["opp_dreb"]),
        "ft_rate": r(s["fta"], s["fga"]),
        "ts": r(s["pts"], 2 * (s["fga"] + 0.44 * s["fta"])),
        "three_rate": r(s["fg3a"], s["fga"]),
        # Four factors, defense (what opponents did)
        "opp_efg": r(s["opp_fgm"] + 0.5 * s["opp_fg3m"], s["opp_fga"]),
        "forced_tov_pct": r(s["opp_tov"], s["opp_fga"] + 0.44 * s["opp_fta"] + s["opp_tov"]),
        "drb_pct": r(s["dreb"], s["dreb"] + s["opp_oreb"]),
        "opp_ft_rate": r(s["opp_fta"], s["opp_fga"]),
        "opp_three_rate": r(s["opp_fg3a"], s["opp_fga"]),
        # Scoring types per 100 possessions
        "paint_pts_100": r(100 * s["pts_paint"], poss),
        "second_chance_100": r(100 * s["pts_second_chance"], poss),
        "fast_break_100": r(100 * s["pts_fast_break"], poss),
        "opp_paint_pts_100": r(100 * s["opp_pts_paint"], poss),
        "opp_second_chance_100": r(100 * s["opp_pts_second_chance"], poss),
        "opp_fast_break_100": r(100 * s["opp_pts_fast_break"], poss),
        "fouls_100": r(100 * s["pf"], poss),
        "pts_for": float(s["pts"]),
        "pts_against": float(s["opp_pts"]),
    }


# -- opponent-adjusted ratings ---------------------------------------------

def _design(df: pd.DataFrame, team_index: dict):
    """One row per offensive observation: offense team, defense team, home edge."""
    n, k = len(df), len(team_index)
    X = np.zeros((n, 2 * k + 1))
    rows = np.arange(n)
    X[rows, df["team_id"].map(team_index).to_numpy()] = 1.0
    X[rows, k + df["opp_id"].map(team_index).to_numpy()] = 1.0
    home = np.where(df["neutral_site"].to_numpy() == 1, 0.0,
                    np.where(df["is_home"].to_numpy() == 1, 1.0, -1.0))
    X[:, 2 * k] = home
    return X


def _fit_ridge(df: pd.DataFrame, team_index: dict, lam: float):
    """Weighted ridge regression of points per 100 possessions.

    ortg_game = league_avg + offense_team + defense_opponent + home * h
    Weights are possessions. Team effects are shrunk toward 0 (league
    average) by `lam`, measured in possessions: a team needs roughly that
    many possessions of evidence before its effect moves halfway from
    average. The home term is not shrunk.
    """
    y = df["ortg"].to_numpy()
    w = df["poss"].to_numpy()
    mu = float(np.average(y, weights=w))
    X = _design(df, team_index)
    pen = np.full(X.shape[1], lam)
    pen[-1] = 1e-6
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X + np.diag(pen), XtW @ (y - mu))
    return mu, beta


def choose_lambda(df: pd.DataFrame, team_index: dict, grid=(100, 200, 400, 800, 1600, 3200, 6400)):
    """Pick the shrinkage strength that best predicts games it has not seen.

    Walk forward through the season: fit on games before each cutoff,
    predict the next 7 days, and keep the lambda with the lowest
    possession-weighted squared error. Returns (lambda, details) or
    (None, reason) when there is not enough history.
    """
    dates = sorted(df["game_date"].unique())
    if len(df) < 400 or len(dates) < 30:
        return None, "not enough games for out-of-sample selection"
    cutoffs = dates[20::7]
    errors = {lam: [0.0, 0.0] for lam in grid}
    for cut in cutoffs:
        train = df[df["game_date"] < cut]
        test = df[(df["game_date"] >= cut) & (df["game_date"] < _plus_days(cut, 7))]
        if test.empty or train["team_id"].nunique() < len(team_index):
            continue
        Xt = _design(test, team_index)
        for lam in grid:
            mu, beta = _fit_ridge(train, team_index, lam)
            pred = mu + Xt @ beta
            err = test["ortg"].to_numpy() - pred
            errors[lam][0] += float(np.sum(test["poss"].to_numpy() * err ** 2))
            errors[lam][1] += float(test["poss"].sum())
    scored = {lam: e[0] / e[1] for lam, e in errors.items() if e[1] > 0}
    if not scored:
        return None, "no usable walk-forward windows"
    best = min(scored, key=scored.get)
    return best, {str(k): round(v, 3) for k, v in scored.items()}


def _plus_days(d: str, n: int) -> str:
    return (pd.Timestamp(d) + pd.Timedelta(days=n)).strftime("%Y-%m-%d")


DEFAULT_LAMBDA = 800  # used only until there is data to choose one (about 8 games' worth)


def adjusted_ratings(df: pd.DataFrame, lam: float):
    teams = sorted(df["team_id"].unique())
    idx = {t: i for i, t in enumerate(teams)}
    mu, beta = _fit_ridge(df, idx, lam)
    k = len(teams)
    out = {}
    for t, i in idx.items():
        o = mu + beta[i]
        d = mu + beta[k + i]
        out[int(t)] = {"adj_ortg": o, "adj_drtg": d, "adj_net": o - d}
    return out, {"league_ortg": mu, "home_edge_per_100": float(2 * beta[-1])}


# -- full team table -------------------------------------------------------

def team_table(df: pd.DataFrame, prior_lambda=None) -> dict:
    """Everything the Team Analytics page shows, for one season."""
    if df.empty:
        return {"teams": [], "model": {"note": "no completed games yet"}}

    teams = sorted(df["team_id"].unique())
    idx = {t: i for i, t in enumerate(teams)}
    lam, cv = choose_lambda(df, idx)
    if lam is not None:
        lam_source = "walk-forward selection on this season"
    elif prior_lambda is not None:
        lam, lam_source = prior_lambda, "selected on the previous season (not enough games yet this season)"
    else:
        lam, lam_source = DEFAULT_LAMBDA, "default (not enough games to select one yet)"
    adj, league = adjusted_ratings(df, lam)

    season_pace = df.groupby("team_id").apply(
        lambda g: 48 * g["poss"].sum() / (g["minutes"].sum() / 5), include_groups=False
    )
    median_pace = float(season_pace.median())
    fast_opps = set(season_pace[season_pace >= median_pace].index)

    rows = []
    for t, g in df.groupby("team_id"):
        g = g.sort_values(["game_date", "game_id"])
        row = {"team_id": int(t), **_sum_ratios(g), **adj[int(t)]}
        row["expected_wins"] = (pythagorean_win_pct(row["pts_for"], row["pts_against"]) or 0) * row["games"]
        opp_adj = [adj[int(o)]["adj_net"] for o in g["opp_id"]]
        row["sos"] = float(np.mean(opp_adj)) if opp_adj else None
        for label, part in (("home", g[g["is_home"] == 1]), ("away", g[g["is_home"] == 0])):
            row[label] = _brief(part)
        row["last"] = {str(n): _brief(g.tail(n)) for n in (5, 10, 20)}
        row["vs_fast"] = _brief(g[g["opp_id"].isin(fast_opps)])
        row["vs_slow"] = _brief(g[~g["opp_id"].isin(fast_opps)])
        rows.append(row)

    return {
        "teams": rows,
        "model": {
            "lambda_possessions": lam,
            "lambda_source": lam_source,
            "cv_error_by_lambda": cv if isinstance(cv, dict) else None,
            "league_ortg": league["league_ortg"],
            "home_edge_per_100": league["home_edge_per_100"],
            "median_pace": median_pace,
            "possessions_method": "box-score estimate (Basketball-Reference formula)",
        },
    }


def _brief(g: pd.DataFrame):
    if g.empty:
        return None
    s = g.sum(numeric_only=True)
    poss = s["poss"]
    return {
        "games": int(len(g)),
        "wins": int(s["win"]),
        "pace": float(48 * poss / (s["minutes"] / 5)),
        "ortg": float(100 * s["pts"] / poss),
        "drtg": float(100 * s["opp_pts"] / poss),
        "net": float(100 * (s["pts"] - s["opp_pts"]) / poss),
    }
