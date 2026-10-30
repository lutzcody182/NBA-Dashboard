"""Weekly scorecard: is the model getting sharper as the season goes on?

Everything is graded from what the model said BEFORE each game (the last
pregame snapshot saved before tip-off: model_runs, game_projections,
player_projections, prop_evals), never from numbers recomputed afterwards.
Each part has something to beat:

  * games: winner accuracy, Brier score and log loss of the win chance, margin
    and total errors; against the closing moneyline from legal books (no-vig)
  * players: minutes, points, rebounds, assists and threes errors for players
    who played; against each player's average over his previous 20 games
  * props: Brier score of the model's over chance; against the market's no-vig
    over chance at the same pull (pushes and players who did not play left out)
  * bets: units won, return on units risked, and closing-line value

Weeks run Monday to Sunday (Eastern). "To date" rows give the season so far.
"""
import math
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..models import props as PM
from ..providers import odds_api as OA

PLAYER_STATS = ("min", "pts", "reb", "ast", "fg3m")
BASE_GAMES = 20
PROP_STAT = {"pts": ("pts",), "reb": ("reb",), "ast": ("ast",), "fg3m": ("fg3m",), "pra": ("pts", "reb", "ast"),
             "pr": ("pts", "reb"), "pa": ("pts", "ast"), "ra": ("reb", "ast")}


def week_of(d: str) -> str:
    x = date.fromisoformat(d)
    return (x - timedelta(days=x.weekday())).isoformat()


def _r(v, n=3):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else round(f, n)


def _games(conn, season):
    return pd.read_sql_query(
        """SELECT game_id, game_date, tip_utc, home_team_id, away_team_id, home_score, away_score FROM games
           WHERE season_id = ? AND season_type IN ('regular', 'playoffs', 'play_in', 'cup_final')
             AND status = 'final' AND home_score IS NOT NULL AND tip_utc IS NOT NULL""", conn, params=[season])


def _last_runs(conn, games):
    """For each game: the last pregame run that started before tip-off."""
    runs = pd.read_sql_query("SELECT run_id, started_at FROM model_runs WHERE run_type = 'pregame'", conn)
    gp = pd.read_sql_query("SELECT run_id, game_id FROM game_projections", conn)
    pp = pd.read_sql_query("SELECT DISTINCT run_id, game_id FROM player_projections", conn)
    if runs.empty or (gp.empty and pp.empty):
        return {}
    both = pd.concat([gp, pp]).drop_duplicates().merge(runs, on="run_id").merge(games[["game_id", "tip_utc"]], on="game_id")
    both = both[both["started_at"] < both["tip_utc"]]
    return both.sort_values("started_at").groupby("game_id").tail(1).set_index("game_id")["run_id"].to_dict()


def _market_home(conn, game_id, tip):
    rows = conn.execute(
        """SELECT book, side, price_american FROM odds_snapshots
           WHERE game_id = ? AND player_id IS NULL AND market = 'h2h' AND fetched_at = (
               SELECT MAX(fetched_at) FROM odds_snapshots WHERE game_id = ? AND player_id IS NULL
                 AND market = 'h2h' AND fetched_at < ?)""", (game_id, game_id, tip)).fetchall()
    by = defaultdict(dict)
    for book, side, price in rows:
        if book in OA.LEGAL_BOOKS:
            by[book][side] = price
    fairs = [PM.devig(s["home"], s["away"])[0] for s in by.values() if "home" in s and "away" in s]
    fairs = [f for f in fairs if f is not None]
    return float(np.mean(fairs)) if fairs else None


def game_rows(conn, games, run_of):
    gp = pd.read_sql_query("SELECT run_id, game_id, home_win_prob, spread_mean, total_mean FROM game_projections", conn)
    out = []
    for g in games.itertuples():
        run = run_of.get(g.game_id)
        if run is None:
            continue
        p = gp[(gp["run_id"] == run) & (gp["game_id"] == g.game_id)]
        if p.empty:
            continue
        p = p.iloc[0]
        margin, total = g.home_score - g.away_score, g.home_score + g.away_score
        out.append({"game_id": g.game_id, "week": week_of(g.game_date), "p": float(p.home_win_prob),
                    "won": int(margin > 0), "margin_err": float(p.spread_mean) - margin,
                    "total_err": float(p.total_mean) - total, "mkt": _market_home(conn, g.game_id, g.tip_utc)})
    return pd.DataFrame(out)


def player_rows(conn, season, games, run_of):
    pp = pd.read_sql_query(
        """SELECT run_id, game_id, player_id, play_prob, minutes_mean, pts_mean, reb_mean, ast_mean, fg3m_mean
           FROM player_projections WHERE pts_mean IS NOT NULL""", conn)
    keep = {(g, r) for g, r in run_of.items()}
    pp = pp[[(g, r) in keep for g, r in zip(pp["game_id"], pp["run_id"])]]
    prev = f"{int(season[:4]) - 1}-{season[2:4]}"
    box = pd.read_sql_query(
        """SELECT s.game_id, s.player_id, g.game_date, s.seconds / 60.0 AS min, s.pts, s.oreb + s.dreb AS reb, s.ast, s.fg3m
           FROM player_game_stats s JOIN games g ON g.game_id = s.game_id
           WHERE g.season_id IN (?, ?) AND g.season_type = 'regular' AND g.status = 'final' AND s.seconds > 0""",
        conn, params=[season, prev]).sort_values(["player_id", "game_date"])
    for k in PLAYER_STATS:   # each player's average over his previous BASE_GAMES games (both seasons)
        box[f"base_{k}"] = box.groupby("player_id")[k].transform(
            lambda s: s.shift(1).rolling(BASE_GAMES, min_periods=3).mean())
    if pp.empty or box.empty:
        return pd.DataFrame()
    d = pp.merge(box, on=["game_id", "player_id"])
    if d.empty:
        return pd.DataFrame()
    d = d.rename(columns={"minutes_mean": "p_min", "pts_mean": "p_pts", "reb_mean": "p_reb", "ast_mean": "p_ast",
                          "fg3m_mean": "p_fg3m"})
    d["week"] = d["game_date"].map(week_of)
    return d


def prop_rows(conn, games):
    try:
        pe = pd.read_sql_query(
            "SELECT fetched_at, game_id, player_id, stat, line, p_over, fair_over FROM prop_evals "
            "WHERE p_over IS NOT NULL AND fair_over IS NOT NULL", conn)
    except Exception:           # the props step creates this table on first use
        return pd.DataFrame()
    if pe.empty:
        return pe
    pe = pe.merge(games[["game_id", "tip_utc", "game_date"]], on="game_id")
    pe = pe[pe["fetched_at"] < pe["tip_utc"]]
    if pe.empty:
        return pd.DataFrame()
    pe = pe.sort_values("fetched_at").groupby(["game_id", "player_id", "stat", "line"]).tail(1)
    box = pd.read_sql_query(
        """SELECT game_id, player_id, pts, oreb + dreb AS reb, ast, fg3m FROM player_game_stats WHERE seconds > 0""", conn)
    pe = pe.merge(box, on=["game_id", "player_id"])
    pe["actual"] = [sum(r[c] for c in PROP_STAT[s]) if s in PROP_STAT else np.nan
                    for s, r in zip(pe["stat"], pe[["pts", "reb", "ast", "fg3m"]].to_dict("records"))]
    pe = pe[pe["actual"].notna() & (pe["actual"] != pe["line"])]
    pe["over"] = (pe["actual"] > pe["line"]).astype(int)
    pe["week"] = pe["game_date"].map(week_of)
    return pe


def bet_rows(conn, games):
    try:
        b = pd.read_sql_query("SELECT game_id, units, pnl, clv, status FROM bets WHERE status IN ('won', 'lost', 'push')",
                              conn)
    except Exception:
        return pd.DataFrame()
    b = b.merge(games[["game_id", "game_date"]], on="game_id")
    if b.empty:
        return pd.DataFrame()
    b["week"] = b["game_date"].map(week_of)
    return b


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _game_block(g):
    if g is None or g.empty:
        return None
    y = g["won"].to_numpy(float)
    m = g[g["mkt"].notna()]
    out = {"n": int(len(g)), "accuracy": _r(((g["p"] > 0.5) == (y == 1)).mean()),
           "brier": _r(np.mean((g["p"] - y) ** 2), 4), "log_loss": _r(_ll(g["p"], y), 4),
           "margin_mae": _r(g["margin_err"].abs().mean(), 2), "margin_bias": _r(g["margin_err"].mean(), 2),
           "total_mae": _r(g["total_err"].abs().mean(), 2), "total_bias": _r(g["total_err"].mean(), 2)}
    if len(m):
        ym = m["won"].to_numpy(float)
        out.update({"mkt_n": int(len(m)), "brier_model_same": _r(np.mean((m["p"] - ym) ** 2), 4),
                    "brier_market": _r(np.mean((m["mkt"] - ym) ** 2), 4)})
    return out


def _player_block(d):
    if d is None or d.empty:
        return None
    out = {"n": int(len(d))}
    for k in PLAYER_STATS:
        h = d[d[f"base_{k}"].notna() & d[f"p_{k}"].notna()]
        if len(h):
            out[k] = {"mae": _r((h[f"p_{k}"] - h[k]).abs().mean(), 2),
                      "base_mae": _r((h[f"base_{k}"] - h[k]).abs().mean(), 2),
                      "bias": _r((h[f"p_{k}"] - h[k]).mean(), 2), "n": int(len(h))}
    return out


def _prop_block(p):
    if p is None or p.empty:
        return None
    y = p["over"].to_numpy(float)
    return {"n": int(len(p)), "brier": _r(np.mean((p["p_over"] - y) ** 2), 4),
            "brier_market": _r(np.mean((p["fair_over"] - y) ** 2), 4), "over_rate": _r(y.mean()),
            "model_over": _r(p["p_over"].mean()), "market_over": _r(p["fair_over"].mean())}


def _bet_block(b):
    if b is None or b.empty:
        return None
    risked = float(b["units"].sum())
    clv = b["clv"].dropna()
    return {"n": int(len(b)), "units": _r(b["pnl"].sum(), 2), "risked": _r(risked, 2),
            "roi": _r(b["pnl"].sum() / risked, 4) if risked else None,
            "avg_clv": _r(clv.mean(), 4) if len(clv) else None,
            "beat_close": _r((clv > 0).mean()) if len(clv) else None}


def build(conn, season) -> dict:
    games = _games(conn, season)
    if games.empty:
        return {"season": season, "weeks": []}
    run_of = _last_runs(conn, games)
    G = game_rows(conn, games, run_of)
    P = player_rows(conn, season, games, run_of)
    R = prop_rows(conn, games)
    B = bet_rows(conn, games)
    weeks = sorted(set(G["week"] if len(G) else []) | set(P["week"] if len(P) else [])
                   | set(R["week"] if len(R) else []) | set(B["week"] if len(B) else []))

    def part(df, w, upto=False):
        if df is None or df.empty:
            return None
        return df[df["week"] <= w] if upto else df[df["week"] == w]
    rows = []
    for w in weeks:
        rows.append({"week": w,
                     "games": _game_block(part(G, w)), "players": _player_block(part(P, w)),
                     "props": _prop_block(part(R, w)), "bets": _bet_block(part(B, w)),
                     "to_date": {"games": _game_block(part(G, w, True)), "players": _player_block(part(P, w, True)),
                                 "props": _prop_block(part(R, w, True)), "bets": _bet_block(part(B, w, True))}})
    return {"season": season, "weeks": rows, "base_games": BASE_GAMES,
            "graded_games": int(len(G)), "first_snapshot_games": int(len(run_of))}
