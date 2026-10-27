"""Who picks up the shots (and assists, rebounds, ...) when someone is out.

Every player has a usage weight for each stat: his share of what his team
does while he is on the floor (shots, free throws, assists, rebounds,
turnovers), shrunk toward one fifth by an amount learned from data.

In any five-man lineup the team's total is split among the five as

    player share = weight ** g / sum of the five players' weight ** g

so when a high-usage player sits, the other four absorb his share: with
g = 1 they absorb it in proportion to their own usage, with g > 1 the
high-usage teammates take more than that. g and the shrinkage are fit by
predicting players' stats in games the fit has not seen, from the actual
lineups of those games; the test is reported separately for the games
where a team's top-usage player was missing, which is the case this is
for.
"""
import numpy as np
import pandas as pd

STATS = {
    "fga": ("fga", "Field goal attempts"),
    "fg3a": ("fg3a", "3-point attempts"),
    "fta": ("fta", "Free throw attempts"),
    "ast": ("ast", "Assists"),
    "oreb": ("oreb", "Offensive rebounds"),
    "dreb": ("dreb", "Defensive rebounds"),
    "tov": ("tov", "Turnovers"),
}
G_GRID = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
K_GRID = (5, 20, 50, 100, 200, 400)


def load(conn, seasons):
    """Stints exploded to player rows, and player-game totals."""
    q = ",".join("?" * len(seasons))
    st = pd.read_sql_query(
        f"""SELECT s.game_id, s.stint_num, s.team_id, s.lineup_key, s.fga, s.fg3a, s.fta, s.ast, s.oreb, s.dreb,
                   s.tov, g.season_id, g.game_date FROM lineup_stints s JOIN games g ON g.game_id = s.game_id
            WHERE s.ok = 1 AND g.season_type = 'regular' AND g.season_id IN ({q})""", conn, params=list(seasons))
    st["player_id"] = st["lineup_key"].str.split("-")
    st = st.explode("player_id")
    st["player_id"] = st["player_id"].astype(int)
    pg = pd.read_sql_query(
        f"""SELECT p.game_id, p.player_id, p.team_id, p.fga, p.fg3a, p.fta, p.ast, p.oreb, p.dreb, p.tov,
                   g.season_id, g.game_date FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
            WHERE g.season_type = 'regular' AND g.status = 'final' AND p.seconds > 0 AND g.season_id IN ({q})""",
        conn, params=list(seasons))
    return st, pg


def weights(st_before: pd.DataFrame, pg_before: pd.DataFrame, stat: str, k: float) -> pd.Series:
    """Shrunk share of the team's `stat` while on the floor, per player (one fifth = average)."""
    on = st_before.groupby("player_id")[stat].sum()
    own = pg_before.groupby("player_id")[stat].sum().reindex(on.index, fill_value=0)
    return (own + k * 0.2) / (on + k)


def predict(st_game: pd.DataFrame, w: pd.Series, stat: str, g: float, default: float = 0.2) -> pd.Series:
    """Each player's expected `stat` from the team totals of the stints he played in."""
    d = st_game[["game_id", "stint_num", "team_id", "player_id", stat]].copy()
    d["u"] = d["player_id"].map(w).fillna(default) ** g
    d["share"] = d["u"] / d.groupby(["game_id", "stint_num", "team_id"])["u"].transform("sum")
    d["pred"] = d["share"] * d[stat]
    return d.groupby(["game_id", "player_id"])["pred"].sum()


def fixed_share(st_game: pd.DataFrame, w: pd.Series, stat: str, default: float = 0.2) -> pd.Series:
    """No redistribution: each player keeps his usual share no matter who is beside him."""
    d = st_game[["game_id", "player_id", stat]].copy()
    d["pred"] = d["player_id"].map(w).fillna(default) * d[stat]
    return d.groupby(["game_id", "player_id"])["pred"].sum()


def _poisson_dev(y, mu):
    mu = np.maximum(mu, 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(t - (y - mu)))


def calibrate(conn, season: str) -> dict:
    """Fit g and the shrinkage on `season`: weights from games before each cutoff, tested on games after."""
    st, pg = load(conn, [season])
    if st.empty:
        return {}
    dates = sorted(pg["game_date"].unique())
    cuts = [dates[int(len(dates) * f)] for f in (0.25, 0.5, 0.75)]
    nxt = {c: dates[min(len(dates) - 1, dates.index(c) + 30)] for c in cuts}   # test on the next ~30 days
    out = {}
    for stat in STATS:
        best, grid = None, {}
        tests = []
        for c in cuts:
            sb, pb = st[st.game_date < c], pg[pg.game_date < c]
            sa = st[(st.game_date >= c) & (st.game_date < nxt[c])]
            pa = pg[(pg.game_date >= c) & (pg.game_date < nxt[c])].set_index(["game_id", "player_id"])[stat]
            tests.append((sb, pb, sa, pa))
        # top-usage player missing: the team's highest-weight player (by FGA) did not play that game
        for k in K_GRID:
            for g in G_GRID:
                dev = 0.0
                for sb, pb, sa, pa in tests:
                    w = weights(sb, pb, stat, k)
                    pr = predict(sa, w, stat, g).reindex(pa.index).fillna(0)
                    dev += _poisson_dev(pa.to_numpy(float), pr.to_numpy(float))
                grid[f"{k}|{g}"] = dev
                if best is None or dev < best[0]:
                    best = (dev, k, g)
        _, k, g = best
        # report: overall and with the team's top shooter out, against keeping shares fixed
        rep = {"all": [0.0, 0.0, 0], "star_out": [0.0, 0.0, 0]}
        for sb, pb, sa, pa in tests:
            w = weights(sb, pb, stat, k)
            wf = weights(sb, pb, "fga", k)
            minutes = pb.groupby("player_id").size()
            top = {}
            for t, grp in pb.groupby("team_id"):
                cand = [p for p in grp["player_id"].unique() if minutes.get(p, 0) >= 10]
                if cand:
                    top[t] = max(cand, key=lambda p: wf.get(p, 0))
            played = sa.groupby(["game_id", "team_id"])["player_id"].agg(set)
            star_out_games = {(gid, t) for (gid, t), ps in played.items() if t in top and top[t] not in ps}
            pr = predict(sa, w, stat, g).reindex(pa.index).fillna(0)
            pf = fixed_share(sa, w, stat).reindex(pa.index).fillna(0)
            team_of = pg.set_index(["game_id", "player_id"])["team_id"]
            keys = pa.index
            mask = np.array([(gid, team_of.get((gid, p))) in star_out_games for gid, p in keys])
            for name, m in (("all", np.ones(len(keys), bool)), ("star_out", mask)):
                y = pa.to_numpy(float)[m]
                rep[name][0] += float(np.sum((y - pr.to_numpy(float)[m]) ** 2))
                rep[name][1] += float(np.sum((y - pf.to_numpy(float)[m]) ** 2))
                rep[name][2] += int(m.sum())
        out[stat] = {
            "label": STATS[stat][1], "k": k, "g": g,
            "rmse": {n: float(np.sqrt(v[0] / v[2])) if v[2] else None for n, v in rep.items()},
            "rmse_fixed_share": {n: float(np.sqrt(v[1] / v[2])) if v[2] else None for n, v in rep.items()},
            "player_games": {n: v[2] for n, v in rep.items()},
        }
    return {"season": season, "stats": out}


def expected_shares(minutes: dict, w: pd.Series, g: float, default: float = 0.2) -> dict:
    """Pregame: each player's expected share of team totals while he is on the floor.

    minutes: {player_id: expected minutes}. Teammates are assumed to share the floor in
    proportion to their minutes, so a player's four teammates carry on average
    4 * (minutes-weighted mean of the others' weight ** g).
    """
    u = {p: float(w.get(p, default)) ** g for p in minutes}
    tot_min = sum(minutes.values())
    out = {}
    for p, m in minutes.items():
        others = tot_min - m
        if others <= 0:
            out[p] = 0.2
            continue
        mean_other = sum(minutes[q] * u[q] for q in minutes if q != p) / others
        out[p] = u[p] / (u[p] + 4 * mean_other)
    return out
