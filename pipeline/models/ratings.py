"""Player ratings: overall impact, offensive and defensive sub-ratings, context.

How it fits together:

1. Every sub-rating stat is shrunk toward league average by an amount
   learned from data (see shrink.py). Calibration uses the most recent
   full season, testing estimates made before a cutoff against what
   happened after it.
2. A box-score model, fit on that season's plus-minus ratings, turns a
   player's shrunk stat profile into a prior for his impact.
3. Regularized plus-minus (rapm.py) on this season plus last season's
   lineup data (at a learned weight) gives the overall offensive and
   defensive ratings, each pulled toward the prior.
4. Context (recent form, minutes and role stability, availability,
   back-to-backs, strong defenses) comes from game logs, with every
   comparison tested against random variation.

Everything learned in steps 1 to 3 is cached per calibration season, and
the out-of-sample test results are exported so the dashboard can show
them.
"""
import json
import logging

import numpy as np
import pandas as pd
from scipy import sparse

from ..util import utc_now_iso
from . import rapm
from .catalog import STAT_BY_KEY, STATS, UNAVAILABLE, _prep
from .features import load_stints, player_games, rest_days
from .shrink import ShrinkFit, calibrate

log = logging.getLogger(__name__)


def _np(o):
    """JSON fallback for NumPy numbers."""
    return o.item() if hasattr(o, "item") else str(o)

PRIOR_FEATURES = ["scoring_vol", "scoring_eff", "three_vol", "three_pt", "assist_rate", "ball_security",
                  "off_reb", "def_reb", "steals", "rim_prot", "fouls", "ft_gen", "usage", "finishing"]
MIN_FULL_SEASON_GAMES = 900
MIN_POSS_FOR_PRIOR_FIT = 300
OVERALL_MIN_POSS = 1000          # pooled possessions before an overall percentile is shown
DEFAULT_LAMBDA, DEFAULT_PREV_W = 4000, 0.5


def prev_season(sid: str) -> str:
    y = int(sid[:4]) - 1
    return f"{y}-{str(y + 1)[-2:]}"


def _season_games(conn):
    return {r[0]: r[1] for r in conn.execute(
        "SELECT season_id, COUNT(*) FROM games WHERE status='final' AND season_type='regular' GROUP BY 1")}


def model_season(conn, season: str):
    """The latest full season at or before `season` (used for calibration)."""
    counts = _season_games(conn)
    full = sorted(s for s, n in counts.items() if n >= MIN_FULL_SEASON_GAMES and s <= season)
    return full[-1] if full else None


def _cutoffs(dates: pd.Series):
    d = sorted(dates.unique())
    if len(d) < 40:
        return []
    return [d[int(len(d) * f)] for f in (0.2, 0.45, 0.7)]


# -- step 1 and 2: shrinkage + box prior ----------------------------------------

def shrunk_profile(now: pd.DataFrame, before: pd.DataFrame, fits: dict, keys=None) -> pd.DataFrame:
    """Shrunk value of every stat for every player, plus evidence counts.

    Returns columns <key> (estimate), <key>_raw, <key>_n (this season tries),
    <key>_n_prev and <key>_weight (share of the estimate from the player).
    """
    keys = keys or [s.key for s in STATS]
    ids = sorted(set(now["player_id"]) | set(before["player_id"] if before is not None else []))
    index = pd.Index(ids, name="player_id")
    cols = {}
    g = now.groupby("player_id")
    gb = before.groupby("player_id") if before is not None and len(before) else None
    zero = pd.Series(0.0, index=index)
    for k in keys:
        s = STAT_BY_KEY[k]
        m = g[s.made].sum().reindex(index, fill_value=0)
        t = g[s.tries].sum().reindex(index, fill_value=0)
        mp = gb[s.made].sum().reindex(index, fill_value=0) if gb is not None else zero
        tp = gb[s.tries].sum().reindex(index, fill_value=0) if gb is not None else zero
        league = (m.sum() / t.sum()) if t.sum() else fits[k].league
        f = fits[k]
        cols[k] = f.apply(m, t, mp, tp, league=league)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"{k}_raw"] = np.where(t > 0, m / t, np.nan)
        cols[f"{k}_n"] = t.to_numpy()
        cols[f"{k}_n_prev"] = tp.to_numpy()
        cols[f"{k}_weight"] = f.weight_on_player(t, tp)
        cols[f"{k}_league"] = np.full(len(index), league)
    mins = g["min"].sum().reindex(index, fill_value=0)
    gp = g.size().reindex(index, fill_value=0).astype(float)
    if gb is not None:
        mins = mins + 0.5 * gb["min"].sum().reindex(index, fill_value=0)
        gp = gp + 0.5 * gb.size().reindex(index, fill_value=0)
    cols["mpg"] = np.where(gp > 0, mins / gp.replace(0, np.nan), 0.0)
    cols["on_poss"] = g["on_poss"].sum().reindex(index, fill_value=0).to_numpy()
    cols["on_poss_prev"] = (gb["on_poss"].sum().reindex(index, fill_value=0).to_numpy()
                            if gb is not None else np.zeros(len(index)))
    out = pd.DataFrame(cols, index=index)
    return out


def _prior_matrix(profile: pd.DataFrame) -> np.ndarray:
    cols = [profile[k] - profile[f"{k}_league"] for k in PRIOR_FEATURES]
    cols.append(profile["mpg"] - 20.0)
    return np.c_[np.ones(len(profile)), np.column_stack(cols)]


def fit_box_prior(stints: pd.DataFrame, games: pd.DataFrame, fits: dict, lam: float):
    """Regress one season's plain plus-minus on that season's shrunk box profile."""
    pure = rapm.fit(stints, np.ones(len(stints)), lam=lam).as_frame().set_index("player_id")
    prof = shrunk_profile(games, None, fits, PRIOR_FEATURES)
    d = prof.join(pure, how="inner")
    d = d[d["on_poss"] >= MIN_POSS_FOR_PRIOR_FIT]
    X = _prior_matrix(d)
    w = d["on_poss"].to_numpy(float)
    ridge = 1e-3 * np.eye(X.shape[1])
    XtW = X.T * w
    coef_o = np.linalg.solve(XtW @ X + ridge, XtW @ d["o_rapm"].to_numpy())
    coef_d = np.linalg.solve(XtW @ X + ridge, XtW @ d["d_rapm"].to_numpy())

    def r2(y, p):
        mu = np.average(y, weights=w)
        return float(1 - np.average((y - p) ** 2, weights=w) / np.average((y - mu) ** 2, weights=w))

    return {
        "coef_off": coef_o.tolist(), "coef_def": coef_d.tolist(),
        "features": ["const"] + PRIOR_FEATURES + ["mpg"],
        "r2_off": r2(d["o_rapm"].to_numpy(), X @ coef_o), "r2_def": r2(d["d_rapm"].to_numpy(), X @ coef_d),
        "n_players": int(len(d)),
    }


def apply_box_prior(profile: pd.DataFrame, box: dict):
    X = _prior_matrix(profile)
    po = X @ np.asarray(box["coef_off"])
    pdef = X @ np.asarray(box["coef_def"])      # + means better defense
    return dict(zip(profile.index, po)), dict(zip(profile.index, -pdef))   # rapm.fit wants + = allows more


# -- calibration (cached) -------------------------------------------------------

def calibrate_all(conn, season_m: str) -> dict:
    """Learn shrinkage, the box prior and the plus-minus settings on a full season."""
    prev_m = prev_season(season_m)
    pg = _prep(player_games(conn, [prev_m, season_m]))
    cur, prev = pg[pg["season_id"] == season_m], pg[pg["season_id"] == prev_m]
    has_prev = len(prev) > 0
    cuts = _cutoffs(cur["game_date"])
    fits = {s.key: calibrate(cur, prev if has_prev else None, s.made, s.tries, cuts) for s in STATS}

    st = load_stints(conn, [prev_m, season_m])
    st_cur, st_prev = st[st["season_id"] == season_m], st[st["season_id"] == prev_m]
    box_m = fit_box_prior(st_cur, cur, fits, lam=2000)

    # Choose lambda and last-season weight by predicting the last part of the season.
    split = cuts[1] if cuts else None
    tests, choice = {}, {"lambda": DEFAULT_LAMBDA, "prev_weight": DEFAULT_PREV_W, "source": "default"}
    if split is not None:
        train, test = st_cur[st_cur["game_date"] < split], st_cur[st_cur["game_date"] >= split]
        g_train = cur[cur["game_date"] < split]
        lg = np.average(100 * train["pts"][train["poss"] > 0] / train["poss"][train["poss"] > 0],
                        weights=train["poss"][train["poss"] > 0])
        tests["league_average"] = rapm.weighted_mse(test, np.full(len(test), lg))
        tt, te = train.copy(), test.copy()
        for d in (tt, te):
            d["lineup_key"], d["opp_lineup_key"] = d["team_id"].astype(str), d["opp_id"].astype(str)
        f_team = rapm.fit(tt, np.ones(len(tt)), lam=500)
        tests["team_ratings"] = rapm.weighted_mse(te, rapm.predict(f_team, te))
        prior_o = prior_d = None
        if has_prev:
            box_prev = fit_box_prior(st_prev, prev, fits, lam=2000)
            prof = shrunk_profile(g_train, prev, fits, PRIOR_FEATURES)
            prior_o, prior_d = apply_box_prior(prof, box_prev)
            f_box = rapm.fit(train, np.ones(len(train)), lam=1e9, prior_off=prior_o, prior_def=prior_d)
            tests["box_score_only"] = rapm.weighted_mse(test, rapm.predict(f_box, test, prior_o, prior_d))
        both = pd.concat([st_prev, train]) if has_prev else train
        best = None
        for pw in (rapm.PREV_WEIGHT_GRID if has_prev else (0.0,)):
            w = np.where(both["season_id"] == prev_m, pw, 1.0)
            for lam in rapm.LAMBDA_GRID:
                f = rapm.fit(both, w, lam=lam, prior_off=prior_o, prior_def=prior_d)
                e = rapm.weighted_mse(test, rapm.predict(f, test, prior_o, prior_d))
                tests[f"rapm_prior|{pw}|{lam}"] = e
                if best is None or e < best[0]:
                    best = (e, pw, lam)
        if not has_prev:
            f = rapm.fit(train, np.ones(len(train)), lam=4000)
            tests["plain_rapm"] = rapm.weighted_mse(test, rapm.predict(f, test))
        else:
            w = np.where(both["season_id"] == prev_m, best[1], 1.0)
            f = rapm.fit(both, w, lam=best[2])
            tests["plain_rapm"] = rapm.weighted_mse(test, rapm.predict(f, test))
        choice = {"lambda": best[2], "prev_weight": best[1], "source": "out-of-sample test",
                  "split_date": split, "test_stints": int(len(test))}

    base = tests.get("league_average")
    summary = {}
    if base:
        for name in ("team_ratings", "plain_rapm", "box_score_only"):
            if name in tests:
                summary[name] = base - tests[name]
        summary["final_model"] = base - min(v for k, v in tests.items() if k.startswith("rapm_prior"))
    return {
        "season": season_m, "prev_season": prev_m if has_prev else None,
        "shrink": {k: f.to_dict() for k, f in fits.items()},
        "box_prior": box_m, "rapm": choice, "tests": tests, "signal_captured": summary,
        "created_at": utc_now_iso(),
    }


def get_calibration(conn, season_m: str, force=False) -> dict:
    key = f"ratings:{season_m}"
    row = conn.execute("SELECT payload FROM model_calibration WHERE key = ?", (key,)).fetchone()
    if row and not force:
        return json.loads(row[0])
    cal = calibrate_all(conn, season_m)
    conn.execute("INSERT OR REPLACE INTO model_calibration (key, payload, created_at) VALUES (?, ?, ?)",
                 (key, json.dumps(cal, default=_np), cal["created_at"]))
    conn.commit()
    return cal


def _fits_from(cal) -> dict:
    out = {}
    for k, d in cal["shrink"].items():
        out[k] = ShrinkFit(k=d["k"], w=d["w"], league=d["league"], error=d["error"], error_raw=d["error_raw"],
                           error_league=d["error_league"], n_players=d["n_players"], note=d.get("note", ""))
    return out


# -- step 3: final ratings -------------------------------------------------------

def _posterior_sd(stints, weights, fit_, prior_o, prior_d):
    """Approximate standard error of each player's offensive and defensive rating."""
    y, w, off, dfn, home = rapm._rows(stints, weights)
    index = {int(p): i for i, p in enumerate(fit_.players)}
    X = rapm._design(off, dfn, home, index)
    n = len(index)
    beta = np.r_[fit_.off, fit_.deff, fit_.home]
    resid = (y - fit_.league) - X @ beta
    sigma2 = float(np.sum(w * resid ** 2) / max(len(y) - 1, 1))
    A = (X.T @ sparse.diags(w) @ X).toarray() + np.diag(np.r_[np.full(2 * n, fit_.lam), 1e-3])
    try:
        cov_diag = np.diag(np.linalg.inv(A))
    except np.linalg.LinAlgError:
        return np.full(n, np.nan), np.full(n, np.nan)
    sd = np.sqrt(sigma2 * cov_diag)
    return sd[:n], sd[n:2 * n]


def _percentile(values: pd.Series, eligible: pd.Series, better: int) -> pd.Series:
    v = values.where(eligible)
    pct = (v * better).rank(pct=True) * 100
    return pct.where(eligible)


def _ebayes_split(pg, mask_a, mask_b, made, tries):
    """Shrunk difference of a rate between two kinds of games (normal-normal empirical Bayes).

    Returns per-player shrunk difference, its raw value, sample sizes and the
    estimated spread of true differences across players (tau). A tau of 0
    means the data show no reliable player-to-player differences at all.
    """
    a = pg[mask_a].groupby("player_id")[[made, tries]].sum()
    b = pg[mask_b].groupby("player_id")[[made, tries]].sum()
    d = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    d = d[(d[f"{tries}_a"] >= 20) & (d[f"{tries}_b"] >= 20)]
    if d.empty:
        return pd.DataFrame(), 0.0
    ra, rb = d[f"{made}_a"] / d[f"{tries}_a"], d[f"{made}_b"] / d[f"{tries}_b"]
    p = (d[f"{made}_a"] + d[f"{made}_b"]) / (d[f"{tries}_a"] + d[f"{tries}_b"])
    var = p * (1 - p).clip(lower=0.05) * (1 / d[f"{tries}_a"] + 1 / d[f"{tries}_b"])
    diff = ra - rb
    tau2 = max(0.0, float(np.var(diff) - np.mean(var)))
    shrunk = diff * tau2 / (tau2 + var) if tau2 > 0 else diff * 0
    out = pd.DataFrame({"raw": diff, "shrunk": shrunk, "se": np.sqrt(var),
                        "n_a": d[f"{tries}_a"], "n_b": d[f"{tries}_b"]})
    return out, float(np.sqrt(tau2))


def context_ratings(pg_cur: pd.DataFrame, pg_both: pd.DataFrame, team_games: pd.DataFrame,
                    strong_def_teams: set) -> pd.DataFrame:
    """Recent form, minutes and role stability, availability, fatigue, strong defenses."""
    rows = {}
    pg_both = pg_both.sort_values(["game_date", "game_id"])
    for pid, g in pg_both.groupby("player_id"):
        cur = g[g["season_id"] == pg_cur["season_id"].iloc[0]] if len(pg_cur) else g.iloc[:0]
        last15 = g.tail(15)
        r = {"games_cur": int(len(cur)), "games_total": int(len(g))}
        r["min_sd_last15"] = float(last15["min"].std(ddof=1)) if len(last15) >= 5 else None
        r["min_avg_last15"] = float(last15["min"].mean()) if len(last15) else None
        r["starts_last15"] = int(last15["started"].sum())
        r["games_last15"] = int(len(last15))
        # Recent form: last 10 vs everything before, for four box-score signals.
        if len(g) >= 20:
            recent, base = g.tail(10), g.iloc[:-10]
            form = {}
            for key, num, den, scale in (("min", "min", None, 1), ("pts36", "pts", "min", 36),
                                         ("ts", "ts_pts", "tsa", 100), ("usage", "usage_n", "on_poss", 100)):
                if den is None:
                    a, b = recent[num], base[num]
                    diff = a.mean() - b.mean()
                    se = np.sqrt(b.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                else:
                    ra = recent[num].sum() / recent[den].sum() if recent[den].sum() else np.nan
                    rb = base[num].sum() / base[den].sum() if base[den].sum() else np.nan
                    per_game = (base[num] / base[den].replace(0, np.nan)).dropna()
                    w = base[den][per_game.index]
                    var_game = np.average((per_game - rb) ** 2, weights=w) if len(per_game) > 3 else np.nan
                    mean_w = w.mean() if len(w) else np.nan
                    diff = ra - rb
                    se = np.sqrt(var_game * mean_w * (1 / max(recent[den].sum(), 1e-9) + 1 / max(base[den].sum(), 1e-9)))
                z = diff / se if se and se > 0 else None
                form[key] = {"recent": float((recent[num].mean() if den is None else
                                              recent[num].sum() / max(recent[den].sum(), 1e-9)) * scale),
                             "baseline": float((base[num].mean() if den is None else
                                                base[num].sum() / max(base[den].sum(), 1e-9)) * scale),
                             "z": None if z is None or not np.isfinite(z) else float(z)}
            r["form"] = form
        rows[pid] = r
    ctx = pd.DataFrame.from_dict(rows, orient="index")
    ctx.index.name = "player_id"

    # Availability: games played out of his team's games since he first appeared for them this season.
    if len(pg_cur):
        avail = {}
        tg = team_games.sort_values("game_date")
        for (pid, tid), g in pg_cur.groupby(["player_id", "team_id"]):
            first = g["game_date"].min()
            last_team_game = tg[(tg["team_id"] == tid)]["game_date"].max()
            span = tg[(tg["team_id"] == tid) & (tg["game_date"] >= first) & (tg["game_date"] <= last_team_game)]
            avail.setdefault(pid, [0, 0])
            avail[pid][0] += len(g)
            avail[pid][1] += len(span)
        ctx["avail_played"] = pd.Series({k: v[0] for k, v in avail.items()})
        ctx["avail_possible"] = pd.Series({k: v[1] for k, v in avail.items()})

    # Fatigue: true shooting on back-to-backs vs with rest (empirical Bayes, both seasons).
    b2b, tau_b2b = _ebayes_split(pg_both, pg_both["rest"] == 1, pg_both["rest"] >= 2, "ts_pts", "tsa")
    if len(b2b):
        ctx = ctx.join(b2b.add_prefix("b2b_"), how="left")
    vs, tau_vs = _ebayes_split(pg_both, pg_both["opp_id"].isin(strong_def_teams),
                               ~pg_both["opp_id"].isin(strong_def_teams), "ts_pts", "tsa")
    if len(vs):
        ctx = ctx.join(vs.add_prefix("vsdef_"), how="left")
    ctx.attrs["tau_b2b"] = tau_b2b
    ctx.attrs["tau_vsdef"] = tau_vs
    return ctx


def compute(conn, season: str, team_table=None) -> dict:
    """All player ratings for `season`, ready to export."""
    m = model_season(conn, season)
    if m is None:
        return {"players": [], "model": {"note": "Ratings need at least one full season of play-by-play."}}
    cal = get_calibration(conn, m)
    fits = _fits_from(cal)
    prev = prev_season(season)
    pg = _prep(player_games(conn, [prev, season]))
    pg["rest"] = rest_days(pg).fillna(3)
    cur, before = pg[pg["season_id"] == season], pg[pg["season_id"] == prev]
    profile = shrunk_profile(cur, before, fits)

    # Priors come from the box model fit on the calibration season.
    prior_o, prior_d = apply_box_prior(profile, cal["box_prior"])
    st = load_stints(conn, [prev, season])
    lam, pw = cal["rapm"]["lambda"], cal["rapm"]["prev_weight"]
    weights = np.where(st["season_id"] == prev, pw, 1.0)
    fit_ = rapm.fit(st, weights, lam=lam, prior_off=prior_o, prior_def=prior_d)
    sd_o, sd_d = _posterior_sd(st, weights, fit_, prior_o, prior_d)
    imp = fit_.as_frame().set_index("player_id")
    imp["o_sd"], imp["d_sd"] = sd_o, sd_d
    imp["net_sd"] = np.sqrt(sd_o ** 2 + sd_d ** 2)
    imp["prior_o"] = [prior_o.get(p, 0.0) for p in imp.index]
    imp["prior_d"] = [-prior_d.get(p, 0.0) for p in imp.index]
    profile = profile.join(imp, how="left")

    # Context
    team_games = pd.read_sql_query(
        """SELECT t.team_id, g.game_date FROM team_game_stats t JOIN games g ON g.game_id = t.game_id
           WHERE g.season_id = ? AND g.status = 'final' AND g.season_type = 'regular'""", conn, params=(season,))
    strong = set()
    if team_table and team_table.get("teams"):
        ranked = sorted(team_table["teams"], key=lambda r: r["adj_drtg"])
        strong = {r["team_id"] for r in ranked[:10]}
    ctx = context_ratings(cur, pg, team_games, strong)

    # Latest team for each player
    last_team = pg.sort_values("game_date").groupby("player_id").tail(1).set_index("player_id")[["team_id", "game_date"]]

    profile = profile.copy()
    pooled_poss = profile["on_poss"] + pw * profile["on_poss_prev"]
    profile["pooled_poss"] = pooled_poss
    elig_overall = pooled_poss >= OVERALL_MIN_POSS
    pct = {}
    for s in STATS:
        evidence = profile[f"{s.key}_n"] + fits[s.key].w * profile[f"{s.key}_n_prev"]
        pct[s.key] = _percentile(profile[s.key], (evidence >= s.min_tries) & elig_overall, s.better)
    pct["overall"] = _percentile(profile["net_rapm"], elig_overall, 1)
    pct["offense"] = _percentile(profile["o_rapm"], elig_overall, 1)
    pct["defense"] = _percentile(profile["d_rapm"], elig_overall, 1)

    players = []
    for pid, row in profile.iterrows():
        if pd.isna(row.get("net_rapm")):
            continue
        c = ctx.loc[pid] if pid in ctx.index else None
        lt = last_team.loc[pid] if pid in last_team.index else None
        rec = {
            "player_id": int(pid),
            "team_id": int(lt["team_id"]) if lt is not None else None,
            "last_game": lt["game_date"] if lt is not None else None,
            "games": int(c["games_cur"]) if c is not None else 0,
            "mpg": float(row["mpg"]),
            "poss": float(row["on_poss"]), "poss_prev": float(row["on_poss_prev"]),
            "overall": {"value": row["net_rapm"], "sd": row["net_sd"], "pct": pct["overall"].get(pid)},
            "offense": {"value": row["o_rapm"], "sd": row["o_sd"], "prior": row["prior_o"], "pct": pct["offense"].get(pid)},
            "defense": {"value": row["d_rapm"], "sd": row["d_sd"], "prior": row["prior_d"], "pct": pct["defense"].get(pid)},
            "stats": {},
        }
        for s in STATS:
            # compact: [estimate, raw this season, tries this season, tries last season,
            #           share of estimate from the player, league average, percentile]
            rec["stats"][s.key] = [
                row[s.key] * s.scale, None if pd.isna(row[f"{s.key}_raw"]) else row[f"{s.key}_raw"] * s.scale,
                round(float(row[f"{s.key}_n"]), 1), round(float(row[f"{s.key}_n_prev"]), 1),
                row[f"{s.key}_weight"], row[f"{s.key}_league"] * s.scale, pct[s.key].get(pid),
            ]
        if c is not None:
            rec["context"] = _context_record(c, ctx.attrs)
        players.append(rec)

    return {
        "season": season,
        "players": players,
        "model": {
            "calibration_season": m,
            "calibration": {k: v for k, v in cal.items() if k not in ("tests",)},
            "tests": cal["tests"],
            "lambda": lam, "prev_weight": pw, "league_ortg": fit_.league, "home_edge_per_100": 2 * fit_.home,
            "overall_min_poss": OVERALL_MIN_POSS,
            "stats": [{"key": s.key, "label": s.label, "group": s.group, "unit": s.unit, "better": s.better,
                       "how": s.how, "min_tries": s.min_tries, "tries_label": s.tries_label, "k": cal["shrink"][s.key]["k"],
                       "w": cal["shrink"][s.key]["w"],
                       "improvement_vs_raw": cal["shrink"][s.key].get("improvement_vs_raw")} for s in STATS],
            "stat_fields": ["value", "raw", "n", "n_prev", "weight", "league", "pct"],
            "unavailable": [{"key": k, "label": v[0], "why": v[1]} for k, v in UNAVAILABLE.items()],
            "tau_b2b": ctx.attrs.get("tau_b2b"), "tau_vsdef": ctx.attrs.get("tau_vsdef"),
            "strong_defenses": sorted(strong),
        },
    }


def _context_record(c, attrs) -> dict:
    def g(k):
        v = c.get(k) if hasattr(c, "get") else None
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return v
    rec = {
        "min_sd_last15": g("min_sd_last15"), "min_avg_last15": g("min_avg_last15"),
        "starts_last15": g("starts_last15"), "games_last15": g("games_last15"),
        "avail_played": g("avail_played"), "avail_possible": g("avail_possible"),
        "form": g("form"),
    }
    if g("b2b_shrunk") is not None:
        rec["b2b"] = {"raw": g("b2b_raw") * 100, "shrunk": g("b2b_shrunk") * 100, "se": g("b2b_se") * 100,
                      "n_b2b": g("b2b_n_a"), "n_rest": g("b2b_n_b"), "tau": attrs.get("tau_b2b", 0) * 100}
    if g("vsdef_shrunk") is not None:
        rec["vs_strong_def"] = {"raw": g("vsdef_raw") * 100, "shrunk": g("vsdef_shrunk") * 100,
                                "se": g("vsdef_se") * 100, "n_strong": g("vsdef_n_a"), "n_other": g("vsdef_n_b"),
                                "tau": attrs.get("tau_vsdef", 0) * 100}
    return rec
