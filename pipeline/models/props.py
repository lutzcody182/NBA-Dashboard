"""Chance of going over a player prop line.

The game simulation gives each player's expected numbers. A line asks a
different question: how likely is he to clear 24.5? That needs the whole
spread of outcomes, built here in two layers:

  * minutes: the minutes model's own 10th-90th percentile range for this
    player if he plays (books void a prop when the player sits), as a
    two-sided normal around the projection
  * the stat at a given minutes total: negative binomial with mean
    rate x minutes. Counts like points and rebounds vary more than a
    Poisson would allow, and the extra spread, k, is fitted per stat on a
    full season of real games: for each player-game, his rate over his
    previous 20 games times the minutes he actually played, against what
    he actually did. Fitting at known minutes keeps minutes noise out of k,
    since the first layer already carries it.

calibration_check() then rebuilds the full chain on that season with the
previous-10-game minutes average standing in for the projection, and
compares predicted chances of going over with how often players did.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.special import gammaln

STATS = {
    "pts": ("pts",), "reb": ("reb",), "ast": ("ast",), "fg3m": ("fg3m",),
    "pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast"), "ra": ("reb", "ast"),
}
K_GRID = np.round(np.exp(np.linspace(np.log(0.5), np.log(400), 60)), 3)
PRIOR_MIN = 60.0          # minutes of league-average play mixed into each rate
WINDOW = 20
Z90 = 1.2816


def _with_stats(logs: pd.DataFrame) -> pd.DataFrame:
    df = logs.copy()
    for s, parts in STATS.items():
        if s not in df or len(parts) > 1:
            df[s] = sum(df[p] for p in parts)
    return df


def prior_rates(logs: pd.DataFrame) -> pd.DataFrame:
    """Per-minute rate for every stat from each player's previous WINDOW games."""
    df = _with_stats(logs).sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    g = df.groupby("player_id")
    df["prior_min"] = g["min"].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).sum())
    df["l10_min"] = g["min"].transform(lambda s: s.shift().rolling(10, min_periods=3).mean())
    lg = {s: df[s].sum() / df["min"].sum() for s in STATS}
    for s in STATS:
        prior = g[s].transform(lambda x: x.shift().rolling(WINDOW, min_periods=1).sum())
        df[f"rate_{s}"] = (prior.fillna(0) + PRIOR_MIN * lg[s]) / (df["prior_min"].fillna(0) + PRIOR_MIN)
    return df


def nb_logpmf(x, mu, k):
    mu = np.maximum(mu, 1e-6)
    return (gammaln(x + k) - gammaln(k) - gammaln(x + 1) + k * np.log(k / (k + mu)) + x * np.log(mu / (k + mu)))


def fit_dispersion(logs: pd.DataFrame, min_minutes=8.0, min_prior=100.0) -> dict:
    df = prior_rates(logs)
    df = df[(df["min"] >= min_minutes) & (df["prior_min"] >= min_prior)]
    out = {}
    for s in STATS:
        x, mu = df[s].to_numpy(float), (df[f"rate_{s}"] * df["min"]).to_numpy(float)
        ll = [float(nb_logpmf(x, mu, k).sum()) for k in K_GRID]
        i = int(np.argmax(ll))
        pois = float(st.poisson.logpmf(x, np.maximum(mu, 1e-6)).sum())
        out[s] = {"k": float(K_GRID[i]), "n": int(len(x)), "loglik_per_game": ll[i] / len(x),
                  "poisson_loglik_per_game": pois / len(x), "at_grid_edge": i in (0, len(K_GRID) - 1)}
    return out


def minutes_points(mean: float, p10: float, p90: float, n: int = 15):
    """Quadrature points (minutes, weight): a two-sided normal with the p10/p90 spread, averaging `mean`.

    The minutes model predicts an average and its range comes from residual
    quantiles, so the points are shifted until they average `mean`.
    """
    lo_sd = max((mean - p10) / Z90, 0.5)
    hi_sd = max((p90 - mean) / Z90, 0.5)
    qs = (np.arange(n) + 0.5) / n
    z = st.norm.ppf(qs)
    m = np.where(z < 0, z * lo_sd, z * hi_sd)
    m = m - m.mean() + mean
    return np.clip(m, 0.5, 48.0), np.full(n, 1.0 / n)


G_GRID = np.round(np.arange(0.0, 3.01, 0.1), 2)


def _cdf(x, mu, shape: dict):
    """P(X <= x) for a continuous x (lines sit on half points), at mean(s) mu."""
    mu = np.maximum(mu, 1e-6)
    if shape["shape"] == "nb":
        k = shape["k"]
        return st.nbinom.cdf(np.floor(x), k, k / (k + mu))
    # shifted gamma with mean mu, variance a*mu + b*mu^2 and skew g*sd/mu
    var = np.maximum(shape["a"] * mu + shape["b"] * mu ** 2, 0.05)
    sd = np.sqrt(var)
    skew = np.maximum(shape["g"] * sd / mu, 1e-3)
    alpha = 4.0 / skew ** 2
    theta = sd * skew / 2.0
    loc = mu - alpha * theta
    return st.gamma.cdf(x - loc, alpha, scale=theta)


def over_under(line: float, rate: float, shape, min_mean: float, p10: float, p90: float):
    """(P over, P under, P push) for one line. `shape`: a fitted shape dict (or a bare NB k)."""
    if not isinstance(shape, dict):
        shape = {"shape": "nb", "k": float(shape)}
    mins, w = minutes_points(min_mean, p10, p90)
    mu = rate * mins
    if float(line).is_integer():
        under = float(w @ _cdf(line - 0.5, mu, shape))
        push = float(w @ _cdf(line + 0.5, mu, shape)) - under
    else:
        under = float(w @ _cdf(line, mu, shape))
        push = 0.0
    return 1.0 - under - push, under, push


def _known_minutes(logs, min_minutes=15.0, seed=1):
    """Player-games with mean = prior rate x actual minutes, and a line near that mean."""
    df = prior_rates(logs)
    df = df[(df["min"] >= min_minutes) & (df["prior_min"] >= 100)]
    rng = np.random.default_rng(seed)
    out = {}
    for s in STATS:
        mu = (df[f"rate_{s}"] * df["min"]).to_numpy(float)
        x = df[s].to_numpy(float)
        line = np.floor(mu + rng.uniform(-0.5, 0.5, len(mu)) * np.maximum(1.0, 0.25 * mu)) + 0.5
        ok = line > 0
        out[s] = (mu[ok], x[ok], line[ok])
    return out


def _brier(shape, mu, x, line):
    p = 1 - _cdf(line, mu, shape)
    return float(np.mean((p - (x > line)) ** 2)), float(p.mean())


def fit_skewed(mu, x, line) -> dict:
    """Variance a*mu + b*mu^2 from the residuals, then the skew that best predicts overs."""
    A = np.vstack([mu, mu ** 2]).T
    a, b = np.linalg.lstsq(A, (x - mu) ** 2, rcond=None)[0]
    a, b = float(max(a, 0.2)), float(max(b, 0.0))
    best = min(((_brier({"shape": "skewed", "a": a, "b": b, "g": g}, mu, x, line)[0], g) for g in G_GRID if g > 0))
    return {"shape": "skewed", "a": round(a, 4), "b": round(b, 5), "g": float(best[1])}


def fit_shapes(train_logs: pd.DataFrame, test_logs: pd.DataFrame) -> dict:
    """Per stat: fit both shapes on one season, keep whichever predicts overs better on the next.

    The kept shape is then refitted on the newer season for live use.
    """
    nb_tr, nb_te = fit_dispersion(train_logs), fit_dispersion(test_logs)
    tr, te = _known_minutes(train_logs), _known_minutes(test_logs, seed=2)
    out = {}
    for s in STATS:
        cand = {"nb": {"shape": "nb", "k": nb_tr[s]["k"]}, "skewed": fit_skewed(*tr[s])}
        mu, x, line = te[s]
        scores = {n: _brier(c, mu, x, line) for n, c in cand.items()}
        pick = min(scores, key=lambda n: scores[n][0])
        live = {"shape": "nb", "k": nb_te[s]["k"]} if pick == "nb" else fit_skewed(*te[s])
        out[s] = {**live, "n": int(len(mu)), "held_out": {
            n: {"brier": round(v[0], 5), "predicted_over": round(v[1], 4)} for n, v in scores.items()},
            "actual_over": round(float((x > line).mean()), 4), "trained_on": "previous season",
            "tested_on": "next season"}
    return out


def calibration_check(logs: pd.DataFrame, shapes: dict, min_sd: float = None) -> dict:
    """Predicted vs actual chance of going over on a past season, minutes not known in advance."""
    df = prior_rates(logs)
    df = df[(df["prior_min"] >= 100) & df["l10_min"].notna() & (df["l10_min"] >= 15) & (df["min"] > 0)].copy()
    if min_sd is None:
        min_sd = float(np.std(df["min"] - df["l10_min"]))
    out = {}
    rng = np.random.default_rng(7)
    for s in ("pts", "reb", "ast", "fg3m", "pra"):
        shape = shapes[s]
        sub = df.sample(min(len(df), 6000), random_state=7)
        preds, hits = [], []
        for r in sub.itertuples():
            rate = getattr(r, f"rate_{s}")
            mean = rate * r.l10_min
            line = math.floor(mean + rng.uniform(-0.5, 0.5) * max(1.0, 0.25 * mean)) + 0.5
            if line < 0.5:
                continue
            po, pu, _ = over_under(line, rate, shape, r.l10_min, r.l10_min - Z90 * min_sd, r.l10_min + Z90 * min_sd)
            preds.append(po)
            hits.append(float(getattr(r, s) > line))
        preds, hits = np.array(preds), np.array(hits)
        bins = np.clip((preds * 10).astype(int), 0, 9)
        table = [{"bin": f"{b * 10}-{b * 10 + 10}%", "n": int((bins == b).sum()),
                  "predicted": round(float(preds[bins == b].mean()), 3),
                  "actual": round(float(hits[bins == b].mean()), 3)} for b in range(10) if (bins == b).sum() >= 30]
        out[s] = {"n": int(len(preds)), "brier": round(float(np.mean((preds - hits) ** 2)), 4),
                  "brier_coinflip": 0.25, "mean_pred": round(float(preds.mean()), 3),
                  "mean_actual": round(float(hits.mean()), 3), "bins": table}
    return {"minutes_sd": round(min_sd, 2), "stats": out}


def devig(over_price: float, under_price: float):
    """Fair over chance from a two-way American-odds pair (proportional margin removal)."""
    po, pu = implied(over_price), implied(under_price)
    if po is None or pu is None or po + pu <= 0:
        return None, None
    return po / (po + pu), po + pu - 1.0


def implied(price):
    if price is None:
        return None
    price = float(price)
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def decimal(price):
    price = float(price)
    return 1 + price / 100.0 if price > 0 else 1 + 100.0 / -price
