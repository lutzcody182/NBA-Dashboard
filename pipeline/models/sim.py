"""Game simulation: thousands of games played possession by possession.

Each possession, for the team with the ball:

  1. turnover, at the matchup's turnover rate, ends it with no points
  2. otherwise a "chance": either a two-shot free-throw trip, or a shot
     from one of five zones at the matchup's shot mix, made at that zone's
     matchup make %, with an and-one now and then
  3. a miss is rebounded by the offense at its offensive-rebound rate,
     which gives another chance in the same possession

Rates come from team_rates.matchup (offense vs. that defense). Things the
rates don't know about (home court, rest and travel, who is in or out
tonight) are applied as a shift in expected points per 100 possessions,
by moving every zone's make % on the log-odds scale until the expected
points match.

Two sources of spread are added on top of possession-by-possession luck,
both set from past games rather than by hand: how many possessions the
game has, and a per-game "form" shift for each team (shooting nights,
matchup quirks the rates miss).
"""
import math

import numpy as np

from .team_rates import THREE, ZONES

VALUE = np.array([3.0 if z in THREE else 2.0 for z in ZONES])
MAX_CHANCES = 6


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _inv(x):
    return 1 / (1 + np.exp(-x))


def _arrays(r):
    shares = np.array([r[f"share_{z}"] for z in ZONES])
    makes = np.array([r[f"fg_{z}"] for z in ZONES])
    return shares, makes


def trip_rate(r) -> float:
    """Chance that a scoring chance is a two-shot trip, matching the matchup's free throws per possession."""
    shares, makes = _arrays(r)
    pbar = float(shares @ makes)
    fgm_per_fga = pbar
    # free throws per possession = 2 * trips + and-ones; solve for trips per possession T first
    t, o = r["tov"], r["oreb"]
    # chances per possession E(q) = (1 - t) / (1 - (1 - q) (1 - pbar) o); fga per possession = (1 - q) E
    # fta per possession = 2 q E + and1 * pbar * (1 - q) E
    target = r["fta"]
    lo, hi = 0.0, 0.6
    for _ in range(50):
        q = (lo + hi) / 2
        E = (1 - t) / (1 - (1 - q) * (1 - pbar) * o)
        fta = 2 * q * E + r["and1"] * fgm_per_fga * (1 - q) * E
        lo, hi = (q, hi) if fta < target else (lo, q)
    return (lo + hi) / 2


def expected_ppp(r, shift: float = 0.0, q: float = None) -> float:
    """Expected points per possession for a matchup, with make % moved by `shift` on the log-odds scale."""
    shares, makes = _arrays(r)
    makes = _inv(_logit(makes) + shift)
    q = trip_rate(r) if q is None else q
    pbar = float(shares @ makes)
    E = (1 - r["tov"]) / (1 - (1 - q) * (1 - pbar) * r["oreb"])
    shot_pts = float(shares @ (makes * (VALUE + r["and1"] * r["ftp"])))
    return E * (q * 2 * r["ftp"] + (1 - q) * shot_pts)


def shift_for(r, delta_per100: float) -> float:
    """Log-odds shift of make % that changes expected points per 100 by delta_per100."""
    if abs(delta_per100) < 1e-9:
        return 0.0
    q = trip_rate(r)
    base = expected_ppp(r, 0.0, q)
    target = base + delta_per100 / 100
    lo, hi = -2.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expected_ppp(r, mid, q) < target else (lo, mid)
    return (lo + hi) / 2


def _possessions(r, shift, n_sims, n_poss, rng):
    """Points scored on each of n_poss possessions in each of n_sims games. shift: per-sim log-odds shift."""
    shares, makes = _arrays(r)
    q = trip_rate(r)
    lm = _logit(makes)[None, None, :] + shift[:, None, None]
    pm = _inv(lm)                                              # (n_sims, 1, zones)
    pts = np.zeros((n_sims, n_poss))
    alive = rng.random((n_sims, n_poss)) >= r["tov"]
    cum = np.cumsum(shares)
    for _ in range(MAX_CHANCES):
        if not alive.any():
            break
        u = rng.random((n_sims, n_poss))
        trip = alive & (u < q)
        ft = (rng.random((n_sims, n_poss, 2)) < r["ftp"]).sum(axis=2)
        pts += np.where(trip, ft, 0)
        shot = alive & ~trip
        z = np.searchsorted(cum, rng.random((n_sims, n_poss)) * cum[-1])
        z = np.minimum(z, len(ZONES) - 1)
        p = np.take_along_axis(np.broadcast_to(pm, (n_sims, n_poss, len(ZONES))), z[..., None], axis=2)[..., 0]
        made = shot & (rng.random((n_sims, n_poss)) < p)
        and1 = made & (rng.random((n_sims, n_poss)) < r["and1"]) & (rng.random((n_sims, n_poss)) < r["ftp"])
        pts += np.where(made, VALUE[z], 0) + and1
        alive = shot & ~made & (rng.random((n_sims, n_poss)) < r["oreb"])
    return pts


class _Prepared:
    """Per-matchup constants, worked out once instead of on every possession."""

    def __init__(self, r):
        shares, makes = _arrays(r)
        self.cut = np.cumsum(shares)[:-1] / shares.sum()
        self.lm = _logit(makes)
        self.q, self.tov, self.ftp, self.and1, self.oreb = trip_rate(r), r["tov"], r["ftp"], r["and1"], r["oreb"]


def _one_possession(P: _Prepared, shift, rng):
    """Points on one possession in each simulated game (shift: per-game log-odds shift)."""
    n = len(shift)
    pts = np.zeros(n)
    alive = rng.random(n) >= P.tov
    pz = _inv(P.lm[None, :] + shift[:, None])
    rows = np.arange(n)
    for _ in range(MAX_CHANCES):
        if not alive.any():
            break
        u = rng.random((6, n))
        trip = alive & (u[0] < P.q)
        pts += trip * ((u[1] < P.ftp).astype(float) + (u[2] < P.ftp))
        shot = alive & ~trip
        z = (u[3][:, None] > P.cut[None, :]).sum(axis=1)
        made = shot & (u[4] < pz[rows, z])
        pts += made * (VALUE[z] + ((u[5] < P.and1) & (u[2] < P.ftp)))
        alive = shot & ~made & (rng.random(n) < P.oreb)
    return pts


def simulate(home_rates, away_rates, *, pace: float, home_shift: float = 0.0, away_shift: float = 0.0,
             form_sd: float = 0.0, pace_sd: float = 0.0, lead_effect: float = 0.0,
             n_sims: int = 10000, seed: int = 0, keep: bool = True) -> dict:
    """Play n_sims games. *_rates: that team's offense vs. the other's defense (team_rates.matchup).

    pace: expected possessions per team in 48 minutes. *_shift: log-odds make % shifts from
    shift_for. form_sd: per-game spread of each team's shift. pace_sd: per-game spread of pace.
    lead_effect: log-odds make % change per point of lead (negative: leaders ease off, trailing
    teams push), applied possession by possession.
    """
    rng = np.random.default_rng(seed)
    home_shift, away_shift, form_sd = (np.asarray(x, float) for x in (home_shift, away_shift, form_sd))
    n_max = int(math.ceil(pace + 4 * pace_sd + 4))
    n_game = np.clip(np.rint(pace + pace_sd * rng.standard_normal(n_sims)), 60, n_max).astype(int)
    hs = home_shift + form_sd * rng.standard_normal(n_sims)
    as_ = away_shift + form_sd * rng.standard_normal(n_sims)
    home = np.zeros(n_sims)
    away = np.zeros(n_sims)

    ph, pa = _Prepared(home_rates), _Prepared(away_rates)

    def play(n_steps, mask_steps, idx):
        for k in range(n_steps):
            live = mask_steps[idx] > k
            lead = home[idx] - away[idx]
            h = _one_possession(ph, hs[idx] + lead_effect * lead, rng)
            home[idx] += np.where(live, h, 0)
            lead = home[idx] - away[idx]
            a = _one_possession(pa, as_[idx] - lead_effect * lead, rng)
            away[idx] += np.where(live, a, 0)

    everyone = np.arange(n_sims)
    play(n_max, n_game, everyone)
    ot = np.zeros(n_sims, int)
    n_ot = max(1, int(round(pace * 5 / 48)))
    for _ in range(4):                     # overtimes: 5 minutes at the same pace
        tied = np.flatnonzero(home == away)
        if not len(tied):
            break
        play(n_ot, np.full(n_sims, n_ot), tied)
        ot[tied] += 1
    still = home == away                   # extremely rare after four overtimes: a coin flip
    home[still] += rng.integers(0, 2, int(still.sum())) * 2 - 1
    margin = home - away
    total = home + away
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    out = {
        "n_sims": n_sims,
        "home_win": float((margin > 0).mean()),
        "margin_mean": float(margin.mean()), "margin_sd": float(margin.std()),
        "total_mean": float(total.mean()), "total_sd": float(total.std()),
        "home_pts_mean": float(home.mean()), "away_pts_mean": float(away.mean()),
        "margin_q": {str(q): float(v) for q, v in zip(qs, np.quantile(margin, qs))},
        "total_q": {str(q): float(v) for q, v in zip(qs, np.quantile(total, qs))},
        "home_pts_q": {str(q): float(v) for q, v in zip(qs, np.quantile(home, qs))},
        "away_pts_q": {str(q): float(v) for q, v in zip(qs, np.quantile(away, qs))},
        "overtime": float((ot > 0).mean()),
    }
    if keep:
        out["_margin"], out["_total"], out["_home"], out["_away"] = margin, total, home, away
    return out


def cover_prob(sim: dict, line: float) -> float:
    """Chance the home team wins by more than `line` points (pushes count half)."""
    m = sim["_margin"]
    return float((m > line).mean() + 0.5 * (m == line).mean())


def over_prob(sim: dict, line: float) -> float:
    t = sim["_total"]
    return float((t > line).mean() + 0.5 * (t == line).mean())
