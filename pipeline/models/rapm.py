"""Overall impact ratings: regularized plus-minus with a box-score prior.

For every stretch of play (a lineup stint), the offense's points per 100
possessions is modelled as

    league average + sum(offense ratings of its 5 players)
                   + sum(defense ratings of the 5 defenders) + home edge

and solved by ridge regression weighted by possessions. Each player's
ratings are pulled toward a prior: what his box-score profile says a
player like him is usually worth (fit on earlier seasons). Players with
lots of minutes move away from the prior; players with few stay close.

The prior strength (lambda) and how much last season counts are chosen
by predicting stints the model has not seen.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

LAMBDA_GRID = (1000, 2000, 4000, 8000, 16000)
PREV_WEIGHT_GRID = (0.25, 0.5, 1.0)


@dataclass
class RapmFit:
    players: np.ndarray          # player ids, column order
    off: np.ndarray              # points per 100 added on offense (+ is good)
    deff: np.ndarray             # points per 100 allowed on defense (- is good)
    home: float                  # home edge per 100 for the offense
    league: float                # league points per 100
    lam: float
    prev_weight: float

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"player_id": self.players, "o_rapm": self.off, "d_rapm": -self.deff,
                             "net_rapm": self.off - self.deff})


def _rows(stints: pd.DataFrame, weights: np.ndarray):
    s = stints[stints["poss"] > 0]
    w = weights[stints["poss"].to_numpy() > 0] * s["poss"].to_numpy()
    y = 100.0 * s["pts"].to_numpy() / s["poss"].to_numpy()
    off = s["lineup_key"].str.split("-").tolist()
    dfn = s["opp_lineup_key"].str.split("-").tolist()
    home = np.where(s["is_home"].to_numpy() == 1, 1.0, -1.0)
    return y, w, off, dfn, home


def _design(off, dfn, home, index):
    n_rows, n = len(off), len(index)
    rows, cols = [], []
    for i, (o, d) in enumerate(zip(off, dfn)):
        for p in o:
            rows.append(i)
            cols.append(index[int(p)])
        for p in d:
            rows.append(i)
            cols.append(n + index[int(p)])
    data = np.ones(len(rows))
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, 2 * n + 1))
    X = X + sparse.csr_matrix((home, (np.arange(n_rows), np.full(n_rows, 2 * n))), shape=X.shape)
    return X


def fit(stints: pd.DataFrame, weights: np.ndarray, lam: float, prior_off=None, prior_def=None,
        prev_weight: float = 1.0, players=None) -> RapmFit:
    """Solve the ridge problem. prior_* are dicts player_id -> prior value (default 0)."""
    y, w, off, dfn, home = _rows(stints, weights)
    if players is None:
        ids = sorted({int(p) for lst in off for p in lst} | {int(p) for lst in dfn for p in lst})
    else:
        ids = list(players)
    index = {p: i for i, p in enumerate(ids)}
    X = _design(off, dfn, home, index)
    league = float(np.average(y, weights=w))
    n = len(ids)
    prior = np.zeros(2 * n + 1)
    if prior_off:
        prior[:n] = [prior_off.get(p, 0.0) for p in ids]
    if prior_def:
        prior[n:2 * n] = [prior_def.get(p, 0.0) for p in ids]
    W = sparse.diags(w)
    XtW = X.T @ W
    pen = np.full(2 * n + 1, float(lam))
    pen[-1] = 1e-3
    A = (XtW @ X) + sparse.diags(pen)
    b = XtW @ (y - league) + pen * prior
    beta = spsolve(A.tocsc(), b)
    return RapmFit(players=np.array(ids), off=beta[:n], deff=beta[n:2 * n], home=float(beta[-1]),
                   league=league, lam=lam, prev_weight=prev_weight)


def predict(fit_: RapmFit, stints: pd.DataFrame, prior_off=None, prior_def=None) -> np.ndarray:
    """Predicted points per 100 for each stint (players the fit never saw get their prior)."""
    o_map = dict(zip(fit_.players.tolist(), fit_.off))
    d_map = dict(zip(fit_.players.tolist(), fit_.deff))
    po, pd_ = prior_off or {}, prior_def or {}
    out = np.empty(len(stints))
    for i, (ok, dk, h) in enumerate(zip(stints["lineup_key"], stints["opp_lineup_key"], stints["is_home"])):
        o = sum(o_map.get(int(p), po.get(int(p), 0.0)) for p in ok.split("-"))
        d = sum(d_map.get(int(p), pd_.get(int(p), 0.0)) for p in dk.split("-"))
        out[i] = fit_.league + o + d + (fit_.home if h == 1 else -fit_.home)
    return out


def weighted_mse(stints: pd.DataFrame, pred: np.ndarray) -> float:
    s = stints["poss"].to_numpy()
    m = s > 0
    y = 100.0 * stints["pts"].to_numpy()[m] / s[m]
    return float(np.average((y - pred[m]) ** 2, weights=s[m]))
