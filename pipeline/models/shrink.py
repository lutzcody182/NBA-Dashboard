"""Bayesian shrinkage for player rates, with strengths learned from data.

Every rate (a percentage like TS% or a per-possession rate like steals per
100) is estimated as

    estimate = (made_now + w * made_last_season + k * league_avg)
               / (tries_now + w * tries_last_season + k)

  k : prior strength, in the stat's own units (attempts, possessions). A
      player needs about k attempts of evidence before his own numbers
      count as much as the league average.
  w : how much last season's evidence is worth, per attempt, relative to
      this season's.

k and w are not chosen by hand. They are picked by testing: estimate each
player from the games before a cutoff date, then score the estimate against
what the player actually did after the cutoff. Several cutoffs are pooled
(early, middle and late season) so the values work at any point of a
season. The chosen values, and how much they beat the raw numbers, are
reported with every rating.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

K_GRID = np.array([0, 5, 10, 20, 40, 80, 150, 300, 600, 1200, 2500, 5000])
W_GRID = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])


@dataclass
class ShrinkFit:
    k: float
    w: float
    league: float
    error: float              # weighted squared error of the chosen setting
    error_raw: float          # error using only this season's raw numbers
    error_league: float       # error using the league average for everyone
    n_players: int
    note: str = ""
    grid: dict = field(default_factory=dict)

    @property
    def improvement_vs_raw(self):
        return None if not self.error_raw else 1 - self.error / self.error_raw

    def apply(self, made_now, tries_now, made_prev=0.0, tries_prev=0.0, league=None):
        mu = self.league if league is None else league
        num = np.asarray(made_now, float) + self.w * np.asarray(made_prev, float) + self.k * mu
        den = np.asarray(tries_now, float) + self.w * np.asarray(tries_prev, float) + self.k
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, mu)

    def weight_on_player(self, tries_now, tries_prev=0.0):
        """Share of the estimate that comes from the player's own numbers (0 to 1)."""
        ev = np.asarray(tries_now, float) + self.w * np.asarray(tries_prev, float)
        return ev / (ev + self.k) if self.k else np.ones_like(ev)

    def to_dict(self):
        return {"k": self.k, "w": self.w, "league": self.league, "error": self.error,
                "error_raw": self.error_raw, "error_league": self.error_league,
                "improvement_vs_raw": self.improvement_vs_raw, "n_players": self.n_players,
                "note": self.note}


def _sums(df, made, tries, by="player_id"):
    g = df.groupby(by)[[made, tries]].sum()
    return g[made], g[tries]


def calibrate(cur: pd.DataFrame, prev: pd.DataFrame, made: str, tries: str,
              cutoffs, min_test: float = 1.0) -> ShrinkFit:
    """Pick k and w for one stat on a season with a previous season behind it.

    cur, prev: player-game frames with columns player_id, game_date, made, tries.
    """
    pm, pt = _sums(prev, made, tries) if prev is not None and len(prev) else (pd.Series(dtype=float),) * 2
    stacks = []
    for cut in cutoffs:
        before, after = cur[cur["game_date"] < cut], cur[cur["game_date"] >= cut]
        bm, bt = _sums(before, made, tries)
        am, at = _sums(after, made, tries)
        league = bm.sum() / bt.sum() if bt.sum() else 0.0
        ids = at[at >= min_test].index
        frame = pd.DataFrame({
            "bm": bm.reindex(ids, fill_value=0), "bt": bt.reindex(ids, fill_value=0),
            "pm": pm.reindex(ids, fill_value=0), "pt": pt.reindex(ids, fill_value=0),
            "am": am.reindex(ids), "at": at.reindex(ids), "league": league,
        })
        stacks.append(frame)
    data = pd.concat(stacks)
    if data.empty or data["at"].sum() == 0:
        return ShrinkFit(k=300, w=0.5, league=0.0, error=np.nan, error_raw=np.nan, error_league=np.nan,
                         n_players=0, note="no data to calibrate; default used")
    target = data["am"] / data["at"]
    wts = data["at"]

    def err(est):
        return float(np.sum(wts * (target - est) ** 2) / np.sum(wts))

    grid, best = {}, None
    for k in K_GRID:
        for w in W_GRID:
            num = data["bm"] + w * data["pm"] + k * data["league"]
            den = data["bt"] + w * data["pt"] + k
            est = np.where(den > 0, num / den, data["league"])
            e = err(est)
            grid[f"{k}|{w}"] = e
            if best is None or e < best[0]:
                best = (e, k, w)
    raw_den = data["bt"].replace(0, np.nan)
    raw = (data["bm"] / raw_den).fillna(data["league"])
    league_cur = cur[made].sum() / cur[tries].sum() if cur[tries].sum() else 0.0
    return ShrinkFit(k=float(best[1]), w=float(best[2]), league=float(league_cur), error=best[0],
                     error_raw=err(raw), error_league=err(data["league"]),
                     n_players=int(data.index.nunique()), grid=grid)
