"""Minutes projections.

For every player expected to be available for a game, predict his minutes
from information known before tip-off:

  recent minutes (last 1, 3, 5, 10 games), season and last-season average,
  starting role, days since he last played, games he has missed, minutes
  left open by teammates who are out, minutes taken back by teammates
  returning, foul rate, back-to-backs, home/away, and how lopsided the
  game is expected to be (blowouts cut starters' minutes).

A gradient-boosted model learns how these combine. Projections for each
team are then scaled so they add up to the minutes a team actually plays
(240 in regulation, slightly more on average because of overtime), with
nobody above 48.

The same feature code runs for backtests (where we know who actually
played) and for live games (where the injury report says who is out).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Only things known before tip-off. (How many players end up getting in is NOT one of
# them: blowouts send in the deep bench, so using it would leak the result.)
FEATURES = ["m_l1", "m_l3", "m_l5", "m_l10", "m_season", "m_prev", "gp_season", "start_l1", "start_rate_l10",
            "days_since", "missed_team_games", "vacated", "vacated_share", "returning",
            "exp_margin_abs", "exp_margin_x_starter", "pf36", "team_rest", "home", "new_team"]


def history(conn, seasons) -> pd.DataFrame:
    q = ",".join("?" * len(seasons))
    df = pd.read_sql_query(
        f"""SELECT p.game_id, g.game_date, g.season_id, p.team_id, p.player_id, p.seconds / 60.0 AS min,
                   p.started, p.pf,
                   CASE WHEN g.home_team_id = p.team_id THEN 1 ELSE 0 END AS home,
                   CASE WHEN g.home_team_id = p.team_id THEN g.away_team_id ELSE g.home_team_id END AS opp_id,
                   CASE WHEN g.home_team_id = p.team_id THEN g.home_score - g.away_score
                        ELSE g.away_score - g.home_score END AS margin
            FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
            WHERE g.status = 'final' AND g.season_type = 'regular' AND p.seconds > 0 AND g.season_id IN ({q})
            ORDER BY g.game_date, g.game_id""", conn, params=list(seasons))
    return df


@dataclass
class Target:
    game_id: str
    game_date: str
    season_id: str
    team_id: int
    opp_id: int
    home: int
    players: list          # player ids expected to play
    n_expected: Optional[float] = None   # live games: expected number who get in (sum of appearance odds)


class FeatureBuilder:
    """Pre-game features for any (game, team, player list), using only earlier games.

    Histories are kept as sorted arrays so each lookup is a binary search.
    """

    def __init__(self, hist: pd.DataFrame):
        h = self.hist = hist.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        h = h.assign(day=pd.to_datetime(h["game_date"]).map(pd.Timestamp.toordinal))
        self.players = {}
        for p, g in h.groupby("player_id", sort=False):
            self.players[p] = {
                "day": g["day"].to_numpy(), "min": g["min"].to_numpy(float), "start": g["started"].to_numpy(),
                "pf": g["pf"].to_numpy(float), "season": g["season_id"].to_numpy(), "team": g["team_id"].to_numpy(),
            }
        tg = (h.groupby(["team_id", "game_id"], as_index=False)
              .agg(day=("day", "first"), margin=("margin", "first"), season=("season_id", "first"))
              .sort_values(["day", "game_id"]))
        self.teams = {}
        for t, g in tg.groupby("team_id"):
            self.teams[t] = {"day": g["day"].to_numpy(), "gid": g["game_id"].to_numpy(),
                             "margin": g["margin"].to_numpy(float), "season": g["season"].to_numpy()}
        self.appear = {(t, gid): dict(zip(g["player_id"], g["min"])) for (t, gid), g in h.groupby(["team_id", "game_id"])}

    @staticmethod
    def _day(date):
        return pd.Timestamp(date).toordinal()

    def _strength(self, team, day, season):
        t = self.teams.get(team)
        if t is None:
            return 0.0
        n = np.searchsorted(t["day"], day)
        m, s = t["margin"][:n], t["season"][:n]
        cur = m[s == season]
        prev = m[s != season][-41:]
        # Point differential per game, last season counted at half weight, pulled toward 0.
        return (cur.sum() + 0.5 * prev.sum()) / (len(cur) + 0.5 * len(prev) + 10)

    def rows(self, targets) -> pd.DataFrame:
        out = []
        for tg in targets:
            day = self._day(tg.game_date)
            t = self.teams.get(tg.team_id)
            n_team = np.searchsorted(t["day"], day) if t is not None else 0
            last10 = t["gid"][max(0, n_team - 10):n_team] if t is not None else []
            regulars = {}
            for gid in last10:
                for p, m in self.appear.get((tg.team_id, gid), {}).items():
                    regulars.setdefault(p, []).append(m)
            regular_avg = {p: float(np.mean(v)) for p, v in regulars.items() if len(v) >= 2}
            playing = set(tg.players)
            vacated = sum(m for p, m in regular_avg.items() if p not in playing and m >= 10)
            returning = 0.0
            last_set = self.appear.get((tg.team_id, last10[-1]), {}) if len(last10) else {}
            hist_cache = {}
            for p in playing:
                ph = self.players.get(p)
                k = np.searchsorted(ph["day"], day) if ph is not None else 0
                hist_cache[p] = (ph, k)
                if len(last10) and p not in last_set and ph is not None and k:
                    returning += float(ph["min"][max(0, k - 10):k].mean())
            team_rest = 3
            if t is not None and n_team:
                team_rest = int(min(day - t["day"][n_team - 1], 4))
            exp_margin = (self._strength(tg.team_id, day, tg.season_id)
                          - self._strength(tg.opp_id, day, tg.season_id) + (2.5 if tg.home else -2.5))
            for p in tg.players:
                ph, k = hist_cache[p]
                r = {"game_id": tg.game_id, "game_date": tg.game_date, "team_id": tg.team_id, "player_id": p,
                     "home": tg.home, "team_rest": team_rest, "vacated": vacated, "returning": returning,
                     "n_available": tg.n_expected if tg.n_expected is not None else len(playing),
                     "exp_margin_abs": abs(exp_margin)}
                if ph is None or k == 0:
                    r.update(m_l1=np.nan, m_l3=np.nan, m_l5=np.nan, m_l10=np.nan, m_season=np.nan, m_prev=np.nan,
                             gp_season=0, start_l1=0, start_rate_l10=0.0, days_since=30, missed_team_games=0,
                             pf36=np.nan, new_team=1)
                else:
                    mins, seas = ph["min"][:k], ph["season"][:k]
                    cur = mins[seas == tg.season_id]
                    prev = mins[seas != tg.season_id]
                    r["m_l1"] = mins[-1]
                    r["m_l3"] = mins[-3:].mean()
                    r["m_l5"] = mins[-5:].mean()
                    r["m_l10"] = mins[-10:].mean()
                    r["m_season"] = cur.mean() if len(cur) else np.nan
                    r["m_prev"] = prev.mean() if len(prev) else np.nan
                    r["gp_season"] = len(cur)
                    r["start_l1"] = int(ph["start"][k - 1])
                    r["start_rate_l10"] = float(ph["start"][max(0, k - 10):k].mean())
                    r["days_since"] = int(min(day - ph["day"][k - 1], 30))
                    with_team = ph["day"][:k][ph["team"][:k] == tg.team_id]
                    r["new_team"] = int(len(with_team) == 0)
                    if len(with_team) and t is not None:
                        r["missed_team_games"] = int(np.sum((t["day"][:n_team] > with_team[-1])))
                    else:
                        r["missed_team_games"] = 0
                    m30, pf30 = mins[-30:], ph["pf"][max(0, k - 30):k]
                    r["pf36"] = 36 * pf30.sum() / m30.sum() if m30.sum() else np.nan
                r["vacated_share"] = vacated * (0 if np.isnan(r["m_l10"]) else r["m_l10"]) / 240.0
                r["exp_margin_x_starter"] = r["exp_margin_abs"] * r["start_rate_l10"]
                out.append(r)
        return pd.DataFrame(out)


def backtest_targets(hist: pd.DataFrame, season: str, start_date: str = None, end_date: str = None):
    """One target per team-game: the players who actually played (as a perfect injury report would say)."""
    h = hist[hist["season_id"] == season]
    if start_date:
        h = h[h["game_date"] >= start_date]
    if end_date:
        h = h[h["game_date"] < end_date]
    out = []
    for (gid, t), g in h.groupby(["game_id", "team_id"]):
        r = g.iloc[0]
        out.append(Target(gid, r["game_date"], season, int(t), int(r["opp_id"]), int(r["home"]),
                          g["player_id"].astype(int).tolist()))
    return out


def baseline(df: pd.DataFrame) -> np.ndarray:
    """Season average, falling back to last 10 games, then last season, then 12 minutes."""
    return (df["m_season"].fillna(df["m_l10"]).fillna(df["m_prev"]).fillna(12.0)).to_numpy()


def fill(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    base = baseline(df)
    for c in ("m_l1", "m_l3", "m_l5", "m_l10", "m_season", "m_prev"):
        X[c] = X[c].fillna(pd.Series(base, index=X.index))
    X["pf36"] = X["pf36"].fillna(3.5)
    return X


def team_total_target(hist: pd.DataFrame) -> float:
    """Average minutes a team plays per game (240 plus overtime), measured from the data."""
    t = hist.groupby(["game_id", "team_id"])["min"].sum()
    return float(t.mean())


def scale_to_team(df: pd.DataFrame, pred: np.ndarray, total: float, cap: float = 48.0) -> np.ndarray:
    """Scale each team-game's projections so they sum to `total`, capping anyone at `cap`."""
    out = np.asarray(pred, float).clip(min=0).copy()
    for _, idx in df.groupby(["game_id", "team_id"]).indices.items():
        p = out[idx]
        for _ in range(10):
            s = p.sum()
            if s <= 0:
                break
            free = p < cap
            room = total - p[~free].sum()
            if free.sum() == 0 or p[free].sum() <= 0:
                break
            p[free] = p[free] * room / p[free].sum()
            if (p <= cap + 1e-9).all():
                break
            p = p.clip(max=cap)
        out[idx] = p
    return out


class MinutesModel:
    def __init__(self):
        self.model = None
        self.total = 241.0
        self.resid_q = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, total: float):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05,
                                                   max_leaf_nodes=31, min_samples_leaf=40, random_state=0)
        self.model.fit(X, y)
        self.total = total
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X).clip(min=0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return scale_to_team(df, self.predict_raw(fill(df)), self.total)

    def set_intervals(self, pred: np.ndarray, actual: np.ndarray):
        """Learn 10th/90th percentile errors by projected-minutes band, from held-out games."""
        bands = np.array([0, 8, 16, 24, 30, 36, 60])
        q = []
        for lo, hi in zip(bands[:-1], bands[1:]):
            m = (pred >= lo) & (pred < hi)
            r = actual[m] - pred[m]
            q.append((lo, hi, float(np.quantile(r, 0.1)) if m.sum() > 30 else -6.0,
                      float(np.quantile(r, 0.9)) if m.sum() > 30 else 6.0))
        self.resid_q = q

    def interval(self, pred: np.ndarray):
        lo, hi = np.zeros_like(pred), np.zeros_like(pred)
        for a, b, q10, q90 in self.resid_q or [(0, 99, -6.0, 6.0)]:
            m = (pred >= a) & (pred < b)
            lo[m], hi[m] = (pred[m] + q10).clip(min=0), (pred[m] + q90).clip(max=48)
        return lo, hi
