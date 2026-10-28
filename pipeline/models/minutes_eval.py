"""Early-season replay of the game-day minutes pipeline.

The weak spot of minutes projections is the first weeks of a season, when
players have changed teams and roles and this season's record is thin.
This replays the first EARLY_DAYS days of a past season the way the live
site sees each game:

  * roster: everyone who played for the team within a month either side of
    the game (the past has no stored rosters, so this stands in for one)
  * out: roster players who sat although they were rotation players
    (15+ minutes a game around then), as the injury report would list them;
    everyone else is available, including bench players who then did not get in
  * chance of getting in: this season's record so far, plus last season's
    record at a capped weight, as the live code does
  * a model trained only on seasons before the replayed one

and scores expected minutes against what happened, overall, for the five
players who actually played most, and for players new to their team.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import minutes as M

EARLY_DAYS = 30
OUT_ROLE_MINUTES = 15.0


@dataclass
class TeamGame:
    target: M.Target
    available: list
    played: dict          # player -> minutes
    odds: dict            # player -> chance of getting in
    new_team: set
    role: dict            # player -> average minutes with the team in the month around the game


def _odds(n_played, n_eligible, prev_played, prev_eligible, weight=0.5, cap=6.0):
    w = min(weight, cap / prev_eligible) if prev_eligible else 0.0
    return (n_played + w * prev_played + 1) / (n_eligible + w * prev_eligible + 2)


def early_games(hist: pd.DataFrame, season: str, days: int = EARLY_DAYS, every: int = 1) -> list:
    """TeamGame records for the first `days` days of `season`."""
    h = hist[hist.season_id == season].copy()
    if h.empty:
        return []
    start = pd.Timestamp(h.game_date.min())
    h["day"] = (pd.to_datetime(h.game_date) - start).dt.days
    prev = hist[hist.season_id < season]
    prev_season = prev[prev.season_id == prev.season_id.max()] if len(prev) else prev
    # last season's appearance record, any team: games played / team games between first and last appearance
    prev_rec = {}
    if len(prev_season):
        team_days = prev_season.groupby("team_id").game_date.unique().apply(sorted).to_dict()
        for (p, t), g in prev_season.groupby(["player_id", "team_id"]):
            d0, d1 = g.game_date.min(), g.game_date.max()
            elig = sum(d0 <= x <= d1 for x in team_days[t])
            a, b = prev_rec.get(p, (0, 0))
            prev_rec[p] = (a + len(g), b + elig)
    team_games = h.groupby("team_id")
    out = []
    games = h[h.day < days][["game_id", "game_date", "team_id", "opp_id", "home", "day"]].drop_duplicates(
        ["game_id", "team_id"]).sort_values(["day", "game_id"])
    for i, g in enumerate(games.itertuples()):
        if i % every:
            continue
        tg_all = team_games.get_group(g.team_id)
        window = tg_all[(tg_all.day >= g.day - 30) & (tg_all.day <= g.day + 30)]
        role = window.groupby("player_id")["min"].mean()
        played = dict(zip(*[tg_all[tg_all.game_id == g.game_id][c] for c in ("player_id", "min")]))
        roster = set(role.index) | set(played)
        out_ids = {p for p in roster if p not in played and role.get(p, 0) >= OUT_ROLE_MINUTES}
        avail = sorted(roster - out_ids)
        before = tg_all[tg_all.day < g.day]
        team_before = before.game_id.unique()
        odds, new = {}, set()
        for p in avail:
            mine = before[before.player_id == p]
            # games he missed while a rotation player were injuries, not coach's decisions
            missed_out = 0
            if role.get(p, 0) >= OUT_ROLE_MINUTES:
                missed_out = len(team_before) - len(mine)
            elig = len(team_before) - missed_out
            pp, pe = prev_rec.get(p, (0, 0))
            odds[p] = _odds(len(mine), elig, pp, pe)
            if not (prev[(prev.player_id == p) & (prev.team_id == g.team_id)].shape[0]) and len(prev):
                new.add(p)
        t = M.Target(g.game_id, g.game_date, season, int(g.team_id), int(g.opp_id), int(g.home), avail,
                     float(sum(odds.values())))
        out.append(TeamGame(t, avail, played, odds, new, {p: float(role.get(p, 0.0)) for p in avail}))
    return out


def score(records: list, expected: list) -> dict:
    """records: TeamGame; expected: per record, {player: expected minutes}."""
    err, top, starters_bias, new_err, sums = [], [], [], [], []
    for r, e in zip(records, expected):
        act = {p: r.played.get(p, 0.0) for p in r.available}
        err += [abs(act[p] - e.get(p, 0.0)) for p in r.available]
        # the team's five biggest roles around that time (not that night's result, which would bias it)
        top5 = sorted(r.available, key=lambda p: -r.role.get(p, 0.0))[:5]
        top += [abs(act[p] - e.get(p, 0.0)) for p in top5]
        starters_bias += [act[p] - e.get(p, 0.0) for p in top5]
        new_err += [abs(act[p] - e.get(p, 0.0)) for p in r.available if p in r.new_team]
        sums.append(sum(e.values()))
    return {"mae": float(np.mean(err)), "top5_mae": float(np.mean(top)), "top5_bias": float(np.mean(starters_bias)),
            "new_team_mae": float(np.mean(new_err)) if new_err else None, "players": len(err),
            "team_games": len(records), "mean_team_total": float(np.mean(sums))}


def allocate(cond: np.ndarray, p_in: np.ndarray, total: float, power: float = 1.0, cap: float = 48.0) -> np.ndarray:
    """Expected minutes for one team-game, summing to `total`.

    power = 1 is the old proportional squeeze. power > 1 squeezes the deep
    bench harder than the regulars, the way coaches shorten a crowded rotation.
    """
    base = np.clip(cond, 0, None) * p_in
    w = base ** power
    if w.sum() <= 0:
        return base
    fixed = np.zeros(len(w), bool)
    out = w * total / w.sum()
    for _ in range(len(w)):
        new = (out > cap) & ~fixed
        if not new.any():
            break
        fixed |= new
        rest = w[~fixed].sum()
        out = np.where(fixed, cap, w * (total - cap * fixed.sum()) / rest if rest > 0 else 0.0)
    return out
