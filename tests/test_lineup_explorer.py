"""Lineup explorer: groups inside five-man units add up. Test use only."""
import pandas as pd
import pytest

from pipeline.models import lineup_explorer as LX

A, B = 1610612738, 1610612747


def _stint(game, n, team, key, poss, pts, secs=120, **kw):
    row = {"game_id": game, "stint_num": n, "team_id": team, "lineup_key": key, "start_sec": 0, "end_sec": secs,
           "poss": poss, "pts": pts, "fga": poss, "fgm": pts // 2, "fg3a": 2, "fg3m": 1, "fta": 2, "ftm": 1, "tov": 1,
           "oreb": 1, "dreb": 3}
    row.update(kw)
    return row


def test_groups_split_five_man_totals():
    rows = []
    for g in range(30):          # enough minutes for every group size
        rows += [_stint(f"g{g}", 1, A, "1-2-3-4-5", 10, 12), _stint(f"g{g}", 1, B, "11-12-13-14-15", 10, 9),
                 _stint(f"g{g}", 2, A, "1-2-3-4-6", 10, 8), _stint(f"g{g}", 2, B, "11-12-13-14-15", 10, 11)]
    out = LX.build(pd.DataFrame(rows), "2026-27")
    t = out["teams"][str(A)]
    fives = {tuple(r[0]): dict(zip(t["5"]["cols"], r)) for r in t["5"]["rows"]}
    assert fives[(1, 2, 3, 4, 5)]["ortg"] == pytest.approx(120.0)
    assert fives[(1, 2, 3, 4, 5)]["drtg"] == pytest.approx(90.0)
    pairs = {tuple(r[0]): dict(zip(t["2"]["cols"], r)) for r in t["2"]["rows"]}
    # 1 and 2 played in both units: their pair is the two units combined
    assert pairs[(1, 2)]["ortg"] == pytest.approx(100.0)
    assert pairs[(1, 2)]["minutes"] == pytest.approx(120.0)
    assert (5, 6) not in pairs                     # never on the floor together
    assert t["team"]["net"] == pytest.approx(0.0)
    # more possessions, smaller margin
    assert pairs[(1, 2)]["margin"] < fives[(1, 2, 3, 4, 5)]["margin"]
