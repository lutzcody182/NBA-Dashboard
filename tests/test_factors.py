import pandas as pd

from pipeline.export import factors as FX


def test_position_group():
    assert FX._group("G-F", 0, 0) == ("G", True)
    assert FX._group(None, 0.2, 0.1) == ("G", False)
    assert FX._group(float("nan"), 0.05, 0.3) == ("C", False)
    assert FX._group("", 0.05, 0.1) == ("F", False)


def test_teammate_splits():
    rows = []
    for i in range(10):
        gid, d = f"g{i}", f"2026-11-{i + 1:02d}"
        rows.append(dict(game_id=gid, player_id=1, team_id=9, season_id="2026-27", game_date=d, m=30.0,
                         pts=10 if i < 5 else 20, reb=5, ast=3, fg3m=1))
        if i < 5:      # star plays the first five games only, but is still on the team after
            rows.append(dict(game_id=gid, player_id=2, team_id=9, season_id="2026-27", game_date=d, m=35.0,
                             pts=30, reb=5, ast=5, fg3m=3))
    rows.append(dict(game_id="g10", player_id=2, team_id=9, season_id="2026-27", game_date="2026-11-30", m=35.0,
                     pts=30, reb=5, ast=5, fg3m=3))
    pg = pd.DataFrame(rows)
    sp = FX._teammate_splits(pg, 1, 9, [(2, "Star")], "2026-27")
    assert len(sp) == 1
    assert sp[0]["with"]["pts"] == 10 and sp[0]["without"]["pts"] == 20
    assert sp[0]["with"]["games"] == 5 and sp[0]["without"]["games"] == 5
    # a bench teammate is not worth a split
    assert FX._teammate_splits(pg.assign(m=10.0), 1, 9, [(2, "Star")], "2026-27") == []
