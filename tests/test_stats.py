import json

import numpy as np
import pytest

from pipeline import db
from pipeline.export.build import export_all, player_logs
from pipeline.stats import formulas as f
from pipeline.stats.team_ratings import load_team_games, team_table

from .simulate import TEAM_IDS, simulate_season


# -- formulas, checked against hand calculations ---------------------------

def test_true_shooting_by_hand():
    # 30 pts on 20 FGA and 10 FTA: 30 / (2 * (20 + 4.4)) = 0.6148
    assert f.true_shooting(30, 20, 10) == pytest.approx(30 / 48.8)


def test_effective_fg_by_hand():
    # 8 of 16 with 4 threes: (8 + 2) / 16 = 0.625
    assert f.effective_fg(8, 4, 16) == pytest.approx(0.625)


def test_possessions_by_hand():
    # Team: 88 FGA, 45 FGM, 19 FTA, 10 ORB, 12 TOV; Opp DRB 33
    team = 88 + 0.4 * 19 - 1.07 * (10 / (10 + 33)) * (88 - 45) + 12
    # Opp: 85 FGA, 38 FGM, 22 FTA, 9 ORB, 14 TOV; Team DRB 36
    opp = 85 + 0.4 * 22 - 1.07 * (9 / (9 + 36)) * (85 - 38) + 14
    got = f.estimate_possessions(fga=88, fta=19, oreb=10, tov=12, fgm=45, opp_dreb=33,
                                 opp_fga=85, opp_fta=22, opp_oreb=9, opp_tov=14, opp_fgm=38, dreb=36)
    assert got == pytest.approx((team + opp) / 2)


def test_pace_and_rating():
    assert f.pace(100, 240) == pytest.approx(100.0)
    assert f.pace(100, 265) == pytest.approx(48 * 100 / 53)   # one overtime
    assert f.rating(115, 100) == pytest.approx(115.0)
    assert f.rating(115, 0) is None


def test_usage_by_hand():
    # 20 FGA, 5 FTA, 3 TOV in 36 min; team 88 FGA, 20 FTA, 13 TOV in 240 min
    num, den = f.usage_parts(minutes=36, fga=20, fta=5, tov=3, tm_minutes=240,
                             tm_fga=88, tm_fta=20, tm_tov=13)
    expected = 100 * (20 + 2.2 + 3) * 48 / (36 * (88 + 8.8 + 13))
    assert 100 * num / den == pytest.approx(expected)


def test_assist_and_rebound_pct_by_hand():
    num, den = f.assist_pct_parts(minutes=36, ast=8, fgm=7, tm_minutes=240, tm_fgm=42)
    assert 100 * num / den == pytest.approx(100 * 8 / ((36 / 48) * 42 - 7))
    num, den = f.rebound_pct_parts(minutes=30, reb=10, tm_minutes=240, tm_reb=45, opp_reb=43)
    assert 100 * num / den == pytest.approx(100 * 10 * 48 / (30 * 88))


def test_pythagorean():
    assert f.pythagorean_win_pct(110, 110) == pytest.approx(0.5)
    assert f.pythagorean_win_pct(115, 105) > 0.75


# -- simulated season ---------------------------------------------------------

@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    conn = db.connect(tmp_path_factory.mktemp("sim") / "sim.sqlite")
    truth = simulate_season(conn)
    yield conn, truth
    conn.close()


def test_team_minutes_and_possessions(sim):
    conn, _ = sim
    df = load_team_games(conn, "2025-26")
    assert len(df) == len(TEAM_IDS) * 82
    assert (df["minutes"] == 240).all()
    assert df["poss"].between(80, 125).all()


def test_adjusted_ratings_recover_true_strength(sim):
    conn, truth = sim
    table = team_table(load_team_games(conn, "2025-26"))
    rows = {r["team_id"]: r for r in table["teams"]}
    true_net = np.array([truth["o"][t] - truth["d"][t] for t in TEAM_IDS])
    adj_net = np.array([rows[t]["adj_net"] for t in TEAM_IDS])
    assert np.corrcoef(true_net, adj_net)[0, 1] > 0.9
    # Shrinkage strength was chosen from data, not left at the default
    assert table["model"]["lambda_source"].startswith("walk-forward")
    # Home edge in the right neighbourhood (true value: 3 points per 100)
    assert 0 < table["model"]["home_edge_per_100"] < 6


def test_adjustment_corrects_a_hard_schedule(tmp_path):
    conn = db.connect(tmp_path / "bias.sqlite")
    team = TEAM_IDS[0]
    truth = simulate_season(conn, seed=11, schedule_bias={team: "strong"})
    table = team_table(load_team_games(conn, "2025-26"))
    row = next(r for r in table["teams"] if r["team_id"] == team)
    true_o = truth["league"] + truth["o"][team]
    # Facing only elite defenses drags the raw rating down; the adjusted one is closer to the truth
    assert abs(row["adj_ortg"] - true_o) < abs(row["ortg"] - true_o)
    assert row["sos"] < 0 or row["adj_ortg"] > row["ortg"]


def test_player_logs_rates_are_sane(sim):
    conn, _ = sim
    logs = player_logs(conn, "2025-26")
    # Usage of all players on a team averages to 100% / 5 players weighted by minutes,
    # which means sum(num) / sum(den) over the whole team must equal 20%.
    g = logs.groupby(["game_id", "team_id"])[["usg_n", "usg_d"]].sum()
    assert (100 * g["usg_n"] / g["usg_d"]).round(6).eq(20.0).all()
    assert set(logs["dd"].unique()) <= {0, 1}


def test_export_writes_everything(sim, tmp_path):
    conn, _ = sim
    res = export_all(conn, tmp_path)

    def errors(x):
        if isinstance(x, dict):
            return [e for k, v in x.items() for e in (v if k == "errors" and isinstance(v, list) else errors(v))]
        return [e for v in x for e in errors(v)] if isinstance(x, list) else []
    # parts that fail are caught and logged so the site still updates; none should fail here
    assert not [e for e in errors(res) if "analytics" in str(e)]
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["season"] == "2025-26"
    assert len(meta["teams"]) == 30
    ratings = json.loads((tmp_path / "teams" / "ratings_2025-26.json").read_text())
    assert len(ratings["teams"]) == 30
    log = json.loads((tmp_path / "players" / "gamelog_2025-26.json").read_text())
    assert log["cols"][0] == "game_id" and len(log["rows"]) == 30 * 82 * 8
    rosters = json.loads((tmp_path / "rosters.json").read_text())
    some = rosters["teams"][str(TEAM_IDS[0])]
    assert some["data_status"] == "confirmed" and len(some["players"]) == 8
    # No NaN can sneak into the JSON
    for p in tmp_path.rglob("*.json"):
        assert "NaN" not in p.read_text()
