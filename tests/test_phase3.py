"""Phase 3: schedule effects, team rates, the simulator and usage redistribution.

The data here is synthetic with a known answer built in, so each test checks
that the method recovers what was put there. Test use only.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.models import game_model as GM
from pipeline.models import schedule as SC
from pipeline.models import sim
from pipeline.models import team_rates as TR
from pipeline.models import usage as U

BOS, LAL, DEN, NYK = 1610612738, 1610612747, 1610612743, 1610612752

LEAGUE = {"tov": 0.145, "fta": 0.234, "ftp": 0.78, "and1": 0.06, "oreb": 0.26, "share_rim": 0.29,
          "share_paint": 0.18, "share_mid": 0.11, "share_c3": 0.11, "share_ab3": 0.31, "fg_rim": 0.66,
          "fg_paint": 0.44, "fg_mid": 0.45, "fg_c3": 0.39, "fg_ab3": 0.35, "pace": 99.0, "ast": 0.62}


# -- schedule ------------------------------------------------------------------------

def _g(gid, day, home, away, season="2025-26"):
    return {"game_id": gid, "season_id": season, "game_date": day, "home_team_id": home, "away_team_id": away}


def test_team_schedule_situations():
    games = pd.DataFrame([
        _g("1", "2025-11-01", BOS, NYK),          # BOS home
        _g("2", "2025-11-02", LAL, BOS),          # BOS at LAL next night: back-to-back, flew west 3 zones
        _g("3", "2025-11-04", DEN, BOS),          # 3 games in 4 nights, at altitude, 2nd road game
        _g("4", "2025-11-08", BOS, DEN),          # home again after 3 days off, flew east
    ])
    s = SC.team_schedule(games).set_index(["game_id", "team_id"])
    b = lambda g: s.loc[(g, BOS)]  # noqa: E731
    assert b("1").rest == 4 and b("1").travel_mi == 0
    assert b("2").b2b == 1 and abs(b("2").travel_mi - 2600) < 50 and b("2").tz_west == 3
    assert b("3").three_in_four == 1 and b("3").visitor_altitude == 1 and b("3").road_trip_n == 2
    assert b("4").rest3plus == 1 and b("4").tz_east == 2 and b("4").road_trip_n == 0
    assert s.loc[("4", DEN)].visitor_altitude == 0       # Denver visiting Boston is not at altitude


def test_schedule_effects_are_recovered():
    rng = np.random.default_rng(3)
    teams = sorted(SC.ARENAS)[:12]
    strength = {t: rng.normal(0, 4) for t in teams}
    rows, day = [], pd.Timestamp("2025-10-20")
    gid = 0
    for d in range(160):
        day_s = (day + pd.Timedelta(days=d)).date().isoformat()
        order = rng.permutation(teams)
        for i in range(0, len(order), 2):
            if rng.random() < 0.5:
                continue
            gid += 1
            rows.append(_g(f"g{gid}", day_s, int(order[i]), int(order[i + 1])))
    games = pd.DataFrame(rows)
    sched = SC.team_schedule(games)
    s = sched.set_index(["game_id", "team_id"])
    true_b2b, hca = -3.0, 2.5
    margin = []
    for r in games.itertuples():
        diff = s.loc[(r.game_id, r.home_team_id)].b2b - s.loc[(r.game_id, r.away_team_id)].b2b
        margin.append(hca + strength[r.home_team_id] - strength[r.away_team_id] + true_b2b * diff + rng.normal(0, 12))
    games["home_score"] = 100 + np.array(margin)
    games["away_score"] = 100.0
    games["h_poss"] = games["a_poss"] = 100.0
    games["neutral_site"] = 0
    eff = SC.fit_effects(games, sched, lam_fatigue=1.0)
    b2b = eff["beta"][SC.FEATURES.index("b2b")]
    assert abs(b2b - true_b2b) < 2.5 * eff["se"][SC.FEATURES.index("b2b")] + 0.5
    assert abs(eff["hca"] - hca) < 1.0


# -- team rates ---------------------------------------------------------------------

def test_matchup_combines_offense_and_defense():
    est = {"league": LEAGUE, "teams": {
        1: {"off": dict(LEAGUE), "def": dict(LEAGUE)},
        2: {"off": {**LEAGUE, "fg_rim": 0.70, "tov": 0.12}, "def": {**LEAGUE, "fg_ab3": 0.33}}}}
    same = TR.matchup(est, 1, 1)
    assert all(abs(same[k] - LEAGUE[k]) < 1e-9 for k in ("tov", "fg_rim", "share_ab3", "fta"))
    r = TR.matchup(est, 2, 2)          # good rim offense against a defense that limits threes
    assert r["fg_rim"] > LEAGUE["fg_rim"] and r["fg_ab3"] < LEAGUE["fg_ab3"] and r["tov"] < LEAGUE["tov"]
    assert abs(sum(r[f"share_{z}"] for z in TR.ZONES) - 1) < 1e-9


def test_shot_quality_splits_mix_from_making():
    good_mix = {**LEAGUE, "share_rim": 0.39, "share_mid": 0.01}      # rim shots instead of mid-range
    est = {"league": LEAGUE, "teams": {1: {"off": good_mix, "def": dict(LEAGUE)},
                                       2: {"off": {**LEAGUE, "fg_ab3": 0.40}, "def": dict(LEAGUE)}}}
    sq = TR.shot_quality(est)["teams"]
    assert sq[1]["off"]["shot_quality"] > sq[2]["off"]["shot_quality"]     # better shots
    assert abs(sq[1]["off"]["shot_making"]) < 1e-9 and sq[2]["off"]["shot_making"] > 0.01   # better shooting


# -- simulator ----------------------------------------------------------------------

def test_expected_points_match_simulation():
    pts = sim._possessions(LEAGUE, np.zeros(400), 400, 100, np.random.default_rng(1))
    assert abs(pts.mean() - sim.expected_ppp(LEAGUE)) < 0.01


def test_shift_hits_target_points():
    for delta in (-6.0, 3.0, 8.0):
        sh = sim.shift_for(LEAGUE, delta)
        assert abs(100 * (sim.expected_ppp(LEAGUE, sh) - sim.expected_ppp(LEAGUE)) - delta) < 0.01


def test_simulated_games():
    even = sim.simulate(LEAGUE, LEAGUE, pace=99, n_sims=4000, seed=2)
    assert abs(even["home_win"] - 0.5) < 0.03 and abs(even["margin_mean"]) < 0.8
    strong = sim.simulate(LEAGUE, LEAGUE, pace=99, home_shift=sim.shift_for(LEAGUE, 5), away_shift=0, n_sims=4000, seed=2)
    assert strong["home_win"] > 0.6 and 3.5 < strong["margin_mean"] < 6.5
    calm = sim.simulate(LEAGUE, LEAGUE, pace=99, lead_effect=-0.006, n_sims=4000, seed=2)
    assert calm["margin_sd"] < even["margin_sd"] - 1.5        # leaders easing off narrows margins
    assert (calm["_margin"] != 0).all()                       # overtime settles every game


# -- usage ---------------------------------------------------------------------------

def _stints():
    rows = []
    for s, lineup, fga in ((1, [1, 2, 3, 4, 5], 40), (2, [6, 2, 3, 4, 5], 40)):
        for p in lineup:
            rows.append({"game_id": "g", "stint_num": s, "team_id": 9, "player_id": p, "fga": fga})
    return pd.DataFrame(rows)


def test_star_absence_moves_shots_to_teammates():
    w = pd.Series({1: 0.40, 2: 0.20, 3: 0.15, 4: 0.15, 5: 0.10, 6: 0.10})
    pred = U.predict(_stints(), w, "fga", g=1.0)
    # stint 1: the star's 40% share is his; stint 2 (star out): his share goes to the others pro rata
    assert pred[("g", 1)] == pytest.approx(16.0)
    assert pred[("g", 2)] == pytest.approx(40 * 0.20 + 40 * 0.20 / 0.70)
    assert pred.sum() == pytest.approx(80.0)
    fixed = U.fixed_share(_stints(), w, "fga")
    assert fixed.sum() < 80.0                                  # keeping shares fixed loses shots
    hi = U.predict(_stints(), w, "fga", g=2.0)
    assert hi[("g", 2)] > pred[("g", 2)]                       # power > 1: the high-usage teammate takes more


def test_expected_shares_follow_minutes():
    w = pd.Series({1: 0.30, 2: 0.20, 3: 0.20, 4: 0.15, 5: 0.15})
    mins = {1: 48, 2: 48, 3: 48, 4: 48, 5: 48}
    sh = U.expected_shares(mins, w, 1.0)
    assert sum(sh.values()) == pytest.approx(1.0) and sh[1] == pytest.approx(0.30)


# -- game model ------------------------------------------------------------------------

def test_roster_effects_are_centred_on_the_slate():
    ratings = {1: (3.0, 1.0, 5000), 2: (-1.0, 0.0, 5000), 3: (0.0, 0.0, 5000)}
    rosters = {("g1", 10): ({1: 240.0}, (0.0, 0.0)), ("g1", 11): ({2: 240.0}, (0.0, 0.0)),
               ("g2", 12): ({3: 240.0}, None)}
    c = GM.center(rosters, ratings, (-1.0, -1.0), 1.0)
    assert abs(np.mean([v[0] for v in c.values()])) < 1e-9
    assert c[("g1", 10)][0] > c[("g1", 11)][0]


def test_floor_sum_is_five_players_average():
    ratings = {p: (1.0, 0.5, 5000) for p in range(1, 9)}
    o, d = GM.floor_sum({p: 30.0 for p in range(1, 9)}, ratings, (0.0, 0.0))
    assert o == pytest.approx(5.0) and d == pytest.approx(2.5)


def test_injury_report_team_names_match_whole_words():
    from pipeline.injuries.ingest import _resolve_team
    lookup = {1610612766: ("hornets", "CHA"), 1610612751: ("nets", "BKN"), 1610612757: ("trail blazers", "POR")}
    assert _resolve_team("Charlotte Hornets", lookup) == 1610612766
    assert _resolve_team("Brooklyn Nets", lookup) == 1610612751
    assert _resolve_team("Portland Trail Blazers", lookup) == 1610612757
