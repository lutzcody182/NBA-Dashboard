"""Player props: odds parsing, margin removal, pull budget and the over/under chances. Test use only."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pipeline.export import props as PX
from pipeline.models import props as PM
from pipeline.providers import odds_api as OA

NOW = datetime(2026, 10, 28, 16, 0, tzinfo=timezone.utc)


def test_names_match_across_spellings():
    assert OA.norm_name("Nikola Jokić") == OA.norm_name("Nikola Jokic")
    assert OA.norm_name("Jaren Jackson Jr.") == OA.norm_name("Jaren Jackson")
    assert OA.norm_name("De'Aaron Fox") == OA.norm_name("DeAaron Fox")


def test_parse_props_keeps_both_sides_and_flags_unknown_players():
    payload = {"bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": [
        {"key": "player_points", "outcomes": [
            {"name": "Over", "description": "Jayson Tatum", "price": -115, "point": 27.5},
            {"name": "Under", "description": "Jayson Tatum", "price": -105, "point": 27.5},
            {"name": "Over", "description": "Somebody Else", "price": -110, "point": 9.5}]},
        {"key": "player_first_basket", "outcomes": [{"name": "Jayson Tatum", "price": 600}]}]}]}
    rows, un = OA.parse_props(payload, {"jayson tatum": 1628369})
    assert len(rows) == 2 and {r["side"] for r in rows} == {"over", "under"}
    assert rows[0]["player_id"] == 1628369 and rows[0]["line"] == 27.5
    assert un == ["Somebody Else"]


def test_event_matches_game_by_teams_and_time():
    teams = {1: ("Los Angeles", "Clippers"), 2: ("Portland", "Trail Blazers"), 3: ("Los Angeles", "Lakers")}
    games = [{"game_id": "g1", "home_team_id": 1, "away_team_id": 2, "tip_utc": "2026-10-29T02:30:00Z"}]
    ev = {"home_team": "LA Clippers", "away_team": "Portland Trail Blazers", "commence_time": "2026-10-29T02:40:00Z"}
    assert OA.match_event(ev, games, teams)["game_id"] == "g1"
    ev["home_team"] = "Los Angeles Lakers"
    assert OA.match_event(ev, games, teams) is None


def test_pull_budget():
    tip = [NOW + timedelta(hours=3)]
    assert OA.should_pull({"pulls": []}, NOW, tip, "2026-10-28")[0]
    assert not OA.should_pull({"pulls": []}, NOW, [NOW + timedelta(hours=20)], "2026-10-28")[0]
    recent = {"pulls": [{"at": "2026-10-28T14:30:00Z", "et_day": "2026-10-28"}]}
    assert not OA.should_pull(recent, NOW, tip, "2026-10-28")[0]
    three = {"pulls": [{"at": f"2026-10-28T0{h}:00:00Z", "et_day": "2026-10-28"} for h in (1, 5, 9)]}
    assert not OA.should_pull(three, NOW, tip, "2026-10-28")[0]
    assert not OA.should_pull({"pulls": [], "remaining": 100}, NOW, tip, "2026-10-28")[0]


def test_devig_and_prices():
    fair, hold = PM.devig(-110, -110)
    assert fair == pytest.approx(0.5) and hold == pytest.approx(0.0476, abs=1e-3)
    fair, _ = PM.devig(-150, 130)
    assert 0.57 < fair < 0.59
    assert PM.decimal(150) == pytest.approx(2.5) and PM.decimal(-200) == pytest.approx(1.5)


def test_minutes_points_average_to_the_projection():
    m, w = PM.minutes_points(30, 20, 36)
    assert float(m @ w) == pytest.approx(30, abs=0.4)


def test_over_under_is_a_proper_distribution():
    po, pu, push = PM.over_under(24.5, 0.7, 8.5, 34, 28, 39)
    assert po + pu == pytest.approx(1.0) and push == 0
    assert 0.4 < po < 0.65                              # mean 23.8: close to a coin flip
    po2, pu2, push2 = PM.over_under(24.0, 0.7, 8.5, 34, 28, 39)
    assert po2 + pu2 + push2 == pytest.approx(1.0) and push2 > 0
    # more minutes, more overs
    assert PM.over_under(24.5, 0.7, 8.5, 38, 32, 42)[0] > po


def test_dispersion_recovers_known_spread():
    rng = np.random.default_rng(3)
    rows = []
    for p in range(120):
        rate = rng.uniform(0.3, 0.9)
        for d in range(40):
            m = 30.0
            mu = rate * m
            k = 8.0
            x = rng.negative_binomial(k, k / (k + mu))
            rows.append({"game_id": f"{p}-{d}", "date": f"2025-11-{d % 28 + 1:02d}-{d:02d}", "player_id": p, "min": m,
                         "pts": x, "reb": rng.poisson(0.2 * m), "ast": rng.poisson(0.1 * m), "fg3m": rng.poisson(0.05 * m)})
    fit = PM.fit_dispersion(pd.DataFrame(rows))
    assert 5 < fit["pts"]["k"] < 13              # true k = 8 (rate estimated from 20 games widens it a little)
    assert fit["reb"]["k"] > 30                  # Poisson: little extra spread


def test_market_table_uses_most_offered_line():
    rows = []
    for book, line, o, u in (("dk", 24.5, -115, -105), ("fd", 24.5, -110, -110), ("mgm", 25.5, 100, -120)):
        rows += [{"game_id": "g", "player_id": 7, "market": "player_points", "book": book, "side": "over", "line": line,
                  "price_american": o},
                 {"game_id": "g", "player_id": 7, "market": "player_points", "book": book, "side": "under", "line": line,
                  "price_american": u}]
    t = PX.market_table(pd.DataFrame(rows))
    assert len(t) == 1 and t[0]["line"] == 24.5 and len(t[0]["books"]) == 2 and t[0]["other_lines"] == [25.5]
    assert t[0]["best_over"] == {"price": -110, "book": "fd"} and t[0]["best_under"] == {"price": -105, "book": "dk"}
    inputs = {("g", 7): {"min": 34, "p10": 28, "p90": 39, "status": "available", "name": "x",
                         "rates": {s: 0.85 for s in PM.STATS}}}
    out = PX.compare(t, inputs, {s: {"shape": "skewed", "a": 2.7, "b": 0.0, "g": 0.6} for s in PM.STATS})
    assert out[0]["model"]["mean"] == pytest.approx(28.9)
    assert out[0]["model"]["edge_over"] > 0


def test_skewed_shape_matches_scaled_counts():
    # points scored in 2s and 3s: spread grows with the square root of the average, about 2.7x a Poisson's
    rng = np.random.default_rng(5)
    mu = rng.uniform(5, 30, 40000)
    x = 2.0 * rng.poisson(mu / 2.0) + rng.binomial(1, 0.3, len(mu)) - 0.3
    line = np.floor(mu) + 0.5
    sh = PM.fit_skewed(mu, x, line)
    assert 1.6 < sh["a"] < 2.6 and sh["b"] < 0.01
    po, pu, push = PM.over_under(20.5, 0.6, sh, 34, 28, 39)
    assert po + pu == pytest.approx(1.0) and push == 0
    po2, pu2, push2 = PM.over_under(20.0, 0.6, sh, 34, 28, 39)
    assert po2 + pu2 + push2 == pytest.approx(1.0) and 0 < push2 < 0.2
