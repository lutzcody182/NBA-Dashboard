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


def test_pull_plan():
    # NOW is 2026-10-28 16:00 UTC (noon Eastern)
    P = lambda state, up, last, now=NOW, day="2026-10-28": OA.plan_pull(state, now, up, last, day)
    h = lambda x: NOW + timedelta(hours=x)
    up = {"early": h(3), "late": h(7), "tomorrow": h(20)}
    # morning: every game of the day not pulled yet today, soonest first; tomorrow's games wait
    assert P({"pulls": []}, up, {})[0] == ["early", "late"]
    # pulled 30 minutes ago: the game 3 hours out waits for the hourly re-pull; the one 7 hours out waits too
    last = {"early": h(-0.5), "late": h(-0.5)}
    ids, why = P({"pulls": []}, up, last)
    assert ids == [] and "next pull in about 25 minutes" in why
    # an hour later: only the game inside the last 4 hours is pulled again
    assert P({"pulls": []}, up, {"early": h(-1), "late": h(-1)})[0] == ["early"]
    # a pull from yesterday doesn't count as today's morning pull
    assert P({"pulls": []}, up, {"early": h(-20), "late": h(-20)})[0] == ["early", "late"]
    # a game about to start is not pulled
    assert P({"pulls": []}, {"now": NOW + timedelta(minutes=3)}, {})[0] == []
    # budget: when short, the soonest tips go first; nothing when the day's or month's share is used
    near_cap = {"pulls": [{"at": "2026-10-28T13:00:00Z", "et_day": "2026-10-28", "cost": OA.DAILY_CREDIT_CAP - OA.pull_cost(1)}]}
    ids, why = P(near_cap, up, {})
    assert ids == ["early"] and "budget covers 1 of 2" in why
    full = {"pulls": [{"at": "2026-10-28T13:00:00Z", "et_day": "2026-10-28", "cost": OA.DAILY_CREDIT_CAP}]}
    assert P(full, up, {}) == ([], "today's NBA odds budget is used")
    month = {"pulls": [{"at": "2026-10-0%dT13:00:00Z" % d, "et_day": "2026-10-0%d" % d, "cost": OA.MONTHLY_CREDIT_CAP / 5}
                       for d in range(1, 6)]}
    assert P(month, up, {}) == ([], "this month's NBA odds budget is used")
    assert P({"pulls": [], "remaining": 100}, up, {})[0] == []                       # shared key nearly empty
    # nothing before 8 AM Eastern
    early = datetime(2026, 10, 30, 9, 56, tzinfo=timezone.utc)          # 5:56 AM EDT
    assert P({"pulls": []}, {"g": early + timedelta(hours=13)}, {}, early, "2026-10-30")[0] == []
    eight = datetime(2026, 10, 30, 12, 5, tzinfo=timezone.utc)          # 8:05 AM EDT
    assert P({"pulls": []}, {"g": eight + timedelta(hours=10)}, {}, eight, "2026-10-30")[0] == ["g"]


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
    for book, line, o, u in (("draftkings", 24.5, -115, -105), ("fanduel", 24.5, -110, -110), ("williamhill_us", 24.5, -112, -108),
                             ("betmgm", 25.5, 100, -120)):
        rows += [{"game_id": "g", "player_id": 7, "market": "player_points", "book": book, "side": "over", "line": line,
                  "price_american": o},
                 {"game_id": "g", "player_id": 7, "market": "player_points", "book": book, "side": "under", "line": line,
                  "price_american": u}]
    t = PX.market_table(pd.DataFrame(rows))
    assert len(t) == 1 and t[0]["line"] == 24.5 and len(t[0]["books"]) == 3 and t[0]["other_lines"] == [25.5]
    # DraftKings has the best under (-105) and counts toward the market, but best prices only come from your books
    assert t[0]["best_over"] == {"price": -110, "book": "fanduel"} and t[0]["best_under"] == {"price": -108, "book": "williamhill_us"}
    inputs = {("g", 7): {"min": 34, "p10": 28, "p90": 39, "status": "available", "name": "x",
                         "rates": {s: 0.85 for s in PM.STATS}}}
    out = PX.compare(t, inputs, {s: {"shape": "skewed", "a": 2.7, "b": 0.0, "g": 0.6} for s in PM.STATS})
    assert out[0]["model"]["mean"] == pytest.approx(28.9)
    assert out[0]["model"]["edge_over"] > 0


def test_no_bet_book_at_the_line_means_no_best_price():
    rows = []
    for book in ("draftkings", "betmgm"):
        for side, price in (("over", -110), ("under", -110)):
            rows.append({"game_id": "g", "player_id": 7, "market": "player_points", "book": book, "side": side,
                         "line": 24.5, "price_american": price})
    t = PX.market_table(pd.DataFrame(rows))
    assert t[0]["best_over"]["price"] is None and t[0]["best_under"]["book"] is None
    inputs = {("g", 7): {"min": 34, "p10": 28, "p90": 39, "status": "available", "name": "x",
                         "rates": {s: 0.85 for s in PM.STATS}}}
    m = PX.compare(t, inputs, {s: {"shape": "skewed", "a": 2.7, "b": 0.0, "g": 0.6} for s in PM.STATS})[0]["model"]
    assert m["ev_over"] is None and m["ev_under"] is None
    from pipeline.export import bets as B
    assert B.prop_candidates([{**t[0], "model": {**m, "games": 20}}]) == []


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


def test_anchor_inputs_pulls_minutes_and_rates_toward_history():
    from pipeline.export import props as XP
    x = {"min": 34.0, "p10": 28.0, "p90": 38.0, "rates": {s: 0.0 for s in XP.PM.STATS}, "status": None}
    x["rates"].update({"pts": 0.60, "reb": 0.20, "ast": 0.10, "fg3m": 0.05})
    hist = {7: {"m5": 26.0, "games": 12, "rate": {"pts": 0.40, "reb": 0.20, "ast": 0.10, "fg3m": 0.05}}}
    out = XP.anchor_inputs({("g", 7): x, ("g", 8): x}, hist)
    a = out[("g", 7)]
    assert a["min"] == 30.0                       # halfway from 26 toward 34, inside the 4-minute cap
    assert a["p10"] == 24.0 and a["p90"] == 34.0  # the range moves with it
    assert abs(a["rates"]["pts"] - 0.57) < 1e-9   # 85% simulation, 15% longer record
    assert abs(a["rates"]["pra"] - (0.57 + 0.20 + 0.10)) < 1e-9
    assert out[("g", 8)] is x                     # no history: unchanged
    far = XP.anchor_inputs({("g", 7): {**x, "min": 46.0}}, hist)[("g", 7)]
    assert far["min"] == 30.0                     # capped at 4 minutes from the last-5 average
    few = XP.anchor_inputs({("g", 7): x}, {7: {**hist[7], "games": 3}})[("g", 7)]
    assert few is x                               # fewer than 5 games: not anchored


def test_pull_fetches_only_due_games(tmp_path, monkeypatch):
    from pipeline import db
    conn = db.connect(tmp_path / "t.sqlite")
    conn.execute("PRAGMA foreign_keys = OFF")          # only odds rows are needed here
    for tid, city, name in ((1610612738, "Boston", "Celtics"), (1610612747, "Los Angeles", "Lakers"),
                            (1610612752, "New York", "Knicks"), (1610612744, "Golden State", "Warriors")):
        conn.execute("INSERT OR IGNORE INTO teams (team_id, abbr, city, name) VALUES (?, ?, ?, ?)", (tid, name[:3].upper(), city, name))
    conn.commit()
    soon, later = NOW + timedelta(hours=2), NOW + timedelta(hours=6)
    games = [{"game_id": "g1", "tip_utc": soon.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "scheduled",
              "home_team_id": 1610612738, "away_team_id": 1610612747},
             {"game_id": "g2", "tip_utc": later.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "scheduled",
              "home_team_id": 1610612752, "away_team_id": 1610612744}]
    # both games had their morning pull an hour ago; only g1 (inside the last 4 hours) is due again
    for gid in ("g1", "g2"):
        conn.execute("""INSERT INTO odds_snapshots (game_id, fetched_at, book, market, player_id, side, line, price_american,
                        is_closing, source) VALUES (?, '2026-10-28T15:00:00Z', 'fanduel', 'player_points', 1, 'over', 20.5, -110, 0, ?)""",
                     (gid, OA.SOURCE))
    conn.commit()
    calls = []
    events = [{"id": "e1", "home_team": "Boston Celtics", "away_team": "Los Angeles Lakers", "commence_time": games[0]["tip_utc"]},
              {"id": "e2", "home_team": "New York Knicks", "away_team": "Golden State Warriors", "commence_time": games[1]["tip_utc"]}]
    def fake_get(path, key, params=None):
        calls.append(path)
        if path == "/events":
            return events, 15000.0
        if path == "/odds":
            return [], 14980.0
        return {"bookmakers": []}, 14990.0
    monkeypatch.setattr(OA, "_get", fake_get)
    monkeypatch.setenv("ODDS_API_KEY", "test")
    out = OA.pull(conn, games, {}, now=NOW, et_day="2026-10-28")
    assert "/events/e1/odds" in calls and "/events/e2/odds" not in calls and out["games"] == 1
    st = OA._state(conn)
    assert st["pulls"][-1]["cost"] == 20.0 and OA.spent(st, month="2026-10") == 20.0


def test_lineups_parse_and_minutes():
    from pipeline.providers import nba_lineups as NL
    payload = {"games": [{"gameId": "0022600100", "homeTeam": {"teamId": 1610612738, "players": [
        {"personId": i, "lineupStatus": "Confirmed"} for i in (1, 2, 3, 4, 5)]},
        "awayTeam": {"teamId": 1610612747, "players": [{"personId": i, "lineupStatus": "Expected"} for i in (6, 7, 8, 9, 10)]}}]}
    g = NL.parse(payload)[0]
    assert g["teams"][1610612738]["status"] == "confirmed" and g["teams"][1610612747]["status"] == "projected"
    x = {"min": 24.0, "p10": 18.0, "p90": 30.0, "rates": {s: 0.0 for s in PM.STATS}, "status": None}
    x["rates"].update({"pts": 0.5, "reb": 0.2, "ast": 0.1, "fg3m": 0.05})
    hist = {1: {"m5": 24.0, "games": 20, "st5": 0.0, "team": 1610612738, "rate": x["rates"]},     # bench player, now starting
            11: {"m5": 32.0, "games": 20, "st5": 1.0, "team": 1610612738, "rate": x["rates"]},    # starter, now benched
            2: {"m5": 24.0, "games": 20, "st5": 1.0, "team": 1610612738, "rate": x["rates"]},     # starter, still starting
            6: {"m5": 24.0, "games": 20, "st5": 0.0, "team": 1610612747, "rate": x["rates"]}}     # only "Expected": no change
    lu = {("g", 1610612738): {"status": "confirmed", "starters": {1, 2, 3, 4, 5}},
          ("g", 1610612747): {"status": "projected", "starters": {6, 7, 8, 9, 10}}}
    out = PX.anchor_inputs({("g", p): {**x, "min": (32.0 if p == 11 else 24.0)} for p in (1, 11, 2, 6)}, hist, lu)
    assert out[("g", 1)]["min"] == 24.0 + PX.LINEUP_UP and out[("g", 1)]["lineup"] == "moved into the starting lineup"
    assert out[("g", 11)]["min"] == 32.0 - PX.LINEUP_DOWN and out[("g", 11)]["lineup"] == "moved out of the starting lineup"
    assert out[("g", 2)]["min"] == 24.0 and out[("g", 2)]["lineup"] == "confirmed starter"
    assert out[("g", 6)]["min"] == 24.0 and "lineup" not in out[("g", 6)]
    assert PX.anchor_inputs({("g", 1): x}, hist)[("g", 1)]["min"] == 24.0          # no lineups: unchanged


def test_lineups_update_stores_changes_only(tmp_path):
    from datetime import date
    from pipeline import db
    from pipeline.providers import nba_lineups as NL
    conn = db.connect(tmp_path / "t.sqlite")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("""INSERT INTO games (game_id, season_id, season_type, game_date, tip_utc, home_team_id, away_team_id, status,
                    source, fetched_at) VALUES ('g1', '2026-27', 'regular', '2026-11-06', '2026-11-07T00:00:00Z', 1610612738,
                    1610612747, 'scheduled', 'test', 'x')""")
    for pid in range(1, 11):
        conn.execute("INSERT INTO players (player_id, full_name, fetched_at) VALUES (?, ?, 'x')", (pid, f"P{pid}"))
    conn.commit()
    def payload(status):
        return {"games": [{"homeTeam": {"teamId": 1610612738, "players": [{"personId": i, "lineupStatus": status} for i in range(1, 6)]},
                           "awayTeam": {"teamId": 1610612747, "players": [{"personId": i, "lineupStatus": "Expected"} for i in range(6, 11)]}}]}
    class S:
        def __init__(self, p): self.p = p
        def get(self, url, headers, timeout):
            p = self.p
            class R:
                def raise_for_status(self): pass
                def json(self): return p
            return R()
    day = date(2026, 11, 6)
    assert NL.update(conn, day, S(payload("Expected")), "2026-11-06T18:00:00Z")["teams_stored"] == 2
    assert NL.update(conn, day, S(payload("Expected")), "2026-11-06T18:15:00Z")["teams_stored"] == 0     # unchanged
    r = NL.update(conn, day, S(payload("Confirmed")), "2026-11-06T23:30:00Z")
    assert r["teams_stored"] == 1 and r["teams_confirmed"] == 1
    lu = NL.latest(conn)
    assert lu[("g1", 1610612738)]["status"] == "confirmed" and lu[("g1", 1610612738)]["starters"] == {1, 2, 3, 4, 5}
    assert lu[("g1", 1610612747)]["status"] == "projected"
