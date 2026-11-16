"""Tracking mode, the extra markets and news pulls (Nov 16, 2026)."""
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pipeline.export import bets as B
from pipeline.export import props as PX
from pipeline.models import props as PM
from pipeline.providers import odds_api as OA

NOW = datetime(2026, 11, 16, 18, 0, tzinfo=timezone.utc)      # 1 PM Eastern


def test_parse_props_double_double_and_alternates():
    payload = {"bookmakers": [{"key": "fanduel", "title": "FanDuel", "markets": [
        {"key": "player_double_double", "outcomes": [
            {"name": "Yes", "description": "Nikola Jokic", "price": -400},
            {"name": "No", "description": "Nikola Jokic", "price": 280}]},
        {"key": "player_points_alternate", "outcomes": [
            {"name": "Over", "description": "Nikola Jokic", "point": 34.5, "price": 210}]},
        {"key": "player_steals", "outcomes": [
            {"name": "Over", "description": "Nikola Jokic", "point": 1.5, "price": 150},
            {"name": "Under", "description": "Nikola Jokic", "point": 1.5, "price": -190}]}]}]}
    rows, un = OA.parse_props(payload, {"nikola jokic": 203999})
    got = {(r["market"], r["side"], r["line"]) for r in rows}
    assert ("player_double_double", "over", 0.5) in got and ("player_double_double", "under", 0.5) in got
    assert ("player_points_alternate", "over", 34.5) in got and ("player_steals", "under", 1.5) in got
    assert un == []


def test_parse_event_game_lines():
    payload = {"home_team": "Denver Nuggets", "away_team": "Utah Jazz", "bookmakers": [{"key": "fanduel", "markets": [
        {"key": "team_totals", "outcomes": [
            {"name": "Over", "description": "Denver Nuggets", "point": 121.5, "price": -110},
            {"name": "Under", "description": "Denver Nuggets", "point": 121.5, "price": -110},
            {"name": "Over", "description": "Utah Jazz", "point": 108.5, "price": -115}]},
        {"key": "h2h_h1", "outcomes": [{"name": "Denver Nuggets", "price": -300}, {"name": "Utah Jazz", "price": 230}]},
        {"key": "spreads_h1", "outcomes": [{"name": "Denver Nuggets", "point": -6.5, "price": -110},
                                            {"name": "Utah Jazz", "point": 6.5, "price": -110}]},
        {"key": "alternate_totals", "outcomes": [{"name": "Over", "point": 235.5, "price": 150}]}]}]}
    rows = OA.parse_event_game_lines(payload, {"game_id": "g"})
    sides = {(r["market"], r["side"], r["line"]) for r in rows}
    assert ("team_totals", "home_over", 121.5) in sides and ("team_totals", "away_over", 108.5) in sides
    assert ("h2h_h1", "home", 0.0) in sides and ("spreads_h1", "away", 6.5) in sides
    assert ("alternate_totals", "over", 235.5) in sides


def test_plan_pull_extra_markets_and_news():
    tip = NOW + timedelta(hours=2)
    state = {"pulls": []}
    ids, _ = OA.plan_pull(state, NOW, {"g": tip}, {}, "2026-11-16")
    assert ids == ["g"] and state["_plan"]["ext"] == ["g"]         # first pull of the day brings the extras
    last = NOW - timedelta(minutes=20)
    state = {"pulls": [], "ext": {"g": [last.strftime("%Y-%m-%dT%H:%M:%SZ")]}}
    ids, _ = OA.plan_pull(state, NOW, {"g": tip}, {"g": last}, "2026-11-16")
    assert ids == []                                              # 20 minutes ago: not due ...
    ids, _ = OA.plan_pull(state, NOW, {"g": tip}, {"g": last}, "2026-11-16", news={"g"})
    assert ids == ["g"] and state["_plan"]["news"] == ["g"]       # ... unless there is news
    assert state["_plan"]["ext"] == []                            # extras: not yet near tip
    near = NOW + timedelta(minutes=60)
    old = NOW - timedelta(hours=5)
    state = {"pulls": [], "ext": {"g": [old.strftime("%Y-%m-%dT%H:%M:%SZ")]}}
    ids, _ = OA.plan_pull(state, NOW, {"g": near}, {"g": NOW - timedelta(minutes=58)}, "2026-11-16")
    assert ids == ["g"] and state["_plan"]["ext"] == ["g"]        # second extras pull, near tip
    assert OA.news_games({"fp": {"g": "a", "h": "b"}}, {"g": "a2", "h": "b", "new": "c"}) == {"g"}


def test_alt_table_and_double_double_model():
    rows = []
    for book, o, u in (("fanduel", 200, None), ("draftkings", 180, None), ("fanatics", 190, -260)):
        rows.append({"game_id": "g", "player_id": 7, "market": "player_points_alternate", "book": book, "side": "over",
                     "line": 30.5, "price_american": o, "fetched_at": "t"})
        if u:
            rows.append({"game_id": "g", "player_id": 7, "market": "player_points_alternate", "book": book,
                         "side": "under", "line": 30.5, "price_american": u, "fetched_at": "t"})
        rows.append({"game_id": "g", "player_id": 7, "market": "player_points_alternate", "book": book, "side": "over",
                     "line": 24.5, "price_american": -110, "fetched_at": "t"})
    t = PX.alt_table(pd.DataFrame(rows), [{"game_id": "g", "player_id": 7, "stat": "pts", "line": 24.5}])
    assert len(t) == 1 and t[0]["line"] == 30.5 and t[0]["stat"] == "alt_pts"     # the main line is left out
    assert t[0]["best_over"] == {"price": 200, "book": "fanduel"}
    assert abs(t[0]["fair_over"] - np.median([PM.implied(200) / 1.08, PM.implied(180) / 1.08,
                                              PM.devig(190, -260)[0]])) < 1e-3
    shapes = {s: {"shape": "nb", "k": 20.0} for s in ("pts", "reb", "ast")}
    big = PM.double_double({"pts": 0.8, "reb": 0.35, "ast": 0.3}, shapes, 34, 30, 38)
    small = PM.double_double({"pts": 0.4, "reb": 0.15, "ast": 0.1}, shapes, 24, 18, 30)
    assert 0.6 < big < 0.99 and small < 0.05


def test_game_markets_and_probs():
    rows = []
    for book in ("fanduel", "draftkings", "fanatics"):
        rows += [{"game_id": "g", "market": "team_totals", "book": book, "side": "home_over", "line": 118.5,
                  "price_american": -130 if book == "fanduel" else -110, "fetched_at": "t"},
                 {"game_id": "g", "market": "team_totals", "book": book, "side": "home_under", "line": 118.5,
                  "price_american": 110 if book == "fanduel" else -110, "fetched_at": "t"},
                 {"game_id": "g", "market": "totals", "book": book, "side": "over", "line": 228.5,
                  "price_american": -110, "fetched_at": "t"},
                 {"game_id": "g", "market": "totals", "book": book, "side": "under", "line": 228.5,
                  "price_american": -110, "fetched_at": "t"}]
        for line in (228.5, 234.5):
            rows += [{"game_id": "g", "market": "alternate_totals", "book": book, "side": "over", "line": line,
                      "price_american": 160, "fetched_at": "t"},
                     {"game_id": "g", "market": "alternate_totals", "book": book, "side": "under", "line": line,
                      "price_american": -200, "fetched_at": "t"}]
    t = B.game_market_table(pd.DataFrame(rows))
    kinds = sorted((e["market"], e.get("team"), e["line"]) for e in t)
    assert kinds == [("alternate_totals", None, 234.5), ("team_totals", "home", 118.5), ("totals", None, 228.5)]
    sim = {"home_win": 0.6, "margin": 3.0, "total": 230.0, "margin_sd": 12.0, "total_sd": 20.0,
           "home_pts": 115.8, "away_pts": 113.5, "home_pts_sd": 12.0, "away_pts_sd": 12.0,
           "h1": {"home_win": 0.55, "tie": 0.03, "margin_mean": 1.5, "margin_sd": 9.0, "total_mean": 114.0,
                  "total_sd": 13.0}}
    rep = {"margin_rmse": 14.0, "total_rmse": 19.0}
    p_h1, push = B.game_probs(sim, "h2h_h1", 0.0, rep)
    assert (p_h1, push) == (0.55, 0.03)
    p_tt, _ = B.game_probs(sim, "team_totals", 118.5, rep, "home")
    assert p_tt < 0.5                                            # projected 116.5 against 118.5
    p_t1, _ = B.game_probs(sim, "totals_h1", 110.5, rep)
    assert p_t1 > 0.5
    cands = B.game_candidates([e for e in t if e["market"] == "team_totals"],
                              {"games": [{"game_id": "g", **sim}], "model": {"report": {"scores": rep}}})
    assert cands and cands[0]["side"] == "home_under" and cands[0]["gate"] == "team_total:under"


def _db():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript("""
        CREATE TABLE games (game_id TEXT PRIMARY KEY, season_id TEXT, game_date TEXT, tip_utc TEXT,
            home_team_id INT, away_team_id INT, status TEXT, home_score INT, away_score INT);
        CREATE TABLE odds_snapshots (game_id TEXT, fetched_at TEXT, book TEXT, market TEXT, player_id INT,
            side TEXT, line REAL, price_american INT, is_closing INT DEFAULT 0, source TEXT);
        CREATE TABLE pbp_poss (game_id TEXT, team_id INT, period INT, pts INT);
    """)
    B.ensure_table(c)
    return c


def test_grade_new_kinds_and_closing_value():
    c = _db()
    c.execute("INSERT INTO games VALUES ('g','2026-27','2026-11-15','2026-11-16T00:00:00Z',1,2,'final',112,108)")
    for team, per, pts in ((1, 1, 30), (1, 2, 28), (1, 3, 27), (1, 4, 27), (2, 1, 25), (2, 2, 26), (2, 3, 30),
                           (2, 4, 27)):
        c.execute("INSERT INTO pbp_poss VALUES ('g', ?, ?, ?)", (team, per, pts))
    ins = """INSERT INTO bets (created_at, game_id, kind, side, line, price, book, p_model, p_market, p_used, ev,
               units, mode) VALUES ('2026-11-15T20:00:00Z','g',?,?,?,-110,'fanduel',0.55,0.5,0.52,0.04,0.5,'track')"""
    c.execute(ins, ("team_total", "home_over", 110.5))      # home scored 112: won
    c.execute(ins, ("spread_h1", "home", -2.5))             # first half 58-51: home by 7, won
    c.execute(ins, ("total_h1", "over", 110.5))             # 109: lost
    c.execute(ins, ("moneyline_h1", "away", None))          # lost
    c.execute(ins, ("alt_total", "under", 225.5))           # 220: won
    # closing prices for the team total: one pull after the pick
    for book in ("fanduel", "draftkings"):
        for side, price in (("home_over", -125), ("home_under", 105)):
            c.execute("""INSERT INTO odds_snapshots VALUES ('g','2026-11-15T23:00:00Z',?,'team_totals',NULL,?,110.5,?,0,
                         'odds_api')""", (book, side, price))
    c.commit()
    assert B.grade(c, "2026-27") == 5
    got = {r["kind"]: (r["status"], r["clv"]) for r in c.execute("SELECT kind, status, clv FROM bets")}
    assert got["team_total"][0] == "won" and got["team_total"][1] > 0          # line moved toward the over
    assert got["spread_h1"][0] == "won" and got["total_h1"][0] == "lost"
    assert got["moneyline_h1"][0] == "lost" and got["alt_total"][0] == "won"
    assert got["alt_total"][1] is None                                         # no later price: no CLV


def test_first_half_waits_for_matching_play_by_play():
    c = _db()
    c.execute("INSERT INTO games VALUES ('g','2026-27','2026-11-15','2026-11-16T00:00:00Z',1,2,'final',112,108)")
    c.execute("INSERT INTO pbp_poss VALUES ('g', 1, 1, 30)")                  # doesn't add up to the final
    c.execute("""INSERT INTO bets (created_at, game_id, kind, side, line, price, book, p_model, p_market, p_used, ev,
                 units) VALUES ('2026-11-15T20:00:00Z','g','total_h1','over',110.5,-110,'fanduel',.55,.5,.52,.04,.5)""")
    c.commit()
    B.grade(c, "2026-27")
    assert c.execute("SELECT status FROM bets").fetchone()[0] == "open"
