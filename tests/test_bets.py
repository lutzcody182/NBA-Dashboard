"""Recommended bets: game-line chances, staking, selection caps and grading. Test use only."""
import pandas as pd
import pytest

from pipeline import db
from pipeline.export import bets as B

REPORT = {"margin_bias": 0.0, "margin_rmse": 14.0, "total_bias": 0.0, "total_rmse": 19.0}


def test_game_probs():
    sim = {"home_win": 0.62, "margin": 4.0, "total": 225.0}
    assert B.game_probs(sim, "h2h", 0.0, REPORT) == (0.62, 0.0)
    p, push = B.game_probs(sim, "spreads", -4.5, REPORT)        # home must win by 5+
    assert 0.45 < p < 0.5 and push == 0
    p, push = B.game_probs(sim, "spreads", -4.0, REPORT)
    assert push > 0
    p, _ = B.game_probs(sim, "totals", 220.5, REPORT)
    assert p > 0.5


def test_kelly_units_are_small_and_capped():
    assert B.kelly_units(0.50, -110) == 0.0
    u = B.kelly_units(0.56, -110)
    assert 0.25 <= u <= 1.5 and (u * 4) % 1 == 0
    assert B.kelly_units(0.9, 200) == B.MAX_UNITS


def test_candidate_needs_model_on_its_side():
    assert B._candidate("total", "g", "over", 220.5, 100, "fanduel", 0.60, 0.50, 0, 5) is not None   # 35% model: 53.5% at even money
    # cheap price but the model likes the other side: no bet
    assert B._candidate("total", "g", "over", 220.5, 150, "fanduel", 0.45, 0.50, 0, 5) is None
    # too big a gap: probably news the model hasn't seen
    assert B._candidate("total", "g", "over", 220.5, -110, "fanduel", 0.75, 0.50, 0, 5) is None
    assert B._candidate("total", "g", "over", 220.5, -110, "fanduel", 0.60, 0.50, 0, 2) is None   # too few books


def test_selection_caps():
    cands = []
    for i in range(30):
        cands.append({"kind": "prop", "game_id": f"g{i % 3}", "player_id": i % 7, "stat": "pts", "side": "under",
                      "ev": 0.05 + i / 1000, "units": 1.0, "gate": "pts:under"})
    earned = {"pts:under": {"status": "earned"}}
    picks = B.select([dict(c) for c in cands], earned)
    # no daily limit: every player's best prop is kept (one per player per game), stakes scaled to the game cap
    keys = {(c["game_id"], c["player_id"]) for c in cands}
    assert len(picks) == len(keys) and all(p["mode"] == "rec" for p in picks)
    for g in ("g0", "g1", "g2"):
        n = sum(1 for p in picks if p["game_id"] == g)
        assert sum(p["units"] for p in picks if p["game_id"] == g) <= max(B.MAX_GAME_UNITS, 0.25 * n)
    keys = [(p["game_id"], p["player_id"]) for p in picks]
    assert len(keys) == len(set(keys))
    # nothing earned: the same picks are only tracked, at their own stakes (no game cap)
    tracked = B.select([dict(c) for c in cands], {})
    assert len(tracked) == len(picks) and all(p["mode"] == "track" and p["units"] == 1.0 for p in tracked)


def test_gates():
    rows = []
    for i in range(200):          # under picks: closing line moves our way, they win a bit more than expected
        rows.append({"kind": "prop", "stat": "pts", "side": "under", "status": "won" if i % 100 < 55 else "lost",
                     "p_market": 0.5, "clv": 0.01 + (0.005 if i % 2 else -0.005), "units": 0.5,
                     "pnl": 0.45 if i % 100 < 55 else -0.5, "gate": None, "trigger": None})
    for i in range(200):          # over picks: no closing-line edge
        rows.append({"kind": "prop", "stat": "pts", "side": "over", "status": "won" if i % 2 else "lost",
                     "p_market": 0.5, "clv": 0.01 if i % 2 else -0.01, "units": 0.5, "pnl": 0.45 if i % 2 else -0.5,
                     "gate": None, "trigger": None})
    g = B.gates(pd.DataFrame(rows))
    assert g["pts:under"]["status"] == "earned" and g["pts:under"]["needs"] == []
    assert g["pts:over"]["status"] == "tracking" and g["pts:over"]["graded"] == 200
    assert B.gate_key("team_total", None, "home_over") == "team_total:over"
    assert B._earned({"gate": "pts:under"}, g) and not B._earned({"gate": "pts:over"}, g)


def test_game_market_table_spread_pairs():
    rows = []
    for book, hl, hp, ap in (("draftkings", -3.5, -110, -110), ("fanduel", -3.5, -105, -115), ("betmgm", -4.0, -110, -110)):
        rows += [{"game_id": "g", "market": "spreads", "book": book, "side": "home", "line": hl, "price_american": hp},
                 {"game_id": "g", "market": "spreads", "book": book, "side": "away", "line": -hl, "price_american": ap}]
    t = B.game_market_table(pd.DataFrame(rows))
    assert len(t) == 1 and t[0]["line"] == -3.5 and len(t[0]["books"]) == 2
    assert t[0]["best"]["home"] == {"price": -105, "book": "fanduel"}


def test_grading_game_lines(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    conn.executescript(B.TABLE)
    conn.execute("INSERT INTO seasons VALUES ('2026-27', '2026-10-20', '2027-04-11')")
    for tid, ab in ((1610612738, "BOS"), (1610612747, "LAL")):
        conn.execute("INSERT OR IGNORE INTO teams (team_id, abbr, city, name) VALUES (?, ?, 'x', ?)", (tid, ab, ab))
    conn.execute("""INSERT INTO games (game_id, season_id, season_type, game_date, tip_utc, home_team_id, away_team_id,
                    status, home_score, away_score, source, fetched_at) VALUES ('g1', '2026-27', 'regular', '2026-10-28',
                    '2026-10-28T23:00:00Z', 1610612738, 1610612747, 'final', 110, 104, 'test', 'x')""")
    for kind, side, line, price in (("spread", "home", -5.5, -110), ("total", "over", 214.0, -110),
                                    ("moneyline", "away", None, 150)):
        conn.execute("""INSERT INTO bets (created_at, game_id, kind, side, line, price, book, p_model, p_market, p_used,
                        ev, units) VALUES ('2026-10-28T16:00:00Z', 'g1', ?, ?, ?, ?, 'fanduel', .55, .5, .525, .05, 1)""",
                     (kind, side, line, price))
    conn.commit()
    assert B.grade(conn, "2026-27") == 3
    res = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT kind, status, pnl FROM bets")}
    assert res["spread"][0] == "won" and res["spread"][1] == pytest.approx(0.9091, abs=1e-3)
    assert res["total"] == ("push", 0.0)
    assert res["moneyline"] == ("lost", -1.0)


def test_beat_close_ignores_rounding_noise():
    # bets locked from the last pull before tip-off have CLV 0 (or float noise); they did not beat the close
    df = pd.DataFrame({"status": ["won", "lost", "won"], "units": [1.0, 1.0, 1.0], "pnl": [0.9, -1.0, 0.9],
                       "clv": [0.0, 1e-17, 0.02]})
    s = B.summary(df)
    assert s["beat_close"] == pytest.approx(1 / 3, abs=1e-3) and s["clv_n"] == 3


def test_track_states(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    B.ensure_table(conn)
    conn.execute("INSERT INTO seasons VALUES ('2026-27', '2026-10-20', '2027-04-11')")
    for tid, ab in ((1610612738, "BOS"), (1610612747, "LAL")):
        conn.execute("INSERT OR IGNORE INTO teams (team_id, abbr, city, name) VALUES (?, ?, 'x', ?)", (tid, ab, ab))
    for gid, tip in (("g1", "2026-11-03T23:00:00Z"), ("g0", "2026-11-03T15:00:00Z")):
        conn.execute("""INSERT INTO games (game_id, season_id, season_type, game_date, tip_utc, home_team_id, away_team_id,
                        status, source, fetched_at) VALUES (?, '2026-27', 'regular', '2026-11-03', ?, 1610612738,
                        1610612747, 'scheduled', 'test', 'x')""", (gid, tip))
    rows = [("g1", 7, "pts", "over", 25.5, -110), ("g1", 8, "reb", "under", 9.5, -105), ("g1", 9, "ast", "over", 6.5, 120),
            ("g0", 7, "pts", "over", 20.5, -110)]
    for gid, pid, stat, side, line, price in rows:
        conn.execute("""INSERT INTO bets (created_at, game_id, kind, player_id, stat, side, line, price, book, p_model,
                        p_market, p_used, ev, units) VALUES ('2026-11-03T14:00:00Z', ?, 'prop', ?, ?, ?, ?, ?, 'fanduel',
                        .6, .5, .535, .05, 1)""", (gid, pid, stat, side, line, price))
    conn.commit()
    pick = {"kind": "prop", "game_id": "g1", "player_id": 7, "stat": "pts", "side": "over", "line": 25.5, "price": -105,
            "book": "draftkings", "units": 0.75}
    prop_rows = [{"game_id": "g1", "player_id": 8, "stat": "reb", "line": 10.5, "fair_over": .5, "best_over": None,
                  "best_under": None, "model": {}},
                 {"game_id": "g1", "player_id": 9, "stat": "ast", "line": 6.5, "fair_over": .5,
                  "best_over": {"price": 100, "book": "fanduel"}, "best_under": {"price": -120, "book": "fanduel"},
                  "model": {"p_over": .52, "p_under": .48}}]
    games = {"g1": {"tip_utc": "2026-11-03T23:00:00Z"}}
    out = B.track(conn, [pick], [pick], prop_rows, [], games, "2026-11-03T17:00:00Z")
    assert out == {"active": 1, "expired": 2, "started": 1}
    st = {r[0]: r[1:] for r in conn.execute("SELECT player_id || stat || game_id, rec_state, end_reason, now_price FROM bets")}
    assert st["7ptsg1"] == ("active", None, -105)
    assert st["8rebg1"][:2] == ("expired", "line moved to 10.5")
    assert st["9astg1"][:2] == ("expired", "price moved to +100")
    assert st["7ptsg0"][0] == "started"
    # an expired one keeps its first reason and time on later runs
    B.track(conn, [], [], prop_rows, [], games, "2026-11-03T18:00:00Z")
    assert tuple(conn.execute("SELECT end_reason, ended_at FROM bets WHERE player_id=8").fetchone()) == ("line moved to 10.5", "2026-11-03T17:00:00Z")
    assert tuple(conn.execute("SELECT rec_state, end_reason FROM bets WHERE player_id=7 AND game_id='g1'").fetchone()) == \
        ("expired", "no longer offered by enough legal books")
    assert B.track(conn, [], [], [], [], games, "2026-11-03T20:00:00Z") == {}      # nothing priced: nothing changes


def test_stake_cap_and_overs_tracked():
    c = B._candidate("prop", "g", "under", 20.5, 120, "fanduel", 0.62, 0.53, 0.0, 5, player_id=7, stat="pts")
    assert c is not None and c["units"] == B.MAX_UNITS == 0.5 and c["units_full"] > c["units"]
    assert c["gate"] == "pts:under"
    row = {"game_id": "g", "player_id": 7, "stat": "pts", "line": 20.5, "fair_over": 0.40,
           "books": [{"book": b} for b in ("fanduel", "draftkings", "betmgm")],
           "best_over": {"price": 140, "book": "fanduel"}, "best_under": {"price": -170, "book": "fanduel"},
           "model": {"p_over": 0.50, "p_under": 0.50, "games": 20, "status": "available", "mean": 20, "min": 30}}
    assert B.PROP_OVERS is False
    out = B.prop_candidates([row])                       # overs are tracked again (Nov 16, 2026) ...
    assert [c["side"] for c in out] == ["over"]
    picks = B.select(out, {})                            # ... but only earn a recommendation through the gate
    assert picks[0]["mode"] == "track"
    alt = {**row, "stat": "alt_pts", "line": 25.5, "alt": True, "books": [{"book": "fanduel"}, {"book": "fanatics"}]}
    assert [c["stat"] for c in B.prop_candidates([alt])] == ["alt_pts"]      # alternates need only two books
