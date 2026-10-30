"""Bet overrides: the odds you got are checked, saved to overrides.json, and graded."""
import json

import pytest

from pipeline import db
from pipeline import overrides as O
from pipeline.export import bets as B


def body(**kw):
    return "I got different odds for this bet.\n\noverride: " + json.dumps(kw) + "\n"


def test_parse_checks_every_field():
    assert O.parse(body(bet_id=3, price=-115, line=26.5, book="FanDuel")) == \
        {"bet_id": 3, "price": -115, "line": 26.5, "book": "FanDuel"}
    assert O.parse(body(bet_id=3, price="+130", line="", book="BetMGM"))["line"] is None
    assert O.parse(body(bet_id=3, clear=True)) == {"bet_id": 3, "clear": True}
    for bad in (dict(bet_id=3, price=-50, book="X"), dict(bet_id=3, price=-110, line=26.3, book="X"),
                dict(bet_id=3, price=-110, book="<script>"), dict(price=-110, book="X"), dict(bet_id=3, price="abc", book="X")):
        with pytest.raises(O.OverrideError):
            O.parse(body(**bad))
    with pytest.raises(O.OverrideError):
        O.parse("no machine line here")


def test_record_saves_and_clears(tmp_path):
    f = tmp_path / "overrides.json"
    O.record(body(bet_id=2, price=-120, line=None, book="Caesars"), f)
    O.record(body(bet_id=1, price=105, line=3.5, book="DraftKings"), f)
    data = json.loads(f.read_text())["bets"]
    assert list(data) == ["1", "2"] and data["2"]["price"] == -120 and data["1"]["line"] == 3.5
    O.record(body(bet_id=2, clear=True), f)
    assert list(json.loads(f.read_text())["bets"]) == ["1"]


def setup(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    B.ensure_table(conn)
    conn.execute("INSERT INTO seasons VALUES ('2026-27', '2026-10-20', '2027-04-11')")
    for tid, ab in ((1610612738, "BOS"), (1610612747, "LAL")):
        conn.execute("INSERT OR IGNORE INTO teams (team_id, abbr, city, name) VALUES (?, ?, 'x', ?)", (tid, ab, ab))
    conn.execute("""INSERT INTO games (game_id, season_id, season_type, game_date, tip_utc, home_team_id, away_team_id,
                    status, home_score, away_score, source, fetched_at) VALUES ('g1', '2026-27', 'regular', '2026-10-28',
                    '2026-10-28T23:00:00Z', 1610612738, 1610612747, 'final', 110, 104, 'test', 'x')""")
    # recommended: home -5.5 at -110 (home won by 6, so it wins)
    conn.execute("""INSERT INTO bets (created_at, game_id, kind, side, line, price, book, p_model, p_market, p_used,
                    ev, units) VALUES ('2026-10-28T16:00:00Z', 'g1', 'spread', 'home', -5.5, -110, 'fanduel', .55, .5,
                    .525, .05, 1)""")
    conn.commit()
    return conn


def test_graded_at_your_odds_and_line(tmp_path):
    conn, f = setup(tmp_path), tmp_path / "overrides.json"
    O.record(body(bet_id=1, price=+100, line=-5.5, book="BetMGM"), f)
    assert O.apply_file(conn, f)["changed"] == 1
    assert O.apply_file(conn, f)["changed"] == 0               # idempotent
    B.grade(conn, "2026-27")
    st, pnl, price, line = conn.execute("SELECT status, pnl, price, line FROM bets").fetchone()
    assert st == "won" and pnl == pytest.approx(1.0) and (price, line) == (-110, -5.5)   # recommendation kept
    # a worse line you got (-6.5) turns the same game into a loss; the graded bet is reopened and regraded
    O.record(body(bet_id=1, price=-105, line=-6.5, book="BetMGM"), f)
    O.apply_file(conn, f)
    assert conn.execute("SELECT status FROM bets").fetchone()[0] == "open"
    B.grade(conn, "2026-27")
    assert tuple(conn.execute("SELECT status, pnl FROM bets").fetchone()) == ("lost", -1.0)
    # back to the recommendation
    O.record(body(bet_id=1, clear=True), f)
    O.apply_file(conn, f)
    B.grade(conn, "2026-27")
    st, pnl, tp = conn.execute("SELECT status, pnl, taken_price FROM bets").fetchone()
    assert st == "won" and pnl == pytest.approx(0.9091, abs=1e-3) and tp is None


def test_unknown_bet_is_reported_not_applied(tmp_path):
    conn, f = setup(tmp_path), tmp_path / "overrides.json"
    O.record(body(bet_id=99, price=-110, book="FanDuel"), f)
    assert O.apply_file(conn, f)["unknown_bets"] == ["99"]


def test_results_by_bet_type():
    assert B.market_of("prop", "fg3m") == "fg3m" and B.market_of("spread", None) == "spread"
