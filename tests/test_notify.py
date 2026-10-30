"""Phone notifications: only new bets, never twice, retried after a failed send."""
import sqlite3

from pipeline import notify

TOPIC = "test-topic"


class FakeSession:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def post(self, url, data, timeout, headers):
        if self.fail:
            raise ConnectionError("offline")
        self.sent.append((url, headers["Title"].decode(), data.decode()))

        class R:
            def raise_for_status(self):
                pass
        return R()


def db():
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE model_calibration (key TEXT PRIMARY KEY, payload TEXT, created_at TEXT);
        CREATE TABLE teams (team_id INTEGER, abbr TEXT);
        CREATE TABLE players (player_id INTEGER, full_name TEXT);
        CREATE TABLE games (game_id TEXT, home_team_id INTEGER, away_team_id INTEGER, tip_utc TEXT);
        CREATE TABLE bets (bet_id INTEGER PRIMARY KEY, game_id TEXT, kind TEXT, player_id INTEGER, stat TEXT,
                           side TEXT, line REAL, price REAL, book TEXT, units REAL, ev REAL);
        INSERT INTO teams VALUES (1, 'NYK'), (2, 'BOS');
        INSERT INTO players VALUES (7, 'Jalen Brunson');
        INSERT INTO games VALUES ('g1', 1, 2, '2026-11-01T23:30:00Z'), ('g0', 2, 1, '2026-10-30T23:00:00Z');
        INSERT INTO bets VALUES (1, 'g1', 'moneyline', NULL, NULL, 'away', 0, 120, 'draftkings', 1.0, 0.05);
    """)
    return c


def add(c, bet_id, game="g1", **kw):
    b = {"kind": "prop", "player_id": 7, "stat": "pts", "side": "over", "line": 26.5, "price": -110,
         "book": "fanduel", "units": 0.75, "ev": 0.06}
    b.update(kw)
    c.execute("INSERT INTO bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (bet_id, game, b["kind"], b["player_id"], b["stat"], b["side"], b["line"], b["price"], b["book"],
               b["units"], b["ev"]))


NOW = "2026-11-01T15:00:00Z"


def test_first_run_starts_from_the_newest_bet():
    c, s = db(), FakeSession()
    assert notify.notify_new_bets(c, TOPIC, NOW, s)["status"] == "nothing new"
    assert s.sent == []                       # the existing ledger is not sent


def test_new_bets_sent_once_with_readable_lines():
    c, s = db(), FakeSession()
    notify.notify_new_bets(c, TOPIC, NOW, s)
    add(c, 2)
    add(c, 3, kind="spread", player_id=None, stat=None, side="home", line=-4.5, price=-108, book="williamhill_us",
        units=1.25)
    r = notify.notify_new_bets(c, TOPIC, NOW, s)
    assert r == {"status": "sent", "bets": 2}
    url, title, body = s.sent[0]
    assert url.endswith("/test-topic") and title == "2 new picks · 2 units"
    assert "BOS @ NYK, 6:30 PM ET" in body          # Nov 1: clocks fell back, UTC-5
    assert "Jalen Brunson over 26.5 points -110 · 0.75u · FanDuel" in body
    assert "NYK -4.5 -108 · 1.25u · Caesars" in body
    assert notify.notify_new_bets(c, TOPIC, NOW, s)["status"] == "nothing new"
    assert len(s.sent) == 1


def test_failed_send_is_retried_and_started_games_skipped():
    c = db()
    notify.notify_new_bets(c, TOPIC, NOW, FakeSession())
    add(c, 2)
    assert notify.notify_new_bets(c, TOPIC, NOW, FakeSession(fail=True))["status"] == "error"
    s = FakeSession()
    assert notify.notify_new_bets(c, TOPIC, NOW, s)["status"] == "sent"      # retried
    add(c, 3, game="g0")                                                     # that game already started
    assert notify.notify_new_bets(c, TOPIC, NOW, s)["status"] == "nothing new"
    assert len(s.sent) == 1


def test_no_topic_does_nothing():
    assert notify.notify_new_bets(db(), "", NOW)["status"] == "skipped"
