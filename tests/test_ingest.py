import json

import pytest

from pipeline import db
from pipeline.ingest.store import store_boxscore, store_schedule
from pipeline.providers.nba_cdn import FeedError, parse_boxscore, parse_schedule
from pipeline.util import duration_to_seconds, season_from_game_id

from .helpers import FIXTURES, make_box, player, real_box

BOS = 1610612738
ORL = 1610612753
NYK = 1610612752


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.sqlite")
    yield c
    c.close()


# -- helpers -------------------------------------------------------------

def test_duration_parsing():
    assert duration_to_seconds("PT25M01.00S") == 1501
    assert duration_to_seconds("PT240M00.00S") == 14400
    assert duration_to_seconds("PT240M") == 14400
    assert duration_to_seconds("") == 0
    with pytest.raises(ValueError):
        duration_to_seconds("25:01")


def test_season_from_game_id():
    assert season_from_game_id("0022600001") == "2026-27"
    assert season_from_game_id("0012500049") == "2025-26"


# -- parsing real NBA responses -------------------------------------------

def test_parse_real_boxscore():
    box = parse_boxscore(real_box())
    assert box.game_id == "0022000180"
    assert box.season_id == "2020-21"
    assert box.game_date == "2021-01-15"
    assert box.status == "final"
    assert box.home.team.tricode == "BOS"
    assert box.home.pts == 124
    assert box.home.seconds == 240 * 60
    jb = box.home.players[0]
    assert jb.full_name == "Jaylen Brown"
    assert jb.seconds == 25 * 60 + 1
    assert (jb.pts, jb.fgm, jb.fga, jb.ftm, jb.fta) == (21, 6, 12, 7, 7)
    assert jb.started and jb.plus_minus == 15
    assert box.officials[0] == (201638, "Brent Barnaky")


def test_parse_real_schedule_skips_all_star():
    data = json.loads((FIXTURES / "schedule_sample.json").read_text())
    games = parse_schedule(data)
    assert games, "expected games"
    assert all(g.game_id[:3] != "003" for g in games)
    first = games[0]
    assert first.game_id == "0012000001"
    assert first.game_date == "2020-12-11"
    assert first.home.tricode == "ATL" and first.away.tricode == "ORL"
    assert (first.home_score, first.away_score) == (112, 116)
    assert first.status == "final"


def test_unexpected_shape_fails_loudly():
    with pytest.raises(FeedError):
        parse_schedule({"wrong": {}})
    with pytest.raises(FeedError):
        parse_boxscore({"game": {}})


# -- storing ---------------------------------------------------------------

def _bos(players):
    return (BOS, "BOS", "Boston", "Celtics", players)


def _orl(players):
    return (ORL, "ORL", "Orlando", "Magic", players)


def test_store_real_boxscore(conn):
    store_boxscore(conn, parse_boxscore(real_box()), "test")
    g = conn.execute("SELECT * FROM games").fetchone()
    assert (g["status"], g["home_score"], g["season_type"]) == ("final", 124, "regular")
    t = conn.execute("SELECT * FROM team_game_stats WHERE team_id = ?", (BOS,)).fetchone()
    assert t["minutes"] == 240 and t["possessions"] > 0
    assert conn.execute("SELECT COUNT(*) FROM player_game_stats").fetchone()[0] == 2
    assert conn.execute("SELECT conference FROM teams WHERE team_id = ?", (BOS,)).fetchone()[0] == "East"


def test_rerun_is_idempotent(conn):
    box = parse_boxscore(real_box())
    store_boxscore(conn, box, "test")
    store_boxscore(conn, box, "test")
    assert conn.execute("SELECT COUNT(*) FROM team_game_stats").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM roster_stints").fetchone()[0] == 2


def test_dnp_and_inactive_players(conn):
    home = [player(1, "A", "Starter", starter=True, position="G"),
            player(2, "B", "Bench", played=False, reason="DNP - Coach's Decision"),
            player(3, "C", "Hurt", played=False, status="INACTIVE", reason="INJURY_ILLNESS")]
    away = [player(10, "X", "Y", starter=True)]
    store_boxscore(conn, parse_boxscore(make_box("0022600001", "2026-10-21", _bos(home), _orl(away))), "t")
    rows = {r["player_id"]: r for r in conn.execute("SELECT * FROM player_game_stats")}
    assert rows[2]["seconds"] == 0 and rows[2]["dnp_reason"] == "DNP - Coach's Decision"
    assert rows[3]["dnp_reason"] == "INJURY_ILLNESS"
    assert rows[1]["dnp_reason"] is None
    # inactive players are still on the game-day roster
    on_roster = {r[0] for r in conn.execute("SELECT player_id FROM roster_stints WHERE team_id = ? AND end_date IS NULL", (BOS,))}
    assert on_roster == {1, 2, 3}
    assert conn.execute("SELECT position FROM players WHERE player_id = 1").fetchone()[0] == "G"


def test_trade_moves_player_and_keeps_history(conn):
    traded = lambda: player(7, "T", "Traded")
    g1 = make_box("0022600001", "2026-10-21", _bos([traded(), player(1, "A", "B")]), _orl([player(10, "X", "Y")]))
    g2 = make_box("0022600050", "2026-11-05",
                  (NYK, "NYK", "New York", "Knicks", [traded()]), _orl([player(10, "X", "Y")]))
    g3 = make_box("0022600060", "2026-11-06", _bos([player(1, "A", "B")]), _orl([player(10, "X", "Y")]))
    for b in (g1, g2, g3):
        store_boxscore(conn, parse_boxscore(b), "t")
    stints = conn.execute(
        "SELECT team_id, start_date, end_date FROM roster_stints WHERE player_id = 7 ORDER BY start_date"
    ).fetchall()
    assert [tuple(s) for s in stints] == [(BOS, "2026-10-21", "2026-11-05"), (NYK, "2026-11-05", None)]


def test_old_game_does_not_rewrite_current_roster(conn):
    newer = make_box("0022600050", "2026-11-05", (NYK, "NYK", "New York", "Knicks", [player(7, "T", "T")]), _orl([player(10, "X", "Y")]))
    older = make_box("0022600001", "2026-10-21", _bos([player(7, "T", "T")]), _orl([player(10, "X", "Y")]))
    store_boxscore(conn, parse_boxscore(newer), "t")
    store_boxscore(conn, parse_boxscore(older), "t")
    cur = conn.execute("SELECT team_id FROM roster_stints WHERE player_id = 7 AND end_date IS NULL").fetchone()
    assert cur[0] == NYK


def test_schedule_never_downgrades_final(conn):
    store_boxscore(conn, parse_boxscore(real_box()), "test")
    data = json.loads((FIXTURES / "schedule_sample.json").read_text())
    games = parse_schedule(data)
    g = games[0]
    g.game_id, g.status = "0022000180", "scheduled"
    store_schedule(conn, [g], "test")
    assert conn.execute("SELECT status FROM games WHERE game_id='0022000180'").fetchone()[0] == "final"


def test_freshness_keeps_last_success_on_failure(conn):
    db.mark_attempt(conn, "schedule", True, "ok")
    first = conn.execute("SELECT last_success_at FROM data_freshness").fetchone()[0]
    db.mark_attempt(conn, "schedule", False, "boom")
    row = conn.execute("SELECT * FROM data_freshness").fetchone()
    assert row["status"] == "failed" and row["last_success_at"] == first


# -- full update job with a stand-in provider ---------------------------------

class FakeProvider:
    name = "fake"

    def __init__(self, schedule, boxes):
        self.schedule, self.boxes, self.requested = schedule, boxes, []

    def get_schedule(self):
        return self.schedule

    def get_boxscore(self, gid):
        self.requested.append(gid)
        return self.boxes.get(gid)

    def polite_pause(self):
        pass


def test_update_job_fetches_only_what_it_needs(conn, monkeypatch):
    from datetime import date
    from pipeline.ingest import update as upd
    from pipeline.providers.base import ScheduledGame, TeamRef

    monkeypatch.setattr(upd, "today_et", lambda: date(2026, 11, 20))
    bos, orl = TeamRef(BOS, "BOS", "Boston", "Celtics"), TeamRef(ORL, "ORL", "Orlando", "Magic")
    mk = lambda gid, day, status, tip: ScheduledGame(gid, "2026-27", day, tip, bos, orl, status,
                                                    None, None, None, None, "TD Garden", False)
    old_final = mk("0022600001", "2026-10-25", "final", "2026-10-25T23:30:00Z")
    recent = mk("0022600200", "2026-11-19", "final", "2026-11-19T23:30:00Z")
    tonight = mk("0022600210", "2026-11-20", "scheduled", "2099-01-01T00:00:00Z")
    started_but_unpublished = mk("0022600209", "2026-11-20", "scheduled", "2000-01-01T00:00:00Z")  # tip time already passed
    box = lambda gid, day: parse_boxscore(make_box(gid, day, _bos([player(1, "A", "B")]), _orl([player(10, "X", "Y")])))
    boxes = {"0022600001": box("0022600001", "2026-10-25"), "0022600200": box("0022600200", "2026-11-19")}
    fake = FakeProvider([old_final, recent, tonight, started_but_unpublished], boxes)

    first = upd.update_current_season(conn, fake)
    assert first["box_fetched"] == 2 and first["box_not_published"] == 1
    assert "0022600210" not in fake.requested           # future game: not requested

    fake.requested.clear()
    upd.update_current_season(conn, fake)
    # Old final is stored and outside the correction window, so it is skipped;
    # the recent one is re-checked for stat corrections.
    assert "0022600001" not in fake.requested and "0022600200" in fake.requested
    statuses = {r["dataset"]: r["status"] for r in conn.execute("SELECT * FROM data_freshness")}
    assert statuses == {"schedule": "ok", "box_scores": "ok", "rosters": "ok"}


def test_update_job_survives_feed_outage(conn):
    from pipeline.ingest.update import update_current_season

    class Down(FakeProvider):
        def get_schedule(self):
            raise FeedError("CDN unreachable")

    res = update_current_season(conn, Down([], {}))
    assert res["errors"] and res["box_fetched"] == 0
    assert conn.execute("SELECT status FROM data_freshness WHERE dataset='schedule'").fetchone()[0] == "failed"


def test_backfill_walks_game_ids_and_resumes(conn):
    from pipeline.ingest.update import backfill_regular_season
    boxes = {f"00225{n:05d}": parse_boxscore(make_box(f"00225{n:05d}", "2025-11-01", _bos([player(1, "A", "B")]), _orl([player(10, "X", "Y")])))
             for n in (1, 2, 3)}
    fake = FakeProvider([], boxes)
    res = backfill_regular_season(conn, fake, 2025, max_games=5)
    assert res["fetched"] == 3 and res["missing"] == ["0022500004", "0022500005"]
    res2 = backfill_regular_season(conn, fake, 2025, max_games=5)
    assert res2["skipped_existing"] == 3 and res2["fetched"] == 0
