"""ESPN source: real responses for BOS at DET, 2026-10-20 (ESPN event 401909088).

fixtures/espn_401909088_box.json is the game summary's header and box score as
ESPN served them (unused fields dropped). fixtures/espn_401909088_plays.psv is
the play-by-play for the 1st quarter and the start of the 2nd, one play per
line, copied field for field.
"""
import json
from datetime import date

import pytest

from pipeline import db
from pipeline.ingest.store import store_boxscore, sync_rosters
from pipeline.pbp.parse import parse_actions
from pipeline.providers.base import TeamRef
from pipeline.providers.espn import (NEW_PLAYER_BASE, EspnProvider, PlayerIds, convert_plays, event_of,
                                     make_game_id, name_keys, parse_box, parse_scoreboard, shot_zone)

from .helpers import FIXTURES

BOS, DET = 1610612738, 1610612765
GID = "00226E401909088"


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.sqlite")
    yield c
    c.close()


def load_plays():
    plays = []
    for line in (FIXTURES / "espn_401909088_plays.psv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("|")
        seq, tid, tt, per, clock, team, parts, hs, as_, sc, sv, sh, pa, x, y = f[:15]
        p = {"sequenceNumber": seq, "type": {"id": tid, "text": tt.replace("\\n", "\n")},
             "text": "|".join(f[15:]).replace("\\n", "\n"), "period": {"number": int(per)},
             "clock": {"displayValue": clock}, "homeScore": int(hs), "awayScore": int(as_),
             "scoringPlay": sc == "1", "scoreValue": int(sv), "shootingPlay": sh == "1",
             "pointsAttempted": int(pa), "coordinate": {"x": int(x), "y": int(y)}}
        if team:
            p["team"] = {"id": team}
        if parts:
            p["participants"] = [{"athlete": {"id": a}} for a in parts.split(",")]
        plays.append(p)
    return plays


def summary():
    s = json.loads((FIXTURES / "espn_401909088_box.json").read_text())
    s["plays"] = load_plays()
    return s


def add_archive_player(conn, pid, name, team, day="2026-04-10"):
    """A player as the archive stores him: short name, last game for `team`."""
    gid = f"00225{pid % 100000:05d}"
    conn.execute("INSERT OR IGNORE INTO teams (team_id, abbr, city, name) VALUES (?, ?, '', '')", (team, str(team)))
    conn.execute("INSERT OR IGNORE INTO teams (team_id, abbr, city, name) VALUES (1, 'Y', '', '')")
    conn.execute("INSERT OR IGNORE INTO seasons (season_id, start_date, end_date) VALUES ('2025-26', ?, ?)",
                 (day, day))
    conn.execute("""INSERT OR IGNORE INTO games (game_id, season_id, season_type, game_date, home_team_id,
                    away_team_id, status, source, fetched_at) VALUES (?, '2025-26', 'regular', ?, ?, 1, 'final',
                    'archive', 'x')""", (gid, day, team))
    conn.execute("INSERT INTO players (player_id, full_name, fetched_at) VALUES (?, ?, 'x')", (pid, name))
    conn.execute("""INSERT INTO player_game_stats (game_id, player_id, team_id, started, seconds, source, fetched_at)
                    VALUES (?, ?, ?, 0, 600, 'archive', 'x')""", (gid, pid, team))


# -- IDs and schedule --------------------------------------------------------------------

def test_game_ids_keep_nba_prefix_and_season():
    from pipeline.ingest.store import season_type
    from pipeline.util import season_from_game_id
    gid = make_game_id(2, 2027, "401909088")
    assert gid == GID and season_from_game_id(gid) == "2026-27" and season_type(gid) == "regular"
    assert make_game_id(1, 2027, "1")[:3] == "001" and make_game_id(3, 2027, "1")[:3] == "004"
    assert make_game_id(4, 2027, "1") is None            # All-Star: skipped
    assert event_of(gid) == "401909088"


def test_scoreboard_event_becomes_scheduled_game():
    event = {"id": "401909088", "date": "2026-10-20T19:00Z", "season": {"year": 2027, "type": 2},
             "competitions": [{"date": "2026-10-20T19:00Z", "neutralSite": False,
                               "venue": {"fullName": "Little Caesars Arena"},
                               "status": {"type": {"name": "STATUS_FINAL", "state": "post", "completed": True,
                                                   "shortDetail": "Final"}},
                               "competitors": [
                                   {"homeAway": "home", "id": "8", "score": "121",
                                    "records": [{"summary": "1-0", "type": "total"}]},
                                   {"homeAway": "away", "id": "2", "score": "113",
                                    "records": [{"summary": "0-1", "type": "total"}]}]}]}
    other_season = dict(event, id="1", season={"year": 2026, "type": 2})
    all_star = dict(event, id="2", season={"year": 2027, "type": 4})
    games = parse_scoreboard({"events": [event, other_season, all_star]}, 2027)
    assert len(games) == 1
    g = games[0]
    assert (g.game_id, g.season_id, g.game_date, g.tip_utc) == (GID, "2026-27", "2026-10-20", "2026-10-20T19:00:00Z")
    assert (g.home.team_id, g.home.tricode, g.home.city, g.away.tricode) == (DET, "DET", "Detroit", "BOS")
    assert (g.status, g.home_score, g.away_score, g.home_record) == ("final", 121, 113, "1-0")


def test_late_game_gets_eastern_date():
    event = {"id": "9", "date": "2026-10-23T02:30Z", "season": {"year": 2027, "type": 2},
             "competitions": [{"date": "2026-10-23T02:30Z", "status": {"type": {"state": "pre"}},
                               "competitors": [{"homeAway": "home", "id": "13"}, {"homeAway": "away", "id": "9"}]}]}
    g = parse_scoreboard({"events": [event]}, 2027)[0]
    assert g.game_date == "2026-10-22" and g.status == "scheduled" and g.home_score is None
    assert g.home.tricode == "LAL" and g.away.tricode == "GSW"


def test_name_keys_cover_short_and_display_names():
    assert "p george" in name_keys("Paul George", "P. George")
    assert "l doncic" in name_keys("Luka Dončić", "")
    assert "j jackson" in name_keys("Jaren Jackson Jr.", "J. Jackson Jr.")


def test_player_matching(conn):
    add_archive_player(conn, 202331, "P. George", 1610612755)      # traded since: name alone decides
    add_archive_player(conn, 1629011, "M. Robinson", 1610612752)
    add_archive_player(conn, 1641718, "K. George", 1610612762)      # two K. Georges: team decides
    add_archive_player(conn, 1642273, "K. George", DET)
    ids = PlayerIds(conn)
    assert ids.get("4251", "Paul George", "P. George", BOS) == 202331
    assert ids.get("999", "Kyle George", "K. George", DET) == 1642273
    assert ids.get("5175647", "Hugo Gonzalez", "H. Gonzalez", BOS) == NEW_PLAYER_BASE + 5175647
    # stored: a second lookup (or a fresh run) gives the same answer without matching again
    assert PlayerIds(conn).get("4251", "", "", 0) == 202331
    methods = dict(conn.execute("SELECT provider_key, match_method FROM id_map WHERE provider = 'espn'").fetchall())
    assert methods["4251"] == "name" and methods["999"] == "name+team" and methods["5175647"].startswith("new")


def test_roster_matching_resolves_name_clashes(conn):
    add_archive_player(conn, 1631114, "J. Williams", 1610612760)
    ids = PlayerIds(conn)
    out = ids.assign_all([("1", "Jalen Williams", "J. Williams", 1610612760),
                          ("2", "Jaylin Williams", "J. Williams", 1610612745)])
    assert out["1"] == 1631114 and out["2"] == NEW_PLAYER_BASE + 2


# -- box score and play-by-play ------------------------------------------------------------

def test_parse_real_box_score(conn):
    box, athletes = parse_box(summary(), PlayerIds(conn))
    assert (box.game_id, box.season_id, box.game_date, box.status) == (GID, "2026-27", "2026-10-20", "final")
    assert box.arena == "Little Caesars Arena" and not box.full_roster
    h, a = box.home, box.away
    assert (h.team.team_id, a.team.team_id, h.pts, a.pts) == (DET, BOS, 121, 113)
    # team totals match ESPN's own team line
    assert (h.fgm, h.fga, h.fg3m, h.fg3a, h.ftm, h.fta) == (39, 86, 15, 40, 28, 32)
    assert (a.fgm, a.fga, a.fg3m, a.fg3a, a.ftm, a.fta) == (40, 88, 17, 42, 16, 21)
    assert (h.oreb, h.dreb, h.ast, h.stl, h.blk, h.tov) == (14, 35, 27, 7, 5, 17)
    assert (a.tov, a.pf, h.pts_paint, h.pts_fast_break, h.pts_off_tov) == (14, 26, 44, 17, 21)
    pg = next(p for p in a.players if p.full_name == "Paul George")
    assert (pg.seconds, pg.pts, pg.fgm, pg.fga, pg.plus_minus, pg.started) == (31 * 60, 38, 12, 21, 9, True)
    conley = next(p for p in a.players if p.full_name == "Mike Conley")
    assert conley.seconds == 0 and conley.dnp_reason.startswith("DNP") and conley.plus_minus is None
    assert len(athletes) == 28 and athletes["4251"][1] == BOS


def test_play_by_play_through_existing_parser(conn):
    s = summary()
    box, athletes = parse_box(s, PlayerIds(conn))
    actions = convert_plays(s, athletes)
    starters = {side.team.team_id: [p.player_id for p in side.players if p.started] for side in (box.home, box.away)}
    g = parse_actions(GID, actions, DET, BOS, starters)
    assert g.bad_events == 0 and all(st.ok for st in g.stints)       # every event fits the lineups
    pid = {v: athletes[k][0] for k, v in {"4065648": "tatum", "4432166": "cade", "4351852": "mitch",
                                         "4395702": "joe", "4683689": "walsh"}.items()}
    # end of the 1st quarter: DET 27, BOS 28
    q1 = [st for st in g.stints if st.period == 1]
    assert sum(st.pts[DET] for st in q1) == 27 and sum(st.pts[BOS] for st in q1) == 28
    assert g.final_score == (32, 33)                                    # last play in the excerpt
    t = g.players[pid["tatum"]]
    # Tatum, Q1 + 1:47 of Q2: 3-9 FG, 0-2 3PT, 1-1 FT (an and-one), 4 assists, all 12 Q1 minutes
    assert (t["pts"], t["fgm"], t["fga"], t["fg3a"], t["ftm"], t["fta"], t["ast"], t["and1"]) == (7, 3, 9, 2, 1, 1, 4, 1)
    assert t["fga_rim"] == 4 and t["fga_mid"] == 2 and t["fga_ab3"] == 2
    assert g.players[pid["mitch"]]["blk"] == 1 and g.players[pid["mitch"]]["tov"] == 2
    assert g.players[pid["joe"]]["blk"] == 1 and g.players[pid["walsh"]]["blk"] == 1
    assert g.players[pid["cade"]]["stl"] == 1 and g.players[pid["cade"]]["tov"] == 2
    # starters from the box score; Walsh (starter) left at 6:14 of the 1st
    assert g.players[pid["walsh"]]["starter"] and round(g.players[pid["walsh"]]["seconds"]) == 346
    # possessions alternate: the two teams can differ by at most one per period
    assert abs(g.team_poss[DET] - g.team_poss[BOS]) <= 2 and 25 <= g.team_poss[DET] <= 33


def test_shot_zones():
    assert shot_zone(2, (29.0, 0.0), "Paul Reed makes 4-foot reverse layup", "Reverse Layup Shot") == "rim"
    assert shot_zone(2, (21.0, 1.0), "Jayson Tatum makes layup", "Layup Shot") == "rim"
    assert shot_zone(2, (20.0, 7.0), "Jayson Tatum makes 8-foot pullup jump shot", "Pullup Jump Shot") == "paint"
    assert shot_zone(2, (20.0, 14.0), "misses 14-foot pullup jump shot", "Pullup Jump Shot") == "mid"
    assert shot_zone(3, (49.0, 6.0), "makes 24-foot three point jumper", "Jump Shot") == "c3"
    assert shot_zone(3, (23.0, 26.0), "misses 25-foot three point jumper", "Jump Shot") == "ab3"
    assert shot_zone(3, None, "misses running pullup jump shot", "") is None


def test_espn_box_does_not_close_injured_players_stints(conn):
    box, _ = parse_box(summary(), PlayerIds(conn))
    injured = NEW_PLAYER_BASE + 4433188           # on BOS's roster, not in the box score
    sync_rosters(conn, [(TeamRef(BOS, "BOS", "Boston", "Celtics"), [
        {"player_id": injured, "full_name": "Devin Carter", "jersey": "99"}])], "espn", "2026-10-19")
    store_boxscore(conn, box, "espn")
    open_team = conn.execute("SELECT team_id FROM roster_stints WHERE player_id = ? AND end_date IS NULL",
                             (injured,)).fetchone()
    assert open_team and open_team[0] == BOS


def test_roster_sync_moves_and_releases(conn):
    bos, det = TeamRef(BOS, "BOS", "Boston", "Celtics"), TeamRef(DET, "DET", "Detroit", "Pistons")
    sync_rosters(conn, [(bos, [{"player_id": 1, "full_name": "A B"}, {"player_id": 2, "full_name": "C D"}])],
                 "espn", "2026-10-01")
    res = sync_rosters(conn, [(bos, [{"player_id": 1, "full_name": "A B"}]),
                              (det, [{"player_id": 3, "full_name": "E F", "birth_date": "2001-02-03"}])],
                       "espn", "2026-10-10")
    assert res == {"players": 2, "added": 1, "moved": 0, "removed": 1}
    res = sync_rosters(conn, [(det, [{"player_id": 1, "full_name": "A B"}, {"player_id": 3, "full_name": "E F"}])],
                       "espn", "2026-10-15")
    assert res["moved"] == 1
    rows = conn.execute("SELECT player_id, team_id, start_date, end_date FROM roster_stints ORDER BY 1, 3").fetchall()
    assert [tuple(r) for r in rows] == [(1, BOS, "2026-10-01", "2026-10-15"), (1, DET, "2026-10-15", None),
                                        (2, BOS, "2026-10-01", "2026-10-10"), (3, DET, "2026-10-10", None)]
    assert conn.execute("SELECT birth_date FROM players WHERE player_id = 3").fetchone()[0] == "2001-02-03"


# -- the whole update job, with ESPN answering from the fixtures -----------------------------

class FakeEspn(EspnProvider):
    pbp_valid_since = ""
    def __init__(self, conn):
        from datetime import datetime
        super().__init__(conn, now=datetime(2026, 10, 22, 12))
        self.calls = []

    def _get_json(self, url):
        self.calls.append(url)
        if "scoreboard" in url:
            if "202610" not in url:
                return {"events": []}
            s = summary()["header"]
            c = s["competitions"][0]
            return {"events": [{"id": s["id"], "date": c["date"], "season": s["season"], "competitions": [c]}]}
        if "summary" in url:
            return summary()
        if "roster" in url:
            return {"athletes": []}
        raise AssertionError(url)

    def polite_pause(self):
        pass


def test_update_job_runs_on_espn(conn, monkeypatch):
    from pipeline.ingest import update as upd
    monkeypatch.setattr(upd, "today_et", lambda: date(2026, 10, 22))
    res = upd.update_current_season(conn, FakeEspn(conn))
    assert res["errors"] == [] and res["box_fetched"] == 1 and res["pbp_fetched"] == 1
    assert conn.execute("SELECT status, home_score, source FROM games WHERE game_id = ?", (GID,)).fetchone()[:] == \
        ("final", 121, "espn")
    # the excerpt stops early in the 2nd quarter, so the totals check flags it instead of trusting it
    notes = conn.execute("SELECT notes, bad_stints, stints FROM pbp_games WHERE game_id = ?", (GID,)).fetchone()
    assert "box score 121" in notes[0] and notes[1] == notes[2]
    # minutes stay as ESPN's whole minutes when the play-by-play is not trusted
    secs = conn.execute("SELECT seconds FROM player_game_stats WHERE game_id = ? AND seconds > 0", (GID,)).fetchall()
    assert all(r[0] % 60 == 0 for r in secs)
    # team names now come through for the dashboard
    assert conn.execute("SELECT city, name FROM teams WHERE team_id = ?", (DET,)).fetchone()[:] == ("Detroit", "Pistons")
    status = {r[0]: r[1] for r in conn.execute("SELECT dataset, status FROM data_freshness")}
    assert status["rosters"] == "ok" and status["play_by_play"] == "ok"


def test_pick_provider_sticks_with_espn_for_the_season(conn):
    from pipeline.run import pick_provider
    from pipeline.providers.nba_cdn import NbaCdnProvider

    class Blocked(NbaCdnProvider):
        def get_schedule(self):
            raise RuntimeError("HTTP 403")

    import pipeline.run as run
    orig = run.NbaCdnProvider
    run.NbaCdnProvider = Blocked
    try:
        assert pick_provider(conn, log=lambda *_: None).name == "espn"
        assert conn.execute("SELECT status FROM data_freshness WHERE dataset = 'data_source'").fetchone()[0] == "ok"
    finally:
        run.NbaCdnProvider = orig


def test_missed_heave_is_not_a_shot_attempt(conn):
    s = summary()
    heave = {"type": {"id": "282", "text": "Heave Jump Shot"}, "text": "Paul George misses heave jump shot",
             "period": {"number": 2}, "clock": {"displayValue": "0.4"}, "scoringPlay": False, "scoreValue": 0,
             "shootingPlay": True, "pointsAttempted": 3, "team": {"id": "2"},
             "participants": [{"athlete": {"id": "4251"}}], "coordinate": {"x": 25, "y": 60}}
    s["plays"].append(heave)
    box, athletes = parse_box(s, PlayerIds(conn))
    acts = convert_plays(s, athletes)
    assert acts[-1]["actionType"] == "heave"
    made = dict(heave, scoringPlay=True, scoreValue=3, text="Paul George makes heave jump shot")
    s["plays"][-1] = made
    assert convert_plays(s, athletes)[-1]["actionType"] == "3pt"
