"""Phase 2 tests: play-by-play, database upgrade, ratings math, minutes, injuries, lineups.

Real NBA play-by-play is used where it matters (fixtures/pbp_*.json are two
real 2025-26 games). Synthetic data is used only where the test needs a
known right answer to compare against.
"""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from pipeline import config, db
from pipeline.injuries.report import match_player, parse_pdf, report_urls_before, split_report_name
from pipeline.models import minutes as M
from pipeline.models import rapm
from pipeline.models.lineups import lineups
from pipeline.models.shrink import calibrate
from pipeline.pbp.parse import parse_actions
from pipeline.providers.archive import infer_home_away

from .helpers import FIXTURES

OKC, HOU = 1610612760, 1610612745


def load_game(gid):
    d = json.loads((FIXTURES / f"pbp_{gid}.json").read_text())
    return d["actions"], d["home"], d["away"]


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.sqlite")
    yield c
    c.close()


# -- play-by-play on real games -------------------------------------------------

def test_real_game_reconciles():
    # Opening night 2025-26: Thunder 125, Rockets 124 in double overtime.
    actions, home, away = load_game("0022500001")
    assert (home, away) == (OKC, HOU)
    g = parse_actions("0022500001", actions, home, away)
    assert g.final_score == (125, 124)
    assert g.periods == 6
    assert g.bad_events == 0 and all(s.ok for s in g.stints)
    pts = {t: sum(p["pts"] for p in g.players.values() if p["team_id"] == t) for t in (home, away)}
    assert (pts[home], pts[away]) == g.final_score
    length = 4 * 720 + 2 * 300
    for t in (home, away):
        secs = sum(p["seconds"] for p in g.players.values() if p["team_id"] == t)
        assert secs == pytest.approx(5 * length)
    starters = [p for p in g.players.values() if p["starter"]]
    assert len(starters) == 10
    # possessions: both teams within a few of each other, and of the box-score estimate
    assert abs(g.team_poss[home] - g.team_poss[away]) <= 4
    assert 100 < g.team_poss[home] < 135


def test_second_real_game_and_home_away_inference():
    actions, home, away = load_game("0022500002")
    assert infer_home_away(actions) == (home, away)
    g = parse_actions("0022500002", actions, home, away)
    pts = {t: sum(p["pts"] for p in g.players.values() if p["team_id"] == t) for t in (home, away)}
    assert (pts[home], pts[away]) == g.final_score
    # every stint's points add up to the final score
    assert sum(s.pts[home] for s in g.stints) == g.final_score[0]
    assert sum(s.pts[away] for s in g.stints) == g.final_score[1]


def test_lineup_error_is_flagged_not_hidden():
    actions, home, away = load_game("0022500002")
    # drop one substitution-in: the incoming player then acts while "off the floor"
    i = next(i for i, a in enumerate(actions) if a["actionType"] == "substitution" and a["subType"] == "in"
             and int(float(a["period"])) == 2)
    broken = actions[:i] + actions[i + 1:]
    g = parse_actions("0022500002", broken, home, away)
    assert g.bad_events > 0 or any(not s.ok for s in g.stints)


def test_store_archive_game_builds_box_scores(conn):
    from pipeline.ingest.pbp_store import store_archive_game
    actions, home, away = load_game("0022500001")
    g = parse_actions("0022500001", actions, home, away)
    store_archive_game(conn, g, actions, "archive")
    row = conn.execute("SELECT status, home_score, away_score, periods, game_date, season_id FROM games").fetchone()
    assert tuple(row) == ("final", 125, 124, 6, "2025-10-21", "2025-26")
    t = conn.execute("SELECT pts, possessions, poss_source, minutes FROM team_game_stats WHERE team_id = ?", (OKC,)).fetchone()
    assert t["pts"] == 125 and t["poss_source"] == "pbp" and t["minutes"] == pytest.approx(290)
    n_stints = conn.execute("SELECT COUNT(*) FROM lineup_stints").fetchone()[0]
    assert n_stints == 2 * len(g.stints)
    pm = conn.execute("SELECT SUM(plus_minus) FROM player_game_stats WHERE team_id = ?", (OKC,)).fetchone()[0]
    assert pm == 5 * (125 - 124)


# -- database upgrade ------------------------------------------------------------

def test_v1_database_is_upgraded(tmp_path):
    path = tmp_path / "old.sqlite"
    c = sqlite3.connect(path)
    c.executescript(config.SCHEMA_PATH.read_text())
    c.execute("PRAGMA user_version = 1")
    c.commit()
    c.close()
    conn = db.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"player_game_pbp", "pbp_games", "model_calibration", "injury_scenarios"} <= tables
    cols = {r[1] for r in conn.execute("PRAGMA table_info(team_game_stats)")}
    assert "poss_source" in cols
    stint_cols = {r[1] for r in conn.execute("PRAGMA table_info(lineup_stints)")}
    assert {"fga_rim", "opp_poss", "ok"} <= stint_cols


# -- shrinkage ----------------------------------------------------------------------

def _binomial_players(rng, n_players, true_sd, tries, mu=0.36, dates=("2026-01-01", "2026-03-01")):
    true = np.clip(rng.normal(mu, true_sd, n_players), 0.05, 0.95)
    rows = []
    for pid, p in enumerate(true):
        for d in dates:
            n = int(rng.integers(tries // 2, tries * 2))
            rows.append({"player_id": pid, "game_date": d, "made": rng.binomial(n, p), "tries": n})
    return pd.DataFrame(rows), true


def test_shrinkage_is_learned_and_beats_raw():
    rng = np.random.default_rng(0)
    cur, _ = _binomial_players(rng, 400, 0.03, 120)
    fit = calibrate(cur, None, "made", "tries", ["2026-02-01"])
    assert fit.improvement_vs_raw > 0.2
    # with true spread 0.03 around 0.36, the ideal prior strength is about p(1-p)/sd^2 ~ 250
    assert 80 <= fit.k <= 600
    est = fit.apply(30, 60)
    assert min(0.5, fit.league) < est < max(0.5, fit.league)


def test_pure_noise_stat_shrinks_hard():
    rng = np.random.default_rng(1)
    cur, _ = _binomial_players(rng, 400, 0.0, 120)
    fit = calibrate(cur, None, "made", "tries", ["2026-02-01"])
    assert fit.k >= 1200


# -- plus-minus ----------------------------------------------------------------------

def _synthetic_stints(rng, n_players=60, n_stints=20000, sd=3.0):
    o = rng.normal(0, sd, n_players)
    d = rng.normal(0, sd, n_players)
    rows = []
    for i in range(n_stints):
        players = rng.choice(n_players, 10, replace=False)
        off, dfn = players[:5], players[5:]
        poss = int(rng.integers(4, 14))
        mean = 1.14 + (o[off].sum() + d[dfn].sum()) / 100
        pts = int(rng.poisson(max(mean, 0.2) * poss))
        rows.append({"lineup_key": "-".join(str(p + 1) for p in sorted(off)),
                     "opp_lineup_key": "-".join(str(p + 1) for p in sorted(dfn)),
                     "poss": poss, "pts": pts, "is_home": int(rng.integers(0, 2))})
    return pd.DataFrame(rows), o, d


def test_rapm_recovers_known_player_effects():
    rng = np.random.default_rng(2)
    st, o, d = _synthetic_stints(rng)
    f = rapm.fit(st, np.ones(len(st)), lam=500)
    fr = f.as_frame().set_index("player_id").sort_index()
    assert np.corrcoef(fr["o_rapm"], o)[0, 1] > 0.8
    assert np.corrcoef(-fr["d_rapm"], d)[0, 1] > 0.8


def test_prior_pulls_low_minute_players():
    rng = np.random.default_rng(3)
    st, o, d = _synthetic_stints(rng, n_stints=300)
    prior = {p: 5.0 for p in range(1, 61)}
    f = rapm.fit(st, np.ones(len(st)), lam=100000, prior_off=prior)
    assert np.allclose(f.off, 5.0, atol=0.5)


# -- minutes ---------------------------------------------------------------------------

def test_scale_to_team_hits_total_and_caps():
    df = pd.DataFrame({"game_id": ["g"] * 9, "team_id": [1] * 9})
    pred = np.array([46, 44, 40, 30, 28, 22, 18, 10, 5], float)
    out = M.scale_to_team(df, pred, 241.0)
    assert out.sum() == pytest.approx(241.0)
    assert out.max() <= 48.0 + 1e-9
    assert (np.diff(out) <= 1e-9).all()      # order kept


def test_minutes_features_use_only_the_past():
    hist = pd.DataFrame([
        {"game_id": f"00225000{i:02d}", "game_date": f"2025-11-{i:02d}", "season_id": "2025-26", "team_id": 1,
         "player_id": 7, "min": float(i), "started": 1, "pf": 2, "home": 1, "opp_id": 2, "margin": 3}
        for i in range(1, 11)])
    fb = M.FeatureBuilder(hist)
    row = fb.rows([M.Target("0022500005", "2025-11-05", "2025-26", 1, 2, 1, [7])]).iloc[0]
    assert row["m_l1"] == 4 and row["m_l3"] == pytest.approx(3) and row["gp_season"] == 4
    assert "n_available" not in M.FEATURES     # would leak who got into the game


# -- injury report ----------------------------------------------------------------------

def test_injury_pdf_in_report_layout(tmp_path):
    from .make_injury_pdf import SAMPLE_P1, SAMPLE_P2, build
    path = tmp_path / "r.pdf"
    build(path, SAMPLE_P1, SAMPLE_P2)
    rep = parse_pdf(path.read_bytes())
    assert rep.ok and rep.report_time_et == "2026-10-21T17:30" and rep.pages == 2
    by = {(e.team, e.player): e for e in rep.entries}
    e = by[("Detroit Pistons", "Example, Charlie")]
    assert e.status == "doubtful" and e.matchup == "BOS@DET" and e.game_date == "2026-10-21"
    assert e.reason == "Injury/Illness - Left Posterior Tibialis Tendon; Surgery Rehabilitation"
    assert by[("Philadelphia 76ers", None)].status == "not_yet_submitted"
    assert by[("Los Angeles Lakers", "Tryout, Golf")].game_time == "10:00 (ET)"
    assert len(rep.entries) == 8


def test_report_urls_cover_both_naming_styles():
    from datetime import datetime
    from pipeline.injuries.report import ET
    urls = [u for _, u in report_urls_before(datetime(2026, 10, 21, 17, 40, tzinfo=ET), steps=4)]
    assert urls[0].endswith("Injury-Report_2026-10-21_05_30PM.pdf")
    assert any(u.endswith("Injury-Report_2026-10-21_05PM.pdf") for u in urls)


@pytest.mark.parametrize("printed,expected", [
    ("Porter Jr., Kevin", ("porter", "kevin")),
    ("Gilgeous-Alexander, Shai", ("gilgeous alexander", "shai")),
    ("Dončić, Luka", ("doncic", "luka")),
])
def test_report_name_split(printed, expected):
    assert split_report_name(printed) == expected


def test_name_matching():
    cands = {1: "Kevin Porter Jr.", 2: "S. Gilgeous-Alexander", 3: "Jalen Williams", 4: "Jaylin Williams",
             5: "Karl-Anthony Towns", 6: "J. Smith", 7: "J. Smith Jr."}
    assert match_player("Porter Jr., Kevin", cands)[0] == 1
    assert match_player("Gilgeous-Alexander, Shai", cands)[0] == 2
    assert match_player("Williams, Jalen", cands)[0] == 3
    assert match_player("Towns, Karl-Anthony", cands)[0] == 5
    assert match_player("Smith, John", cands)[0] is None       # two J. Smiths: refuse to guess
    assert match_player("Nobody, Real", cands)[0] is None


def test_play_rates_need_enough_listings(conn):
    from pipeline.injuries.ingest import MIN_RATE_SAMPLE, play_rates
    conn.execute("INSERT INTO seasons VALUES ('2025-26', '2025-10-21', '2026-04-12')")
    conn.execute("INSERT INTO teams (team_id, abbr, city, name) VALUES (1, 'AAA', 'A', 'A'), (2, 'BBB', 'B', 'B')")
    conn.execute("INSERT INTO injury_reports (report_time_et, source_url, fetched_at, parse_ok) VALUES ('2026-01-01T17:00', 'u', 'x', 1)")
    n = MIN_RATE_SAMPLE + 10
    for i in range(n):
        gid = f"00225{i:05d}"
        conn.execute("INSERT INTO games (game_id, season_id, season_type, game_date, tip_utc, home_team_id, away_team_id, status, source, fetched_at) "
                     "VALUES (?, '2025-26', 'regular', '2026-01-01', '2026-01-02T00:30:00Z', 1, 2, 'final', 't', 'x')", (gid,))
        conn.execute("INSERT OR IGNORE INTO players (player_id, full_name, fetched_at) VALUES (?, 'P', 'x')", (100 + i,))
        conn.execute("INSERT INTO injury_report_entries (report_id, game_id, team_id, player_id, status) VALUES (1, ?, 1, ?, 'questionable')",
                     (gid, 100 + i))
        if i % 2 == 0:
            conn.execute("INSERT INTO player_game_stats (game_id, player_id, team_id, seconds, source, fetched_at) VALUES (?, ?, 1, 900, 't', 'x')",
                         (gid, 100 + i))
    conn.commit()
    r = play_rates(conn)["questionable"]
    assert r["n"] == n and r["rate"] == pytest.approx(sum(1 for i in range(n) if i % 2 == 0) / n)


# -- lineups ---------------------------------------------------------------------------

def test_no_synergy_when_lineups_are_additive():
    rng = np.random.default_rng(4)
    st, o, d = _synthetic_stints(rng, n_players=20, n_stints=15000)
    net = {p + 1: o[p] + d[p] * -1 for p in range(20)}   # synthetic "true" ratings (d: + = allows more)
    rows = []
    for i, r in st.iterrows():
        for side in (0, 1):
            rows.append({"game_id": "g", "stint_num": i, "team_id": side + 1, "opp_poss": r["poss"],
                         "lineup_key": r["lineup_key"] if side == 0 else r["opp_lineup_key"],
                         "poss": r["poss"], "pts": r["pts"] if side == 0 else int(rng.poisson(1.14 * r["poss"])),
                         "opp_pts": 0, "start_sec": 0.0, "end_sec": 60.0})
    df = pd.DataFrame(rows)
    df["opp_pts"] = df.groupby("stint_num")["pts"].transform("sum") - df["pts"]
    out = lineups(df, net)
    assert out["tau"] < 2.0
