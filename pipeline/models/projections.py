"""Game-day rotations: who is available, projected minutes, starters and injury scenarios.

Steps for each team in each upcoming game:
  1. Roster: players listed in the team's latest box score (confirmed), or,
     before a team has played this season, last season's players who have
     not appeared for anyone else since (estimated).
  2. Injury report: Out players are removed. Questionable and Doubtful
     players are uncertain and get scenarios. Everyone else is available.
  3. How often each available player actually gets into games (from
     box-score "coach's decision" records), and his minutes if he plays,
     from the minutes model. Expected minutes are scaled to the team total.
  4. One scenario per combination of uncertain players sitting (up to 4
     players, 16 scenarios). Each scenario is projected separately, so
     their minutes flow to the right teammates. When enough past reports
     exist, each scenario gets a probability from how often players with
     that status really played; until then probabilities are left blank.
"""
import itertools
import json
import logging

import numpy as np
import pandas as pd

from ..util import utc_now_iso
from . import minutes as M
from .ratings import model_season, prev_season

log = logging.getLogger(__name__)


def _np(o):
    """JSON fallback for NumPy numbers."""
    return o.item() if hasattr(o, "item") else str(o)
MAX_UNCERTAIN = 4


# -- minutes model: train + calibrate --------------------------------------------

def _hist_seasons(season):
    """Three seasons of history: two to train on, and one more so early-season games have a past."""
    p1 = prev_season(season)
    return [prev_season(p1), p1, season]


def _live_like(fb, hist_games, team_games, test_targets, model):
    """Replay games the way the live pipeline sees them, without knowing who will get in.

    For each team-game: the roster is everyone who played for the team in its last
    10 games. Regulars (10+ minutes a game) who sat are treated as listed Out, the way
    the injury report would show them; everyone else is available, including bench
    players who then did not get in. Appearance odds come from those 10 games.
    Returns expected minutes (after scaling to the team total) and actual minutes
    for every available player, zeros included.
    """
    exp_all, act_all, cond_all, played_all, base_all = [], [], [], [], []
    for tg in test_targets:
        prev_ids = team_games.get(tg.team_id)
        if prev_ids is None:
            continue
        k = int(np.searchsorted(prev_ids["day"], pd.Timestamp(tg.game_date).toordinal()))
        last10 = prev_ids["gid"][max(0, k - 10):k]
        if len(last10) < 5:
            continue
        apps = {}
        for gid in last10:
            for p, m in fb.appear.get((tg.team_id, gid), {}).items():
                apps.setdefault(p, []).append(m)
        played_now = fb.appear.get((tg.team_id, tg.game_id), {})
        roster = set(apps) | set(played_now)
        out, odds = set(), {}
        for p in roster:
            mins = apps.get(p, [])
            regular = len(mins) >= 2 and np.mean(mins) >= 10
            if regular and p not in played_now:
                out.add(p)
                continue
            n_app = len(mins)
            n_healthy = len(last10) if not regular else n_app + 0   # regulars' absences were injuries
            odds[p] = (n_app + 1) / (max(n_healthy, n_app) + 2)
            # baseline: his average over the last 10 team games he was healthy for, zeros included
            odds[p] = (odds[p], sum(mins) / max(n_healthy, n_app, 1))
        avail = sorted(p for p in roster if p not in out)
        if not avail:
            continue
        t = M.Target(tg.game_id, tg.game_date, tg.season_id, tg.team_id, tg.opp_id, tg.home, avail)
        rows = fb.rows([t])
        cond = model.predict_raw(M.fill(rows))
        pv = np.array([odds[p][0] for p in rows["player_id"]])
        base_all.append(np.array([odds[p][1] for p in rows["player_id"]]))
        exp = M.scale_to_team(rows, cond * pv, model.total)
        act = np.array([played_now.get(p, 0.0) for p in rows["player_id"]])
        exp_all.append(exp), act_all.append(act), cond_all.append(cond)
        played_all.append(np.array([p in played_now for p in rows["player_id"]]))
    if not exp_all:
        return None
    return tuple(np.concatenate(x) for x in (exp_all, act_all, cond_all, played_all, base_all))


def minutes_calibration(conn, season_m: str) -> dict:
    """Walk-forward backtests on a full season: the error bands and the test results the dashboard shows.

    Two tests, both using only games before each test window to train:
      * minutes when a player gets in (the model's main job)
      * the full game-day pipeline, with no knowledge of who gets in
    """
    key = f"minutes-v2:{season_m}"
    row = conn.execute("SELECT payload FROM model_calibration WHERE key = ?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    prev = prev_season(season_m)
    hist = M.history(conn, _hist_seasons(season_m))
    fb = M.FeatureBuilder(hist)
    has_prev = (hist["season_id"] == prev).any()
    prev_rows = fb.rows(M.backtest_targets(hist, prev)) if has_prev else pd.DataFrame()
    cur_targets = M.backtest_targets(hist, season_m)
    cur_rows = fb.rows(cur_targets)
    mins = {(r.player_id, r.game_id): r.min for r in hist.itertuples()}

    def y(d):
        return np.array([mins[(p, g)] for p, g in zip(d.player_id, d.game_id)])

    total = M.team_total_target(hist[hist["season_id"] == season_m])
    dates = sorted(cur_rows["game_date"].unique())
    edges = [dates[int(len(dates) * f)] for f in (0.0, 0.2, 0.4, 0.6, 0.8)] + ["9999"]
    windows, P, Y, B, L5 = [], [], [], [], []
    LE, LA, LB = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        train = pd.concat([prev_rows, cur_rows[cur_rows["game_date"] < a]]) if has_prev else cur_rows[cur_rows["game_date"] < a]
        test = cur_rows[(cur_rows["game_date"] >= a) & (cur_rows["game_date"] < b)]
        if len(train) < 2000 or test.empty:
            continue
        m = M.MinutesModel().fit(M.fill(train), y(train), total)
        # Scored on the raw prediction: scaling in this test would use the set of players
        # who actually got in, which is not known before the game.
        p, yy = m.predict_raw(M.fill(test)), y(test)
        base = M.baseline(test)
        l5 = test["m_l5"].fillna(pd.Series(base, index=test.index)).to_numpy()
        win = {"from": a, "to": b if b != "9999" else dates[-1], "rows": int(len(test)),
               "model": float(np.mean(np.abs(yy - p))), "season_avg": float(np.mean(np.abs(yy - base))),
               "last5": float(np.mean(np.abs(yy - l5)))}
        live = _live_like(fb, None, fb.teams, [t for t in cur_targets if a <= t.game_date < b][::3], m)
        if live:
            e, act, cond, played, lb = live
            win["live_model"] = float(np.mean(np.abs(act - e)))
            win["live_baseline"] = float(np.mean(np.abs(act - lb)))
            LE.append(e), LA.append(act), LB.append(lb)
        windows.append(win)
        P.append(p), Y.append(yy), B.append(base), L5.append(l5)
    if not P:
        return {"season": season_m, "intervals": None, "backtest": [], "note": "not enough data"}
    P, Y, B, L5 = map(np.concatenate, (P, Y, B, L5))
    half = len(P) // 2
    tmp = M.MinutesModel()
    tmp.set_intervals(P, Y)
    check = M.MinutesModel()
    check.set_intervals(P[:half], Y[:half])
    lo, hi = check.interval(P[half:])
    live = None
    if LE:
        LE, LA, LB = np.concatenate(LE), np.concatenate(LA), np.concatenate(LB)
        live = {"model": float(np.mean(np.abs(LA - LE))), "last10_avg": float(np.mean(np.abs(LA - LB))),
                "rows": int(len(LE)), "bias": float(np.mean(LA - LE))}
    early = _early_season_test(hist, fb, prev_rows, season_m, y, total) if has_prev else None
    out = {
        "season": season_m, "total_minutes": total, "intervals": tmp.resid_q,
        "early_season": early,
        "backtest": windows,
        "overall": {"model": float(np.mean(np.abs(Y - P))), "season_avg": float(np.mean(np.abs(Y - B))),
                    "last5": float(np.mean(np.abs(Y - L5))), "rows": int(len(P)),
                    "band_coverage": float(np.mean((Y[half:] >= lo) & (Y[half:] <= hi))),
                    "bias": float(np.mean(Y - P))},
        "live_test": live,
        "assumption": ("The first test scores minutes for players who got in. The game-day test replays "
                       "games without knowing who would get in: regulars who sat stand in for players "
                       "the injury report lists as out."),
        "created_at": utc_now_iso(),
    }
    conn.execute("INSERT OR REPLACE INTO model_calibration (key, payload, created_at) VALUES (?, ?, ?)",
                 (key, json.dumps(out, default=_np), out["created_at"]))
    conn.commit()
    return out


def _early_season_test(hist, fb, prev_rows, season_m, y, total):
    """First month of season_m replayed as game days, with a model trained only on earlier seasons."""
    from . import minutes_eval as E
    m = M.MinutesModel().fit(M.fill(prev_rows), y(prev_rows), total)
    recs = E.early_games(hist, season_m)
    if not recs:
        return None
    model_exp, base_exp = [], []
    for r in recs:
        rows = fb.rows([r.target])
        p_in = np.array([r.odds[q] for q in rows["player_id"]])
        model_exp.append(dict(zip(rows["player_id"], M.scale_to_team(rows, m.predict_raw(M.fill(rows)) * p_in, total))))
        base_exp.append(dict(zip(rows["player_id"], M.scale_to_team(rows, M.baseline(rows) * p_in, total))))
    return {"days": E.EARLY_DAYS, "model": E.score(recs, model_exp), "recent_average": E.score(recs, base_exp)}


def train_minutes_model(conn, season: str):
    seasons = _hist_seasons(season)
    hist = M.history(conn, seasons)
    fb = M.FeatureBuilder(hist)
    rows = []
    for s in seasons[1:]:
        if (hist["season_id"] == s).any():
            rows.append(fb.rows(M.backtest_targets(hist, s)))
    if not rows:
        return None, fb, hist
    train = pd.concat(rows)
    mins = {(r.player_id, r.game_id): r.min for r in hist.itertuples()}
    yv = np.array([mins[(p, g)] for p, g in zip(train.player_id, train.game_id)])
    model = M.MinutesModel().fit(M.fill(train), yv, M.team_total_target(hist[hist["season_id"].isin(seasons[1:])]))
    return model, fb, hist


# -- availability ------------------------------------------------------------------

def _is_coach_dnp(reason: str) -> bool:
    r = (reason or "").lower()
    return "coach" in r


PREV_GAME_WEIGHT = 0.5
PREV_GAMES_CAP = 6
APPEAR_DEFAULT = {"m0": 0.5, "s": 2.0, "w": PREV_GAME_WEIGHT, "cap": PREV_GAMES_CAP,
                  "source": "default (not enough of this season's game-day rosters yet)"}
MIN_APPEAR_ROWS = 300


def _this_season_record(rows):
    """(played, eligible) from a player's box-score rows, newest first: healthy games only, last 10."""
    played = eligible = 0
    for secs, reason in rows:
        if secs and secs > 0:
            played, eligible = played + 1, eligible + 1
        elif _is_coach_dnp(reason):
            eligible += 1
        if eligible >= 10:
            break
    return played, eligible


def _prev_record(conn, p, team_id, prev, cache):
    key = (p, team_id)
    if key not in cache:
        pp, pe, src = _last_season_odds(conn, p, team_id, prev)
        if pe == 0:
            pp, pe, src = _last_season_any_team(conn, p, prev)
        cache[key] = (pp, pe, src)
    return cache[key]


def _p_hat(k, n, pp, pe, prm):
    w = min(prm["w"], prm["cap"] / pe) if pe else 0.0
    return (k + w * pp + prm["s"] * prm["m0"]) / (n + w * pe + prm["s"])


def fit_appearance(conn, season: str) -> dict:
    """Learn the chance-of-getting-in formula from this season's game-day rosters.

    Every healthy, dressed player-game this season (box-score rows: played, or
    'coach's decision') is predicted from the games before it, exactly as the
    live code would have: the starting guess (m0, worth s games), last season's
    weight per game (w) and its cap in games (cap) are picked to make those
    predictions as likely as possible.
    """
    prev = prev_season(season)
    rows = conn.execute(
        """SELECT p.player_id, p.team_id, p.seconds, p.dnp_reason, g.game_date, g.game_id FROM player_game_stats p
           JOIN games g ON g.game_id = p.game_id WHERE g.season_id = ? AND g.status = 'final'
             AND g.season_type = 'regular' ORDER BY g.game_date, g.game_id""", (season,)).fetchall()
    hist, obs, cache = {}, [], {}
    for pid, tid, secs, reason, _, _ in rows:
        played_now = bool(secs and secs > 0)
        if not played_now and not _is_coach_dnp(reason):
            continue
        before = hist.setdefault((pid, tid), [])
        k, n = _this_season_record(list(reversed(before)))
        pp, pe, _ = _prev_record(conn, pid, tid, prev, cache)
        obs.append((k, n, pp, pe, played_now))
        before.append((secs, reason))
    if len(obs) < MIN_APPEAR_ROWS:
        return {**APPEAR_DEFAULT, "rows": len(obs)}
    o = np.array(obs, dtype=float)

    def loglik(prm):
        w = np.where(o[:, 3] > 0, np.minimum(prm["w"], prm["cap"] / np.maximum(o[:, 3], 1)), 0.0)
        p = (o[:, 0] + w * o[:, 2] + prm["s"] * prm["m0"]) / (o[:, 1] + w * o[:, 3] + prm["s"])
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return float(np.mean(o[:, 4] * np.log(p) + (1 - o[:, 4]) * np.log(1 - p)))

    best = None
    for m0 in (0.4, 0.5, 0.6, 0.7, 0.8):
        for s_ in (1.0, 2.0, 4.0, 8.0):
            for w in (0.1, 0.25, 0.5, 1.0):
                for cap in (2.0, 4.0, 6.0, 12.0, 25.0):
                    prm = {"m0": m0, "s": s_, "w": w, "cap": cap}
                    ll = loglik(prm)
                    if best is None or ll > best[0]:
                        best = (ll, prm)
    return {**best[1], "rows": len(obs), "log_loss": -best[0], "log_loss_default": -loglik(APPEAR_DEFAULT),
            "played_share": float(o[:, 4].mean()),
            "source": f"learned from {len(obs):,} healthy player-games this season"}


def appearance_odds(conn, team_id: int, season: str, players, params: dict = None, cache: dict = None) -> dict:
    """How often each player gets into games when he is healthy.

    This season: his last 10 healthy games with this team, from box scores
    ('coach's decision' counts as not getting in; inactive, injured or G
    League games are left out). With no record this season: last season with
    this team, leaving out games he was on that day's injury report, and
    absences of 3 or more games in a row when no report is stored (almost
    always injuries). A 1-in-2 starting guess (one game each way) keeps tiny
    samples from reading as certainties.
    """
    out = {}
    prev = prev_season(season)
    prm = params or APPEAR_DEFAULT
    cache = {} if cache is None else cache
    for p in players:
        rows = conn.execute(
            """SELECT p.seconds, p.dnp_reason FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
               WHERE p.player_id = ? AND p.team_id = ? AND g.season_id = ? AND g.status = 'final'
                 AND g.season_type = 'regular' ORDER BY g.game_date DESC LIMIT 25""",
            (p, team_id, season)).fetchall()
        played, eligible = _this_season_record(rows)
        # Last season's record is the prior, each game worth w of a game this season up to `cap`
        # games' worth, on top of a starting guess; all learned from this season's rosters.
        pp, pe, psrc = _prev_record(conn, p, team_id, prev, cache)
        p_hat = _p_hat(played, eligible, pp, pe, prm)
        source = "this season" + (f" + {psrc}" if pe else "") if eligible else (psrc if pe else "no record")
        out[p] = {"p": p_hat, "played": played, "eligible": eligible, "prev_played": pp, "prev_eligible": pe,
                  "source": source}
    return out


def _last_season_any_team(conn, player_id, prev):
    """Last season's record across every team he played for (traded or signed since)."""
    teams = [r[0] for r in conn.execute(
        """SELECT DISTINCT p.team_id FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
           WHERE p.player_id = ? AND g.season_id = ? AND p.seconds > 0""", (player_id, prev))]
    played = eligible = 0
    src = "no record"
    for t in teams:
        a, b, src = _last_season_odds(conn, player_id, t, prev)
        played, eligible = played + a, eligible + b
    return played, eligible, (src.replace("last season", "last season, other team") if teams else src)


def _last_season_odds(conn, player_id, team_id, prev):
    apps = {r[0] for r in conn.execute(
        """SELECT g.game_id FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
           WHERE p.player_id = ? AND p.team_id = ? AND g.season_id = ? AND p.seconds > 0""",
        (player_id, team_id, prev))}
    if not apps:
        return 0, 0, "no record"
    team_games = [r[0] for r in conn.execute(
        """SELECT game_id FROM games WHERE season_id = ? AND status = 'final' AND season_type = 'regular'
           AND (home_team_id = ? OR away_team_id = ?) ORDER BY game_date, game_id""", (prev, team_id, team_id))]
    idx = [i for i, g in enumerate(team_games) if g in apps]
    span = team_games[idx[0]:idx[-1] + 1]
    listed = {r[0] for r in conn.execute(
        "SELECT DISTINCT game_id FROM injury_report_entries WHERE player_id = ? AND game_id IS NOT NULL",
        (player_id,))}
    reported_days = {r[0] for r in conn.execute(
        """SELECT DISTINCT e.game_id FROM injury_report_entries e WHERE e.team_id = ? AND e.game_id IS NOT NULL""",
        (team_id,))}
    eligible, run = 0, []

    def flush(run):
        # a run of missed games: counted as "did not get in" only if short and not explained
        n = 0
        for g in run:
            if g in listed:
                continue
            if g not in reported_days and len(run) >= 3:
                continue
            n += 1
        return n

    for g in span:
        if g in apps:
            eligible += 1 + flush(run)
            run = []
        else:
            run.append(g)
    eligible += flush(run)
    source = "last season" + (" (injury reports used)" if reported_days else " (long absences left out)")
    return len(apps), eligible, source


# -- rosters + injuries ---------------------------------------------------------------

def team_roster(conn, team_id: int, season: str):
    rows = conn.execute(
        """SELECT s.player_id, pl.full_name FROM roster_stints s JOIN players pl ON pl.player_id = s.player_id
           WHERE s.team_id = ? AND s.end_date IS NULL""", (team_id,)).fetchall()
    last_box = conn.execute(
        """SELECT g.game_date, g.season_type FROM games g JOIN team_game_stats t ON t.game_id = g.game_id
           WHERE t.team_id = ? AND g.season_id = ? AND g.status = 'final'
           ORDER BY g.game_date DESC LIMIT 1""", (team_id, season)).fetchone()
    if rows and last_box:
        status = "confirmed" if last_box[1] != "preseason" else "estimated"
        return {r[0]: r[1] for r in rows}, status, last_box[0]
    # No box score this season yet: last season's players still tied to this team.
    prev = prev_season(season)
    rows = conn.execute(
        """WITH last_team AS (
               SELECT p.player_id, p.team_id, MAX(g.game_date) AS d FROM player_game_stats p
               JOIN games g ON g.game_id = p.game_id WHERE g.season_id IN (?, ?) AND p.seconds > 0
               GROUP BY p.player_id, p.team_id),
           latest AS (SELECT player_id, team_id, d, ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY d DESC) rn
                      FROM last_team)
           SELECT l.player_id, pl.full_name FROM latest l JOIN players pl ON pl.player_id = l.player_id
           WHERE l.rn = 1 AND l.team_id = ?""", (prev, season, team_id)).fetchall()
    return {r[0]: r[1] for r in rows}, "estimated", None


def latest_report(conn):
    r = conn.execute(
        "SELECT report_id, report_time_et, source_url, parse_ok, parse_notes FROM injury_reports "
        "ORDER BY report_time_et DESC LIMIT 1").fetchone()
    return dict(r) if r else None


def report_statuses(conn, report_id, game_id, team_id, game_date):
    rows = conn.execute(
        """SELECT player_id, player_name_raw, status, reason FROM injury_report_entries
           WHERE report_id = ? AND team_id = ? AND (game_id = ? OR game_id IS NULL)""",
        (report_id, team_id, game_id)).fetchall()
    return [dict(r) for r in rows]


# -- projection -------------------------------------------------------------------------

def _project(model, fb, game, team_id, opp_id, home, season, playing, odds):
    if not playing:
        return {}
    # The model was trained on players who actually got in, so it is told how many
    # players are expected to get in, not the size of the whole roster.
    n_exp = float(sum(odds[p]["p"] for p in playing))
    tg = M.Target(game["game_id"], game["game_date"], season, team_id, opp_id, home, sorted(playing), n_exp)
    rows = fb.rows([tg])
    cond = model.predict_raw(M.fill(rows))
    p_in = np.array([odds[p]["p"] for p in rows["player_id"]])
    exp = cond * p_in
    scaled = M.scale_to_team(rows, exp, model.total)
    factor = scaled.sum() / exp.sum() if exp.sum() else 1.0
    if_plays = np.minimum(cond * factor, 48.0)
    lo, hi = model.interval(if_plays)
    out = {}
    for i, p in enumerate(rows["player_id"]):
        out[int(p)] = {"min": float(scaled[i]), "if_plays": float(if_plays[i]), "p10": float(lo[i]), "p90": float(hi[i]),
                       "start_rate": float(rows["start_rate_l10"].iloc[i]),
                       "m_l10": None if np.isnan(rows["m_l10"].iloc[i]) else float(rows["m_l10"].iloc[i])}
    return out


def project_games(conn, season: str, games: list, rates: dict) -> dict:
    """Projections for a list of upcoming games (dicts from the games table)."""
    if not games:
        return {"games": []}
    model, fb, _ = train_minutes_model(conn, season)
    m_season = model_season(conn, season)
    cal = minutes_calibration(conn, m_season) if m_season else None
    if model is None:
        return {"games": [], "note": "no history to train the minutes model"}
    if cal and cal.get("intervals"):
        model.resid_q = [tuple(x) for x in cal["intervals"]]
    rep = latest_report(conn)
    appear = fit_appearance(conn, season)
    prev_cache = {}
    out = []
    for g in games:
        teams = {}
        for team_id, opp_id, home in ((g["home_team_id"], g["away_team_id"], 1), (g["away_team_id"], g["home_team_id"], 0)):
            roster, roster_status, roster_date = team_roster(conn, team_id, season)
            entries = report_statuses(conn, rep["report_id"], g["game_id"], team_id, g["game_date"]) if rep else []
            listed = {e["player_id"]: e for e in entries if e["player_id"]}
            submitted = not any(e["status"] == "not_yet_submitted" for e in entries)
            out_ids = {p for p, e in listed.items() if e["status"] == "out"}
            uncertain = [p for p, e in listed.items() if e["status"] in ("questionable", "doubtful") and p in roster]
            candidates = [p for p in roster if p not in out_ids]
            odds = appearance_odds(conn, team_id, season, candidates, appear, prev_cache)
            base = _project(model, fb, g, team_id, opp_id, home, season, set(candidates), odds)
            # limit scenarios to the uncertain players with the most minutes at stake
            uncertain.sort(key=lambda p: -base.get(p, {}).get("min", 0))
            in_play = uncertain[:MAX_UNCERTAIN]
            fixed_out = {p for p in uncertain[MAX_UNCERTAIN:] if listed[p]["status"] == "doubtful"}
            scenarios = []
            for k in range(len(in_play) + 1):
                for sitting in itertools.combinations(in_play, k):
                    sit = set(sitting) | fixed_out
                    proj = _project(model, fb, g, team_id, opp_id, home, season, set(candidates) - sit, odds)
                    prob = 1.0
                    for p in in_play:
                        r = rates.get(listed[p]["status"], {}).get("rate")
                        if r is None:
                            prob = None
                            break
                        prob *= (1 - r) if p in sitting else r
                    scenarios.append({"sitting": sorted(int(p) for p in sit), "probability": prob,
                                      "minutes": {str(p): round(v["min"], 1) for p, v in proj.items()}})
            # expected minutes across scenarios when probabilities exist
            if scenarios and all(s["probability"] is not None for s in scenarios):
                expected = {}
                for s in scenarios:
                    for p, mnt in s["minutes"].items():
                        expected[p] = expected.get(p, 0) + s["probability"] * mnt
            else:
                expected = None
            ranked = sorted(base.items(), key=lambda kv: (-kv[1]["start_rate"], -(kv[1]["m_l10"] or 0)))
            starters = [p for p, v in ranked if odds[p]["p"] >= 0.5][:5]
            players = []
            for p in sorted(roster, key=lambda p: -base.get(p, {}).get("min", 0)):
                e = listed.get(p)
                b = base.get(p)
                players.append({
                    "player_id": int(p), "name": roster[p],
                    "status": e["status"] if e else ("available" if submitted else "unknown"),
                    "reason": e["reason"] if e else None,
                    "min": round(b["min"], 1) if b else 0.0, "p10": round(b["p10"], 1) if b else 0.0,
                    "p90": round(b["p90"], 1) if b else 0.0,
                    "expected_min": round(expected[str(p)], 1) if expected and str(p) in expected else None,
                    "appear": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in odds[p].items()} if p in odds else None,
                    "projected_starter": p in starters,
                })
            unmatched = [e for e in entries if not e["player_id"] and e["status"] != "not_yet_submitted"]
            teams[str(team_id)] = {
                "roster_status": roster_status, "roster_as_of": roster_date,
                "report_submitted": submitted, "players": players, "scenarios": scenarios,
                "unmatched_report_names": [{"name": e["player_name_raw"], "status": e["status"]} for e in unmatched],
                "starters_status": "projected",
            }
        out.append({"game_id": g["game_id"], "game_date": g["game_date"], "tip_utc": g["tip_utc"], "teams": teams})
    return {
        "games": out,
        "injury_report": rep,
        "play_rates": rates,
        "minutes_model": {k: v for k, v in (cal or {}).items() if k != "intervals"},
        "team_minutes": model.total,
        "appearance_model": appear,
    }
