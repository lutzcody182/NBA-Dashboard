"""Game predictions: team rates + schedule + home court + who's playing, through the simulator.

Calibration (cached per season in model_calibration, like the player ratings):
  * team-rate shrinkage, fit on the calibration season (team_rates.calibrate)
  * schedule and home-court effects, fit on the seasons before it
  * score effect (how much a lead changes efficiency), measured from lineup stints
  * game-to-game spread (pace and form), picked by replaying the calibration
    season day by day with only what was known before each game

The replay doubles as the model's report card: every game's prediction is
kept, so accuracy and calibration can be shown on the dashboard.
"""
import json
import logging
import math
import zlib

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from ..util import utc_now_iso
from . import schedule as S
from . import sim
from . import team_rates as TR

log = logging.getLogger(__name__)
FORM_GRID = (0.0, 0.05, 0.1, 0.15)
N_SIMS_LIVE = 10000
N_SIMS_BACKTEST = 800


def prev_season(sid: str) -> str:
    y = int(sid[:4]) - 1
    return f"{y}-{str(y + 1)[-2:]}"


def lead_slope(conn, seasons) -> float:
    """Points per possession lost per point of lead, with team and opponent strength held fixed."""
    q = ",".join("?" * len(seasons))
    st = pd.read_sql_query(
        f"""SELECT s.team_id, s.opp_id, s.poss, s.pts, s.margin_start, g.season_id FROM lineup_stints s
            JOIN games g ON g.game_id = s.game_id
            WHERE g.season_type = 'regular' AND g.season_id IN ({q}) AND s.poss > 0""", conn, params=list(seasons))
    if st.empty:
        return 0.0
    lead = st["margin_start"].clip(-30, 30).to_numpy(float)
    ts = pd.factorize(st["season_id"] + "_" + st["team_id"].astype(str))[0]
    os_ = pd.factorize(st["season_id"] + "_" + st["opp_id"].astype(str))[0] + ts.max() + 1
    n = len(st)
    X = sparse.csr_matrix((np.r_[np.ones(n), np.ones(n), lead],
                           (np.r_[np.arange(n), np.arange(n), np.arange(n)],
                            np.r_[ts, os_, np.full(n, os_.max() + 1)])))
    w = np.sqrt(st["poss"].to_numpy(float))
    y = (st["pts"] / st["poss"]).to_numpy(float)
    return float(lsqr(sparse.diags(w) @ X, w * y)[0][-1])


class Context:
    """Everything needed to predict games on one date."""

    def __init__(self, est, effects, sched, lead_effect, league_ppp, form_sd, pace_sd, strength_scale=1.0):
        self.est, self.effects, self.sched = est, effects, sched
        self.lead_effect, self.league_ppp = lead_effect, league_ppp
        self.form_sd, self.pace_sd = form_sd, pace_sd
        self.strength_scale = strength_scale

    def inputs(self, game_id, home, away, neutral=False, roster=None):
        """Matchup rates, pace and points-per-100 shifts for one game, with each piece kept for display."""
        est, lg = self.est, self.est["league"]
        rh, ra = TR.matchup(est, home, away), TR.matchup(est, away, home)
        pace = est["teams"][home]["off"]["pace"] * est["teams"][away]["off"]["pace"] / lg["pace"]
        hca = 0.0 if neutral else self.effects["home_court"]
        sh = self.sched.get((game_id, home))
        sa = self.sched.get((game_id, away))
        fat = 0.0
        fat_detail = {}
        if sh is not None and sa is not None:
            for f in S.FEATURES:
                d = self.effects["effects"][f]["per100"] * (sh[f] - sa[f])
                if d:
                    fat_detail[f] = round(d, 2)
                fat += d
        # the rates' expected points run slightly below real scoring (free-throw rebounds, technicals ...)
        bias = 100 * (self.league_ppp - sim.expected_ppp(lg))
        r_h, r_a = (roster or {}).get(home, (0.0, 0.0)), (roster or {}).get(away, (0.0, 0.0))
        # roster tuple: (offense change, defense change) in points per 100, + is better
        # Each rate is shrunk on its own, which over-shrinks the team as a whole; the
        # strength scale (learned by replaying a season) widens matchup edges back out.
        base = sim.expected_ppp(lg)
        edge_h = 100 * (sim.expected_ppp(rh) - base) * (self.strength_scale - 1)
        edge_a = 100 * (sim.expected_ppp(ra) - base) * (self.strength_scale - 1)
        d_home = bias + edge_h + (hca + fat) / 2 + r_h[0] - r_a[1]
        d_away = bias + edge_a - (hca + fat) / 2 + r_a[0] - r_h[1]
        return {"rates": (rh, ra), "pace": pace,
                "shift": (sim.shift_for(rh, d_home), sim.shift_for(ra, d_away)),
                "parts": {"home_court": hca, "schedule": round(fat, 2), "schedule_detail": fat_detail,
                          "edge_scale": (round(edge_h, 2), round(edge_a, 2)),
                          "roster_home": r_h, "roster_away": r_a, "league_adjust": round(bias, 2)}}

    def predict(self, game_id, home, away, neutral=False, roster=None, n_sims=N_SIMS_LIVE, seed=0, keep=False):
        x = self.inputs(game_id, home, away, neutral, roster)
        res = sim.simulate(x["rates"][0], x["rates"][1], pace=x["pace"], home_shift=x["shift"][0],
                           away_shift=x["shift"][1], form_sd=self.form_sd, pace_sd=self.pace_sd,
                           lead_effect=self.lead_effect, n_sims=n_sims, seed=seed, keep=keep)
        res["inputs"] = {"pace": x["pace"], **x["parts"],
                         "expected_ppp_home": sim.expected_ppp(x["rates"][0], x["shift"][0]),
                         "expected_ppp_away": sim.expected_ppp(x["rates"][1], x["shift"][1])}
        return res


def _sched_lookup(sched: pd.DataFrame) -> dict:
    return {(g, t): row for g, t, row in zip(sched["game_id"], sched["team_id"], sched[S.FEATURES].to_dict("records"))}


def _all_games(conn):
    return pd.read_sql_query(
        """SELECT game_id, season_id, season_type, game_date, home_team_id, away_team_id, neutral_site, status,
                  home_score, away_score FROM games WHERE season_type = 'regular'""", conn)


# -- who's playing ----------------------------------------------------------------

PREV_COMPOSITION_WEIGHT = 0.35     # last season's lineup mix, per game, vs. this season's


def replacement_value(ratings: dict) -> tuple:
    """Rating used for players without one: the average of low-minute players."""
    low = [(o, d) for o, d, poss in ratings.values() if poss < 1500]
    if not low:
        return (0.0, 0.0)
    return (float(np.mean([x[0] for x in low])), float(np.mean([x[1] for x in low])))


def floor_sum(minutes: dict, ratings: dict, repl: tuple) -> tuple:
    """Average offense and defense ratings summed over the five on the floor, from minutes per player."""
    tot = sum(minutes.values()) or 1.0
    o = sum(m * ratings.get(p, repl)[0] for p, m in minutes.items()) * 5 / tot
    d = sum(m * ratings.get(p, repl)[1] for p, m in minutes.items()) * 5 / tot
    return o, d


class RosterHistory:
    """Season-so-far (and last season's) minutes, to compare tonight's roster with the one the rates saw."""

    def __init__(self, conn, season):
        prev = prev_season(season)
        self.pg = pd.read_sql_query(
            """SELECT p.game_id, p.player_id, p.team_id, p.seconds, g.game_date, g.season_id
               FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
               WHERE g.season_type = 'regular' AND g.status = 'final' AND p.seconds > 0
                 AND g.season_id IN (?, ?)""", conn, params=(prev, season))
        self.season, self.prev = season, prev

    def base(self, team, day, ratings, repl):
        """Ratings of the floor the team rates were built on (this season so far plus a share of last)."""
        cur = self.pg[(self.pg.season_id == self.season) & (self.pg.team_id == team) & (self.pg.game_date < day)]
        prv = self.pg[(self.pg.season_id == self.prev) & (self.pg.team_id == team)]
        n_c, n_p = cur.game_id.nunique(), prv.game_id.nunique() * PREV_COMPOSITION_WEIGHT
        parts = []
        for frame, n in ((cur, n_c), (prv, n_p)):
            if n:
                parts.append((floor_sum(frame.groupby("player_id")["seconds"].sum().to_dict(), ratings, repl), n))
        if not parts:
            return None
        tot = sum(n for _, n in parts)
        return (sum(v[0] * n for v, n in parts) / tot, sum(v[1] * n for v, n in parts) / tot)

    def expected_minutes(self, team, day, players):
        """Recent minutes for the players who will play (last 10 games with the team, else last season)."""
        cur = self.pg[(self.pg.season_id == self.season) & (self.pg.game_date < day)]
        out = {}
        for p in players:
            mine = cur[(cur.player_id == p) & (cur.team_id == team)].sort_values("game_date").tail(10)
            if len(mine):
                out[p] = float(mine.seconds.mean() / 60)
                continue
            old = self.pg[(self.pg.season_id == self.prev) & (self.pg.player_id == p)]
            out[p] = float(old.seconds.mean() / 60) if len(old) else 12.0
        tot = sum(out.values()) or 1.0
        return {p: m * 240 / tot for p, m in out.items()}


def roster_effect(minutes: dict, base, ratings, repl, scale: float) -> tuple:
    if base is None or not minutes:
        return (0.0, 0.0)
    o, d = floor_sum(minutes, ratings, repl)
    return (scale * (o - base[0]), scale * (d - base[1]))


def center(rosters: dict, ratings, repl, scale: float) -> dict:
    """Roster effects for one day's slate, minus the slate's average.

    Unrated players (mostly rookies) get a below-average value, which would
    otherwise pull every early-season total down; centring keeps who's in
    and out relative to everyone else playing that day.
    """
    eff = {k: roster_effect(m, b, ratings, repl, scale) for k, (m, b) in rosters.items()}
    if not eff:
        return eff
    mo = float(np.mean([v[0] for v in eff.values()]))
    md = float(np.mean([v[1] for v in eff.values()]))
    return {k: (v[0] - mo, v[1] - md) for k, v in eff.items()}


def replay(conn, season, fits, effects, lead_effect, form_sds=(0.0,), pace_sd=None, n_sims=N_SIMS_BACKTEST,
           max_games=None, ratings=None, roster_scales=(0.0,), strength_scales=(1.0,)) -> pd.DataFrame:
    """Predict every game of a finished season using only games before its date.

    With `ratings` ({player: (offense, defense, possessions)} known before the season), each
    team's rating is adjusted for who actually played that night, at their recent minutes.
    """
    prev = prev_season(season)
    hist = RosterHistory(conn, season) if ratings else None
    repl = replacement_value(ratings) if ratings else None
    played = hist.pg.groupby(["game_id", "team_id"])["player_id"].agg(list).to_dict() if hist else {}
    cur_all = TR.team_games(conn, [season])
    prev_tg = TR.team_games(conn, [prev])
    games = _all_games(conn)
    sched = _sched_lookup(S.team_schedule(games))
    target = games[(games.season_id == season) & (games.status == "final")].sort_values(["game_date", "game_id"])
    if max_games:
        target = target.head(max_games)
    poss = cur_all.groupby("game_id")["poss"].mean()
    rows = []
    teams = sorted(set(target.home_team_id) | set(target.away_team_id))
    for day, todays in target.groupby("game_date"):
        cur = cur_all[cur_all.game_date < day]
        est = TR.estimates(cur, prev_tg, fits, teams)
        league_ppp = float(cur.pts.sum() / cur.poss.sum()) if len(cur) >= 60 else \
            float(prev_tg.pts.sum() / prev_tg.poss.sum())
        rost = {}
        if hist:
            for g in todays.itertuples():
                for t in (g.home_team_id, g.away_team_id):
                    mins = hist.expected_minutes(t, day, played.get((g.game_id, t), []))
                    rost[(g.game_id, t)] = (mins, hist.base(t, day, ratings, repl))
        centred = {sc: center(rost, ratings, repl, sc) for sc in roster_scales} if hist else {}
        ctxs = {ks: Context(est, effects, sched, lead_effect, league_ppp, 0.0, pace_sd or 0.0, ks)
                for ks in strength_scales}
        combos = [(fs, ks, sc) for fs in form_sds for ks in strength_scales for sc in roster_scales]
        for g in todays.itertuples():
            # every setting being compared is simulated in one batch: same rates and pace,
            # different shifts and form spread per block of n_sims games
            hs, as_, fsd, inp = [], [], [], None
            for fs, ks, sc in combos:
                roster = None
                if hist and sc:
                    roster = {t: centred[sc][(g.game_id, t)] for t in (g.home_team_id, g.away_team_id)}
                x = ctxs[ks].inputs(g.game_id, g.home_team_id, g.away_team_id, bool(g.neutral_site), roster)
                inp = inp or x
                hs.append(np.full(n_sims, x["shift"][0]))
                as_.append(np.full(n_sims, x["shift"][1]))
                fsd.append(np.full(n_sims, fs))
            r = sim.simulate(inp["rates"][0], inp["rates"][1], pace=inp["pace"], home_shift=np.concatenate(hs),
                             away_shift=np.concatenate(as_), form_sd=np.concatenate(fsd), pace_sd=pace_sd or 0.0,
                             lead_effect=lead_effect, n_sims=n_sims * len(combos),
                             seed=zlib.crc32(g.game_id.encode()), keep=True)
            m, tot = g.home_score - g.away_score, g.home_score + g.away_score
            for i, (fs, ks, sc) in enumerate(combos):
                mg = r["_margin"][i * n_sims:(i + 1) * n_sims]
                tt = r["_total"][i * n_sims:(i + 1) * n_sims]
                rows.append({"game_id": g.game_id, "game_date": day, "home": g.home_team_id,
                             "away": g.away_team_id, "form_sd": fs, "roster_scale": sc, "strength_scale": ks,
                             "p_home": float((mg > 0).mean()),
                             "margin_pred": float(mg.mean()), "total_pred": float(tt.mean()),
                             "margin": m, "total": tot,
                             "pace_pred": inp["pace"], "poss": float(poss.get(g.game_id, np.nan)),
                             "pit_margin": float((mg < m).mean() + 0.5 * (mg == m).mean()),
                             "pit_total": float((tt < tot).mean() + 0.5 * (tt == tot).mean()),
                             "sim_margin_sd": float(mg.std()), "sim_total_sd": float(tt.std())})
    return pd.DataFrame(rows)


def scores(df: pd.DataFrame) -> dict:
    y = (df["margin"] > 0).astype(float)
    p = df["p_home"].clip(0.005, 0.995)
    cover80 = lambda pit: float(((pit > 0.1) & (pit < 0.9)).mean())  # noqa: E731
    return {
        "games": int(len(df)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(((p > 0.5) == (y == 1)).mean()),
        "margin_rmse": float(np.sqrt(np.mean((df["margin"] - df["margin_pred"]) ** 2))),
        "margin_mae": float(np.mean(np.abs(df["margin"] - df["margin_pred"]))),
        "total_rmse": float(np.sqrt(np.mean((df["total"] - df["total_pred"]) ** 2))),
        "total_bias": float(np.mean(df["total"] - df["total_pred"])),
        "margin_bias": float(np.mean(df["margin"] - df["margin_pred"])),
        "margin_80_coverage": cover80(df["pit_margin"]),
        "total_80_coverage": cover80(df["pit_total"]),
    }


def calibration_table(df: pd.DataFrame, bins=10) -> list:
    d = df.copy()
    d["bin"] = np.minimum((d["p_home"] * bins).astype(int), bins - 1)
    out = []
    for b, g in d.groupby("bin"):
        out.append({"from": b / bins, "to": (b + 1) / bins, "games": int(len(g)),
                    "predicted": float(g["p_home"].mean()), "actual": float((g["margin"] > 0).mean())})
    return out


def baseline_scores(df: pd.DataFrame, hca_pts: float) -> dict:
    """What a simple 'home team by a few points' guess would have scored, for comparison."""
    b = df.copy()
    b["margin_pred"] = hca_pts
    sd = float(np.std(b["margin"]))
    b["p_home"] = 0.5 * (1 + np.vectorize(math.erf)(hca_pts / (sd * math.sqrt(2))))
    b["total_pred"] = float(df["total"].mean())
    b["pit_margin"] = 0.5
    b["pit_total"] = 0.5
    s = scores(b)
    return {k: s[k] for k in ("brier", "log_loss", "accuracy", "margin_rmse", "total_rmse")}


ROSTER_GRID = (1.0, 1.5, 2.0)
STRENGTH_GRID = (1.0, 1.3, 1.6, 2.0)
N_SIMS_GRID = 300


def calibrate(conn, season_m: str) -> dict:
    """Fit everything for predictions made during the season after season_m, and grade it on season_m."""
    from .ratings import compute as rate_players
    prev = prev_season(season_m)
    log.info("Game model: calibrating on %s", season_m)
    fits = TR.calibrate(conn, season_m, prev)["fits"]
    seasons_before = [r[0] for r in conn.execute(
        "SELECT DISTINCT season_id FROM games WHERE season_type = 'regular' AND status = 'final' AND season_id < ? "
        "ORDER BY 1", (season_m,))]
    eff_train = S.calibrate(conn, seasons_before) if seasons_before else _no_effects()
    slope = lead_slope(conn, seasons_before or [season_m])
    # dPPP / d(shift) at league rates, to turn the slope into the simulator's units
    lg = TR.estimates(TR.team_games(conn, [season_m]), TR.team_games(conn, [prev]), fits, [])["league"]
    step = sim.expected_ppp(lg, 0.05) - sim.expected_ppp(lg, -0.05)
    lead_effect = slope / (step / 0.1)
    # player ratings as they stood before season_m, to value who was in and out each night
    try:
        rp = rate_players(conn, prev)
        ratings = {p["player_id"]: (p["offense"]["value"], p["defense"]["value"], p["poss"] + p.get("poss_prev", 0))
                   for p in rp["players"]} or None
    except Exception:
        log.exception("Ratings for %s unavailable; game model calibrated without roster effects", prev)
        ratings = None

    def pick(graded):
        return min(graded, key=lambda k: graded[k]["log_loss"] + abs(graded[k]["margin_80_coverage"] - 0.8) / 10)

    roster_grid = ROSTER_GRID if ratings else (0.0,)
    # 1. edge scale and roster scale together (the two interact), with no extra form spread
    df1 = replay(conn, season_m, fits, eff_train, lead_effect, ratings=ratings, roster_scales=roster_grid,
                 strength_scales=STRENGTH_GRID, n_sims=N_SIMS_GRID)
    pace_sd = float(np.nanstd(df1["poss"] - df1["pace_pred"]))
    graded_1 = {(ks, sc): scores(g) for (ks, sc), g in df1.groupby(["strength_scale", "roster_scale"])}
    best_ks, best_scale = pick(graded_1)
    # 2. game-to-game form spread at those settings, with the pace spread included
    df2 = replay(conn, season_m, fits, eff_train, lead_effect, form_sds=FORM_GRID, pace_sd=pace_sd,
                 ratings=ratings, roster_scales=(best_scale,), strength_scales=(best_ks,), n_sims=N_SIMS_GRID)
    graded_form = {fs: scores(df2[df2.form_sd == fs]) for fs in FORM_GRID}
    best_form = pick(graded_form)
    # 3. the report card: a full-size replay at the chosen settings, plus one without who's playing
    df3 = replay(conn, season_m, fits, eff_train, lead_effect, form_sds=(best_form,), pace_sd=pace_sd,
                 ratings=ratings, roster_scales=tuple(sorted({0.0, best_scale})), strength_scales=(best_ks,))
    final = df3[df3.roster_scale == best_scale]
    graded_roster = {str(sc): v for (ks, sc), v in graded_1.items() if ks == best_ks}
    eff_all = S.calibrate(conn, seasons_before + [season_m])
    return {
        "season": season_m, "created_at": utc_now_iso(),
        "team_rate_fits": fits,
        "schedule": eff_all,                     # used for live predictions (all finished seasons)
        "schedule_for_replay": eff_train,
        "lead_slope_ppp": slope, "lead_effect": lead_effect,
        "form_sd": best_form, "pace_sd": pace_sd, "roster_scale": best_scale, "strength_scale": best_ks,
        "form_grid": {str(k): v for k, v in graded_form.items()},
        "roster_grid": graded_roster,
        "strength_grid": {f"{ks}|{sc}": v for (ks, sc), v in graded_1.items()},
        "report": {"scores": scores(final), "calibration": calibration_table(final),
                   "baseline": baseline_scores(final, eff_all["home_court"] * final["pace_pred"].mean() / 100),
                   "without_roster": scores(df3[df3.roster_scale == 0.0]) if best_scale else scores(final)},
        "replay": final[["game_id", "game_date", "home", "away", "p_home", "margin_pred", "total_pred", "margin",
                         "total"]].round(4).to_dict("records"),
    }


def _no_effects():
    return {"home_court": 0.0, "effects": {f: {"per100": 0.0} for f in S.FEATURES}}


def get_calibration(conn, season_m: str, force=False) -> dict:
    key = f"game_model:{season_m}"
    if not force:
        row = conn.execute("SELECT payload FROM model_calibration WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
    cal = calibrate(conn, season_m)
    conn.execute("INSERT OR REPLACE INTO model_calibration (key, payload, created_at) VALUES (?, ?, ?)",
                 (key, json.dumps(cal, default=float), cal["created_at"]))
    conn.commit()
    return cal
