"""Schedule and fatigue: rest, back-to-backs, travel, time zones and altitude.

Every team-game gets its schedule situation, worked out from the games
table alone. The effect of each situation on scoring margin is then
measured from past regular seasons, with each team's strength in that
season held fixed so that only the schedule is being measured. Effects
are ridge-shrunk toward zero by an amount picked by leaving one season out
at a time, and each is reported with its standard error so the dashboard
can show how sure we are.
"""
import math
from zoneinfo import ZoneInfo
from datetime import datetime

import numpy as np
import pandas as pd

# NBA team id -> (arena latitude, longitude, IANA time zone, arena elevation in feet)
ARENAS = {
    1610612737: (33.757, -84.396, "America/New_York", 1050),     # ATL State Farm Arena
    1610612738: (42.366, -71.062, "America/New_York", 20),       # BOS TD Garden
    1610612739: (41.496, -81.688, "America/New_York", 650),      # CLE Rocket Arena
    1610612740: (29.949, -90.082, "America/Chicago", 5),         # NOP Smoothie King Center
    1610612741: (41.881, -87.674, "America/Chicago", 595),       # CHI United Center
    1610612742: (32.790, -96.810, "America/Chicago", 430),       # DAL American Airlines Center
    1610612743: (39.749, -105.008, "America/Denver", 5280),      # DEN Ball Arena
    1610612744: (37.768, -122.388, "America/Los_Angeles", 10),   # GSW Chase Center
    1610612745: (29.751, -95.362, "America/Chicago", 50),        # HOU Toyota Center
    1610612746: (33.945, -118.343, "America/Los_Angeles", 100),  # LAC Intuit Dome
    1610612747: (34.043, -118.267, "America/Los_Angeles", 270),  # LAL Crypto.com Arena
    1610612748: (25.781, -80.188, "America/New_York", 5),        # MIA Kaseya Center
    1610612749: (43.045, -87.917, "America/Chicago", 600),       # MIL Fiserv Forum
    1610612750: (44.979, -93.276, "America/Chicago", 830),       # MIN Target Center
    1610612751: (40.683, -73.976, "America/New_York", 30),       # BKN Barclays Center
    1610612752: (40.751, -73.993, "America/New_York", 30),       # NYK Madison Square Garden
    1610612753: (28.539, -81.384, "America/New_York", 100),      # ORL Kia Center
    1610612754: (39.764, -86.156, "America/New_York", 715),      # IND Gainbridge Fieldhouse
    1610612755: (39.901, -75.172, "America/New_York", 10),       # PHI Wells Fargo Center
    1610612756: (33.446, -112.071, "America/Phoenix", 1085),     # PHX Footprint Center
    1610612757: (45.532, -122.667, "America/Los_Angeles", 50),   # POR Moda Center
    1610612758: (38.580, -121.500, "America/Los_Angeles", 30),   # SAC Golden 1 Center
    1610612759: (29.427, -98.438, "America/Chicago", 650),       # SAS Frost Bank Center
    1610612760: (35.463, -97.515, "America/Chicago", 1200),      # OKC Paycom Center
    1610612761: (43.643, -79.379, "America/Toronto", 250),       # TOR Scotiabank Arena
    1610612762: (40.768, -111.901, "America/Denver", 4265),      # UTA Delta Center
    1610612763: (35.138, -90.051, "America/Chicago", 260),       # MEM FedExForum
    1610612764: (38.898, -77.021, "America/New_York", 50),       # WAS Capital One Arena
    1610612765: (42.341, -83.055, "America/New_York", 600),      # DET Little Caesars Arena
    1610612766: (35.225, -80.839, "America/New_York", 750),      # CHA Spectrum Center
}
HIGH_ALTITUDE_FT = 4000

FEATURES = ["b2b", "rest2", "rest3plus", "three_in_four", "travel_1000mi", "tz_west", "tz_east",
            "visitor_altitude", "road_trip_long"]
LABELS = {
    "b2b": "Second night of a back-to-back",
    "rest2": "Two days' rest (vs one)",
    "rest3plus": "Three or more days' rest (vs one)",
    "three_in_four": "Third game in four nights",
    "travel_1000mi": "Per 1,000 miles travelled since last game",
    "tz_west": "Per hour of time zones crossed going west",
    "tz_east": "Per hour of time zones crossed going east",
    "visitor_altitude": "Visiting team at altitude (Denver, Utah)",
    "road_trip_long": "Fourth or later game of a road trip",
}


def miles(a, b) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 3958.8 * math.asin(math.sqrt(h))


def _utc_offset(tz: str, when: str) -> float:
    d = datetime.fromisoformat(when + "T19:00:00").replace(tzinfo=ZoneInfo(tz))
    return d.utcoffset().total_seconds() / 3600


def team_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with its schedule situation.

    games: game_id, season_id, game_date, home_team_id, away_team_id
    (any status: future games get their situation too, from the games
    scheduled before them).
    """
    rows = []
    for side, opp, home in (("home_team_id", "away_team_id", 1), ("away_team_id", "home_team_id", 0)):
        part = games[["game_id", "season_id", "game_date", side, opp, "home_team_id"]].copy()
        part.columns = ["game_id", "season_id", "game_date", "team_id", "opp_id", "site_team"]
        part["home"] = home
        rows.append(part)
    tg = pd.concat(rows, ignore_index=True)
    tg = tg[tg.site_team.isin(ARENAS) & tg.team_id.isin(ARENAS)]
    tg = tg.sort_values(["team_id", "season_id", "game_date", "game_id"]).reset_index(drop=True)
    tg["date"] = pd.to_datetime(tg.game_date)
    out = []
    for (_, _), g in tg.groupby(["team_id", "season_id"], sort=False):
        g = g.copy()
        gap = g.date.diff().dt.days
        g["rest"] = (gap - 1).fillna(4).clip(lower=0, upper=4)
        g["b2b"] = (g.rest == 0).astype(int)
        g["rest2"] = (g.rest == 2).astype(int)
        g["rest3plus"] = (g.rest >= 3).astype(int)
        span = (g.date - g.date.shift(2)).dt.days
        g["three_in_four"] = (span <= 3).astype(int)
        prev_site = g.site_team.shift(1)
        trav, tzs = [], []
        for site, before, day in zip(g.site_team, prev_site, g.game_date):
            if pd.isna(before):
                trav.append(0.0)
                tzs.append(0.0)
                continue
            a, b = ARENAS[int(before)], ARENAS[int(site)]
            trav.append(miles(a[:2], b[:2]))
            tzs.append(_utc_offset(b[2], day) - _utc_offset(a[2], day))   # negative = moved west
        g["travel_mi"] = trav
        g["travel_1000mi"] = g.travel_mi / 1000
        tz = np.array(tzs)
        g["tz_west"] = np.clip(-tz, 0, None)
        g["tz_east"] = np.clip(tz, 0, None)
        g["visitor_altitude"] = ((g.home == 0) & g.site_team.map(lambda t: ARENAS[int(t)][3] >= HIGH_ALTITUDE_FT)).astype(int)
        trip, n = [], 0
        for h in g.home:
            n = 0 if h else n + 1
            trip.append(n)
        g["road_trip_n"] = trip
        g["road_trip_long"] = (g.road_trip_n >= 4).astype(int)
        out.append(g)
    res = pd.concat(out, ignore_index=True) if out else tg
    return res.drop(columns=["date"])


# -- measuring the effects ----------------------------------------------------------

def _game_frame(conn, seasons) -> pd.DataFrame:
    q = f"""SELECT g.game_id, g.season_id, g.game_date, g.home_team_id, g.away_team_id, g.home_score, g.away_score,
                   g.neutral_site, h.possessions AS h_poss, a.possessions AS a_poss
            FROM games g JOIN team_game_stats h ON h.game_id = g.game_id AND h.team_id = g.home_team_id
            JOIN team_game_stats a ON a.game_id = g.game_id AND a.team_id = g.away_team_id
            WHERE g.status = 'final' AND g.season_type = 'regular'
              AND g.season_id IN ({",".join("?" * len(seasons))})"""
    return pd.read_sql_query(q, conn, params=list(seasons))


def design(games: pd.DataFrame, sched: pd.DataFrame):
    """Rows: games. Columns: home-court, team-season strengths (+home, -away), fatigue differences."""
    s = sched.set_index(["game_id", "team_id"])
    h = s.loc[list(zip(games.game_id, games.home_team_id)), FEATURES].to_numpy(float)
    a = s.loc[list(zip(games.game_id, games.away_team_id)), FEATURES].to_numpy(float)
    keys = sorted(set(zip(games.season_id, games.home_team_id)) | set(zip(games.season_id, games.away_team_id)))
    idx = {k: i for i, k in enumerate(keys)}
    T = np.zeros((len(games), len(keys)))
    for r, (sid, ht, at) in enumerate(zip(games.season_id, games.home_team_id, games.away_team_id)):
        T[r, idx[(sid, ht)]] += 1
        T[r, idx[(sid, at)]] -= 1
    hca = (1 - games.neutral_site.fillna(0).to_numpy(float))[:, None]
    return hca, T, h - a, idx


def margin_per100(games: pd.DataFrame) -> np.ndarray:
    poss = (games.h_poss + games.a_poss) / 2
    return (100 * (games.home_score - games.away_score) / poss).to_numpy(float)


def _solve(X, y, pen):
    A = X.T @ X + np.diag(pen)
    return np.linalg.solve(A, X.T @ y)


def fit_effects(games, sched, lam_fatigue: float, lam_team: float = 5.0):
    hca, T, F, idx = design(games, sched)
    X = np.hstack([hca, T, F])
    y = margin_per100(games)
    pen = np.r_[0.0, np.full(T.shape[1], lam_team), np.full(F.shape[1], lam_fatigue)]
    beta = _solve(X, y, pen)
    resid = y - X @ beta
    sigma2 = resid @ resid / max(1, len(y) - X.shape[1])
    cov = sigma2 * np.linalg.inv(X.T @ X + np.diag(pen))
    k = 1 + T.shape[1]
    return {"hca": beta[0], "hca_se": math.sqrt(cov[0, 0]), "beta": beta[k:], "se": np.sqrt(np.diag(cov)[k:]),
            "sigma": math.sqrt(sigma2), "n": len(y)}


def _holdout_mse(train, test, sched, lam_f, lam_team=5.0):
    """Fatigue effects learned on `train`; team strengths and home court refit on `test` with them fixed."""
    eff = fit_effects(train, sched, lam_f, lam_team)
    hca, T, F, _ = design(test, sched)
    y = margin_per100(test) - F @ eff["beta"]
    X = np.hstack([hca, T])
    # Split the test season in two by date: strengths from one half predict the other half.
    order = np.argsort(test.game_date.to_numpy())
    half = np.zeros(len(test), bool)
    half[order[: len(order) // 2]] = True
    err = 0.0
    for fit_on in (half, ~half):
        pen = np.r_[0.0, np.full(T.shape[1], lam_team)]
        b = _solve(X[fit_on], y[fit_on], pen)
        err += float(np.sum((y[~fit_on] - X[~fit_on] @ b) ** 2))
    return err / len(test)


def calibrate(conn, seasons, grid=(1.0, 30.0, 100.0, 300.0, 1000.0, 1e9)) -> dict:
    """Pick the fatigue shrinkage by leaving each season out; report effects fit on all seasons."""
    games = _game_frame(conn, seasons)
    all_games = pd.read_sql_query(
        "SELECT game_id, season_id, game_date, home_team_id, away_team_id FROM games WHERE season_type = 'regular'",
        conn)
    sched = team_schedule(all_games)
    scores = {}
    for lam in grid:
        tot = 0.0
        for s in seasons:
            tot += _holdout_mse(games[games.season_id != s], games[games.season_id == s], sched, lam) * \
                (games.season_id == s).sum()
        scores[lam] = tot / len(games)
    best = min(scores, key=scores.get)
    eff = fit_effects(games, sched, best)
    return {
        "seasons": list(seasons), "games": int(len(games)), "lambda": best,
        "holdout_mse": {str(k): round(v, 3) for k, v in scores.items()},
        "holdout_gain": round(scores[1e9] - scores[best], 3),   # vs. no schedule effects at all
        "home_court": round(eff["hca"], 2), "home_court_se": round(eff["hca_se"], 2),
        "sigma": round(eff["sigma"], 2),
        "effects": {f: {"label": LABELS[f], "per100": round(float(b), 2), "se": round(float(se), 2)}
                    for f, b, se in zip(FEATURES, eff["beta"], eff["se"])},
    }


def adjustment(effects: dict, team_row: pd.Series, opp_row: pd.Series) -> float:
    """Schedule effect on this team's margin per 100 possessions in one game."""
    return float(sum(effects[f]["per100"] * (team_row[f] - opp_row[f]) for f in FEATURES))
