"""A simulated league for testing the stats code end to end.

Each team gets a known true offensive and defensive strength. Games are
generated as real-format NBA box-score JSON, run through the real parser
and loader, and the outputs are compared against the known truth.
Test use only; nothing here ever reaches the dashboard.
"""
import numpy as np

from pipeline import config
from pipeline.ingest.store import store_boxscore
from pipeline.providers.nba_cdn import parse_boxscore

from .helpers import make_box, player

TEAM_IDS = sorted(config.NBA_TEAM_IDS)
MINUTES = [36, 34, 32, 30, 28, 26, 24, 30]   # 240 total
SHARES = np.array([0.24, 0.19, 0.15, 0.12, 0.10, 0.08, 0.05, 0.07])


def _split(total, shares):
    raw = shares * total
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: int(total - base.sum())]:
        base[i] += 1
    return base.tolist()


def _team_counts(rng, poss, eff):
    fga = int(round(poss * 0.87))
    fta = int(round(poss * 0.26))
    fg3a = int(round(fga * 0.40))
    ftm = int(round(fta * 0.78))
    target = eff * poss / 100
    fg3m = min(fg3a, int(round(fg3a * 0.36)))
    fgm = int(round((target - ftm - fg3m) / 2))
    fgm = max(fg3m, min(fga, fgm))
    pts = 2 * fgm + fg3m + ftm
    miss = fga - fgm
    oreb = int(round(miss * 0.27))
    dreb = int(round(34 + rng.normal(0, 2)))
    tov = max(5, int(round(poss - (fga + 0.44 * fta) + oreb * 1.0)))
    return dict(pts=pts, fgm=fgm, fga=fga, fg3m=fg3m, fg3a=fg3a, ftm=ftm, fta=fta,
                oreb=oreb, dreb=dreb, ast=int(fgm * 0.6), stl=7, blk=5, tov=tov, pf=19)


def _players(team_idx, c, starters_pos=("G", "G", "F", "F", "C")):
    split = {k: _split(v, SHARES) for k, v in c.items()}
    out = []
    for i in range(8):
        pid = 1_000_000 + team_idx * 100 + i
        kw = {k: split[k][i] for k in split}
        out.append(player(pid, f"P{i}", f"T{team_idx}", starter=i < 5, secs=MINUTES[i] * 60,
                          position=starters_pos[i] if i < 5 else None, **kw))
    return out


def simulate_season(conn, season_start_year=2025, games_per_team=82, seed=7, noise=11.0,
                    schedule_bias=None):
    """Load a simulated regular season into `conn`. Returns the true strengths.

    schedule_bias: optional {team_id: 'strong'|'weak'} to give a team an
    unbalanced schedule (plays only the strongest or weakest defenses).
    """
    rng = np.random.default_rng(seed)
    true_o = dict(zip(TEAM_IDS, rng.normal(0, 4, len(TEAM_IDS))))
    true_d = dict(zip(TEAM_IDS, rng.normal(0, 4, len(TEAM_IDS))))   # + = allows more
    pace = dict(zip(TEAM_IDS, rng.normal(99, 2.5, len(TEAM_IDS))))
    league, home_edge = 114.0, 1.5
    by_def = sorted(TEAM_IDS, key=lambda t: true_d[t])

    n_games = len(TEAM_IDS) * games_per_team // 2
    yy = season_start_year % 100
    day0 = np.datetime64(f"{season_start_year}-10-21")
    gid = 0
    for n in range(n_games):
        h, a = rng.choice(TEAM_IDS, 2, replace=False)
        if schedule_bias:
            for team, kind in schedule_bias.items():
                if team in (h, a):
                    pool = by_def[:5] if kind == "strong" else by_def[-5:]
                    other = rng.choice([t for t in pool if t != team])
                    h, a = (team, other) if rng.random() < 0.5 else (other, team)
        poss = (pace[h] + pace[a]) / 2 + rng.normal(0, 3)
        eff_h = league + true_o[h] + true_d[a] + home_edge + rng.normal(0, noise)
        eff_a = league + true_o[a] + true_d[h] - home_edge + rng.normal(0, noise)
        ch, ca = _team_counts(rng, poss, eff_h), _team_counts(rng, poss, eff_a)
        if ch["pts"] == ca["pts"]:
            ch["ftm"] += 1; ch["fta"] += 1; ch["pts"] += 1
        gid += 1
        date = str(day0 + np.timedelta64(int(n * 170 / n_games), "D"))
        hi, ai = TEAM_IDS.index(h), TEAM_IDS.index(a)
        box = make_box(f"002{yy:02d}{gid:05d}", date,
                       (int(h), f"T{hi:02d}", f"City{hi}", f"Team{hi}", _players(hi, ch)),
                       (int(a), f"T{ai:02d}", f"City{ai}", f"Team{ai}", _players(ai, ca)))
        store_boxscore(conn, parse_boxscore(box), "simulated")
    return {"o": true_o, "d": true_d, "pace": pace, "league": league, "home_edge": home_edge}
