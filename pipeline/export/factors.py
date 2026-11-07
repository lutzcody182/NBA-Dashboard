"""Matchup factors shown beside each prop: teammates out, opponent by position, rebound chances, expected threes.

These are for reading, not inputs to the model. Each was tested against past seasons
(fit on 2023-25, scored on 2025-26) before being added here:
  * teammates out: the game simulation already moves shots, assists and rebounds to
    teammates when someone sits (usage.py); player-specific with/without splits on top
    of that did not improve predictions
  * opponent: the simulation already uses each opponent's rates; splitting the
    opponent's defence by position on top of that added nothing
  * rebound chances: the expected number of missed shots in the game added about 0.2%
  * expected threes (attempts times a shrunk three-point percentage): about 0.2%
  * referee crews: crews' foul and free-throw tendencies did not carry from one season
    to the next, and no daily assignment feed is stored, so they are left out
  * travel and schedule (rest, back-to-backs, miles, time zones, altitude) are already
    in the game model and shown on each game card
So they explain the number rather than change it.
"""
import numpy as np
import pandas as pd

STATS = ("pts", "reb", "ast", "fg3m")
OUT_STATUSES = ("out", "doubtful")
KEY_TEAMMATE_MIN = 20.0        # only teammates averaging this many minutes are worth a split
MIN_GAMES_WITHOUT = 3
OPP_PRIOR_MIN = 2000.0         # minutes of league-average play an opponent's allowed rate is shrunk with
THREE_PRIOR_ATT = 200.0        # attempts of a player's longer record (then the league) his 3P% is shrunk with
THREE_RECENT = 15
LEAGUE_3P = 0.36


def _group(pos, ast_pm, reb_pm):
    p = pos.strip().upper() if isinstance(pos, str) else ""
    if p[:1] in ("G", "F", "C"):
        return p[:1], True
    if ast_pm >= 0.13:
        return "G", False
    if reb_pm >= 0.25:
        return "C", False
    return "F", False


def _load(conn, seasons):
    q = ",".join("?" * len(seasons))
    pg = pd.read_sql_query(
        f"""SELECT s.game_id, s.player_id, s.team_id, g.season_id, g.game_date, s.seconds / 60.0 AS m,
                   s.pts, s.oreb + s.dreb AS reb, s.ast, s.fg3m, s.fg3a, p.position,
                   CASE WHEN g.home_team_id = s.team_id THEN g.away_team_id ELSE g.home_team_id END AS opp
            FROM player_game_stats s JOIN games g ON g.game_id = s.game_id
            LEFT JOIN players p ON p.player_id = s.player_id
            WHERE g.season_id IN ({q}) AND g.season_type = 'regular' AND g.status = 'final' AND s.seconds > 0""",
        conn, params=list(seasons))
    tg = pd.read_sql_query(
        f"""SELECT t.game_id, t.team_id, g.season_id, t.fga - t.fgm AS miss, t.fta - t.ftm AS ftmiss, t.oreb + t.dreb AS reb
            FROM team_game_stats t JOIN games g ON g.game_id = t.game_id
            WHERE g.season_id IN ({q}) AND g.season_type = 'regular' AND g.status = 'final'""",
        conn, params=list(seasons))
    return pg, tg


def _teammate_splits(pg, pid, team, outs, season):
    """Per-game numbers for pid with and without each teammate listed out, over games for this team."""
    mine = pg[(pg.player_id == pid) & (pg.team_id == team)]
    res = []
    for oid, oname in outs:
        theirs = pg[(pg.player_id == oid) & (pg.team_id == team)]
        if theirs.empty or theirs.m.mean() < KEY_TEAMMATE_MIN:
            continue
        seasons = set(theirs.season_id)
        pool = mine[mine.season_id.isin(seasons)]
        # games the teammate was on the team: between his first and last game for it in each season
        span = theirs.groupby("season_id").game_date.agg(["min", "max"])
        pool = pool[[span.loc[s, "min"] <= d <= span.loc[s, "max"] if s in span.index else False
                     for s, d in zip(pool.season_id, pool.game_date)]] if len(pool) else pool
        with_ = pool[pool.game_id.isin(set(theirs.game_id))]
        without = pool[~pool.game_id.isin(set(theirs.game_id))]
        if len(without) < MIN_GAMES_WITHOUT or len(with_) < MIN_GAMES_WITHOUT:
            continue

        def line(d):
            return {"games": int(len(d)), "min": round(float(d.m.mean()), 1),
                    **{s: round(float(d[s].mean()), 1) for s in STATS}}
        res.append({"player_id": int(oid), "name": oname, "minutes": round(float(theirs.m.mean()), 1),
                    "with": line(with_), "without": line(without)})
    return res


def _opp_allowed(pg, season, prev):
    """(opponent team, position group) -> allowed per-minute rate / league rate, per stat."""
    lg = {s: pg[s].sum() / pg.m.sum() for s in STATS}
    out = {}
    for (opp, grp), d in pg.groupby(["opp", "grp"]):
        cur, old = d[d.season_id == season], d[d.season_id == prev]
        lgg = pg[pg.grp == grp]
        lgr = {s: lgg[s].sum() / lgg.m.sum() for s in STATS}
        r = {}
        for s in STATS:
            prior = (old[s].sum() + OPP_PRIOR_MIN * lgr[s]) / (old.m.sum() + OPP_PRIOR_MIN)
            rate = (cur[s].sum() + OPP_PRIOR_MIN * prior) / (cur.m.sum() + OPP_PRIOR_MIN)
            r[s] = round(float(rate / lgr[s] - 1), 3)
        r["games"] = int(cur.game_id.nunique())
        out[(int(opp), grp)] = r
    return out, lg


def build(conn, season: str, rows: list, proj: dict) -> dict:
    """'game_id|player_id' -> factors for every player with a prop listed."""
    from ..models.game_model import prev_season
    prev = prev_season(season)
    pg, tg = _load(conn, (season, prev))
    if pg.empty:
        return {}
    car = pg.groupby("player_id").agg(m=("m", "sum"), ast=("ast", "sum"), reb=("reb", "sum"), pos=("position", "first"))
    grp = {int(p): _group(r.pos, r.ast / max(r.m, 1), r.reb / max(r.m, 1)) for p, r in car.iterrows()}
    pg["grp"] = pg.player_id.map(lambda p: grp.get(int(p), ("F", False))[0])
    opp_allowed, _ = _opp_allowed(pg, season, prev)

    # rebound chances: missed shots (and missed last free throws, roughly 40% of FT misses) by both teams
    tg["chances"] = tg.miss + 0.4 * tg.ftmiss
    both = tg.merge(tg[["game_id", "team_id", "chances"]].rename(columns={"team_id": "o", "chances": "oc"}), on="game_id")
    both = both[both.team_id != both.o]
    both["avail"] = both.chances + both.oc
    cur_t = both[both.season_id == season] if (both.season_id == season).sum() >= 30 else both
    lg_avail = float(cur_t.avail.mean())
    own_miss = cur_t.groupby("team_id").chances.agg(["sum", "count"])
    forced = cur_t.groupby("o").oc.agg(["sum", "count"])   # misses a team's opponents take = misses it forces
    lg_side = float(cur_t.chances.mean())

    def shrunk(tab, t, k=10):
        if t not in tab.index:
            return lg_side
        s, n = tab.loc[t, "sum"], tab.loc[t, "count"]
        return float((s + k * lg_side) / (n + k))
    avail_by_game = dict(zip(zip(both.game_id, both.team_id), both.avail))

    games = {g["game_id"]: g for g in (proj or {}).get("games", [])}
    want = {(r["game_id"], int(r["player_id"])) for r in rows}
    out = {}
    for gid, pid in want:
        g = games.get(gid)
        if not g:
            continue
        team = next((int(t) for t, v in g["teams"].items() if any(p["player_id"] == pid for p in v["players"])), None)
        if team is None:
            continue
        opp = next((int(t) for t in g["teams"] if int(t) != team), None)
        f = {}
        outs = [(p["player_id"], p["name"]) for p in g["teams"][str(team)]["players"]
                if p.get("status") in OUT_STATUSES and p["player_id"] != pid]
        if outs:
            sp = _teammate_splits(pg, pid, team, outs, season)
            if sp:
                f["teammates_out"] = sp
        gp = grp.get(pid, ("F", False))
        if opp is not None and (opp, gp[0]) in opp_allowed:
            f["opponent"] = {"team_id": opp, "group": gp[0], "group_listed": gp[1], **opp_allowed[(opp, gp[0])]}
        mine = pg[pg.player_id == pid].sort_values("game_date")
        if len(mine) >= 5:
            rec = mine.tail(THREE_RECENT)
            # rebound share: his rebounds over the chances while he was on the floor (minutes share of the game)
            av = np.array([avail_by_game.get((a, b), lg_avail) for a, b in zip(rec.game_id, rec.team_id)])
            on = (rec.m / 48.0).clip(upper=1.0).to_numpy() * av
            share = float(rec.reb.sum() / max(on.sum(), 1))
            exp_avail = shrunk(own_miss, team) + shrunk(forced, opp) if opp is not None else lg_avail
            usual = float(av.mean())
            f["rebounds"] = {"share": round(share, 3), "usual_chances": round(usual, 1),
                             "game_chances": round(exp_avail, 1), "per_48_usual": round(share * usual, 1),
                             "per_48_game": round(share * exp_avail, 1)}
            att, made = mine.fg3a.sum(), mine.fg3m.sum()
            if rec.fg3a.mean() >= 1.0:
                pct_long = (made + THREE_PRIOR_ATT * LEAGUE_3P) / (att + THREE_PRIOR_ATT)
                ra, rm = rec.fg3a.sum(), rec.fg3m.sum()
                pct_recent = (rm + THREE_PRIOR_ATT * pct_long) / (ra + THREE_PRIOR_ATT)
                f["threes"] = {"games": int(len(rec)), "att": round(float(rec.fg3a.mean()), 1),
                               "made": round(float(rec.fg3m.mean()), 2), "pct": round(float(rm / ra), 3) if ra else None,
                               "pct_expected": round(float(pct_recent), 3),
                               "expected": round(float(rec.fg3a.mean() * pct_recent), 2)}
        if f:
            out[f"{gid}|{pid}"] = f
    return out
