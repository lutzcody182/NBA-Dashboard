"""Deeper analytics from the play log (pbp_plays, pbp_poss) and lineup stints.

One file per season (teams/analytics_{season}.json) with:
  * expected shooting: shot quality and shot-making for teams (both ends) and
    players, with a calibration table for the model (models/xfg.py)
  * luck-adjusted team ratings: opponents' threes and free throws counted at
    expected rates (backtested: predicts the rest of a season better than the
    raw net rating)
  * assisted vs. self-created makes by zone
  * transition vs. half-court: possessions, points per possession, by how the
    possession began, for offense and defense
  * rebounding by where the miss came from, for teams and players
  * situations: clutch, garbage time, close and lopsided games, and how much
    starters' minutes drop in blowouts
  * usage shifts: what each teammate does with a key player on vs. off the floor
  * rotation maps: when each player is on the floor, minute by minute
  * consistency: the spread of each player's recent games
  * plain adjusted plus-minus with error margins
"""
import logging

import numpy as np
import pandas as pd
from scipy import sparse

from ..models import rapm, xfg

log = logging.getLogger(__name__)

THREE = ("c3", "ab3")
MISS_GROUPS = {"rim": "rim", "paint": "short", "mid": "mid", "c3": "three", "ab3": "three"}
TRANSITION_SECONDS = 8.0
REG_SECONDS = 2880
MIN_PLAYER_FGA = 25
KEY_PLAYERS = 6
MIN_OFF_POSS = 40


def _r(v, n=3):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, n)


def _div(a, b):
    return a / b if b else None


def prev_season(sid: str) -> str:
    y = int(sid[:4]) - 1
    return f"{y}-{str(y + 1)[-2:]}"


# -- loading ---------------------------------------------------------------------------

def _load(conn, season):
    plays = pd.read_sql_query(
        """SELECT p.*, g.game_date, g.home_team_id, g.away_team_id FROM pbp_plays p JOIN games g ON g.game_id = p.game_id
           WHERE g.season_id = ? AND g.season_type IN ('regular', 'playoffs', 'play_in')""", conn, params=[season])
    poss = pd.read_sql_query(
        """SELECT p.*, g.home_team_id, g.away_team_id FROM pbp_poss p JOIN games g ON g.game_id = p.game_id
           WHERE g.season_id = ? AND g.season_type IN ('regular', 'playoffs', 'play_in')""", conn, params=[season])
    stints = pd.read_sql_query(
        """SELECT s.*, g.game_date FROM lineup_stints s JOIN games g ON g.game_id = s.game_id
           WHERE g.season_id = ? AND g.season_type IN ('regular', 'playoffs', 'play_in')""", conn, params=[season])
    return plays, poss, stints


def _opp(df, home="home_team_id", away="away_team_id", team="team_id"):
    return np.where(df[team] == df[home], df[away], df[home])


# -- expected shooting -----------------------------------------------------------------

def _cache_get(conn, key):
    import json
    row = conn.execute("SELECT payload FROM model_calibration WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _cache_put(conn, key, payload):
    import json
    from ..util import utc_now_iso
    conn.execute("INSERT OR REPLACE INTO model_calibration (key, payload, created_at) VALUES (?, ?, ?)",
                 (key, json.dumps(payload, default=str), utc_now_iso()))
    conn.commit()


def shooting(conn, season, shots_all, day=None):
    """Expected FG% for this season's shots. The model is fit on this season and last, once a day
    (cached in model_calibration); the calibration check runs once per season."""
    if shots_all.empty:
        return shots_all, {}
    cur = shots_all[shots_all["season_id"] == season].copy()
    key = f"xfg:{season}"
    cached = _cache_get(conn, key) if conn is not None else None
    if cached and cached.get("day") == day and day is not None:
        payload, cal = cached["model"], cached.get("calibration", {})
    else:
        cal = (cached or {}).get("calibration") or xfg.calibrate(
            shots_all[shots_all["season_id"] == prev_season(season)]) or xfg.calibrate(shots_all)
        model = xfg.fit(shots_all)
        payload = xfg.to_payload(model, xfg.features(shots_all.head(5)).columns)
        if conn is not None and day is not None:
            _cache_put(conn, key, {"day": day, "model": payload, "calibration": cal})
    if cur.empty:
        return cur, cal
    cur["xfg"] = xfg.predict_payload(payload, cur)
    return cur, cal


def _shoot_block(g):
    fga = len(g)
    if not fga:
        return None
    three = g["value"] == 3
    efg = (g["made"].sum() + 0.5 * (g["made"] * three).sum()) / fga
    xefg = (g["xfg"] * g["value"] / 2).sum() / fga
    return {"fga": int(fga), "efg": _r(efg), "xefg": _r(xefg), "making": _r(efg - xefg),
            "pps": _r((g["made"] * g["value"]).sum() / fga), "xpps": _r((g["xfg"] * g["value"]).sum() / fga)}


def team_shooting(sh):
    out = {}
    sh = sh.assign(def_team=_opp(sh))
    for tid, g in sh.groupby("team_id"):
        out.setdefault(str(int(tid)), {})["off"] = _shoot_block(g)
    for tid, g in sh.groupby("def_team"):
        out.setdefault(str(int(tid)), {})["def"] = _shoot_block(g)
    return out


def player_shooting(sh):
    out = {}
    for pid, g in sh.groupby("player_id"):
        if len(g) < MIN_PLAYER_FGA or not pid:
            continue
        b = _shoot_block(g)
        # standard error of shot-making in points per shot (binomial on each shot's expected value)
        var = (g["xfg"] * (1 - g["xfg"]) * g["value"] ** 2).sum() / len(g) ** 2
        b["making_pps"] = _r((g["made"] * g["value"]).sum() / len(g) - (g["xfg"] * g["value"]).sum() / len(g))
        b["making_se"] = _r(np.sqrt(var))
        fam = {}
        for f, h in g.groupby("family"):
            fam[f] = [int(len(h)), int(h["made"].sum()), _r(h["xfg"].sum(), 1)]
        b["families"] = fam
        zones = {}
        for z, h in g.groupby("zone"):
            made = h[h["made"] == 1]
            zones[z] = [int(len(h)), int(len(made)), int(made["other_id"].notna().sum()), _r(h["xfg"].sum(), 1)]
        b["zones"] = zones
        made = g[g["made"] == 1]
        b["assisted"] = _r(made["other_id"].notna().mean()) if len(made) else None
        tr = g[(g["poss_age"] <= TRANSITION_SECONDS) & g["poss_kind"].isin(["dreb", "steal"])]
        b["transition_share"] = _r(len(tr) / len(g))
        b["putbacks"] = int(g["putback"].sum())
        b["late_clock_share"] = _r((g["poss_age"] >= 20).mean())
        out[str(int(pid))] = b
    return out


def league_shooting(sh):
    fam = {}
    for f, g in sh.groupby("family"):
        fam[f] = {"share": _r(len(g) / len(sh)), "fg": _r(g["made"].mean()), "xfg": _r(g["xfg"].mean()),
                  "avg_dist": _r(g["dist"].mean(), 1)}
    return {"fga": int(len(sh)), "efg": _shoot_block(sh)["efg"], "families": fam,
            "assisted": _r(sh.loc[sh["made"] == 1, "other_id"].notna().mean()),
            "zones_assisted": {z: _r(g.loc[g["made"] == 1, "other_id"].notna().mean())
                               for z, g in sh.groupby("zone")}}


# -- luck-adjusted ratings ---------------------------------------------------------------

def luck(plays, poss, sh):
    """Per team: actual ratings, and ratings with opponents' threes and free throws at expected rates."""
    if poss.empty:
        return {}, {}
    ft = plays[(plays["kind"] == "ft") & (plays["tech"] == 0)]
    lg_ft = float(ft["made"].mean()) if len(ft) else 0.78
    p = poss.groupby(["game_id", "team_id"]).agg(poss=("n", "size"), pts=("pts", "sum")).reset_index()
    three = sh[sh["value"] == 3].groupby(["game_id", "team_id"]).agg(m3=("made", "sum"), x3=("xfg", "sum"))
    f = ft.groupby(["game_id", "team_id"]).agg(ftm=("made", "sum"), fta=("made", "size"))
    p = p.merge(three, on=["game_id", "team_id"], how="left").merge(f, on=["game_id", "team_id"], how="left").fillna(0)
    o = p.rename(columns={c: "o_" + c for c in p.columns if c != "game_id"})
    t = p.merge(o, on="game_id")
    t = t[t["team_id"] != t["o_team_id"]]
    t["opp_luck"] = 3 * (t["o_m3"] - t["o_x3"]) + (t["o_ftm"] - lg_ft * t["o_fta"])
    t["own_luck"] = 3 * (t["m3"] - t["x3"]) + (t["ftm"] - lg_ft * t["fta"])
    out = {}
    for tid, g in t.groupby("team_id"):
        n = (g["poss"].sum() + g["o_poss"].sum()) / 2
        ortg = 100 * g["pts"].sum() / g["poss"].sum()
        drtg = 100 * g["o_pts"].sum() / g["o_poss"].sum()
        adj_d = 100 * (g["o_pts"].sum() - g["opp_luck"].sum()) / g["o_poss"].sum()
        out[str(int(tid))] = {
            "games": int(len(g)), "ortg": _r(ortg, 1), "drtg": _r(drtg, 1), "net": _r(ortg - drtg, 1),
            "adj_drtg": _r(adj_d, 1), "adj_net": _r(ortg - adj_d, 1),
            "opp_three_luck": _r(100 * 3 * (g["o_m3"].sum() - g["o_x3"].sum()) / g["o_poss"].sum(), 1),
            "opp_ft_luck": _r(100 * (g["o_ftm"].sum() - lg_ft * g["o_fta"].sum()) / g["o_poss"].sum(), 1),
            "own_three_luck": _r(100 * 3 * (g["m3"].sum() - g["x3"].sum()) / g["poss"].sum(), 1),
            "poss_per_game": _r(n / len(g), 1),
        }
    return out, {"league_ft": _r(lg_ft)}


# -- transition ----------------------------------------------------------------------------

def _poss_class(poss):
    tr = poss["kind"].isin(["dreb", "steal"]) & (poss["first_age"].fillna(99) <= TRANSITION_SECONDS)
    return np.where(tr, "transition", "halfcourt")


def transition(poss):
    if poss.empty:
        return {}, {}
    poss = poss.assign(cls=_poss_class(poss), def_team=_opp(poss))
    lg = {"transition_share": _r((poss["cls"] == "transition").mean()),
          "ppp": {c: _r(g["pts"].mean()) for c, g in poss.groupby("cls")},
          "by_start": {k: [int(len(g)), _r(g["pts"].mean())] for k, g in poss.groupby("kind")}}

    def block(g):
        tr, hc = g[g["cls"] == "transition"], g[g["cls"] == "halfcourt"]
        return {"poss": int(len(g)), "share": _r(len(tr) / len(g)), "ppp_tr": _r(tr["pts"].mean()),
                "ppp_hc": _r(hc["pts"].mean()), "ppp": _r(g["pts"].mean()),
                "after_steal": [int((g["kind"] == "steal").sum()), _r(g.loc[g["kind"] == "steal", "pts"].mean())],
                "after_dreb": [int((g["kind"] == "dreb").sum()), _r(g.loc[g["kind"] == "dreb", "pts"].mean())],
                "after_made": [int(g["kind"].isin(["made", "ft"]).sum()),
                               _r(g.loc[g["kind"].isin(["made", "ft"]), "pts"].mean())],
                "second_chance": _r((g["oreb"] > 0).mean())}
    out = {}
    for tid, g in poss.groupby("team_id"):
        out.setdefault(str(int(tid)), {})["off"] = block(g)
    for tid, g in poss.groupby("def_team"):
        out.setdefault(str(int(tid)), {})["def"] = block(g)
    return out, lg


# -- lineups on the floor for each play ---------------------------------------------------

def _stint_players(stints):
    """One row per (game, stint, team, player) with seconds and possessions."""
    s = stints[["game_id", "stint_num", "team_id", "lineup_key", "start_sec", "end_sec", "poss", "opp_poss",
                "period", "margin_start"]].copy()
    s["player_id"] = s["lineup_key"].str.split("-")
    s = s.explode("player_id")
    s["player_id"] = s["player_id"].astype(int)
    s["secs"] = s["end_sec"] - s["start_sec"]
    return s


# -- rebounding by miss location --------------------------------------------------------

def rebounding(plays, sp):
    misses = plays[(plays["kind"] == "fg") & (plays["made"] == 0) & plays["reb_off"].notna()].copy()
    if misses.empty:
        return {}, {}, {}
    misses["grp"] = misses["zone"].map(MISS_GROUPS).fillna("mid")
    misses["def_team"] = _opp(misses)
    lg = {g: _r(h["reb_off"].mean()) for g, h in misses.groupby("grp")}
    lg["all"] = _r(misses["reb_off"].mean())
    teams = {}
    for tid, g in misses.groupby("team_id"):
        teams.setdefault(str(int(tid)), {})["off"] = {
            k: [int(len(h)), _r(h["reb_off"].mean())] for k, h in [("all", g), *g.groupby("grp")]}
    for tid, g in misses.groupby("def_team"):
        teams.setdefault(str(int(tid)), {})["def"] = {
            k: [int(len(h)), _r(1 - h["reb_off"].mean())] for k, h in [("all", g), *g.groupby("grp")]}
    # players: misses while on the floor (offense = his team shot it), and how many he grabbed
    on = sp[["game_id", "stint_num", "team_id", "player_id"]]
    m_off = misses.merge(on, on=["game_id", "stint_num", "team_id"], suffixes=("", "_on"))
    m_def = misses.merge(on.rename(columns={"team_id": "def_team"}), on=["game_id", "stint_num", "def_team"],
                         suffixes=("", "_on"))
    players = {}
    for side, m in (("off", m_off), ("def", m_def)):
        m = m.assign(got=(m["reb_player"] == m["player_id_on"]).astype(int))
        agg = m.groupby(["player_id_on", "grp"]).agg(ch=("got", "size"), got=("got", "sum")).reset_index()
        tot = m.groupby("player_id_on").agg(ch=("got", "size"), got=("got", "sum"))
        for pid, r in tot.iterrows():
            if r["ch"] < 40:
                continue
            d = players.setdefault(str(int(pid)), {})
            d[side] = {"all": [int(r["ch"]), int(r["got"])]}
        for r in agg.itertuples():
            d = players.get(str(int(r.player_id_on)))
            if d and side in d:
                d[side][r.grp] = [int(r.ch), int(r.got)]
    return teams, players, lg


# -- situations -----------------------------------------------------------------------------

def _garbage_stints(stints, conn, season):
    """Stints in garbage time: 4th quarter, a big lead for the time left, and at most two starters on the floor."""
    st = conn.execute(
        """SELECT p.game_id, p.player_id FROM player_game_pbp p JOIN games g ON g.game_id = p.game_id
           WHERE g.season_id = ? AND p.starter = 1""", (season,)).fetchall()
    starters = {}
    for gid, pid in st:
        starters.setdefault(gid, set()).add(int(pid))
    s = stints.copy()
    left = REG_SECONDS - s["start_sec"]
    lead = s["margin_start"].abs()
    big = (s["period"] == 4) & (((left <= 720) & (left > 540) & (lead >= 25)) | ((left <= 540) & (left > 360) & (lead >= 20))
                                | ((left <= 360) & (lead >= 10)))
    keys = (s["lineup_key"] + "-" + s["opp_lineup_key"]).str.split("-")
    n_start = [sum(int(p) in starters.get(g, ()) for p in k) for g, k in zip(s["game_id"], keys)]
    s["garbage"] = big & (np.array(n_start) <= 2)
    return s[["game_id", "stint_num", "garbage"]].drop_duplicates(["game_id", "stint_num"])


def situations(conn, season, plays, poss, stints, sp):
    if poss.empty or stints.empty:
        return {}, {}, {}
    garb = _garbage_stints(stints, conn, season)
    gset = set(map(tuple, garb.loc[garb["garbage"], ["game_id", "stint_num"]].to_numpy()))
    poss = poss.assign(def_team=_opp(poss))
    poss["garbage"] = [(g, s) in gset for g, s in zip(poss["game_id"], poss["stint_num"])]
    left = np.where(poss["period"] >= 5, 0, REG_SECONDS - poss["start_t"])
    poss["clutch"] = (poss["period"] >= 4) & ((poss["period"] >= 5) | (left <= 300)) & (poss["margin"].abs() <= 5)
    teams = {}
    for tid, g in poss.groupby("team_id"):
        d = poss[poss["def_team"] == tid]
        c_o, c_d = g[g["clutch"]], d[d["clutch"]]
        ng_o, ng_d = g[~g["garbage"]], d[~d["garbage"]]
        teams[str(int(tid))] = {
            "clutch": {"poss": int(len(c_o)), "ortg": _r(100 * c_o["pts"].mean(), 1) if len(c_o) else None,
                       "drtg": _r(100 * c_d["pts"].mean(), 1) if len(c_d) else None},
            "garbage_share": _r(g["garbage"].mean()),
            "net_no_garbage": _r(100 * (ng_o["pts"].mean() - ng_d["pts"].mean()), 1),
            "net": _r(100 * (g["pts"].mean() - d["pts"].mean()), 1),
        }
    # players: per-game production with and without garbage time, and in the clutch
    pl = plays[plays["player_id"].notna()].copy()
    pl["garbage"] = [(g, s) in gset for g, s in zip(pl["game_id"], pl["stint_num"])]
    pl["pts"] = np.where(pl["kind"].isin(["fg", "ft"]), pl["made"].fillna(0) * pl["value"].fillna(0), 0)
    pl["reb"] = (pl["kind"] == "reb").astype(int)
    pl["fga"] = (pl["kind"] == "fg").astype(int)
    pl["tov"] = (pl["kind"] == "tov").astype(int)
    ast = plays[(plays["kind"] == "fg") & (plays["made"] == 1) & plays["other_id"].notna()][
        ["game_id", "stint_num", "other_id", "period", "t", "margin"]].rename(columns={"other_id": "player_id"})
    ast = ast.assign(ast=1, garbage=[(g, s) in gset for g, s in zip(ast["game_id"], ast["stint_num"])])
    pl["ast"] = 0
    left = np.where(pl["period"] >= 5, 0, REG_SECONDS - pl["t"])
    pl["clutch"] = (pl["period"] >= 4) & ((pl["period"] >= 5) | (left <= 300)) & (pl["margin"].abs() <= 5)
    al = np.where(ast["period"] >= 5, 0, REG_SECONDS - ast["t"])
    ast["clutch"] = (ast["period"] >= 4) & ((ast["period"] >= 5) | (al <= 300)) & (ast["margin"].abs() <= 5)
    cols = ["pts", "reb", "ast", "fga", "tov"]
    both = pd.concat([pl[["game_id", "player_id", "garbage", "clutch"] + cols],
                      ast.assign(pts=0, reb=0, fga=0, tov=0)[["game_id", "player_id", "garbage", "clutch"] + cols]])
    both["player_id"] = both["player_id"].astype(int)
    sp = sp.merge(garb, on=["game_id", "stint_num"], how="left")
    sp["garbage"] = sp["garbage"].fillna(False).astype(bool)
    mins_all = sp.groupby("player_id")["secs"].sum() / 60
    mins_ng = sp[~sp["garbage"]].groupby("player_id")["secs"].sum() / 60
    games = sp.groupby("player_id")["game_id"].nunique()
    tot_all = both.groupby("player_id")[cols].sum()
    tot_ng = both[~both["garbage"]].groupby("player_id")[cols].sum()
    tot_cl = both[both["clutch"]].groupby("player_id")[cols].sum()
    fga_cl = pl[pl["clutch"] & (pl["kind"] == "fg")].groupby("player_id")["made"].agg(["size", "sum"])
    # clutch minutes: stint time inside the last 5 minutes of the 4th/OT with a close score at the stint's start
    sc = sp.copy()
    lo = np.where(sc["period"] >= 5, sc["start_sec"], np.maximum(sc["start_sec"], REG_SECONDS - 300))
    hi = np.where(sc["period"] >= 5, sc["end_sec"], np.minimum(sc["end_sec"], REG_SECONDS))
    sc["csecs"] = np.where((sc["period"] >= 4) & (sc["margin_start"].abs() <= 5), np.maximum(hi - lo, 0), 0)
    cmins = sc.groupby("player_id")["csecs"].sum() / 60
    players = {}
    for pid in games.index:
        g = int(games[pid])
        if g < 3:
            continue
        a, n = tot_all.loc[pid] if pid in tot_all.index else None, tot_ng.loc[pid] if pid in tot_ng.index else None
        if a is None:
            continue
        d = {"games": g, "min": _r(mins_all.get(pid, 0) / g, 1), "min_ng": _r(mins_ng.get(pid, 0) / g, 1),
             "all": {c: _r(a[c] / g, 1) for c in cols},
             "no_garbage": {c: _r((n[c] if n is not None else 0) / g, 1) for c in cols}}
        cm = float(cmins.get(pid, 0))
        if pid in tot_cl.index and cm >= 5:
            c = tot_cl.loc[pid]
            f = fga_cl.loc[pid] if pid in fga_cl.index else None
            d["clutch"] = {"min": _r(cm, 1), **{k: _r(c[k] * 36 / cm, 1) for k in cols},
                           "fg": [int(f["sum"]), int(f["size"])] if f is not None else [0, 0]}
        players[str(int(pid))] = d
    return teams, players, {"garbage_rule": "4th quarter; lead of 25+ with 9-12 minutes left, 20+ with 6-9, "
                                            "10+ in the last 6; at most two starters on the floor",
                            "clutch_rule": "last 5 minutes of the 4th quarter or overtime, score within 5"}


def close_and_blowout(conn, season):
    """Per player: per-36 in close games (final margin <= 5) vs lopsided ones (>= 15); and how
    much starters' minutes drop by final margin (both seasons, for the blowout-risk estimate)."""
    df = pd.read_sql_query(
        """SELECT s.player_id, s.game_id, s.started, s.seconds, s.pts, s.oreb + s.dreb AS reb, s.ast, g.season_id,
                  ABS(g.home_score - g.away_score) AS fm FROM player_game_stats s JOIN games g ON g.game_id = s.game_id
           WHERE g.season_id IN (?, ?) AND g.season_type = 'regular' AND g.status = 'final' AND s.seconds > 0""",
        conn, params=[season, prev_season(season)])
    if df.empty:
        return {}, {}
    st = df[df["started"] == 1].copy()
    st["avg"] = st.groupby(["player_id", "season_id"])["seconds"].transform("mean")
    st["bucket"] = pd.cut(st["fm"], [-1, 9, 14, 19, 24, 99], labels=["0-9", "10-14", "15-19", "20-24", "25+"])
    cut = st.groupby("bucket", observed=True).apply(
        lambda g: _r((g["seconds"] - g["avg"]).mean() / 60, 1), include_groups=False).to_dict()
    share = st.groupby("bucket", observed=True)["game_id"].nunique()
    blow = {"starter_min_change": {str(k): v for k, v in cut.items()},
            "games": {str(k): int(v) for k, v in share.items()}}
    cur = df[df["season_id"] == season]
    out = {}
    for pid, g in cur.groupby("player_id"):
        close, lop = g[g["fm"] <= 5], g[g["fm"] >= 15]
        if len(g) < 3:
            continue

        def p36(h):
            m = h["seconds"].sum() / 60
            return None if m < 20 else {"games": int(len(h)), "min": _r(m / len(h), 1),
                                        **{k: _r(h[k].sum() * 36 / m, 1) for k in ("pts", "reb", "ast")}}
        out[str(int(pid))] = {"close": p36(close), "lopsided": p36(lop)}
    return out, blow


# -- usage shifts ---------------------------------------------------------------------------

def usage_shifts(plays, sp, stints):
    """For each team's key players: every teammate's per-100 numbers with that player on vs. off."""
    pl = plays[plays["player_id"].notna() & (plays["tech"] == 0)].copy()
    pl["player_id"] = pl["player_id"].astype(int)
    pl["fga"] = (pl["kind"] == "fg").astype(int)
    pl["fta"] = (pl["kind"] == "ft").astype(int)
    pl["tov"] = (pl["kind"] == "tov").astype(int)
    pl["reb"] = (pl["kind"] == "reb").astype(int)
    pl["pts"] = np.where(pl["kind"].isin(["fg", "ft"]), pl["made"].fillna(0) * pl["value"].fillna(0), 0)
    ast = plays[(plays["kind"] == "fg") & (plays["made"] == 1) & plays["other_id"].notna()]
    ast = ast[["game_id", "stint_num", "team_id", "other_id"]].rename(columns={"other_id": "player_id"})
    ast = ast.assign(player_id=ast["player_id"].astype(int), ast=1)
    per = pl.groupby(["game_id", "stint_num", "player_id"])[["fga", "fta", "tov", "reb", "pts"]].sum()
    per = per.join(ast.groupby(["game_id", "stint_num", "player_id"])["ast"].sum(), how="outer").fillna(0).reset_index()
    base = sp.merge(per, on=["game_id", "stint_num", "player_id"], how="left").fillna(
        {"fga": 0, "fta": 0, "tov": 0, "reb": 0, "pts": 0, "ast": 0})
    key = stints[["game_id", "stint_num", "team_id", "lineup_key"]]
    out = {}
    for tid, b in base.groupby("team_id"):
        mins = b.groupby("player_id")["secs"].sum().sort_values(ascending=False)
        keys = list(mins.index[:KEY_PLAYERS])
        on_sets = key[key["team_id"] == tid].set_index(["game_id", "stint_num"])["lineup_key"].str.split("-").map(
            lambda l: {int(x) for x in l})
        b = b.join(on_sets.rename("five"), on=["game_id", "stint_num"])
        rows = []
        for x in keys:
            has_x = b["five"].map(lambda s, x=x: x in s if isinstance(s, set) else False)
            mates = []
            for y in mins.index[:10]:
                if y == x:
                    continue
                yb = b[b["player_id"] == y]
                hx = has_x[yb.index]
                on, off = yb[hx], yb[~hx]
                if off["poss"].sum() < MIN_OFF_POSS or on["poss"].sum() < MIN_OFF_POSS:
                    continue

                def rates(h):
                    p = h["poss"].sum()
                    return {"poss": int(p), "min": _r(h["secs"].sum() / 60, 0),
                            "usg": _r(100 * (h["fga"].sum() + 0.44 * h["fta"].sum() + h["tov"].sum()) / p, 1),
                            **{k: _r(100 * h[k].sum() / p, 1) for k in ("pts", "fga", "fta", "ast", "reb", "tov")}}
                mates.append({"player_id": int(y), "on": rates(on), "off": rates(off)})
            if mates:
                team_off = b[~has_x].drop_duplicates(["game_id", "stint_num"])
                rows.append({"player_id": int(x), "off_poss": int(team_off["poss"].sum()), "mates": mates})
        out[str(int(tid))] = rows
    return out


# -- rotations ------------------------------------------------------------------------------

def rotations(sp, stints, n_recent=5):
    out = {}
    if sp.empty:
        return out
    games_by_team = stints.groupby("team_id")["game_id"].nunique()
    for tid, b in sp.groupby("team_id"):
        n_games = int(games_by_team.get(tid, 0)) or 1
        share = {}
        for r in b.itertuples():
            lo, hi = float(r.start_sec), float(r.end_sec)
            m = int(lo // 60)
            while m * 60 < hi and m < 63:
                a, z = max(lo, m * 60), min(hi, (m + 1) * 60)
                if z > a:
                    share.setdefault(r.player_id, np.zeros(64))[m] += (z - a) / 60
                m += 1
        mins = b.groupby("player_id")["secs"].sum().sort_values(ascending=False)
        players = []
        for pid in mins.index:
            if mins[pid] / 60 / n_games < 3:
                continue
            arr = share.get(pid, np.zeros(64))[:48] / n_games
            players.append({"player_id": int(pid), "mpg": _r(mins[pid] / 60 / n_games, 1),
                            "bins": [round(float(v), 2) for v in arr]})
        recent = []
        dates = stints[stints["team_id"] == tid][["game_id", "game_date", "opp_id"]].drop_duplicates("game_id")
        for g in dates.sort_values("game_date").tail(n_recent).itertuples():
            gb = b[b["game_id"] == g.game_id].sort_values("start_sec")
            iv = {}
            for r in gb.itertuples():
                cur = iv.setdefault(int(r.player_id), [])
                if cur and abs(cur[-1][1] - r.start_sec) < 0.5:
                    cur[-1][1] = round(float(r.end_sec), 1)
                else:
                    cur.append([round(float(r.start_sec), 1), round(float(r.end_sec), 1)])
            end = float(gb["end_sec"].max()) if len(gb) else REG_SECONDS
            recent.append({"game_id": g.game_id, "date": g.game_date, "opp": int(g.opp_id), "end": round(end, 1),
                           "players": [{"player_id": p, "on": v} for p, v in
                                       sorted(iv.items(), key=lambda kv: -sum(b - a for a, b in kv[1]))]})
        out[str(int(tid))] = {"games": n_games, "players": players, "recent": recent}
    return out


# -- consistency ----------------------------------------------------------------------------

LINES = {"pts": (10, 15, 20, 25, 30), "reb": (4, 6, 8, 10, 12), "ast": (2, 4, 6, 8, 10), "fg3m": (1, 2, 3, 4, 5),
         "pra": (20, 25, 30, 35, 40)}


def consistency(conn, season, last_n=30):
    df = pd.read_sql_query(
        """SELECT s.player_id, g.game_date, s.seconds / 60.0 AS min, s.pts, s.oreb + s.dreb AS reb, s.ast, s.fg3m
           FROM player_game_stats s JOIN games g ON g.game_id = s.game_id
           WHERE g.season_id IN (?, ?) AND g.season_type = 'regular' AND g.status = 'final' AND s.seconds > 0""",
        conn, params=[season, prev_season(season)])
    cur = set(pd.read_sql_query(
        """SELECT DISTINCT s.player_id FROM player_game_stats s JOIN games g ON g.game_id = s.game_id
           WHERE g.season_id = ? AND s.seconds > 0""", conn, params=[season])["player_id"])
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    out = {}
    for pid, g in df.sort_values("game_date").groupby("player_id"):
        if pid not in cur:
            continue
        g = g.tail(last_n)
        if len(g) < 8:
            continue
        d = {"games": int(len(g)), "from": g["game_date"].iloc[0], "min": {
            "mean": _r(g["min"].mean(), 1), "sd": _r(g["min"].std(), 1)}}
        for k in ("pts", "reb", "ast", "fg3m", "pra"):
            v = g[k].to_numpy(float)
            q = np.percentile(v, [10, 25, 50, 75, 90])
            d[k] = {"mean": _r(v.mean(), 1), "sd": _r(v.std(ddof=1), 1), "q": [_r(x, 1) for x in q],
                    "over": {str(t): _r((v > t - 0.5).mean(), 2) for t in LINES[k]}}
        out[str(int(pid))] = d
    return out


# -- plain adjusted plus-minus --------------------------------------------------------------

def plain_apm(stints, lam=3000.0, min_poss=300):
    st = stints[(stints["ok"] == 1) & (stints["poss"] > 0)]
    if len(st) < 2000:
        return {}
    f = rapm.fit(st, np.ones(len(st)), lam=lam)
    y, w, off, dfn, home = rapm._rows(st, np.ones(len(st)))
    index = {int(p): i for i, p in enumerate(f.players)}
    X = rapm._design(off, dfn, home, index)
    pred = rapm.predict(f, st[st["poss"] > 0])
    s2 = float(np.mean(w * (y - pred) ** 2))
    W = sparse.diags(w)
    pen = np.full(X.shape[1], lam)
    pen[-1] = 1e-3
    A = (X.T @ W @ X + sparse.diags(pen)).toarray()
    var = np.diag(np.linalg.inv(A)) * s2
    n = len(f.players)
    poss = {}
    for k, p in zip(st["lineup_key"], st["poss"]):
        for x in k.split("-"):
            poss[int(x)] = poss.get(int(x), 0) + p
    out = {}
    for i, pid in enumerate(f.players):
        if poss.get(int(pid), 0) < min_poss:
            continue
        o, d = float(f.off[i]), float(-f.deff[i])
        se = float(np.sqrt(var[i] + var[n + i]))
        out[str(int(pid))] = {"o": _r(o, 2), "d": _r(d, 2), "net": _r(o + d, 2), "net_moe": _r(1.96 * se, 2),
                              "o_moe": _r(1.96 * np.sqrt(var[i]), 2), "d_moe": _r(1.96 * np.sqrt(var[n + i]), 2),
                              "poss": int(poss[int(pid)])}
    return {"lambda": lam, "players": out}


# -- build ------------------------------------------------------------------------------------

def build(conn, season, day=None) -> dict:
    plays, poss, stints = _load(conn, season)
    out = {"season": season, "games": int(plays["game_id"].nunique()) if len(plays) else 0}
    if plays.empty:
        return out
    shots_all = xfg.shots(conn, [prev_season(season), season])
    sh, cal = shooting(conn, season, shots_all, day)
    out["model"] = cal
    sp = _stint_players(stints) if len(stints) else pd.DataFrame()
    steps = {
        "league_shooting": lambda: league_shooting(sh),
        "team_shooting": lambda: team_shooting(sh),
        "player_shooting": lambda: player_shooting(sh),
        "luck": lambda: luck(plays, poss, sh),
        "transition": lambda: transition(poss),
        "rebounding": lambda: rebounding(plays, sp),
        "situations": lambda: situations(conn, season, plays, poss, stints, sp),
        "close_blowout": lambda: close_and_blowout(conn, season),
        "usage": lambda: usage_shifts(plays, sp, stints),
        "rotations": lambda: rotations(sp, stints),
        "consistency": lambda: consistency(conn, season),
        "apm": lambda: plain_apm(stints),
    }
    res, errors = {}, []
    for name, fn in steps.items():
        try:
            res[name] = fn()
        except Exception as exc:     # one failing piece should not take the rest down
            log.exception("analytics %s failed", name)
            errors.append(f"{name}: {exc}")
            res[name] = None
    teams, players = {}, {}

    def put(dst, key, part):
        for k, v in (part or {}).items():
            dst.setdefault(k, {})[key] = v
    put(teams, "shooting", res["team_shooting"])
    lk = res["luck"] or ({}, {})
    put(teams, "luck", lk[0])
    tr = res["transition"] or ({}, {})
    put(teams, "transition", tr[0])
    rb = res["rebounding"] or ({}, {}, {})
    put(teams, "rebounding", rb[0])
    si = res["situations"] or ({}, {}, {})
    put(teams, "situations", si[0])
    put(teams, "usage", res["usage"])
    put(teams, "rotation", res["rotations"])
    put(players, "shooting", res["player_shooting"])
    put(players, "rebounding", rb[1])
    put(players, "situations", si[1])
    cb = res["close_blowout"] or ({}, {})
    put(players, "close", cb[0])
    put(players, "consistency", res["consistency"])
    put(players, "apm", (res["apm"] or {}).get("players"))
    out.update({
        "league": {"shooting": res["league_shooting"], "luck": lk[1], "transition": tr[1], "rebounding": rb[2],
                   "situations": si[2], "blowout": cb[1], "apm_lambda": (res["apm"] or {}).get("lambda")},
        "teams": teams, "players": players, "errors": errors,
    })
    return out


def write_files(conn, out_dir, season, write, day) -> list:
    """teams/analytics_{sid}.json (league + teams) and players/analytics_{sid}.json, for this
    season and last. Last season's is built once and kept in model_calibration."""
    files = []
    for sid in (prev_season(season), season):
        if sid != season:
            data = _cache_get(conn, f"analytics:{sid}")
            if not data or not data.get("games"):
                data = build(conn, sid, day)
                if data.get("games"):
                    _cache_put(conn, f"analytics:{sid}", data)
        else:
            data = build(conn, sid, day)
        if not data.get("games"):
            continue
        players = data.pop("players", {})
        write(out_dir / "teams" / f"analytics_{sid}.json", data)
        write(out_dir / "players" / f"analytics_{sid}.json", {"season": sid, "players": players})
        files += [f"teams/analytics_{sid}.json", f"players/analytics_{sid}.json"]
    return files

