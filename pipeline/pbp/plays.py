"""A play-by-play log with context: every shot, free throw, turnover and
rebound, and every possession.

Runs on the same NBA-style actions as parse.py, after parse_actions has
marked each action with its game time ("_t") and lineup stint ("_stint").
Both sources work: the NBA archive (past seasons) and ESPN (this season,
converted in providers/espn.py).

Possessions are tracked from the events themselves:
  * a made basket or a made last free throw hands the ball over ("made", "ft")
  * a defensive rebound starts a possession ("dreb")
  * a turnover starts one for the other team: "steal" if it was stolen
    (a live ball), otherwise "deadball"
  * the first possession of each period is "period"
An offensive rebound continues the same possession. And-one free throws are
credited to the possession that scored the basket. Technical free throws
belong to no possession.

The context saved with each shot (how and how long ago the possession began,
whether it is a putback, the score, the lineup on the floor) is what the
expected-shooting, transition and situation analyses use.
"""
import math
import re

from .parse import ZONES, _clock_seconds, _int, period_length, period_start

PUTBACK_SECONDS = 3.0


def shot_family(a) -> str:
    """dunk, tip, layup, hook, floater, jumper, pullup, stepback, fadeaway (or '' for a heave)."""
    s = " ".join(str(a.get(k) or "") for k in ("espnType", "subType", "descriptor")).lower()
    if "heave" in s:
        return ""
    if "dunk" in s:
        return "dunk"
    if "tip" in s or "putback" in s:
        return "tip"
    if "hook" in s:
        return "hook"
    if "layup" in s or "finger roll" in s:
        return "layup"
    if "float" in s:
        return "floater"
    if "step back" in s or "stepback" in s:
        return "stepback"
    if "fade" in s or "turnaround" in s:
        return "fadeaway"
    if "pullup" in s or "pull-up" in s or "pull up" in s:
        return "pullup"
    return "jumper"


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def shot_xy(a):
    """(x, y, distance) in feet, basket at (0, 0), y out toward half court."""
    x = y = None
    if _num(a.get("xFt")) is not None and _num(a.get("yFt")) is not None:      # ESPN
        x, y = _num(a["xFt"]) - 25.0, _num(a["yFt"])
    elif _num(a.get("xLegacy")) is not None and _num(a.get("yLegacy")) is not None:  # NBA (tenths of a foot)
        x, y = _num(a["xLegacy"]) / 10.0, _num(a["yLegacy"]) / 10.0
    dist = _num(a.get("shotDistance"))
    if dist is None:
        m = re.search(r"(\d+)-foot", str(a.get("description") or ""))
        if m:
            dist = float(m.group(1))
    if dist is None and x is not None:
        dist = math.hypot(x, y)
    return x, y, dist


def _trip(a):
    """(this free throw's number, trip length) or (None, None)."""
    m = re.search(r"(\d)\s*of\s*(\d)", str(a.get("subType") or ""))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _is_tech_ft(a) -> bool:
    s = (str(a.get("subType") or "") + " " + str(a.get("description") or "")).lower()
    return "technical" in s


def extract(game_id: str, actions: list, home_id: int, away_id: int):
    """(plays, possessions): lists of dicts matching pbp_plays and pbp_poss."""
    home_id, away_id = int(home_id), int(away_id)
    teams = (home_id, away_id)
    other = {home_id: away_id, away_id: home_id}
    acts = sorted(actions, key=lambda a: (_int(a.get("period")), _int(a.get("orderNumber")) or _int(a.get("actionNumber"))))
    plays, poss = [], []
    score = {home_id: 0, away_id: 0}
    st = {"cur": None, "period": 0, "last_miss": None, "last_oreb": None, "last_tov": None, "last_fg": None}

    def T(a):
        if a.get("_t") is not None:
            return float(a["_t"])
        per = _int(a.get("period"))
        return period_start(per) + period_length(per) - _clock_seconds(a.get("clock"))

    def start(team, t, kind, a):
        cur = st["cur"]
        if cur is not None:
            cur["end_t"] = t
        cur = {"game_id": game_id, "n": len(poss) + 1, "team_id": team, "period": _int(a.get("period")),
               "start_t": t, "end_t": t, "kind": kind, "pts": 0, "fga": 0, "fta": 0, "tov": 0, "oreb": 0,
               "first_age": None, "stint_num": a.get("_stint"), "margin": score[team] - score[other[team]]}
        poss.append(cur)
        st["cur"] = cur
        st["last_oreb"] = None
        return cur

    def owner(team, t, a):
        cur = st["cur"]
        if cur is not None and cur["team_id"] == team:
            return cur
        kind = "period" if cur is None else "unknown"
        p_start = period_start(_int(a.get("period")))
        return start(team, p_start if cur is None else t, kind, a)

    def touch(cur, t):
        if cur["first_age"] is None:
            cur["first_age"] = round(t - cur["start_t"], 1)

    def row(a, t, team, kind, **kw):
        r = {"game_id": game_id, "n": len(plays) + 1, "period": _int(a.get("period")), "t": round(t, 1),
             "team_id": team, "player_id": _int(a.get("personId")) or None, "kind": kind, "value": None,
             "made": None, "x": None, "y": None, "dist": None, "zone": None, "family": None, "other_id": None,
             "blocked": 0, "poss_n": None, "poss_age": None, "poss_kind": None, "putback": 0,
             "margin": score[team] - score[other[team]], "stint_num": a.get("_stint"), "reb_off": None,
             "reb_player": None, "tech": 0}
        r.update(kw)
        plays.append(r)
        return r

    def with_poss(r, cur, t):
        r.update(poss_n=cur["n"], poss_age=round(t - cur["start_t"], 1), poss_kind=cur["kind"])

    for a in acts:
        kind = a.get("actionType")
        per = _int(a.get("period"))
        if per != st["period"]:
            if st["cur"] is not None and st["period"]:
                st["cur"]["end_t"] = period_start(st["period"]) + period_length(st["period"])
            st.update(cur=None, period=per, last_miss=None, last_oreb=None, last_tov=None, last_fg=None)
        if kind in ("period", "game"):
            if st["cur"] is not None:
                st["cur"]["end_t"] = period_start(per) + period_length(per)
            continue
        t = T(a)
        team = _int(a.get("teamId"))

        if kind in ("2pt", "3pt"):
            if team not in teams:
                continue
            fam = shot_family(a)
            made = a.get("shotResult") == "Made"
            cur = owner(team, t, a)
            cur["fga"] += 1
            touch(cur, t)
            x, y, dist = shot_xy(a)
            zone = a.get("zone") or ZONES.get(a.get("area"))
            lo = st["last_oreb"]
            r = row(a, t, team, "fg", value=3 if kind == "3pt" else 2, made=int(made),
                    x=None if x is None else round(x, 1), y=None if y is None else round(y, 1),
                    dist=None if dist is None else round(dist, 1), zone=zone, family=fam or "heave",
                    other_id=_int(a.get("assistPersonId")) or None,
                    putback=int(lo is not None and lo[0] == cur["n"] and t - lo[1] <= PUTBACK_SECONDS))
            with_poss(r, cur, t)
            st["last_fg"] = r
            if made:
                pts = 3 if kind == "3pt" else 2
                cur["pts"] += pts
                score[team] += pts
                st["last_miss"] = None
                start(other[team], t, "made", a)
            else:
                st["last_miss"] = r
        elif kind == "block":
            lf = st["last_fg"]
            if lf is not None and abs(lf["t"] - t) < 1.5 and not lf["made"]:
                lf["blocked"] = 1
        elif kind == "freethrow":
            if team not in teams:
                continue
            made = a.get("shotResult") == "Made"
            if _is_tech_ft(a):
                row(a, t, team, "ft", value=1, made=int(made), tech=1)
                score[team] += int(made)
                continue
            cur = st["cur"]
            prev = poss[-2] if len(poss) >= 2 else None
            and_one = (cur is not None and cur["team_id"] != team and prev is not None and prev["team_id"] == team
                       and cur["fga"] == cur["fta"] == cur["tov"] == 0 and t - prev["end_t"] <= 1.0)
            target = prev if and_one else owner(team, t, a)
            target["fta"] += 1
            touch(target, t)
            r = row(a, t, team, "ft", value=1, made=int(made))
            with_poss(r, target, t)
            if made:
                target["pts"] += 1
                score[team] += 1
            k, n = _trip(a)
            last = k is None or k == n
            if last and not and_one:
                if made:
                    st["last_miss"] = None
                    start(other[team], t, "ft", a)
                else:
                    st["last_miss"] = r
            elif last and and_one and not made:
                st["last_miss"] = r
        elif kind == "rebound":
            if team not in teams:
                continue
            off = a.get("subType") == "offensive"
            lm = st["last_miss"]
            if lm is not None:
                lm["reb_off"] = int(lm["team_id"] == team)
                lm["reb_player"] = _int(a.get("personId")) or None
                st["last_miss"] = None
            elif off is False and st["cur"] is not None and st["cur"]["team_id"] == team:
                # a defensive rebound with no recorded miss (e.g. after a heave): keep the possession
                pass
            if off:
                cur = owner(team, t, a)
                cur["oreb"] += 1
                st["last_oreb"] = (cur["n"], t)
            else:
                cur = st["cur"]
                if not (cur is not None and cur["team_id"] == team and cur["fga"] == cur["fta"] == cur["tov"] == 0):
                    cur = start(team, t, "dreb", a)
                elif cur["kind"] in ("made", "ft", "unknown"):
                    cur["kind"] = "dreb"
            r = row(a, t, team, "reb", made=int(off))
            with_poss(r, st["cur"], t)
        elif kind == "turnover":
            if team not in teams:
                continue
            cur = owner(team, t, a)
            cur["tov"] += 1
            touch(cur, t)
            r = row(a, t, team, "tov", made=0)
            with_poss(r, cur, t)
            st["last_tov"] = r
            st["last_miss"] = None
            start(other[team], t, "deadball", a)
            stealer = _int(a.get("stealPersonId"))
            if stealer:
                r["made"], r["other_id"] = 1, stealer
                st["cur"]["kind"] = "steal"
        elif kind == "steal":
            lt = st["last_tov"]
            if lt is not None and abs(lt["t"] - t) < 1.5:
                lt["made"], lt["other_id"] = 1, _int(a.get("personId")) or None
                if st["cur"] is not None and st["cur"]["n"] == lt["poss_n"] + 1:
                    st["cur"]["kind"] = "steal"
    if st["cur"] is not None and st["period"]:
        st["cur"]["end_t"] = max(st["cur"]["end_t"], period_start(st["period"]) + period_length(st["period"]))
    for p in poss:
        p["start_t"], p["end_t"] = round(p["start_t"], 1), round(p["end_t"], 1)
    return plays, poss
