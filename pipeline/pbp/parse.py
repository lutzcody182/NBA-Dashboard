"""Turn NBA play-by-play ("actions") into lineup stints and player totals.

Input is the action list from NBA's live play-by-play feed (the same
schema the historical archive uses). Output:

  * stints   - stretches with the same 10 players on the floor, with exact
               possessions, points and counting stats for both teams
  * players  - per-player totals for the game, including seconds on court
               and shots split by court zone
  * team possessions, counted from the feed's possession marker
  * quality  - how many events disagreed with the inferred lineups, so bad
               games can be left out of the ratings instead of polluting them

Nobody is ever guessed into a lineup silently: when the starters of a
period cannot be worked out from the events, the stint is flagged.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..util import duration_to_seconds

REG_PERIOD = 720
OT_PERIOD = 300

ZONES = {
    "Restricted Area": "rim",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range": "mid",
    "Left Corner 3": "c3",
    "Right Corner 3": "c3",
    "Above the Break 3": "ab3",
}
ZONE_KEYS = ("rim", "paint", "mid", "c3", "ab3")

TEAM_STAT_KEYS = (
    "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "tov", "pf", "ast", "stl", "blk",
    "pts_paint", "pts_2nd", "pts_fb", "pts_off_tov",
    *(f"fga_{z}" for z in ZONE_KEYS), *(f"fgm_{z}" for z in ZONE_KEYS),
)
PLAYER_STAT_KEYS = ("pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "ast",
                    "stl", "blk", "tov", "pf", "and1",
                    *(f"fga_{z}" for z in ZONE_KEYS), *(f"fgm_{z}" for z in ZONE_KEYS))


def period_start(period: int) -> float:
    if period <= 4:
        return (period - 1) * REG_PERIOD
    return 4 * REG_PERIOD + (period - 5) * OT_PERIOD


def period_length(period: int) -> float:
    return REG_PERIOD if period <= 4 else OT_PERIOD


def _clock_seconds(value) -> float:
    if not value:
        return 0.0
    s = str(value)
    if s.startswith("PT"):
        m = s[2:].split("M")
        mins = float(m[0]) if len(m) > 1 else 0.0
        secs = float(m[-1].rstrip("S") or 0)
        return mins * 60 + secs
    return float(duration_to_seconds(s))


def _int(v) -> int:
    try:
        if v is None or v == "" or (isinstance(v, float) and v != v):
            return 0
        return int(float(v))
    except (TypeError, ValueError):
        return 0


@dataclass
class Stint:
    period: int
    start: float          # elapsed game seconds
    end: float
    lineups: dict         # team_id -> tuple of 5 player ids (sorted)
    poss: dict            # team_id -> offensive possessions
    pts: dict             # team_id -> points scored
    stats: dict           # team_id -> {stat: count}
    margin_start: int     # home minus away at stint start
    ok: bool = True

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass
class GamePbp:
    game_id: str
    home_id: int
    away_id: int
    stints: list = field(default_factory=list)
    players: dict = field(default_factory=dict)      # player_id -> stats dict (incl. team_id, seconds, starter)
    team_poss: dict = field(default_factory=dict)
    final_score: tuple = (0, 0)
    periods: int = 4
    events: int = 0
    bad_events: int = 0                               # events contradicting the inferred lineup
    notes: list = field(default_factory=list)

    @property
    def quality(self) -> float:
        return 1.0 - (self.bad_events / self.events if self.events else 0.0)


# -- helpers ---------------------------------------------------------------

def _qualifiers(a) -> set:
    q = a.get("qualifiers")
    if not q:
        return set()
    if isinstance(q, str):
        return {x.strip() for x in q.split(",") if x.strip()}
    return set(q)


def _is_technical(a) -> bool:
    return a.get("actionType") == "foul" and (a.get("subType") == "technical" or
                                              "technical" in str(a.get("descriptor") or "").lower())


def _people_in(a, team_of) -> list:
    """(player_id, team_id) pairs this event shows to be on the floor."""
    t = a.get("actionType")
    if t in ("substitution", "timeout", "period", "game", "ejection") or _is_technical(a):
        return []
    out = []
    pid = _int(a.get("personId"))
    if pid and pid in team_of:
        out.append(pid)
    for key in ("assistPersonId", "jumpBallWonPersonId", "jumpBallLostPersonId", "jumpBallRecoverdPersonId",
                "stealPersonId", "blockPersonId"):
        p = _int(a.get(key))
        if p and p in team_of:
            out.append(p)
    if t == "foul" and a.get("subType") in ("personal", "offensive"):
        p = _int(a.get("foulDrawnPersonId"))
        if p and p in team_of:
            out.append(p)
    return out


def _period_starters(events, team_of, teams, prev_end, box_starters, game):
    """Work out who started a period for each team. Returns {team: set}, ok flag."""
    first = {}  # player -> 'in' | 'on'
    for a in events:
        if a.get("actionType") == "substitution":
            p = _int(a.get("personId"))
            if p and p not in first:
                first[p] = "in" if a.get("subType") == "in" else "on"
            continue
        for p in _people_in(a, team_of):
            first.setdefault(p, "on")
    period = _int(events[0].get("period")) if events else 1
    lineups, ok = {}, True
    for t in teams:
        on = [p for p, how in first.items() if how == "on" and team_of.get(p) == t]
        # order: earliest appearance first (dict keeps insertion order)
        if len(on) < 5:
            candidates = []
            if period == 1 and box_starters and box_starters.get(t):
                candidates = list(box_starters[t])
            elif prev_end.get(t):
                candidates = list(prev_end[t])
            for p in candidates:
                if len(on) >= 5:
                    break
                if p not in on and first.get(p) != "in":
                    on.append(p)
        if len(on) != 5:
            ok = False
            game.notes.append(f"period {period}: found {len(on)} starters for team {t}")
            on = on[:5]
        lineups[t] = set(on)
    return lineups, ok


def _new_stats():
    return {k: 0 for k in TEAM_STAT_KEYS}


def _new_player(team_id):
    d = {k: 0 for k in PLAYER_STAT_KEYS}
    d.update(team_id=team_id, seconds=0.0, starter=False, on_court_poss=0, opp_on_court_poss=0)
    return d


# -- main ------------------------------------------------------------------

def parse_actions(game_id: str, actions: list, home_id: int, away_id: int,
                  box_starters: Optional[dict] = None) -> GamePbp:
    game = GamePbp(game_id=game_id, home_id=int(home_id), away_id=int(away_id))
    teams = (game.home_id, game.away_id)
    acts = sorted(actions, key=lambda a: (_int(a.get("period")), _int(a.get("orderNumber")) or _int(a.get("actionNumber"))))

    team_of = {}
    for a in acts:
        p, t = _int(a.get("personId")), _int(a.get("teamId"))
        if p and t in teams:
            team_of.setdefault(p, t)
    for t, ps in (box_starters or {}).items():
        for p in ps:
            team_of.setdefault(p, t)

    by_period = defaultdict(list)
    for a in acts:
        per = _int(a.get("period"))
        if per:
            by_period[per].append(a)
    game.periods = max(by_period) if by_period else 0

    players = {}

    def P(pid):
        if pid not in players:
            players[pid] = _new_player(team_of.get(pid))
        return players[pid]

    team_poss = {t: 0 for t in teams}
    prev_end = {}
    score = [0, 0]  # home, away

    for per in sorted(by_period):
        events = by_period[per]
        lineups, per_ok = _period_starters(events, team_of, teams, prev_end, box_starters, game)
        if per == 1:
            for t in teams:
                for p in lineups[t]:
                    P(p)["starter"] = True
        p_start, p_len = period_start(per), period_length(per)

        def new_stint(t0):
            return {
                "start": t0, "lineups": None, "pts0": tuple(score),
                "poss": {t: 0 for t in teams}, "stats": {t: _new_stats() for t in teams},
                "events": 0, "ok": per_ok,
            }

        stint = new_stint(p_start)
        poss_team, poss_stint = 0, None

        def close(st, t_end):
            if st["lineups"] is None:
                st["lineups"] = {t: tuple(sorted(lineups[t])) for t in teams}
                if any(len(v) != 5 for v in st["lineups"].values()):
                    st["ok"] = False
            if t_end - st["start"] <= 0 and st["events"] == 0 and not any(st["poss"].values()):
                return
            pts_home = score[0] - st["pts0"][0]
            pts_away = score[1] - st["pts0"][1]
            s = Stint(
                period=per, start=st["start"], end=t_end, lineups=st["lineups"],
                # poss is shared with the open dict on purpose: a possession that
                # ends here can be credited after a substitution closes the stint.
                poss=st["poss"], pts={teams[0]: pts_home, teams[1]: pts_away},
                stats=st["stats"], margin_start=st["pts0"][0] - st["pts0"][1], ok=st["ok"],
            )
            game.stints.append(s)

        for a in events:
            kind = a.get("actionType")
            t_now = p_start + p_len - _clock_seconds(a.get("clock"))

            if kind == "substitution":
                if stint["lineups"] is not None or stint["events"]:
                    close(stint, t_now)
                    stint = new_stint(t_now)
                p, t = _int(a.get("personId")), _int(a.get("teamId"))
                if t in lineups and p:
                    if a.get("subType") == "in":
                        lineups[t].add(p)
                    else:
                        lineups[t].discard(p)
                continue

            # for the play-by-play log (plays.py): game time and the stint this play belongs to
            a["_t"] = t_now
            a["_stint"] = len(game.stints) + 1

            if stint["lineups"] is None and kind not in ("period", "game"):
                stint["lineups"] = {t: tuple(sorted(lineups[t])) for t in teams}
                if any(len(v) != 5 for v in stint["lineups"].values()):
                    stint["ok"] = False
                    game.notes.append(f"period {per} {a.get('clock')}: lineup sizes "
                                      f"{[len(v) for v in stint['lineups'].values()]}")

            # Possession accounting: credit a finished possession to the stint
            # where that team's last event of it happened.
            pos = _int(a.get("possession"))
            if pos in team_poss and kind not in ("period", "game"):
                if pos != poss_team:
                    if poss_team:
                        team_poss[poss_team] += 1
                        poss_stint["poss"][poss_team] += 1
                    poss_team = pos
                poss_stint = stint

            sh, sa = _int(a.get("scoreHome")), _int(a.get("scoreAway"))
            if sh or sa:
                score[0], score[1] = max(score[0], sh), max(score[1], sa)

            if kind in ("period", "game", "timeout"):
                continue
            stint["events"] += 1
            game.events += 1

            # Lineup check: anyone credited with an action must be on the floor.
            for p in _people_in(a, team_of):
                t = team_of.get(p)
                if t in lineups and p not in lineups[t]:
                    game.bad_events += 1
                    stint["ok"] = False
                    break

            t = _int(a.get("teamId"))
            if t not in teams:
                continue
            st = stint["stats"][t]
            pid = _int(a.get("personId"))
            made = a.get("shotResult") == "Made"
            if made and kind in ("2pt", "3pt", "freethrow"):
                value = 3 if kind == "3pt" else 2 if kind == "2pt" else 1
                q = _qualifiers(a)
                if "pointsinthepaint" in q:
                    st["pts_paint"] += value
                if "2ndchance" in q:
                    st["pts_2nd"] += value
                if "fastbreak" in q:
                    st["pts_fb"] += value
                if "fromturnover" in q:
                    st["pts_off_tov"] += value

            if kind in ("2pt", "3pt"):
                zone = ZONES.get(a.get("area"))
                st["fga"] += 1
                st["fgm"] += made
                if kind == "3pt":
                    st["fg3a"] += 1
                    st["fg3m"] += made
                if zone:
                    st[f"fga_{zone}"] += 1
                    st[f"fgm_{zone}"] += made
                if pid:
                    pl = P(pid)
                    pl["fga"] += 1
                    pl["fgm"] += made
                    if kind == "3pt":
                        pl["fg3a"] += 1
                        pl["fg3m"] += made
                    if zone:
                        pl[f"fga_{zone}"] += 1
                        pl[f"fgm_{zone}"] += made
                    if made:
                        pl["pts"] += 3 if kind == "3pt" else 2
                    ast = _int(a.get("assistPersonId"))
                    if made and ast:
                        P(ast)["ast"] += 1
                        st["ast"] += 1
            elif kind == "freethrow":
                st["fta"] += 1
                st["ftm"] += made
                if pid:
                    pl = P(pid)
                    pl["fta"] += 1
                    pl["ftm"] += made
                    pl["pts"] += made
                    if a.get("subType") == "1 of 1" and "technical" not in str(a.get("description") or "").lower():
                        pl["and1"] += 1
            elif kind == "rebound" and pid:
                key = "oreb" if a.get("subType") == "offensive" else "dreb"
                st[key] += 1
                P(pid)[key] += 1
            elif kind == "turnover":
                st["tov"] += 1
                if pid:
                    P(pid)["tov"] += 1
            elif kind == "steal" and pid:
                st["stl"] += 1
                P(pid)["stl"] += 1
            elif kind == "block" and pid:
                st["blk"] += 1
                P(pid)["blk"] += 1
            elif kind == "foul" and not _is_technical(a) and pid:
                st["pf"] += 1
                P(pid)["pf"] += 1

        # end of period: the open possession ends here
        if poss_team:
            team_poss[poss_team] += 1
            poss_stint["poss"][poss_team] += 1
        close(stint, p_start + p_len)
        prev_end = {t: set(lineups[t]) for t in teams}

    for s in game.stints:
        for t in teams:
            o = teams[1] if t == teams[0] else teams[0]
            for p in s.lineups[t]:
                pl = P(p)
                pl["seconds"] += s.seconds
                pl["on_court_poss"] += s.poss[t]
                pl["opp_on_court_poss"] += s.poss[o]
    game.players = players
    game.team_poss = team_poss
    game.final_score = tuple(score)
    return game
