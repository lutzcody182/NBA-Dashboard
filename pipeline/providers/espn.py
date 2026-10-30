"""ESPN's public JSON feeds: schedule, box scores, play-by-play and rosters.

Used when NBA's own CDN refuses GitHub's servers. ESPN uses its own team,
player and game IDs, so this module translates everything into ours:

  * teams   - fixed table below (ESPN id -> NBA id, NBA tricode)
  * players - matched to the NBA IDs already in the database by name (and
              team when two names collide); a player we have never seen gets
              a new ID of 2,000,000,000 + his ESPN ID. Every match is stored
              in id_map with how it was made, so it never changes later.
  * games   - '<NBA season-type prefix><yy>E<ESPN event id>', for example
              '00226E401909088'. The first five characters follow NBA's
              format, so season and season type read the same way.

Play-by-play is converted into the same action format NBA's feed uses, so
the existing parser (lineups, possessions, zones) runs on it unchanged.
"""
import math
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from .. import config
from .base import BoxScore, PlayerLine, ScheduledGame, TeamLine, TeamRef

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SCOREBOARD_URL = BASE + "/scoreboard?dates={month}&limit=500"
SUMMARY_URL = BASE + "/summary?event={event}"
ROSTER_URL = BASE + "/teams/{team}/roster"
ET = ZoneInfo("America/New_York")
NEW_PLAYER_BASE = 2_000_000_000

# ESPN team id -> (NBA team id, NBA tricode, city, name). NBA tricodes are kept
# because the official injury report uses them.
TEAMS = {
    1: (1610612737, "ATL", "Atlanta", "Hawks"),
    2: (1610612738, "BOS", "Boston", "Celtics"),
    3: (1610612740, "NOP", "New Orleans", "Pelicans"),
    4: (1610612741, "CHI", "Chicago", "Bulls"),
    5: (1610612739, "CLE", "Cleveland", "Cavaliers"),
    6: (1610612742, "DAL", "Dallas", "Mavericks"),
    7: (1610612743, "DEN", "Denver", "Nuggets"),
    8: (1610612765, "DET", "Detroit", "Pistons"),
    9: (1610612744, "GSW", "Golden State", "Warriors"),
    10: (1610612745, "HOU", "Houston", "Rockets"),
    11: (1610612754, "IND", "Indiana", "Pacers"),
    12: (1610612746, "LAC", "LA", "Clippers"),
    13: (1610612747, "LAL", "Los Angeles", "Lakers"),
    14: (1610612748, "MIA", "Miami", "Heat"),
    15: (1610612749, "MIL", "Milwaukee", "Bucks"),
    16: (1610612750, "MIN", "Minnesota", "Timberwolves"),
    17: (1610612751, "BKN", "Brooklyn", "Nets"),
    18: (1610612752, "NYK", "New York", "Knicks"),
    19: (1610612753, "ORL", "Orlando", "Magic"),
    20: (1610612755, "PHI", "Philadelphia", "76ers"),
    21: (1610612756, "PHX", "Phoenix", "Suns"),
    22: (1610612757, "POR", "Portland", "Trail Blazers"),
    23: (1610612758, "SAC", "Sacramento", "Kings"),
    24: (1610612759, "SAS", "San Antonio", "Spurs"),
    25: (1610612760, "OKC", "Oklahoma City", "Thunder"),
    26: (1610612762, "UTA", "Utah", "Jazz"),
    27: (1610612764, "WAS", "Washington", "Wizards"),
    28: (1610612761, "TOR", "Toronto", "Raptors"),
    29: (1610612763, "MEM", "Memphis", "Grizzlies"),
    30: (1610612766, "CHA", "Charlotte", "Hornets"),
}
NBA_TO_ESPN = {v[0]: k for k, v in TEAMS.items()}

# ESPN season type -> NBA game ID prefix
PREFIX = {1: "001", 2: "002", 3: "004", 5: "005"}

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
    "Accept": "application/json",
}


class FeedError(RuntimeError):
    pass


def team_ref(espn_team_id) -> Optional[TeamRef]:
    t = TEAMS.get(int(espn_team_id or 0))
    return TeamRef(team_id=t[0], tricode=t[1], city=t[2], name=t[3]) if t else None


def make_game_id(season_type: int, season_year: int, event_id) -> Optional[str]:
    """season_year is ESPN's (the year the season ends): 2027 for 2026-27."""
    prefix = PREFIX.get(int(season_type))
    if not prefix:
        return None
    return f"{prefix}{(int(season_year) - 1) % 100:02d}E{event_id}"


def event_of(game_id: str) -> str:
    if "E" not in game_id:
        raise FeedError(f"{game_id} is not an ESPN-sourced game")
    return game_id.split("E", 1)[1]


def _utc(value: str) -> datetime:
    v = value.replace("Z", "+00:00")
    if len(v) == 22 and v[16] == "+":           # '2026-10-20T19:00+00:00' (no seconds)
        v = v[:16] + ":00" + v[16:]
    return datetime.fromisoformat(v).astimezone(timezone.utc)


def _tip(value: str):
    dt = _utc(value)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), dt.astimezone(ET).date().isoformat()


def _status(status_type: dict) -> str:
    name = (status_type.get("name") or "").upper()
    if "POSTPONED" in name or "SUSPENDED" in name:
        return "postponed"
    if "CANCEL" in name:
        return "cancelled"
    state = status_type.get("state")
    if state == "post" or status_type.get("completed"):
        return "final"
    if state == "in":
        return "live"
    return "scheduled"


# -- schedule ----------------------------------------------------------------------

def parse_scoreboard(data: dict, season_year: int) -> list:
    """Scoreboard events -> ScheduledGame, for one ESPN season (2027 = 2026-27)."""
    games = []
    for e in data.get("events", []):
        season = e.get("season") or {}
        if int(season.get("year") or 0) != int(season_year):
            continue
        gid = make_game_id(season.get("type") or 0, season_year, e["id"])
        if not gid:
            continue
        c = (e.get("competitions") or [{}])[0]
        sides = {x.get("homeAway"): x for x in c.get("competitors", [])}
        if set(sides) != {"home", "away"}:
            continue
        home, away = team_ref(sides["home"].get("id")), team_ref(sides["away"].get("id"))
        if not home or not away:
            continue                      # All-Star or exhibition against a non-NBA team
        status = _status((c.get("status") or e.get("status") or {}).get("type") or {})
        tip, day = _tip(c.get("date") or e["date"])
        played = status in ("live", "final")

        def record(x):
            for r in x.get("records") or []:
                if r.get("type") == "total":
                    return r.get("summary")
            return None

        games.append(ScheduledGame(
            game_id=gid, season_id=f"{season_year - 1}-{str(season_year)[-2:]}", game_date=day, tip_utc=tip,
            home=home, away=away, status=status,
            home_score=int(sides["home"].get("score") or 0) if played else None,
            away_score=int(sides["away"].get("score") or 0) if played else None,
            home_record=record(sides["home"]), away_record=record(sides["away"]),
            arena=(c.get("venue") or {}).get("fullName"), neutral_site=bool(c.get("neutralSite")),
            status_text=((c.get("status") or {}).get("type") or {}).get("shortDetail") or "",
        ))
    return games


def season_months(start_year: int) -> list:
    """'YYYYMM' strings from September to June."""
    return [f"{start_year}{m:02d}" for m in range(9, 13)] + [f"{start_year + 1}{m:02d}" for m in range(1, 7)]


# -- player IDs --------------------------------------------------------------------

_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[.,']", "", s.lower()).replace("-", " ")
    return " ".join(s.split())


def _strip_suffix(key: str) -> str:
    parts = key.split()
    while len(parts) > 2 and parts[-1] in _SUFFIX:
        parts.pop()
    return " ".join(parts)


def name_keys(display: str, short: str = "") -> list:
    """Short-name keys ('p george') an ESPN athlete could be stored under."""
    keys = []
    if short:
        keys.append(norm_name(short))
    parts = (display or "").split()
    if len(parts) >= 2:
        keys.append(norm_name(f"{parts[0][0]}. {' '.join(parts[1:])}"))
    keys += [_strip_suffix(k) for k in keys]
    seen, out = set(), []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


class PlayerIds:
    """Translate ESPN athlete IDs into the database's player IDs."""

    def __init__(self, conn):
        self.conn = conn
        self.known = {r[0]: int(r[1]) for r in conn.execute(
            "SELECT provider_key, internal_id FROM id_map WHERE entity_type = 'player' AND provider = 'espn'")}
        self.taken = set(self.known.values())
        self._index = None
        self.new_ids = 0

    def _build_index(self):
        last_team = {r[0]: r[1] for r in self.conn.execute(
            """SELECT player_id, team_id FROM (
                   SELECT p.player_id, p.team_id, ROW_NUMBER() OVER (
                       PARTITION BY p.player_id ORDER BY g.game_date DESC) rn
                   FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
                   WHERE p.seconds > 0 AND g.game_date >= date('now', '-1100 days'))
               WHERE rn = 1""")}
        idx = {}
        for pid, name in self.conn.execute("SELECT player_id, full_name FROM players"):
            if pid in self.taken or pid not in last_team:
                continue
            for k in {norm_name(name), _strip_suffix(norm_name(name))}:
                idx.setdefault(k, []).append((pid, last_team[pid]))
        self._index = idx

    def candidates(self, display, short, team_id):
        if self._index is None:
            self._build_index()
        for k in name_keys(display, short):
            hits = [h for h in self._index.get(k, []) if h[0] not in self.taken]
            if len(hits) == 1:
                return hits[0][0], "name" + ("+team" if hits[0][1] == team_id else "")
            if len(hits) > 1:
                same = [h for h in hits if h[1] == team_id]
                if len(same) == 1:
                    return same[0][0], "name+team"
                return None, "ambiguous"
        return None, "no match"

    def _save(self, espn_id, pid, how):
        self.conn.execute(
            """INSERT OR REPLACE INTO id_map (entity_type, provider, provider_key, internal_id, match_method)
               VALUES ('player', 'espn', ?, ?, ?)""", (str(espn_id), str(pid), how))
        self.known[str(espn_id)] = pid
        self.taken.add(pid)

    def get(self, espn_id, display, short, team_id) -> int:
        key = str(espn_id)
        if key in self.known:
            return self.known[key]
        pid, how = self.candidates(display, short, team_id)
        if pid is None:
            pid, how = NEW_PLAYER_BASE + int(espn_id), f"new ({how})"
            self.new_ids += 1
        self._save(key, pid, how)
        return pid

    def assign_all(self, athletes) -> dict:
        """Match a whole league's rosters at once: [(espn_id, display, short, team_id)].

        When two ESPN players point at the same stored player, the one on that
        player's last team gets him and the other is treated as new.
        """
        out, claims = {}, {}
        for espn_id, display, short, team_id in athletes:
            key = str(espn_id)
            if key in self.known:
                out[key] = self.known[key]
                continue
            pid, how = self.candidates(display, short, team_id)
            if pid is not None:
                claims.setdefault(pid, []).append((key, how))
        for pid, cl in claims.items():
            if len(cl) > 1:
                cl = [c for c in cl if c[1] == "name+team"] or []
            if len(cl) == 1:
                self._save(cl[0][0], pid, cl[0][1])
                out[cl[0][0]] = pid
        for espn_id, *_ in athletes:
            key = str(espn_id)
            if key not in out:
                out[key] = self.known.get(key) or self._new(key)
        return out

    def _new(self, key):
        pid = NEW_PLAYER_BASE + int(key)
        self._save(key, pid, "new")
        self.new_ids += 1
        return pid


# -- box score ---------------------------------------------------------------------

def _split(v):
    try:
        a, b = str(v).split("-")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return 0, 0


def _toint(v, default=0):
    try:
        return int(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return default


def summary_periods(summary: dict) -> int:
    per = [int((p.get("period") or {}).get("number") or 0) for p in summary.get("plays") or []]
    return max(per) if per else 4


def parse_box(summary: dict, ids: PlayerIds, season_year: Optional[int] = None):
    """ESPN game summary -> (BoxScore, {espn athlete id: (player id, team id)})."""
    h = summary.get("header") or {}
    comp = (h.get("competitions") or [{}])[0]
    season = h.get("season") or {}
    year = int(season.get("year") or 0) or season_year
    gid = make_game_id(season.get("type") or 0, year, h.get("id"))
    if not gid:
        raise FeedError(f"ESPN event {h.get('id')} has an unknown season type")
    status = _status((comp.get("status") or {}).get("type") or {})
    tip, day = _tip(comp["date"])
    sides = {x.get("homeAway"): x for x in comp.get("competitors", [])}
    score = {int(x.get("id")): _toint(x.get("score")) for x in comp.get("competitors", [])}
    periods = summary_periods(summary)
    team_secs = 5 * (2880 + 300 * max(0, periods - 4))

    team_stats = {}
    for t in (summary.get("boxscore") or {}).get("teams", []):
        team_stats[int(t["team"]["id"])] = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
    athletes = {}
    lines = {}
    for block in (summary.get("boxscore") or {}).get("players", []):
        espn_team = int(block["team"]["id"])
        ref = team_ref(espn_team)
        if not ref:
            raise FeedError(f"Unknown ESPN team {espn_team}")
        plist = []
        for st in block.get("statistics", [])[:1]:
            names = st.get("names") or []
            for a in st.get("athletes", []):
                ath = a["athlete"]
                pid = ids.get(ath["id"], ath.get("displayName", ""), ath.get("shortName", ""), ref.team_id)
                athletes[str(ath["id"])] = (pid, ref.team_id)
                s = dict(zip(names, a.get("stats") or []))
                played = bool(s) and not a.get("didNotPlay") and _toint(s.get("MIN")) > 0
                fgm, fga = _split(s.get("FG"))
                f3m, f3a = _split(s.get("3PT"))
                ftm, fta = _split(s.get("FT"))
                display = ath.get("displayName") or ""
                first, _, last = display.partition(" ")
                plist.append(PlayerLine(
                    player_id=pid, team_id=ref.team_id, first_name=first, last_name=last, full_name=display,
                    position=((ath.get("position") or {}).get("abbreviation")) or None,
                    jersey=ath.get("jersey") or None, on_roster_status="active",
                    started=bool(a.get("starter")),
                    # ESPN gives whole minutes; the play-by-play replaces this with exact seconds.
                    seconds=_toint(s.get("MIN")) * 60 if played else 0,
                    pts=_toint(s.get("PTS")), fgm=fgm, fga=fga, fg3m=f3m, fg3a=f3a, ftm=ftm, fta=fta,
                    oreb=_toint(s.get("OREB")), dreb=_toint(s.get("DREB")), ast=_toint(s.get("AST")),
                    stl=_toint(s.get("STL")), blk=_toint(s.get("BLK")), tov=_toint(s.get("TO")),
                    pf=_toint(s.get("PF")),
                    plus_minus=_toint(s.get("+/-"), None) if played else None,
                    # ESPN lists dressed players only; injured and inactive players are left out.
                    dnp_reason=None if played else "DNP - Coach's Decision (ESPN)",
                ))
        lines[espn_team] = (ref, plist)

    def team_line(side):
        espn_team = int(side["team"]["id"]) if "team" in side else int(side["id"])
        ref, plist = lines.get(espn_team, (team_ref(espn_team), []))
        ts = team_stats.get(espn_team, {})
        tot = lambda k: sum(getattr(p, k) for p in plist)  # noqa: E731
        tov = ts.get("totalTurnovers")
        return TeamLine(
            team=ref, is_home=side.get("homeAway") == "home", seconds=team_secs,
            pts=score.get(espn_team, tot("pts")), fgm=tot("fgm"), fga=tot("fga"), fg3m=tot("fg3m"),
            fg3a=tot("fg3a"), ftm=tot("ftm"), fta=tot("fta"), oreb=tot("oreb"), dreb=tot("dreb"),
            ast=tot("ast"), stl=tot("stl"), blk=tot("blk"),
            tov=_toint(tov) if tov is not None else tot("tov"), pf=tot("pf"),
            pts_paint=_toint(ts.get("pointsInPaint"), None), pts_second_chance=None,
            pts_fast_break=_toint(ts.get("fastBreakPoints"), None),
            pts_off_tov=_toint(ts.get("turnoverPoints"), None), players=plist,
        )

    if set(sides) != {"home", "away"}:
        raise FeedError(f"ESPN event {h.get('id')} is missing a team")
    return BoxScore(
        game_id=gid, season_id=f"{year - 1}-{str(year)[-2:]}", game_date=day, tip_utc=tip, status=status,
        periods=periods, arena=((summary.get("gameInfo") or {}).get("venue") or {}).get("fullName"),
        home=team_line(sides["home"]), away=team_line(sides["away"]), officials=[],
        full_roster=False,    # ESPN box scores leave out injured players: don't close roster stints
    ), athletes


# -- play-by-play ------------------------------------------------------------------

SENTINEL = -1_000_000
TURNOVER_TYPES = {"traveling"}
DEF_FOULS = ("shooting", "personal", "take", "flagrant", "clear path", "away from play", "transition")


def _clock(v) -> str:
    s = str(v or "0")
    if ":" in s:
        m, sec = s.split(":")
        return f"PT{int(m):02d}M{float(sec):05.2f}S"
    return f"PT00M{float(s):05.2f}S"


def _coord(p):
    c = p.get("coordinate") or {}
    x, y = c.get("x"), c.get("y")
    if x is None or y is None or x < -5 or x > 55 or y < -10 or y > 100:
        return None
    return float(x), float(y)


RIM_SHOTS = ("layup", "dunk", "tip", "putback", "finger roll", "alley oop")


def shot_zone(points: int, xy, text: str, shot_type: str = "") -> Optional[str]:
    """Court zone in NBA's terms.

    ESPN's coordinates are whole feet with the hoop at x=25, y=0, so they are
    only good to about a foot; the distance written in the play text is used
    first when there is one. Layups and dunks without a written distance count
    as at the rim when the coordinates put them within 6 feet.
    """
    m = re.search(r"(\d+)-foot", text or "")
    said = int(m.group(1)) if m else None
    if points == 3:
        if xy:
            return "c3" if xy[1] <= 9 else "ab3"
        return None
    near = math.hypot(xy[0] - 25, xy[1]) if xy else None
    if said is not None and said <= 4:
        return "rim"
    if said is None and near is not None and (near <= 4 or (near <= 6 and any(
            k in shot_type.lower() for k in RIM_SHOTS))):
        return "rim"
    if xy:
        return "paint" if abs(xy[0] - 25) <= 8 and xy[1] <= 13.75 else "mid"
    if said is not None:
        return "mid" if said >= 15 else None
    return None


def classify(type_text: str, text: str, shooting: bool) -> str:
    tt, tx = type_text.lower(), text.lower()
    if tt.startswith("free throw"):
        return "freethrow"
    if shooting:
        return "shot"
    if tt == "substitution":
        return "substitution"
    if "rebound" in tt:
        return "rebound"
    if "turnover" in tt or tt in TURNOVER_TYPES or ("turnover" in tx and "foul" not in tt):
        return "turnover"
    if "foul" in tt or "technical" in tt or "charge" in tt:
        return "foul"
    if "timeout" in tt:
        return "timeout"
    if tt == "jumpball":
        return "jumpball"
    if tt == "end period":
        return "period"
    if tt == "end game":
        return "game"
    if "ejection" in tt or "ejected" in tx:
        return "ejection"
    if "review" in tt or "challenge" in tt:
        return "instantreplay"
    return "violation"


def convert_plays(summary: dict, athletes: dict) -> list:
    """ESPN plays -> NBA-style actions for parse_actions.

    athletes: {espn athlete id: (player id, team id)} from parse_box. Anyone
    not in the box score (coaches drawing technicals) is left anonymous.
    """
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    home = away = None
    for x in comp.get("competitors", []):
        ref = team_ref(x.get("id"))
        if ref and x.get("homeAway") == "home":
            home = ref.team_id
        elif ref:
            away = ref.team_id
    other = {home: away, away: home}
    actions, score = [], {home: 0, away: 0}

    def who(part, i):
        if len(part) > i:
            a = athletes.get(str(((part[i] or {}).get("athlete") or {}).get("id")))
            if a:
                return a
        return (0, 0)

    for n, p in enumerate(summary.get("plays") or []):
        tt = " ".join(str((p.get("type") or {}).get("text") or "").split())
        text = " ".join(str(p.get("text") or "").split())
        part = p.get("participants") or []
        team = (team_ref((p.get("team") or {}).get("id")) or TeamRef(0, "", "", "")).team_id
        p0, t0 = who(part, 0)
        if not team and t0:
            team = t0
        kind = classify(tt, text, bool(p.get("shootingPlay")))
        base = {"period": int((p.get("period") or {}).get("number") or 0), "clock": _clock(
            (p.get("clock") or {}).get("displayValue")), "description": text, "espnType": tt}
        if p.get("scoringPlay") and team in score:
            score[team] += int(p.get("scoreValue") or 0)
        base["scoreHome"], base["scoreAway"] = score[home], score[away]
        out = []

        def act(**kw):
            a = dict(base)
            a.update(kw)
            a.setdefault("teamId", team)
            a.setdefault("possession", 0)
            out.append(a)

        lower = tt.lower()
        if kind == "shot" and "heave" in lower and not p.get("scoringPlay"):
            # A missed end-of-quarter heave is not a field goal attempt in the box score.
            act(actionType="heave", personId=p0, possession=team)
        elif kind == "shot":
            pts = int(p.get("pointsAttempted") or (3 if "three point" in text.lower() else 2))
            zone = shot_zone(pts, _coord(p), text, tt)
            made = bool(p.get("scoringPlay"))
            q = ["pointsinthepaint"] if made and zone in ("rim", "paint") else []
            xy = _coord(p)
            a = {"actionType": "3pt" if pts == 3 else "2pt", "personId": p0, "shotResult": "Made" if made else "Missed",
                 "xFt": xy[0] if xy else None, "yFt": xy[1] if xy else None, "zone": zone,
                 "area": {"rim": "Restricted Area", "paint": "In The Paint (Non-RA)", "mid": "Mid-Range",
                          "c3": "Left Corner 3", "ab3": "Above the Break 3"}.get(zone), "qualifiers": q,
                 "possession": team}
            if made and "assist" in text.lower():
                a["assistPersonId"] = who(part, 1)[0]
            act(**a)
            if "blocks" in text.lower():
                b, bt = who(part, 1)
                if b:
                    act(actionType="block", personId=b, teamId=bt, possession=team)
        elif kind == "freethrow":
            m = re.search(r"(\d) of (\d)", tt)
            plain = re.fullmatch(r"free throw - \d of \d", lower) is not None
            sub = f"{m.group(1)} of {m.group(2)}" if (m and plain) else lower.replace("free throw - ", "")
            act(actionType="freethrow", subType=sub, personId=p0,
                shotResult="Made" if p.get("scoringPlay") else "Missed",
                possession=0 if "technical" in lower else team)
        elif kind == "rebound":
            act(actionType="rebound", subType="offensive" if "offensive" in lower else "defensive",
                personId=p0, possession=team)
        elif kind == "turnover":
            act(actionType="turnover", subType=lower, personId=p0, possession=team)
            if "steal" in text.lower():
                s, st = who(part, 1)
                if s:
                    act(actionType="steal", personId=s, teamId=st, possession=st)
        elif kind == "foul":
            technical = "technical" in lower or "techfoul" in text.lower()
            offensive = "offensive" in lower or "charge" in lower
            sub = "technical" if technical else "offensive" if offensive else (
                "flagrant" if "flagrant" in lower else "personal")
            pos = team if offensive else other.get(team, 0) if any(f in lower for f in DEF_FOULS) else 0
            act(actionType="foul", subType=sub, descriptor=tt, personId=p0,
                possession=0 if technical else pos)
        elif kind == "substitution":
            pin, tin = who(part, 0)
            pout, tout = who(part, 1)
            if pout:
                act(actionType="substitution", subType="out", personId=pout, teamId=tout or team)
            if pin:
                act(actionType="substitution", subType="in", personId=pin, teamId=tin or team)
        elif kind == "jumpball":
            gainer = who(part, 2)[0] if len(part) >= 3 else 0
            act(actionType="jumpball", personId=gainer, jumpBallWonPersonId=who(part, 0)[0],
                jumpBallLostPersonId=who(part, 1)[0], possession=team)
        elif kind in ("timeout", "period", "game", "instantreplay"):
            act(actionType=kind, personId=0, subType="end" if kind == "period" else lower)
        elif kind == "ejection":
            act(actionType="ejection", personId=p0)
        else:
            pos = other.get(team, 0) if ("goaltending" in lower and "offensive" not in lower) or "kicked" in lower else 0
            act(actionType="violation", subType=lower, personId=p0, possession=pos)
        for k, a in enumerate(out):
            a["orderNumber"] = n * 4 + k
            a["actionNumber"] = n * 4 + k
        actions.extend(out)
    return actions


# -- provider ----------------------------------------------------------------------

class EspnProvider:
    name = "espn"
    minutes_from_pbp = True   # box minutes are whole numbers; exact seconds come from play-by-play
    pbp_valid_since = "2026-10-30T04:00:00Z"   # games parsed earlier are re-read (now also saves the play log)

    def __init__(self, conn, session: Optional[requests.Session] = None, retries: int = 3,
                 now: Optional[datetime] = None):
        self.conn = conn
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.retries = retries
        self.ids = PlayerIds(conn)
        now = now or datetime.now(ET)
        self.start_year = now.year if now.month >= 7 else now.year - 1
        self._summaries = {}
        self._athletes = {}

    def _get_json(self, url: str) -> Optional[dict]:
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException as exc:
                last = exc
            else:
                if r.status_code == 404:
                    return None
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError as exc:
                        raise FeedError(f"{url} did not return JSON") from exc
                last = f"HTTP {r.status_code}"
            time.sleep(2 ** attempt)
        raise FeedError(f"Could not fetch {url}: {last}")

    def polite_pause(self) -> None:
        time.sleep(config.REQUEST_DELAY / 2)

    def get_schedule(self) -> list:
        games = {}
        for month in season_months(self.start_year):
            data = self._get_json(SCOREBOARD_URL.format(month=month))
            if not data:
                continue
            for g in parse_scoreboard(data, self.start_year + 1):
                games[g.game_id] = g
        if not games:
            raise FeedError("ESPN returned no games for this season")
        return sorted(games.values(), key=lambda g: (g.game_date, g.game_id))

    def _summary(self, game_id: str) -> Optional[dict]:
        ev = event_of(game_id)
        if ev not in self._summaries:
            self._summaries.clear()   # keep memory flat during long runs
            self._summaries[ev] = self._get_json(SUMMARY_URL.format(event=ev))
        return self._summaries[ev]

    def get_boxscore(self, game_id: str) -> Optional[BoxScore]:
        s = self._summary(game_id)
        if not s or not (s.get("boxscore") or {}).get("players"):
            return None
        box, athletes = parse_box(s, self.ids, self.start_year + 1)
        if box.game_id != game_id:
            raise FeedError(f"{game_id}: ESPN describes this event as {box.game_id}")
        if all(not p.seconds for side in (box.home, box.away) for p in side.players):
            return None       # listed, but nobody has played yet
        self._athletes[game_id] = athletes
        self.conn.commit()
        return box

    def get_playbyplay(self, game_id: str) -> Optional[list]:
        s = self._summary(game_id)
        if not s or not s.get("plays"):
            return None
        athletes = self._athletes.get(game_id)
        if athletes is None:
            _, athletes = parse_box(s, self.ids, self.start_year + 1)
            self.conn.commit()
        return convert_plays(s, athletes)

    def get_rosters(self) -> list:
        """[(TeamRef, [roster dicts with player_id])] for all 30 teams."""
        raw = []
        for espn_team in TEAMS:
            data = self._get_json(ROSTER_URL.format(team=espn_team))
            if data is None:
                raise FeedError(f"No roster for ESPN team {espn_team}")
            raw.append((team_ref(espn_team), data.get("athletes") or []))
            self.polite_pause()
        ids = self.ids.assign_all([(a["id"], a.get("displayName", ""), a.get("shortName", ""), ref.team_id)
                                   for ref, athletes in raw for a in athletes])
        self.conn.commit()
        out = []
        for ref, athletes in raw:
            players = []
            for a in athletes:
                players.append({
                    "player_id": ids[str(a["id"])],
                    "full_name": a.get("displayName") or a.get("fullName") or "",
                    "first_name": a.get("firstName"), "last_name": a.get("lastName"),
                    "position": (a.get("position") or {}).get("abbreviation"),
                    "jersey": a.get("jersey"),
                    "birth_date": (a.get("dateOfBirth") or "")[:10] or None,
                    "height_in": a.get("height"), "weight_lb": a.get("weight"),
                })
            out.append((ref, players))
        return out

