"""NBA's public CDN feeds (cdn.nba.com): schedule and box scores.

These are the JSON files nba.com itself loads. They are free but
undocumented, so every parser here fails loudly on an unexpected shape
rather than guessing. Parsing is kept separate from fetching so it can be
tested against saved responses.
"""
import time
from typing import Optional

import requests

from .. import config
from ..util import duration_to_seconds, et_date, season_from_game_id
from .base import BoxScore, PlayerLine, ScheduledGame, TeamLine, TeamRef

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
PLAYBYPLAY_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

_STATUS = {1: "scheduled", 2: "live", 3: "final"}


class FeedError(RuntimeError):
    """The feed answered, but not with what we expected."""


class NbaCdnProvider:
    name = "nba_cdn"

    def __init__(self, session: Optional[requests.Session] = None, retries: int = 3):
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.retries = retries
        self.last_status = {}

    # -- fetching ---------------------------------------------------------

    def _get_json(self, url: str) -> Optional[dict]:
        """GET a JSON file. Returns None when the file does not exist yet.

        The CDN answers 403 or 404 for box scores of games that have not
        started, so those are treated as "not published", not as errors.
        """
        last_error = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException as exc:
                last_error = exc
            else:
                self.last_status[url] = r.status_code
                if r.status_code in (403, 404):
                    return None
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError as exc:
                        raise FeedError(f"{url} did not return JSON") from exc
                last_error = FeedError(f"{url} returned HTTP {r.status_code}")
            time.sleep(2 ** attempt)
        raise FeedError(f"Could not fetch {url}: {last_error}")

    def get_schedule(self) -> list:
        data = self._get_json(SCHEDULE_URL)
        if data is None:
            code = self.last_status.get(SCHEDULE_URL)
            raise FeedError(f"Schedule file not served by the CDN (HTTP {code})")
        return parse_schedule(data)

    def get_boxscore(self, game_id: str) -> Optional[BoxScore]:
        data = self._get_json(BOXSCORE_URL.format(game_id=game_id))
        if data is None:
            return None
        return parse_boxscore(data)

    def get_playbyplay(self, game_id: str) -> Optional[list]:
        """The list of play-by-play actions for one game, or None if not published."""
        data = self._get_json(PLAYBYPLAY_URL.format(game_id=game_id))
        if data is None:
            return None
        try:
            return data["game"]["actions"]
        except (KeyError, TypeError) as exc:
            raise FeedError(f"Play-by-play for {game_id} is missing {exc}") from exc

    def polite_pause(self) -> None:
        time.sleep(config.REQUEST_DELAY)


# -- parsing (pure functions, unit-tested) ---------------------------------


def _team_ref(t: dict) -> TeamRef:
    return TeamRef(
        team_id=int(t["teamId"]),
        tricode=t["teamTricode"],
        city=t.get("teamCity") or "",
        name=t.get("teamName") or "",
    )


def parse_schedule(data: dict) -> list:
    """Turn scheduleLeagueV2_1.json into ScheduledGame records.

    Skips All-Star events and exhibitions against non-NBA teams.
    """
    try:
        league = data["leagueSchedule"]
        dates = league["gameDates"]
    except KeyError as exc:
        raise FeedError(f"Schedule is missing {exc}") from exc

    games = []
    for day in dates:
        for g in day.get("games", []):
            gid = g["gameId"]
            if gid[:3] not in config.SEASON_TYPE_BY_PREFIX:
                continue
            home, away = g["homeTeam"], g["awayTeam"]
            if int(home["teamId"]) not in config.NBA_TEAM_IDS or int(away["teamId"]) not in config.NBA_TEAM_IDS:
                continue
            status = _STATUS.get(int(g["gameStatus"]), "scheduled")
            # postponedStatus is 'A' for normal games. Anything else on a game
            # that has not been played is treated as postponed.
            if g.get("postponedStatus", "A") != "A" and status != "final":
                status = "postponed"
            played = status in ("live", "final")
            games.append(
                ScheduledGame(
                    game_id=gid,
                    season_id=season_from_game_id(gid),
                    game_date=et_date(g["gameDateEst"]),
                    tip_utc=g.get("gameDateTimeUTC"),
                    home=_team_ref(home),
                    away=_team_ref(away),
                    status=status,
                    home_score=int(home["score"]) if played else None,
                    away_score=int(away["score"]) if played else None,
                    home_record=f'{home.get("wins", 0)}-{home.get("losses", 0)}',
                    away_record=f'{away.get("wins", 0)}-{away.get("losses", 0)}',
                    arena=g.get("arenaName"),
                    neutral_site=bool(g.get("isNeutral")),
                    status_text=g.get("gameStatusText", ""),
                )
            )
    return games


def _int(stats: dict, key: str, default=0):
    v = stats.get(key)
    return default if v is None else int(v)


def _parse_player(p: dict, team_id: int) -> PlayerLine:
    s = p.get("statistics") or {}
    played = str(p.get("played", "0")) == "1"
    reason = p.get("notPlayingReason") or p.get("notPlayingDescription")
    first = p.get("firstName") or ""
    last = p.get("familyName") or ""
    return PlayerLine(
        player_id=int(p["personId"]),
        team_id=team_id,
        first_name=first,
        last_name=last,
        full_name=p.get("name") or f"{first} {last}".strip(),
        position=p.get("position") or None,
        jersey=p.get("jerseyNum") or None,
        on_roster_status="active" if p.get("status") == "ACTIVE" else "inactive",
        started=str(p.get("starter", "0")) == "1",
        seconds=duration_to_seconds(s.get("minutes")) if played else 0,
        pts=_int(s, "points"),
        fgm=_int(s, "fieldGoalsMade"),
        fga=_int(s, "fieldGoalsAttempted"),
        fg3m=_int(s, "threePointersMade"),
        fg3a=_int(s, "threePointersAttempted"),
        ftm=_int(s, "freeThrowsMade"),
        fta=_int(s, "freeThrowsAttempted"),
        oreb=_int(s, "reboundsOffensive"),
        dreb=_int(s, "reboundsDefensive"),
        ast=_int(s, "assists"),
        stl=_int(s, "steals"),
        blk=_int(s, "blocks"),
        tov=_int(s, "turnovers"),
        pf=_int(s, "foulsPersonal"),
        plus_minus=int(round(s["plusMinusPoints"])) if s.get("plusMinusPoints") is not None and played else None,
        dnp_reason=None if played else (reason or "DNP"),
    )


def _parse_team(t: dict, is_home: bool) -> TeamLine:
    s = t.get("statistics")
    if not s:
        raise FeedError(f"Team {t.get('teamTricode')} has no statistics block")
    ref = _team_ref(t)
    tov = s.get("turnoversTotal")
    return TeamLine(
        team=ref,
        is_home=is_home,
        seconds=duration_to_seconds(s.get("minutes")),
        pts=_int(s, "points"),
        fgm=_int(s, "fieldGoalsMade"),
        fga=_int(s, "fieldGoalsAttempted"),
        fg3m=_int(s, "threePointersMade"),
        fg3a=_int(s, "threePointersAttempted"),
        ftm=_int(s, "freeThrowsMade"),
        fta=_int(s, "freeThrowsAttempted"),
        oreb=_int(s, "reboundsOffensive"),
        dreb=_int(s, "reboundsDefensive"),
        ast=_int(s, "assists"),
        stl=_int(s, "steals"),
        blk=_int(s, "blocks"),
        tov=int(tov) if tov is not None else _int(s, "turnovers"),
        pf=_int(s, "foulsPersonal"),
        pts_paint=_int(s, "pointsInThePaint", None),
        pts_second_chance=_int(s, "pointsSecondChance", None),
        pts_fast_break=_int(s, "pointsFastBreak", None),
        pts_off_tov=_int(s, "pointsFromTurnovers", None),
        players=[_parse_player(p, ref.team_id) for p in t.get("players", [])],
    )


def parse_boxscore(data: dict) -> BoxScore:
    try:
        g = data["game"]
        gid = g["gameId"]
    except KeyError as exc:
        raise FeedError(f"Box score is missing {exc}") from exc
    status = _STATUS.get(int(g.get("gameStatus", 1)), "scheduled")
    arena = (g.get("arena") or {}).get("arenaName")
    officials = [
        (int(o["personId"]), o.get("name") or "")
        for o in g.get("officials", [])
        if o.get("personId")
    ]
    return BoxScore(
        game_id=gid,
        season_id=season_from_game_id(gid),
        game_date=et_date(g.get("gameEt") or g.get("gameTimeLocal") or ""),
        tip_utc=g.get("gameTimeUTC"),
        status=status,
        periods=int(g.get("period") or 0),
        arena=arena,
        home=_parse_team(g["homeTeam"], True),
        away=_parse_team(g["awayTeam"], False),
        officials=officials,
    )
