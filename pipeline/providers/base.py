"""Provider interface: the standard shape every data source must return.

The rest of the pipeline only ever sees these classes. To switch a data
source, write a new provider that returns the same shapes; nothing
downstream has to change.
"""
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class TeamRef:
    team_id: int
    tricode: str
    city: str
    name: str


@dataclass
class ScheduledGame:
    game_id: str
    season_id: str
    game_date: str            # Eastern calendar date, 'YYYY-MM-DD'
    tip_utc: Optional[str]
    home: TeamRef
    away: TeamRef
    status: str               # scheduled | live | final | postponed | cancelled
    home_score: Optional[int]
    away_score: Optional[int]
    home_record: Optional[str]
    away_record: Optional[str]
    arena: Optional[str]
    neutral_site: bool
    status_text: str = ""


@dataclass
class PlayerLine:
    player_id: int
    team_id: int
    first_name: str
    last_name: str
    full_name: str
    position: Optional[str]   # only listed for starters in NBA box scores
    jersey: Optional[str]
    on_roster_status: str     # 'active' or 'inactive' (game-day roster)
    started: bool
    seconds: int
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0
    plus_minus: Optional[int] = None
    dnp_reason: Optional[str] = None


@dataclass
class TeamLine:
    team: TeamRef
    is_home: bool
    seconds: int              # total player-seconds (240 min regulation)
    pts: int
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    oreb: int                 # player offensive rebounds (excludes team rebounds)
    dreb: int
    ast: int
    stl: int
    blk: int
    tov: int                  # total turnovers, including team turnovers
    pf: int
    pts_paint: Optional[int]
    pts_second_chance: Optional[int]
    pts_fast_break: Optional[int]
    pts_off_tov: Optional[int]
    players: list = field(default_factory=list)


@dataclass
class BoxScore:
    game_id: str
    season_id: str
    game_date: str
    tip_utc: Optional[str]
    status: str
    periods: int
    arena: Optional[str]
    home: TeamLine
    away: TeamLine
    officials: list = field(default_factory=list)   # [(official_id, name)]


class GamesProvider(Protocol):
    """Anything that can supply the schedule and box scores."""

    name: str

    def get_schedule(self) -> list:
        """Every game on the current season's schedule, as ScheduledGame."""
        ...

    def get_boxscore(self, game_id: str) -> Optional[BoxScore]:
        """The box score for one game, or None if it is not published yet."""
        ...
