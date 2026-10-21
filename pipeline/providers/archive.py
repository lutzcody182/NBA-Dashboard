"""Historical play-by-play from the public nba_data archive on GitHub.

https://github.com/shufinskiy/nba_data publishes every game's play-by-play
from NBA's own CDN feed, one compressed CSV per season. It is used to load
past seasons quickly and reliably (GitHub is never IP-blocked the way NBA's
servers sometimes are). The current season always comes from NBA directly.
"""
import io
import tarfile
from collections import defaultdict

import pandas as pd
import requests

URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/cdnnba_{year}.tar.xz"

KEEP = ["gameId", "actionNumber", "orderNumber", "clock", "period", "actionType", "subType",
        "personId", "teamId", "teamTricode", "possession", "scoreHome", "scoreAway", "shotResult",
        "area", "descriptor", "description", "assistPersonId", "stealPersonId", "blockPersonId",
        "foulDrawnPersonId", "jumpBallWonPersonId", "jumpBallLostPersonId", "jumpBallRecoverdPersonId",
        "playerName", "playerNameI", "timeActual", "qualifiers"]


def download_season(start_year: int, session=None) -> pd.DataFrame:
    """Download one regular season's play-by-play (about 20 MB compressed)."""
    s = session or requests.Session()
    r = s.get(URL.format(year=start_year), timeout=180)
    r.raise_for_status()
    return read_archive(io.BytesIO(r.content))


def read_archive(fileobj) -> pd.DataFrame:
    with tarfile.open(fileobj=fileobj, mode="r:xz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".csv"))
        df = pd.read_csv(tar.extractfile(member), low_memory=False, usecols=lambda c: c in KEEP)
    return df


def game_id_str(raw) -> str:
    return f"{int(raw):010d}"


def split_games(df: pd.DataFrame):
    """Yield (game_id, actions, home_id, away_id) for each game in an archive frame."""
    df = df.astype(object).where(pd.notna(df), None)
    for gid, g in df.groupby("gameId", sort=True):
        actions = g.to_dict("records")
        home, away = infer_home_away(actions)
        if home and away:
            yield game_id_str(gid), actions, home, away


def infer_home_away(actions):
    """Find which team is home: the team whose made shots move scoreHome."""
    votes = defaultdict(int)
    prev_h = prev_a = 0
    teams = set()
    for a in sorted(actions, key=lambda x: (int(x.get("period") or 0), int(x.get("orderNumber") or x.get("actionNumber") or 0))):
        t = a.get("teamId")
        if t:
            teams.add(int(float(t)))
        h, aw = int(float(a.get("scoreHome") or 0)), int(float(a.get("scoreAway") or 0))
        if t and a.get("shotResult") == "Made":
            if h > prev_h:
                votes[int(float(t))] += 1
            elif aw > prev_a:
                votes[int(float(t))] -= 1
        prev_h, prev_a = max(prev_h, h), max(prev_a, aw)
    teams = [t for t in teams if t > 1610000000]
    if len(teams) != 2:
        return None, None
    home = max(teams, key=lambda t: votes[t])
    away = next(t for t in teams if t != home)
    return home, away
