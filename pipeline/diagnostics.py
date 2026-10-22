"""Check which data sources this machine can reach.

Run on every update so the Actions log (and the dashboard's data status)
shows exactly which feeds answer from GitHub's servers. Cheap: one small
request per source.
"""
import requests

from .providers.nba_cdn import HEADERS

PROBES = [
    ("nba_schedule", "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json", True),
    ("nba_schedule_plain", "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json", False),
    ("nba_scoreboard", "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json", True),
    ("nba_boxscore", "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022600001.json", True),
    ("nba_playbyplay", "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_0022600001.json", True),
    ("nba_stats", "https://stats.nba.com/stats/scoreboardv3?GameDate=2026-10-21&LeagueID=00", True),
    ("espn_scoreboard", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", False),
    ("espn_summary", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=401909088", False),
]


def probe(timeout=15) -> dict:
    out = {}
    for name, url, with_headers in PROBES:
        try:
            r = requests.get(url, headers=HEADERS if with_headers else {"User-Agent": "Mozilla/5.0"},
                             timeout=timeout)
            out[name] = {"status": r.status_code, "bytes": len(r.content),
                         "server": r.headers.get("Server"), "type": r.headers.get("Content-Type")}
        except requests.RequestException as exc:
            out[name] = {"error": type(exc).__name__ + ": " + str(exc)[:120]}
    return out
