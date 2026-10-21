"""Test helpers: build box-score JSON in NBA's real feed format.

Field names and nesting copy NBA's live box score exactly (see
fixtures/boxscore_0022000180.json, a real response). The numbers are
made up for tests only and never reach the dashboard.
"""
import copy
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def real_box() -> dict:
    return json.loads((FIXTURES / "boxscore_0022000180.json").read_text())


def player(pid, first, last, *, starter=False, played=True, secs=1500, pts=10, fgm=4, fga=9,
           fg3m=1, fg3a=3, ftm=1, fta=2, oreb=1, dreb=3, ast=2, stl=1, blk=0, tov=1, pf=2,
           pm=0, status="ACTIVE", reason=None, position=None):
    m, s = divmod(secs, 60)
    p = {
        "status": status,
        "order": 1,
        "personId": pid,
        "jerseyNum": str(pid % 100),
        "starter": "1" if starter else "0",
        "oncourt": "0",
        "played": "1" if played else "0",
        "statistics": {
            "assists": ast, "blocks": blk, "fieldGoalsAttempted": fga, "fieldGoalsMade": fgm,
            "foulsPersonal": pf, "freeThrowsAttempted": fta, "freeThrowsMade": ftm,
            "minutes": f"PT{m:02d}M{s:02d}.00S" if played else "",
            "plusMinusPoints": float(pm), "points": pts,
            "reboundsDefensive": dreb, "reboundsOffensive": oreb, "reboundsTotal": oreb + dreb,
            "steals": stl, "threePointersAttempted": fg3a, "threePointersMade": fg3m,
            "turnovers": tov,
        },
        "name": f"{first} {last}", "nameI": f"{first[0]}. {last}",
        "firstName": first, "familyName": last,
    }
    if position:
        p["position"] = position
    if reason:
        p["notPlayingReason"] = reason
    return p


def team_block(team_id, tricode, city, name, players):
    played = [p for p in players if p["played"] == "1"]
    tot = lambda k: sum(p["statistics"][k] for p in played)
    secs = sum(
        int(p["statistics"]["minutes"][2:4]) * 60 + int(p["statistics"]["minutes"][5:7]) for p in played
    )
    m = secs // 60
    return {
        "teamId": team_id, "teamName": name, "teamCity": city, "teamTricode": tricode,
        "score": tot("points"), "periods": [],
        "players": players,
        "statistics": {
            "points": tot("points"), "fieldGoalsMade": tot("fieldGoalsMade"),
            "fieldGoalsAttempted": tot("fieldGoalsAttempted"),
            "threePointersMade": tot("threePointersMade"),
            "threePointersAttempted": tot("threePointersAttempted"),
            "freeThrowsMade": tot("freeThrowsMade"), "freeThrowsAttempted": tot("freeThrowsAttempted"),
            "reboundsOffensive": tot("reboundsOffensive"), "reboundsDefensive": tot("reboundsDefensive"),
            "assists": tot("assists"), "steals": tot("steals"), "blocks": tot("blocks"),
            "turnovers": tot("turnovers"), "turnoversTotal": tot("turnovers"),
            "foulsPersonal": tot("foulsPersonal"),
            "minutes": f"PT{m}M00.00S",
            "pointsInThePaint": 40, "pointsSecondChance": 12, "pointsFastBreak": 10,
            "pointsFromTurnovers": 15,
        },
    }


def make_box(game_id, date, home, away, status=3, period=4):
    """home/away: (team_id, tricode, city, name, [players])."""
    b = copy.deepcopy(real_box())
    g = b["game"]
    g.update({
        "gameId": game_id, "gameStatus": status, "period": period,
        "gameEt": f"{date}T19:30:00-04:00", "gameTimeUTC": f"{date}T23:30:00Z",
    })
    g["homeTeam"] = team_block(*home)
    g["awayTeam"] = team_block(*away)
    return b
