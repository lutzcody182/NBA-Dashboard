"""The play log: possessions, shot context and rebound outcomes from NBA-style actions."""
from pipeline.pbp.parse import parse_actions
from pipeline.pbp.plays import extract, shot_family, shot_xy

H, A = 1610612737, 1610612738
HP = [1, 2, 3, 4, 5]
AP = [11, 12, 13, 14, 15]


def act(n, clock, kind, team, person, **kw):
    a = {"actionNumber": n, "orderNumber": n, "period": 1, "clock": clock, "actionType": kind, "teamId": team,
         "personId": person, "possession": team}
    a.update(kw)
    return a


def game():
    acts = [
        act(1, "PT11M50.00S", "2pt", H, 1, shotResult="Made", area="Restricted Area", subType="Layup",
            descriptor="driving", assistPersonId=2, xLegacy=0, yLegacy=10, shotDistance=1.0, scoreHome=2),
        act(2, "PT11M30.00S", "3pt", A, 11, shotResult="Missed", area="Above the Break 3", subType="Jump Shot",
            descriptor="pullup", xLegacy=100, yLegacy=230, shotDistance=25.1),
        act(3, "PT11M28.00S", "rebound", A, 12, subType="offensive"),
        act(4, "PT11M26.00S", "2pt", A, 12, shotResult="Made", area="Restricted Area", subType="Layup",
            descriptor="putback", xLegacy=0, yLegacy=5, shotDistance=0.5, scoreAway=2),
        act(5, "PT11M26.00S", "freethrow", A, 12, subType="1 of 1", shotResult="Made", scoreAway=3),
        act(6, "PT11M10.00S", "turnover", H, 3, subType="bad pass", stealPersonId=13),
        act(7, "PT11M10.00S", "steal", A, 13),
        act(8, "PT11M05.00S", "2pt", A, 13, shotResult="Missed", area="Restricted Area", subType="DUNK",
            xLegacy=0, yLegacy=3, shotDistance=0.3),
        act(9, "PT11M03.00S", "rebound", H, 4, subType="defensive"),
        act(10, "PT10M50.00S", "2pt", H, 4, shotResult="Missed", area="Mid-Range", subType="Jump Shot",
            descriptor="step back", xLegacy=-120, yLegacy=100, shotDistance=15.6),
        act(11, "PT10M48.00S", "rebound", A, 0, subType="defensive"),
    ]
    return acts


def test_possessions_and_context():
    acts = game()
    box = {H: HP, A: AP}
    parse_actions("g1", acts, H, A, box)
    plays, poss = extract("g1", acts, H, A)
    fg = [p for p in plays if p["kind"] == "fg"]
    # points: possessions add up to the score (the and-one free throw belongs to the putback possession)
    assert sum(p["pts"] for p in poss if p["team_id"] == H) == 2
    assert sum(p["pts"] for p in poss if p["team_id"] == A) == 3
    kinds = [(p["team_id"], p["kind"]) for p in poss]
    assert kinds[:4] == [(H, "period"), (A, "made"), (H, "made"), (A, "steal")]
    assert poss[1]["oreb"] == 1 and poss[1]["pts"] == 3
    # the missed three was rebounded by the offense; the putback is marked
    assert fg[1]["reb_off"] == 1 and fg[1]["reb_player"] == 12
    assert fg[2]["putback"] == 1 and fg[2]["family"] == "tip"
    # the steal starts the next possession, and the dunk 5 seconds later is in it
    assert fg[3]["poss_kind"] == "steal" and fg[3]["poss_age"] == 5.0 and fg[3]["reb_off"] == 0
    assert [p for p in plays if p["kind"] == "tov"][0]["other_id"] == 13
    # the team rebound after the step-back miss is a defensive one with no player
    assert fg[4]["family"] == "stepback" and fg[4]["reb_off"] == 0 and fg[4]["reb_player"] is None
    assert fg[0]["other_id"] == 2 and fg[0]["dist"] == 1.0
    assert all(p["stint_num"] == 1 for p in plays)


def test_shot_family_and_location():
    assert shot_family({"espnType": "Driving Floating Bank Jump Shot"}) == "floater"
    assert shot_family({"espnType": "Alley Oop Dunk Shot"}) == "dunk"
    assert shot_family({"espnType": "Layup Shot Putback"}) == "tip"
    assert shot_family({"espnType": "Turnaround Fade Away Jump Shot"}) == "fadeaway"
    assert shot_family({"espnType": "Running Pullup Jump Shot"}) == "pullup"
    assert shot_family({"espnType": "Jump Shot"}) == "jumper"
    assert shot_family({"subType": "Jump Shot", "descriptor": "step back"}) == "stepback"
    assert shot_family({"espnType": "Heave Jump Shot"}) == ""
    x, y, d = shot_xy({"xFt": 37, "yFt": 13, "description": "makes 17-foot pullup jump shot"})
    assert (x, y, d) == (12.0, 13.0, 17.0)
    x, y, d = shot_xy({"xLegacy": -179, "yLegacy": 179, "shotDistance": 25.33})
    assert (x, y, d) == (-17.9, 17.9, 25.33)
