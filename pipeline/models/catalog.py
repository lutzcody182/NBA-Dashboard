"""The sub-ratings: what each one measures, and how it is computed.

Each entry is a rate: a numerator column over a denominator column, summed
over a player's games. `better` says which direction is good. Everything
here is computed from box scores and play-by-play; nothing needs player
tracking data. Ratings that would need tracking data are listed in
UNAVAILABLE with the reason, rather than faked.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Stat:
    key: str
    label: str
    group: str            # offense | defense
    made: str             # numerator column
    tries: str            # denominator column
    scale: float          # display multiplier (100 for %, 100 for per-100 poss)
    unit: str             # how the value is shown
    better: int           # +1 higher is better, -1 lower is better
    how: str              # plain-language definition shown on the profile
    min_tries: float      # evidence needed before a percentile is shown

    @property
    def tries_label(self) -> str:
        return TRIES_LABELS.get(self.tries, "chances")


def _prep(df):
    """Add the derived numerator/denominator columns the catalog refers to."""
    df["ts_pts"] = df["pts"] / 2.0
    df["fga_mid_all"] = df["fga_mid"]
    df["fga_close"] = df["fga_rim"]
    df["tov_chances"] = df["fga"] + 0.44 * df["fta"] + df["tov"]
    df["ast_chances"] = (df["on_fgm"] - df["fgm"]).clip(lower=0)
    df["oreb_chances"] = df["on_oreb"] + df["on_opp_dreb"]
    df["dreb_chances"] = df["on_dreb"] + df["on_opp_oreb"]
    df["opp_efg_made"] = df["on_opp_fgm"] + 0.5 * df["on_opp_fg3m"]
    df["opp_3pa"] = df["on_opp_fg3a"]
    df["usage_n"] = df["fga"] + 0.44 * df["fta"] + df["tov"]
    return df


STATS = [
    # Offense
    Stat("scoring_eff", "Scoring efficiency", "offense", "ts_pts", "tsa", 100, "TS%", 1,
         "True shooting: points per shooting possession, counting threes and free throws.", 150),
    Stat("scoring_vol", "Scoring volume", "offense", "pts", "on_poss", 100, "pts / 100 poss", 1,
         "Points scored per 100 of his team's possessions while he is on the floor.", 500),
    Stat("three_pt", "Three-point shooting", "offense", "fg3m", "fg3a", 100, "3P%", 1,
         "Share of three-point attempts made.", 60),
    Stat("three_vol", "Three-point volume", "offense", "fg3a", "on_poss", 100, "3PA / 100 poss", 1,
         "Three-point attempts per 100 possessions on the floor.", 500),
    Stat("midrange", "Midrange scoring", "offense", "fgm_mid", "fga_mid", 100, "FG% mid", 1,
         "Field goal % on two-pointers outside the paint.", 40),
    Stat("finishing", "Finishing at the rim", "offense", "fgm_rim", "fga_rim", 100, "FG% at rim", 1,
         "Field goal % in the restricted area (within 4 feet).", 50),
    Stat("playmaking", "Playmaking", "offense", "ast", "ast_chances", 100, "AST%", 1,
         "Share of teammates' baskets he assisted while on the floor.", 150),
    Stat("assist_rate", "Assist creation", "offense", "ast", "on_poss", 100, "AST / 100 poss", 1,
         "Assists per 100 possessions on the floor.", 500),
    Stat("ball_security", "Turnover avoidance", "offense", "tov", "tov_chances", 100, "TOV%", -1,
         "Turnovers per possession he ends (shots, free-throw trips, turnovers). Lower is better.", 150),
    Stat("off_reb", "Offensive rebounding", "offense", "oreb", "oreb_chances", 100, "OREB%", 1,
         "Share of his team's missed shots he rebounds while on the floor.", 150),
    Stat("ft_gen", "Free throw generation", "offense", "fta", "on_poss", 100, "FTA / 100 poss", 1,
         "Free throw attempts drawn per 100 possessions on the floor.", 500),
    Stat("usage", "Usage", "offense", "usage_n", "on_poss", 100, "plays / 100 poss", 1,
         "Shots, free-throw trips and turnovers per 100 possessions on the floor.", 500),
    # Defense
    Stat("rim_prot", "Rim protection", "defense", "blk", "on_opp_fga_rim", 100, "BLK per 100 opp rim shots", 1,
         "Blocks per 100 opponent shots at the rim while he is on the floor.", 150),
    Stat("rim_fg_allowed", "Opponent rim FG%", "defense", "on_opp_fgm_rim", "on_opp_fga_rim", 100,
         "opp FG% at rim", -1,
         "Opponents' field goal % at the rim while he is on the floor. Team defense affects this too.", 300),
    Stat("perimeter", "Perimeter defense", "defense", "opp_3pa", "on_opp_poss", 100, "opp 3PA / 100 poss", -1,
         "Opponent three-point attempts per 100 possessions while he is on the floor. Tracks how well "
         "a defense runs shooters off the line; three-point % allowed is mostly luck, so it is not used.", 800),
    Stat("steals", "Steal generation", "defense", "stl", "on_opp_poss", 100, "STL / 100 opp poss", 1,
         "Steals per 100 opponent possessions on the floor.", 500),
    Stat("def_reb", "Defensive rebounding", "defense", "dreb", "dreb_chances", 100, "DREB%", 1,
         "Share of opponent misses he rebounds while on the floor.", 150),
    Stat("fouls", "Foul avoidance", "defense", "pf", "on_opp_poss", 100, "fouls / 100 opp poss", -1,
         "Personal fouls per 100 opponent possessions on the floor. Lower is better.", 500),
    Stat("shot_supp", "Opponent shot suppression", "defense", "opp_efg_made", "on_opp_fga", 100, "opp eFG%", -1,
         "Opponents' effective field goal % while he is on the floor. Team defense affects this too.", 500),
]

STAT_BY_KEY = {s.key: s for s in STATS}

TRIES_LABELS = {
    "tsa": "shooting possessions", "on_poss": "possessions", "fg3a": "three-point attempts",
    "fga_mid": "midrange attempts", "fga_rim": "shots at the rim", "ast_chances": "teammate baskets",
    "tov_chances": "plays", "oreb_chances": "rebound chances", "dreb_chances": "rebound chances",
    "on_opp_fga_rim": "opponent shots at the rim", "on_opp_poss": "opponent possessions",
    "on_opp_fga": "opponent shots",
}

UNAVAILABLE = {
    "def_versatility": ("Defensive versatility",
                        "Needs player-tracking matchup data (who guarded whom), which this dashboard does not use."),
    "matchup_sensitivity": ("Matchup sensitivity", "Added with the matchup analysis in Phase 4."),
}
