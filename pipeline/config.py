"""Shared settings: file locations and the fixed team reference table."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "db"
DB_PATH = DB_DIR / "nba.sqlite"
SCHEMA_PATH = DB_DIR / "schema.sql"
DATA_DIR = ROOT / "data"

# Conference and division for the 30 franchises, keyed by NBA team ID.
# Names, cities and tricodes are NOT hard-coded: they come from the feeds,
# so a rename or relocation is picked up automatically.
TEAM_GROUPS = {
    1610612737: ("East", "Southeast"),  # ATL
    1610612738: ("East", "Atlantic"),   # BOS
    1610612739: ("East", "Central"),    # CLE
    1610612740: ("West", "Southwest"),  # NOP
    1610612741: ("East", "Central"),    # CHI
    1610612742: ("West", "Southwest"),  # DAL
    1610612743: ("West", "Northwest"),  # DEN
    1610612744: ("West", "Pacific"),    # GSW
    1610612745: ("West", "Southwest"),  # HOU
    1610612746: ("West", "Pacific"),    # LAC
    1610612747: ("West", "Pacific"),    # LAL
    1610612748: ("East", "Southeast"),  # MIA
    1610612749: ("East", "Central"),    # MIL
    1610612750: ("West", "Northwest"),  # MIN
    1610612751: ("East", "Atlantic"),   # BKN
    1610612752: ("East", "Atlantic"),   # NYK
    1610612753: ("East", "Southeast"),  # ORL
    1610612754: ("East", "Central"),    # IND
    1610612755: ("East", "Atlantic"),   # PHI
    1610612756: ("West", "Pacific"),    # PHX
    1610612757: ("West", "Northwest"),  # POR
    1610612758: ("West", "Pacific"),    # SAC
    1610612759: ("West", "Southwest"),  # SAS
    1610612760: ("West", "Northwest"),  # OKC
    1610612761: ("East", "Atlantic"),   # TOR
    1610612762: ("West", "Northwest"),  # UTA
    1610612763: ("West", "Southwest"),  # MEM
    1610612764: ("East", "Southeast"),  # WAS
    1610612765: ("East", "Central"),    # DET
    1610612766: ("East", "Southeast"),  # CHA
}
NBA_TEAM_IDS = frozenset(TEAM_GROUPS)

# Team nicknames, used to read team names in the injury report PDF.
NICKNAMES = {
    1610612737: "Hawks", 1610612738: "Celtics", 1610612739: "Cavaliers", 1610612740: "Pelicans",
    1610612741: "Bulls", 1610612742: "Mavericks", 1610612743: "Nuggets", 1610612744: "Warriors",
    1610612745: "Rockets", 1610612746: "Clippers", 1610612747: "Lakers", 1610612748: "Heat",
    1610612749: "Bucks", 1610612750: "Timberwolves", 1610612751: "Nets", 1610612752: "Knicks",
    1610612753: "Magic", 1610612754: "Pacers", 1610612755: "76ers", 1610612756: "Suns",
    1610612757: "Trail Blazers", 1610612758: "Kings", 1610612759: "Spurs", 1610612760: "Thunder",
    1610612761: "Raptors", 1610612762: "Jazz", 1610612763: "Grizzlies", 1610612764: "Wizards",
    1610612765: "Pistons", 1610612766: "Hornets",
}

# Game ID prefix -> season type. Prefix 003 (All-Star) is skipped entirely.
SEASON_TYPE_BY_PREFIX = {
    "001": "preseason",
    "002": "regular",
    "004": "playoffs",
    "005": "play_in",
    "006": "cup_final",
}

# Only these count toward stats, leaders and team ratings in Phase 1.
STAT_SEASON_TYPES = ("regular",)

# Polite delay between box-score requests during backfills (seconds).
REQUEST_DELAY = 0.6
