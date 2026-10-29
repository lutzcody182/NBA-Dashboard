"""Database access: open the SQLite file, create tables, record data freshness.

The schema is versioned with SQLite's user_version. A database saved by an
older version of the pipeline is upgraded in place the next time it opens.
"""
import sqlite3
from pathlib import Path

from . import config
from .util import utc_now_iso

SCHEMA_VERSION = 3

_STINT_STATS = (
    "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "tov", "pf", "ast", "stl", "blk",
    "pts_paint", "pts_2nd", "pts_fb", "pts_off_tov",
    "fga_rim", "fga_paint", "fga_mid", "fga_c3", "fga_ab3",
    "fgm_rim", "fgm_paint", "fgm_mid", "fgm_c3", "fgm_ab3",
)
_PLAYER_ZONES = ("fga_rim", "fga_paint", "fga_mid", "fga_c3", "fga_ab3",
                 "fgm_rim", "fgm_paint", "fgm_mid", "fgm_c3", "fgm_ab3")

MIGRATIONS = {
    # Version 2: play-by-play derived tables, injury scenarios, model outputs.
    2: f"""
        DROP TABLE IF EXISTS lineup_stints;
        CREATE TABLE lineup_stints (
            game_id       TEXT NOT NULL,
            stint_num     INTEGER NOT NULL,
            team_id       INTEGER NOT NULL,
            opp_id        INTEGER NOT NULL,
            is_home       INTEGER NOT NULL,
            period        INTEGER NOT NULL,
            start_sec     REAL NOT NULL,
            end_sec       REAL NOT NULL,
            lineup_key    TEXT NOT NULL,      -- 5 player IDs, sorted, '-' joined
            opp_lineup_key TEXT NOT NULL,
            poss          INTEGER NOT NULL,   -- this team's offensive possessions
            opp_poss      INTEGER NOT NULL,
            pts           INTEGER NOT NULL,
            opp_pts       INTEGER NOT NULL,
            margin_start  INTEGER NOT NULL,   -- this team's lead when the stint began
            ok            INTEGER NOT NULL,   -- 0 if the lineup could not be verified
            {", ".join(f"{c} INTEGER NOT NULL DEFAULT 0" for c in _STINT_STATS)},
            PRIMARY KEY (game_id, stint_num, team_id)
        );
        CREATE INDEX idx_stints_team ON lineup_stints(team_id);

        CREATE TABLE IF NOT EXISTS player_game_pbp (
            game_id       TEXT NOT NULL,
            player_id     INTEGER NOT NULL,
            team_id       INTEGER NOT NULL,
            seconds       REAL NOT NULL,
            starter       INTEGER NOT NULL,
            on_poss       INTEGER NOT NULL,   -- team offensive possessions while on court
            opp_poss      INTEGER NOT NULL,   -- opponent possessions while on court
            and1          INTEGER NOT NULL DEFAULT 0,
            {", ".join(f"{c} INTEGER NOT NULL DEFAULT 0" for c in _PLAYER_ZONES)},
            PRIMARY KEY (game_id, player_id)
        );
        CREATE INDEX IF NOT EXISTS idx_pgp_player ON player_game_pbp(player_id);

        CREATE TABLE IF NOT EXISTS pbp_games (
            game_id       TEXT PRIMARY KEY,
            events        INTEGER NOT NULL,
            bad_events    INTEGER NOT NULL,
            quality       REAL NOT NULL,
            stints        INTEGER NOT NULL,
            bad_stints    INTEGER NOT NULL,
            notes         TEXT,
            source        TEXT NOT NULL,
            fetched_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_calibration (
            key           TEXT PRIMARY KEY,   -- e.g. 'ratings:2025-26'
            payload       TEXT NOT NULL,      -- JSON
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS injury_scenarios (
            run_id        INTEGER NOT NULL,
            game_id       TEXT NOT NULL,
            team_id       INTEGER NOT NULL,
            scenario_key  TEXT NOT NULL,      -- sorted player IDs assumed OUT among the uncertain ones
            probability   REAL,               -- NULL until play rates are calibrated
            minutes_json  TEXT NOT NULL,      -- {{player_id: [mean, p10, p90]}}
            PRIMARY KEY (run_id, game_id, team_id, scenario_key)
        );
    """,
}

MIGRATIONS[3] = """
    CREATE TABLE IF NOT EXISTS shot_locations (
        game_id       TEXT NOT NULL,
        shot_num      INTEGER NOT NULL,   -- order in the game
        team_id       INTEGER NOT NULL,   -- shooting team
        def_team_id   INTEGER NOT NULL,   -- defending team
        player_id     INTEGER NOT NULL,
        period        INTEGER NOT NULL,
        clock         TEXT,
        x             REAL NOT NULL,      -- feet across the court, basket at 25
        y             REAL NOT NULL,      -- feet out from the basket
        zone          TEXT,               -- rim, paint, mid, c3, ab3
        points        INTEGER NOT NULL,   -- 2 or 3 attempted
        made          INTEGER NOT NULL,
        PRIMARY KEY (game_id, shot_num)
    );
    CREATE INDEX IF NOT EXISTS idx_shots_team ON shot_locations(team_id);
    CREATE INDEX IF NOT EXISTS idx_shots_def ON shot_locations(def_team_id);
    CREATE INDEX IF NOT EXISTS idx_shots_player ON shot_locations(player_id);
"""

# Columns added to existing tables (ALTER TABLE cannot run inside executescript safely
# when the column may already exist, so these are applied one by one).
ADDED_COLUMNS = {
    2: [
        ("team_game_stats", "poss_source", "TEXT"),
    ],
}


def connect(path: Path = config.DB_PATH) -> sqlite3.Connection:
    """Open the database, creating or upgrading the tables as needed."""
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not _has_tables(conn):
        conn.executescript(config.SCHEMA_PATH.read_text())
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    migrate(conn)
    return conn


def migrate(conn) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0] or 1
    for v in range(version + 1, SCHEMA_VERSION + 1):
        if v in MIGRATIONS:
            conn.executescript(MIGRATIONS[v])
        for table, col, decl in ADDED_COLUMNS.get(v, []):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        conn.execute(f"PRAGMA user_version = {v}")
        conn.commit()


def _has_tables(conn) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='games'"
    ).fetchone()
    return row is not None


def mark_attempt(conn, dataset: str, ok: bool, detail: str = "") -> None:
    """Record that an update was tried, and whether it worked.

    A failed attempt keeps the old last_success_at, so the dashboard can show
    how stale that dataset really is.
    """
    now = utc_now_iso()
    existing = conn.execute(
        "SELECT last_success_at FROM data_freshness WHERE dataset = ?", (dataset,)
    ).fetchone()
    last_success = now if ok else (existing["last_success_at"] if existing else None)
    conn.execute(
        """INSERT INTO data_freshness (dataset, last_attempt_at, last_success_at, status, detail)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(dataset) DO UPDATE SET
             last_attempt_at = excluded.last_attempt_at,
             last_success_at = excluded.last_success_at,
             status = excluded.status,
             detail = excluded.detail""",
        (dataset, now, last_success, "ok" if ok else "failed", detail[:500]),
    )
    conn.commit()
