-- =====================================================================
-- NBA Dashboard — database schema (SQLite)
-- Goes in: db/schema.sql
--
-- Conventions
--   * IDs: NBA's own IDs are canonical (team_id like 1610612738,
--     player_id like 1628369, game_id like '0022600001').
--     Other providers' IDs map in through id_map.
--   * Times: stored as ISO-8601 text in UTC ('2026-10-20T23:30:00Z').
--     Game dates are the US/Eastern calendar date ('2026-10-20').
--   * data_status on anything that can be uncertain:
--       confirmed | projected | estimated | missing
--   * fetched_at = when we pulled it; source = which provider.
--   * Nothing is overwritten for history: rosters use date ranges,
--     injuries and odds are stored as timestamped snapshots.
-- =====================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- 1. Reference data
-- ---------------------------------------------------------------------

CREATE TABLE seasons (
    season_id     TEXT PRIMARY KEY,          -- '2026-27'
    start_date    TEXT NOT NULL,
    end_date      TEXT NOT NULL
);

CREATE TABLE teams (
    team_id       INTEGER PRIMARY KEY,       -- NBA team ID
    abbr          TEXT NOT NULL UNIQUE,      -- 'BOS'
    city          TEXT NOT NULL,
    name          TEXT NOT NULL,             -- 'Celtics'
    conference    TEXT CHECK (conference IN ('East','West')),
    division      TEXT,
    arena_lat     REAL,                      -- for travel distance
    arena_lon     REAL,
    time_zone     TEXT                       -- 'America/New_York'
);

CREATE TABLE players (
    player_id     INTEGER PRIMARY KEY,       -- NBA player ID
    full_name     TEXT NOT NULL,
    first_name    TEXT,
    last_name     TEXT,
    birth_date    TEXT,
    height_in     INTEGER,
    weight_lb     INTEGER,
    position      TEXT,                      -- as listed: 'G', 'F-C', ...
    first_season  TEXT,
    fetched_at    TEXT NOT NULL
);

-- Maps other providers' IDs (ESPN, Odds API, injury-PDF names) to ours.
CREATE TABLE id_map (
    entity_type   TEXT NOT NULL CHECK (entity_type IN ('team','player','game')),
    provider      TEXT NOT NULL,             -- 'espn', 'odds_api', 'injury_pdf'
    provider_key  TEXT NOT NULL,             -- their ID, or raw name string
    internal_id   TEXT NOT NULL,             -- our ID (as text)
    match_method  TEXT,                      -- 'exact', 'manual', 'fuzzy'
    PRIMARY KEY (entity_type, provider, provider_key)
);

-- ---------------------------------------------------------------------
-- 2. Rosters and transactions (date ranges, so history is preserved)
-- ---------------------------------------------------------------------

CREATE TABLE roster_stints (
    player_id     INTEGER NOT NULL REFERENCES players(player_id),
    team_id       INTEGER NOT NULL REFERENCES teams(team_id),
    start_date    TEXT NOT NULL,
    end_date      TEXT,                      -- NULL = still on the team
    contract_type TEXT,                      -- 'standard', 'two_way', '10_day'
    jersey        TEXT,
    source        TEXT NOT NULL,
    fetched_at    TEXT NOT NULL,
    PRIMARY KEY (player_id, team_id, start_date)
);
CREATE INDEX idx_roster_team ON roster_stints(team_id, end_date);

CREATE TABLE transactions (
    txn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_date      TEXT NOT NULL,
    player_id     INTEGER REFERENCES players(player_id),
    from_team_id  INTEGER REFERENCES teams(team_id),
    to_team_id    INTEGER REFERENCES teams(team_id),
    txn_type      TEXT,                      -- 'trade', 'signing', 'waived', ...
    description   TEXT,
    source        TEXT NOT NULL,
    fetched_at    TEXT NOT NULL
);

-- ---------------------------------------------------------------------
-- 3. Games and results
-- ---------------------------------------------------------------------

CREATE TABLE games (
    game_id       TEXT PRIMARY KEY,          -- '0022600001'
    season_id     TEXT NOT NULL REFERENCES seasons(season_id),
    season_type   TEXT NOT NULL CHECK (season_type IN
                    ('preseason','regular','play_in','playoffs','cup_final')),
    game_date     TEXT NOT NULL,             -- Eastern date
    tip_utc       TEXT,
    home_team_id  INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id  INTEGER NOT NULL REFERENCES teams(team_id),
    arena         TEXT,
    neutral_site  INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL CHECK (status IN
                    ('scheduled','live','final','postponed','cancelled')),
    home_score    INTEGER,
    away_score    INTEGER,
    periods       INTEGER,                   -- 4 = regulation, 5+ = OT
    source        TEXT NOT NULL,
    fetched_at    TEXT NOT NULL
);
CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_home ON games(home_team_id, game_date);
CREATE INDEX idx_games_away ON games(away_team_id, game_date);

CREATE TABLE game_officials (
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    official_id   INTEGER NOT NULL,
    official_name TEXT NOT NULL,
    PRIMARY KEY (game_id, official_id)
);

-- Team box score, one row per team per game.
CREATE TABLE team_game_stats (
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    team_id       INTEGER NOT NULL REFERENCES teams(team_id),
    is_home       INTEGER NOT NULL,
    minutes       REAL,                      -- 240 regulation, +25 per OT
    pts INTEGER, fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER,
    ftm INTEGER, fta INTEGER, oreb INTEGER, dreb INTEGER, ast INTEGER,
    stl INTEGER, blk INTEGER, tov INTEGER, pf INTEGER,
    pts_paint     INTEGER,
    pts_second_chance INTEGER,
    pts_fast_break INTEGER,
    pts_off_tov   INTEGER,
    possessions   REAL,                      -- computed from play-by-play
    source        TEXT NOT NULL,
    fetched_at    TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id)
);

-- Player box score, one row per player per game (DNPs included).
CREATE TABLE player_game_stats (
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    player_id     INTEGER NOT NULL REFERENCES players(player_id),
    team_id       INTEGER NOT NULL REFERENCES teams(team_id),
    started       INTEGER NOT NULL DEFAULT 0,
    seconds       INTEGER NOT NULL DEFAULT 0,
    pts INTEGER, fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER,
    ftm INTEGER, fta INTEGER, oreb INTEGER, dreb INTEGER, ast INTEGER,
    stl INTEGER, blk INTEGER, tov INTEGER, pf INTEGER,
    plus_minus    INTEGER,
    dnp_reason    TEXT,                      -- NULL if played
    source        TEXT NOT NULL,
    fetched_at    TEXT NOT NULL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX idx_pgs_player ON player_game_stats(player_id);

-- ---------------------------------------------------------------------
-- 4. Play-by-play and lineup stints
--    (Large: kept in a separate file per season, db/pbp_2026-27.sqlite,
--     so the main database stays well under GitHub's 100 MB limit.)
-- ---------------------------------------------------------------------

CREATE TABLE pbp_events (
    game_id       TEXT NOT NULL,
    event_num     INTEGER NOT NULL,
    period        INTEGER NOT NULL,
    clock_sec     REAL NOT NULL,             -- seconds left in the period
    team_id       INTEGER,
    player1_id    INTEGER,                   -- shooter / fouler / etc.
    player2_id    INTEGER,                   -- assister / fouled / etc.
    player3_id    INTEGER,                   -- blocker / etc.
    event_type    TEXT NOT NULL,             -- 'shot','ft','rebound','tov','foul','sub',...
    sub_type      TEXT,
    shot_value    INTEGER,                   -- 2 or 3
    shot_made     INTEGER,
    shot_x        REAL,
    shot_y        REAL,
    shot_dist_ft  REAL,
    home_score    INTEGER,
    away_score    INTEGER,
    description   TEXT,
    PRIMARY KEY (game_id, event_num)
);

-- Continuous stretches where the same 10 players are on the floor.
-- Built from play-by-play; powers on/off, lineup ratings and pace splits.
CREATE TABLE lineup_stints (
    game_id       TEXT NOT NULL,
    stint_num     INTEGER NOT NULL,
    team_id       INTEGER NOT NULL,
    period        INTEGER NOT NULL,
    start_sec     REAL NOT NULL,             -- elapsed game seconds
    end_sec       REAL NOT NULL,
    lineup_key    TEXT NOT NULL,             -- 5 player IDs, sorted, '-' joined
    opp_lineup_key TEXT NOT NULL,
    possessions   REAL NOT NULL,
    pts_for       INTEGER NOT NULL,
    pts_against   INTEGER NOT NULL,
    score_margin_start INTEGER,              -- for leading/trailing splits
    PRIMARY KEY (game_id, stint_num, team_id)
);
CREATE INDEX idx_stints_lineup ON lineup_stints(lineup_key);

-- ---------------------------------------------------------------------
-- 5. Injuries (every official report kept, never overwritten)
-- ---------------------------------------------------------------------

CREATE TABLE injury_reports (
    report_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    report_time_et TEXT NOT NULL UNIQUE,     -- time printed on the PDF
    source_url    TEXT NOT NULL,
    fetched_at    TEXT NOT NULL,
    parse_ok      INTEGER NOT NULL,          -- 0 if the PDF failed to parse cleanly
    parse_notes   TEXT
);

CREATE TABLE injury_report_entries (
    report_id     INTEGER NOT NULL REFERENCES injury_reports(report_id),
    game_id       TEXT REFERENCES games(game_id),
    team_id       INTEGER NOT NULL REFERENCES teams(team_id),
    player_id     INTEGER REFERENCES players(player_id),  -- NULL if unmatched
    player_name_raw TEXT,                    -- exactly as printed
    status        TEXT NOT NULL CHECK (status IN
                    ('out','doubtful','questionable','probable','available',
                     'not_yet_submitted')),
    reason        TEXT,
    category      TEXT                       -- 'injury_illness','g_league','personal',...
);
CREATE INDEX idx_inj_game ON injury_report_entries(game_id, team_id);

-- ---------------------------------------------------------------------
-- 6. Lineups (projected until confirmed at tip)
-- ---------------------------------------------------------------------

CREATE TABLE game_lineups (
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    team_id       INTEGER NOT NULL REFERENCES teams(team_id),
    player_id     INTEGER NOT NULL REFERENCES players(player_id),
    is_starter    INTEGER NOT NULL,
    data_status   TEXT NOT NULL CHECK (data_status IN
                    ('confirmed','projected','estimated','missing')),
    as_of         TEXT NOT NULL,
    source        TEXT NOT NULL,             -- 'model' or provider
    PRIMARY KEY (game_id, team_id, player_id, as_of)
);

-- ---------------------------------------------------------------------
-- 7. Odds (Phase 5 — defined now so the shape is settled)
-- ---------------------------------------------------------------------

CREATE TABLE odds_snapshots (
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    fetched_at    TEXT NOT NULL,
    book          TEXT NOT NULL,             -- 'draftkings', 'fanduel', ...
    market        TEXT NOT NULL,             -- 'h2h','spreads','totals','player_points',...
    player_id     INTEGER REFERENCES players(player_id),  -- NULL for game markets
    side          TEXT NOT NULL,             -- team abbr, 'over', 'under'
    line          REAL,                      -- NULL for moneyline
    price_american INTEGER NOT NULL,
    is_closing    INTEGER NOT NULL DEFAULT 0,
    source        TEXT NOT NULL,
    PRIMARY KEY (game_id, fetched_at, book, market, player_id, side, line)
);

-- ---------------------------------------------------------------------
-- 8. Model runs, pregame snapshots, projections
--    Everything the dashboard shows for a game is tied to a run, so a
--    backtest can see exactly what was known before tip-off.
-- ---------------------------------------------------------------------

CREATE TABLE model_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    run_type      TEXT NOT NULL,             -- 'morning','pregame','backtest'
    model_version TEXT NOT NULL,             -- e.g. 'sim-0.1.0'
    git_sha       TEXT,
    data_cutoff   TEXT NOT NULL,             -- latest timestamp of inputs allowed
    status        TEXT NOT NULL CHECK (status IN ('running','ok','failed')),
    notes         TEXT
);

CREATE TABLE game_projections (
    run_id        INTEGER NOT NULL REFERENCES model_runs(run_id),
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    n_sims        INTEGER NOT NULL,
    home_win_prob REAL NOT NULL,
    home_pts_mean REAL, away_pts_mean REAL,
    home_pts_median REAL, away_pts_median REAL,
    spread_mean   REAL,                      -- home minus away
    total_mean    REAL,
    total_p10     REAL, total_p90 REAL,
    possessions_mean REAL,
    ot_prob       REAL,
    blowout_prob  REAL,                      -- margin >= 20
    inputs_path   TEXT,                      -- data/snapshots/... JSON of inputs
    PRIMARY KEY (run_id, game_id)
);

CREATE TABLE player_projections (
    run_id        INTEGER NOT NULL REFERENCES model_runs(run_id),
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    player_id     INTEGER NOT NULL REFERENCES players(player_id),
    availability  TEXT NOT NULL,             -- scenario label, e.g. 'active','if_out'
    play_prob     REAL NOT NULL,
    minutes_mean  REAL, minutes_p10 REAL, minutes_p90 REAL,
    pts_mean REAL, reb_mean REAL, ast_mean REAL, fg3m_mean REAL,
    stl_mean REAL, blk_mean REAL, tov_mean REAL,
    dist_path     TEXT,                      -- JSON of full simulated distributions
    PRIMARY KEY (run_id, game_id, player_id, availability)
);

-- ---------------------------------------------------------------------
-- 9. Data freshness (feeds meta.json and the "last updated" labels)
-- ---------------------------------------------------------------------

CREATE TABLE data_freshness (
    dataset       TEXT PRIMARY KEY,          -- 'schedule','rosters','injuries','box_scores','odds'
    last_attempt_at TEXT,
    last_success_at TEXT,
    status        TEXT NOT NULL CHECK (status IN ('ok','stale','failed')),
    detail        TEXT
);
