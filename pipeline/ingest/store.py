"""Write provider data into the database.

Every write is an upsert, so re-running the same day is safe and stat
corrections published later simply replace the old numbers.
"""
from .. import config
from ..providers.base import BoxScore, TeamRef
from ..stats.formulas import estimate_possessions
from ..util import utc_now_iso


def season_type(game_id: str) -> str:
    return config.SEASON_TYPE_BY_PREFIX[game_id[:3]]


def upsert_team(conn, t: TeamRef) -> None:
    conf, div = config.TEAM_GROUPS.get(t.team_id, (None, None))
    conn.execute(
        """INSERT INTO teams (team_id, abbr, city, name, conference, division)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(team_id) DO UPDATE SET
             abbr = excluded.abbr, city = excluded.city, name = excluded.name,
             conference = excluded.conference, division = excluded.division""",
        (t.team_id, t.tricode, t.city, t.name, conf, div),
    )


def ensure_season(conn, season_id: str, game_date: str) -> None:
    """Create the season row, widening its date range as games arrive."""
    conn.execute(
        """INSERT INTO seasons (season_id, start_date, end_date) VALUES (?, ?, ?)
           ON CONFLICT(season_id) DO UPDATE SET
             start_date = MIN(start_date, excluded.start_date),
             end_date   = MAX(end_date, excluded.end_date)""",
        (season_id, game_date, game_date),
    )


def _upsert_game(conn, *, game_id, season_id, game_date, tip_utc, home_id, away_id,
                 arena, neutral, status, home_score, away_score, periods, source, fetched_at):
    conn.execute(
        """INSERT INTO games (game_id, season_id, season_type, game_date, tip_utc,
                              home_team_id, away_team_id, arena, neutral_site, status,
                              home_score, away_score, periods, source, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(game_id) DO UPDATE SET
             game_date = excluded.game_date, tip_utc = excluded.tip_utc,
             arena = COALESCE(excluded.arena, games.arena),
             neutral_site = excluded.neutral_site,
             status = excluded.status,
             home_score = COALESCE(excluded.home_score, games.home_score),
             away_score = COALESCE(excluded.away_score, games.away_score),
             periods = COALESCE(excluded.periods, games.periods),
             source = excluded.source, fetched_at = excluded.fetched_at""",
        (game_id, season_id, season_type(game_id), game_date, tip_utc, home_id, away_id,
         arena, int(neutral), status, home_score, away_score, periods, source, fetched_at),
    )


def store_schedule(conn, games: list, source: str) -> int:
    now = utc_now_iso()
    for g in games:
        upsert_team(conn, g.home)
        upsert_team(conn, g.away)
        ensure_season(conn, g.season_id, g.game_date)
        # A final box score is more authoritative than the schedule file, so
        # never let the schedule downgrade a game we already have as final.
        current = conn.execute("SELECT status FROM games WHERE game_id = ?", (g.game_id,)).fetchone()
        status = g.status
        if current and current["status"] == "final" and status != "final":
            status = "final"
        _upsert_game(
            conn, game_id=g.game_id, season_id=g.season_id, game_date=g.game_date,
            tip_utc=g.tip_utc, home_id=g.home.team_id, away_id=g.away.team_id,
            arena=g.arena, neutral=g.neutral_site, status=status,
            home_score=g.home_score, away_score=g.away_score, periods=None,
            source=source, fetched_at=now,
        )
    conn.commit()
    return len(games)


_TEAM_COLS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
              "ast", "stl", "blk", "tov", "pf")


def store_boxscore(conn, box: BoxScore, source: str) -> None:
    now = utc_now_iso()
    for side in (box.home, box.away):
        upsert_team(conn, side.team)
    ensure_season(conn, box.season_id, box.game_date)
    _upsert_game(
        conn, game_id=box.game_id, season_id=box.season_id, game_date=box.game_date,
        tip_utc=box.tip_utc, home_id=box.home.team.team_id, away_id=box.away.team.team_id,
        arena=box.arena, neutral=False, status=box.status,
        home_score=box.home.pts, away_score=box.away.pts, periods=box.periods or None,
        source=source, fetched_at=now,
    )
    # Neutral-site flag comes from the schedule; keep it if it was set there.
    poss = {}
    for side, opp in ((box.home, box.away), (box.away, box.home)):
        poss[side.team.team_id] = estimate_possessions(
            fga=side.fga, fta=side.fta, oreb=side.oreb, tov=side.tov, fgm=side.fgm,
            opp_dreb=opp.dreb,
            opp_fga=opp.fga, opp_fta=opp.fta, opp_oreb=opp.oreb, opp_tov=opp.tov,
            opp_fgm=opp.fgm, dreb=side.dreb,
        )
    for side in (box.home, box.away):
        vals = [getattr(side, c) for c in _TEAM_COLS]
        conn.execute(
            f"""INSERT INTO team_game_stats (game_id, team_id, is_home, minutes,
                    {", ".join(_TEAM_COLS)}, pts_paint, pts_second_chance,
                    pts_fast_break, pts_off_tov, possessions, source, fetched_at)
                VALUES (?, ?, ?, ?, {", ".join("?" * len(_TEAM_COLS))}, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, team_id) DO UPDATE SET
                  is_home = excluded.is_home, minutes = excluded.minutes,
                  {", ".join(f"{c} = excluded.{c}" for c in _TEAM_COLS)},
                  pts_paint = excluded.pts_paint, pts_second_chance = excluded.pts_second_chance,
                  pts_fast_break = excluded.pts_fast_break, pts_off_tov = excluded.pts_off_tov,
                  possessions = excluded.possessions, source = excluded.source,
                  fetched_at = excluded.fetched_at""",
            (box.game_id, side.team.team_id, int(side.is_home), side.seconds / 60, *vals,
             side.pts_paint, side.pts_second_chance, side.pts_fast_break, side.pts_off_tov,
             poss[side.team.team_id], source, now),
        )
        for p in side.players:
            _store_player_line(conn, box.game_id, p, source, now)
    for oid, name in box.officials:
        conn.execute(
            "INSERT OR REPLACE INTO game_officials (game_id, official_id, official_name) VALUES (?, ?, ?)",
            (box.game_id, oid, name),
        )
    update_rosters_from_box(conn, box, source, now)
    conn.commit()


def _store_player_line(conn, game_id, p, source, now):
    conn.execute(
        """INSERT INTO players (player_id, full_name, first_name, last_name, position, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(player_id) DO UPDATE SET
             full_name = excluded.full_name, first_name = excluded.first_name,
             last_name = excluded.last_name,
             position = COALESCE(excluded.position, players.position),
             fetched_at = excluded.fetched_at""",
        (p.player_id, p.full_name, p.first_name, p.last_name, p.position, now),
    )
    conn.execute(
        """INSERT INTO player_game_stats (game_id, player_id, team_id, started, seconds,
               pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf,
               plus_minus, dnp_reason, source, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(game_id, player_id) DO UPDATE SET
             team_id = excluded.team_id, started = excluded.started, seconds = excluded.seconds,
             pts = excluded.pts, fgm = excluded.fgm, fga = excluded.fga,
             fg3m = excluded.fg3m, fg3a = excluded.fg3a, ftm = excluded.ftm, fta = excluded.fta,
             oreb = excluded.oreb, dreb = excluded.dreb, ast = excluded.ast, stl = excluded.stl,
             blk = excluded.blk, tov = excluded.tov, pf = excluded.pf,
             plus_minus = excluded.plus_minus, dnp_reason = excluded.dnp_reason,
             source = excluded.source, fetched_at = excluded.fetched_at""",
        (game_id, p.player_id, p.team_id, int(p.started), p.seconds, p.pts, p.fgm, p.fga,
         p.fg3m, p.fg3a, p.ftm, p.fta, p.oreb, p.dreb, p.ast, p.stl, p.blk, p.tov, p.pf,
         p.plus_minus, p.dnp_reason, source, now),
    )


def update_rosters_from_box(conn, box: BoxScore, source: str, now: str) -> None:
    """Keep roster history in step with game-day rosters.

    A box score lists every player on a team's game-day roster (active and
    inactive). So:
      * a player listed for a new team -> close the old stint, open a new one
      * a player with an open stint who is missing from his team's box score
        -> close that stint (waived, traded or released)
    Changes are only applied when this game is at least as recent as the
    stint it would change, so processing old games never rewrites the present.
    """
    for side in (box.home, box.away):
        team_id = side.team.team_id
        listed = {p.player_id: p for p in side.players}
        for pid, p in listed.items():
            open_stint = conn.execute(
                "SELECT team_id, start_date FROM roster_stints WHERE player_id = ? AND end_date IS NULL",
                (pid,),
            ).fetchone()
            if open_stint and open_stint["team_id"] == team_id:
                conn.execute(
                    "UPDATE roster_stints SET jersey = COALESCE(?, jersey), fetched_at = ? "
                    "WHERE player_id = ? AND team_id = ? AND end_date IS NULL",
                    (p.jersey, now, pid, team_id),
                )
                continue
            if open_stint and open_stint["start_date"] > box.game_date:
                continue  # we already know about a later move
            if open_stint:
                conn.execute(
                    "UPDATE roster_stints SET end_date = ? WHERE player_id = ? AND end_date IS NULL",
                    (box.game_date, pid),
                )
            conn.execute(
                """INSERT OR IGNORE INTO roster_stints
                     (player_id, team_id, start_date, end_date, jersey, source, fetched_at)
                   VALUES (?, ?, ?, NULL, ?, ?, ?)""",
                (pid, team_id, box.game_date, p.jersey, source, now),
            )
        for row in conn.execute(
            "SELECT player_id, start_date FROM roster_stints WHERE team_id = ? AND end_date IS NULL",
            (team_id,),
        ).fetchall():
            if row["player_id"] not in listed and row["start_date"] <= box.game_date:
                conn.execute(
                    "UPDATE roster_stints SET end_date = ? WHERE player_id = ? AND team_id = ? AND end_date IS NULL",
                    (box.game_date, row["player_id"], team_id),
                )
