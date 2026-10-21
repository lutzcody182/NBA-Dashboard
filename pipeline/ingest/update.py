"""The update jobs: refresh the current season, or backfill a past one."""
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .. import db
from ..pbp.parse import parse_actions
from .pbp_store import store_archive_game, store_pbp
from .store import store_boxscore, store_schedule

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def today_et() -> date:
    return datetime.now(ET).date()


def _games_needing_box(conn, schedule, refresh_days: int, now_utc_iso: str) -> list:
    """Pick which box scores to (re)fetch.

    * any game the schedule marks live or final that we have no final box for
    * any game whose tip time has passed but the schedule has not caught up
    * anything from the last `refresh_days` days, to pick up stat corrections
    """
    cutoff = (today_et() - timedelta(days=refresh_days)).isoformat()
    have_final = {
        r["game_id"]
        for r in conn.execute(
            "SELECT g.game_id FROM games g JOIN team_game_stats t ON t.game_id = g.game_id "
            "WHERE g.status = 'final'"
        )
    }
    wanted = []
    for g in schedule:
        if g.status in ("postponed", "cancelled"):
            continue
        started = g.status in ("live", "final") or (g.tip_utc and g.tip_utc <= now_utc_iso)
        if not started:
            continue
        if g.game_id not in have_final or g.game_date >= cutoff:
            wanted.append(g.game_id)
    return wanted


def update_current_season(conn, provider, refresh_days: int = 2) -> dict:
    from ..util import utc_now_iso

    summary = {"schedule_games": 0, "box_fetched": 0, "box_not_published": 0, "pbp_fetched": 0,
               "pbp_low_quality": 0, "errors": []}
    try:
        schedule = provider.get_schedule()
        summary["schedule_games"] = store_schedule(conn, schedule, provider.name)
        db.mark_attempt(conn, "schedule", True, f"{len(schedule)} games")
    except Exception as exc:  # keep going with what is already stored
        log.exception("Schedule update failed")
        db.mark_attempt(conn, "schedule", False, str(exc))
        summary["errors"].append(f"schedule: {exc}")
        return summary

    for gid in _games_needing_box(conn, schedule, refresh_days, utc_now_iso()):
        try:
            box = provider.get_boxscore(gid)
        except Exception as exc:
            log.warning("Box score %s failed: %s", gid, exc)
            summary["errors"].append(f"{gid}: {exc}")
            continue
        if box is None:
            summary["box_not_published"] += 1
        else:
            store_boxscore(conn, box, provider.name)
            summary["box_fetched"] += 1
            if box.status == "final" and hasattr(provider, "get_playbyplay"):
                _fetch_pbp(conn, provider, box.game_id, summary)
        provider.polite_pause()

    # Catch up on finals that have a box score but no play-by-play yet.
    missing = [r[0] for r in conn.execute(
        """SELECT g.game_id FROM games g JOIN team_game_stats t ON t.game_id = g.game_id AND t.is_home = 1
           LEFT JOIN pbp_games p ON p.game_id = g.game_id
           WHERE g.status = 'final' AND p.game_id IS NULL AND g.season_id = (SELECT MAX(season_id) FROM games)
           LIMIT 60""")]
    if missing and hasattr(provider, "get_playbyplay"):
        for gid in missing:
            _fetch_pbp(conn, provider, gid, summary)
            provider.polite_pause()
    if hasattr(provider, "get_playbyplay"):
        db.mark_attempt(conn, "play_by_play", True,
                        f'{summary["pbp_fetched"]} games parsed, {summary["pbp_low_quality"]} with unverified lineups')

    ok = not summary["errors"]
    db.mark_attempt(
        conn, "box_scores", ok,
        f'{summary["box_fetched"]} fetched, {summary["box_not_published"]} not yet published'
        + (f', {len(summary["errors"])} errors' if summary["errors"] else ""),
    )
    db.mark_attempt(conn, "rosters", ok, "from game-day box scores")
    return summary


def backfill_regular_season(conn, provider, start_year: int, max_games: int = 1230) -> dict:
    """Load a finished regular season by walking its game IDs (00225 00001 ... 01230).

    A past season's schedule file is not on the CDN, but its game IDs are
    sequential, and each box score carries the date, teams and arena.
    Games already stored as final are skipped, so a backfill can be resumed.
    """
    yy = f"{start_year % 100:02d}"
    have = {
        r["game_id"]
        for r in conn.execute(
            "SELECT g.game_id FROM games g JOIN team_game_stats t ON t.game_id = g.game_id "
            "WHERE g.status = 'final' AND g.game_id LIKE ?",
            (f"002{yy}%",),
        )
    }
    summary = {"fetched": 0, "skipped_existing": 0, "missing": [], "errors": []}
    for n in range(1, max_games + 1):
        gid = f"002{yy}{n:05d}"
        if gid in have:
            summary["skipped_existing"] += 1
            continue
        try:
            box = provider.get_boxscore(gid)
        except Exception as exc:
            summary["errors"].append(f"{gid}: {exc}")
            continue
        if box is None:
            summary["missing"].append(gid)
        elif box.status == "final":
            store_boxscore(conn, box, provider.name)
            summary["fetched"] += 1
        provider.polite_pause()
        if n % 100 == 0:
            log.info("Backfill %s: %d/%d", start_year, n, max_games)
    db.mark_attempt(
        conn, f"backfill_{start_year}", not summary["errors"],
        f'{summary["fetched"]} fetched, {len(summary["missing"])} missing, '
        f'{len(summary["errors"])} errors',
    )
    return summary


def _fetch_pbp(conn, provider, game_id, summary) -> None:
    """Fetch and store one game's play-by-play. The box score must already be stored."""
    row = conn.execute("SELECT home_team_id, away_team_id FROM games WHERE game_id = ?", (game_id,)).fetchone()
    if not row:
        return
    try:
        actions = provider.get_playbyplay(game_id)
    except Exception as exc:
        summary["errors"].append(f"pbp {game_id}: {exc}")
        return
    if not actions:
        return
    starters = {}
    for r in conn.execute("SELECT team_id, player_id FROM player_game_stats WHERE game_id = ? AND started = 1",
                          (game_id,)):
        starters.setdefault(r[0], []).append(r[1])
    g = parse_actions(game_id, actions, row[0], row[1], starters)
    store_pbp(conn, g, provider.name)
    conn.commit()
    summary["pbp_fetched"] += 1
    if g.quality < 0.995 or any(not s.ok for s in g.stints):
        summary["pbp_low_quality"] += 1


def load_archive_season(conn, start_year: int, frame=None) -> dict:
    """Load a full past regular season from the play-by-play archive.

    Skips games already parsed. `frame` lets tests pass a prepared DataFrame.
    """
    from ..providers.archive import download_season, split_games

    df = frame if frame is not None else download_season(start_year)
    have = {r[0] for r in conn.execute("SELECT game_id FROM pbp_games")}
    summary = {"season_start": start_year, "games": 0, "skipped": 0, "low_quality": 0}
    for gid, actions, home, away in split_games(df):
        if gid in have:
            summary["skipped"] += 1
            continue
        g = parse_actions(gid, actions, home, away)
        # A box score from NBA (if one was loaded earlier) is better than one rebuilt from
        # play-by-play, so only fill box-score rows when none exist.
        has_box = conn.execute("SELECT 1 FROM team_game_stats WHERE game_id = ?", (gid,)).fetchone()
        if has_box:
            store_pbp(conn, g, "archive")
        else:
            store_archive_game(conn, g, actions, "archive")
        summary["games"] += 1
        summary["low_quality"] += int(g.quality < 0.995 or any(not s.ok for s in g.stints))
        if summary["games"] % 200 == 0:
            conn.commit()
            log.info("Archive %s: %d games", start_year, summary["games"])
    conn.commit()
    db.mark_attempt(conn, f"archive_{start_year}", True,
                    f'{summary["games"]} games loaded, {summary["low_quality"]} with unverified lineups')
    return summary
