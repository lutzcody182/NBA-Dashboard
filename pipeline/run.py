"""Command-line entry point, used by the GitHub Actions jobs.

    python -m pipeline.run update            # refresh this season, injuries, models, and data/
    python -m pipeline.run setup             # first-time history load (three past seasons + injury reports)
    python -m pipeline.run backfill 2025     # load one past season from NBA box scores (slow)
    python -m pipeline.run export            # rebuild data/ from the database only
"""
import argparse
import json
import logging
import sys
from datetime import datetime

from . import db
from .export.build import export_all
from .ingest.update import (PLAYS_FROM_YEAR, backfill_plays, backfill_regular_season, load_archive_season,
                            update_current_season)
from .providers.nba_cdn import NbaCdnProvider


def _last_completed_start_year(now=None) -> int:
    """Start year of the most recent finished regular season (2025 for 2025-26 once it is over)."""
    now = now or datetime.utcnow()
    return now.year - 1 if now.month >= 7 else now.year - 2


def pick_provider(conn, log=print):
    """NBA's own feed when it answers; ESPN otherwise.

    Once a season has games stored from ESPN it stays on ESPN, so the same
    game is never stored twice under two different IDs.
    """
    from .providers.espn import EspnProvider
    from .providers.nba_cdn import FeedError
    espn = EspnProvider(conn)
    sid = f"{espn.start_year}-{str(espn.start_year + 1)[-2:]}"
    if conn.execute("SELECT 1 FROM games WHERE season_id = ? AND source = 'espn' LIMIT 1", (sid,)).fetchone():
        db.mark_attempt(conn, "data_source", True, f"ESPN (used for all of {sid} once it started there)")
        return espn
    nba = NbaCdnProvider()
    try:
        nba.get_schedule()
        db.mark_attempt(conn, "data_source", True, "NBA (cdn.nba.com)")
        return nba
    except (FeedError, Exception) as exc:
        log(f"NBA schedule unavailable ({exc}); using ESPN")
        db.mark_attempt(conn, "data_source", True, f"ESPN (NBA's feed: {exc})")
        return espn


def run_injuries(conn) -> dict:
    from .injuries.ingest import fetch_latest
    try:
        res = fetch_latest(conn)
        ok = res.get("status") in ("stored", "already stored")
        db.mark_attempt(conn, "injuries", ok, json.dumps(res)[:400])
        return res
    except Exception as exc:  # the dashboard keeps the last good report
        db.mark_attempt(conn, "injuries", False, str(exc))
        return {"status": "error", "error": str(exc)}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("update", help="refresh the current season and rebuild the JSON")
    up.add_argument("--refresh-days", type=int, default=2)
    sub.add_parser("setup", help="load three past seasons of play-by-play and last season's injury reports")
    bf = sub.add_parser("backfill", help="load a past regular season from NBA box scores")
    bf.add_argument("start_year", type=int, help="e.g. 2025 for the 2025-26 season")
    sub.add_parser("export", help="rebuild the JSON from the database only")
    sub.add_parser("notify-test", help="send a test phone notification (needs NTFY_TOPIC)")
    args = parser.parse_args(argv)
    if args.cmd == "notify-test":
        from .notify import send_test
        print(json.dumps(send_test(), indent=2))
        return 0

    conn = db.connect()
    provider = pick_provider(conn, logging.info) if args.cmd in ("update", "setup") else NbaCdnProvider()
    result = {"provider": provider.name}
    if args.cmd == "setup":
        last = _last_completed_start_year()
        # Three seasons: ratings use the last two; the minutes model needs one more so that
        # early-season games in its training data have a previous season behind them.
        result["archive"] = [load_archive_season(conn, y) for y in (last - 2, last - 1, last)]
        from .injuries.ingest import backfill_season
        sid = f"{last}-{str(last + 1)[-2:]}"
        try:
            result["injury_history"] = backfill_season(conn, sid)
            db.mark_attempt(conn, "injury_history", True, json.dumps(result["injury_history"]))
        except Exception as exc:
            result["injury_history"] = {"error": str(exc)}
            db.mark_attempt(conn, "injury_history", False, str(exc))
        result["update"] = update_current_season(conn, provider)
        result["injuries"] = run_injuries(conn)
    elif args.cmd == "update":
        # one time: build last season's play log from the archive (used by the analytics)
        done = conn.execute("SELECT 1 FROM data_freshness WHERE dataset = ? AND status = 'ok'",
                            (f"plays_{PLAYS_FROM_YEAR}",)).fetchone()
        if not done:
            try:
                result["plays_backfill"] = backfill_plays(conn, PLAYS_FROM_YEAR)
            except Exception as exc:
                result["plays_backfill"] = {"error": str(exc)}
                db.mark_attempt(conn, f"plays_{PLAYS_FROM_YEAR}", False, str(exc))
        result["update"] = update_current_season(conn, provider, args.refresh_days)
        result["injuries"] = run_injuries(conn)
    if args.cmd in ("update", "setup"):
        from .diagnostics import probe
        result["sources"] = probe()
        db.mark_attempt(conn, "source_check", True,
                        ", ".join(f"{k}: {v.get('status', v.get('error'))}" for k, v in result["sources"].items()))
    elif args.cmd == "backfill":
        bf_result = backfill_regular_season(conn, provider, args.start_year)
        bf_result["missing_count"] = len(bf_result["missing"])
        bf_result["missing"] = bf_result["missing"][:20]
        result["backfill"] = bf_result
    result["export"] = export_all(conn)
    if args.cmd == "update":
        from .notify import notify_new_bets
        try:
            result["notify"] = notify_new_bets(conn)
        except Exception as exc:          # a notification problem must never stop the site updating
            result["notify"] = {"status": "error", "error": str(exc)}
    conn.close()
    print(json.dumps(result, indent=2, default=str))

    # Always publish what we have: a feed that fails shows as a red "source failed" marker on
    # the dashboard's data status instead of stopping the site from updating at all.
    return 0


if __name__ == "__main__":
    sys.exit(main())
