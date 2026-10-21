"""Fetch the official injury report, link it to games and players, and learn play rates."""
import logging
import time
from datetime import datetime

import requests

from .. import config
from ..util import utc_now_iso
from .report import ET, match_player, parse_pdf, report_urls_before

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
UNCERTAIN = ("questionable", "doubtful")
MIN_RATE_SAMPLE = 40


def _get_pdf(session, url):
    try:
        r = session.get(url, timeout=30, headers=HEADERS)
    except requests.RequestException as exc:
        return None, str(exc)
    if r.status_code == 200 and r.content[:4] == b"%PDF":
        return r.content, None
    return None, f"HTTP {r.status_code}"


def fetch_latest(conn, session=None, now_et=None, steps=12) -> dict:
    """Find and store the newest report published at or before now. Returns a summary."""
    session = session or requests.Session()
    now_et = now_et or datetime.now(ET)
    for stamp, url in report_urls_before(now_et, steps):
        data, err = _get_pdf(session, url)
        if data is None:
            continue
        have = conn.execute("SELECT report_id FROM injury_reports WHERE source_url = ?", (url,)).fetchone()
        if have:
            return {"url": url, "status": "already stored", "report_id": have[0]}
        parsed = parse_pdf(data)
        rid = store_report(conn, parsed, url)
        return {"url": url, "status": "stored", "report_id": rid, "entries": len(parsed.entries),
                "parse_ok": parsed.ok, "unparsed_lines": parsed.unparsed_lines}
    return {"status": "no report found", "tried": steps}


def _team_lookup(conn):
    names = {}
    for r in conn.execute("SELECT team_id, abbr, city, name FROM teams"):
        tid = r[0]
        if tid not in config.NBA_TEAM_IDS:
            continue
        nick = config.NICKNAMES.get(tid) or r[3]
        names[tid] = (nick.lower(), r[1])
    return names


def _resolve_team(printed, lookup):
    if not printed:
        return None
    p = printed.lower()
    hits = [tid for tid, (nick, _) in lookup.items() if nick and p.endswith(nick)]
    return hits[0] if len(hits) == 1 else None


def _abbr_to_id(lookup):
    return {abbr: tid for tid, (_, abbr) in lookup.items()}


def _candidates(conn, team_id, date):
    """Players who could be on this team on this date: open roster stints plus recent appearances."""
    rows = conn.execute(
        """SELECT DISTINCT pl.player_id, pl.full_name FROM players pl
           WHERE pl.player_id IN (
               SELECT player_id FROM roster_stints WHERE team_id = ? AND start_date <= ?
                      AND (end_date IS NULL OR end_date >= ?)
               UNION
               SELECT p.player_id FROM player_game_stats p JOIN games g ON g.game_id = p.game_id
               WHERE p.team_id = ? AND g.game_date <= ? AND g.game_date >= date(?, '-400 days'))""",
        (team_id, date, date, team_id, date, date)).fetchall()
    return {r[0]: r[1] for r in rows}


def store_report(conn, parsed, url) -> int:
    now = utc_now_iso()
    report_time = parsed.report_time_et or now[:16]
    lookup = _team_lookup(conn)
    by_abbr = _abbr_to_id(lookup)
    notes = []
    cur = conn.execute(
        """INSERT INTO injury_reports (report_time_et, source_url, fetched_at, parse_ok, parse_notes)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(report_time_et) DO UPDATE SET source_url = excluded.source_url,
             fetched_at = excluded.fetched_at, parse_ok = excluded.parse_ok, parse_notes = excluded.parse_notes""",
        (report_time, url, now, int(parsed.ok), None))
    rid = cur.lastrowid or conn.execute(
        "SELECT report_id FROM injury_reports WHERE report_time_et = ?", (report_time,)).fetchone()[0]
    conn.execute("DELETE FROM injury_report_entries WHERE report_id = ?", (rid,))
    unmatched = 0
    for e in parsed.entries:
        team_id = _resolve_team(e.team, lookup)
        if team_id is None:
            notes.append(f"unknown team {e.team!r}")
            continue
        game_id = None
        if e.matchup and e.game_date:
            away, home = e.matchup.split("@")
            row = conn.execute(
                "SELECT game_id FROM games WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?",
                (e.game_date, by_abbr.get(home), by_abbr.get(away))).fetchone()
            game_id = row[0] if row else None
        player_id = None
        if e.player:
            pid, how = match_player(e.player, _candidates(conn, team_id, e.game_date or report_time[:10]))
            if pid is None:
                unmatched += 1
                notes.append(f"{e.player} ({e.team}): {how}")
            player_id = pid
        reason = e.reason or ""
        category = ("g_league" if "g league" in reason.lower() else
                    "personal" if "personal" in reason.lower() else
                    "rest" if "rest" in reason.lower() else
                    "injury_illness" if "injury" in reason.lower() or "illness" in reason.lower() else None)
        conn.execute(
            """INSERT INTO injury_report_entries (report_id, game_id, team_id, player_id, player_name_raw,
                   status, reason, category) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (rid, game_id, team_id, player_id, e.player, e.status, reason, category))
    detail = f"{len(parsed.entries)} entries, {unmatched} names not matched, {parsed.unparsed_lines} lines not read"
    if parsed.used_fallback_edges:
        detail += "; header not found, used stored column positions"
    conn.execute("UPDATE injury_reports SET parse_notes = ? WHERE report_id = ?",
                 (detail + ("; " + "; ".join(notes[:15]) if notes else ""), rid))
    conn.commit()
    return rid


def backfill_season(conn, season_id: str, session=None, hour_et: int = 17, pause: float = 0.4) -> dict:
    """Store one report per game day of a past season (the last one before about 5:30 PM ET).

    These are used to learn how often questionable, doubtful and probable players end up playing.
    """
    session = session or requests.Session()
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT game_date FROM games WHERE season_id = ? AND season_type = 'regular' ORDER BY 1",
        (season_id,))]
    summary = {"dates": len(dates), "stored": 0, "missing": 0}
    for d in dates:
        when = datetime.strptime(d, "%Y-%m-%d").replace(hour=hour_et, minute=45, tzinfo=ET)
        got = False
        for _, url in report_urls_before(when, steps=8):
            if conn.execute("SELECT 1 FROM injury_reports WHERE source_url = ?", (url,)).fetchone():
                got = True
                break
            data, _ = _get_pdf(session, url)
            time.sleep(pause)
            if data is None:
                continue
            store_report(conn, parse_pdf(data), url)
            summary["stored"] += 1
            got = True
            break
        if not got:
            summary["missing"] += 1
    return summary


def play_rates(conn) -> dict:
    """How often players with each status ended up playing, from stored reports and box scores.

    Uses each player's last listing published before his game's tip-off. A
    rate is only reported when at least MIN_RATE_SAMPLE listings back it up.
    """
    rows = conn.execute(
        """SELECT e.game_id, e.player_id, e.status, r.report_time_et, g.tip_utc,
                  COALESCE(p.seconds, 0) AS seconds
           FROM injury_report_entries e JOIN injury_reports r ON r.report_id = e.report_id
           JOIN games g ON g.game_id = e.game_id
           LEFT JOIN player_game_stats p ON p.game_id = e.game_id AND p.player_id = e.player_id
           WHERE e.player_id IS NOT NULL AND g.status = 'final' AND r.parse_ok = 1""").fetchall()
    last = {}
    for gid, pid, status, rt, tip, secs in rows:
        if tip:
            tip_et = datetime.fromisoformat(tip.replace("Z", "+00:00")).astimezone(ET).strftime("%Y-%m-%dT%H:%M")
            if rt > tip_et:
                continue
        key = (gid, pid)
        if key not in last or rt > last[key][0]:
            last[key] = (rt, status, secs > 0)
    counts = {}
    for _, status, played in last.values():
        c = counts.setdefault(status, [0, 0])
        c[0] += 1
        c[1] += int(played)
    return {s: {"n": n, "played": k, "rate": k / n if n >= MIN_RATE_SAMPLE else None}
            for s, (n, k) in counts.items()}
