"""Small helpers used across the pipeline."""
import re
from datetime import datetime, timezone

_DURATION = re.compile(r"^PT(?:(\d+)M)?(?:([\d.]+)S)?$")


def utc_now_iso() -> str:
    """Current time in UTC as '2026-10-20T23:30:00Z'."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def duration_to_seconds(value) -> int:
    """Convert NBA's ISO-8601 durations ('PT25M01.00S', 'PT240M') to whole seconds.

    Empty or missing values (players who did not play) return 0.
    """
    if not value:
        return 0
    m = _DURATION.match(str(value).strip())
    if not m:
        raise ValueError(f"Unrecognised duration: {value!r}")
    minutes = int(m.group(1) or 0)
    seconds = float(m.group(2) or 0)
    return int(round(minutes * 60 + seconds))


def season_from_game_id(game_id: str) -> str:
    """'0022600001' -> '2026-27'. Characters 4-5 hold the season's start year."""
    start = 2000 + int(game_id[3:5])
    return f"{start}-{str(start + 1)[-2:]}"


def et_date(value: str) -> str:
    """Take the calendar date from an Eastern-time timestamp ('2026-10-20T19:30:00-04:00')."""
    return str(value)[:10]


def safe_div(num, den):
    """Division that returns None instead of failing when the denominator is 0 or missing."""
    if num is None or not den:
        return None
    return num / den
