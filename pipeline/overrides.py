"""Record the odds, line and book you actually got for a recommended bet.

The Bets page opens a prefilled GitHub issue titled "Bet override #<bet_id>"
whose body carries one machine-readable line:

    override: {"bet_id": 12, "price": -115, "line": 26.5, "book": "FanDuel"}

or, to go back to the recommended odds:

    override: {"bet_id": 12, "clear": true}

The override workflow (only for issues opened by the repository owner) reads
that one line, checks every field (whole American odds of at least +/-100, a
whole or half line, a short plain book name) and saves it in overrides.json
in the repository: the permanent record. Every update run then applies the
file to the bets table (apply_file), so an override can never be lost to a
skipped run. The recommended price, line and book stay on the bet; results
and units are graded at what you got. A bet that was already graded is set
back to open so the next grading uses your numbers.

The parsing and file parts use only the standard library, so the override
workflow runs without installing anything.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

LINE_RE = re.compile(r"^\s*override:\s*(\{.*\})\s*$", re.MULTILINE)
BOOK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .'&+-]{0,29}$")
FILE = Path("overrides.json")


class OverrideError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(body: str) -> dict:
    m = LINE_RE.search(body or "")
    if not m:
        raise OverrideError("no 'override: {...}' line found")
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise OverrideError(f"the override line is not valid: {exc}") from exc
    if not isinstance(d, dict):
        raise OverrideError("the override line is not valid")
    try:
        bet_id = int(d.get("bet_id"))
        if bet_id <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise OverrideError("missing bet number") from exc
    if d.get("clear") is True:
        return {"bet_id": bet_id, "clear": True}
    try:
        price = int(round(float(d.get("price"))))
    except (TypeError, ValueError) as exc:
        raise OverrideError("odds must be a number like -115 or +130") from exc
    if abs(price) < 100 or abs(price) > 20000:
        raise OverrideError("odds must be American odds, -100 or lower or +100 or higher")
    line = d.get("line")
    if line not in (None, ""):
        try:
            line = float(line)
        except (TypeError, ValueError) as exc:
            raise OverrideError("line must be a number") from exc
        if abs(line) > 200 or abs(line * 2 - round(line * 2)) > 1e-9:
            raise OverrideError("line must be a whole or half number")
    else:
        line = None
    book = str(d.get("book") or "").strip()
    if not BOOK_RE.match(book):
        raise OverrideError("book name must be 1 to 30 letters, numbers or spaces")
    return {"bet_id": bet_id, "price": price, "line": line, "book": book}


def load(path: Path = FILE) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text() or "{}")
    return {str(k): v for k, v in (data.get("bets") or {}).items()}


def record(body: str, path: Path = FILE) -> dict:
    """Validate one override and save it to the file. Returns what was saved."""
    o = parse(body)
    bets = load(path)
    key = str(o["bet_id"])
    if o.get("clear"):
        bets.pop(key, None)
        out = {"bet_id": o["bet_id"], "cleared": True}
    else:
        bets[key] = {"price": o["price"], "line": o["line"], "book": o["book"], "at": _now()}
        out = dict(o)
    Path(path).write_text(json.dumps({"bets": dict(sorted(bets.items(), key=lambda kv: int(kv[0])))}, indent=1) + "\n")
    return out


def message(r: dict) -> str:
    if r.get("cleared"):
        return f"Bet #{r['bet_id']}: back to the recommended odds. The site updates in a few minutes."
    odds = f"+{r['price']}" if r["price"] > 0 else str(r["price"])
    line = "the recommended line" if r["line"] is None else f"line {r['line']:g}"
    return (f"Recorded bet #{r['bet_id']} at {odds}, {line}, {r['book']}. It will be graded at these numbers; "
            "the site updates in a few minutes.")


def apply_file(conn, path: Path = FILE) -> dict:
    """Bring the bets table in line with overrides.json (idempotent; runs on every update)."""
    from .export.bets import ensure_table
    ensure_table(conn)
    want = load(path)
    rows = conn.execute("SELECT bet_id, line, status, taken_price, taken_line, taken_book FROM bets").fetchall()
    changed, unknown = 0, sorted(set(want) - {str(r[0]) for r in rows}, key=int)
    for bet_id, line, status, tp, tl, tb in rows:
        o = want.get(str(bet_id))
        new = (o["price"], o["line"] if o["line"] is not None else line, o["book"]) if o else (None, None, None)
        if (tp, tl, tb) == new:
            continue
        reset = ", status = 'open', result = NULL, pnl = NULL, graded_at = NULL" \
            if status in ("won", "lost", "push") else ""
        conn.execute(f"UPDATE bets SET taken_price = ?, taken_line = ?, taken_book = ?, taken_at = ?{reset} "
                     "WHERE bet_id = ?", (*new, o["at"] if o else None, bet_id))
        changed += 1
    conn.commit()
    return {"overrides": len(want), "changed": changed, "unknown_bets": unknown}


def main() -> int:
    """For the override workflow: record ISSUE_BODY into overrides.json, write the reply to override_result.txt."""
    import os
    try:
        msg = message(record(os.environ.get("ISSUE_BODY", "")))
        ok = True
    except OverrideError as exc:
        msg, ok = f"Could not record this override: {exc}. Nothing was changed.", False
    print(msg)
    Path("override_result.txt").write_text(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
