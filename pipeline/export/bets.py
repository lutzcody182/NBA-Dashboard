"""Recommended bets, with stakes, and a ledger that grades every one.

Candidates
  * player props: the props comparison (export/props.py), state-licensed US
    books only
  * game lines: moneyline, spread and total from the same books, against the
    game simulation. The spread and total use a normal curve around the
    simulated margin and total, with the miss size measured when the
    simulation was replayed on last season (its RMSE), after removing that
    replay's average bias
How much to trust the model
  The model has no live record yet, so the chance used for a bet is the
  average of the model's and the market's (BLEND). A bet needs:
    * expected return of at least MIN_EV at the best legal price
    * the model rating the side at least MIN_GAP above the market, and not
      more than MAX_GAP above it (bigger gaps usually mean news the model
      hasn't seen)
    * at least MIN_BOOKS legal books quoting it
    * props: the player has 5+ games of history and is not listed as
      questionable or doubtful
Stakes
  A quarter of the Kelly stake for the blended chance, where 1 unit = 1% of
  bankroll, rounded to 0.25 units, at most MAX_UNITS per bet and
  MAX_GAME_UNITS per game. One prop per player, and one side bet (moneyline
  or spread) per game. At most MAX_DAY_PICKS bets and MAX_DAY_UNITS units a
  day, best expected return first.
Ledger
  A pick is locked the first time it qualifies on prices at most
  FRESH_MINUTES old (any site update, not only right after a pull), at that
  price and book, and
  never changed. After the game it is graded from the box score or final
  score; props for players who don't play are void. Closing line value
  compares the market's fair chance at the last pull before tip-off with
  the fair chance when the bet was made.
"""
import json
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy import stats as st

from ..models import props as PM
from ..providers import odds_api as OA
from ..util import utc_now_iso

log = logging.getLogger(__name__)

BLEND = 0.5
MIN_EV = 0.04
MAX_GAP = 0.10
MIN_GAP = 0.02
MIN_BOOKS = 3
KELLY = 0.25
UNIT_PCT = 1.0
MAX_UNITS = 1.5
MAX_GAME_UNITS = 3.0
MAX_DAY_PICKS = 15
MAX_DAY_UNITS = 12.0
FRESH_MINUTES = 100       # picks are locked only from prices at most this old
MIN_PRICE, MAX_PRICE = -250, 250

TABLE = """
CREATE TABLE IF NOT EXISTS bets (
    bet_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    game_id       TEXT NOT NULL,
    kind          TEXT NOT NULL,         -- 'prop', 'moneyline', 'spread', 'total'
    player_id     INTEGER,               -- props only
    stat          TEXT,                  -- props only
    side          TEXT NOT NULL,         -- 'over'/'under' or 'home'/'away'
    line          REAL,
    price         INTEGER NOT NULL,
    book          TEXT NOT NULL,
    p_model       REAL NOT NULL,
    p_market      REAL NOT NULL,
    p_used        REAL NOT NULL,
    ev            REAL NOT NULL,
    units         REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',   -- open, won, lost, push, void
    result        REAL,
    pnl           REAL,
    close_fair    REAL,
    clv           REAL,
    graded_at     TEXT,
    settings      TEXT
);
CREATE INDEX IF NOT EXISTS idx_bets_game ON bets(game_id);
"""
KIND_LABEL = {"prop": "Player prop", "moneyline": "Moneyline", "spread": "Spread", "total": "Total"}


# ---------------------------------------------------------------- game lines

def game_market_table(odds: pd.DataFrame) -> list:
    """Per game and market: the most-offered line, the median fair chance and the best price on each side."""
    if odds.empty:
        return []
    out = []
    for (gid, market), g in odds.groupby(["game_id", "market"]):
        if market not in OA.GAME_MARKETS:
            continue
        a_side, b_side = ("over", "under") if market == "totals" else ("home", "away")
        if market == "h2h":
            g = g.assign(key_line=0.0)
        elif market == "spreads":
            # a spread pair shares one number: home -3.5 with away +3.5; key it by the home number
            g = g.assign(key_line=np.where(g["side"] == "home", g["line"], -g["line"]))
        else:
            g = g.assign(key_line=g["line"])
        pairs = g.pivot_table(index=["book", "key_line"], columns="side", values="price_american", aggfunc="first")
        if a_side not in pairs or b_side not in pairs:
            continue
        pairs = pairs.dropna(subset=[a_side, b_side]).reset_index()
        if pairs.empty:
            continue
        counts = Counter(pairs["key_line"])
        top = max(counts.values())
        mid = float(np.median(pairs["key_line"]))
        line = min((l for l, c in counts.items() if c == top), key=lambda l: (abs(l - mid), l))
        at = pairs[pairs["key_line"] == line]
        books = []
        for r in at.itertuples():
            fair, hold = PM.devig(getattr(r, a_side), getattr(r, b_side))
            if fair is not None:
                books.append({"book": r.book, a_side: int(getattr(r, a_side)), b_side: int(getattr(r, b_side)),
                              "fair": round(fair, 4), "hold": round(hold, 4)})
        if not books:
            continue
        best = {s: max(books, key=lambda b: PM.decimal(b[s])) for s in (a_side, b_side)}
        out.append({"game_id": gid, "market": market, "line": float(line), "sides": (a_side, b_side),
                    "fair": round(float(np.median([b["fair"] for b in books])), 4), "books": books,
                    "best": {s: {"price": best[s][s], "book": best[s]["book"]} for s in (a_side, b_side)}})
    return out


def game_probs(sim: dict, market: str, line: float, report: dict):
    """(P first side wins, P push) from the simulation: home for moneyline/spread, over for totals."""
    if market == "h2h":
        return float(sim["home_win"]), 0.0
    if market == "spreads":
        mu = float(sim["margin"]) - float(report.get("margin_bias") or 0.0)
        sd = float(report.get("margin_rmse") or 14.0)
        thr = -line                      # home covers when margin > -line
    else:
        mu = float(sim["total"]) - float(report.get("total_bias") or 0.0)
        sd = float(report.get("total_rmse") or 19.0)
        thr = line
    if float(thr).is_integer():
        win = 1 - st.norm.cdf(thr + 0.5, mu, sd)
        push = st.norm.cdf(thr + 0.5, mu, sd) - st.norm.cdf(thr - 0.5, mu, sd)
    else:
        win, push = 1 - st.norm.cdf(thr, mu, sd), 0.0
    return float(win), float(push)


# ---------------------------------------------------------------- candidates

def kelly_units(p: float, price: int) -> float:
    b = PM.decimal(price) - 1
    f = (p * b - (1 - p)) / b
    if f <= 0:
        return 0.0
    u = KELLY * f * 100 / UNIT_PCT
    u = min(MAX_UNITS, round(u * 4) / 4)
    return u if u >= 0.25 else 0.0


def _candidate(kind, game_id, side, line, price, book, p_model, p_market, push, books, **extra):
    """p_model / p_market: chance this side wins, pushes excluded."""
    if price is None or not (MIN_PRICE <= price <= MAX_PRICE) or books < MIN_BOOKS:
        return None
    gap = p_model - p_market
    if gap < MIN_GAP or gap > MAX_GAP:
        return None
    p = BLEND * p_model + (1 - BLEND) * p_market
    ev = (1 - push) * (p * (PM.decimal(price) - 1) - (1 - p))
    if ev < MIN_EV:
        return None
    units = kelly_units(p, price)
    if units <= 0:
        return None
    return {"kind": kind, "game_id": game_id, "side": side, "line": line, "price": int(price), "book": book,
            "p_model": round(p_model, 4), "p_market": round(p_market, 4), "p_used": round(p, 4),
            "ev": round(ev, 4), "units": units, "books": books, **extra}


def prop_candidates(rows: list) -> list:
    out = []
    for r in rows:
        m = r.get("model")
        if not m or m.get("status") in ("out", "doubtful", "questionable") or (m.get("games") or 0) < 5:
            continue
        po = m["p_over"] / (m["p_over"] + m["p_under"]) if m["p_over"] + m["p_under"] > 0 else None
        if po is None:
            continue
        n = len(r["books"])
        for side, p_mod, p_mkt in (("over", po, r["fair_over"]), ("under", 1 - po, 1 - r["fair_over"])):
            best = r["best_over"] if side == "over" else r["best_under"]
            c = _candidate("prop", r["game_id"], side, r["line"], best["price"], best["book"], p_mod, p_mkt,
                           m.get("p_push", 0.0), n, player_id=r["player_id"], stat=r["stat"],
                           model_mean=m["mean"], model_min=m["min"])
            if c:
                out.append(c)
    return out


def game_candidates(table: list, sims: dict) -> list:
    by_game = {g["game_id"]: g for g in (sims or {}).get("games", [])}
    report = ((sims or {}).get("model") or {}).get("report", {}).get("scores", {})
    out = []
    for t in table:
        sim = by_game.get(t["game_id"])
        if not sim:
            continue
        p_first, push = game_probs(sim, t["market"], t["line"], report)
        p_first_np = p_first / (1 - push) if push < 1 else 0.5
        a, b = t["sides"]
        kind = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}[t["market"]]
        for side, p_mod, p_mkt in ((a, p_first_np, t["fair"]), (b, 1 - p_first_np, 1 - t["fair"])):
            line = None if kind == "moneyline" else (t["line"] if side in ("home", "over", "under") else -t["line"])
            c = _candidate(kind, t["game_id"], side, line, t["best"][side]["price"], t["best"][side]["book"],
                           p_mod, p_mkt, push, len(t["books"]),
                           sim_margin=sim["margin"], sim_total=sim["total"])
            if c:
                out.append(c)
    return out


def select(cands: list) -> list:
    """One prop per player, one side bet per game, then the per-game stake cap."""
    cands = sorted(cands, key=lambda c: -c["ev"])
    picked, seen = [], set()
    for c in cands:
        if c["kind"] == "prop":
            key = ("prop", c["game_id"], c["player_id"])
        elif c["kind"] in ("moneyline", "spread"):
            key = ("side", c["game_id"])
        else:
            key = ("total", c["game_id"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
    by_game = defaultdict(list)
    for c in picked:
        by_game[c["game_id"]].append(c)
    out = []
    for gid, cs in by_game.items():
        tot = sum(c["units"] for c in cs)
        if tot > MAX_GAME_UNITS:
            f = MAX_GAME_UNITS / tot
            for c in cs:
                c["units"] = max(0.25, math.floor(c["units"] * f * 4) / 4)
            # still over after rounding up the small ones: drop the weakest
            cs.sort(key=lambda c: -c["ev"])
            while sum(c["units"] for c in cs) > MAX_GAME_UNITS and len(cs) > 1:
                cs.pop()
        out += cs
    out.sort(key=lambda c: -c["ev"])
    keep, units = [], 0.0
    for c in out:
        if len(keep) >= MAX_DAY_PICKS or units + c["units"] > MAX_DAY_UNITS:
            continue
        keep.append(c)
        units += c["units"]
    return keep


# ---------------------------------------------------------------- ledger

def lock(conn, picks: list, games: dict, now_iso: str) -> int:
    """Store picks not already on the ledger for that game, market and player. Returns how many were added."""
    settings = json.dumps({"blend": BLEND, "min_ev": MIN_EV, "max_gap": MAX_GAP, "kelly": KELLY,
                           "max_units": MAX_UNITS, "max_game_units": MAX_GAME_UNITS})
    added = 0
    for c in picks:
        g = games.get(c["game_id"])
        if not g or not g.get("tip_utc") or g["tip_utc"] <= now_iso:
            continue
        n_day, u_day = conn.execute(
            """SELECT COUNT(*), COALESCE(SUM(units), 0) FROM bets b JOIN games x ON x.game_id = b.game_id
               WHERE x.game_date = ?""", (g["game_date"],)).fetchone()
        if n_day >= MAX_DAY_PICKS or u_day + c["units"] > MAX_DAY_UNITS:
            continue
        if c["kind"] == "prop":
            exists = conn.execute("SELECT 1 FROM bets WHERE game_id=? AND kind='prop' AND player_id=? AND stat=?",
                                  (c["game_id"], c["player_id"], c["stat"])).fetchone()
        elif c["kind"] in ("moneyline", "spread"):
            exists = conn.execute("SELECT 1 FROM bets WHERE game_id=? AND kind IN ('moneyline','spread')",
                                  (c["game_id"],)).fetchone()
        else:
            exists = conn.execute("SELECT 1 FROM bets WHERE game_id=? AND kind='total'", (c["game_id"],)).fetchone()
        if exists:
            continue
        conn.execute(
            """INSERT INTO bets (created_at, game_id, kind, player_id, stat, side, line, price, book, p_model,
                 p_market, p_used, ev, units, settings) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_iso, c["game_id"], c["kind"], c.get("player_id"), c.get("stat"), c["side"], c["line"], c["price"],
             c["book"], c["p_model"], c["p_market"], c["p_used"], c["ev"], c["units"], settings))
        added += 1
    conn.commit()
    return added


def _closing_fair(conn, bet: dict, tip: str):
    """Market fair chance for the bet's side at the last legal-book pull before tip-off, at the same line."""
    q = """SELECT book, side, line, price_american FROM odds_snapshots
           WHERE game_id=? AND source=? AND fetched_at=(SELECT MAX(fetched_at) FROM odds_snapshots
                 WHERE game_id=? AND source=? AND fetched_at < ? AND {w}) AND {w}"""
    if bet["kind"] == "prop":
        mk = {v: k for k, v in OA.MARKETS.items()}[bet["stat"]]
        w = f"player_id={int(bet['player_id'])} AND market='{mk}'"
    else:
        mk = {"moneyline": "h2h", "spread": "spreads", "total": "totals"}[bet["kind"]]
        w = f"player_id IS NULL AND market='{mk}'"
    rows = conn.execute(q.format(w=w), (bet["game_id"], OA.SOURCE, bet["game_id"], OA.SOURCE, tip)).fetchall()
    by_book = defaultdict(dict)
    for book, side, line, price in rows:
        if book not in OA.LEGAL_BOOKS:
            continue
        if bet["kind"] == "spread":
            key = line if side == "home" else -line          # home number
            want = bet["line"] if bet["side"] == "home" else -bet["line"]
        elif bet["kind"] == "moneyline":
            key, want = 0.0, 0.0
        else:
            key, want = line, bet["line"]
        if abs(key - want) < 1e-9:
            by_book[book][side] = price
    other = {"over": "under", "under": "over", "home": "away", "away": "home"}[bet["side"]]
    fairs = []
    for b, sides in by_book.items():
        if bet["side"] in sides and other in sides:
            f, _ = PM.devig(sides[bet["side"]], sides[other])
            if f is not None:
                fairs.append(f)
    return float(np.median(fairs)) if fairs else None


def grade(conn, season: str) -> int:
    open_bets = [dict(r) for r in conn.execute(
        """SELECT b.*, g.status AS gstatus, g.home_score, g.away_score, g.tip_utc FROM bets b
           JOIN games g ON g.game_id = b.game_id WHERE b.status = 'open'""")]
    if not open_bets:
        return 0
    finals = [b for b in open_bets if b["gstatus"] == "final"]
    voided = [b for b in open_bets if b["gstatus"] in ("postponed", "cancelled")]
    logs = None
    if any(b["kind"] == "prop" for b in finals):
        from .build import player_logs
        logs = player_logs(conn, season)
        for s, parts in PM.STATS.items():
            if s not in logs or len(parts) > 1:
                logs[s] = sum(logs[p] for p in parts)
        logs = logs.set_index(["game_id", "player_id"])
    now = utc_now_iso()
    n = 0
    for b in voided:
        conn.execute("UPDATE bets SET status='void', pnl=0, graded_at=? WHERE bet_id=?", (now, b["bet_id"]))
        n += 1
    for b in finals:
        if b["kind"] == "prop":
            key = (b["game_id"], b["player_id"])
            if key not in logs.index or float(logs.loc[key, "min"]) <= 0:
                conn.execute("UPDATE bets SET status='void', pnl=0, graded_at=? WHERE bet_id=?", (now, b["bet_id"]))
                n += 1
                continue
            val = float(logs.loc[key, b["stat"]])
            diff = val - b["line"] if b["side"] == "over" else b["line"] - val
        else:
            if b["home_score"] is None or b["away_score"] is None:
                continue
            margin = b["home_score"] - b["away_score"]
            total = b["home_score"] + b["away_score"]
            if b["kind"] == "moneyline":
                val = margin
                diff = margin if b["side"] == "home" else -margin
            elif b["kind"] == "spread":
                val = margin
                diff = (margin + b["line"]) if b["side"] == "home" else (-margin + b["line"])
            else:
                val = total
                diff = total - b["line"] if b["side"] == "over" else b["line"] - total
        status = "won" if diff > 0 else "lost" if diff < 0 else "push"
        pnl = b["units"] * (PM.decimal(b["price"]) - 1) if status == "won" else -b["units"] if status == "lost" else 0.0
        close = _closing_fair(conn, b, b["tip_utc"] or now)
        clv = close - b["p_market"] if close is not None else None
        conn.execute("UPDATE bets SET status=?, result=?, pnl=?, close_fair=?, clv=?, graded_at=? WHERE bet_id=?",
                     (status, val, round(pnl, 4), close, clv, now, b["bet_id"]))
        n += 1
    conn.commit()
    return n


def summary(df: pd.DataFrame) -> dict:
    done = df[df["status"].isin(["won", "lost", "push"])]
    if done.empty:
        return {"bets": int(len(df)), "graded": 0}
    risked = float(done["units"].sum())
    pnl = float(done["pnl"].sum())
    res = (done["pnl"] / done["units"]).to_numpy()
    clv = done["clv"].dropna()
    return {"bets": int(len(df)), "graded": int(len(done)), "won": int((done["status"] == "won").sum()),
            "lost": int((done["status"] == "lost").sum()), "push": int((done["status"] == "push").sum()),
            "units_risked": round(risked, 2), "units": round(pnl, 2), "roi": round(pnl / risked, 4) if risked else None,
            "roi_se": round(float(res.std(ddof=1) / math.sqrt(len(res))), 4) if len(res) > 1 else None,
            "avg_clv": round(float(clv.mean()), 4) if len(clv) else None,
            "beat_close": round(float((clv > 0).mean()), 3) if len(clv) else None, "clv_n": int(len(clv))}


def export(conn, out_dir, season: str, games: list, sims: dict, prop_rows: list, pulled: dict, write) -> dict:
    conn.executescript(TABLE)
    from .props import _latest_odds
    gmap = {g["game_id"]: g for g in games}
    table = game_market_table(_latest_odds(conn, list(gmap), props=False))
    cands = prop_candidates(prop_rows or []) + game_candidates(table, sims)
    picks = select(cands)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # lock on any run while the stored prices are fresh, not only right after a pull
    fresh = {r[0]: r[1] for r in conn.execute(
        "SELECT game_id, MAX(fetched_at) FROM odds_snapshots WHERE source = ? GROUP BY game_id", (OA.SOURCE,))}
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=FRESH_MINUTES)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lockable = [c for c in picks if (fresh.get(c["game_id"]) or "") >= cutoff]
    added = lock(conn, lockable, gmap, now_iso)
    for c in picks:
        c["odds_at"] = fresh.get(c["game_id"])
    graded = grade(conn, season)
    ledger = pd.read_sql_query(
        """SELECT b.*, g.game_date, g.home_team_id, g.away_team_id, g.tip_utc, g.home_score, g.away_score
           FROM bets b JOIN games g ON g.game_id = b.game_id ORDER BY b.created_at DESC, b.bet_id DESC""", conn)
    names = dict(conn.execute("SELECT player_id, full_name FROM players").fetchall())
    ledger["player"] = ledger["player_id"].map(lambda p: names.get(int(p)) if pd.notna(p) else None)
    by_kind = {k: summary(ledger[ledger["kind"] == k]) for k in KIND_LABEL if (ledger["kind"] == k).any()} \
        if len(ledger) else {}
    cols = ["bet_id", "created_at", "game_id", "game_date", "home_team_id", "away_team_id", "tip_utc", "kind",
            "player_id", "player", "stat", "side", "line", "price", "book", "p_model", "p_market", "p_used", "ev",
            "units", "status", "result", "pnl", "close_fair", "clv", "home_score", "away_score"]
    led = ledger[cols].astype(object).where(pd.notna(ledger[cols]), None) if len(ledger) else ledger
    game_lines = [{k: v for k, v in t.items() if k != "sides"} | {"sides": list(t["sides"])} for t in table]
    payload = {
        "generated_at": utc_now_iso(), "season": season,
        "settings": {"blend": BLEND, "min_ev": MIN_EV, "min_gap": MIN_GAP, "max_gap": MAX_GAP, "min_books": MIN_BOOKS, "kelly": KELLY,
                     "unit_pct": UNIT_PCT, "max_units": MAX_UNITS, "max_game_units": MAX_GAME_UNITS,
                     "max_day_picks": MAX_DAY_PICKS, "max_day_units": MAX_DAY_UNITS,
                     "price_range": [MIN_PRICE, MAX_PRICE], "legal_books": sorted(OA.LEGAL_BOOKS)},
        "current": picks, "candidates": len(cands), "added_this_run": added, "graded_this_run": graded,
        "odds_as_of": max((v for k, v in fresh.items() if k in gmap), default=None),
        "next_pull": pulled.get("reason") if pulled.get("status") == "skipped" else None,
        "game_lines": game_lines,
        "ledger": {"cols": cols, "rows": led.values.tolist() if len(led) else []},
        "summary": summary(ledger) if len(ledger) else {"bets": 0, "graded": 0}, "by_kind": by_kind,
    }
    write(out_dir / "bets.json", payload)
    return {"picks": len(picks), "added": added, "graded": graded, "ledger": int(len(ledger))}
