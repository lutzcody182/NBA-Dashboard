"""Player props: sportsbook lines against the model, and how the comparisons have done.

Every run:
  * pulls new odds when the budget allows (providers.odds_api)
  * for each upcoming game, takes the latest stored odds and, per player and
    stat, the line most books are offering; removes each book's margin
    from its over/under pair and takes the median as the market's fair
    chance of going over
  * computes the model's chance from the simulation's per-minute numbers,
    the minutes model's range for the player if he plays, and the fitted
    spread of each stat (models.props), after anchoring both to the
    player's own history (anchor_inputs): minutes are pulled halfway
    toward his last 5 games and kept within MIN_CAP of them, and each
    per-minute rate takes RATE_LONG of his rate over this season and last.
    Graded props from the first ~4,000 lines of 2026-27 showed the
    unanchored numbers drifting from the line on noise: where the model
    liked an over most, it had players' minutes 2.6 too high and scoring
    rates about 20% too high
  * when a pull happened, stores those comparisons in prop_evals, and grades
    every stored comparison whose game is final: the last one before tip-off
    for each player and stat, against the box score. Players who did not
    play are left out (books void those bets).
Writes data/props.json.
"""
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..models import props as PM
from ..models.ratings import model_season
from ..providers import odds_api as OA
from ..providers import nba_lineups as NL
from ..util import utc_now_iso

log = logging.getLogger(__name__)
# anchoring the props inputs to each player's own record (see anchor_inputs)
MIN_PULL = 0.5        # keep this share of the minutes model's difference from his last-5 average
MIN_CAP = 4.0         # and never more than this many minutes away from it
RATE_LONG = 0.15      # share of each per-minute rate taken from his rate over this season and last
ANCHOR_MIN_GAMES = 5
# confirmed starting lineups (providers.nba_lineups): a player confirmed in the starting five who has mostly
# come off the bench lately (started under half his last 5 games) gets LINEUP_UP more minutes; one confirmed
# out of it after mostly starting gets LINEUP_DOWN fewer. Fitted on 1,446 player-games of 2026-27 (players
# moved into the lineup played 5.3 more minutes than projected, players moved out 8.0 fewer): minutes error
# fell from 4.72 to 4.51 overall and from 7.4 to 5.5 for those players. Only confirmed lineups count.
LINEUP_UP = 5.0
LINEUP_DOWN = 6.0
STAT_LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "Threes", "pra": "Pts+Reb+Ast",
              "pr": "Pts+Reb", "pa": "Pts+Ast", "ra": "Reb+Ast", "stl": "Steals", "blk": "Blocks",
              "stocks": "Steals+Blocks", "tov": "Turnovers", "ftm": "Free throws made", "fgm": "Field goals made",
              "dd": "Double-double"}
STAT_LABEL.update({f"alt_{k}": f"{v} (alt line)" for k, v in list(STAT_LABEL.items()) if k in
                   ("pts", "reb", "ast", "fg3m", "pra", "pr", "pa", "ra", "stl", "blk", "tov")})
# a one-sided alternate price (most alt lines are offered as an over only) has its margin taken out at this rate
ALT_HOLD = 0.08

TABLE = """
CREATE TABLE IF NOT EXISTS prop_evals (
    fetched_at    TEXT NOT NULL,
    game_id       TEXT NOT NULL,
    player_id     INTEGER NOT NULL,
    stat          TEXT NOT NULL,
    line          REAL NOT NULL,
    model_mean    REAL,
    p_over        REAL,
    p_under       REAL,
    fair_over     REAL,
    books         INTEGER,
    best_over     INTEGER, best_over_book TEXT,
    best_under    INTEGER, best_under_book TEXT,
    PRIMARY KEY (fetched_at, game_id, player_id, stat)
);
CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_snapshots(game_id, fetched_at);
"""


def calibration(conn, season_m: str) -> dict:
    key = f"props-v3:{season_m}"   # v3 (Nov 16, 2026): steals, blocks, turnovers, made FT and FG
    row = conn.execute("SELECT payload FROM model_calibration WHERE key = ?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    from ..models.game_model import prev_season
    from .build import player_logs
    logs = player_logs(conn, season_m)
    prev = player_logs(conn, prev_season(season_m))
    prev2 = player_logs(conn, prev_season(prev_season(season_m)))
    if prev is None or prev.empty:
        nb = PM.fit_dispersion(logs)
        shapes = {s: {"shape": "nb", "k": v["k"], "n": v["n"]} for s, v in nb.items()}
        check = PM.calibration_check(logs, shapes)
        check["out_of_sample"] = False
    else:
        # shape picked on held-out games; live settings fitted on the latest full season
        shapes = PM.fit_shapes(prev, logs)
        # the check uses settings fitted only on earlier seasons, so it is out of sample
        older = PM.fit_shapes(prev2, prev) if prev2 is not None and not prev2.empty else shapes
        check = PM.calibration_check(logs, older)
        check["out_of_sample"] = older is not shapes
    cal = {"season": season_m, "shapes": shapes, "check": check}
    conn.execute("INSERT OR REPLACE INTO model_calibration (key, payload, created_at) VALUES (?, ?, ?)",
                 (key, json.dumps(cal), utc_now_iso()))
    conn.commit()
    return cal


def _rosters(proj: dict, conn) -> dict:
    """game_id -> {normalized name: player_id} from the projected rosters (plus alternate spellings seen before)."""
    out = {}
    extra = {}
    for r in conn.execute("SELECT provider_key, internal_id FROM id_map WHERE entity_type='player' AND provider=?",
                          (OA.SOURCE,)):
        extra[OA.norm_name(r[0])] = int(r[1])
    for g in (proj or {}).get("games", []):
        m = {}
        for t in g.get("teams", {}).values():
            for p in t.get("players", []):
                m[OA.norm_name(p["name"])] = p["player_id"]
        ids = set(m.values())
        m.update({k: v for k, v in extra.items() if v in ids})
        out[g["game_id"]] = m
    return out


def _latest_odds(conn, game_ids, props: bool = True, legal_only: bool = True):
    """Latest stored odds per game and market, player props or game lines, from state-licensed US books by default.

    Per market, because the extra markets are pulled less often than the main ones."""
    if not game_ids:
        return pd.DataFrame()
    q = ",".join("?" * len(game_ids))
    which = "player_id IS NOT NULL" if props else "player_id IS NULL"
    df = pd.read_sql_query(
        f"""SELECT o.* FROM odds_snapshots o
            JOIN (SELECT game_id, market, MAX(fetched_at) AS f FROM odds_snapshots
                  WHERE source = ? AND game_id IN ({q}) AND {which} GROUP BY game_id, market) m
              ON m.game_id = o.game_id AND m.market = o.market AND m.f = o.fetched_at
            WHERE o.source = ? AND o.{which}""",
        conn, params=[OA.SOURCE, *game_ids, OA.SOURCE])
    if legal_only and not df.empty:
        df = df[df["book"].isin(OA.LEGAL_BOOKS)]
    return df


def market_table(odds: pd.DataFrame) -> list:
    """One entry per player and stat at the line most books offer."""
    if odds.empty:
        return []
    odds = odds.copy()
    odds["stat"] = odds["market"].map(OA.PROP_MARKETS)
    odds = odds[odds["stat"].notna() & ~odds["stat"].astype(str).str.startswith("alt_")]
    out = []
    for (gid, pid, stat), g in odds.groupby(["game_id", "player_id", "stat"]):
        # books with both sides at the same line
        pairs = g.pivot_table(index=["book", "line"], columns="side", values="price_american", aggfunc="first")
        if "over" not in pairs or "under" not in pairs:
            continue
        pairs = pairs.dropna(subset=["over", "under"]).reset_index()
        if pairs.empty:
            continue
        counts = Counter(pairs["line"])
        top = max(counts.values())
        # most-offered line; ties go to the one closest to the middle of the offered lines
        cands = [l for l, c in counts.items() if c == top]
        mid = float(np.median(pairs["line"]))
        line = min(cands, key=lambda l: (abs(l - mid), l))
        at = pairs[pairs["line"] == line]
        books = []
        for r in at.itertuples():
            fair, hold = PM.devig(r.over, r.under)
            if fair is not None:
                books.append({"book": r.book, "over": int(r.over), "under": int(r.under),
                              "fair_over": round(fair, 4), "hold": round(hold, 4)})
        if not books:
            continue
        mine = [b for b in books if b["book"] in OA.BET_BOOKS]     # best prices only from the books you bet at
        bo = max(mine, key=lambda b: PM.decimal(b["over"])) if mine else {"over": None, "book": None}
        bu = max(mine, key=lambda b: PM.decimal(b["under"])) if mine else {"under": None, "book": None}
        out.append({"game_id": gid, "player_id": int(pid), "stat": stat, "line": float(line),
                    "fetched_at": str(g["fetched_at"].max()) if "fetched_at" in g else None,
                    "fair_over": round(float(np.median([b["fair_over"] for b in books])), 4),
                    "books": sorted(books, key=lambda b: b["book"]),
                    "best_over": {"price": bo["over"], "book": bo["book"]},
                    "best_under": {"price": bu["under"], "book": bu["book"]},
                    "other_lines": sorted(float(l) for l in counts if l != line)})
    return out


def alt_fair(sides: dict):
    """Fair over chance at one book and line: both sides with the margin removed, or an over alone at ALT_HOLD."""
    if sides.get("over") is not None and sides.get("under") is not None:
        return PM.devig(sides["over"], sides["under"])[0]
    if sides.get("over") is not None:
        return min(0.99, PM.implied(sides["over"]) / (1 + ALT_HOLD))
    if sides.get("under") is not None:
        return max(0.01, 1 - PM.implied(sides["under"]) / (1 + ALT_HOLD))
    return None


def alt_table(odds: pd.DataFrame, main_rows: list) -> list:
    """Alternate player lines: every line other than the main one, with a fair chance and the best prices."""
    if odds.empty:
        return []
    odds = odds.copy()
    odds["stat"] = odds["market"].map(OA.PROP_MARKETS)
    odds = odds[odds["stat"].astype(str).str.startswith("alt_")]
    main = {(r["game_id"], r["player_id"], r["stat"]): r["line"] for r in main_rows}
    out = []
    for (gid, pid, stat, line), g in odds.groupby(["game_id", "player_id", "stat", "line"]):
        base = stat[4:]
        if main.get((gid, int(pid), base)) == float(line):
            continue
        books = []
        for book, b in g.groupby("book"):
            sides = {r.side: int(r.price_american) for r in b.itertuples()}
            fair = alt_fair(sides)
            if fair is not None:
                books.append({"book": book, **sides, "fair_over": round(float(fair), 4)})
        if not books:
            continue
        mine = [b for b in books if b["book"] in OA.BET_BOOKS]
        best = {}
        for side in ("over", "under"):
            have = [b for b in mine if b.get(side) is not None]
            bb = max(have, key=lambda b: PM.decimal(b[side])) if have else None
            best[side] = {"price": bb[side] if bb else None, "book": bb["book"] if bb else None}
        out.append({"game_id": gid, "player_id": int(pid), "stat": stat, "base_stat": base, "line": float(line),
                    "fetched_at": str(g["fetched_at"].max()) if "fetched_at" in g else None,
                    "fair_over": round(float(np.median([b["fair_over"] for b in books])), 4),
                    "books": sorted(books, key=lambda b: b["book"]),
                    "best_over": best["over"], "best_under": best["under"], "alt": True})
    return out


def fingerprints(proj: dict, lineups: dict) -> dict:
    """game_id -> a short text of who is out, doubtful or questionable and the confirmed starters: when it
    changes, there is news the books may not have priced yet."""
    out = {}
    for g in (proj or {}).get("games", []):
        parts = []
        for tid, t in sorted(g.get("teams", {}).items()):
            st = sorted(f"{p['player_id']}:{p.get('status')}" for p in t.get("players", [])
                        if p.get("status") in ("out", "doubtful", "questionable"))
            lu = (lineups or {}).get((g["game_id"], int(tid)))
            starters = sorted(lu["starters"]) if lu and lu.get("status") == "confirmed" else []
            parts.append(f"{tid}|{','.join(st)}|{','.join(map(str, starters))}")
        out[g["game_id"]] = ";".join(parts)
    return out


def model_inputs(proj: dict, sims: dict) -> dict:
    """(game_id, player_id) -> minutes range if he plays and per-minute rates for every stat."""
    pinfo = {}
    for g in (proj or {}).get("games", []):
        for t in g.get("teams", {}).values():
            for p in t.get("players", []):
                pinfo[(g["game_id"], p["player_id"])] = p
    out = {}
    for g in (sims or {}).get("games", []):
        for t in g.get("teams", {}).values():
            for sp in t.get("players", []):
                key = (g["game_id"], sp["player_id"])
                p = pinfo.get(key)
                if not p or not sp.get("min"):
                    continue
                m_if = p.get("min_if_plays") or p.get("min")
                if not m_if:
                    continue
                base = {s: sp[s] for s in PM.BASE_STATS if sp.get(s) is not None}
                for s in ("pts", "reb", "ast", "fg3m"):
                    base.setdefault(s, 0.0)
                rates = {s: sum(base[x] for x in parts) / sp["min"] for s, parts in PM.STATS.items()
                         if all(x in base for x in parts)}
                # p10/p90 are the minutes model's range around minutes-if-he-plays
                p10 = p.get("p10") if p.get("p10") is not None else m_if - 6
                p90 = p.get("p90") if p.get("p90") is not None else m_if + 6
                ap = p.get("appear") or {}
                out[key] = {"min": m_if, "p10": min(p10, m_if), "p90": max(p90, m_if), "rates": rates,
                            "status": p.get("status"), "name": p.get("name"),
                            "games": int((ap.get("played") or 0) + (ap.get("prev_played") or 0))}
    return out


def player_history(conn, season: str) -> dict:
    """player_id -> {"m5": last-5 minutes, "games": n, "rate": {stat: per-minute rate}} from this season and last."""
    from ..models.game_model import prev_season
    df = pd.read_sql_query(
        """SELECT s.player_id, s.team_id, s.started, g.game_date, s.seconds / 60.0 AS m, s.pts, s.oreb + s.dreb AS reb, s.ast, s.fg3m,
                  s.stl, s.blk, s.tov, s.ftm, s.fgm
           FROM player_game_stats s JOIN games g ON g.game_id = s.game_id
           WHERE g.season_id IN (?, ?) AND g.season_type = 'regular' AND g.status = 'final' AND s.seconds > 0""",
        conn, params=[season, prev_season(season)])
    if df.empty:
        return {}
    base = PM.BASE_STATS
    df[list(base)] = df[list(base)].fillna(0)
    lg = {k: df[k].sum() / df["m"].sum() for k in base}
    out = {}
    for pid, h in df.sort_values("game_date").groupby("player_id"):
        mins = h["m"].sum()
        out[int(pid)] = {"m5": float(h["m"].tail(5).mean()), "games": int(len(h)),
                         "st5": float(h["started"].tail(5).fillna(0).mean()), "team": int(h["team_id"].iloc[-1]),
                         "rate": {k: float((h[k].sum() + PM.PRIOR_MIN * lg[k]) / (mins + PM.PRIOR_MIN)) for k in base}}
    return out


def anchor_inputs(inputs: dict, hist: dict, lineups: dict = None) -> dict:
    """Pull each player's minutes toward his last 5 games and his per-minute rates toward his longer record,
    then adjust minutes for a confirmed change of role in the starting lineup."""
    out = {}
    for key, x in inputs.items():
        h = hist.get(key[1])
        if not h or h["games"] < ANCHOR_MIN_GAMES:
            out[key] = x
            continue
        m = h["m5"] + float(np.clip(MIN_PULL * (x["min"] - h["m5"]), -MIN_CAP, MIN_CAP))
        role = None
        lu = (lineups or {}).get((key[0], h.get("team")))
        if lu and lu["status"] == "confirmed" and "st5" in h:
            starts = key[1] in lu["starters"]
            if starts and h["st5"] < 0.5:
                m, role = m + LINEUP_UP, "moved into the starting lineup"
            elif not starts and h["st5"] >= 0.5:
                m, role = m - LINEUP_DOWN, "moved out of the starting lineup"
            else:
                role = "confirmed starter" if starts else "confirmed bench"
        m = float(np.clip(m, 1.0, 48.0))
        shift = m - x["min"]
        # stats the simulation doesn't hand out (steals, blocks, made FT/FG) come from his own record
        hr = h["rate"]
        base = {k: ((1 - RATE_LONG) * x["rates"][k] + RATE_LONG * hr[k]) if k in x["rates"] and k in hr
                else x["rates"][k] if k in x["rates"] else hr[k]
                for k in PM.BASE_STATS if k in x["rates"] or k in hr}
        rates = {s: sum(base[p] for p in parts) for s, parts in PM.STATS.items() if all(p in base for p in parts)}
        out[key] = {**x, "min": m, "p10": max(0.0, min(x["p10"] + shift, m)), "p90": max(x["p90"] + shift, m),
                    "rates": rates, "model_min": x["min"], **({"lineup": role} if role else {})}
    return out


def compare(rows: list, inputs: dict, disp: dict) -> list:
    out = []
    for r in rows:
        x = inputs.get((r["game_id"], r["player_id"]))
        if x is None or x["status"] == "out":
            r.update({"model": None})
            out.append(r)
            continue
        stat = r.get("base_stat", r["stat"])
        if stat == "dd":
            if not all(s in x["rates"] for s in PM.DD_PARTS):
                r.update({"model": None})
                out.append(r)
                continue
            rate = None
            po = PM.double_double(x["rates"], disp, x["min"], x["p10"], x["p90"])
            pu, push = 1.0 - po, 0.0
        else:
            if stat not in x["rates"] or stat not in disp:
                r.update({"model": None})
                out.append(r)
                continue
            rate, shape = x["rates"][stat], disp[stat]
            po, pu, push = PM.over_under(r["line"], rate, shape, x["min"], x["p10"], x["p90"])
        bo_p, bu_p = r["best_over"]["price"], r["best_under"]["price"]
        dec_o, dec_u = (PM.decimal(bo_p) if bo_p is not None else None), (PM.decimal(bu_p) if bu_p is not None else None)
        fair_o = r["fair_over"]
        # compare like with like: the model's over chance among results that are not a push
        po_np = po / (po + pu) if po + pu > 0 else None
        r["model"] = {"mean": round(rate * x["min"], 2) if rate is not None else round(po, 3),
                      "min": round(x["min"], 1), "p_over": round(po, 4),
                      "p_under": round(pu, 4), "p_push": round(push, 4),
                      "edge_over": round(po_np - fair_o, 4) if po_np is not None else None,
                      "ev_over": round(po * (dec_o - 1) - pu, 4) if dec_o else None,
                      "ev_under": round(pu * (dec_u - 1) - po, 4) if dec_u else None,
                      "status": x["status"], "games": x.get("games"),
                      **({"lineup": x["lineup"]} if x.get("lineup") else {})}
        out.append(r)
    return out


def save_evals(conn, fetched_at: str, rows: list):
    conn.executemany(
        """INSERT OR REPLACE INTO prop_evals (fetched_at, game_id, player_id, stat, line, model_mean, p_over, p_under,
             fair_over, books, best_over, best_over_book, best_under, best_under_book)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(fetched_at, r["game_id"], r["player_id"], r["stat"], r["line"], r["model"]["mean"], r["model"]["p_over"],
          r["model"]["p_under"], r["fair_over"], len(r["books"]), r["best_over"]["price"], r["best_over"]["book"],
          r["best_under"]["price"], r["best_under"]["book"]) for r in rows
         if r.get("model") and r.get("fetched_at", fetched_at) == fetched_at])
    conn.commit()


def track_record(conn, season: str) -> dict:
    """Grade the last pre-tip comparison for every finished game."""
    df = pd.read_sql_query(
        """SELECT e.*, g.tip_utc FROM prop_evals e JOIN games g ON g.game_id = e.game_id
           WHERE g.status = 'final' AND g.season_id = ? AND (g.tip_utc IS NULL OR e.fetched_at < g.tip_utc)""",
        conn, params=[season])
    if df.empty:
        return {"graded": 0}
    df = df.sort_values("fetched_at").groupby(["game_id", "player_id", "stat"]).tail(1)
    from .build import player_logs
    logs = player_logs(conn, season)
    logs = logs[logs["game_id"].isin(df["game_id"].unique())].copy()
    for s, parts in PM.STATS.items():
        logs[s] = sum(logs[p] for p in parts)
    long = logs.melt(id_vars=["game_id", "player_id", "min"], value_vars=[*PM.STATS, "dd"], var_name="stat",
                     value_name="actual")
    m = df.merge(long, on=["game_id", "player_id", "stat"], how="inner")
    m = m[m["min"] > 0]
    m = m[m["actual"] != m["line"]]                      # pushes are refunded
    if m.empty:
        return {"graded": 0}
    m["over"] = (m["actual"] > m["line"]).astype(float)
    m["p_model"] = m["p_over"] / (m["p_over"] + m["p_under"])
    m["edge"] = m["p_model"] - m["fair_over"]

    def summary(x):
        if x.empty:
            return None
        return {"n": int(len(x)), "over_rate": round(float(x["over"].mean()), 3),
                "brier_model": round(float(((x["p_model"] - x["over"]) ** 2).mean()), 4),
                "brier_market": round(float(((x["fair_over"] - x["over"]) ** 2).mean()), 4)}

    def picks(x, thr):
        """Bet the side the model favors by at least `thr` over the market, at the best price stored."""
        res = []
        for r in x.itertuples():
            if r.edge >= thr and pd.notna(r.best_over):
                res.append(PM.decimal(r.best_over) - 1 if r.over else -1.0)
            elif r.edge <= -thr and pd.notna(r.best_under):
                res.append(PM.decimal(r.best_under) - 1 if not r.over else -1.0)
        if not res:
            return {"threshold": thr, "n": 0}
        a = np.array(res)
        return {"threshold": thr, "n": int(len(a)), "won": int((a > 0).sum()), "units": round(float(a.sum()), 2),
                "roi": round(float(a.mean()), 4), "roi_se": round(float(a.std(ddof=1) / math.sqrt(len(a))), 4)
                if len(a) > 1 else None}

    bins = []
    edges = [0, .3, .4, .45, .5, .55, .6, .7, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = m[(m["p_model"] >= lo) & (m["p_model"] < hi)]
        if len(x) >= 10:
            bins.append({"bin": f"{round(lo * 100)}-{min(round(hi * 100), 100)}%", "n": int(len(x)),
                         "model": round(float(x["p_model"].mean()), 3), "market": round(float(x["fair_over"].mean()), 3),
                         "actual": round(float(x["over"].mean()), 3)})
    return {"graded": int(len(m)), "games": int(m["game_id"].nunique()),
            "first": str(m["fetched_at"].min())[:10], "last": str(m["fetched_at"].max())[:10],
            "overall": summary(m), "by_stat": {s: summary(m[m["stat"] == s]) for s in [*PM.STATS, "dd"] if (m["stat"] == s).any()},
            "picks": [picks(m, t) for t in (0.03, 0.05, 0.08)], "calibration": bins}


def export(conn, out_dir, season: str, games: list, proj: dict, sims: dict, write) -> dict:
    conn.executescript(TABLE)
    m = model_season(conn, season)
    if m is None:
        write(out_dir / "props.json", {"generated_at": utc_now_iso(), "props": [], "note": "no finished season to fit on"})
        return {"status": "no model season"}
    cal = calibration(conn, m)
    from ..ingest.update import today_et
    now = datetime.now(timezone.utc)
    try:                                   # starting lineups first: free, and they change the minutes below
        lineup_status = NL.update(conn, today_et())
    except Exception as exc:
        log.warning("Lineups not read: %s", exc)
        lineup_status = {"status": "error", "error": str(exc)[:200]}
    upcoming = [g["game_id"] for g in games]
    try:
        lineups = NL.latest(conn, set(upcoming))
    except Exception:
        log.exception("Reading stored lineups failed")
        lineups = {}
    try:
        pulled = OA.pull(conn, games, _rosters(proj, conn), now=now, et_day=today_et().isoformat(),
                         fingerprints=fingerprints(proj, lineups))
    except Exception as exc:
        log.exception("Odds pull failed")
        pulled = {"status": "error", "errors": [str(exc)]}
    from .. import db as dbmod
    if pulled["status"] == "stored":
        dbmod.mark_attempt(conn, "odds", True, f"{pulled['rows']} prices for {pulled['games']} games; "
                           f"{int(pulled['remaining']) if pulled.get('remaining') is not None else '?'} credits left")
    elif pulled.get("key_ok"):
        dbmod.mark_attempt(conn, "odds", True, f"key works; no pull yet ({pulled.get('reason')}); "
                           f"{int(pulled['remaining']) if pulled.get('remaining') is not None else '?'} credits left")
    elif pulled["status"] == "error":
        dbmod.mark_attempt(conn, "odds", False, "; ".join(pulled.get("errors", []))[:400])
    odds = _latest_odds(conn, upcoming)
    rows = market_table(odds)
    alt_rows = alt_table(odds, rows)
    try:
        hist = player_history(conn, season)
    except Exception:
        log.exception("Player history for props failed; using the unanchored inputs")
        hist = {}
    inputs = anchor_inputs(model_inputs(proj, sims), hist, lineups)
    rows = compare(rows, inputs, cal["shapes"])
    alt_rows = compare(alt_rows, inputs, cal["shapes"])
    if pulled["status"] == "stored":
        save_evals(conn, pulled["fetched_at"], rows)
    fetched = {r[0]: r[1] for r in conn.execute(
        f"SELECT game_id, MAX(fetched_at) FROM odds_snapshots WHERE source=? AND game_id IN "
        f"({','.join('?' * len(upcoming)) or 'NULL'}) GROUP BY game_id", [OA.SOURCE, *upcoming])} if upcoming else {}
    try:
        from . import factors as FX
        factors = FX.build(conn, season, rows, proj)
    except Exception:
        log.exception("Matchup factors failed")
        factors = {}
    state = OA._state(conn)
    payload = {
        "generated_at": utc_now_iso(), "season": season, "stat_labels": STAT_LABEL,
        "pull": {k: v for k, v in pulled.items() if k != "errors"} | {"errors": pulled.get("errors", [])},
        "credits_remaining": state.get("remaining"), "last_pulls": state.get("pulls", [])[-6:],
        "odds_as_of": fetched, "props": rows, "factors": factors,
        "model": {"season": cal["season"], "shapes": cal["shapes"],
                  "check": cal["check"]},
        "track_record": track_record(conn, season),
        "lineups": lineup_status,
        "budget": {"near_hours": OA.NEAR_HOURS, "repull_minutes": OA.REPULL_MINUTES,
                   "daily_cap": OA.DAILY_CREDIT_CAP, "monthly_cap": OA.MONTHLY_CREDIT_CAP,
                   "month_used": OA.spent(OA._state(conn), month=today_et().isoformat()[:7]),
                   "markets": len(OA.MARKETS), "reserve": OA.RESERVE_CREDITS},
    }
    write(out_dir / "props.json", payload)
    return {"status": pulled["status"], "reason": pulled.get("reason"), "props": len(rows), "alt_lines": len(alt_rows),
            "_rows": rows, "_alt_rows": alt_rows, "_pulled": pulled}
