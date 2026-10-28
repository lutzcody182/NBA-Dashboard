"""Player props: sportsbook lines against the model, and how the comparisons have done.

Every run:
  * pulls new odds when the budget allows (providers.odds_api)
  * for each upcoming game, takes the latest stored odds and, per player and
    stat, the line most books are offering; removes each book's margin
    from its over/under pair and takes the median as the market's fair
    chance of going over
  * computes the model's chance from the simulation's per-minute numbers,
    the minutes model's range for the player if he plays, and the fitted
    spread of each stat (models.props)
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
from ..util import utc_now_iso

log = logging.getLogger(__name__)
STAT_LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "Threes", "pra": "Pts+Reb+Ast",
              "pr": "Pts+Reb", "pa": "Pts+Ast", "ra": "Reb+Ast"}

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
    key = f"props-v2:{season_m}"
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


def _latest_odds(conn, game_ids):
    if not game_ids:
        return pd.DataFrame()
    q = ",".join("?" * len(game_ids))
    df = pd.read_sql_query(
        f"""SELECT o.* FROM odds_snapshots o
            JOIN (SELECT game_id, MAX(fetched_at) AS f FROM odds_snapshots
                  WHERE source = ? AND game_id IN ({q}) AND player_id IS NOT NULL GROUP BY game_id) m
              ON m.game_id = o.game_id AND m.f = o.fetched_at
            WHERE o.source = ? AND o.player_id IS NOT NULL""",
        conn, params=[OA.SOURCE, *game_ids, OA.SOURCE])
    return df


def market_table(odds: pd.DataFrame) -> list:
    """One entry per player and stat at the line most books offer."""
    if odds.empty:
        return []
    odds = odds.copy()
    odds["stat"] = odds["market"].map(OA.MARKETS)
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
        bo = max(books, key=lambda b: PM.decimal(b["over"]))
        bu = max(books, key=lambda b: PM.decimal(b["under"]))
        out.append({"game_id": gid, "player_id": int(pid), "stat": stat, "line": float(line),
                    "fair_over": round(float(np.median([b["fair_over"] for b in books])), 4),
                    "books": sorted(books, key=lambda b: b["book"]),
                    "best_over": {"price": bo["over"], "book": bo["book"]},
                    "best_under": {"price": bu["under"], "book": bu["book"]},
                    "other_lines": sorted(float(l) for l in counts if l != line)})
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
                base = {s: sp.get(s, 0.0) for s in ("pts", "reb", "ast", "fg3m")}
                rates = {s: sum(base[x] for x in parts) / sp["min"] for s, parts in PM.STATS.items()}
                # p10/p90 are the minutes model's range around minutes-if-he-plays
                p10 = p.get("p10") if p.get("p10") is not None else m_if - 6
                p90 = p.get("p90") if p.get("p90") is not None else m_if + 6
                out[key] = {"min": m_if, "p10": min(p10, m_if), "p90": max(p90, m_if), "rates": rates,
                            "status": p.get("status"), "name": p.get("name")}
    return out


def compare(rows: list, inputs: dict, disp: dict) -> list:
    out = []
    for r in rows:
        x = inputs.get((r["game_id"], r["player_id"]))
        if x is None or x["status"] == "out":
            r.update({"model": None})
            out.append(r)
            continue
        rate, shape = x["rates"][r["stat"]], disp[r["stat"]]
        po, pu, push = PM.over_under(r["line"], rate, shape, x["min"], x["p10"], x["p90"])
        dec_o, dec_u = PM.decimal(r["best_over"]["price"]), PM.decimal(r["best_under"]["price"])
        fair_o = r["fair_over"]
        # compare like with like: the model's over chance among results that are not a push
        po_np = po / (po + pu) if po + pu > 0 else None
        r["model"] = {"mean": round(rate * x["min"], 2), "min": round(x["min"], 1), "p_over": round(po, 4),
                      "p_under": round(pu, 4), "p_push": round(push, 4),
                      "edge_over": round(po_np - fair_o, 4) if po_np is not None else None,
                      "ev_over": round(po * (dec_o - 1) - pu, 4), "ev_under": round(pu * (dec_u - 1) - po, 4),
                      "status": x["status"]}
        out.append(r)
    return out


def save_evals(conn, fetched_at: str, rows: list):
    conn.executemany(
        """INSERT OR REPLACE INTO prop_evals (fetched_at, game_id, player_id, stat, line, model_mean, p_over, p_under,
             fair_over, books, best_over, best_over_book, best_under, best_under_book)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(fetched_at, r["game_id"], r["player_id"], r["stat"], r["line"], r["model"]["mean"], r["model"]["p_over"],
          r["model"]["p_under"], r["fair_over"], len(r["books"]), r["best_over"]["price"], r["best_over"]["book"],
          r["best_under"]["price"], r["best_under"]["book"]) for r in rows if r.get("model")])
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
    long = logs.melt(id_vars=["game_id", "player_id", "min"], value_vars=list(PM.STATS), var_name="stat",
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
            if r.edge >= thr:
                res.append(PM.decimal(r.best_over) - 1 if r.over else -1.0)
            elif r.edge <= -thr:
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
            "overall": summary(m), "by_stat": {s: summary(m[m["stat"] == s]) for s in PM.STATS if (m["stat"] == s).any()},
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
    try:
        pulled = OA.pull(conn, games, _rosters(proj, conn), now=now, et_day=today_et().isoformat())
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
    upcoming = [g["game_id"] for g in games]
    rows = market_table(_latest_odds(conn, upcoming))
    rows = compare(rows, model_inputs(proj, sims), cal["shapes"])
    if pulled["status"] == "stored":
        save_evals(conn, pulled["fetched_at"], rows)
    fetched = {r[0]: r[1] for r in conn.execute(
        f"SELECT game_id, MAX(fetched_at) FROM odds_snapshots WHERE source=? AND game_id IN "
        f"({','.join('?' * len(upcoming)) or 'NULL'}) GROUP BY game_id", [OA.SOURCE, *upcoming])} if upcoming else {}
    state = OA._state(conn)
    payload = {
        "generated_at": utc_now_iso(), "season": season, "stat_labels": STAT_LABEL,
        "pull": {k: v for k, v in pulled.items() if k != "errors"} | {"errors": pulled.get("errors", [])},
        "credits_remaining": state.get("remaining"), "last_pulls": state.get("pulls", [])[-6:],
        "odds_as_of": fetched, "props": rows,
        "model": {"season": cal["season"], "shapes": cal["shapes"],
                  "check": cal["check"]},
        "track_record": track_record(conn, season),
        "budget": {"max_pulls_per_day": OA.MAX_PULLS_PER_DAY, "hours_between": OA.MIN_HOURS_BETWEEN,
                   "markets": len(OA.MARKETS), "reserve": OA.RESERVE_CREDITS},
    }
    write(out_dir / "props.json", payload)
    return {"status": pulled["status"], "reason": pulled.get("reason"), "props": len(rows)}
