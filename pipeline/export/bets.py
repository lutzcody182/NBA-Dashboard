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
  The model has no live record yet, so the chance used for a bet is a
  weighted average of the model's and the market's: BLEND model, the rest
  market. It started at 0.5; after the first ~220 graded bets showed the
  model running well above both the market and the results, it was cut to
  0.35. A bet needs:
    * expected return of at least MIN_EV at the best legal price
    * the model rating the side at least MIN_GAP above the market, and not
      more than MAX_GAP above it (bigger gaps usually mean news the model
      hasn't seen)
    * at least MIN_BOOKS legal books quoting it (they all set the market's fair chance), and a price at one
      of the books you bet at (odds_api.BET_BOOKS: FanDuel, theScore Bet / ESPN BET, Caesars, Fanatics)
    * props: the player has 5+ games of history and is not listed as
      questionable or doubtful; unders only while PROP_OVERS is off (paused Nov 6, 2026)
Stakes
  A quarter of the Kelly stake for the blended chance, where 1 unit = 1% of
  bankroll, rounded to 0.25 units, at most MAX_UNITS per bet (the uncapped stake, up to
  FULL_MAX_UNITS, is saved as units_full to rank picks) and
  MAX_GAME_UNITS per game (a game's stakes are scaled down to fit, never
  below 0.25 units, and no qualifying bet is dropped). One prop per player,
  and one side bet (moneyline or spread) per game. No daily limit: every bet
  that qualifies is recommended.
Tracking mode (since Nov 16, 2026)
  Through ~450 graded bets the picks lost about 11% and did not beat the
  closing line, so every pick is now tracked first. A pick is only
  recommended (shown as a bet, sent to the phone, counted in the record) when
  its market has earned it (gate_table): at least GATE_MIN_GRADED graded picks
  and GATE_MIN_CLV with a closing price after the pick, the closing line moving
  toward the picks on average by at least GATE_CLV_Z standard errors, and the
  picks winning at least as often as the market's own chances said. Markets are
  split by side (prop unders and overs earn separately). Tracked picks get the
  same stake a recommendation would, so their results read in units, but no
  game cap. Picks made on a news pull (odds_api: an injury or lineup change near
  tip-off) are also scored together, as "news", and earn on their own.
  Extra markets tracked from the start: steals, blocks, steals+blocks,
  turnovers, made FT and FG, double-doubles, alternate player lines (alt_*),
  team totals, first-half lines and alternate spreads/totals.
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

# A bet "beat the close" only if the market moved toward it by more than rounding noise. Many bets are
# locked from the last pull before tip-off, so their closing price is their own price (CLV exactly 0).
BEAT_EPS = 5e-5

BLEND = 0.35             # share of the model in the chance used (was 0.5 until Nov 2, 2026)
MIN_EV = 0.04
MAX_GAP = 0.10
MIN_GAP = 0.02
MIN_BOOKS = 3
KELLY = 0.25
UNIT_PCT = 1.0
MAX_UNITS = 0.5          # stake cap while the model is unproven (was 1.5 until Nov 6, 2026); the uncapped
FULL_MAX_UNITS = 1.5     # stake is still worked out (up to 1.5) and saved as units_full, to rank the strongest picks
PROP_OVERS = False       # prop overs paused Nov 6, 2026: through 383 graded bets and 7,350 graded lines they lost in every
                         # band of model-vs-market difference (overs hit 46.6% vs the market's 49.7%); unders stay
MAX_GAME_UNITS = 3.0
MAX_DAY_PICKS = None      # no daily limit on bets (was 15)
MAX_DAY_UNITS = None      # no daily limit on units (was 12)
FRESH_MINUTES = 100       # picks are locked only from prices at most this old
GATE_MIN_GRADED = 150     # a market earns recommendations only with this many graded picks
GATE_MIN_CLV = 100        # ... this many with a closing price taken after the pick
GATE_CLV_Z = 2.0          # ... closing line value above zero by this many standard errors
MIN_BOOKS_ALT = 2         # alternate lines are quoted by fewer books
# Retired Nov 26, 2026: after ~450 graded recommendations (-11%) and ~1,500 tracked picks (-9%, no closing-line
# edge, and fading the model lost too), no new picks are made or tracked. Open picks are still graded and the
# full record stays on the dashboard. Set True (and odds_api.ODDS_ENABLED) to start again.
PICKS_ENABLED = False
# "Moderate disagreement" rule (tracked since Nov 20, 2026). On ~6,500 graded main props from Nov 12-19 the
# model's moderate leans beat the market (model 60-70% over: 57% went over vs the market's 52.5%; model 30-40%:
# 43% vs 47%), while its strong leans did not (under 30%: 49% went over). So: main props where the model gives
# the over MOD_OVER, or the under MOD_UNDER (the over 30-45%), at the best legal price, with the model on the same
# side as the market's fair chance moves. Flat MOD_UNITS stake, scored together as one market ("moderate").
MOD_STATS = ("pts", "reb", "ast", "fg3m", "pra", "pr", "pa", "ra")
MOD_OVER = (0.55, 0.70)
MOD_UNDER = (0.30, 0.45)                   # the model's chance of the over
MOD_UNITS = 0.5
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
KIND_LABEL = {"prop": "Player prop", "moneyline": "Moneyline", "spread": "Spread", "total": "Total",
              "team_total": "Team total", "moneyline_h1": "1st-half moneyline", "spread_h1": "1st-half spread",
              "total_h1": "1st-half total", "alt_spread": "Alt spread", "alt_total": "Alt total"}
# Odds API market key -> our kind, for the game markets
GAME_KIND = {"h2h": "moneyline", "spreads": "spread", "totals": "total", "h2h_h1": "moneyline_h1",
             "spreads_h1": "spread_h1", "totals_h1": "total_h1", "team_totals": "team_total",
             "alternate_spreads": "alt_spread", "alternate_totals": "alt_total"}
KIND_MARKET = {v: k for k, v in GAME_KIND.items()}
TOTALS_LIKE = {"totals", "totals_h1", "alternate_totals", "team_totals"}
SPREADS_LIKE = {"spreads", "spreads_h1", "alternate_spreads"}
ALT_GAME = {"alternate_spreads": "spreads", "alternate_totals": "totals"}
# results are also broken out by bet type: each prop stat, then the game markets
MARKET_ORDER = ("pts", "reb", "ast", "fg3m", "pra", "pr", "pa", "ra", "stl", "blk", "stocks", "tov", "ftm", "fgm",
                "dd", "alt_pts", "alt_reb", "alt_ast", "alt_fg3m", "alt_pra", "alt_pr", "alt_pa", "alt_ra",
                "alt_stl", "alt_blk", "alt_tov", "moneyline", "spread", "total", "team_total", "moneyline_h1",
                "spread_h1", "total_h1", "alt_spread", "alt_total")

# What you actually got, when it differs from the recommendation (set through an override,
# pipeline/overrides.py). Results and units are graded at these; the recommended price, line and
# book are kept as they were, and closing-line value still measures the recommended pick.
TAKEN_COLUMNS = (("taken_price", "INTEGER"), ("taken_line", "REAL"), ("taken_book", "TEXT"), ("taken_at", "TEXT"),
                 ("units_full", "REAL"))
# Where each recommendation stands now (track()): 'active' while the latest prices still recommend it,
# 'expired' once they stop (with the reason and when), 'started' at tip-off. Nothing is deleted.
STATE_COLUMNS = (("rec_state", "TEXT"), ("state_at", "TEXT"), ("ended_at", "TEXT"), ("end_reason", "TEXT"),
                 ("now_price", "INTEGER"), ("now_line", "REAL"), ("now_units", "REAL"), ("now_book", "TEXT"))
# Tracking mode: 'rec' (a recommendation) or 'track' (tracked only); rows from before Nov 16, 2026 have none and
# were recommendations. gate: the market it is scored in. trigger: 'news' when picked on a news pull.
MODE_COLUMNS = (("mode", "TEXT"), ("gate", "TEXT"), ("trigger", "TEXT"))


def ensure_table(conn) -> None:
    conn.executescript(TABLE)
    have = {r[1] for r in conn.execute("PRAGMA table_info(bets)")}
    for col, decl in TAKEN_COLUMNS + STATE_COLUMNS + MODE_COLUMNS:
        if col not in have:
            conn.execute(f"ALTER TABLE bets ADD COLUMN {col} {decl}")
    conn.commit()


def market_of(kind: str, stat) -> str:
    return stat if kind == "prop" else kind


def gate_key(kind: str, stat, side: str, trigger: str = None) -> str:
    """The market a pick is scored and earns recommendations in: props and totals split by side."""
    if trigger == "moderate":
        return "moderate"
    if kind == "prop":
        return f"{stat}:{side}"
    if kind in ("total", "total_h1", "alt_total", "team_total"):
        return f"{kind}:{str(side).split('_')[-1]}"
    return kind


def gate_label(key: str) -> str:
    from .props import STAT_LABEL
    m, _, side = key.partition(":")
    name = STAT_LABEL.get(m) or KIND_LABEL.get(m) or {"news": "News-driven picks",
                                                       "moderate": "Moderate disagreement (main props)"}.get(m, m)
    if m == "dd" and side:
        side = "yes" if side == "over" else "no"
    return f"{name} {side}".strip()


def gates(ledger: pd.DataFrame) -> dict:
    """Per market (gate_key), how its graded picks have done, and whether it has earned recommendations."""
    if ledger is None or ledger.empty:
        return {}
    df = ledger.copy()
    if "gate" not in df or df["gate"].isna().any():
        keys = [gate_key(k, s, sd) for k, s, sd in zip(df["kind"], df["stat"], df["side"])]
        df["gate"] = df["gate"].where(df["gate"].notna(), pd.Series(keys, index=df.index)) if "gate" in df \
            else keys
    groups = {k: g for k, g in df.groupby("gate")}
    if "trigger" in df and (df["trigger"] == "news").any():
        groups["news"] = df[df["trigger"] == "news"]
    out = {}
    for key, g in groups.items():
        done = g[g["status"].isin(["won", "lost", "push"])]
        dec = done[done["status"] != "push"]
        clv = done["clv"].dropna()
        n_clv = int(len(clv))
        clv_mean = float(clv.mean()) if n_clv else None
        clv_se = float(clv.std(ddof=1) / math.sqrt(n_clv)) if n_clv > 1 else None
        win = float((dec["status"] == "won").mean()) if len(dec) else None
        exp = float(dec["p_market"].mean()) if len(dec) else None
        risked = float(done["units"].sum())
        pnl = float(done["pnl"].sum()) if len(done) else 0.0
        z = clv_mean / clv_se if clv_mean is not None and clv_se else None
        short = []
        if len(done) < GATE_MIN_GRADED:
            short.append(f"{GATE_MIN_GRADED - len(done)} more graded picks")
        if n_clv < GATE_MIN_CLV:
            short.append(f"{GATE_MIN_CLV - n_clv} more with a closing price")
        if z is None or z < GATE_CLV_Z:
            short.append("the closing line moving our way" if clv_mean is None or clv_mean <= 0
                         else "a clearer edge on the closing line")
        if win is None or exp is None or win < exp:
            short.append("winning as often as the market expected")
        out[key] = {"key": key, "label": gate_label(key), "picks": int(len(g)), "graded": int(len(done)),
                    "won": int((done["status"] == "won").sum()), "lost": int((done["status"] == "lost").sum()),
                    "win_rate": round(win, 4) if win is not None else None,
                    "expected": round(exp, 4) if exp is not None else None,
                    "avg_clv": round(clv_mean, 4) if clv_mean is not None else None,
                    "clv_z": round(z, 2) if z is not None else None, "clv_n": n_clv,
                    "beat_close": round(float((clv > BEAT_EPS).mean()), 3) if n_clv else None,
                    "units_risked": round(risked, 2), "units": round(pnl, 2),
                    "roi": round(pnl / risked, 4) if risked else None,
                    "status": "earned" if not short else "tracking", "needs": short}
    return out


def _earned(c: dict, gate_info: dict) -> bool:
    if (gate_info.get(c["gate"]) or {}).get("status") == "earned":
        return True
    return c.get("trigger") == "news" and (gate_info.get("news") or {}).get("status") == "earned"


# ---------------------------------------------------------------- game lines

def _entry(gid, market, g, a_side, b_side, line, team=None):
    """One game market at one line: the median fair chance over books with both sides, and the best prices."""
    pairs = g.pivot_table(index=["book", "key_line"], columns="side", values="price_american", aggfunc="first")
    if a_side not in pairs or b_side not in pairs:
        return None
    pairs = pairs.dropna(subset=[a_side, b_side]).reset_index()
    at = pairs[pairs["key_line"] == line]
    books = []
    for r in at.itertuples():
        fair, hold = PM.devig(getattr(r, a_side), getattr(r, b_side))
        if fair is not None:
            books.append({"book": r.book, a_side: int(getattr(r, a_side)), b_side: int(getattr(r, b_side)),
                          "fair": round(fair, 4), "hold": round(hold, 4)})
    if not books:
        return None
    mine = [b for b in books if b["book"] in OA.BET_BOOKS]     # best prices only from the books you bet at
    best = {s: (max(mine, key=lambda b: PM.decimal(b[s])) if mine else {s: None, "book": None}) for s in (a_side, b_side)}
    e = {"game_id": gid, "market": market, "line": float(line), "sides": (a_side, b_side),
         "fair": round(float(np.median([b["fair"] for b in books])), 4), "books": books,
         "best": {s: {"price": best[s][s], "book": best[s]["book"]} for s in (a_side, b_side)},
         "fetched_at": str(g["fetched_at"].max()) if "fetched_at" in g else None}
    if team:
        e["team"] = team
    return e


def _keyed(g: pd.DataFrame, market: str) -> pd.DataFrame:
    if market.startswith("h2h"):
        return g.assign(key_line=0.0)
    if market in SPREADS_LIKE:
        # a spread pair shares one number: home -3.5 with away +3.5; key it by the home number
        return g.assign(key_line=np.where(g["side"] == "home", g["line"], -g["line"]))
    return g.assign(key_line=g["line"])


def _main_line(g: pd.DataFrame, a_side: str, b_side: str):
    pairs = g.pivot_table(index=["book", "key_line"], columns="side", values="price_american", aggfunc="first")
    if a_side not in pairs or b_side not in pairs:
        return None
    pairs = pairs.dropna(subset=[a_side, b_side]).reset_index()
    if pairs.empty:
        return None
    counts = Counter(pairs["key_line"])
    top = max(counts.values())
    mid = float(np.median(pairs["key_line"]))
    return min((l for l, c in counts.items() if c == top), key=lambda l: (abs(l - mid), l))


def game_market_table(odds: pd.DataFrame) -> list:
    """Per game and market: the most-offered line (every line for alternates), the median fair chance and
    the best price on each side. Team totals get one entry per team."""
    if odds.empty:
        return []
    out = []
    mains = {}
    for (gid, market), g in sorted(odds.groupby(["game_id", "market"]), key=lambda x: x[0][1] in ALT_GAME):
        if market not in OA.ALL_GAME_MARKETS:
            continue
        if market == "team_totals":
            for team in ("home", "away"):
                sub = g[g["side"].str.startswith(team + "_")]
                sub = sub.assign(side=sub["side"].str.split("_").str[-1], key_line=sub["line"])
                line = _main_line(sub, "over", "under")
                e = _entry(gid, market, sub, "over", "under", line, team) if line is not None else None
                if e:
                    out.append(e)
            continue
        a_side, b_side = ("over", "under") if market in TOTALS_LIKE else ("home", "away")
        g = _keyed(g, market)
        if market in ALT_GAME:
            main = mains.get((gid, ALT_GAME[market]))
            for line in sorted(set(g["key_line"])):
                if main is not None and abs(line - main) < 1e-9:
                    continue
                e = _entry(gid, market, g, a_side, b_side, line)
                if e:
                    out.append(e)
            continue
        line = _main_line(g, a_side, b_side)
        if line is None:
            continue
        mains[(gid, market)] = line
        e = _entry(gid, market, g, a_side, b_side, line)
        if e:
            out.append(e)
    return out


def _normal(mu: float, sd: float, thr: float):
    """(P above thr, P exactly thr) for a whole-point score around a normal curve."""
    if float(thr).is_integer():
        win = 1 - st.norm.cdf(thr + 0.5, mu, sd)
        push = st.norm.cdf(thr + 0.5, mu, sd) - st.norm.cdf(thr - 0.5, mu, sd)
    else:
        win, push = 1 - st.norm.cdf(thr, mu, sd), 0.0
    return float(win), float(push)


def game_probs(sim: dict, market: str, line: float, report: dict, team: str = None):
    """(P first side wins, P push) from the simulation: home for moneyline/spread, over for totals.

    Spreads and totals use a normal curve with the size of miss measured when the model was replayed on a
    past season (RMSE). First halves and team totals scale that miss by the simulation's own spread of each
    number against the full-game one. The replay's average bias is no longer applied (until Nov 16, 2026 it
    was, with the wrong sign: subtracted where it should have been added): it was measured before the
    Nov 11 scoring-level fix (team_rates relevel), which removed most of it."""
    m_rmse = float(report.get("margin_rmse") or 14.0)
    t_rmse = float(report.get("total_rmse") or 19.0)
    m_bias = t_bias = 0.0
    h1 = sim.get("h1") or {}
    if market == "h2h":
        return float(sim["home_win"]), 0.0
    if market == "h2h_h1":
        if not h1:
            raise KeyError("no first-half simulation")
        return float(h1["home_win"]), float(h1["tie"])
    if market in ("spreads", "alternate_spreads"):
        return _normal(float(sim["margin"]) + m_bias, m_rmse, -line)      # home covers when margin > -line
    if market in ("totals", "alternate_totals"):
        return _normal(float(sim["total"]) + t_bias, t_rmse, line)
    if market == "spreads_h1":
        sd = m_rmse * float(h1["margin_sd"]) / float(sim["margin_sd"])
        return _normal(float(h1["margin_mean"]) + m_bias / 2, sd, -line)
    if market == "totals_h1":
        sd = t_rmse * float(h1["total_sd"]) / float(sim["total_sd"])
        return _normal(float(h1["total_mean"]) + t_bias / 2, sd, line)
    if market == "team_totals":
        sd = t_rmse * float(sim[f"{team}_pts_sd"]) / float(sim["total_sd"])
        return _normal(float(sim[f"{team}_pts"]) + t_bias / 2, sd, line)
    raise KeyError(market)


# ---------------------------------------------------------------- candidates

def kelly_units(p: float, price: int, cap: float = None) -> float:
    b = PM.decimal(price) - 1
    f = (p * b - (1 - p)) / b
    if f <= 0:
        return 0.0
    u = KELLY * f * 100 / UNIT_PCT
    u = min(MAX_UNITS if cap is None else cap, round(u * 4) / 4)
    return u if u >= 0.25 else 0.0


def _candidate(kind, game_id, side, line, price, book, p_model, p_market, push, books, min_books=MIN_BOOKS, **extra):
    """p_model / p_market: chance this side wins, pushes excluded."""
    if price is None or not (MIN_PRICE <= price <= MAX_PRICE) or books < min_books:
        return None
    gap = p_model - p_market
    if gap < MIN_GAP or gap > MAX_GAP:
        return None
    p = BLEND * p_model + (1 - BLEND) * p_market
    ev = (1 - push) * (p * (PM.decimal(price) - 1) - (1 - p))
    if ev < MIN_EV:
        return None
    full = kelly_units(p, price, cap=FULL_MAX_UNITS)
    units = min(full, MAX_UNITS)
    if units <= 0:
        return None
    c = {"kind": kind, "game_id": game_id, "side": side, "line": line, "price": int(price), "book": book,
         "p_model": round(p_model, 4), "p_market": round(p_market, 4), "p_used": round(p, 4),
         "ev": round(ev, 4), "units": units, "units_full": full, "books": books, **extra}
    c["gate"] = gate_key(kind, c.get("stat"), side)
    return c


def prop_candidates(rows: list) -> list:
    """Every prop side that passes the filters, overs included: in tracking mode overs are tracked, and they
    can only be recommended once their market earns it (PROP_OVERS stays False: no override)."""
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
            best = (r["best_over"] if side == "over" else r["best_under"]) or {}
            c = _candidate("prop", r["game_id"], side, r["line"], best.get("price"), best.get("book"), p_mod, p_mkt,
                           m.get("p_push", 0.0), n, MIN_BOOKS_ALT if r.get("alt") else MIN_BOOKS,
                           player_id=r["player_id"], stat=r["stat"],
                           model_mean=m["mean"], model_min=m["min"], fetched_at=r.get("fetched_at"))
            if c:
                out.append(c)
    return out


def moderate_candidates(rows: list) -> list:
    """The moderate-disagreement rule (see MOD_*): no EV or gap filter, the model's lean decides the side.
    select() keeps one per player: the one with the best expected value."""
    out = []
    for r in rows:
        m = r.get("model")
        if r.get("alt") or r["stat"] not in MOD_STATS or not m or (m.get("games") or 0) < 5 or \
                m.get("status") in ("out", "doubtful", "questionable"):
            continue
        if m["p_over"] + m["p_under"] <= 0 or len(r["books"]) < MIN_BOOKS:
            continue
        po = m["p_over"] / (m["p_over"] + m["p_under"])
        if MOD_OVER[0] <= po <= MOD_OVER[1]:
            side, p_mod, p_mkt, best = "over", po, r["fair_over"], r["best_over"]
        elif MOD_UNDER[0] <= po <= MOD_UNDER[1]:
            side, p_mod, p_mkt, best = "under", 1 - po, 1 - r["fair_over"], r["best_under"]
        else:
            continue
        price = (best or {}).get("price")
        if price is None or not (MIN_PRICE <= price <= MAX_PRICE) or p_mod <= p_mkt:
            continue
        p = BLEND * p_mod + (1 - BLEND) * p_mkt
        ev = p * PM.decimal(price) - 1
        out.append({"kind": "prop", "game_id": r["game_id"], "side": side, "line": r["line"], "price": int(price),
                    "book": best.get("book"), "p_model": round(p_mod, 4), "p_market": round(p_mkt, 4),
                    "p_used": round(p, 4), "ev": round(ev, 4), "units": MOD_UNITS, "units_full": MOD_UNITS,
                    "books": len(r["books"]), "player_id": r["player_id"], "stat": r["stat"],
                    "model_mean": m["mean"], "model_min": m["min"], "fetched_at": r.get("fetched_at"),
                    "trigger": "moderate", "gate": "moderate"})
    return out


def game_candidates(table: list, sims: dict) -> list:
    by_game = {g["game_id"]: g for g in (sims or {}).get("games", [])}
    report = ((sims or {}).get("model") or {}).get("report", {}).get("scores", {})
    out = []
    for t in table:
        sim = by_game.get(t["game_id"])
        if not sim:
            continue
        try:
            p_first, push = game_probs(sim, t["market"], t["line"], report, t.get("team"))
        except (KeyError, TypeError, ZeroDivisionError):        # e.g. a simulation from before first halves
            continue
        p_first_np = p_first / (1 - push) if push < 1 else 0.5
        a, b = t["sides"]
        kind = GAME_KIND[t["market"]]
        for side, p_mod, p_mkt in ((a, p_first_np, t["fair"]), (b, 1 - p_first_np, 1 - t["fair"])):
            line = None if kind.startswith("moneyline") else \
                (t["line"] if side in ("home", "over", "under") else -t["line"])
            c = _candidate(kind, t["game_id"], f"{t['team']}_{side}" if t.get("team") else side, line,
                           t["best"][side]["price"], t["best"][side]["book"],
                           p_mod, p_mkt, push, len(t["books"]),
                           MIN_BOOKS_ALT if t["market"] in ALT_GAME else MIN_BOOKS,
                           sim_margin=sim["margin"], sim_total=sim["total"], fetched_at=t.get("fetched_at"))
            if c:
                out.append(c)
    return out


def pick_key(c: dict) -> tuple:
    """One pick per player and market, and per game and market family (each rule on its own)."""
    if c.get("trigger") == "moderate":      # one per player (his stats move together), the biggest edge
        return ("moderate", c["game_id"], c["player_id"])
    if c["kind"] == "prop":
        return ("prop", c["game_id"], c["player_id"], c["stat"])
    if c["kind"] in ("moneyline", "spread"):
        return ("side", c["game_id"])
    if c["kind"] in ("moneyline_h1", "spread_h1"):
        return ("side_h1", c["game_id"])
    if c["kind"] == "team_total":
        return ("team_total", c["game_id"], c["side"].split("_")[0])
    return (c["kind"], c["game_id"])


def select(cands: list, gate_info: dict = None) -> list:
    """The picks: one per player and market, one per game and market family. Each is marked 'rec' when its
    market has earned recommendations, else 'track'. Recommendations then keep one prop per player and fit
    the per-game stake cap; tracked picks keep their stakes (no game cap: they are not bets)."""
    cands = sorted(cands, key=lambda c: -c["ev"])
    picked, seen = [], set()
    for c in cands:
        key = pick_key(c)
        if key in seen:
            continue
        seen.add(key)
        c["mode"] = "rec" if _earned(c, gate_info or {}) else "track"
        picked.append(c)
    seen_player = set()
    for c in picked:                     # recommendations: one prop per player
        if c["mode"] == "rec" and c["kind"] == "prop":
            k = (c["game_id"], c["player_id"])
            if k in seen_player:
                c["mode"] = "track"
            seen_player.add(k)
    by_game = defaultdict(list)
    for c in picked:
        if c["mode"] == "rec":
            by_game[c["game_id"]].append(c)
    for gid, cs in by_game.items():
        tot = sum(c["units"] for c in cs)
        if tot > MAX_GAME_UNITS:
            f = MAX_GAME_UNITS / tot
            for c in cs:
                c["units"] = max(0.25, math.floor(c["units"] * f * 4) / 4)
            # every qualifying bet stays; if the 0.25 minimum keeps the game a little over, so be it
    picked.sort(key=lambda c: (c["mode"] != "rec", -c["ev"]))
    keep, units = [], 0.0
    for c in picked:
        if c["mode"] == "rec" and ((MAX_DAY_PICKS is not None and len(keep) >= MAX_DAY_PICKS) or
                                   (MAX_DAY_UNITS is not None and units + c["units"] > MAX_DAY_UNITS)):
            continue
        keep.append(c)
        units += c["units"] if c["mode"] == "rec" else 0.0
    return keep


# ---------------------------------------------------------------- ledger

def lock(conn, picks: list, games: dict, now_iso: str) -> int:
    """Store picks not already on the ledger for that game, market and player. Returns how many were added."""
    settings = json.dumps({"blend": BLEND, "min_ev": MIN_EV, "max_gap": MAX_GAP, "kelly": KELLY,
                           "max_units": MAX_UNITS, "max_game_units": MAX_GAME_UNITS,
                           "prop_overs": PROP_OVERS, "tracking": True})
    added = 0
    for c in picks:
        g = games.get(c["game_id"])
        if not g or not g.get("tip_utc") or g["tip_utc"] <= now_iso:
            continue
        n_day, u_day = conn.execute(
            """SELECT COUNT(*), COALESCE(SUM(units), 0) FROM bets b JOIN games x ON x.game_id = b.game_id
               WHERE x.game_date = ?""", (g["game_date"],)).fetchone()
        if (MAX_DAY_PICKS is not None and n_day >= MAX_DAY_PICKS) or \
                (MAX_DAY_UNITS is not None and u_day + c["units"] > MAX_DAY_UNITS):
            continue
        if c["kind"] == "prop":
            rule = "COALESCE(trigger,'') = 'moderate'" if c.get("trigger") == "moderate" else \
                "COALESCE(trigger,'') != 'moderate'"
            exists = conn.execute(f"SELECT 1 FROM bets WHERE game_id=? AND kind='prop' AND player_id=? AND stat=? AND {rule}",
                                  (c["game_id"], c["player_id"], c["stat"])).fetchone()
        elif c["kind"] in ("moneyline", "spread"):
            exists = conn.execute("SELECT 1 FROM bets WHERE game_id=? AND kind IN ('moneyline','spread')",
                                  (c["game_id"],)).fetchone()
        elif c["kind"] in ("moneyline_h1", "spread_h1"):
            exists = conn.execute("SELECT 1 FROM bets WHERE game_id=? AND kind IN ('moneyline_h1','spread_h1')",
                                  (c["game_id"],)).fetchone()
        elif c["kind"] == "team_total":
            exists = conn.execute("SELECT 1 FROM bets WHERE game_id=? AND kind='team_total' AND side LIKE ?",
                                  (c["game_id"], c["side"].split("_")[0] + "_%")).fetchone()
        else:
            exists = conn.execute("SELECT 1 FROM bets WHERE game_id=? AND kind=?", (c["game_id"], c["kind"])).fetchone()
        if exists:
            continue
        conn.execute(
            """INSERT INTO bets (created_at, game_id, kind, player_id, stat, side, line, price, book, p_model,
                 p_market, p_used, ev, units, units_full, settings, mode, gate, trigger)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_iso, c["game_id"], c["kind"], c.get("player_id"), c.get("stat"), c["side"], c["line"], c["price"],
             c["book"], c["p_model"], c["p_market"], c["p_used"], c["ev"], c["units"], c.get("units_full"), settings,
             c.get("mode", "rec"), c.get("gate"), c.get("trigger")))
        added += 1
    conn.commit()
    return added


def _same(c: dict, b: dict) -> bool:
    if c["game_id"] != b["game_id"] or c["kind"] != b["kind"] or c["side"] != b["side"]:
        return False
    if (c.get("trigger") == "moderate") != (b.get("trigger") == "moderate"):
        return False
    return c["kind"] != "prop" or (int(c["player_id"]) == int(b["player_id"]) and c["stat"] == b["stat"])


def _odds(p) -> str:
    return f"+{int(p)}" if p > 0 else str(int(p))


def _why(b: dict, cands: list, prop_rows: list, table: list) -> str:
    """Plain reason a recommendation no longer qualifies at the latest prices."""
    if any(_same(c, b) for c in cands):
        return "another bet on the same player or game now looks better"
    if b["kind"] == "prop":
        alt = str(b["stat"]).startswith("alt_")
        r = next((r for r in prop_rows if r["game_id"] == b["game_id"] and r["player_id"] == b["player_id"]
                  and r["stat"] == b["stat"] and (not alt or r["line"] == b["line"])), None)
        if r is None:
            return "no longer offered by enough legal books"
        m = r.get("model") or {}
        if m.get("status") in ("out", "doubtful", "questionable"):
            return f"player listed as {m['status']}"
        if r["line"] != b["line"]:
            return f"line moved to {r['line']:g}"
        best = (r["best_over"] if b["side"] == "over" else r["best_under"]) or {}
        price, fair = best.get("price"), r["fair_over"] if b["side"] == "over" else 1 - r["fair_over"]
        po = m["p_over"] / (m["p_over"] + m["p_under"]) if m and m.get("p_over") is not None and m["p_over"] + m["p_under"] > 0 else None
        mod = po if b["side"] == "over" else (1 - po if po is not None else None)
    else:
        mk = KIND_MARKET[b["kind"]]
        team, side = (b["side"].split("_") if b["kind"] == "team_total" else (None, b["side"]))
        alt = mk in ALT_GAME
        t = next((t for t in table if t["game_id"] == b["game_id"] and t["market"] == mk
                  and t.get("team") == team and (not alt or b["line"] is None or abs(
                      (t["line"] if side in ("home", "over", "under") else -t["line"]) - b["line"]) < 1e-9)), None)
        if t is None:
            return "no longer offered by enough legal books"
        line = None if mk.startswith("h2h") else (t["line"] if side in ("home", "over", "under") else -t["line"])
        if line is not None and b["line"] is not None and abs(line - b["line"]) > 1e-9:
            return f"line moved to {line:+g}" if mk in SPREADS_LIKE else f"line moved to {line:g}"
        price = (t["best"].get(side) or {}).get("price")
        fair = t["fair"] if side in t["sides"][:1] else 1 - t["fair"]
        mod = None
    if price is not None and PM.decimal(price) < PM.decimal(b["price"]) - 1e-9:
        return f"price moved to {_odds(price)}"
    if mod is not None:
        return f"edge dropped: model {100 * mod:.0f}%, market {100 * fair:.0f}%"
    return f"edge dropped: market now {100 * fair:.0f}%"


def track(conn, picks: list, cands: list, prop_rows: list, table: list, games: dict, now_iso: str) -> dict:
    """Update each open recommendation's state: still recommended, no longer recommended (and why), or started."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(bets)")]
    rows = [dict(zip(cols, r)) for r in conn.execute(
        "SELECT * FROM bets WHERE status = 'open' AND COALESCE(rec_state, '') != 'started'").fetchall()]
    tips = dict(conn.execute("SELECT game_id, tip_utc FROM games WHERE game_id IN (%s)" % ",".join("?" * len(rows)),
                             [b["game_id"] for b in rows]).fetchall()) if rows else {}
    prop_games = {r["game_id"] for r in prop_rows}
    line_games = {t["game_id"] for t in table}
    n = Counter()
    for b in rows:
        tip = tips.get(b["game_id"]) or (games.get(b["game_id"]) or {}).get("tip_utc")
        if tip and tip <= now_iso:
            conn.execute("UPDATE bets SET rec_state='started', state_at=?, ended_at=COALESCE(ended_at, ?) WHERE bet_id=?",
                         (now_iso, tip, b["bet_id"]))
            n["started"] += 1
            continue
        if b["game_id"] not in games:          # not on this run's slate: leave as it was
            continue
        if b["game_id"] not in (prop_games if b["kind"] == "prop" else line_games):
            continue                           # no prices for this game this run (e.g. a failed step): no change
        live = next((c for c in picks if _same(c, b)), None)
        if live is not None and (b["kind"] == "moneyline" or live["line"] == b["line"]):
            conn.execute("""UPDATE bets SET rec_state='active', state_at=?, ended_at=NULL, end_reason=NULL,
                            now_price=?, now_line=?, now_units=?, now_book=? WHERE bet_id=?""",
                         (now_iso, live["price"], live["line"], live["units"], live["book"], b["bet_id"]))
            n["active"] += 1
        elif b.get("rec_state") != "expired":
            why = f"line moved to {live['line']:g}" if live is not None else _why(b, cands, prop_rows, table)
            conn.execute("UPDATE bets SET rec_state='expired', state_at=?, ended_at=?, end_reason=? WHERE bet_id=?",
                         (now_iso, now_iso, why, b["bet_id"]))
            n["expired"] += 1
    conn.commit()
    return dict(n)


def _closing_fair(conn, bet: dict, tip: str):
    """Market fair chance for the bet's side at the last legal-book pull before tip-off, at the same line.

    None when that last pull is the one the pick was made from (or earlier): there is no later price to
    compare with, so the pick says nothing about the closing line (until Nov 16, 2026 these counted as 0)."""
    q = """SELECT book, side, line, price_american, fetched_at FROM odds_snapshots
           WHERE game_id=? AND source=? AND fetched_at=(SELECT MAX(fetched_at) FROM odds_snapshots
                 WHERE game_id=? AND source=? AND fetched_at < ? AND {w}) AND {w}"""
    kind, side = bet["kind"], bet["side"]
    team = None
    if kind == "prop":
        mk = OA.STAT_MARKET.get(bet["stat"])
        if mk is None:
            return None
        w = f"player_id={int(bet['player_id'])} AND market='{mk}'"
    else:
        mk = KIND_MARKET[kind]
        w = f"player_id IS NULL AND market='{mk}'"
        if kind == "team_total":
            team, side = side.split("_")
    rows = conn.execute(q.format(w=w), (bet["game_id"], OA.SOURCE, bet["game_id"], OA.SOURCE, tip)).fetchall()
    if not rows or (bet.get("created_at") and rows[0][4] <= bet["created_at"]):
        return None
    by_book = defaultdict(dict)
    for book, s_, line, price, _ in rows:
        if book not in OA.LEGAL_BOOKS:
            continue
        if team is not None:
            if not s_.startswith(team + "_"):
                continue
            s_ = s_.split("_")[-1]
        if kind != "prop" and mk in SPREADS_LIKE:
            key = line if s_ == "home" else -line          # home number
            want = bet["line"] if side == "home" else -bet["line"]
        elif kind != "prop" and mk.startswith("h2h"):
            key, want = 0.0, 0.0
        else:
            key, want = line, bet["line"]
        if abs(key - want) < 1e-9:
            by_book[book][s_] = price
    other = {"over": "under", "under": "over", "home": "away", "away": "home"}[side]
    fairs = []
    for b, sides in by_book.items():
        if kind == "prop" and str(bet["stat"]).startswith("alt_"):
            from .props import alt_fair
            fo = alt_fair(sides)
            if fo is not None:
                fairs.append(fo if side == "over" else 1 - fo)
        elif side in sides and other in sides:
            f, _ = PM.devig(sides[side], sides[other])
            if f is not None:
                fairs.append(f)
    return float(np.median(fairs)) if fairs else None


def first_half(conn, game_ids) -> dict:
    """game_id -> (home, away) first-half points from the play-by-play, only where the play-by-play adds up
    to the final score (otherwise a first-half bet waits, and is void after three days)."""
    if not game_ids:
        return {}
    q = ",".join("?" * len(game_ids))
    rows = conn.execute(
        f"""SELECT g.game_id, g.home_score, g.away_score,
                   SUM(CASE WHEN p.team_id = g.home_team_id THEN p.pts END),
                   SUM(CASE WHEN p.team_id = g.away_team_id THEN p.pts END),
                   SUM(CASE WHEN p.team_id = g.home_team_id AND p.period <= 2 THEN p.pts END),
                   SUM(CASE WHEN p.team_id = g.away_team_id AND p.period <= 2 THEN p.pts END)
            FROM games g JOIN pbp_poss p ON p.game_id = g.game_id
            WHERE g.game_id IN ({q}) GROUP BY g.game_id""", list(game_ids)).fetchall()
    return {r[0]: (r[5], r[6]) for r in rows if r[1] == r[3] and r[2] == r[4] and r[5] is not None}


def grade(conn, season: str) -> int:
    open_bets = [dict(r) for r in conn.execute(
        """SELECT b.*, b.line AS rec_line, g.status AS gstatus, g.home_score, g.away_score, g.tip_utc FROM bets b
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
    try:
        halves = first_half(conn, [b["game_id"] for b in finals if b["kind"].endswith("_h1")])
    except Exception:                           # no play-by-play table
        halves = {}
    old = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    now = utc_now_iso()
    n = 0
    for b in voided:
        conn.execute("UPDATE bets SET status='void', pnl=0, graded_at=? WHERE bet_id=?", (now, b["bet_id"]))
        n += 1
    for b in finals:
        if b.get("taken_line") is not None:
            b["line"] = b["taken_line"]           # graded at the line you got
        if b["kind"] == "prop":
            key = (b["game_id"], b["player_id"])
            if key not in logs.index or float(logs.loc[key, "min"]) <= 0:
                conn.execute("UPDATE bets SET status='void', pnl=0, graded_at=? WHERE bet_id=?", (now, b["bet_id"]))
                n += 1
                continue
            stat = b["stat"][4:] if str(b["stat"]).startswith("alt_") else b["stat"]
            val = float(logs.loc[key, stat])
            diff = val - b["line"] if b["side"] == "over" else b["line"] - val
        else:
            if b["home_score"] is None or b["away_score"] is None:
                continue
            hs, as_ = b["home_score"], b["away_score"]
            if b["kind"].endswith("_h1"):
                if b["game_id"] not in halves:
                    if (b["tip_utc"] or now) < old:
                        conn.execute("UPDATE bets SET status='void', pnl=0, graded_at=? WHERE bet_id=?",
                                     (now, b["bet_id"]))
                        n += 1
                    continue
                hs, as_ = halves[b["game_id"]]
            margin, total = hs - as_, hs + as_
            side = b["side"]
            if b["kind"] in ("moneyline", "moneyline_h1"):
                val = margin
                diff = margin if side == "home" else -margin
            elif b["kind"] in ("spread", "spread_h1", "alt_spread"):
                val = margin
                diff = (margin + b["line"]) if side == "home" else (-margin + b["line"])
            elif b["kind"] == "team_total":
                team, side = side.split("_")
                val = hs if team == "home" else as_
                diff = val - b["line"] if side == "over" else b["line"] - val
            else:
                val = total
                diff = total - b["line"] if side == "over" else b["line"] - total
        status = "won" if diff > 0 else "lost" if diff < 0 else "push"
        price = b["taken_price"] if b.get("taken_price") is not None else b["price"]
        pnl = b["units"] * (PM.decimal(price) - 1) if status == "won" else -b["units"] if status == "lost" else 0.0
        rec = dict(b, line=b["rec_line"])         # closing-line value grades the recommended pick
        close = _closing_fair(conn, rec, b["tip_utc"] or now)
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
            "beat_close": round(float((clv > BEAT_EPS).mean()), 3) if len(clv) else None, "clv_n": int(len(clv))}


def _ledger(conn) -> pd.DataFrame:
    ledger = pd.read_sql_query(
        """SELECT b.*, g.game_date, g.home_team_id, g.away_team_id, g.tip_utc, g.home_score, g.away_score
           FROM bets b JOIN games g ON g.game_id = b.game_id ORDER BY b.created_at DESC, b.bet_id DESC""", conn)
    if len(ledger):
        ledger["gate"] = [g if isinstance(g, str) and g else gate_key(k, st_, sd)
                          for g, k, st_, sd in zip(ledger["gate"], ledger["kind"], ledger["stat"], ledger["side"])]
        ledger["mode"] = ledger["mode"].fillna("rec")
    return ledger


TRACK_DAYS = 10          # tracked picks older than this stay in the scorecard but are left out of bets.json


def export(conn, out_dir, season: str, games: list, sims: dict, prop_rows: list, pulled: dict, write) -> dict:
    ensure_table(conn)
    from .props import _latest_odds
    gmap = {g["game_id"]: g for g in games}
    table = game_market_table(_latest_odds(conn, list(gmap), props=False))
    table = [t for t in table if t["market"] not in OA.OLD_EVENT_GAME_MARKETS]      # no longer tracked
    cands = prop_candidates(prop_rows or []) + game_candidates(table, sims)
    news = set((pulled or {}).get("news_games") or [])
    for c in cands:                     # picked from a news pull's prices
        if c["game_id"] in news and c.get("fetched_at") == (pulled or {}).get("fetched_at"):
            c["trigger"] = "news"
    cands += moderate_candidates(prop_rows or [])
    if not PICKS_ENABLED:
        cands = []
    gate_info = gates(_ledger(conn))
    picks = select(cands, gate_info)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # lock on any run while the pick's own prices are fresh, not only right after a pull
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=FRESH_MINUTES)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lockable = [c for c in picks if (c.get("fetched_at") or "") >= cutoff]
    added = lock(conn, lockable, gmap, now_iso)
    try:
        states = track(conn, picks, cands, prop_rows or [], table, gmap, now_iso) if PICKS_ENABLED else {}
    except Exception:
        log.exception("Tracking recommendation states failed")
        states = {}
    for c in picks:
        c["odds_at"] = c.get("fetched_at")
    graded = grade(conn, season)
    ledger = _ledger(conn)
    gate_info = gates(ledger)
    names = dict(conn.execute("SELECT player_id, full_name FROM players").fetchall())
    ledger["player"] = ledger["player_id"].map(lambda p: names.get(int(p)) if pd.notna(p) else None) \
        if len(ledger) else None
    real = ledger[ledger["mode"] == "rec"] if len(ledger) else ledger      # the bets: recommendations only
    by_kind = {k: summary(real[real["kind"] == k]) for k in KIND_LABEL if (real["kind"] == k).any()} \
        if len(real) else {}
    by_market = {}
    if len(real):
        mk = [market_of(k, s_) for k, s_ in zip(real["kind"], real["stat"])]
        real = real.assign(market=mk)
        by_market = {m: summary(real[real["market"] == m]) for m in MARKET_ORDER if m in set(mk)}
    tracked = ledger[ledger["mode"] == "track"] if len(ledger) else ledger
    cols = ["bet_id", "created_at", "game_id", "game_date", "home_team_id", "away_team_id", "tip_utc", "kind",
            "player_id", "player", "stat", "side", "line", "price", "book", "p_model", "p_market", "p_used", "ev",
            "units", "status", "result", "pnl", "close_fair", "clv", "home_score", "away_score",
            "taken_price", "taken_line", "taken_book", "taken_at", "units_full"] + \
        [c for c, _ in STATE_COLUMNS] + [c for c, _ in MODE_COLUMNS]
    if len(ledger):
        since = (datetime.now(timezone.utc) - timedelta(days=TRACK_DAYS)).strftime("%Y-%m-%d")
        shown = ledger[(ledger["mode"] == "rec") | (ledger["game_date"] >= since)]
        led = shown[cols].astype(object).where(pd.notna(shown[cols]), None)
    else:
        led = ledger
    game_lines = [{k: v for k, v in t.items() if k != "sides"} | {"sides": list(t["sides"])} for t in table
                  if t["market"] in OA.GAME_MARKETS]
    order = {k: i for i, k in enumerate(MARKET_ORDER)}
    scorecard = sorted(gate_info.values(), key=lambda g: (g["key"] == "news", g["status"] != "earned",
                                                          order.get(g["key"].split(":")[0], 99), g["key"]))
    payload = {
        "generated_at": utc_now_iso(), "season": season,
        "settings": {"blend": BLEND, "min_ev": MIN_EV, "min_gap": MIN_GAP, "max_gap": MAX_GAP, "min_books": MIN_BOOKS, "kelly": KELLY,
                     "unit_pct": UNIT_PCT, "max_units": MAX_UNITS, "full_max_units": FULL_MAX_UNITS, "prop_overs": PROP_OVERS, "max_game_units": MAX_GAME_UNITS,
                     "max_day_picks": MAX_DAY_PICKS, "max_day_units": MAX_DAY_UNITS,
                     "price_range": [MIN_PRICE, MAX_PRICE], "legal_books": sorted(OA.LEGAL_BOOKS), "bet_books": sorted(OA.BET_BOOKS),
                     "tracking": True, "picks_enabled": PICKS_ENABLED, "gate": {"min_graded": GATE_MIN_GRADED, "min_clv": GATE_MIN_CLV,
                                                "clv_z": GATE_CLV_Z}, "track_days": TRACK_DAYS},
        "current": [c for c in picks if c["mode"] == "rec"],
        "tracking": [c for c in picks if c["mode"] == "track"],
        "candidates": len(cands), "added_this_run": added, "graded_this_run": graded,
        "odds_as_of": max((c["fetched_at"] for c in picks if c.get("fetched_at")), default=None),
        "next_pull": pulled.get("reason") if pulled.get("status") == "skipped" else None,
        "game_lines": game_lines,
        "ledger": {"cols": cols, "rows": led.values.tolist() if len(led) else []},
        "summary": summary(real) if len(real) else {"bets": 0, "graded": 0}, "by_kind": by_kind,
        "by_market": by_market,
        "tracked_summary": summary(tracked) if len(tracked) else {"bets": 0, "graded": 0},
        "scorecard": scorecard,
    }
    write(out_dir / "bets.json", payload)
    return {"picks": len(payload["current"]), "tracked": len(payload["tracking"]), "added": added, "graded": graded,
            "ledger": int(len(ledger)), "states": states}
