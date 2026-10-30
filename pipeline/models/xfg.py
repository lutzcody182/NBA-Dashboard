"""Expected field goal %: the chance a league-average shooter makes a shot,
given where and how it was taken.

Inputs per shot (from pbp_plays): distance (piecewise), zone, shot type
(dunk, tip, layup, hook, floater, jumper, pullup, stepback, fadeaway), how
and how long ago the possession began (transition, putback, late in the
possession), home/away, close late game, and the angle to the basket.
Nothing that is only known after the shot (assists, blocks) is used.

Fit as a logistic regression. calibrate() checks it on the last 30% of a
season using a model fit on the first 70%; the export shows that table.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ZONES = ("rim", "paint", "mid", "c3", "ab3")
FAMILIES = ("dunk", "tip", "layup", "hook", "floater", "jumper", "pullup", "stepback", "fadeaway")
KNOTS = (0, 3, 6, 10, 16, 22, 26, 30)


def features(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    dist = d["dist"].astype(float).fillna(d["dist"].median()).clip(0, 40)
    for z in ZONES:
        X["z_" + z] = (d["zone"] == z).astype(float)
    for k in KNOTS:
        X[f"d{k}"] = np.maximum(dist - k, 0)
    for f in FAMILIES:
        X["f_" + f] = (d["family"] == f).astype(float)
    for f in ("layup", "jumper", "pullup", "floater", "hook"):
        X["fd_" + f] = X["f_" + f] * dist
    age = d["poss_age"].astype(float).fillna(12)
    X["putback"] = d["putback"].fillna(0).astype(float)
    X["transition"] = ((age <= 8) & d["poss_kind"].isin(["dreb", "steal"])).astype(float)
    X["after_steal"] = (d["poss_kind"] == "steal").astype(float)
    X["late"] = (age >= 20).astype(float)
    X["age"] = age.clip(0, 30)
    X["home"] = (d["team_id"] == d["home_team_id"]).astype(float)
    X["close_late"] = ((d["period"] >= 4) & (d["margin"].abs() <= 5)).astype(float)
    X["angle"] = np.abs(np.arctan2(d["x"].astype(float).fillna(0), d["y"].astype(float).fillna(1).clip(lower=0.1)))
    return X


def shots(conn, seasons) -> pd.DataFrame:
    q = ",".join("?" * len(seasons))
    return pd.read_sql_query(
        f"""SELECT p.*, g.game_date, g.season_id, g.home_team_id, g.away_team_id FROM pbp_plays p
            JOIN games g ON g.game_id = p.game_id
            WHERE p.kind = 'fg' AND p.family != 'heave' AND g.season_id IN ({q})
              AND g.season_type IN ('regular', 'playoffs', 'play_in')""", conn, params=list(seasons))


def fit(d: pd.DataFrame) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=3000).fit(features(d), d["made"].astype(int))


def predict(model, d: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features(d))[:, 1]


def calibrate(d: pd.DataFrame) -> dict:
    """Fit on the first 70% of the shots by date, test on the rest."""
    d = d.sort_values(["game_date", "game_id", "n"])
    cut = int(len(d) * 0.7)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    if len(tr) < 5000 or len(te) < 2000:
        return {}
    m = fit(tr)
    p = predict(m, te)
    y = te["made"].to_numpy(float)
    base = np.full(len(y), y.mean())
    zone_rate = tr.groupby("zone")["made"].mean()
    pz = te["zone"].map(zone_rate).fillna(tr["made"].mean()).to_numpy(float)

    def ll(q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    t = te.assign(p=p)
    bins = pd.qcut(t["p"], 10, duplicates="drop")
    table = t.groupby(bins, observed=True).agg(pred=("p", "mean"), actual=("made", "mean"), n=("made", "size"))
    fam = t.groupby("family").agg(pred=("p", "mean"), actual=("made", "mean"), n=("made", "size"))
    return {
        "train_shots": int(len(tr)), "test_shots": int(len(te)), "test_from": str(te["game_date"].iloc[0]),
        "log_loss": round(ll(p), 4), "log_loss_zone_only": round(ll(pz), 4), "log_loss_flat": round(ll(base), 4),
        "bins": [[round(float(r.pred), 3), round(float(r.actual), 3), int(r.n)] for r in table.itertuples()],
        "families": {f: [round(float(r.pred), 3), round(float(r.actual), 3), int(r.n)] for f, r in fam.iterrows()},
    }


def to_payload(model, columns) -> dict:
    return {"columns": list(columns), "coef": [float(c) for c in model.coef_[0]], "intercept": float(model.intercept_[0])}


def predict_payload(payload: dict, d: pd.DataFrame) -> np.ndarray:
    X = features(d)[payload["columns"]].to_numpy(float)
    z = X @ np.array(payload["coef"]) + payload["intercept"]
    return 1.0 / (1.0 + np.exp(-z))

