"""
Score every shot on goal with the saved xG model -> data/xg/shot_xg.parquet.
Runs in CI after normalize.py (no training there, so xG values only change when a
new model is promoted or the underlying data changes):

  python xg/score.py                 # model named in models/current.txt
  python xg/score.py --version v4

Models with season_models in their meta (one booster per season year, weighted
towards it; see train.py HALF_LIFE) score each shot with its season's model; seasons
after the last trained one use the next-season model.

In-season calibration (models with offset_k in their meta, see train.py OFFSET_K):
each season year gets an offset b on the log-odds scale, xg = sigmoid(logit(p) + b),
  b = G / (G + K) * b_hat
b_hat makes the season's model-scored shots of the training season types (regular +
playoffs) add up to their goals; G = their unadjusted xG. A season with no such
shots yet gets b = 0. Recomputed on every run, so a running season's xG moves
slightly as games come in. Fixed-value rows (empty net, penalty shots) are not
adjusted. xg_unadjusted and season_offset are in the output.

Output: one row per `shot` event (goals included) and per `penaltyshot` event,
keyed by (game_id, event_idx) like data/clean/events; goal_event_idx links a goal to
its `goal` row. xg_source says where xg comes from:
  model         the model
  empty_net     fixed XG_EMPTY_NET (1.0)
  penalty_shot  fixed XG_PENALTY_SHOT (0.33)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

import features as F
from train import settings

MODELS = F.ROOT / "models"
OUT = F.ROOT / "data" / "xg" / "shot_xg.parquet"
OUT_COLS = ["game_id", "event_idx", "goal_event_idx", "season_id", "season_year", "period", "period_time_s",
            "team_id", "shooter_id", "goalie_id", "is_goal", "xg_source", "xg", "xg_unadjusted",
            "season_offset", "model"]


def load(version=None):
    version = version or (MODELS / "current.txt").read_text().strip()
    meta = json.loads((MODELS / f"xg_{version}.meta.json").read_text())
    now = {"features": F.feature_names(), "categorical": F.CATEGORICAL, "settings": settings()}
    changed = [k for k, v in now.items() if json.loads(json.dumps(v)) != meta.get(k)]
    if changed:
        raise SystemExit(f"features.py no longer matches model {version} ({', '.join(changed)}): "
                         "train a new version instead of editing features under an old one")
    # one model per season year (season_models), or one model for all seasons (v1-v3)
    files = meta.get("season_models") or {"all": f"xg_{version}.json"}
    boosters = {}
    for year, name in files.items():
        boosters[year] = xgb.Booster()
        boosters[year].load_model(MODELS / name)
    return boosters, meta


def model_year(meta, season_year):
    """The season model for a season year: its own, else the nearest one (seasons after
    the last trained one use the next-season model)."""
    if "season_models" not in meta:
        return "all"
    years = sorted(int(y) for y in meta["season_models"])
    return str(min(max(int(season_year), years[0]), years[-1]))


def predict(boosters, meta, df):
    """xG for shot rows: the model where xg_source = model, fixed values otherwise."""
    p = np.full(len(df), np.nan)
    which = df["season_year"].map(lambda y: model_year(meta, y)).to_numpy()
    for year, b in boosters.items():
        m = which == year
        if m.any():
            p[m] = b.predict(xgb.DMatrix(df.loc[m, meta["features"]], enable_categorical=True))
    return np.select([df["xg_source"] == "model", df["xg_source"] == "empty_net",
                      df["xg_source"] == "penalty_shot"],
                     [p, F.XG_EMPTY_NET, F.XG_PENALTY_SHOT], np.nan)


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_offset(p, goals):
    """b with sum(sigmoid(logit(p) + b)) = goals (Newton; 0 if no shots)."""
    p = np.asarray(p, float)
    if len(p) == 0:
        return 0.0
    z, b = logit(p), 0.0
    for _ in range(50):
        q = sigmoid(z + b)
        step = (q.sum() - goals) / max((q * (1 - q)).sum(), 1e-9)
        b -= step
        if abs(step) < 1e-10:
            break
    return float(b)


def shrink(b_hat, G, K):
    """Shrink the raw offset towards 0: weight G / (G + K) (K = 0: none, inf: no offset)."""
    if K == np.inf:
        return 0.0
    if K == 0:
        return b_hat
    return G / (G + K) * b_hat


def season_offsets(df, p, K):
    """{season year: {b, b_hat, weight, xg_unadjusted, goals, shots}} from model-scored
    shots of the training season types; p = their unadjusted model xG."""
    m = (df["xg_source"] == "model").to_numpy() & df["season_type"].isin(F.TRAIN_SEASON_TYPES).to_numpy()
    out = {}
    for yr in sorted(df.loc[m, "season_year"].unique()):
        r = m & (df["season_year"] == yr).to_numpy()
        G, goals = float(np.sum(p[r])), int(df.loc[r, "is_goal"].sum())
        b_hat = fit_offset(p[r], goals)
        out[str(int(yr))] = {"b": round(shrink(b_hat, G, K), 6), "b_hat": round(b_hat, 6),
                             "weight": round(G / (G + K), 4), "xg_unadjusted": round(G, 1),
                             "goals": goals, "shots": int(r.sum())}
    return out


def scored(version=None):
    booster, meta = load(version)
    shots = F.build()
    shots["xg_unadjusted"] = predict(booster, meta, shots)
    shots["season_offset"] = 0.0
    offsets = {}
    if meta.get("offset_k") is not None:
        offsets = season_offsets(shots, shots["xg_unadjusted"].to_numpy(), meta["offset_k"])
        shots["season_offset"] = shots["season_year"].map(lambda y: offsets.get(str(int(y)), {"b": 0.0})["b"])
    model = shots["xg_source"] == "model"
    shots["xg"] = shots["xg_unadjusted"]
    shots.loc[model, "xg"] = sigmoid(logit(shots.loc[model, "xg_unadjusted"]) + shots.loc[model, "season_offset"])
    shots.loc[~model, "season_offset"] = 0.0
    ps = F.penalty_shots()
    ps["xg"] = ps["xg_unadjusted"] = F.XG_PENALTY_SHOT
    ps["season_offset"] = 0.0
    out = pd.concat([shots[F.ID_COLS + ["xg", "xg_unadjusted", "season_offset"]], ps], ignore_index=True)
    for c in ("xg", "xg_unadjusted", "season_offset"):
        out[c] = out[c].astype(float).round(5)  # rounded: stable bytes across platforms
    out["model"] = meta["version"]
    out = out.sort_values(["game_id", "event_idx"]).reset_index(drop=True)
    meta = {**meta, "season_offsets": offsets}
    return out[OUT_COLS].astype({"goal_event_idx": "Int64", "shooter_id": "Int64", "goalie_id": "Int64",
                                 "is_goal": "int64"}), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version")
    a = ap.parse_args()
    out, meta = scored(a.version)
    t = pa.Table.from_pandas(out, preserve_index=False)
    if OUT.exists() and pq.read_table(OUT).equals(t):
        print("shot_xg unchanged")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    pq.write_table(t, tmp, compression="zstd")
    os.replace(tmp, OUT)
    by = out.groupby("xg_source").agg(shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum")).round(1)
    print(f"wrote {OUT.relative_to(F.ROOT)} with model {meta['version']}:\n{by.to_string()}")
    if meta["season_offsets"]:
        print("season offsets:\n" + pd.DataFrame(meta["season_offsets"]).T.to_string())


if __name__ == "__main__":
    main()
