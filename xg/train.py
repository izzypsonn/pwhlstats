"""
Train the xG model and save it to models/. Run by hand, not in CI:

  python xg/train.py --version v1             # fit + save models/xg_v1.json (+ .meta.json)
  python xg/train.py --version v2 --promote   # ... and make it the model score.py uses

Steps: 5-fold cross-validation grouped by game (a game's shots are all in one fold)
picks the number of boosting rounds with early stopping and reports out-of-fold
log loss / AUC / calibration. The final model is then fit on all training shots
with that many rounds.

Saved files:
  models/xg_<version>.json        the booster, in XGBoost's own JSON format (portable
                                  across XGBoost versions, unlike pickle)
  models/xg_<version>.meta.json   features, category levels, feature settings,
                                  parameters, training data and CV metrics: what
                                  score.py checks against
  models/current.txt              the version score.py uses (only with --promote, or
                                  for the first model)
An existing version is never overwritten: use a new version name.
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb

import features as F

MODELS = F.ROOT / "models"
SEED = 20261125
N_FOLDS = 5
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_cat_to_onehot": 8,
    "seed": SEED,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100
# Season weighting (xg/tune.py): a training shot from season year s gets weight
# 0.5 ** (|s - target| / HALF_LIFE) in the model for season `target` (None = equal
# weights). ROUNDS None = pick the number of rounds by early stopping.
# Tuned 2026-11-25 with xg/tune.py (models/tuning.csv): highest AUC with maximum
# in-season calibration error <= 3% = PARAMS at half-life 0.25, 150 rounds (max error
# 2.3%, 1.6%, 1.7% on three fold splits; AUC 0.716 vs 0.729 with equal weights) = v4.
# Not used since v5: equal weights + a per-season offset (OFFSET_K) calibrates every
# season with no loss of AUC (max error 0.1-0.2% out of fold, AUC 0.729-0.731).
HALF_LIFE = None
ROUNDS = 150
# Per-season calibration offset applied by score.py: xG = sigmoid(logit(p) + b), with
# b = G / (G + OFFSET_K) * b_hat; b_hat makes the season's model-scored shots (training
# season types) add up to their goals, G = their unadjusted xG. Re-estimated on every
# scoring run, so a running season's xG changes slightly as games come in. K = 100
# from the leave-one-season-out replay (xg/replay.py; 25-160 all good, 0 and inf worse).
OFFSET_K = 100


def settings():
    """Feature settings a model depends on (stored in its meta; score.py checks them)."""
    return {"rebound_s": list(F.REBOUND_S), "royal_road_deg": F.ROYAL_ROAD_DEG,
            "x_per_ft": F.X_PER_FT, "y_per_ft": F.Y_PER_FT, "goal_line_ft": F.GOAL_LINE_FT,
            "xg_empty_net": F.XG_EMPTY_NET, "xg_penalty_shot": F.XG_PENALTY_SHOT}


def dmatrix(df, feats, label=True):
    return xgb.DMatrix(df[feats], label=df["is_goal"] if label else None, enable_categorical=True)


def game_folds(game_ids, k=N_FOLDS, seed=SEED):
    games = np.unique(game_ids)
    fold_of = dict(zip(np.random.default_rng(seed).permutation(games), np.arange(len(games)) % k))
    f = np.array([fold_of[g] for g in game_ids])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def log_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    r = np.argsort(np.argsort(p)) + 1.0  # ranks (ties are rare with continuous p)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def calibration(y, p, bins=10):
    """Shots split into deciles of predicted xG: mean xG vs goal rate per bin."""
    order = np.argsort(p)
    return [{"mean_xg": round(float(p[i].mean()), 4), "goal_rate": round(float(y[i].mean()), 4),
             "shots": int(len(i))} for i in np.array_split(order, bins)]


def cross_validate(tr, feats, seed=SEED, params=PARAMS):
    """Grouped-by-game CV: (number of rounds, out-of-fold predictions)."""
    params = {**params, "seed": seed}
    folds = game_folds(tr["game_id"].to_numpy(), seed=seed)
    cv = xgb.cv(params, dmatrix(tr, feats), num_boost_round=MAX_ROUNDS, folds=folds,
                early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    rounds = len(cv)
    oof = np.zeros(len(tr))
    for tri, tei in folds:
        b = xgb.train(params, dmatrix(tr.iloc[tri], feats), num_boost_round=rounds)
        oof[tei] = b.predict(dmatrix(tr.iloc[tei], feats))
    return rounds, oof


def main():
    import tune  # season folds / weights / in-season calibration (tune imports train)
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="model name, e.g. v4")
    ap.add_argument("--promote", action="store_true", help="make score.py use this model")
    a = ap.parse_args()
    meta_path = MODELS / f"xg_{a.version}.meta.json"
    if meta_path.exists() or any(MODELS.glob(f"xg_{a.version}_*.json")):
        raise SystemExit(f"model {a.version} exists: pick a new --version")

    feats = F.feature_names()
    df = F.build()
    tr = df[F.train_mask(df)].reset_index(drop=True)
    y = tr["is_goal"].to_numpy()
    print(f"training shots: {len(tr)}  goals: {int(y.sum())}  games: {tr['game_id'].nunique()}  "
          f"half-life: {HALF_LIFE}")

    # number of rounds: tuned (ROUNDS) or early stopping on grouped CV
    rounds = ROUNDS or cross_validate(tr, feats)[0]
    # out-of-sample evaluation the way the models are used: single-season folds, each
    # scored by a model weighted towards its season
    oof = tune.oof_predictions(tr, feats, PARAMS, rounds, HALF_LIFE, tune.season_folds(tr))
    base = log_loss(y, np.full(len(y), y.mean()))
    metrics = {"rounds": rounds, "cv_logloss": round(log_loss(y, oof), 5),
               "baseline_logloss": round(base, 5), "cv_auc": round(auc(y, oof), 4),
               "goals": int(y.sum()), "xg_sum_oof": round(float(oof.sum()), 1),
               "in_season": tune.summarize(tr, oof), "calibration_oof": calibration(y, oof)}
    print(f"rounds {rounds}  CV log loss {metrics['cv_logloss']} (constant rate: {metrics['baseline_logloss']})"
          f"  AUC {metrics['cv_auc']}  in-season calibration {metrics['in_season']}")
    if OFFSET_K is not None:
        # out of fold WITH the shrunk season offset: each fold's offset comes from the
        # other folds of its season, as for a game not yet played
        import score
        folds = tune.season_folds(tr)
        adj = oof.copy()
        for yr, te in folds:
            same = np.concatenate([t for yy, t in folds if yy == yr and t is not te])
            b = score.shrink(score.fit_offset(oof[same], y[same].sum()), oof[same].sum(), OFFSET_K)
            adj[te] = score.sigmoid(score.logit(oof[te]) + b)
        metrics["in_season_with_offsets"] = tune.summarize(tr, adj)
        print(f"  with season offsets (K={OFFSET_K}): {metrics['in_season_with_offsets']}")

    MODELS.mkdir(exist_ok=True)
    years = sorted(int(v) for v in tr["season_year"].unique())
    season_models = None
    if HALF_LIFE is None:  # equal weights: one model for every season
        b = xgb.train(PARAMS, dmatrix(tr, feats), num_boost_round=rounds)
        saved = [MODELS / f"xg_{a.version}.json"]
        b.save_model(saved[0])
    else:  # one model per scored season year, plus the next season (scores new games
        # until the next retrain adds them to training)
        season_models, saved = {}, []
        for target in years + [years[-1] + 1]:
            w = tune.season_weights(tr["season_year"], target, HALF_LIFE)
            b = xgb.train(PARAMS, xgb.DMatrix(tr[feats], label=y, weight=w, enable_categorical=True),
                          num_boost_round=rounds)
            path = MODELS / f"xg_{a.version}_{target}.json"
            b.save_model(path)
            season_models[str(target)] = path.name
            saved.append(path)
    if OFFSET_K is not None:  # in-season offsets as score.py will apply them (in sample)
        import score
        metrics["season_offsets_in_sample"] = score.season_offsets(tr, b.predict(dmatrix(tr, feats)), OFFSET_K)
    gain = b.get_score(importance_type="gain")  # (per-season models: the next-season one)
    metrics["importance_gain"] = {k: round(v, 3) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}
    print("gain:", metrics["importance_gain"])

    meta = {
        "version": a.version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xgboost_version": xgb.__version__,
        "features": feats,
        "categorical": F.CATEGORICAL,
        "settings": settings(),
        "params": PARAMS,
        "half_life": HALF_LIFE,
        "offset_k": OFFSET_K,
        "train": {"season_types": list(F.TRAIN_SEASON_TYPES),
                  "season_ids": sorted(int(s) for s in tr["season_id"].unique()),
                  "season_years": years, "shots": len(tr), "games": int(tr["game_id"].nunique())},
        "metrics": metrics,
    }
    if season_models:  # season year -> model file; later years use the last, earlier the first
        meta["season_models"] = season_models
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {', '.join(p.name for p in saved)} and {meta_path.name}")
    if a.promote or not (MODELS / "current.txt").exists():
        (MODELS / "current.txt").write_text(a.version + "\n")
        print(f"score.py now uses {a.version}")


if __name__ == "__main__":
    main()
