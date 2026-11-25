"""
Hyperparameter + half-life search under a maximum in-season calibration constraint,
following https://hockeystats.com/methodology/expected-goals (run by hand):

  python xg/tune.py                    # search, writes models/tuning.csv
  python xg/tune.py --combos 20        # smaller search

Method (as in the article):
  - calibration of a season = sum(xG) / goals, out of sample; maximum in-season
    calibration error = the largest |calibration - 100%| over the seasons;
  - selection: highest AUC with maximum in-season calibration error <= MAX_CAL_ERR;
  - half-life season weighting: a training shot from season s gets weight
    0.5 ** (|s - target| / half_life) in the model that scores season `target`
    (half_life None = equal weights), so every scored season gets its own model;
  - single-season folds: each fold is the shots of a random set of games from ONE
    season (K_PER_SEASON per season); a fold is scored by a model trained on all
    other folds, weighted towards the fold's season.
  Seasons are season years (features.season_years: preseason + regular + playoffs).
  Not done (too costly for a CPU run): the article's nested outer loop. Instead the
  best candidates are re-scored on fresh fold splits (--recheck-seeds) to see
  whether they still meet the constraint; tuning.csv keeps every candidate.

Noise floor: a season with G goals has a calibration sd of about 1/sqrt(G) for a
PERFECT model (Poisson). PWHL seasons have 359-561 goals (sd 4.2-5.3%), so a 3% cap
is tighter than chance alone; the article's NHL seasons have ~8,000 (sd ~1.1%).
"""
import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import features as F
import train as T

MAX_CAL_ERR = 0.03
K_PER_SEASON = 5
HALF_LIVES = [0.25, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, None]
FOLD_SEED = 1


def season_folds(tr, k=K_PER_SEASON, seed=FOLD_SEED):
    """[(season_year, row indices)], k game-grouped folds within each season."""
    folds = []
    for yr, g in tr.groupby("season_year"):
        games = np.random.default_rng(seed * 10_000 + int(yr)).permutation(g["game_id"].unique())
        fold_of = {gm: i % k for i, gm in enumerate(games)}
        f = g["game_id"].map(fold_of).to_numpy()
        folds += [(int(yr), g.index[f == i].to_numpy()) for i in range(k)]
    return folds


def season_weights(years, target, half_life):
    if half_life is None:
        return np.ones(len(years))
    w = 0.5 ** (np.abs(np.asarray(years) - target) / half_life)
    # scaled so the nearest season has weight 1: min_child_weight counts weighted shots,
    # so a model for a season with no data yet (all weights <= 0.5 ** (1 / half_life))
    # would otherwise hardly split. No change when the target season is in the data.
    return w / w.max()


def oof_predictions(tr, feats, params, rounds, half_life, folds):
    y, yrs = tr["is_goal"].to_numpy(), tr["season_year"].to_numpy()
    oof = np.full(len(tr), np.nan)
    for yr, te in folds:
        keep = np.ones(len(tr), bool)
        keep[te] = False
        tri = np.flatnonzero(keep)
        d = xgb.DMatrix(tr.iloc[tri][feats], label=y[tri], weight=season_weights(yrs[tri], yr, half_life),
                        enable_categorical=True)
        b = xgb.train(params, d, num_boost_round=rounds)
        oof[te] = b.predict(xgb.DMatrix(tr.iloc[te][feats], enable_categorical=True))
    return oof


def summarize(tr, oof):
    y = tr["is_goal"].to_numpy()
    cal = pd.Series(oof).groupby(tr["season_year"].to_numpy()).sum() / pd.Series(y).groupby(tr["season_year"].to_numpy()).sum()
    out = {f"cal_{yr}": round(float(c), 4) for yr, c in cal.items()}
    out.update(max_cal_err=round(float((cal - 1).abs().max()), 4), auc=round(T.auc(y, oof), 4),
               logloss=round(T.log_loss(y, oof), 5), cal_all=round(float(oof.sum() / y.sum()), 4))
    return out


def sample(rng):
    params = {**T.PARAMS,
              "max_depth": int(rng.integers(2, 7)),
              "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
              "min_child_weight": float(np.exp(rng.uniform(0, np.log(50)))),
              "subsample": float(rng.uniform(0.6, 1.0)),
              "colsample_bytree": float(rng.uniform(0.5, 1.0)),
              "reg_lambda": float(np.exp(rng.uniform(np.log(0.5), np.log(20))))}
    return params, int(rng.integers(100, 800)), HALF_LIVES[int(rng.integers(len(HALF_LIVES)))]


def pick(res):
    ok = res[res["max_cal_err"] <= MAX_CAL_ERR]
    return (ok.sort_values("auc", ascending=False) if len(ok) else res.sort_values("max_cal_err")), len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", type=int, default=60, help="random hyperparameter sets (plus the fixed sweep)")
    ap.add_argument("--recheck-seeds", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--out", default=str(F.ROOT / "models" / "tuning.csv"))
    a = ap.parse_args()
    t0 = time.time()
    feats = F.feature_names()
    df = F.build()
    tr = df[F.train_mask(df)].reset_index(drop=True)
    g = tr.groupby("season_year")["is_goal"].agg(["size", "sum"])
    print("training shots / goals by season:", {int(k): (int(r["size"]), int(r["sum"])) for k, r in g.iterrows()},
          "| calibration noise sd for a perfect model:", {int(k): f"{100 / np.sqrt(r['sum']):.1f}%" for k, r in g.iterrows()})
    folds = season_folds(tr)

    # fixed sweep: current (v3) parameters at every half-life; v3 used 150 rounds
    cands = [({**T.PARAMS}, 150, hl, "v3 params") for hl in HALF_LIVES]
    rng = np.random.default_rng(20261125)
    cands += [(*sample(rng), "random") for _ in range(a.combos)]
    rows = []
    for i, (params, rounds, hl, kind) in enumerate(cands, 1):
        r = summarize(tr, oof_predictions(tr, feats, params, rounds, hl, folds))
        rows.append({"kind": kind, "half_life": hl, "rounds": rounds,
                     **{k: params[k] for k in ("max_depth", "learning_rate", "min_child_weight", "subsample",
                                               "colsample_bytree", "reg_lambda")}, **r})
        print(f"[{i}/{len(cands)}] {kind:<9} hl={str(hl):<5} depth={params['max_depth']} lr={params['learning_rate']:.3f} "
              f"rounds={rounds:<4} max_err={r['max_cal_err']:.3f} auc={r['auc']:.4f} "
              f"cal={[r[c] for c in r if c.startswith('cal_2')]}  ({time.time() - t0:.0f} s)", flush=True)
    res = pd.DataFrame(rows)
    ranked, n_ok = pick(res)
    print(f"\n{n_ok} of {len(res)} candidates meet max in-season calibration error <= {MAX_CAL_ERR:.0%}")

    # re-score the best candidates on fresh fold splits
    top = ranked.head(5).copy()
    for s in a.recheck_seeds:
        f2 = season_folds(tr, seed=s)
        errs, aucs = [], []
        for _, r in top.iterrows():
            params = {**T.PARAMS, **{k: r[k] for k in ("max_depth", "learning_rate", "min_child_weight", "subsample",
                                                       "colsample_bytree", "reg_lambda")}}
            params["max_depth"] = int(params["max_depth"])
            hl = None if pd.isna(r["half_life"]) else float(r["half_life"])
            out = summarize(tr, oof_predictions(tr, feats, params, int(r["rounds"]), hl, f2))
            errs.append(out["max_cal_err"])
            aucs.append(out["auc"])
        top[f"max_err_seed{s}"], top[f"auc_seed{s}"] = errs, aucs
    res = res.merge(top[[c for c in top.columns if "_seed" in c]], left_index=True, right_index=True, how="left")
    res.to_csv(a.out, index=False)
    cols = ["kind", "half_life", "rounds", "max_depth", "learning_rate", "min_child_weight"] + \
           [c for c in res.columns if c.startswith("cal_2")] + ["max_cal_err", "auc", "logloss"] + \
           [c for c in top.columns if "_seed" in c]
    print("\nbest candidates (re-checked on other fold splits):\n" + top[cols].to_string(index=False))
    print("\nfixed sweep (v3 params, 150 rounds) by half-life:\n" +
          res[res["kind"] == "v3 params"][["half_life"] + [c for c in res.columns if c.startswith("cal_2")] +
                                          ["max_cal_err", "auc", "logloss"]].to_string(index=False))
    print(f"\nwrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
