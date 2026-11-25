"""
Feature-setting experiments for the xG model (run by hand: python xg/experiments.py).

Every variant is scored with the same grouped-by-game CV as train.py, on the same
folds, over several seeds, so differences between variants are paired. Reported:
out-of-fold log loss (lower is better) and its change vs the baseline (current
settings in features.py), with the spread of that change across seeds.

  --only rebound|royal_road|ablation   run one group

Results on 2026-11-25 (17,940 training shots; log loss deltas in 1e-4, mean of 3
seeds; seed-to-seed spread is about +-2):
  rebound windows vs (3,): none +1.2, 1 s +2.6, 2 s +1.5, 4 s 0.0, 5 s +1.4, 6 s +2.0,
    (2, 5) +0.3, (1, 3) +0.8, (1, 3, 6) -0.1. 3-4 s is best for one window; extra
    windows add nothing (prev_event / prev_same_team / secs_since_prev already
    let the trees find rebounds). Chosen: (3,).
  royal road cutoff vs 0 deg: left out +5.9 (every seed worse), 10 -0.6, 20 -1.4
    (every seed better), 30 +1.4, 45 +2.9, 60 +5.1. Chosen: 20 deg.
  ablation (one feature left out): off_hand -0.6, glove_side -0.8, period +
    time_remaining -0.4, prev_to_shot_speed_ft_s -0.7, score_diff -0.6,
    prev_same_team -0.6: none measurable on its own at this sample size.
  v3 round (same method; script was a one-off): base = v2 features + behind_net.
    - behind_net -1.0 (seeds +0.9 -0.4 -3.4: noise; kept, 105 shots).
    + shooter_shift_s and opp_avg_shift_s with NaN when missing -5.2 (every seed),
      but only through missingness: shots without a shooter shift get mean xG 6.35%
      vs 7.98% for found ones (goal rates 5.8% vs 8.1%: the shift data is more complete
      around goals). With missing values imputed at random: +1.7 (worse, every seed);
      shooter only +1.3, opponents only +0.1. Left out of the model (INFO_COLS).
    training seasons (fixed 150 rounds, scored on regular + playoff shots): adding
      preseasons 4 and 7 +0.1 (no effect); dropping season 1 +7.1 (worse, every seed;
      +7.4 scored on seasons 3+ only). Kept: regular + playoffs, all seasons. The
      inaugural preseason (season 2) has no stored games (events but no shifts).
"""
import argparse
import time

import numpy as np

import features as F
import train as T

SEEDS = (1, 2, 3)
REBOUND_VARIANTS = [(), (1,), (2,), (3,), (4,), (5,), (6,), (2, 5), (1, 3), (1, 3, 6)]
ROYAL_ROAD_VARIANTS = [None, 0, 10, 20, 30, 45, 60, 90]  # None = feature left out


def evaluate(raw, drop=(), **settings):
    old = {k: getattr(F, k) for k in settings}
    for k, v in settings.items():
        setattr(F, k, v)
    try:
        df = F.add_features(raw.copy())
        feats = [f for f in F.feature_names() if f not in drop]
    finally:
        for k, v in old.items():
            setattr(F, k, v)
    tr = df[F.train_mask(df)].reset_index(drop=True)
    y = tr["is_goal"].to_numpy()
    out = []
    for s in SEEDS:
        rounds, oof = T.cross_validate(tr, feats, seed=s)
        out.append((T.log_loss(y, oof), T.auc(y, oof), rounds))
    return np.array(out)


def report(name, res, base):
    d = res[:, 0] - base[:, 0]
    print(f"  {name:<28} logloss {res[:, 0].mean():.5f}  delta {d.mean() * 1e4:+6.1f}e-4 "
          f"(seeds {' '.join(f'{x * 1e4:+.1f}' for x in d)})  AUC {res[:, 1].mean():.4f}  "
          f"rounds {int(res[:, 2].mean())}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["rebound", "royal_road", "ablation"])
    a = ap.parse_args()
    t0 = time.time()
    raw = F.load_raw()
    base = evaluate(raw)
    print(f"baseline (REBOUND_S={F.REBOUND_S}, ROYAL_ROAD_DEG={F.ROYAL_ROAD_DEG}): "
          f"logloss {base[:, 0].mean():.5f}  AUC {base[:, 1].mean():.4f}")

    if a.only in (None, "rebound"):
        print("rebound windows (seconds):")
        for v in REBOUND_VARIANTS:
            report(str(v) if v else "none", evaluate(raw, REBOUND_S=v), base)
    if a.only in (None, "royal_road"):
        print("royal road angle cutoff (degrees):")
        for v in ROYAL_ROAD_VARIANTS:
            res = evaluate(raw, drop=("royal_road",)) if v is None else evaluate(raw, ROYAL_ROAD_DEG=v)
            report("left out" if v is None else f">= {v}", res, base)
    if a.only in (None, "ablation"):
        print("ablation (feature group left out):")
        groups = {"off_hand": ["off_hand"], "glove_side": ["glove_side"],
                  "period + time_remaining": ["period", "time_remaining_s"],
                  "prev_to_shot_speed_ft_s": ["prev_to_shot_speed_ft_s"], "score_diff": ["score_diff"],
                  "prev_same_team": ["prev_same_team"]}
        for name, cols in groups.items():
            report(f"- {name}", evaluate(raw, drop=cols), base)
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
