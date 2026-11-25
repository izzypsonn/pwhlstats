"""
Replay test for the in-season calibration offset's shrinkage (run by hand):

  python xg/replay.py

The offset: xG = sigmoid(logit(p) + b), one b per season. The live estimate before
a game day uses only that season's earlier games:
  b_hat = the b that makes the earlier games' xG add up to their goals
  b     = w * b_hat, w = G / (G + K), G = the earlier games' (unadjusted) xG
K = 0: no shrinkage; K = inf: no offset.

Leave one season out: for each season, the shot model is trained on the other two
(regular + playoffs, train.PARAMS, equal weights, train.ROUNDS or 150 rounds). The
season is then replayed by game day in date order; each day's shots are scored with
the b known that morning, for every K.

Metrics per K (lower is better unless noted):
  day_dev    Poisson deviance of each game day's goals vs its xG, summed: the
             season-level part, which shot log loss barely sees
  logloss    shot-level log loss (should not get worse)
  rest_N     xG / goals of the rest of the season with b frozen after N games
             (closer to 1 is better): where early overfitting would show
  b_jumps    sd of the day-to-day change in b (how jumpy the offset is)

Results 2026-11-25 (all three seasons; day_dev summed, change vs no offset):
  K=0 -4.3, 25 -6.7, 50 -7.1, 100 -7.0, 160 -6.6, 250 -5.9, 500 -4.6, 1000 -3.1, inf 0.
  Every offset forecasts later games better than none (season-level differences
  persist within a season); no shrinkage overfits a little (b jumps 0.041/day vs
  0.009 at K=100). By season: 2025 prefers K=25, 2024 and 2026 100-250. Shot log
  loss unchanged (0.2474-0.2476). Chosen K=100 (train.OFFSET_K).
"""
import numpy as np
import pandas as pd
import xgboost as xgb

import features as F
import train as T
from score import fit_offset, logit, shrink, sigmoid

KS = [0, 25, 50, 100, 160, 250, 500, 1000, np.inf]
CHECKPOINTS = (10, 30, 60)


def poisson_dev(goals, mu):
    goals, mu = np.asarray(goals, float), np.asarray(mu, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(goals > 0, goals * np.log(goals / mu), 0.0)
    return float(2 * np.sum(t - (goals - mu)))


def replay_season(tr, year, feats, rounds):
    train_rows, test = tr[tr["season_year"] != year], tr[tr["season_year"] == year].copy()
    b = xgb.train(T.PARAMS, T.dmatrix(train_rows, feats), num_boost_round=rounds)
    test["p"] = b.predict(T.dmatrix(test, feats))
    test = test.sort_values(["date", "game_id", "event_idx"])
    days = sorted(test["date"].unique())
    game_no = test.groupby("game_id")["date"].first().rank(method="first")  # games in date order
    test["game_no"] = test["game_id"].map(game_no)

    rows = []
    for K in KS:
        xg = np.empty(len(test))
        bs, day_goals, day_xg = [], [], []
        pos = 0
        for d in days:
            before = test.iloc[:pos]
            day = test["date"].iloc[pos:] == d
            n = int(day.to_numpy().cumprod().sum())  # rows of this day (sorted by date)
            b_hat = fit_offset(before["p"].to_numpy(), before["is_goal"].sum())
            bk = shrink(b_hat, before["p"].sum(), K)
            q = sigmoid(logit(test["p"].iloc[pos:pos + n].to_numpy()) + bk)
            xg[pos:pos + n] = q
            bs.append(bk)
            day_goals.append(test["is_goal"].iloc[pos:pos + n].sum())
            day_xg.append(q.sum())
            pos += n
        y = test["is_goal"].to_numpy()
        r = {"season": year, "K": K, "day_dev": round(poisson_dev(day_goals, day_xg), 2),
             "logloss": round(T.log_loss(y, xg), 5), "b_jumps": round(float(np.std(np.diff(bs))), 4),
             "b_final": round(bs[-1], 3)}
        for c in CHECKPOINTS:  # b frozen after c games, applied to the rest of the season
            seen = test["game_no"] <= c
            if seen.all():
                continue
            bh = fit_offset(test.loc[seen, "p"].to_numpy(), test.loc[seen, "is_goal"].sum())
            bk = shrink(bh, test.loc[seen, "p"].sum(), K)
            rest = test[~seen]
            r[f"rest_{c}"] = round(float(sigmoid(logit(rest["p"]) + bk).sum() / rest["is_goal"].sum()), 4)
        rows.append(r)
    full_b = fit_offset(test["p"].to_numpy(), test["is_goal"].sum())
    return rows, {"season": year, "games": int(test["game_id"].nunique()), "days": len(days),
                  "goals": int(test["is_goal"].sum()), "xg_raw": round(float(test["p"].sum()), 1),
                  "b_full_season": round(full_b, 3)}


def main():
    feats = F.feature_names()
    df = F.build()
    tr = df[F.train_mask(df)].reset_index(drop=True)
    dates = pd.read_parquet(F.CLEAN / "games.parquet")[["game_id", "date_played"]]
    tr = tr.merge(dates, on="game_id", how="left").rename(columns={"date_played": "date"})
    rounds = T.ROUNDS or 150
    rows, info = [], []
    for year in sorted(tr["season_year"].unique()):
        r, i = replay_season(tr, int(year), feats, rounds)
        rows += r
        info.append(i)
    print("replayed seasons (model trained on the other two):")
    print(pd.DataFrame(info).to_string(index=False))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nper season:")
    print(res.to_string(index=False))
    tot = res.groupby("K").agg(day_dev=("day_dev", "sum"), logloss=("logloss", "mean"),
                               b_jumps=("b_jumps", "mean"),
                               **{f"rest_{c}_abs_err": (f"rest_{c}", lambda s: float(np.nanmean(np.abs(s - 1))))
                                  for c in CHECKPOINTS})
    tot["day_dev_vs_no_offset"] = tot["day_dev"] - tot.loc[np.inf, "day_dev"]
    print("\nall three seasons (day_dev summed; others averaged; rest_N_abs_err = mean |xG/goals - 1|):")
    print(tot.round(4).to_string())


if __name__ == "__main__":
    main()
