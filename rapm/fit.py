"""
xG-based regularized adjusted plus-minus (RAPM), one fit per season year (regular
season + playoffs; preseason left out). Two models (MODELS):

  5v5   even strength: offence (off) and defence (def) per skater
  pp    special teams: power-play offence (ppo) and shorthanded defence (shd)

Runs in CI after xg/score.py (files are only rewritten when the results change);
by hand:

  python rapm/fit.py                    # both models, all season years
  python rapm/fit.py --model pp         # one model (always all season years: each
                                        # season uses the seasons before it)
  python rapm/fit.py --prior vanilla    # another method than the model's own (MODELS
                                        # "method"), written to rapm_<model>_<method>_xg

Data: stints from rapm/stints.py, both goalies in, duration > 0. Games with
games.exclude_shifts = 1 are left out there; nothing else is filtered (see
stints.py on skater lists that don't have 5 names).
  5v5: state 5v5. Each stint gives two rows, one per team attacking.
  pp:  state 5v4 / 4v5 (5v3 and 4v3 are left out: a different game, < 0.6 h per
       season). One row per stint, the power-play team attacking; the
       shorthanded team's own chances are not modelled.

Each row:
  y = xG for the attacking team per 60 minutes in the stint
  w = stint length in hours
  X = +1 for each attacking skater's offence column, +1 for each defending
      skater's defence column, plus the model's context columns, all from the
      attacking team's side:
        home                    attacking team is at home
        lead_m2 / lead_m1 / lead_p1 / lead_p2
                                attacking team's lead at the stint start:
                                <= -2 / -1 / +1 / >= +2 (tied = baseline)
        start_oz / start_dz / start_nz
                                stint began with a faceoff in the attacking team's
                                offensive / defensive / neutral zone (on the fly =
                                baseline)
        b2b_att / b2b_def       attacking / defending team played the day before (5v5 only)
        pen_exp_att / pen_exp_def
                                stint began as the attacking / defending team's
                                penalty expired (5v5 only)
      The pp model leaves out the back-to-back and penalty-expiry terms: they have
      almost no power-play time.
Weighted ridge throughout: minimise sum w (y - Xb)^2 + lambda * |b - b0|^2 over the
player columns (prior mean b0, 0 unless stated), intercepts and context terms not
penalised. Each model has a method (MODELS "method"): both use "halflife"
(2026-11-25: for 5v5 it beat the Bacon prior out of sample; for pp no method was
distinguishable from vanilla in a paired per-game CV comparison, and half-life was
chosen so both models' ratings mean the same kind of thing). "bacon" (steps 3-4)
and "vanilla" (step 2 only) remain options. Steps, per model:
  1. pooled context: all season years in one fit, with an intercept per season
     year, the context terms shared by all seasons, and one offence and one
     defence column per player-season. Gives the context effects (meta
     "context_xg60"), estimated from every season at once (some terms, e.g.
     back-to-backs and penalty expiry, have only a few hours of data per season).
     Steps 2-4 take them out of y and fit intercept + players only.
  2. vanilla RAPM per season year (b0 = 0, one lambda).
  3. trend (Patrick Bacon / HockeyStats): weighted linear fit of a player's vanilla
     rating in season t on their vanilla rating in season t - 1, separately for
     offence and defence and for forwards and defenders (TREND_BY_POSITION; a
     player's position = their most common roster position over all games, C / LW /
     RW / F = forward, LD / RD / D = defender; no position -> the all-skater trend),
     pooled over every pair of consecutive season years. Weight: harmonic mean of
     the two seasons' minutes in that role (offence: attacking-side time, defence:
     defending-side time; 5v5: both are 5v5 TOI, pp: PP / PK time). Players with no
     time in the role in either season are left out. With only a few seasons the
     pairs include the season being fitted, so its data enters the prior through
     these numbers.
  4. final RAPM per season year: returning players are shrunk towards their
     predicted rating, prior = intercept + slope * vanilla rating in season t - 1;
     players with no time in that role in the previous season (rookies, and
     everyone in the first season) towards 0, i.e. average (PWHL rookies arrive
     older than NHL ones and can be dominant from the start, so no below-average
     rookie prior). One lambda for all players (a separate rookie lambda was tried
     2026-11-25: it ran to the edge of the grid in 2025 for a negligible CV gain).
  4b. halflife (instead of 3-4): season T's ratings from season T and every earlier
     season, rows of season s weighted by 0.5 ** ((T - s) / half_life), one column
     per player across them and an intercept per season (fit_halflife). One
     half-life for all seasons, the one in HALF_LIVES with the lowest CV MSE summed
     over the seasons that have earlier data (choose_half_life): per-season
     choices were noise on a flat curve (2026-11-25: 3.0 and 1.5). A rating is then
     a multi-season estimate weighted towards season T.
     The half-life is chosen from completed season years only (complete_years), so
     a season in progress does not move the earlier seasons' ratings; season years
     with fewer than MIN_GAMES games are not fitted.
lambda is picked from LAMBDAS by N_FOLDS-fold cross-validation grouped by game
(weighted squared error on held-out games; a player not in the training folds gets
their prior). Folds are seeded per fit, so the vanilla and final fits of a season
use the same folds (meta: cv_mse_vanilla vs cv_mse). Everything is solved from the
per-fold sums X'WX, X'Wy, y'Wy, so the design matrix is never held in memory.

Output, per model: data/rapm/rapm_<model>_xg.parquet, one row per player and season year:
  5v5:  off    xGF/60 the player adds on offence (above an average skater)
        def    xGA/60 the player saves on defence (sign flipped: positive = good)
        net    off + def
        toi_5v5_s  5v5 seconds in the stints used (this season only)
        toi_weighted_s  5v5 seconds weighted the way the fit weights seasons:
               halflife: sum over this and earlier seasons of
               0.5 ** (seasons back / half_life) * that season's seconds (the time
               the rating rests on; use it for TOI cutoffs); bacon / vanilla: this
               season only (their fits use only this season's data)
  pp:   ppo    PP xGF/60 the player adds on the power play
        shd    PP xGA/60 the player saves on the penalty kill (positive = good)
        pp_toi_s / pk_toi_s  5v4 seconds on the power play / penalty kill
        pp_toi_weighted_s / pk_toi_weighted_s  the same, weighted as toi_weighted_s
  qualified (5v5) / pp_qualified, pk_qualified (pp)  meets both leaderboard cutoffs
        in MIN_TOI: this season's TOI and weighted TOI. Only players who played in
        the season have a row, so history alone never qualifies anyone.
        (no net: the two are measured in different situations)
  both: vanilla_<x> (step 2) and prior_<x> (step 4 prior mean; NaN: no prior, 0)
and data/rapm/rapm_<model>_xg.meta.json with the pooled fit, the trend and lambda,
CV curve and intercept per season.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import stints as S

OUT_DIR = S.ROOT / "data" / "rapm"
LAMBDAS = np.round(np.logspace(-1, 3, 17), 4)  # in stint-hours
N_FOLDS = 10
HALF_LIVES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf)  # in seasons; 0 = this season only
TREND_BY_POSITION = True
# Leaderboard cutoffs (seconds): a player is "qualified" in a role with at least
# "season" seconds this season AND "weighted" seconds of weighted TOI (the time the
# rating rests on; see toi_weighted_s). Both are needed: the season cutoff keeps out
# players with a long history but little play this season, the weighted one keeps
# out rookies with too small a sample. Ratings are computed for everyone; this only
# flags who to show.
MIN_TOI = {"5v5": {"season": 100 * 60, "weighted": 200 * 60},
           "pp": {"season": 15 * 60, "weighted": 30 * 60}}
MIN_GAMES = N_FOLDS  # season years with fewer games (start of a season) are not fitted yet
SEED = 20261125
CHUNK = 5000
COVS = ["home", "lead_m2", "lead_m1", "lead_p1", "lead_p2", "start_oz", "start_dz", "start_nz",
        "b2b_att", "b2b_def", "pen_exp_att", "pen_exp_def"]
PP_COVS = ["home", "lead_m2", "lead_m1", "lead_p1", "lead_p2", "start_oz", "start_dz", "start_nz"]
FLIP_ZONE = {"oz": "dz", "dz": "oz", "nz": "nz", "otf": "otf"}


def both_goalies(st):
    return st[(st["home_goalie_in"] == 1) & (st["visitor_goalie_in"] == 1) & (st["duration_s"] > 0)]


def five_on_five(st):
    return both_goalies(st[st["state"] == "5v5"])


def power_play(st):
    return both_goalies(st[st["state"].isin(["5v4", "4v5"])])


def rows(st, attackers=None):
    """One row per stint and attacking team: game_id, season_year, off / def player
    lists, the context columns, y and w. attackers: None = both teams attack in
    every stint (5v5); 'pp' = only the team with more skaters (5v4: home, 4v5:
    visitor)."""
    out = []
    for att, opp in (("home", "visitor"), ("visitor", "home")):
        s = st
        if attackers == "pp":
            s = st[st["state"] == ("5v4" if att == "home" else "4v5")]
        lead = s["score_home"] * (1 if att == "home" else -1)
        start = s["start_home"] if att == "home" else s["start_home"].map(FLIP_ZONE)
        r = pd.DataFrame({"game_id": s["game_id"], "season_year": s["season_year"],
                          "off": s[f"{att}_players"], "def": s[f"{opp}_players"],
                          "home": int(att == "home"),
                          "lead_m2": (lead <= -2).astype(int), "lead_m1": (lead == -1).astype(int),
                          "lead_p1": (lead == 1).astype(int), "lead_p2": (lead >= 2).astype(int),
                          "start_oz": (start == "oz").astype(int), "start_dz": (start == "dz").astype(int),
                          "start_nz": (start == "nz").astype(int),
                          "b2b_att": s[f"{att}_b2b"], "b2b_def": s[f"{opp}_b2b"],
                          "pen_exp_att": s[f"pen_exp_{att}"], "pen_exp_def": s[f"pen_exp_{opp}"],
                          "team_off": s[f"{att}_team_id"],
                          "xg": s[f"{att}_xg"], "dur": s["duration_s"]})
        out.append(r)
    r = pd.concat(out, ignore_index=True)
    r["w"] = r["dur"] / 3600
    r["y"] = r["xg"] / r["w"]
    return r


MODELS = {
    "5v5": {"stints": five_on_five, "attackers": None, "covs": COVS, "names": ("off", "def"),
            "states": "5v5, both goalies in", "method": "halflife"},
    "pp": {"stints": power_play, "attackers": "pp", "covs": PP_COVS, "names": ("ppo", "shd"),
           "states": "5v4 / 4v5, both goalies in; power-play team attacking", "method": "halflife"},
}


def sums(r, fixed, col):
    """X'WX, X'Wy, y'Wy, sum w for rows r. Columns: the unpenalised columns `fixed`
    (names of columns of r), then an offence and a defence column per player key
    (col: (season_year, player_id) -> index)."""
    nf, n_p = len(fixed), len(col)
    n_cols = nf + 2 * n_p
    G = np.zeros((n_cols, n_cols))
    b = np.zeros(n_cols)
    for i in range(0, len(r), CHUNK):
        c = r.iloc[i:i + CHUNK]
        X = np.zeros((len(c), n_cols))
        X[:, :nf] = c[fixed].to_numpy(float)
        for j, (yr, o, d) in enumerate(zip(c["season_year"], c["off"], c["def"])):
            X[j, [nf + col[(yr, p)] for p in o]] = 1
            X[j, [nf + n_p + col[(yr, p)] for p in d]] = 1
        w, y = c["w"].to_numpy(), c["y"].to_numpy()
        G += X.T @ (X * w[:, None])
        b += X.T @ (w * y)
    return G, b, float((r["w"] * r["y"] ** 2).sum()), float(r["w"].sum())


def solve(G, b, lam, n_fixed, b0=None):
    """Ridge towards b0 (default 0); lam: one value or one per column; the first
    n_fixed columns are unpenalised."""
    P = np.broadcast_to(np.asarray(lam, float), len(b)).copy()
    P[:n_fixed] = 0  # intercepts and context terms unpenalised
    rhs = b if b0 is None else b + P * b0
    return np.linalg.solve(G + np.diag(P), rhs)


def cv_ridge(r, fixed, prior=None):
    """Ridge on rows r with unpenalised columns `fixed` and player-season columns,
    lambda by grouped CV. prior: {(season_year, player_id): (off, def coefficient)}
    prior means; players not in it are shrunk towards 0. Returns (beta, players,
    lambda, cv curve)."""
    players = sorted(set((yr, p) for yr, ps in zip(r["season_year"], r["off"]) for p in ps)
                     | set((yr, p) for yr, ps in zip(r["season_year"], r["def"]) for p in ps))
    col = {k: i for i, k in enumerate(players)}
    nf, n = len(fixed), len(players)
    b0 = np.zeros(nf + 2 * n)
    if prior:
        for k, i in col.items():
            if k in prior:
                b0[nf + i], b0[nf + n + i] = prior[k]
    games = np.sort(r["game_id"].unique())
    fold_of = dict(zip(np.random.default_rng(SEED).permutation(games), np.arange(len(games)) % N_FOLDS))
    fold = r["game_id"].map(fold_of).to_numpy()
    folds = [sums(r[fold == k], fixed, col) for k in range(N_FOLDS)]
    G_all = sum(f[0] for f in folds)
    b_all = sum(f[1] for f in folds)
    cv = {}
    for lam in LAMBDAS:
        sse = sw = 0.0
        for G_k, b_k, yy_k, w_k in folds:
            beta = solve(G_all - G_k, b_all - b_k, lam, nf, b0)
            sse += yy_k - 2 * beta @ b_k + beta @ G_k @ beta
            sw += w_k
        cv[float(lam)] = sse / sw
    lam = min(cv, key=cv.get)
    if lam in (LAMBDAS[0], LAMBDAS[-1]):
        print(f"  warning: lambda {lam} is at the edge of the grid")
    return solve(G_all, b_all, lam, nf, b0), players, lam, cv


def fit_pooled(r, covs):
    """Step 1: season intercepts + shared context + player-season columns."""
    years = sorted(r["season_year"].unique())
    yr_cols = [f"yr_{y}" for y in years]
    r = r.copy()
    for y, c in zip(years, yr_cols):
        r[c] = (r["season_year"] == y).astype(int)
    beta, _, lam, cv = cv_ridge(r, yr_cols + covs)
    nf = len(yr_cols)
    return {"lambda": lam, "cv_mse": cv,
            "intercept_xg60": {str(int(y)): round(float(v), 4) for y, v in zip(years, beta[:nf])},
            "context_xg60": {k: round(float(v), 4) for k, v in zip(covs, beta[nf:nf + len(covs)])},
            "context_hours": {k: round(float((r[k] * r["w"]).sum()), 2) for k in covs}}


def fit_season(r, context, prior=None):
    """One season's rows, pooled context effects taken out of y. prior: DataFrame
    (player_id, prior_off, prior_def) in output signs (def positive = good; NaN =
    no prior, shrunk towards 0); None = vanilla RAPM. Returns (ratings: player_id,
    off, def, net; info)."""
    r = r.copy()
    covs = list(context)
    r["y"] = r["y"] - r[covs].to_numpy(float) @ np.array([context[k] for k in covs])
    r["const"] = 1
    yr = int(r["season_year"].iloc[0])
    pr = None
    if prior is not None:  # def is stored sign-flipped; its coefficient is -def
        pr = {(yr, p): (0.0 if pd.isna(o) else o, 0.0 if pd.isna(d) else -d)
              for p, o, d in zip(prior["player_id"], prior["prior_off"], prior["prior_def"])}
    beta, players, lam, cv = cv_ridge(r, ["const"], pr)
    n = len(players)
    res = pd.DataFrame({"player_id": [p for _, p in players], "off": beta[1:1 + n], "def": -beta[1 + n:]})
    res["net"] = res["off"] + res["def"]
    info = {"lambda": lam, "cv_mse": min(cv.values()), "cv_curve": cv, "intercept_xg60": float(beta[0]),
            "players": n, "games": int(r["game_id"].nunique()), "hours": round(float(r["w"].sum()), 2),
            "xg": round(float(r["xg"].sum()), 2)}
    if prior is not None:
        info["players_with_prior"] = int(prior[["prior_off", "prior_def"]].notna().any(axis=1).sum())
    return res, info


def fit_halflife(by_year, T, context, half_lives=HALF_LIVES):
    """Alternative to the Bacon prior: season T's ratings from season T and every
    earlier season, rows of season s weighted by 0.5 ** ((T - s) / half_life), one
    column per player across them, an unpenalised intercept per season and the
    pooled context taken out of y. half_life and lambda are picked together by
    grouped CV on season T's games (same folds as cv_ridge on season T alone);
    earlier seasons are always in training. half_life 0 = vanilla RAPM. Returns
    (ratings of season T's players, info)."""
    covs = list(context)
    years = [y for y in sorted(by_year) if y <= T]
    fixed = [f"yr_{y}" for y in years]
    parts = {}
    for y in years:
        r = by_year[y].copy()
        r["y"] = r["y"] - r[covs].to_numpy(float) @ np.array([context[k] for k in covs])
        r["season_year"] = T  # one column per player across seasons
        for yy in years:
            r[f"yr_{yy}"] = int(y == yy)
        parts[y] = r
    players = sorted(set((T, p) for r in parts.values() for ps in pd.concat([r["off"], r["def"]]) for p in ps))
    col = {k: i for i, k in enumerate(players)}
    nf, n = len(fixed), len(players)
    rT = parts[T]
    games = np.sort(rT["game_id"].unique())
    fold_of = dict(zip(np.random.default_rng(SEED).permutation(games), np.arange(len(games)) % N_FOLDS))
    fold = rT["game_id"].map(fold_of).to_numpy()
    folds = [sums(rT[fold == k], fixed, col) for k in range(N_FOLDS)]
    G_T, b_T = sum(f[0] for f in folds), sum(f[1] for f in folds)
    old = {y: sums(parts[y], fixed, col)[:2] for y in years if y < T}
    eps = np.zeros(nf + 2 * n)
    eps[:nf] = 1e-9  # keeps unused season intercepts (half_life 0) solvable

    def weights(hl):
        return {y: (0.0 if hl == 0 else 0.5 ** ((T - y) / hl)) for y in old}

    cv = {}
    for hl in (half_lives if old else (0.0,)):
        f = weights(hl)
        G_o = sum((f[y] * old[y][0] for y in old), np.zeros_like(G_T))
        b_o = sum((f[y] * old[y][1] for y in old), np.zeros_like(b_T))
        for lam in LAMBDAS:
            sse = sw = 0.0
            for G_k, b_k, yy_k, w_k in folds:
                beta = solve(G_T - G_k + G_o + np.diag(eps), b_T - b_k + b_o, lam, nf)
                sse += yy_k - 2 * beta @ b_k + beta @ G_k @ beta
                sw += w_k
            cv[(float(hl), float(lam))] = sse / sw
    hl, lam = min(cv, key=cv.get)
    if lam in (LAMBDAS[0], LAMBDAS[-1]):
        print(f"  warning: lambda {lam} is at the edge of the grid")
    f = weights(hl)
    G = G_T + sum((f[y] * old[y][0] for y in old), np.zeros_like(G_T)) + np.diag(eps)
    b = b_T + sum((f[y] * old[y][1] for y in old), np.zeros_like(b_T))
    beta = solve(G, b, lam, nf)
    in_T = set(p for ps in pd.concat([rT["off"], rT["def"]]) for p in ps)
    res = pd.DataFrame({"player_id": [p for _, p in players], "off": beta[nf:nf + n], "def": -beta[nf + n:]})
    res = res[res["player_id"].isin(in_T)].reset_index(drop=True)
    res["net"] = res["off"] + res["def"]
    info = {"half_life": hl, "lambda": lam, "cv_mse": cv[(hl, lam)],
            "cv_curve": {f"half_life={k[0]:g},lambda={k[1]:g}": float(v) for k, v in cv.items()},
            "season_weights": {str(y): round(w, 4) for y, w in f.items()}, "intercept_xg60": float(beta[years.index(T)]),
            "players": int(len(res)), "games": int(len(games)), "hours": round(float(rT["w"].sum()), 2)}
    return res, info


def positions():
    """player_id -> 'F' / 'D' (most common roster position over all games; goalies left out)."""
    r = pd.read_parquet(S.ROOT / "data" / "clean" / "game_rosters")[["player_id", "position"]].dropna()
    r = r[r["position"] != "G"]
    r["pos"] = np.where(r["position"].isin(["LD", "RD", "D"]), "D", "F")
    return r.groupby("player_id")["pos"].agg(lambda x: x.mode().iloc[0])


def _trend(m, c):
    x, y, w = m[f"{c}_0"].to_numpy(), m[f"{c}_1"].to_numpy(), m["w"].to_numpy()
    slope, icpt = np.polyfit(x, y, 1, w=np.sqrt(w))  # polyfit weights the residuals, not their squares
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    r = np.average((x - xm) * (y - ym), weights=w) / np.sqrt(np.average((x - xm) ** 2, weights=w)
                                                           * np.average((y - ym) ** 2, weights=w))
    return {"slope": round(float(slope), 4), "intercept": round(float(icpt), 5), "r": round(float(r), 3),
            "pairs": int(len(m))}


def fit_trend(vanilla, toi, pos=None):
    """Weighted linear fit of vanilla rating in season t on season t - 1, pooled over
    consecutive season pairs, for off and def. vanilla: {year: ratings}; toi:
    (season_year, player_id, toi_off_s, toi_def_s) minutes in each role (weights;
    players with none in a season are left out of that role's fit); pos:
    player_id -> group (e.g. F / D) for one trend per group. Returns
    {"all": {"off": ..., "def": ...}, <group>: ...}."""
    t = toi.set_index(["season_year", "player_id"])
    out = {"all": {}}
    groups = sorted(pos.dropna().unique()) if pos is not None else []
    for g in groups:
        out[g] = {}
    for c in ("off", "def"):
        pairs = []
        h = t[f"toi_{c}_s"] / 3600
        for yr in sorted(vanilla):
            if yr - 1 in vanilla:
                m = vanilla[yr - 1].merge(vanilla[yr], on="player_id", suffixes=("_0", "_1"))
                h0 = h.reindex(pd.MultiIndex.from_arrays([[yr - 1] * len(m), m["player_id"]])).fillna(0).to_numpy()
                h1 = h.reindex(pd.MultiIndex.from_arrays([[yr] * len(m), m["player_id"]])).fillna(0).to_numpy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    m["w"] = np.where((h0 > 0) & (h1 > 0), 2 * h0 * h1 / (h0 + h1), 0.0)
                pairs.append(m)
        if not pairs:
            return {}
        m = pd.concat(pairs, ignore_index=True)
        m = m[m["w"] > 0]
        out["all"][c] = _trend(m, c)
        if pos is not None:
            gm = m["player_id"].map(pos)
            for g in groups:
                out[g][c] = _trend(m[gm == g], c)
    return out


def predicted_prior(prev_vanilla, trend, pos=None, prev_toi=None):
    """Prior means for the next season from the previous season's vanilla ratings,
    with the player's group trend (the all-skater trend if none). prev_toi:
    (player_id, toi_off_s, toi_def_s) of the previous season: no time in a role ->
    no prior (NaN) for it."""
    grp = prev_vanilla["player_id"].map(pos) if pos is not None else pd.Series("all", index=prev_vanilla.index)
    grp = grp.where(grp.isin(list(trend)), "all")
    out = pd.DataFrame({"player_id": prev_vanilla["player_id"]})
    for c in ("off", "def"):
        a = grp.map(lambda g: trend[g][c]["intercept"])
        b = grp.map(lambda g: trend[g][c]["slope"])
        out[f"prior_{c}"] = a + b * prev_vanilla[c]
        if prev_toi is not None:
            t = prev_vanilla["player_id"].map(prev_toi.set_index("player_id")[f"toi_{c}_s"]).fillna(0)
            out.loc[t.to_numpy() <= 0, f"prior_{c}"] = np.nan
    return out


def season_weight(T, y, half_life):
    """Weight of season y's data in season T's fit: 0.5 ** ((T - y) / half_life);
    half_life None (bacon / vanilla) or 0 = season T only, inf = all seasons equal."""
    if y > T:
        return 0.0
    if not half_life:
        return float(y == T)
    return 0.5 ** ((T - y) / half_life)


def weighted_toi(toi, years, half_life):
    """(season_year, player_id, toi_off_w_s, toi_def_w_s): each player's role time
    over season T and earlier fitted seasons, weighted by season_weight, for every
    season T in `years`."""
    toi = toi[toi["season_year"].isin(years)]
    out = []
    for T in years:
        x = toi[toi["season_year"] <= T].copy()
        w = x["season_year"].map(lambda y: season_weight(T, int(y), half_life))
        x["toi_off_w_s"], x["toi_def_w_s"] = x["toi_off_s"] * w, x["toi_def_s"] * w
        g = x.groupby("player_id")[["toi_off_w_s", "toi_def_w_s"]].sum().reset_index()
        out.append(g.assign(season_year=T))
    return pd.concat(out, ignore_index=True)


def role_toi(r):
    """(season_year, player_id, toi_off_s, toi_def_s, team_id): seconds with the
    player in the attacking / defending list, and the team they spent the most
    time with."""
    parts = []
    for c in ("off", "def"):
        x = r[["season_year", c, "dur", "team_off"]].explode(c).dropna(subset=[c])
        x = x.rename(columns={c: "player_id"}).assign(role=c)
        parts.append(x)
    x = pd.concat(parts)
    x["player_id"] = x["player_id"].astype(int)
    toi = x.pivot_table(index=["season_year", "player_id"], columns="role", values="dur", aggfunc="sum",
                        fill_value=0).rename(columns={"off": "toi_off_s", "def": "toi_def_s"})
    return toi.reset_index()


def main_team(st):
    """(season_year, player_id) -> team with the most time in the stints."""
    parts = []
    for side in ("home", "visitor"):
        x = st[["season_year", "duration_s", f"{side}_team_id", f"{side}_players"]].explode(f"{side}_players")
        x.columns = ["season_year", "duration_s", "team_id", "player_id"]
        parts.append(x.dropna())
    x = pd.concat(parts)
    x["player_id"] = x["player_id"].astype(int)
    by_team = x.groupby(["season_year", "player_id", "team_id"])["duration_s"].sum().reset_index()
    return by_team.sort_values("duration_s").groupby(["season_year", "player_id"]).tail(1)[
        ["season_year", "player_id", "team_id"]]


PRIOR_TEXT = {"bacon": "trend of vanilla RAPM year over year (Bacon)",
              "halflife": "none: earlier seasons in the fit, weighted by one half-life chosen on all seasons' CV",
              "vanilla": "none: each season on its own"}


def choose_half_life(curves):
    """One half-life for every season: the one with the lowest CV MSE summed over the
    seasons that have earlier data (each season at its own best lambda).
    curves: {year: fit_halflife info["cv_curve"]}. Returns (half_life, totals)."""
    tot = {}
    for cv in curves.values():
        best = {}
        for k, v in cv.items():
            hl = float(k.split(",")[0].split("=")[1])
            best[hl] = min(best.get(hl, np.inf), v)
        for hl, v in best.items():
            tot[hl] = tot.get(hl, 0.0) + v
    hl = min(tot, key=tot.get)
    return hl, {f"{h:g}": round(v, 6) for h, v in sorted(tot.items())}


def complete_years(today=None):
    """Season years whose regular season and playoffs have all ended (seasons.end_date
    before today). The half-life is chosen from these only, so a season in progress
    does not move every season's ratings as its first games come in."""
    s = pd.read_parquet(S.ROOT / "data" / "clean" / "seasons.parquet")
    s = s[s["season_type"].isin(S.SEASON_TYPES)]
    s["year"] = s["season_id"].map(S.F.season_years())
    today = pd.Timestamp(today or datetime.now(timezone.utc).date())
    ended = s.groupby("year")["end_date"].apply(lambda e: (pd.to_datetime(e) < today).all())
    return set(int(y) for y in ended[ended].index)


def _rounded(x, nd=6):
    """Floats in nested meta rounded, so the file only changes when results do
    (and not with the last bits of the linear algebra on another platform)."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _rounded(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v, nd) for v in x]
    return x


def write_if_changed(df, meta, path, meta_path):
    """Write the table and meta unless both equal what is on disk (meta compared
    without its "built" time). Returns True if written."""
    t = pa.Table.from_pandas(df, preserve_index=False)
    meta = _rounded(meta)
    if path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        same_meta = {k: v for k, v in old.items() if k != "built"} ==             json.loads(json.dumps({k: v for k, v in meta.items() if k != "built"}))
        if same_meta and pq.read_table(path).equals(t):
            return False
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(t, tmp, compression="zstd")
    os.replace(tmp, path)
    meta_path.write_text(json.dumps(meta, indent=2))
    return True


def run(model, stints_all, method=None):
    spec = MODELS[model]
    method = method or spec["method"]
    st = spec["stints"](stints_all)
    covs = spec["covs"]
    r_all = rows(st, spec["attackers"])
    toi = role_toi(r_all)
    teams_of = main_team(st)
    meta = {"model": f"rapm_{model}_xg", "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": "xG per 60 (data/xg/shot_xg.parquet, model + empty_net rows)",
            "states": spec["states"], "context": covs, "season_types": list(S.SEASON_TYPES),
            "method": method, "prior": PRIOR_TEXT[method],
            "trend_by_position": TREND_BY_POSITION, "half_lives": [float(h) for h in HALF_LIVES],
            "lambdas": LAMBDAS.tolist(), "n_folds": N_FOLDS, "seed": SEED, "seasons": {}}
    print(f"[{model}] pooled: {st['game_id'].nunique()} games, {r_all['w'].sum():.1f} row-hours")
    pooled = fit_pooled(r_all, covs)
    meta["pooled"] = pooled
    ctx = pooled["context_xg60"]
    print(f"  lambda {pooled['lambda']}; context (xG/60): " + ", ".join(f"{k} {v:+.2f}" for k, v in ctx.items()))
    n_games = r_all.groupby("season_year")["game_id"].nunique()
    years = sorted(int(y) for y, n in n_games.items() if n >= MIN_GAMES)
    for y, n in n_games.items():
        if n < MIN_GAMES:
            print(f"  season {y}: only {n} games, not fitted until it has {MIN_GAMES}")
    by_year = {y: r_all[r_all["season_year"] == y] for y in years}
    vanilla, v_info = {}, {}
    for yr in years:
        vanilla[yr], v_info[yr] = fit_season(by_year[yr], ctx)
        print(f"  vanilla {yr}: lambda {v_info[yr]['lambda']}")
    pos = positions() if TREND_BY_POSITION else None
    trend = fit_trend(vanilla, toi, pos)
    meta["trend"] = trend
    for g, tr in trend.items():
        for c, t in tr.items():
            print(f"  trend {g} {spec['names'][c == 'def']}: next = {t['intercept']:+.4f} + {t['slope']:.3f} * this "
                  f"(r {t['r']:+.2f}, {t['pairs']} pairs)")
    hl = None
    if method == "halflife":
        done = complete_years()
        pick = [yr for yr in years if yr > years[0] and yr in done] or [yr for yr in years if yr > years[0]]
        curves = {yr: fit_halflife(by_year, yr, ctx)[1]["cv_curve"] for yr in pick}
        hl, totals = choose_half_life(curves)
        meta["half_life"], meta["half_life_cv_total"], meta["half_life_seasons"] = hl, totals, pick
        print(f"  half-life {hl:g} (CV MSE summed over {', '.join(map(str, curves))}: {totals})")
    out = []
    for yr in years:
        prior = None
        if method == "halflife":
            res, info = fit_halflife(by_year, yr, ctx, (hl,))
        elif method == "vanilla":
            res, info = vanilla[yr], dict(v_info[yr])
        else:
            if yr - 1 in vanilla and trend:
                prior = predicted_prior(vanilla[yr - 1], trend, pos, toi[toi["season_year"] == yr - 1])
            res, info = fit_season(by_year[yr], ctx, prior)
        info["cv_mse_vanilla"] = v_info[yr]["cv_mse"]
        info["lambda_vanilla"] = v_info[yr]["lambda"]
        v = vanilla[yr][["player_id", "off", "def"]].rename(columns={"off": "vanilla_off", "def": "vanilla_def"})
        res = res.merge(v, on="player_id", how="left")
        if prior is not None:
            res = res.merge(prior, on="player_id", how="left")
        else:
            res["prior_off"] = res["prior_def"] = np.nan
        res.insert(0, "season_year", yr)
        out.append(res)
        meta["seasons"][str(yr)] = info
        print(f"  season {yr}: {info['games']} games; lambda {info['lambda']}; "
              + (f"half-life {info['half_life']}; " if method == "halflife" else "") +
              f"CV MSE {info['cv_mse']:.4f} vs vanilla {info['cv_mse_vanilla']:.4f}")
    df = pd.concat(out, ignore_index=True)
    df = df.merge(toi, on=["season_year", "player_id"], how="left").merge(teams_of, on=["season_year", "player_id"],
                                                                         how="left")
    df = df.merge(weighted_toi(toi, years, hl if method == "halflife" else None),
                  on=["season_year", "player_id"], how="left")
    df[["toi_off_w_s", "toi_def_w_s"]] = df[["toi_off_w_s", "toi_def_w_s"]].fillna(0.0).round(1)
    players = pd.read_parquet(S.ROOT / "data" / "clean" / "players.parquet")
    teams = pd.read_parquet(S.ROOT / "data" / "clean" / "teams.parquet")[["team_id", "abbreviation"]]
    df = df.merge(teams, on="team_id", how="left").rename(columns={"abbreviation": "team"})
    df = df.merge(players[["player_id", "first_name", "last_name"]], on="player_id", how="left")
    df["player"] = df["first_name"] + " " + df["last_name"]
    o, d = spec["names"]
    if model == "5v5":
        df["toi_5v5_s"] = df["toi_off_s"]  # each stint is a row for both teams: same time in both roles
        df["toi_weighted_s"] = df["toi_off_w_s"]
        cut = MIN_TOI["5v5"]
        df["qualified"] = (df["toi_5v5_s"] >= cut["season"]) & (df["toi_weighted_s"] >= cut["weighted"])
        tcols = ["toi_5v5_s", "toi_weighted_s", "qualified"]
        rate = ["off", "def", "net"]
    else:
        df = df.rename(columns={"off": o, "def": d, "toi_off_s": "pp_toi_s", "toi_def_s": "pk_toi_s",
                                "toi_off_w_s": "pp_toi_weighted_s", "toi_def_w_s": "pk_toi_weighted_s"})
        cut = MIN_TOI["pp"]
        for r in ("pp", "pk"):
            df[f"{r}_qualified"] = (df[f"{r}_toi_s"] >= cut["season"]) & (df[f"{r}_toi_weighted_s"] >= cut["weighted"])
        tcols = ["pp_toi_s", "pk_toi_s", "pp_toi_weighted_s", "pk_toi_weighted_s", "pp_qualified", "pk_qualified"]
        rate = [o, d]
        # a player with no time in a role has no rating for it (the fit leaves it at its prior)
        df.loc[df["pp_toi_s"] <= 0, [o, "vanilla_off", "prior_off"]] = np.nan
        df.loc[df["pk_toi_s"] <= 0, [d, "vanilla_def", "prior_def"]] = np.nan
    df = df.rename(columns={"vanilla_off": f"vanilla_{o}", "vanilla_def": f"vanilla_{d}",
                            "prior_off": f"prior_{o}", "prior_def": f"prior_{d}"})
    cols = rate + [f"vanilla_{o}", f"vanilla_{d}", f"prior_{o}", f"prior_{d}"]
    df[cols] = df[cols].astype(float).round(5)
    df = df[["season_year", "player_id", "player", "team_id", "team", *tcols, *cols]]
    key = "net" if model == "5v5" else o
    df = df.sort_values(["season_year", key, "player_id"], ascending=[True, False, True]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if method == spec["method"] else f"_{method}"
    path = OUT_DIR / f"rapm_{model}{suffix}_xg.parquet"
    if write_if_changed(df, meta, path, OUT_DIR / f"rapm_{model}{suffix}_xg.meta.json"):
        print(f"  wrote {path.relative_to(S.ROOT)} ({len(df)} rows)")
    else:
        print(f"  {path.relative_to(S.ROOT)} unchanged")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=[*MODELS, "all"], default="all")
    ap.add_argument("--prior", choices=list(PRIOR_TEXT),
                    help="override the model's method (MODELS[...]['method']); the output then goes to "
                         "rapm_<model>_<method>_xg instead of rapm_<model>_xg")
    a = ap.parse_args()
    stints_all = S.build()
    for m in (MODELS if a.model == "all" else [a.model]):
        run(m, stints_all, a.prior)


if __name__ == "__main__":
    main()
