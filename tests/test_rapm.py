"""Unit tests for rapm/ (stint boundaries and the ridge fit). No data needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rapm"))

import fit  # noqa: E402
import stints  # noqa: E402


def _strength(rows):
    return pd.DataFrame(rows, columns=["start_s", "end_s", "state", "home_goalie_in", "visitor_goalie_in"])


def _shifts(rows, L=1200):
    """rows: (player_id, side, elapsed start, elapsed end) -> countdown clock columns."""
    return pd.DataFrame([{"player_id": p, "side": s, "clock_in_s": L - a, "clock_out_s": L - b}
                         for p, s, a, b in rows])


def test_stints_split_at_shift_and_strength_changes_and_merge_otherwise():
    sh = _shifts([(1, "home", 0, 60), (2, "home", 60, 120), (3, "visitor", 0, 120)])
    st = _strength([(0, 90, "5v5", 1, 1), (90, 1200, "5v4", 1, 1)])
    out = stints.period_stints(sh, st, 120, 1200)
    assert [(s["start_s"], s["end_s"], s["home_players"], s["state"]) for s in out] == [
        (0, 60, [1], "5v5"), (60, 90, [2], "5v5"), (90, 120, [2], "5v4")]
    assert all(s["visitor_players"] == [3] for s in out)


def test_stints_stop_at_period_end_given():
    sh = _shifts([(1, "home", 0, 300)], L=300)
    out = stints.period_stints(sh, _strength([(0, 300, "3v3", 1, 1)]), 136, 300)
    assert out[-1]["end_s"] == 136


def test_shot_at_boundary_goes_to_the_stint_ending_there():
    st = pd.DataFrame({"game_id": 1, "period": 1, "start_s": [0, 60], "end_s": [60, 120], "home_team_id": 10})
    shots = pd.DataFrame({"game_id": [1, 1, 1], "period": [1, 1, 1], "period_time_s": [0, 60, 61],
                          "team_id": [10, 10, 20], "xg": [0.1, 0.2, 0.3], "is_goal": [0, 1, 0]})
    out = stints.add_shots(st, shots)
    assert out["home_xg"].round(6).tolist() == [0.3, 0.0]
    assert out["home_goals"].tolist() == [1, 0]
    assert out["visitor_xg"].round(6).tolist() == [0.0, 0.3]


def test_ridge_recovers_player_effects_on_synthetic_data():
    rng = np.random.default_rng(0)
    n_p, n = 40, 6000
    off, dfn = rng.normal(0, 0.4, n_p), rng.normal(0, 0.4, n_p)
    rows = []
    for g in range(n):
        ps = rng.choice(n_p, 10, replace=False)
        a, d = sorted(ps[:5].tolist()), sorted(ps[5:].tolist())
        w = rng.uniform(0.005, 0.03)
        home = int(rng.integers(2))
        mu = 2.0 + 0.15 * home + off[a].sum() + dfn[d].sum()
        rows.append({"game_id": g, "off": a, "def": d, "home": home, "w": w, "y": mu + rng.normal(0, 0.2) / np.sqrt(w)})
    r = pd.DataFrame(rows)
    r["season_year"] = 2026
    r["const"] = 1
    r["start_oz"] = rng.integers(2, size=len(r))
    r["y"] += 0.8 * r["start_oz"]
    fixed = ["const", "home", "start_oz"]
    col = {(2026, p): p for p in range(n_p)}
    G, b, _, _ = fit.sums(r, fixed, col)
    beta = fit.solve(G, b, 1.0, len(fixed))
    nf = len(fixed)
    assert np.corrcoef(beta[nf:nf + n_p], off)[0, 1] > 0.9
    assert np.corrcoef(beta[nf + n_p:], dfn)[0, 1] > 0.9
    assert abs(beta[1] - 0.15) < 0.05
    assert abs(beta[2] - 0.8) < 0.05


def test_season_fit_takes_pooled_context_out_of_y():
    rng = np.random.default_rng(1)
    n = 4000
    r = pd.DataFrame({"game_id": np.arange(n) % 200, "season_year": 2026,
                      "off": [[int(i)] for i in rng.integers(10, size=n)],
                      "def": [[int(i) + 10] for i in rng.integers(10, size=n)], "w": 0.01, "xg": 0.0})
    for c in fit.COVS:
        r[c] = rng.integers(2, size=n)
    ctx = {c: 0.5 for c in fit.COVS}
    r["y"] = 2.0 + r[fit.COVS].sum(axis=1) * 0.5
    res, info = fit.fit_season(r, ctx)
    assert abs(info["intercept_xg60"] - 2.0) < 1e-6
    assert res[["off", "def"]].abs().max().max() < 1e-6


def test_rows_flip_context_to_the_attacking_side():
    st = pd.DataFrame({"game_id": [1], "season_year": [2026], "home_team_id": [10], "visitor_team_id": [20],
                       "home_players": [[1]], "visitor_players": [[2]], "score_home": [2],
                       "start_home": ["oz"], "home_b2b": [1], "visitor_b2b": [0], "pen_exp_home": [0],
                       "pen_exp_visitor": [1], "home_xg": [0.1], "visitor_xg": [0.2], "duration_s": [36]})
    r = fit.rows(st).set_index("home")
    assert r.loc[1, ["lead_p2", "start_oz", "b2b_att", "pen_exp_def"]].tolist() == [1, 1, 1, 1]
    assert r.loc[0, ["lead_m2", "start_dz", "b2b_def", "pen_exp_att"]].tolist() == [1, 1, 1, 1]
    assert r.loc[0, "y"] == 20.0


def test_faceoff_zone_relative_to_home_attacking_end():
    assert stints.faceoff_zone(500, True) == "oz"
    assert stints.faceoff_zone(100, True) == "dz"
    assert stints.faceoff_zone(100, False) == "oz"
    assert stints.faceoff_zone(300, None) == "nz"
    assert stints.faceoff_zone(500, None) is None


def test_faceoff_starts_a_new_stint_even_with_the_same_players():
    sh = _shifts([(1, "home", 0, 120), (3, "visitor", 0, 120)])
    out = stints.period_stints(sh, _strength([(0, 1200, "5v5", 1, 1)]), 120, 1200, faceoff_t=[0, 50])
    assert [(s["start_s"], s["end_s"]) for s in out] == [(0, 50), (50, 120)]


def test_solve_shrinks_towards_the_prior():
    # one unpenalised constant + one player column with no information: lands on the prior
    G = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([2.0, 0.0])
    beta = fit.solve(G, b, 5.0, 1, np.array([0.0, 0.3]))
    assert np.allclose(beta, [2.0, 0.3])


def test_trend_recovers_a_known_year_over_year_slope():
    rng = np.random.default_rng(2)
    p = np.arange(200)
    x = rng.normal(0, 0.1, 200)
    v = {2025: pd.DataFrame({"player_id": p, "off": x, "def": x}),
         2026: pd.DataFrame({"player_id": p, "off": 0.01 + 0.5 * x, "def": 0.2 * x})}
    toi = pd.DataFrame({"season_year": [2025] * 200 + [2026] * 200, "player_id": list(p) * 2,
                        "toi_off_s": 36000, "toi_def_s": 36000})
    t = fit.fit_trend(v, toi)["all"]
    assert abs(t["off"]["slope"] - 0.5) < 1e-6 and abs(t["off"]["intercept"] - 0.01) < 1e-6
    assert abs(t["def"]["slope"] - 0.2) < 1e-6
    pr = fit.predicted_prior(v[2025], {"all": t})
    assert np.allclose(pr["prior_off"], 0.01 + 0.5 * x)


def test_trend_by_group_with_fallback():
    rng = np.random.default_rng(3)
    p = np.arange(200)
    x = rng.normal(0, 0.1, 200)
    pos = pd.Series(np.where(p < 100, "F", "D"), index=p)
    slope = np.where(p < 100, 0.6, 0.1)
    v = {2025: pd.DataFrame({"player_id": p, "off": x, "def": x}),
         2026: pd.DataFrame({"player_id": p, "off": slope * x, "def": slope * x})}
    toi = pd.DataFrame({"season_year": [2025] * 200 + [2026] * 200, "player_id": list(p) * 2,
                        "toi_off_s": 36000, "toi_def_s": 36000})
    t = fit.fit_trend(v, toi, pos)
    assert abs(t["F"]["off"]["slope"] - 0.6) < 1e-6 and abs(t["D"]["off"]["slope"] - 0.1) < 1e-6
    pos2 = pos.drop(0)  # player 0 has no position -> all-skater trend
    pr = fit.predicted_prior(v[2025], t, pos2).set_index("player_id")
    assert np.isclose(pr.loc[1, "prior_off"], t["F"]["off"]["intercept"] + 0.6 * x[1])
    assert np.isclose(pr.loc[0, "prior_off"], t["all"]["off"]["intercept"] + t["all"]["off"]["slope"] * x[0])


def test_pp_rows_only_have_the_power_play_team_attacking():
    st = pd.DataFrame({"game_id": [1, 1], "season_year": 2026, "state": ["5v4", "4v5"], "home_team_id": 10,
                       "visitor_team_id": 20, "home_players": [[1], [1]], "visitor_players": [[2], [2]],
                       "score_home": 0, "start_home": ["oz", "dz"], "home_b2b": 0, "visitor_b2b": 0,
                       "pen_exp_home": 0, "pen_exp_visitor": 0, "home_xg": [0.3, 0.0], "visitor_xg": [0.1, 0.2],
                       "duration_s": 60})
    r = fit.rows(st, "pp")
    assert len(r) == 2
    assert r[["home", "team_off", "xg"]].values.tolist() == [[1, 10, 0.3], [0, 20, 0.2]]
    assert r["start_oz"].tolist() == [1, 1]  # 4v5 stint: dz for home = oz for the visitor's power play
    assert r["off"].tolist() == [[1], [2]]


def test_predicted_prior_needs_time_in_the_role():
    v = pd.DataFrame({"player_id": [1, 2], "off": [0.2, 0.0], "def": [0.1, 0.1]})
    tr = {"all": {"off": {"intercept": 0.0, "slope": 0.5}, "def": {"intercept": 0.0, "slope": 0.5}}}
    toi = pd.DataFrame({"player_id": [1, 2], "toi_off_s": [600, 0], "toi_def_s": [600, 600]})
    pr = fit.predicted_prior(v, tr, None, toi).set_index("player_id")
    assert pr.loc[1, "prior_off"] == 0.1 and np.isnan(pr.loc[2, "prior_off"])
    assert pr.loc[2, "prior_def"] == 0.05


def test_choose_half_life_sums_best_cv_over_seasons():
    curves = {2025: {"half_life=0,lambda=1": 10.0, "half_life=1,lambda=1": 9.0, "half_life=1,lambda=10": 8.5,
                     "half_life=2,lambda=1": 8.0},
              2026: {"half_life=0,lambda=1": 10.0, "half_life=1,lambda=1": 9.0, "half_life=2,lambda=1": 9.8}}
    hl, tot = fit.choose_half_life(curves)
    assert hl == 1.0 and tot["1"] == 17.5 and tot["2"] == 17.8
