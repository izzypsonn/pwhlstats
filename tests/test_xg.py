"""Checks for the xG features and the saved model (xg/). Run in CI after xg/score.py."""
import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xg"))
CLEAN = ROOT / "data" / "clean"
pytestmark = pytest.mark.skipif(not (CLEAN / "games.parquet").exists(), reason="no exported data")

import features  # noqa: E402

EV = f"read_parquet('{(CLEAN / 'events').as_posix()}/*.parquet', hive_partitioning = false)"


@pytest.fixture(scope="module")
def raw():
    return features.load_raw()


@pytest.fixture(scope="module")
def shots(raw):
    return features.add_features(raw.copy())


def test_every_goal_row_pairs_with_exactly_one_shot_or_penalty_shot(raw):
    """The feed logs each goal twice: a `shot` (is_goal = 1) or `penaltyshot` (is_goal = 1)
    event AND a `goal` row at the same game/period/second/team. features.py counts the
    shot / penalty shot and only takes flags from the goal row."""
    unmatched = duckdb.sql(f"""
        SELECT g.* FROM {EV} g
        WHERE g.event = 'goal' AND NOT EXISTS (
            SELECT 1 FROM {EV} s WHERE s.event IN ('shot', 'penaltyshot') AND s.is_goal = 1
              AND s.game_id = g.game_id AND s.period = g.period
              AND s.period_time_s = g.period_time_s AND s.team_id = g.team_id)""").df()
    assert unmatched.empty, unmatched
    goals = raw[raw["is_goal"] == 1]
    assert goals["goal_event_idx"].notna().all(), "scoring shot without a goal row"
    assert not goals.duplicated(["game_id", "goal_event_idx"]).any(), "goal row matched twice"
    ps = features.penalty_shots()
    assert ps.loc[ps["is_goal"] == 1, "goal_event_idx"].notna().all()


def test_one_row_per_shot(shots, raw):
    assert len(shots) == len(raw)
    assert not shots.duplicated(["game_id", "event_idx"]).any()


def test_categories_cover_the_feed(raw, shots):
    for col, levels in features.CATEGORICAL.items():
        src = {"prev_event": raw["prev_event_raw"].fillna("none"), "shot_type": raw["shot_type"]}.get(col)
        if src is None:  # derived in add_features: NaN = a value outside the levels
            assert shots[col].notna().all(), col
            continue
        new = set(src.dropna()) - set(levels)
        assert not new, f"new {col} values in the feed: {new} (add them and train a new model version)"


def test_feature_ranges(shots):
    assert shots["skaters_for"].notna().all() and shots["skaters_against"].notna().all()
    assert shots["skaters_for"].between(3, 6).all() and shots["skaters_against"].between(3, 6).all()
    assert shots["distance_ft"].between(0, 200).all()
    assert (shots["x_from_goal_ft"] > -12).mean() > 0.999  # almost nothing far behind the net
    assert shots["time_remaining_s"].between(0, 1200).all()
    # handedness from the rosters: every shot but those with no goalie (id 0) has both
    assert shots["shooter_shoots"].notna().mean() > 0.995
    assert shots["goalie_catches"].notna().mean() > 0.995


def test_side_convention(shots):
    """Shooter's-left convention (features.lateral_ft): left defenders shoot from
    their left point. Breaks if the feed's y axis is ever flipped."""
    pos = duckdb.sql(f"""SELECT DISTINCT game_id, player_id, position FROM
        read_parquet('{(CLEAN / 'game_rosters').as_posix()}/*.parquet', hive_partitioning = false)""").df()
    d = shots.merge(pos, left_on=["game_id", "shooter_id"], right_on=["game_id", "player_id"])
    lat = features.lateral_ft(d["y"], d["att_far"])
    point = d["x_from_goal_ft"] > 40
    assert lat[point & (d["position"] == "LD")].mean() > 5
    assert lat[point & (d["position"] == "RD")].mean() < -5


def test_saved_shots_are_never_empty_net(shots):
    assert not ((shots["xg_source"] == "empty_net") & (shots["is_goal"] == 0)).any()


def test_saved_model_matches_features_and_is_calibrated(shots):
    if not (ROOT / "models" / "current.txt").exists():
        pytest.skip("no model trained yet")
    import score
    booster, meta = score.load()  # also fails if features/categories/settings changed under the model
    m = shots[shots["xg_source"] == "model"]
    p = score.predict(booster, meta, m)
    assert ((p > 0) & (p < 1)).all()
    # total xG within 10% of goals (v2 on 2026-11-25: see models/xg_v2.meta.json)
    assert abs(p.sum() / m["is_goal"].sum() - 1) < 0.10


def test_offset_math():
    import numpy as np
    import score
    p = np.array([0.02, 0.05, 0.1, 0.3])
    b = score.fit_offset(p, 1.0)
    assert abs(score.sigmoid(score.logit(p) + b).sum() - 1.0) < 1e-8
    assert score.fit_offset([], 0) == 0.0
    assert score.shrink(0.1, 100, 100) == 0.05 and score.shrink(0.1, 5, np.inf) == 0.0
    assert score.shrink(0.1, 5, 0) == 0.1


def test_season_offsets_applied(shots):
    out = ROOT / "data" / "xg" / "shot_xg.parquet"
    if not out.exists():
        pytest.skip("not scored yet")
    import json
    import numpy as np
    import pandas as pd
    import score
    x = pd.read_parquet(out)
    meta = json.loads((ROOT / "models" / f"xg_{x['model'].iloc[0]}.meta.json").read_text())
    if meta.get("offset_k") is None:
        pytest.skip("model without in-season offsets")
    m = x[x["xg_source"] == "model"]
    assert np.allclose(m["xg"], score.sigmoid(score.logit(m["xg_unadjusted"]) + m["season_offset"]), atol=2e-5)
    assert (x.loc[x["xg_source"] != "model", "season_offset"] == 0).all()
    assert (m.groupby("season_year")["season_offset"].nunique() == 1).all()
    # shrunk offsets: each training season's model xG within 3% of its goals, but not forced to equal them
    t = m.merge(shots[["game_id", "event_idx", "season_type"]], on=["game_id", "event_idx"])
    t = t[t["season_type"].isin(features.TRAIN_SEASON_TYPES)]
    cal = t.groupby("season_year")["xg"].sum() / t.groupby("season_year")["is_goal"].sum()
    assert ((cal - 1).abs() < 0.03).all(), cal


def test_scored_output_is_current(shots):
    out = ROOT / "data" / "xg" / "shot_xg.parquet"
    if not out.exists():
        pytest.skip("not scored yet")
    import pandas as pd
    x = pd.read_parquet(out)
    assert len(x) == len(shots) + len(features.penalty_shots())
    assert not x.duplicated(["game_id", "event_idx"]).any()
    assert (x["model"] == (ROOT / "models" / "current.txt").read_text().strip()).all()
    assert x["xg"].notna().all()
    assert (x.loc[x["xg_source"] == "empty_net", "xg"] == features.XG_EMPTY_NET).all()
    assert (x.loc[x["xg_source"] == "penalty_shot", "xg"] == features.XG_PENALTY_SHOT).all()
