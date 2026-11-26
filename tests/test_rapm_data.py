"""Data checks on the RAPM outputs (data/rapm). Run in CI after rapm/fit.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rapm"))
RAPM = ROOT / "data" / "rapm"
pytestmark = pytest.mark.skipif(not (RAPM / "rapm_5v5_xg.parquet").exists(), reason="no RAPM output")

import fit  # noqa: E402


@pytest.fixture(scope="module", params=list(fit.MODELS))
def model(request):
    m = request.param
    df = pd.read_parquet(RAPM / f"rapm_{m}_xg.parquet")
    meta = json.loads((RAPM / f"rapm_{m}_xg.meta.json").read_text())
    return m, df, meta


def test_one_row_per_player_and_season(model):
    _, df, _ = model
    assert not df.duplicated(["season_year", "player_id"]).any()
    assert df["player"].notna().all()


def test_every_fitted_season_has_players_and_meta(model):
    _, df, meta = model
    assert sorted(meta["seasons"]) == sorted(str(y) for y in df["season_year"].unique())
    for y, info in meta["seasons"].items():
        assert info["games"] >= fit.MIN_GAMES
        assert np.isfinite(info["cv_mse"])


def test_method_and_half_life_are_recorded(model):
    m, _, meta = model
    assert meta["method"] == fit.MODELS[m]["method"]
    if meta["method"] == "halflife" and len(meta["seasons"]) > 1:
        assert meta["half_life"] in [float(h) for h in fit.HALF_LIVES]


def test_ratings_are_finite_and_in_a_sane_range(model):
    m, df, _ = model
    if m == "5v5":
        r = df[["off", "def", "net"]]
        assert r.notna().all().all() and (df["toi_5v5_s"] > 0).all()
        assert np.allclose(df["net"], df["off"] + df["def"], atol=2e-5)
        assert r.abs().max().max() < 1.5  # xG/60 above average; the top is ~0.45
    else:
        # a rating exists exactly when the player had time in that role
        assert (df["ppo"].notna() == (df["pp_toi_s"] > 0)).all()
        assert (df["shd"].notna() == (df["pk_toi_s"] > 0)).all()
        assert df[["ppo", "shd"]].abs().max().max() < 4  # PP rates are ~5 xG/60; the top is ~0.6


def test_ratings_centre_near_average(model):
    """Ridge shrinks towards 0 = average skater: the time-weighted mean rating stays small."""
    m, df, _ = model
    cols = [("off", "toi_5v5_s"), ("def", "toi_5v5_s")] if m == "5v5" else [("ppo", "pp_toi_s"), ("shd", "pk_toi_s")]
    for y, g in df.groupby("season_year"):
        for c, t in cols:
            x = g[g[t] > 0]
            assert abs(np.average(x[c], weights=x[t])) < 0.15, (m, y, c)
