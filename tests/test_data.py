"""Data checks on the committed Parquet (data/raw, data/clean). In CI they run after
ingest + normalize and before anything is committed or deployed, so a feed change
that breaks parsing stops the run. Thresholds sit a little below the values seen on
2026-11-22 (in comments) so that normal noise passes but a real regression fails."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN = ROOT / "data" / "raw", ROOT / "data" / "clean"
pytestmark = pytest.mark.skipif(not (CLEAN / "games.parquet").exists(), reason="no exported data")


def table(name):
    d = CLEAN / name
    if d.is_dir():
        return pd.concat([pd.read_parquet(p) for p in sorted(d.glob("season=*.parquet"))], ignore_index=True)
    return pd.read_parquet(CLEAN / f"{name}.parquet")


@pytest.fixture(scope="module")
def t():
    return {n: table(n) for n in ("games", "seasons", "events", "event_players", "event_on_ice", "shifts", "strength")} | {
        "quality": pd.read_parquet(CLEAN / "quality" / "game_strength_quality.parquet")}


def test_every_final_game_is_stored_or_logged_as_failed():
    stored = set()
    for p in (RAW / "payloads").glob("season=*.parquet"):
        d = pd.read_parquet(p)
        feeds = d.groupby("game_id").feed.nunique()
        stored |= set(feeds[feeds == 2].index)
    failed = set(pd.read_csv(RAW / "failures.csv").game_id) if (RAW / "failures.csv").exists() else set()
    finals = set()
    for p in (RAW / "schedule").glob("season=*.parquet"):
        s = pd.read_parquet(p)
        f = s["final"].eq("1") | s["game_status"].str.lower().str.startswith("final")
        finals |= set(s.loc[f.fillna(False), "game_id"].astype(int))
    assert not (finals - stored - failed), f"final games neither stored nor failed: {sorted(finals - stored - failed)[:10]}"


def test_keys_unique(t):
    assert not t["games"].game_id.duplicated().any()
    assert not t["events"].duplicated(["game_id", "event_idx"]).any()
    assert not t["strength"].duplicated(["game_id", "period", "start_s"]).any()


def test_every_played_game_has_events_shifts_strength(t):
    played = set(t["games"].loc[t["games"].has_pbp == 1, "game_id"])
    for name in ("events", "shifts", "strength"):
        missing = played - set(t[name].game_id)
        assert not missing, f"{name} missing for {sorted(missing)[:10]}"


def test_known_bad_games_excluded(t):
    g = t["games"].set_index("game_id")
    assert (g.loc[[142, 114, 9, 61], "exclude_shifts"] == 1).all()
    assert g.exclude_shifts.sum() == 4


def test_events_look_like_hockey(t):
    ev = t["events"]
    per_game = ev.groupby("game_id").event.value_counts().unstack(fill_value=0)
    assert per_game["shot"].median() >= 45                 # ~56 shots on goal per game
    assert per_game["faceoff"].median() >= 45              # ~55
    pens = ev[ev.event == "penalty"]
    assert pens.team_id.notna().all(), "penalties without a team (againstteam parsing)"
    goals = ev[ev.event == "goal"]
    assert goals.team_id.notna().mean() > 0.99
    shots = ev[ev.event.isin(["shot", "goal"]) & ev.x.notna()]
    assert shots.x.between(0, 600).all() and shots.y.between(0, 300).all()
    sog = ev[ev.event == "shot"]
    assert 0.0 < (sog.coords_mirrored == 1).mean() < 0.03  # 1.3%


def test_goals_are_also_logged_as_shots(t):
    ev = t["events"]
    g = ev[ev.event == "goal"][["game_id", "period", "period_time_s"]]
    s = ev[(ev.event == "shot") & (ev.is_goal == 1)][["game_id", "period", "period_time_s"]]
    assert len(g.merge(s.drop_duplicates())) / len(g) > 0.98   # 1564 / 1566: xG must not double count


def test_shifts_are_consistent(t):
    sh = t["shifts"]
    ok = sh.clock_in_s.notna()
    assert (sh.loc[ok, "clock_in_s"] >= sh.loc[ok, "clock_out_s"]).all()
    assert (sh.loc[ok, "duration_s"] == sh.loc[ok, "clock_in_s"] - sh.loc[ok, "clock_out_s"]).all()
    assert sh.loc[ok, "clock_in_s"].between(0, 1200).all()
    assert ok.mean() > 0.99                                  # 99.5% of shifts have times
    assert (sh.end_inferred == 1).sum() < 0.01 * len(sh)
    stuck = sh[ok & (sh.clock_out_s == 0) & (sh.duration_s > 180) & (sh.end_inferred == 0)]
    # only goalies may still run > 3 min to the period end
    ros = table("game_rosters")
    g_ids = set(zip(ros.loc[ros.position == "G", "game_id"], ros.loc[ros.position == "G", "player_id"]))
    assert all((gid, pid) in g_ids for gid, pid in zip(stuck.game_id, stuck.player_id)), "uncapped stuck skater shifts"
    assert ((sh.end_capped == 1) & (sh.duration_s != 60)).sum() == 0


def test_strength_tiles_every_period(t):
    st = t["strength"].sort_values(["game_id", "period", "start_s"])
    g = st.groupby(["game_id", "period"])
    assert (g.start_s.min() == 0).all()
    assert (g.duration_s.sum() == g.end_s.max()).all()
    assert g.end_s.max().isin([300, 1200]).all()
    assert st.home_skaters.between(3, 5).all() and st.visitor_skaters.between(3, 5).all()
    share = st.groupby("state").duration_s.sum() / st.duration_s.sum()
    assert 0.75 < share["5v5"] < 0.88                        # 80.5%


def test_goal_lists_repaired(t):
    o = t["event_on_ice"]
    assert set(o.source) == {"feed", "shifts"}
    n = o.groupby(["game_id", "event_idx", "side"]).size().unstack(fill_value=0)
    short = n[(n.get("plus", 0) < 3) | (n.get("minus", 0) < 3)]
    # only game 335 (preseason; tracking outage from the start of period 2) stays short
    assert set(short.index.get_level_values(0)) <= {335}, short


def test_strength_matches_goals(t):
    ev, st, games = t["events"], t["strength"], t["games"]
    pm = t["event_on_ice"].groupby(["game_id", "event_idx"]).side.value_counts().unstack(fill_value=0).reset_index()
    goals = ev[(ev.event == "goal") & ev.period_time_s.notna() & (ev.period <= 4)].merge(pm, on=["game_id", "event_idx"])
    goals = goals.merge(games[["game_id", "home_team_id", "exclude_shifts"]], on="game_id")
    goals = goals[goals.exclude_shifts == 0]
    m = goals.merge(st, on=["game_id", "period"])
    m = m[(m.start_s < m.period_time_s) & (m.period_time_s <= m.end_s)]
    home = m.team_id == m.home_team_id
    hs = m.home_skaters + 1 - m.home_goalie_in
    vs = m.visitor_skaters + 1 - m.visitor_goalie_in
    match = (hs.where(home, vs) == m.plus) & (vs.where(home, hs) == m.minus)
    assert match.mean() > 0.96                               # 98.1%
    en = m[m.is_empty_net == 1]
    def_in = en.visitor_goalie_in.where(en.team_id == en.home_team_id, en.home_goalie_in)
    assert (def_in == 0).mean() > 0.97                       # 100%


def test_shift_quality_not_regressed(t):
    q = t["quality"].merge(t["games"][["game_id", "exclude_shifts"]], on="game_id")
    q = q[q.exclude_shifts == 0]
    assert q.pct_ok.median() > 82                            # 87.8
    shq = pd.read_parquet(CLEAN / "quality" / "game_shift_quality.parquet")
    assert (shq.pct_on_ice * shq.shots).sum() / shq.shots.sum() > 86   # 89.7% of shooters on the ice
