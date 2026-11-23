"""Unit tests for ingest/pwhl_ingest.py (no network)."""
from datetime import date

import pandas as pd

import pwhl_ingest as pi


def test_parse_json_strips_jsonp():
    assert pi.parse_json('angular.callbacks._0({"a": 1});') == {"a": 1}
    assert pi.parse_json(' {"a": [1, 2]} ') == {"a": [1, 2]}


def test_n_events_and_n_shifts():
    assert pi.n_events([{"event": "shot"}, {"event": "goal"}]) == 2
    assert pi.n_events({"x": []}) == 0
    good = {"SiteKit": {"Gameshifts": {"home": [{"shifts": [{}, {}]}], "visitor": [{"shifts": [{}]}]}}}
    assert pi.n_shifts(good) == 3
    assert pi.n_shifts({"SiteKit": {"Gameshifts": {"home": [{"shifts": [{}]}], "visitor": []}}}) == 0
    assert pi.n_shifts({"error": "x"}) == 0


def test_upsert_game_keeps_timestamp_when_unchanged():
    df = pi.read_payloads("does-not-exist")
    df, changed = pi.upsert_game(df, 5, '{"e":1}', '{"s":1}', "t1")
    assert changed and pi.stored_ids(df) == {5}
    df2, changed = pi.upsert_game(df, 5, '{"e":1}', '{"s":1}', "t2")
    assert not changed and set(df2.fetched_at) == {"t1"}
    df3, changed = pi.upsert_game(df, 5, '{"e":2}', '{"s":1}', "t3")
    assert changed and dict(zip(df3.feed, df3.fetched_at)) == {"events": "t3", "shifts": "t1"}


def test_write_if_changed_is_a_no_op_for_identical_data(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = tmp_path / "f.parquet"
    assert pi.write_if_changed(df, p) is True
    assert pi.write_if_changed(df.copy(), p) is False
    assert pi.write_if_changed(df.assign(a=[1, 3]), p) is True


def test_finals_and_season_complete():
    sched = pd.DataFrame({"game_id": ["1", "2", "3"], "final": ["1", "0", None],
                          "game_status": ["Final", "7:00 pm", "Final OT"]}).astype("string")
    assert list(pi.finals(sched).game_id) == ["1", "3"]
    done = pd.DataFrame({"game_id": ["1", "3"], "final": ["1", "1"], "game_status": ["Final", "Final"]}).astype("string")
    assert pi.season_complete(done, {1, 3}, date(2025, 1, 1), date(2026, 1, 1))
    assert not pi.season_complete(done, {1}, date(2025, 1, 1), date(2026, 1, 1))        # game 3 missing
    assert not pi.season_complete(done, {1, 3}, date(2025, 12, 30), date(2026, 1, 1))   # ended < 7 days ago
