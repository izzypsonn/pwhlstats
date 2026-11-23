"""Unit tests for the strength rules in ingest/strength.py (penalties, goalies, blank shifts)."""
import json

import numpy as np
import pytest

import strength
from strength import fill_blank_ends, game_strength, goalie_in_net, refine_goalie, skaters

HOME, AWAY = 1, 2


def pen(t, team, player, mins=2, pp=1, desc="Tripping", period=1):
    return {"event": "penalty", "period": period, "period_time_s": t, "team_id": team, "penalty_minutes": mins,
            "is_power_play": pp, "served_by": player, "penalty_description": desc}


def goal(t, team, period=1, en=0, plus_n=5):
    return {"event": "goal", "period": period, "period_time_s": t, "team_id": team, "is_empty_net": en, "plus_n": plus_n}


def end_marker(period=3):
    # a harmless event so the game has periods 1..period
    return {"event": "faceoff", "period": period, "period_time_s": 0}


def state(rows, t, period=1):
    """(home_skaters, visitor_skaters) in force at elapsed second t (start_s <= t < end_s)."""
    r = next(r for r in rows if r["period"] == period and r["start_s"] <= t < r["end_s"])
    return r["home_skaters"], r["visitor_skaters"]


def run(events, season_type="regular", jailbreak=True, goalie_side=None, shifts=None):
    rows, _ = game_strength(1, season_type, HOME, events + [end_marker()], shifts or [], goalie_side or {},
                            jailbreak=jailbreak)
    return rows


def test_minor_expires_after_two_minutes():
    rows = run([pen(100, HOME, 7)])
    assert state(rows, 99) == (5, 5)
    assert state(rows, 150) == (4, 5)
    assert state(rows, 220) == (5, 5)


def test_power_play_goal_ends_minor():
    rows = run([pen(100, HOME, 7), goal(160, AWAY)])
    assert state(rows, 159) == (4, 5)
    assert state(rows, 160) == (5, 5)


def test_jailbreak_shorthanded_goal_ends_minor():
    assert state(run([pen(100, HOME, 7), goal(160, HOME)]), 170) == (5, 5)
    assert state(run([pen(100, HOME, 7), goal(160, HOME)], jailbreak=False), 170) == (4, 5)


def test_goal_at_even_strength_releases_nobody():
    rows = run([pen(100, HOME, 7), pen(100, AWAY, 8), goal(150, AWAY)])
    assert state(rows, 170) == (4, 4)


def test_coincidental_and_misconduct_penalties_ignored():
    rows = run([pen(100, HOME, 7, pp=0), pen(100, AWAY, 8, mins=10, pp=1)])
    assert state(rows, 150) == (5, 5)


def test_major_not_released_by_goal():
    rows = run([pen(100, HOME, 7, mins=5), goal(150, AWAY)])
    assert state(rows, 200) == (4, 5)
    assert state(rows, 399) == (4, 5)
    assert state(rows, 400) == (5, 5)


def test_third_penalty_waits_for_a_free_slot():
    rows = run([pen(100, HOME, 7), pen(110, HOME, 8), pen(120, HOME, 9)])
    assert state(rows, 125) == (3, 5)          # never below 3 skaters
    assert state(rows, 225) == (3, 5)          # first ended at 220, third started then
    assert state(rows, 235) == (4, 5)          # second ended at 230
    assert state(rows, 339) == (4, 5)          # third runs 220-340
    assert state(rows, 340) == (5, 5)


def test_double_minor_same_time_is_one_player_for_four_minutes():
    rows = run([pen(100, HOME, 7), pen(100, HOME, 7)])
    assert state(rows, 150) == (4, 5)
    assert state(rows, 339) == (4, 5)
    assert state(rows, 340) == (5, 5)


def test_double_minor_goal_in_first_half_restarts_second_half():
    rows = run([pen(100, HOME, 7), pen(100, HOME, 7), goal(150, AWAY)])
    assert state(rows, 200) == (4, 5)          # second half 150-270
    assert state(rows, 270) == (5, 5)


def test_double_minor_logged_two_minutes_apart():
    # feed quirk: second half logged at the first half's scheduled end (220)
    rows = run([pen(100, HOME, 7), goal(150, AWAY), pen(220, HOME, 7)])
    assert state(rows, 260) == (4, 5)          # second half started at the goal (150)...
    assert state(rows, 275) == (5, 5)          # ...so it ends at 270, not 340


def test_penalty_carries_into_next_period():
    rows = run([pen(1150, HOME, 7)])
    assert state(rows, 1199) == (4, 5)
    assert state(rows, 50, period=2) == (4, 5)
    assert state(rows, 70, period=2) == (5, 5)


@pytest.mark.parametrize("nh,nv,ot,expected", [
    (0, 0, True, (3, 3)), (1, 0, True, (3, 4)), (2, 0, True, (3, 5)), (0, 1, True, (4, 3)),
    (1, 1, True, (3, 3)), (1, 0, False, (4, 5)), (2, 1, False, (3, 4))])
def test_skaters_mapping(nh, nv, ot, expected):
    assert skaters(nh, nv, ot) == expected


def test_regular_season_overtime_is_3v3_for_300_seconds():
    rows, _ = game_strength(1, "regular", HOME, [{"event": "faceoff", "period": 4, "period_time_s": 0}], [], {})
    ot = [r for r in rows if r["period"] == 4]
    assert ot[0]["start_s"] == 0 and ot[-1]["end_s"] == 300
    assert state(rows, 10, period=4) == (3, 3)


def test_intervals_tile_each_period():
    rows = run([pen(100, HOME, 7), goal(160, AWAY), pen(1150, AWAY, 9)])
    for p in (1, 2, 3):
        pr = [r for r in rows if r["period"] == p]
        assert pr[0]["start_s"] == 0 and pr[-1]["end_s"] == 1200
        assert all(a["end_s"] == b["start_s"] for a, b in zip(pr, pr[1:]))
        assert sum(r["duration_s"] for r in pr) == 1200


# --- goalies -----------------------------------------------------------------
def test_goalie_in_net_short_pull_and_ignored_long_pull():
    a = goalie_in_net([(100, False), (130, True)], 1200)
    assert a[:100].all() and not a[100:130].any() and a[130:].all()
    b = goalie_in_net([(100, False)], 1200)        # never came back: > MAX_PULL_S, ignored
    assert b.all()


def make_refine_inputs(L=1200):
    in_net = np.ones(L, bool)
    gcov = np.ones(L, bool)
    extra = np.zeros(L, bool)
    return in_net, gcov, extra


def test_refine_goalie_shift_gap_with_extra_skater_is_a_pull():
    in_net, gcov, extra = make_refine_inputs()
    gcov[1000:1100] = False
    extra[1010:1100] = True
    out = ~refine_goalie(in_net, gcov, gcov.mean(), extra, [], [], 1200)
    assert out[1000:1100].all() and not out[:1000].any() and not out[1100:].any()


def test_refine_goalie_gap_without_extra_skater_is_not_a_pull():
    in_net, gcov, extra = make_refine_inputs()
    gcov[1000:1100] = False
    assert refine_goalie(in_net, gcov, gcov.mean(), extra, [], [], 1200).all()


def test_refine_goalie_ignores_gap_at_period_start():
    in_net, gcov, extra = make_refine_inputs()
    gcov[0:40] = False
    extra[0:40] = True
    assert refine_goalie(in_net, gcov, gcov.mean(), extra, [], [], 1200).all()


def test_refine_goalie_ignores_gaps_when_goalie_shifts_mostly_missing():
    in_net, gcov, extra = make_refine_inputs()
    gcov[:700] = False
    gcov[1000:1100] = False
    extra[1000:1100] = True
    assert refine_goalie(in_net, gcov, gcov.mean(), extra, [], [], 1200).all()


def test_refine_goalie_pull_starts_after_last_save():
    in_net, gcov, extra = make_refine_inputs()
    gcov[1000:1100] = False
    extra[1000:1100] = True
    out = ~refine_goalie(in_net, gcov, gcov.mean(), extra, [], [1050], 1200)
    assert not out[:1050].any() and out[1050:1100].all()


def test_refine_goalie_goal_anchor_without_other_evidence():
    in_net, gcov, extra = make_refine_inputs()
    out = ~refine_goalie(in_net, gcov, gcov.mean(), extra, [900], [], 1200)
    assert out[899] and out.sum() == 1


def test_empty_net_goal_moves_early_goalie_return():
    ev = [{"event": "goalie_change", "period": 3, "period_time_s": 1000, "team_id": HOME, "goalie_out": 30},
          {"event": "goalie_change", "period": 3, "period_time_s": 1059, "team_id": HOME, "goalie_in": 30},
          goal(1060, AWAY, period=3, en=1)]
    rows, _ = game_strength(1, "regular", HOME, ev, [], {30: "home"})
    r = next(r for r in rows if r["period"] == 3 and r["start_s"] < 1060 <= r["end_s"])
    assert r["home_goalie_in"] == 0 and r["state"] == "6v5"


# --- blank shift ends ----------------------------------------------------------
def shift(pid, cin, cout, side="home", period=1, blank=False, start=None):
    extra = {"start_time": start or f"{cin // 60:02d}:{cin % 60:02d}", "end_time": "" if blank else "x"}
    return {"player_id": pid, "side": side, "period": period, "clock_in_s": None if blank else cin,
            "clock_out_s": None if blank else cout, "duration_s": None if blank else cin - cout,
            "extra": json.dumps(extra), "end_inferred": 0}


def test_fill_blank_ends_stops_when_team_would_be_over_and_caps_length():
    L = 1200
    full = [shift(p, 1200, 0) for p in (1, 2, 3, 4)]            # four skaters all period
    fifth = shift(5, 1200, 600)                                  # fifth skater until elapsed 600
    no_room = shift(6, None, None, blank=True, start="11:40")    # elapsed 500: team already has 5
    room = shift(7, None, None, blank=True, start="09:10")       # elapsed 650: room for one
    ps = full + [fifth, no_room, room]
    prows = [{"period": 1, "start_s": 0, "end_s": L, "home_skaters": 5, "visitor_skaters": 5,
              "home_goalie_in": 1, "visitor_goalie_in": 1}]
    fill_blank_ends(ps, prows, {}, L)
    assert no_room["clock_in_s"] is None and no_room["end_inferred"] == 1
    assert room["clock_in_s"] == 550 and room["duration_s"] == strength.MAX_INFERRED_SKATER_S
