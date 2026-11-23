"""Unit tests for parsing in ingest/normalize.py."""
import normalize as nz


def test_mmss():
    assert nz.mmss("12:34") == 754 and nz.mmss("0:05") == 5 and nz.mmss("") is None and nz.mmss(None) is None


def test_parse_shifts_countdown_clock():
    payload = {"SiteKit": {"Gameshifts": {"home": [{"player_id": "10", "first_name": "A", "last_name": "B", "shifts": [
        {"period": "1", "start_time": "05:00", "end_time": "04:10"},
        {"period": "1", "start_time": "03:00", "end_time": ""}]}], "visitor": []}}}
    rows = nz.parse_shifts(1, payload, nz.Collector())
    assert (rows[0]["clock_in_s"], rows[0]["clock_out_s"], rows[0]["duration_s"]) == (300, 250, 50)
    assert rows[1]["clock_in_s"] is None and rows[1]["end_inferred"] == 0


def test_penalty_team_comes_from_againstteam():
    payload = [{"event": "penalty", "details": {
        "period": {"id": "2"}, "time": "5:07", "minutes": "2.00", "description": "Tripping", "isPowerPlay": True,
        "againstTeam": {"id": 4, "name": "Team", "abbreviation": "TT"},
        "takenBy": {"id": 99, "firstName": "X", "lastName": "Y"}, "servedBy": {"id": 99, "firstName": "X", "lastName": "Y"}}}]
    ev, eplayers, _ = nz.parse_events(1, payload, nz.Collector())
    assert ev[0]["team_id"] == 4 and ev[0]["period"] == 2 and ev[0]["period_time_s"] == 307
    assert {r for _, _, _, r in eplayers} == {"penalty_taken_by", "penalty_served_by"}


def test_flag_mirrored_coords():
    shot = lambda i, x, team=1: {"event": "shot", "event_idx": i, "period": 1, "period_time_s": 100 + i,
                                 "team_id": team, "x": x, "y": 150}
    ev = [shot(i, 500) for i in range(5)] + [shot(10, 50), shot(11, 280)]
    rows = [{"period": 1, "start_s": 0, "end_s": 1200, "home_goalie_in": 1, "visitor_goalie_in": 1}]
    nz.flag_mirrored_coords(ev, rows, home_team_id=1)
    assert [e["coords_mirrored"] for e in ev] == [0, 0, 0, 0, 0, 1, 0]   # deep wrong half only
    rows[0]["visitor_goalie_in"] = 0                                       # empty net: genuine long shot
    nz.flag_mirrored_coords(ev, rows, home_team_id=1)
    assert ev[5]["coords_mirrored"] == 0


def test_cap_stuck_shifts():
    mk = lambda pid, cin, cout, inferred=0: {"player_id": pid, "clock_in_s": cin, "clock_out_s": cout,
                                             "duration_s": cin - cout, "end_inferred": inferred, "end_capped": 0}
    stuck, normal, late_long, goalie, inferred = mk(1, 1048, 0), mk(2, 50, 0), mk(3, 400, 100), mk(30, 1200, 0), mk(4, 900, 0, 1)
    n = nz.cap_stuck_shifts([stuck, normal, late_long, goalie, inferred], goalie_ids={30})
    assert n == 1
    assert (stuck["clock_out_s"], stuck["duration_s"], stuck["end_capped"]) == (988, 60, 1)
    assert normal["end_capped"] == late_long["end_capped"] == goalie["end_capped"] == inferred["end_capped"] == 0


def test_fill_goal_lists_tops_up_short_sides_from_shifts():
    sh = lambda pid, side, cin, cout: {"player_id": pid, "side": side, "period": 1, "clock_in_s": cin, "clock_out_s": cout}
    shifts = [sh(p, "home", 1000, 700) for p in (1, 2, 3, 4, 5)] + [sh(p, "visitor", 1000, 700) for p in (11, 12, 13, 14, 15)] \
        + [sh(6, "home", 790, 600)]                     # arrives at the goal second (clock 800 -> not on, in-play rule)
    ev = [{"event": "goal", "event_idx": 7, "game_id": 1, "period": 1, "period_time_s": 400, "team_id": 1}]
    onice = [(1, 7, 1, "plus", "feed")]                  # incomplete entry: one scorer-side player, no minus list
    eplayers = [(1, 7, 2, "goal_scorer")]
    rows = [{"period": 1, "start_s": 0, "end_s": 1200, "home_skaters": 5, "visitor_skaters": 5,
             "home_goalie_in": 1, "visitor_goalie_in": 1}]
    added = nz.fill_goal_lists(ev, eplayers, onice, shifts, {}, "regular", 1, rows)
    plus = {r[2] for r in added if r[3] == "plus"}
    minus = {r[2] for r in added if r[3] == "minus"}
    assert plus == {2, 3, 4, 5} and minus == {11, 12, 13, 14, 15}
    assert all(r[4] == "shifts" for r in added)
    complete = [(1, 7, p, "plus", "feed") for p in (1, 2, 3, 4, 5)] + [(1, 7, p, "minus", "feed") for p in (11, 12, 13, 14, 15)]
    assert nz.fill_goal_lists(ev, eplayers, complete, shifts, {}, "regular", 1, rows) == []
