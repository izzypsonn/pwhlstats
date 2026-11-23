#!/usr/bin/env python3
"""
Build a clean, typed, normalized database from the raw JSON that pwhl_ingest.py
stores under data/raw/. No API calls; safe to re-run (rebuilds everything).

  python ingest/normalize.py            # -> pwhl_clean.db (local SQLite) + data/clean/*.parquet

Outputs
  pwhl_clean.db       SQLite with the tables and views below (not committed; for local SQL)
  data/clean/         the same tables as Parquet (committed; used by the site's data loaders)
                      per-game tables are partitioned: data/clean/<table>/season=N.parquet

Tables
  seasons(season_id, season_name, shortname, season_type, start_date, end_date)
  games(game_id, season_id, date_played, home/visiting team + goals, game_status, exclude_shifts ...)
  players(player_id, first_name, last_name, birth_date)
  teams(team_id, name, city, nickname, abbreviation)
  game_rosters(game_id, player_id, side, jersey_number, position)
  events(game_id, event_idx, event, period, period_time_s, team_id, x, y, ...)
  event_players(game_id, event_idx, player_id, role)   -- shooter, goalie, assist_1 ...
  event_on_ice(game_id, event_idx, player_id, side)    -- from plus/minus player lists
  shifts(game_id, player_id, side, period, clock_in_s, clock_out_s, duration_s, extra)
      -- the shift clock counts DOWN: clock_in_s = clock reading when the player stepped on
      -- (the larger number), clock_out_s = reading when they stepped off
  strength(game_id, period, start_s, end_s, home/visitor_skaters, home/visitor_goalie_in, manpower, state)
      -- strength state per stretch of play; rules and time conventions in ingest/strength.py
  strength_check, game_strength_quality (view) -- per-second skater count from shifts vs. strength
  shift_coverage, shooter_check, game_shift_quality (views) -- how well shifts agree with shooters, per game
  event_on_ice_derived (view) -- on-ice players for every event, from shifts
Anything not mapped to a column is kept as JSON in the `extra` column.
"""
import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pwhl_ingest import list_of_dicts
from strength import game_strength

SCHEMA = """
CREATE TABLE seasons (
    season_id INTEGER PRIMARY KEY, season_name TEXT, shortname TEXT, season_type TEXT,
    is_playoff INTEGER, is_career INTEGER, start_date TEXT, end_date TEXT);
CREATE TABLE games (
    game_id INTEGER PRIMARY KEY, season_id INTEGER REFERENCES seasons, game_number INTEGER,
    date_played TEXT, game_datetime TEXT, home_team_id INTEGER, visiting_team_id INTEGER,
    home_goals INTEGER, visiting_goals INTEGER, periods INTEGER, overtime INTEGER,
    shootout INTEGER, final INTEGER, game_status TEXT, attendance INTEGER, venue_name TEXT,
    has_pbp INTEGER, exclude_shifts INTEGER, exclude_reason TEXT);
CREATE TABLE players (player_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, birth_date TEXT);
CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT, city TEXT, nickname TEXT, abbreviation TEXT);
CREATE TABLE game_rosters (
    game_id INTEGER, player_id INTEGER REFERENCES players, side TEXT,
    jersey_number INTEGER, position TEXT, PRIMARY KEY (game_id, player_id));
CREATE TABLE events (
    game_id INTEGER, event_idx INTEGER, event TEXT, period INTEGER, period_time_s INTEGER,
    team_id INTEGER REFERENCES teams, x REAL, y REAL, is_goal INTEGER, shot_type TEXT,
    shot_quality TEXT, home_win INTEGER, penalty_minutes REAL, penalty_description TEXT,
    rule_number TEXT, is_power_play INTEGER, is_short_handed INTEGER, is_empty_net INTEGER,
    is_penalty_shot INTEGER, is_insurance_goal INTEGER, is_game_winning_goal INTEGER,
    is_bench INTEGER,
    -- 1 = shot/goal whose coordinates are almost certainly mirrored (recorded >25 ft
    -- behind the red line on the wrong side for its team that period, against a goalie
    -- in net); 0 = other shots/goals with coordinates. See flag_mirrored_coords().
    coords_mirrored INTEGER,
    extra TEXT, PRIMARY KEY (game_id, event_idx));
CREATE TABLE event_players (
    game_id INTEGER, event_idx INTEGER, player_id INTEGER REFERENCES players, role TEXT,
    PRIMARY KEY (game_id, event_idx, player_id, role));
-- side: plus (scoring team) / minus (conceding team). source: feed = the feed's own
-- plus/minus lists; shifts = filled from shift data where the feed's list for a side
-- had fewer than 3 skaters (see fill_goal_lists).
CREATE TABLE event_on_ice (
    game_id INTEGER, event_idx INTEGER, player_id INTEGER REFERENCES players, side TEXT, source TEXT,
    PRIMARY KEY (game_id, event_idx, player_id, side));
CREATE TABLE shifts (
    game_id INTEGER, player_id INTEGER REFERENCES players, side TEXT, period INTEGER,
    clock_in_s INTEGER, clock_out_s INTEGER, duration_s INTEGER,
    end_inferred INTEGER,  -- 1 = end_time was blank; end inferred (see strength.fill_blank_ends)
    end_capped INTEGER,    -- 1 = "stuck" shift (ran to 00:00 for > 3 min); cut to 60 s (cap_stuck_shifts)
    extra TEXT);
CREATE INDEX shifts_game ON shifts (game_id, player_id);

-- Period length in seconds: 1200 for periods 1-3; overtime periods use the
-- largest shift start (the clock counts DOWN: clock_in_s is the reading when
-- the player stepped on, clock_out_s the reading when they stepped off).
CREATE VIEW period_len AS
    SELECT game_id, period,
           CASE WHEN period <= 3 THEN 1200 ELSE MAX(clock_in_s) END AS len
    FROM shifts GROUP BY game_id, period;

-- On-ice players for EVERY event (goalies included), derived from shifts.
-- c = countdown clock at the event = period length - elapsed time.
-- Stoppage events (faceoff, goalie change): a player arriving at c is on, one
-- leaving at c is off. In-play events: a player finishing a shift at c is on,
-- one who just arrived at c is not. Shootout / untimed events get no rows.
CREATE VIEW event_on_ice_derived AS
    SELECT e.game_id, e.event_idx, s.player_id, s.side
    FROM events e
    JOIN period_len pl ON pl.game_id = e.game_id AND pl.period = e.period
    JOIN shifts s ON s.game_id = e.game_id AND s.period = e.period
    WHERE e.period_time_s IS NOT NULL AND (
        (COALESCE(e.event, '') IN ('faceoff', 'goalie_change')
            AND s.clock_out_s <  pl.len - e.period_time_s AND pl.len - e.period_time_s <= s.clock_in_s)
        OR
        (COALESCE(e.event, '') NOT IN ('faceoff', 'goalie_change')
            AND s.clock_out_s <= pl.len - e.period_time_s AND pl.len - e.period_time_s <  s.clock_in_s));

-- Data-quality check: a shooter must be on the ice when they shoot, so how often
-- the shift data agrees tells us how far to trust it for that game.
-- gap_s = seconds from the shot to the nearest shift of that player (0 if inside).
CREATE VIEW shooter_check AS
SELECT game_id, event_idx, player_id, period, c,
       EXISTS(SELECT 1 FROM shifts s
              WHERE s.game_id = q.game_id AND s.player_id = q.player_id AND s.period = q.period
                AND s.clock_out_s <= q.c AND q.c <= s.clock_in_s) AS on_ice,
       (SELECT MIN(MAX(s.clock_out_s - q.c, q.c - s.clock_in_s, 0)) FROM shifts s
         WHERE s.game_id = q.game_id AND s.player_id = q.player_id AND s.period = q.period) AS gap_s
FROM (SELECT ep.game_id, ep.event_idx, ep.player_id, e.period, pl.len - e.period_time_s AS c
      FROM event_players ep
      JOIN events e ON e.game_id = ep.game_id AND e.event_idx = ep.event_idx
      JOIN period_len pl ON pl.game_id = e.game_id AND pl.period = e.period
      WHERE ep.role = 'shooter' AND e.period_time_s IS NOT NULL
        AND COALESCE(e.event, '') <> 'shootout') q;

-- Independent of events: in regulation each side has 5 skaters + a goalie = 6 players
-- on the ice for 1200 s, so total shift time should be near 7200 player-seconds
-- (a bit less with penalties). Far below 100 = missing shifts; above 100 = overlapping
-- or duplicated shifts. Regulation periods only (overtime has fewer players).
CREATE VIEW shift_coverage AS
SELECT game_id, side, period,
       COUNT(*) AS n_shifts,
       SUM(duration_s) AS player_seconds,
       ROUND(100.0 * SUM(duration_s) / (6 * 1200), 1) AS pct_of_full
FROM shifts WHERE period BETWEEN 1 AND 3 GROUP BY game_id, side, period;

-- Strength state per stretch of play (see ingest/strength.py for the rules).
CREATE TABLE strength (
    game_id INTEGER, period INTEGER, start_s INTEGER, end_s INTEGER, duration_s INTEGER,
    home_skaters INTEGER, visitor_skaters INTEGER, home_goalie_in INTEGER, visitor_goalie_in INTEGER,
    manpower TEXT, state TEXT, PRIMARY KEY (game_id, period, start_s));
-- Per game/period/side: seconds where the number of SKATERS on the ice per the
-- shifts equals / is below / is above what the strength table expects (goalies
-- excluded; their shifts are too incomplete). A shift-data quality measure.
CREATE TABLE strength_check (
    game_id INTEGER, period INTEGER, side TEXT, seconds INTEGER, sec_ok INTEGER,
    sec_under INTEGER, sec_over INTEGER, player_sec_missing INTEGER, player_sec_extra INTEGER,
    PRIMARY KEY (game_id, period, side));
CREATE VIEW game_strength_quality AS
SELECT game_id, SUM(seconds) AS side_seconds,
       ROUND(100.0 * SUM(sec_ok) / SUM(seconds), 1) AS pct_ok,
       ROUND(100.0 * SUM(sec_under) / SUM(seconds), 1) AS pct_under,
       ROUND(100.0 * SUM(sec_over) / SUM(seconds), 1) AS pct_over
FROM strength_check GROUP BY game_id;

CREATE VIEW game_shift_quality AS
SELECT game_id, COUNT(*) AS shots,
       ROUND(100.0 * SUM(on_ice) / COUNT(*), 1) AS pct_on_ice,
       ROUND(100.0 * SUM(COALESCE(gap_s, 999) <= 15) / COUNT(*), 1) AS pct_within_15s
FROM shooter_check GROUP BY game_id;
"""

# raw `details` key (lower-cased) -> role name in event_players
ROLE_NAMES = {
    "takenby": "penalty_taken_by", "servedby": "penalty_served_by", "scoredby": "goal_scorer",
    "homeplayer": "faceoff_home", "visitingplayer": "faceoff_visitor",
    "goaliecomingin": "goalie_in", "goaliegoingout": "goalie_out",
}
# raw scalar `details` key (lower-cased) -> events column
SCALARS = {
    "isgoal": "is_goal", "shottype": "shot_type", "shotquality": "shot_quality",
    "homewin": "home_win", "minutes": "penalty_minutes", "description": "penalty_description",
    "rulenumber": "rule_number", "ispowerplay": "is_power_play", "isbench": "is_bench",
    "xlocation": "x", "ylocation": "y",
}
PROPS = {  # details.properties.*
    "ispowerplay": "is_power_play", "isshorthanded": "is_short_handed",
    "isemptynet": "is_empty_net", "ispenaltyshot": "is_penalty_shot",
    "isinsuranceGoal".lower(): "is_insurance_goal", "isgamewinninggoal": "is_game_winning_goal",
}
BOOLS = {"is_goal", "home_win", "is_power_play", "is_short_handed", "is_empty_net",
         "is_penalty_shot", "is_insurance_goal", "is_game_winning_goal", "is_bench"}
# `againstteam` is the PENALIZED team on penalty events (matches the side of the
# player serving the penalty for every penalty in the data).
TEAM_KEYS = ("team", "shooterteam", "shooter_team", "againstteam")


# --- small converters -------------------------------------------------------
def to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_bool(v):
    if v is None or v == "":
        return None
    if isinstance(v, str):
        return 1 if v.strip().lower() in ("1", "true", "yes", "y") else 0
    return int(bool(v))


def mmss(v):
    """'12:34' -> 754"""
    m = re.fullmatch(r"\s*(\d+):(\d{1,2})\s*", str(v)) if v is not None else None
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def lower(d):
    return {str(k).lower(): v for k, v in d.items()}


def is_team(d):
    return isinstance(d, dict) and ("abbreviation" in lower(d) or "nickname" in lower(d))


def is_player(d):
    return isinstance(d, dict) and not is_team(d) and to_int(lower(d).get("id")) is not None


# --- collectors -------------------------------------------------------------
class Collector:
    def __init__(self):
        self.players, self.teams, self.rosters = {}, {}, {}

    def player(self, d, game_id=None, side=None):
        d = lower(d)
        pid = to_int(d.get("id", d.get("player_id")))
        if pid is None:
            return None
        row = self.players.setdefault(pid, {})
        for src, dst in (("firstname", "first_name"), ("first_name", "first_name"),
                         ("lastname", "last_name"), ("last_name", "last_name"),
                         ("birthdate", "birth_date")):
            if d.get(src) not in (None, ""):
                row[dst] = d[src]
        if game_id is not None:
            r = self.rosters.setdefault((game_id, pid), {})
            jersey = to_int(d.get("jerseynumber", d.get("jersey_number")))
            if jersey is not None:
                r["jersey_number"] = jersey
            if d.get("position"):
                r["position"] = d["position"]
            if side:
                r["side"] = side
        return pid

    def team(self, d):
        if isinstance(d, dict):
            d = lower(d)
            tid = to_int(d.get("id"))
            if tid is None:
                return None
            row = self.teams.setdefault(tid, {})
            for k in ("name", "city", "nickname", "abbreviation"):
                if d.get(k):
                    row[k] = d[k]
            return tid
        tid = to_int(d)
        if tid is not None:
            self.teams.setdefault(tid, {})
        return tid


# --- events -----------------------------------------------------------------
def find_events(payload):
    if isinstance(payload, list):
        return payload
    for _, rows in list_of_dicts(payload):
        if rows and "event" in lower(rows[0]):
            return rows
    return []


def parse_events(game_id, payload, col):
    events, eplayers, onice = [], [], []
    for idx, raw in enumerate(find_events(payload)):
        raw = lower(raw)
        det = lower(raw.get("details") or {})
        row = {"game_id": game_id, "event_idx": idx, "event": raw.get("event")}
        extra = {}
        team_id = None
        for key, val in det.items():
            if key == "period" and isinstance(val, dict):
                row["period"] = to_int(lower(val).get("id"))
            elif key == "period_id" or key == "periodid":
                row["period"] = to_int(val)
            elif key == "time":
                row["period_time_s"] = mmss(val)
            elif key == "assists" and isinstance(val, list):
                for n, a in enumerate(val, 1):
                    pid = col.player(a, game_id) if isinstance(a, dict) else to_int(a)
                    if pid is not None:
                        col.players.setdefault(pid, {})
                        eplayers.append((game_id, idx, pid, f"assist_{n}"))
            elif key in ("plus_players", "minus_players") and isinstance(val, list):
                side = "plus" if key.startswith("plus") else "minus"
                for a in val:
                    pid = col.player(a, game_id) if isinstance(a, dict) else to_int(a)
                    if pid is not None:
                        col.players.setdefault(pid, {})
                        onice.append((game_id, idx, pid, side, "feed"))
            elif key == "properties" and isinstance(val, dict):
                for pk, pv in lower(val).items():
                    if pk in PROPS:
                        row[PROPS[pk]] = to_bool(pv)
                    else:
                        extra[f"properties_{pk}"] = pv
            elif is_team(val):
                tid = col.team(val)
                if key in TEAM_KEYS and team_id is None:
                    team_id = tid
                elif key not in TEAM_KEYS:
                    extra[f"{key}_id"] = tid
            elif is_player(val):
                pid = col.player(val, game_id)
                eplayers.append((game_id, idx, pid, ROLE_NAMES.get(key, key)))
            elif key in ("teamid", "shooterteamid") and val not in (None, ""):
                tid = col.team(val)
                team_id = team_id if team_id is not None else tid
            elif key == "team_id":
                team_id = team_id if team_id is not None else col.team(val)
            elif key in SCALARS:
                row[SCALARS[key]] = val
            elif val not in (None, "", [], {}):
                extra[key] = val
        row["team_id"] = team_id
        for b in BOOLS:
            if b in row:
                row[b] = to_bool(row[b])
        for f in ("x", "y", "penalty_minutes"):
            if f in row:
                row[f] = to_float(row[f])
        row["extra"] = json.dumps(extra) if extra else None
        events.append(row)
    return events, eplayers, onice


# --- shifts -----------------------------------------------------------------
def first(d, *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def parse_shifts(game_id, payload, col):
    out = []
    try:
        gs = payload["SiteKit"]["Gameshifts"]
    except (KeyError, TypeError):
        return out
    for side in ("home", "visitor"):
        for p in gs.get(side) or []:
            pid = col.player(p, game_id, side)
            if pid is None:
                continue
            for sh in p.get("shifts") or []:
                sh = lower(sh) if isinstance(sh, dict) else {"value": sh}
                per = first(sh, "period", "period_id", "periodid")
                if isinstance(per, dict):
                    per = lower(per).get("id")
                a = mmss(first(sh, "start_time", "starttime", "start", "time_on"))
                b = mmss(first(sh, "end_time", "endtime", "end", "time_off"))
                # Countdown clock: start_time is the LARGER reading. Stored as the
                # larger (clock_in_s) / smaller (clock_out_s) value, whichever way it arrives.
                ok = a is not None and b is not None
                cin, cout = (max(a, b), min(a, b)) if ok else (None, None)
                out.append({"game_id": game_id, "player_id": pid, "side": side,
                            "period": to_int(per), "clock_in_s": cin, "clock_out_s": cout,
                            "duration_s": cin - cout if ok else None, "end_inferred": 0, "end_capped": 0,
                            "extra": json.dumps(sh)})
    return out


# --- driver -----------------------------------------------------------------
# Games whose shift data is known to be unusable. They stay in every table but
# are flagged in games.exclude_shifts; consumers (TOI, RAPM, ...) filter on it.
KNOWN_BAD_SHIFTS = {
    142: "period 1: 36 shifts all start at 20:00; period 2: 4 shifts; no period 3 shifts",
    114: "39% of skater-seconds with the right count (league median 88%), 72% of shooters on the ice, "
         "67% of plus/minus players found at goals",
    9: "57% of shooters on the ice (league 90%), 62% of skater-seconds with the right count, "
       "72% of plus/minus players found at goals",
    61: "shift tracking stopped ~3 min into period 1 (9 skaters stuck on to 00:00, no later shifts); "
        "NY shifts missing in periods 2-3 (53% right skater count, 75% of NY shooters on the ice)",
}

# A skater shift that "ends" at 00:00 after more than STUCK_MIN_S is a tracking
# outage: the player's later shifts in the period were never recorded (see game 61).
# Simulation: cutting these to 60 s helps RAPM slightly (too few players on the ice is
# less harmful than too many); dropping the affected periods hurts.
STUCK_MIN_S, STUCK_CAP_S = 180, 60


def cap_stuck_shifts(shifts, goalie_ids):
    """Cut stuck skater shifts to STUCK_CAP_S (in place). Returns how many were cut."""
    n = 0
    for s in shifts:
        if (s["player_id"] not in goalie_ids and s.get("clock_out_s") == 0 and not s.get("end_inferred")
                and s.get("duration_s") is not None and s["duration_s"] > STUCK_MIN_S):
            s["clock_out_s"] = s["clock_in_s"] - STUCK_CAP_S
            s["duration_s"] = STUCK_CAP_S
            s["end_capped"] = 1
            n += 1
    return n


def insert(con, table, rows):
    if not rows:
        return
    tcols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    if isinstance(rows[0], dict):
        # rows may have different optional keys; use the table's columns
        cols = [c for c in tcols if any(c in r for r in rows)]
        data = [tuple(r.get(c) for c in cols) for r in rows]
    else:
        cols, data = tcols[:len(rows[0])], rows
    con.executemany(
        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", data)


def season_key(path):
    return int(re.search(r"season=(\d+)", path.name).group(1))


def season_rows(seasons):
    out = []
    for r in seasons.to_dict("records"):
        playoff = to_int(r.get("playoff"))
        name = r.get("season_name") or ""
        out.append({"season_id": to_int(r["season_id"]), "season_name": name,
                    "shortname": r.get("shortname"),
                    "season_type": "playoffs" if playoff == 1 else
                                   "preseason" if "pre" in name.lower() else "regular",
                    "is_playoff": playoff, "is_career": to_int(r.get("career")),
                    "start_date": r.get("start_date"), "end_date": r.get("end_date")})
    return out


def game_rows(sched, have_pbp, col):
    out = []
    for r in sched.to_dict("records"):
        gid = to_int(r["game_id"])
        for side in ("home", "visiting"):  # schedule has the most complete team info
            col.team({"id": r.get(f"{side}_team"), "name": r.get(f"{side}_team_name"),
                      "city": r.get(f"{side}_team_city"), "nickname": r.get(f"{side}_team_nickname"),
                      "abbreviation": r.get(f"{side}_team_code")})
        out.append({"game_id": gid, "season_id": to_int(r.get("season_id")),
                    "game_number": to_int(r.get("game_number")), "date_played": r.get("date_played"),
                    "game_datetime": r.get("GameDateISO8601"),
                    "home_team_id": to_int(r.get("home_team")), "visiting_team_id": to_int(r.get("visiting_team")),
                    "home_goals": to_int(r.get("home_goal_count")),
                    "visiting_goals": to_int(r.get("visiting_goal_count")),
                    "periods": to_int(r.get("period")), "overtime": to_bool(r.get("overtime")),
                    "shootout": to_bool(r.get("shootout")), "final": to_bool(r.get("final")),
                    "game_status": r.get("game_status"), "attendance": to_int(r.get("attendance")),
                    "venue_name": r.get("venue_name"), "has_pbp": int(gid in have_pbp),
                    "exclude_shifts": int(gid in KNOWN_BAD_SHIFTS),
                    "exclude_reason": KNOWN_BAD_SHIFTS.get(gid)})
    return out


def flag_mirrored_coords(ev, strength_rows, home_team_id):
    """Set coords_mirrored on shot/goal rows (in place). A team shoots at one end per
    period; its end is where most of its shots that period are (x > 300 or not).
    A shot recorded in the other half, more than 25 ft (75 units) behind the red
    line, while the defending goalie is in net, is flagged: its goal rate matches
    its mirrored position (11 goals vs 15 expected) and a real on-goal shot from
    there is implausible. Shots into an empty net are not flagged (genuine long
    shots), nor are wrong-half shots from the neutral zone (could be real dump-ins;
    near-zero xG either way)."""
    shots = [e for e in ev if e.get("event") in ("shot", "goal") and e.get("x") is not None
             and e.get("period") and e.get("period_time_s") is not None and e.get("team_id") is not None]
    ends = {}
    for e in shots:
        if e["event"] == "shot":
            ends.setdefault((e["period"], e["team_id"]), []).append(e["x"] > 300)
    st = {}
    for r in strength_rows:
        st.setdefault(r["period"], []).append(r)
    for e in shots:
        far = ends.get((e["period"], e["team_id"]))
        if not far:
            e["coords_mirrored"] = 0
            continue
        att_far = sum(far) / len(far) > 0.5
        t = e["period_time_s"]
        row = next((r for r in st.get(e["period"], []) if r["start_s"] < t <= r["end_s"]), None)
        home = e["team_id"] == home_team_id
        goalie_in = row is not None and (row["visitor_goalie_in"] if home else row["home_goalie_in"]) == 1
        e["coords_mirrored"] = int((e["x"] > 300) != att_far and abs(e["x"] - 300) > 75 and goalie_in)


def fill_goal_lists(ev, eplayers, onice, shifts, goalie_side, season_type, home_team_id, strength_rows,
                    min_list=3):
    """Repair incomplete plus/minus lists from the shifts. A goal is repaired when its
    plus or minus list has fewer than `min_list` skaters (6 goals, 5 of them in Ottawa
    home games: the conceding side was simply not entered). For such a goal, each side
    below the strength table's expected skater count keeps its listed players (and,
    for the scoring side, the scorer and assisters) and is topped up with the skaters
    the shifts put on the ice at the goal (in-play rule: leaving at the goal second =
    on, arriving = off), most clearly on first (furthest from a shift change), until
    the expected count is reached. Returns rows to add (source = 'shifts')."""
    from strength import period_len
    lists = {}
    for _, i, pid, side, _src in onice:
        lists.setdefault(i, {"plus": set(), "minus": set()})[side].add(pid)
    named = {}
    for _, i, pid, role in eplayers:
        if role in ("goal_scorer", "assist_1", "assist_2"):
            named.setdefault(i, set()).add(pid)
    out = []
    for e in ev:
        i = e["event_idx"]
        if e.get("event") != "goal" or i not in lists or e.get("period_time_s") is None or not e.get("period"):
            continue
        if min(len(lists[i]["plus"]), len(lists[i]["minus"])) >= min_list:
            continue
        p, t = int(e["period"]), e["period_time_s"]
        c = period_len(p, season_type) - t
        row = next((r for r in strength_rows if r["period"] == p and r["start_s"] < t <= r["end_s"]), None)
        if row is None:
            continue
        expected = {sd: row[f"{sd}_skaters"] + 1 - row[f"{sd}_goalie_in"] for sd in ("home", "visitor")}
        scoring = "home" if e.get("team_id") == home_team_id else "visitor"
        margin = {"home": {}, "visitor": {}}   # player -> seconds to the nearer edge of the covering shift
        for s in shifts:
            if s["period"] == p and s["clock_in_s"] is not None and s["player_id"] not in goalie_side                     and s["clock_out_s"] <= c < s["clock_in_s"]:
                margin[s["side"]][s["player_id"]] = min(c - s["clock_out_s"], s["clock_in_s"] - c)
        for pm_side, team in (("plus", scoring), ("minus", "visitor" if scoring == "home" else "home")):
            have = set(lists[i][pm_side])
            keep = have | (named.get(i, set()) if pm_side == "plus" else set())
            add = sorted(keep - have)
            cands = sorted((pid for pid in margin[team] if pid not in keep), key=lambda x: -margin[team][x])
            while len(keep) + len([x for x in add if x not in keep]) < expected[team] and cands:
                add.append(cands.pop(0))
            for pid in sorted(set(add)):
                out.append((e["game_id"], i, pid, pm_side, "shifts"))
    return out


# --- Parquet export ------------------------------------------------------------
PARTITIONED = {  # table -> ORDER BY (deterministic output = no-op commits stay no-ops)
    "events": "t.game_id, t.event_idx",
    "event_players": "t.game_id, t.event_idx, t.role, t.player_id",
    "event_on_ice": "t.game_id, t.event_idx, t.side, t.player_id",
    "shifts": "t.game_id, t.rowid",
    "game_rosters": "t.game_id, t.player_id",
    "strength": "t.game_id, t.period, t.start_s",
}
SINGLE = {"seasons": "season_id", "games": "game_id", "players": "player_id", "teams": "team_id",
          "shift_coverage": "game_id, side, period", "game_shift_quality": "game_id",
          "strength_check": "game_id, period, side", "game_strength_quality": "game_id"}
QUALITY = {"shift_coverage", "game_shift_quality", "strength_check", "game_strength_quality"}
ARROW = {"INTEGER": pa.int64(), "REAL": pa.float64(), "TEXT": pa.string()}


def arrow_table(con, name, sql, args=()):
    decl = {r[1]: ARROW.get(r[2].upper()) for r in con.execute(f"PRAGMA table_info({name})")}
    cur = con.execute(sql, args)
    names = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cols = list(zip(*rows)) if rows else [()] * len(names)
    return pa.table({n: pa.array(list(c), type=decl.get(n)) for n, c in zip(names, cols)})


def write_table(t, path):
    """Atomic write, skipped when the file already holds identical data."""
    if path.exists():
        try:
            if pq.ParquetFile(path).read().equals(t):
                return False
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(t, tmp, compression="zstd")
    os.replace(tmp, path)
    return True


def export(con, clean):
    written, keep = 0, set()
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT g.season_id FROM games g JOIN events e USING (game_id) ORDER BY 1")]
    for table, order in PARTITIONED.items():
        for s in seasons:
            path = clean / table / f"season={s}.parquet"
            sql = (f"SELECT t.* FROM {table} t JOIN games g ON g.game_id = t.game_id "
                   f"WHERE g.season_id = ? ORDER BY {order}")
            written += write_table(arrow_table(con, table, sql, (s,)), path)
            keep.add(path)
        for stale in (clean / table).glob("*.parquet"):
            if stale not in keep:
                stale.unlink()
    for table, order in SINGLE.items():
        sub = "quality" if table in QUALITY else ""
        path = clean / sub / f"{table}.parquet"
        written += write_table(arrow_table(con, table, f"SELECT * FROM {table} ORDER BY {order}"), path)
    return written


def main(raw, out, clean):
    raw, clean = Path(raw), Path(clean)
    if os.path.exists(out):
        os.remove(out)
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)
    col = Collector()
    tally = {"games": 0, "events": 0, "shifts": 0, "unresolved_shift_fields": 0}
    have_pbp = set()
    seasons = pd.read_parquet(raw / "seasons.parquet")
    sched = pd.concat([pd.read_parquet(p) for p in sorted((raw / "schedule").glob("season=*.parquet"),
                                                          key=season_key)], ignore_index=True)
    srows = season_rows(seasons)
    stype = {r["season_id"]: r["season_type"] for r in srows}
    ginfo = {to_int(r["game_id"]): (stype.get(to_int(r["season_id"])), to_int(r["home_team"]))
             for r in sched.to_dict("records")}
    n_strength = 0
    for p in sorted((raw / "payloads").glob("season=*.parquet"), key=season_key):
        df = pd.read_parquet(p)
        for gid, grp in df.groupby("game_id", sort=True):
            gid = int(gid)
            payload = dict(zip(grp["feed"], grp["payload"]))
            ev, eplayers, onice = (parse_events(gid, json.loads(payload["events"]), col)
                                   if "events" in payload else ([], [], []))
            sh = parse_shifts(gid, json.loads(payload["shifts"]), col) if "shifts" in payload else []
            have_pbp.add(gid)
            tally["games"] += 1
            tally["events"] += len(ev)
            tally["shifts"] += len(sh)
            tally["unresolved_shift_fields"] += sum(1 for r in sh if r["period"] is None or r["clock_in_s"] is None)
            insert(con, "event_players", eplayers)
            if gid in ginfo:
                roles = {}
                for _, i, pid, role in eplayers:
                    if role in ("penalty_served_by", "goalie_in", "goalie_out"):
                        roles.setdefault(i, {})["served_by" if role == "penalty_served_by" else role] = pid
                # skaters in the feed's plus/minus lists per goal (lists exclude goalies)
                pmn = {}
                for _, i, pid, pside, _src in onice:
                    pmn.setdefault(i, {"plus_n": 0, "minus_n": 0})[f"{pside}_n"] += 1
                evs = [{**e, **roles.get(e["event_idx"], {}), **pmn.get(e["event_idx"], {})} for e in ev]
                goalie_side = {pid: r.get("side") for (g, pid), r in col.rosters.items()
                               if g == gid and r.get("position") == "G"}
                tally["stuck_capped"] = tally.get("stuck_capped", 0) + cap_stuck_shifts(sh, goalie_side)
                st, chk = game_strength(gid, *ginfo[gid], evs, sh, goalie_side)
                insert(con, "strength", st)
                insert(con, "strength_check", chk)
                n_strength += len(st)
            if gid in ginfo:
                flag_mirrored_coords(ev, st, ginfo[gid][1])
                added = fill_goal_lists(ev, eplayers, onice, sh, goalie_side, ginfo[gid][0], ginfo[gid][1], st)
                onice += added
                tally["goal_list_players_filled"] = tally.get("goal_list_players_filled", 0) + len(added)
            insert(con, "event_on_ice", onice)
            insert(con, "events", ev)
            insert(con, "shifts", sh)  # after game_strength, which fills blank shift ends in place
    insert(con, "seasons", srows)
    insert(con, "games", game_rows(sched, have_pbp, col))
    insert(con, "players", [{"player_id": k, **v} for k, v in col.players.items()])
    insert(con, "teams", [{"team_id": k, **v} for k, v in col.teams.items()])
    insert(con, "game_rosters", [{"game_id": g, "player_id": p, **v} for (g, p), v in col.rosters.items()])
    con.commit()
    orphans = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT game_id FROM events) "
                          "WHERE game_id NOT IN (SELECT game_id FROM games)").fetchone()[0]
    if orphans:
        print(f"WARNING: {orphans} games have events but no schedule row; they are not exported")
    n = export(con, clean)
    tally["blank_end_filled"] = con.execute(
        "SELECT COUNT(*) FROM shifts WHERE end_inferred = 1 AND clock_in_s IS NOT NULL").fetchone()[0]
    print(f"{tally['games']} games -> {out} and {clean} ({n} Parquet files changed): {tally}")
    if tally["unresolved_shift_fields"]:
        print(f"note: {tally['unresolved_shift_fields']} shifts arrived without a usable start/end time; "
              f"{tally['blank_end_filled']} blank-end shifts got an inferred end (shifts.end_inferred = 1). "
              "The rest (empty 00:00 placeholders, or no room for another player) keep NULL clock columns "
              "and are ignored by the views.")
    con.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=root / "data" / "raw", help="raw Parquet store (from pwhl_ingest.py)")
    ap.add_argument("--out", default=root / "pwhl_clean.db", help="SQLite file to (re)build")
    ap.add_argument("--clean", default=root / "data" / "clean", help="folder for the clean Parquet export")
    a = ap.parse_args()
    main(a.raw, a.out, a.clean)
