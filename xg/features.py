"""
Shot features for the xG model, built from the clean Parquet export (data/clean).
train.py and score.py both call build(), so a model always sees features computed
exactly the way it was trained on.

One row per unblocked shot on goal: the feed's `shot` events (goals included,
is_goal = 1). The feed has no missed shots, so this is an xG model for shots on
goal, and blocked shots have no outcome to learn from.

Duplicates in the feed (normalize.py keeps them all):
  - each goal is a `shot` row (is_goal = 1, with shot_type) AND a `goal` row a few
    events later (same game, period, second and team; empty-net / penalty-shot flags,
    scorer, assists). The shot row is the shot; the goal row only adds its flags,
    matched on (game, period, second, team). tests/test_xg.py checks that it is 1:1;
  - each penalty shot is a `penaltyshot` event (is_goal set, no coordinates), plus a
    `goal` row when it scores. Penalty shots come from the `penaltyshot` events
    (penalty_shots()) and get a fixed xG; they are not model rows.
Goal rows are left out of the "previous event" sequence (otherwise every goal would
be the previous event of the next shot).

Coordinates: x 0-600 along the rink, y 0-300 across, absolute (not per team). A team
shoots at one end per period. The end is where most of its shots that period are,
as in normalize.flag_mirrored_coords. Shots flagged coords_mirrored are flipped back.
Every shot is then put into one quadrant: x_from_goal_ft out from the goal line of
the net it attacks, y_abs_ft sideways from the centre line. Which side of the ice it
came from only enters through the shooter's hand (off_hand) and the goalie's
glove side (glove_side), and through whether the previous event was on the other
side (royal_road).
Side convention, checked on 2026-11-25 from where left and right defenders shoot
from (LD point shots average 45 units to one side, RD the other): attacking the far
net (x > 300), the shooter's left is y < 150; attacking the near net, y > 150.

xG source (`xg_source`):
  model         scored by the model (trained on regular season + playoffs)
  empty_net     goal flagged empty-net, or goal with the defending goalie out: xG 1.0
                (saved shots are never empty-net; see add_features)
  penalty_shot  xG 0.33
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
sys.path.insert(0, str(ROOT / "ingest"))
from strength import period_len  # noqa: E402  (same period lengths as the strength table)

# Rink geometry in feed units: 600 x 300 units for an NHL-size 200 x 85 ft rink (PWHL
# rinks are NHL size), goal line 11 ft from the end boards. Faceoff-dot positions in
# the feed only roughly match a real rink: the feed's drawing is not exactly to scale.
X_PER_FT = 600 / 200
Y_PER_FT = 300 / 85
GOAL_LINE_FT = 11
Y_CENTRE = 150

TRAIN_SEASON_TYPES = ("regular", "playoffs")
XG_EMPTY_NET = 1.0
XG_PENALTY_SHOT = 0.33

# Tuned with xg/experiments.py (see that file for the results).
REBOUND_S = (3,)          # one is_rebound_<n>s flag per window: same team's shot on goal <= n s before
ROYAL_ROAD_DEG = 20       # crossed = previous event on the other side AND angle change >= this
CLOCK_RESOLUTION_S = 1    # times are whole seconds: speed uses at least 1 s

# Fixed category levels. Never infer them from the data, or a season missing a
# level would shift the codes the saved model was trained on. New values in the
# feed become NaN (XGBoost treats them as missing); tests flag them.
CATEGORICAL = {
    "shot_type": ["Wrist", "Snap", "Slap", "Backhand", "Tip", "Default"],
    "prev_event": ["faceoff", "hit", "penalty", "goalie_change", "shot", "blocked_shot",
                   "penaltyshot", "none"],
    "royal_road": ["crossed", "not_crossed", "none"],
}


def feature_names():
    return [
        "distance_ft", "angle_deg", "x_from_goal_ft", "y_abs_ft", "behind_net",
        "shot_type", "off_hand", "glove_side",
        "skaters_for", "skaters_against", "own_goalie_pulled",
        "period", "time_remaining_s", "score_diff",
        "prev_event", "prev_same_team", "secs_since_prev", "prev_dist_ft", "prev_to_shot_speed_ft_s",
        "royal_road", *[f"is_rebound_{n}s" for n in REBOUND_S],
    ]


FEATURES = feature_names()
ID_COLS = ["game_id", "event_idx", "goal_event_idx", "season_id", "season_year", "season_type", "period",
           "period_time_s", "team_id", "shooter_id", "goalie_id", "is_goal", "xg_source"]
# Built and returned by build(), but NOT model features. Shift lengths leak the outcome
# through missingness: the shift data is more complete around goals (within a game,
# shots with no shooter shift found score 4 points less often at the same distance /
# rebound rate). With NaN kept, CV improves only because the model learns "no shift
# data -> fewer goals"; with missing values imputed at random they make it worse
# (xg/experiments.py, 2026-11-25).
INFO_COLS = ["shooter_shift_s", "opp_avg_shift_s"]


def _src(name):
    p = CLEAN / name
    if p.is_dir():
        return f"read_parquet('{p.as_posix()}/*.parquet', hive_partitioning = false)"
    return f"read_parquet('{p.as_posix()}.parquet')"


SQL = """
WITH games AS (
    SELECT g.game_id, g.season_id, g.home_team_id, g.visiting_team_id, s.season_type
    FROM {games} g JOIN {seasons} s USING (season_id)
), seq AS (           -- the event sequence, without the duplicate goal rows and shootout
    SELECT e.*,
           lag(event)         OVER w AS prev_event_raw,
           lag(ev_team)       OVER w AS prev_team_id,
           lag(period_time_s) OVER w AS prev_t,
           lag(x)             OVER w AS prev_x,
           lag(y)             OVER w AS prev_y
    FROM (
        -- faceoffs carry no team_id; they belong to the team that won them (home_win)
        SELECT e.*, CASE WHEN e.event = 'faceoff' AND e.home_win IS NOT NULL
                         THEN CASE WHEN e.home_win = 1 THEN g.home_team_id ELSE g.visiting_team_id END
                         ELSE e.team_id END AS ev_team
        FROM {events} e JOIN games g USING (game_id)
        WHERE e.event NOT IN ('goal', 'shootout') AND e.period IS NOT NULL
    ) e
    WINDOW w AS (PARTITION BY game_id, period ORDER BY event_idx)
), shots AS (
    SELECT * FROM seq WHERE event = 'shot' AND x IS NOT NULL AND period_time_s IS NOT NULL
), goal_rows AS (
    SELECT game_id, event_idx AS goal_event_idx, period, period_time_s, team_id,
           is_empty_net, is_penalty_shot
    FROM {events} WHERE event = 'goal'
), ends AS (          -- the end each team attacks, per period
    SELECT game_id, period, team_id, avg((x > 300)::INT) > 0.5 AS att_far
    FROM shots GROUP BY ALL
), score AS (         -- goals scored by each team before each shot (shootout excluded)
    SELECT s.game_id, s.event_idx,
           count(*) FILTER (WHERE g.team_id = s.team_id)  AS goals_for,
           count(*) FILTER (WHERE g.team_id <> s.team_id) AS goals_against
    FROM shots s LEFT JOIN {events} g
      ON g.game_id = s.game_id AND g.event = 'goal' AND g.event_idx < s.event_idx
    GROUP BY ALL
), players AS (
    SELECT game_id, event_idx,
           max(player_id) FILTER (WHERE role = 'shooter') AS shooter_id,
           max(player_id) FILTER (WHERE role = 'goalie')  AS goalie_id
    FROM {event_players} GROUP BY ALL
)
SELECT s.game_id, s.event_idx, gr.goal_event_idx, ga.season_id, ga.season_type,
       s.period, s.period_time_s, s.team_id, p.shooter_id, p.goalie_id,
       coalesce(s.is_goal, 0) AS is_goal,
       CASE WHEN s.coords_mirrored = 1 THEN 600 - s.x ELSE s.x END AS x,
       s.y, en.att_far, s.shot_type,
       sp.shoots AS shooter_shoots, gp.catches AS goalie_catches,
       -- strength.state = skaters on the ice (manpower + an extra attacker when the goalie
       -- is pulled), 'home v visitor'; *_skaters columns are manpower only
       CAST(split_part(st.state, 'v', CASE WHEN s.team_id = ga.home_team_id THEN 1 ELSE 2 END) AS INT) AS skaters_for,
       CAST(split_part(st.state, 'v', CASE WHEN s.team_id = ga.home_team_id THEN 2 ELSE 1 END) AS INT) AS skaters_against,
       CASE WHEN s.team_id = ga.home_team_id THEN st.home_goalie_in ELSE st.visitor_goalie_in END AS own_goalie_in,
       CASE WHEN s.team_id = ga.home_team_id THEN st.visitor_goalie_in ELSE st.home_goalie_in END AS opp_goalie_in,
       coalesce(gr.is_empty_net, 0) AS goal_empty_net,
       coalesce(gr.is_penalty_shot, 0) AS goal_penalty_shot,
       sc.goals_for - sc.goals_against AS score_diff,
       s.prev_event_raw, s.prev_team_id, s.period_time_s - s.prev_t AS secs_since_prev,
       s.prev_x, s.prev_y
FROM shots s
JOIN games ga USING (game_id)
JOIN ends en USING (game_id, period, team_id)
JOIN score sc USING (game_id, event_idx)
LEFT JOIN players p USING (game_id, event_idx)
LEFT JOIN {players_tbl} sp ON sp.player_id = p.shooter_id
LEFT JOIN {players_tbl} gp ON gp.player_id = p.goalie_id
LEFT JOIN goal_rows gr
       ON s.is_goal = 1 AND gr.game_id = s.game_id AND gr.period = s.period
      AND gr.period_time_s = s.period_time_s AND gr.team_id = s.team_id
-- in-play events at second t take the state with start_s < t <= end_s (strength.py);
-- a shot at 0:00 takes the first state
LEFT JOIN {strength} st
       ON st.game_id = s.game_id AND st.period = s.period AND s.period_time_s <= st.end_s
      AND (st.start_s < s.period_time_s OR (s.period_time_s = 0 AND st.start_s = 0))
ORDER BY s.game_id, s.event_idx
"""

PENALTY_SHOT_SQL = """
SELECT e.game_id, e.event_idx, gr.goal_event_idx, g.season_id, s.season_type, e.period,
       e.period_time_s, e.team_id, p.shooter_id, p.goalie_id, coalesce(e.is_goal, 0) AS is_goal
FROM {events} e
JOIN {games} g USING (game_id) JOIN {seasons} s USING (season_id)
LEFT JOIN (SELECT game_id, event_idx,
                  max(player_id) FILTER (WHERE role = 'shooter') AS shooter_id,
                  max(player_id) FILTER (WHERE role = 'goalie')  AS goalie_id
           FROM {event_players} GROUP BY ALL) p USING (game_id, event_idx)
LEFT JOIN (SELECT game_id, event_idx AS goal_event_idx, period, period_time_s, team_id
           FROM {events} WHERE event = 'goal' AND is_penalty_shot = 1) gr
       ON e.is_goal = 1 AND gr.game_id = e.game_id AND gr.period = e.period
      AND gr.period_time_s = e.period_time_s AND gr.team_id = e.team_id
WHERE e.event = 'penaltyshot'
ORDER BY e.game_id, e.event_idx
"""
# Time into the current shift at each shot, from the shift data. Same rules as
# normalize.py's event_on_ice_derived view: c = countdown clock at the shot (period
# length - elapsed; period length 1200, overtime = the largest shift start, as in the
# period_len view); for an in-play event a player is on when clock_out_s <= c <
# clock_in_s (one finishing a shift at c is on, one who just arrived is not), for the
# shooter too. 319 shots (2026-11-25) have the shooter's shift starting in the very
# second of the shot, 13% of them goals: taken as shift times snapped to the goal /
# stoppage, i.e. a data error, so those shots count as not found. Time into the shift = clock_in_s - c. Goalies (roster position G) are left out of the
# opponents. Games with games.exclude_shifts = 1 get NULL. If the shooter has two
# overlapping shifts, the later start counts. Not found (shift data gaps, ~10% of
# shots) stays NULL; shifts > 3 min (a few dozen shots, data errors) are left as is.
SHIFT_SQL = """
WITH sh AS (
    SELECT s.* FROM {shifts} s JOIN {games} g USING (game_id)
    WHERE g.exclude_shifts = 0 AND s.clock_in_s IS NOT NULL AND s.clock_out_s IS NOT NULL
), plen AS (
    SELECT game_id, period, CASE WHEN period <= 3 THEN 1200 ELSE max(clock_in_s) END AS len
    FROM {shifts} GROUP BY game_id, period
), goalies AS (
    SELECT DISTINCT game_id, player_id FROM {game_rosters} WHERE position = 'G'
), shots AS (
    SELECT e.game_id, e.event_idx, e.period, pl.len - e.period_time_s AS c, p.player_id AS shooter_id,
           CASE WHEN e.team_id = g.home_team_id THEN 'home' ELSE 'visitor' END AS side
    FROM {events} e
    JOIN {games} g USING (game_id)
    JOIN plen pl ON pl.game_id = e.game_id AND pl.period = e.period
    LEFT JOIN {event_players} p ON p.game_id = e.game_id AND p.event_idx = e.event_idx AND p.role = 'shooter'
    WHERE e.event = 'shot' AND e.period_time_s IS NOT NULL AND g.exclude_shifts = 0
), shooter AS (
    SELECT q.game_id, q.event_idx, min(s.clock_in_s - q.c) AS shooter_shift_s
    FROM shots q JOIN sh s ON s.game_id = q.game_id AND s.period = q.period
     AND s.player_id = q.shooter_id AND s.clock_out_s <= q.c AND q.c < s.clock_in_s
    GROUP BY ALL
), opp AS (
    SELECT q.game_id, q.event_idx, avg(s.clock_in_s - q.c) AS opp_avg_shift_s, count(*) AS opp_skaters_found
    FROM shots q JOIN sh s ON s.game_id = q.game_id AND s.period = q.period AND s.side <> q.side
     AND s.clock_out_s <= q.c AND q.c < s.clock_in_s
    WHERE NOT EXISTS (SELECT 1 FROM goalies gk WHERE gk.game_id = s.game_id AND gk.player_id = s.player_id)
    GROUP BY ALL
)
SELECT q.game_id, q.event_idx, shooter.shooter_shift_s, opp.opp_avg_shift_s, opp.opp_skaters_found
FROM shots q LEFT JOIN shooter USING (game_id, event_idx) LEFT JOIN opp USING (game_id, event_idx)
"""
SOURCES = ("games", "seasons", "events", "event_players", "strength", "shifts", "game_rosters")


def season_years():
    """season_id -> season year (the year the regular season ends: 2024 = the inaugural
    2024 season incl. its preseason and playoffs, 2025 = 2024-25, ...). A season that
    ends after August (a preseason in Nov/Dec) belongs to the next year's season."""
    s = pd.read_parquet(CLEAN / "seasons.parquet")
    end = pd.to_datetime(s["end_date"])
    return dict(zip(s["season_id"], np.where(end.dt.month <= 8, end.dt.year, end.dt.year + 1)))


def load_raw(con=None):
    """One row per shot with the raw columns the features are made from."""
    con = con or duckdb.connect()
    src = {n: _src(n) for n in SOURCES}
    df = con.sql(SQL.format(**src, players_tbl=_src("players"))).df()
    sh = con.sql(SHIFT_SQL.format(**src)).df()
    df = df.merge(sh, on=["game_id", "event_idx"], how="left", validate="one_to_one")
    df["season_year"] = df["season_id"].map(season_years()).astype(int)
    return df


def lateral_ft(y, att_far):
    """Sideways distance from the centre line in feet, positive = the shooter's left
    (see the side convention in the module docstring)."""
    return np.where(att_far, Y_CENTRE - y, y - Y_CENTRE) / Y_PER_FT


def add_features(df):
    """Raw shot rows -> model features (in place; also returns df)."""
    goal_x = np.where(df["att_far"], 600 - GOAL_LINE_FT * X_PER_FT, GOAL_LINE_FT * X_PER_FT)

    def out_from_goal(x):  # feet out from the goal line (negative = behind the net)
        return np.where(df["att_far"], goal_x - x, x - goal_x) / X_PER_FT

    # one quadrant
    df["x_from_goal_ft"] = out_from_goal(df["x"])
    lat = lateral_ft(df["y"], df["att_far"])
    df["y_abs_ft"] = np.abs(lat)
    df["distance_ft"] = np.hypot(df["x_from_goal_ft"], df["y_abs_ft"])
    df["angle_deg"] = np.degrees(np.arctan2(df["y_abs_ft"], df["x_from_goal_ft"]))  # 0 = straight on
    df["behind_net"] = (df["x_from_goal_ft"] < 0).astype(int)

    # side of the ice, relative to the shooter's stick and the goalie's glove. A shot
    # exactly on the centre line has no side (NaN), as does an unknown hand.
    side = np.sign(lat)                                  # +1 shooter's left, -1 right, 0 centre
    shoots = df["shooter_shoots"].map({"L": 1.0, "R": -1.0})
    df["off_hand"] = np.where(side == 0, np.nan, (side != shoots).astype(float))
    df.loc[shoots.isna(), "off_hand"] = np.nan
    # a goalie who catches L has the glove on their left = the shooter's right
    glove = df["goalie_catches"].map({"L": -1.0, "R": 1.0})
    df["glove_side"] = np.where(side == 0, np.nan, (side == glove).astype(float))
    df.loc[glove.isna(), "glove_side"] = np.nan

    df["own_goalie_pulled"] = np.where(df["own_goalie_in"].isna(), np.nan, (df["own_goalie_in"] == 0).astype(float))
    plen = [period_len(p, t) for p, t in zip(df["period"], df["season_type"])]
    df["time_remaining_s"] = (np.array(plen) - df["period_time_s"]).clip(lower=0).astype(float)

    df["prev_event"] = df["prev_event_raw"].fillna("none")
    df["prev_same_team"] = np.where(df["prev_team_id"].isna(), np.nan,
                                    (df["prev_team_id"] == df["team_id"]).astype(float))
    df["prev_dist_ft"] = np.hypot((df["x"] - df["prev_x"]) / X_PER_FT, (df["y"] - df["prev_y"]) / Y_PER_FT)
    df["prev_to_shot_speed_ft_s"] = df["prev_dist_ft"] / df["secs_since_prev"].clip(lower=CLOCK_RESOLUTION_S)

    # royal road: previous event on the other side of the ice, measured against the
    # net this shot attacks, with the angle (signed, 0 = straight on) changing by at
    # least ROYAL_ROAD_DEG
    prev_lat = lateral_ft(df["prev_y"], df["att_far"])
    ang = np.degrees(np.arctan2(lat, df["x_from_goal_ft"]))
    prev_ang = np.degrees(np.arctan2(prev_lat, out_from_goal(df["prev_x"])))
    crossed = (np.sign(lat) * np.sign(prev_lat) < 0) & (np.abs(ang - prev_ang) >= ROYAL_ROAD_DEG)
    df["royal_road"] = np.where(df["prev_x"].isna() | df["secs_since_prev"].isna(), "none",
                                np.where(crossed, "crossed", "not_crossed"))

    for n in REBOUND_S:
        df[f"is_rebound_{n}s"] = ((df["prev_event"] == "shot") & (df["prev_same_team"] == 1)
                                  & (df["secs_since_prev"] <= n)).astype(int)

    for col, levels in CATEGORICAL.items():
        df[col] = pd.Categorical(df[col], categories=levels)
    for col in ("skaters_for", "skaters_against", "score_diff", "secs_since_prev", "period",
                "prev_to_shot_speed_ft_s", "prev_dist_ft", "shooter_shift_s", "opp_avg_shift_s"):
        df[col] = df[col].astype("float")

    # Empty net = a goal flagged empty-net, or a goal with the defending goalie out in the
    # strength table. A SAVED shot is never empty-net: the feed has no missed shots, so
    # a shot on goal that didn't score had a goalie in. (2 saved shots fall in a "goalie
    # out" interval on 2026-11-25: delayed-penalty pulls whose goalie_in is logged in the
    # same second as the save, after it in feed order; games 165 and 208.)
    empty = (df["goal_empty_net"] == 1) | ((df["opp_goalie_in"] == 0) & (df["is_goal"] == 1))
    df["xg_source"] = np.where(df["goal_penalty_shot"] == 1, "penalty_shot",
                               np.where(empty, "empty_net", "model"))
    return df


def build(con=None):
    """Shot table: ID_COLS + FEATURES, one row per shot on goal (goals included)."""
    df = add_features(load_raw(con))
    cols = ID_COLS + [f for f in feature_names() + INFO_COLS if f not in ID_COLS]  # period is both
    return df[cols].reset_index(drop=True)


def penalty_shots(con=None):
    """One row per penalty shot (the feed's `penaltyshot` events), ID_COLS only."""
    con = con or duckdb.connect()
    df = con.sql(PENALTY_SHOT_SQL.format(**{n: _src(n) for n in SOURCES})).df()
    df["xg_source"] = "penalty_shot"
    df["season_year"] = df["season_id"].map(season_years()).astype(int)
    return df[ID_COLS]


def train_mask(df):
    return (df["xg_source"] == "model") & df["season_type"].isin(TRAIN_SEASON_TYPES)


if __name__ == "__main__":  # quick look: python xg/features.py
    d = build()
    print(d.describe(include="all").T.to_string())
    print(f"\n{len(d)} shots, {int(d['is_goal'].sum())} goals, {d['xg_source'].value_counts().to_dict()}, "
          f"{int(train_mask(d).sum())} in training set; {len(penalty_shots())} penalty shots")
