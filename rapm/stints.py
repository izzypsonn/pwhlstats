"""
Stints for RAPM: maximal stretches of a period with the same skaters on the ice for
both teams and the same strength row, also split at every faceoff. Built from
data/clean (shifts, strength, events, games) and data/xg/shot_xg.parquet; no API calls.

Which games: final games of regular-season and playoff seasons with shift data,
except games.exclude_shifts = 1. Preseason is left out. Seasons are grouped by
season year (features.season_years: 2024 = inaugural regular season + playoffs, ...).

Time: elapsed seconds in the period, as in the strength table. A shift covers
(a, b] with a = L - clock_in_s, b = L - clock_out_s (the shift clock counts down),
L = strength.period_len (1200; regular-season / preseason OT 300). That is the
in-play convention used for event_on_ice_derived and the xG shift features: a player
finishing a shift at t is on for an event at t, one arriving at t is not. A stint
(start_s, end_s] gets the shots with start_s < t <= end_s (the first stint of a
period also takes t = 0), matching how the strength table assigns in-play events.
A faceoff at t is a stoppage event and belongs to the stint starting at t.

Boundaries are every shift start/end, every strength row boundary and every
faceoff. Overtime is cut at the first goal of the period (sudden death): the
strength table runs to the end of the period regardless.

Goalies (roster position G) are not in the skater lists. Skater lists are taken from
the shifts as they are (no repair): a stint may list more or fewer skaters than the
strength state says (n_home / n_visitor vs state). Stretches with nobody on the ice
for a side (shift data gaps) are kept; that side then has no player columns.

Shots: shot_xg rows with xg_source model or empty_net (penalty shots are left out).

Context at the start of each stint, from the home team's side (fit.py turns it
into the attacking team's side):
  score_home        home goals - visitor goals, counting goals at or before start_s
                    (the stint starting at a goal's second is after the goal). From
                    the `goal` rows; shootout goals are not goal rows.
  start_home        how the stint began: 'oz' / 'dz' / 'nz' = with a faceoff in the
                    home team's offensive / defensive / neutral zone, 'otf' = no
                    faceoff at start_s (change on the fly, penalty expiring, ...).
                    Zone from the faceoff's x: the feed puts zone dots at x ~100-140
                    and ~460-500, neutral-zone dots at ~200-250 / ~350-400, centre
                    300; x < ZONE_X_NEAR or x > ZONE_X_FAR is in a zone. Which end a
                    team attacks in a period is where most of its shots that period
                    are (as in xg/features.py); if a team has no shots that period,
                    the opposite of the other team's; neither -> 'otf'.
  home_b2b / visitor_b2b
                    1 if the team played a final game (any season type, preseason
                    included) the day before (games.date_played).
  pen_exp_home / pen_exp_visitor
                    1 if the stint starts where that team's penalty expired: its
                    manpower (strength.<side>_skaters) goes up at start_s compared
                    with the previous strength row of the period, with no goal at
                    that second (a goal ending a penalty is not an expiry).
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xg"))
sys.path.insert(0, str(ROOT / "ingest"))
import features as F  # noqa: E402  (season_years, _src)
from strength import period_len  # noqa: E402

SEASON_TYPES = ("regular", "playoffs")
SHOT_XG = ROOT / "data" / "xg" / "shot_xg.parquet"
ZONE_X_NEAR, ZONE_X_FAR = 175, 425
STINT_COLS = ["game_id", "season_id", "season_year", "season_type", "period", "start_s", "end_s",
              "duration_s", "state", "home_goalie_in", "visitor_goalie_in", "home_team_id",
              "visitor_team_id", "home_players", "visitor_players", "n_home", "n_visitor",
              "score_home", "start_home", "home_b2b", "visitor_b2b", "pen_exp_home", "pen_exp_visitor",
              "home_xg", "visitor_xg", "home_goals", "visitor_goals", "home_shots", "visitor_shots"]


def load(con=None):
    con = con or duckdb.connect()
    src = {n: F._src(n) for n in ("games", "seasons", "shifts", "strength", "game_rosters", "events")}
    games = con.sql(f"""
        SELECT g.game_id, g.season_id, s.season_type, g.home_team_id, g.visiting_team_id AS visitor_team_id
        FROM {src['games']} g JOIN {src['seasons']} s USING (season_id)
        WHERE s.season_type IN {SEASON_TYPES} AND g.final = 1 AND g.exclude_shifts = 0
          AND EXISTS (SELECT 1 FROM {src['shifts']} sh WHERE sh.game_id = g.game_id)
    """).df()
    ids = tuple(games["game_id"].tolist()) or (-1,)
    d = {"games": games}
    d["shifts"] = con.sql(f"""
        SELECT sh.game_id, sh.player_id, sh.side, sh.period, sh.clock_in_s, sh.clock_out_s
        FROM {src['shifts']} sh
        WHERE sh.game_id IN {ids} AND sh.clock_in_s IS NOT NULL AND sh.clock_out_s IS NOT NULL
          AND sh.period IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM {src['game_rosters']} r
                          WHERE r.game_id = sh.game_id AND r.player_id = sh.player_id AND r.position = 'G')
    """).df()
    d["strength"] = con.sql(f"SELECT * FROM {src['strength']} WHERE game_id IN {ids}").df()
    d["goals"] = con.sql(f"""
        SELECT game_id, period, period_time_s AS t, team_id FROM {src['events']}
        WHERE event = 'goal' AND period IS NOT NULL AND period_time_s IS NOT NULL AND game_id IN {ids}
    """).df()
    d["faceoffs"] = con.sql(f"""
        SELECT game_id, period, period_time_s AS t, x FROM {src['events']}
        WHERE event = 'faceoff' AND period IS NOT NULL AND period_time_s IS NOT NULL AND game_id IN {ids}
        QUALIFY row_number() OVER (PARTITION BY game_id, period, period_time_s ORDER BY event_idx DESC) = 1
    """).df()
    d["ends"] = con.sql(f"""
        SELECT game_id, period, team_id, avg((x > 300)::INT) > 0.5 AS att_far FROM {src['events']}
        WHERE event = 'shot' AND x IS NOT NULL AND period IS NOT NULL AND game_id IN {ids}
        GROUP BY ALL
    """).df()
    d["b2b"] = con.sql(f"""
        WITH t AS (SELECT game_id, date_played::DATE AS d, home_team_id AS team_id FROM {src['games']} WHERE final = 1
                   UNION ALL
                   SELECT game_id, date_played::DATE, visiting_team_id FROM {src['games']} WHERE final = 1)
        SELECT a.game_id, a.team_id FROM t a
        WHERE EXISTS (SELECT 1 FROM t b WHERE b.team_id = a.team_id AND b.d = a.d - 1)
    """).df()
    shots = pd.read_parquet(SHOT_XG)
    d["shots"] = shots[shots["game_id"].isin(games["game_id"]) & shots["xg_source"].isin(["model", "empty_net"])]
    return d


def period_stints(sh, st, end, L, faceoff_t=()):
    """Stints of one period. sh: that period's skater shifts; st: its strength rows;
    end: where the period stops (L, or the OT winner); faceoff_t: faceoff times
    (each starts a new stint). Returns a list of dicts."""
    a = np.clip(L - sh["clock_in_s"].to_numpy(), 0, end)
    b = np.clip(L - sh["clock_out_s"].to_numpy(), 0, end)
    side = sh["side"].to_numpy()
    pid = sh["player_id"].to_numpy()
    st = st[st["start_s"] < end].sort_values("start_s")
    fo = set(int(t) for t in faceoff_t)
    cuts = np.unique(np.concatenate([[0, end], a, b, st["start_s"], np.minimum(st["end_s"], end),
                                     [t for t in fo if 0 <= t <= end]]))
    cuts = cuts[(cuts >= 0) & (cuts <= end)]
    s_start, s_end = st["start_s"].to_numpy(), st["end_s"].to_numpy()
    out, prev = [], None
    for t0, t1 in zip(cuts[:-1], cuts[1:]):
        on = (a <= t0) & (b >= t1)
        home = tuple(sorted(set(pid[on & (side == "home")].tolist())))
        vis = tuple(sorted(set(pid[on & (side == "visitor")].tolist())))
        k = np.flatnonzero((s_start <= t0) & (s_end >= t1))
        row = st.iloc[k[0]] if len(k) else None
        key = (home, vis, None if row is None else (row["state"], row["home_goalie_in"], row["visitor_goalie_in"]))
        if prev is not None and prev["key"] == key and prev["end_s"] == t0 and int(t0) not in fo:
            prev["end_s"] = int(t1)
            continue
        prev = {"key": key, "start_s": int(t0), "end_s": int(t1), "home_players": list(home),
                "visitor_players": list(vis),
                "state": None if row is None else row["state"],
                "home_goalie_in": None if row is None else int(row["home_goalie_in"]),
                "visitor_goalie_in": None if row is None else int(row["visitor_goalie_in"])}
        out.append(prev)
    return out


def build(con=None):
    """One row per stint (STINT_COLS), all seasons."""
    d = load(con)
    games, shifts, strength = d["games"], d["shifts"], d["strength"]
    years = F.season_years()
    ot = d["goals"][d["goals"]["period"] > 3].groupby(["game_id", "period"])["t"].min()
    sh_by = dict(tuple(shifts.groupby(["game_id", "period"])))
    st_by = dict(tuple(strength.groupby(["game_id", "period"])))
    fo_by = d["faceoffs"].groupby(["game_id", "period"])["t"].apply(list).to_dict()
    rows = []
    for g in games.itertuples():
        for p in sorted(strength.loc[strength["game_id"] == g.game_id, "period"].unique()):
            L = period_len(p, g.season_type)
            end = min(L, ot.get((g.game_id, p), L))
            sh = sh_by.get((g.game_id, p), shifts.iloc[:0])
            for s in period_stints(sh, st_by[(g.game_id, p)], end, L, fo_by.get((g.game_id, p), ())):
                s.pop("key")
                rows.append({"game_id": g.game_id, "season_id": g.season_id,
                             "season_year": int(years[g.season_id]), "season_type": g.season_type,
                             "period": int(p), "home_team_id": g.home_team_id,
                             "visitor_team_id": g.visitor_team_id, **s})
    st = pd.DataFrame(rows)
    st["duration_s"] = st["end_s"] - st["start_s"]
    st["n_home"] = st["home_players"].str.len()
    st["n_visitor"] = st["visitor_players"].str.len()
    st = add_score(st, d["goals"])
    st = add_start(st, d["faceoffs"], d["ends"])
    st = add_b2b(st, d["b2b"])
    st = add_penalty_expiry(st, strength, d["goals"])
    st = add_shots(st, d["shots"])
    return st[STINT_COLS].sort_values(["game_id", "period", "start_s"]).reset_index(drop=True)


def add_score(st, goals):
    """score_home: home - visitor goals at or before (period, start_s)."""
    st = st.copy()
    g = goals.merge(st[["game_id", "home_team_id"]].drop_duplicates(), on="game_id")
    g["d"] = np.where(g["team_id"] == g["home_team_id"], 1, -1)
    g["k"] = g["period"] * 100000 + g["t"]
    g = g.sort_values(["game_id", "k"])
    by = {k: (v["k"].to_numpy(), np.cumsum(v["d"].to_numpy())) for k, v in g.groupby("game_id")}
    key = (st["period"] * 100000 + st["start_s"]).to_numpy()
    score = np.zeros(len(st), dtype=int)
    for gid, idx in st.groupby("game_id").indices.items():
        if gid in by:
            ks, cs = by[gid]
            j = np.searchsorted(ks, key[idx], side="right")
            score[idx] = np.where(j > 0, cs[np.maximum(j - 1, 0)], 0)
    st["score_home"] = score
    return st


def faceoff_zone(x, home_att_far):
    """'oz' / 'dz' / 'nz' for the home team from a faceoff's x (NaN att_far -> None)."""
    if x > ZONE_X_FAR:
        far = True
    elif x < ZONE_X_NEAR:
        far = False
    else:
        return "nz"
    if home_att_far is None or pd.isna(home_att_far):
        return None
    return "oz" if far == bool(home_att_far) else "dz"


def add_start(st, faceoffs, ends):
    """start_home: zone of a faceoff at start_s for the home team, else 'otf'."""
    st = st.copy()
    e = ends.merge(st[["game_id", "home_team_id", "visitor_team_id"]].drop_duplicates(), on="game_id")
    home = e[e["team_id"] == e["home_team_id"]].set_index(["game_id", "period"])["att_far"]
    vis = e[e["team_id"] == e["visitor_team_id"]].set_index(["game_id", "period"])["att_far"]
    att = pd.concat([home.rename("h"), (~vis.astype(bool)).rename("v")], axis=1)
    att = att["h"].astype(object).where(att["h"].notna(), att["v"]).rename("home_att_far").reset_index()
    f = faceoffs.rename(columns={"t": "start_s"}).merge(att, on=["game_id", "period"], how="left")
    f["start_home"] = [faceoff_zone(x, a) for x, a in zip(f["x"], f["home_att_far"])]
    st = st.merge(f[["game_id", "period", "start_s", "start_home"]], on=["game_id", "period", "start_s"], how="left")
    st["start_home"] = st["start_home"].fillna("otf")
    return st


def add_b2b(st, b2b):
    st = st.copy()
    pairs = set(zip(b2b["game_id"], b2b["team_id"]))
    st["home_b2b"] = [int((g, t) in pairs) for g, t in zip(st["game_id"], st["home_team_id"])]
    st["visitor_b2b"] = [int((g, t) in pairs) for g, t in zip(st["game_id"], st["visitor_team_id"])]
    return st


def add_penalty_expiry(st, strength, goals):
    """pen_exp_<side>: the stint starts where that side's manpower goes up (vs the
    previous strength row of the period) with no goal at that second."""
    st = st.copy()
    s = strength.sort_values(["game_id", "period", "start_s"]).copy()
    grp = s.groupby(["game_id", "period"])
    goal_at = set(zip(goals["game_id"], goals["period"], goals["t"]))
    no_goal = np.array([(g, p, t) not in goal_at for g, p, t in zip(s["game_id"], s["period"], s["start_s"])])
    for side in ("home", "visitor"):
        prev = grp[f"{side}_skaters"].shift()
        s[f"pen_exp_{side}"] = ((s[f"{side}_skaters"] > prev) & no_goal & (s["start_s"] > 0)).astype(int)
    s = s[["game_id", "period", "start_s", "pen_exp_home", "pen_exp_visitor"]]
    st = st.merge(s, on=["game_id", "period", "start_s"], how="left")
    st[["pen_exp_home", "pen_exp_visitor"]] = st[["pen_exp_home", "pen_exp_visitor"]].fillna(0).astype(int)
    return st


def add_shots(st, shots):
    """Sum xG, goals and shots per side into the stint holding each shot
    (start_s < t <= end_s; t = 0 goes to the first stint)."""
    st = st.sort_values(["game_id", "period", "start_s"]).reset_index(drop=True)
    for side in ("home", "visitor"):
        for c in ("xg", "goals", "shots"):
            st[f"{side}_{c}"] = 0.0 if c == "xg" else 0
    pos = {k: (v.index.to_numpy(), v["end_s"].to_numpy()) for k, v in st.groupby(["game_id", "period"])}
    home_team = st.groupby("game_id")["home_team_id"].first()
    hits = []
    for r in shots.itertuples():
        k = (r.game_id, int(r.period))
        if k not in pos:
            continue
        idx, ends = pos[k]
        j = np.searchsorted(ends, r.period_time_s, side="left")
        if j < len(idx):
            hits.append((idx[j], "home" if r.team_id == home_team[r.game_id] else "visitor", r.xg, r.is_goal))
    if hits:
        h = pd.DataFrame(hits, columns=["row", "side", "xg", "goal"])
        agg = h.groupby(["row", "side"]).agg(xg=("xg", "sum"), goals=("goal", "sum"), shots=("xg", "size"))
        for side in ("home", "visitor"):
            a = agg.xs(side, level="side")
            st.loc[a.index, f"{side}_xg"] = a["xg"].to_numpy()
            st.loc[a.index, f"{side}_goals"] = a["goals"].to_numpy().astype(int)
            st.loc[a.index, f"{side}_shots"] = a["shots"].to_numpy().astype(int)
    return st


if __name__ == "__main__":
    s = build()
    print(s.groupby(["season_year", "state"]).agg(stints=("game_id", "size"), hours=("duration_s", lambda d: d.sum() / 3600),
                                                  xg=("home_xg", "sum")).round(1).to_string())
