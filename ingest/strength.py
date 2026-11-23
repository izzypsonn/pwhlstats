"""
Strength state for every moment of every game, plus a per-second check of the
shift data against it. Used by normalize.py; no API calls.

Manpower comes from the penalty events (it does not depend on shift data):
  - each team has 5 skaters (3 in regular-season / preseason overtime);
  - a penalty with is_power_play = 1 and 2 or 5 minutes puts a player in the box.
    is_power_play = 0 marks coincidental penalties, which don't change manpower;
    10/20-minute misconducts never do;
  - at most 2 players per team serve at once; further penalties wait in a queue
    and start when a slot frees. A player's own penalties run one after another
    (double minors arrive either as two 2-minute penalties at the same time, or
    as two minors with the second timed exactly 2:00 later; the latter is re-timed
    so the second part starts as soon as the first ends, early or not);
  - a minor ends early on a goal while the teams are at unequal manpower.
    PWHL "jailbreak" rule: this happens when EITHER team scores, not just the
    power-play team; the shorthanded team's minor that would expire first ends.
    Majors never end early;
  - regular-season overtime is 3v3; a penalty adds a skater to the other team
    (4v3, 5v3) instead of removing one.
Goalie in/out comes from the goalie_change events, NOT the goalie shifts. Scored
against all goals' plus/minus lists, events were right far more often (96.9% vs
78.5% of goals): goalie shifts are missing for whole periods in 134 games and have
gaps that look like pulls. The events have their own errors, handled here:
  - the side is taken from the roster side of the goalie named in the event
    (5 events carry the wrong team);
  - each period starts with the goalie in;
  - a pull lasting more than MAX_PULL_S is ignored: those are a goalie_out that
    was never followed by a goalie_in (some last nearly a whole period);
  - a goalie_in logged up to 3 s before an empty-net goal against that side is
    moved to the goal (it was logged a second early);
  - pulls the events never logged are added from the shifts and goals
    (refine_goalie): a gap in the goalie's shifts while the team has an extra
    skater recorded, and goals that prove the goalie was out (empty-net flag on the
    goal event; 6 skaters in the scorer's plus list). Saved shots on goal prove the
    goalie was in, so inferred pulls start after them.
    Checked against goals: 11 of 93 empty-net goals had the goalie "in" from events
    alone; the shift-gap rule alone (no goal information) catches all of them.
With the goalie out, the team has an extra skater (extra attacker / delayed penalty).

Time convention: elapsed seconds in the period. A state starts at start_s and
runs to end_s. Something that happens AT a boundary t (a penalty, a goal) ends
the old state at t and starts the new one at t, so:
  - in-play events at t (shots, goals) take the state of the interval with
    start_s < t <= end_s  (the goal that ends a power play is a PP goal);
  - stoppage events at t (faceoffs) take start_s <= t < end_s.
"""
import json
import re

import numpy as np

JAILBREAK = True  # PWHL rule: a minor also ends when the shorthanded team scores
MAX_INFERRED_SKATER_S = 60  # cap for skater shifts whose end had to be inferred
MAX_PULL_S = 300
# Corrections to event-based goalie state (see refine_goalie); switchable for testing.
GOALIE_FIX = {"shift_gaps": True, "early_return": True, "goal_anchors": True}
MIN_EXTRA_S = 5   # a goalie-shift gap counts as a pull if the team has an extra skater this long  # longer event-based pulls are treated as errors (best fit to goals: tried 60-600)


def period_len(period, season_type):
    return 1200 if period <= 3 or season_type == "playoffs" else 300


# --- penalty box simulation ---------------------------------------------------
def box_counts(items, end_t, jailbreak=JAILBREAK):
    """items: chronological list of dicts with keys t (absolute game seconds),
    kind ('pen'|'goal'), side ('home'|'visitor'), and for penalties secs,
    minor, player. Returns change points [(t, n_home_in_box, n_visitor_in_box)]."""
    box = {"home": [], "visitor": []}    # active penalties: [end_t, is_minor, player]
    queue = {"home": [], "visitor": []}  # waiting penalties
    changes = [(0, 0, 0)]

    def record(t):
        st = (len(box["home"]), len(box["visitor"]))
        if changes[-1][1:] != st:
            if changes[-1][0] == t:
                changes[-1] = (t, *st)
                if len(changes) > 1 and changes[-2][1:] == st:  # back to the previous state
                    changes.pop()
            else:
                changes.append((t, *st))

    def start_queued(t):
        for side in box:
            q, i = queue[side], 0
            while i < len(q) and len(box[side]) < 2:
                p = q[i]
                if p["player"] is not None and any(a[2] == p["player"] for a in box[side]):
                    i += 1
                    continue
                box[side].append([t + p["secs"], p["minor"], p["player"]])
                q.pop(i)

    def advance(t):
        while True:
            due = [a[0] for side in box for a in box[side] if a[0] <= t]
            if not due:
                return
            e = min(due)
            for side in box:
                box[side] = [a for a in box[side] if a[0] != e]
            start_queued(e)
            record(e)

    for it in items:
        advance(it["t"])
        if it["kind"] == "pen":
            queue[it["side"]].append(it)
            start_queued(it["t"])
        else:
            nh, nv = len(box["home"]), len(box["visitor"])
            if nh != nv:
                short = "home" if nh > nv else "visitor"
                if it["side"] != short or jailbreak:
                    minors = [a for a in box[short] if a[1]]
                    if minors:
                        box[short].remove(min(minors, key=lambda a: a[0]))
                        start_queued(it["t"])
        record(it["t"])
    advance(end_t)
    return changes


def skaters(n_home, n_visitor, ot_3v3):
    if not ot_3v3:
        return 5 - n_home, 5 - n_visitor
    d = n_home - n_visitor
    if d == 0:
        return 3, 3
    return (3, 3 + min(d, 2)) if d > 0 else (3 + min(-d, 2), 3)


# --- blank shift ends -------------------------------------------------------
def mmss(v):
    m = re.fullmatch(r"\s*(\d+):(\d{1,2})\s*", str(v)) if v is not None else None
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def fill_blank_ends(ps, prows, goalie_side, L):
    """Give shifts with a blank end_time an end (in place). The shift runs from its
    recorded start until the first second at which this player being on would
    give the team MORE players than it should have, capped at the player's own
    next shift, the period end, and (skaters only) MAX_INFERRED_SKATER_S. "Should have" = the strength table's skaters
    (+1 extra attacker with the goalie pulled) for skaters; 1 goalie for goalies.
    Shifts that would get no time stay NULL. Sets end_inferred = 1."""
    if not any(s["clock_in_s"] is None for s in ps):
        return
    skaters = {"home": np.zeros(L, int), "visitor": np.zeros(L, int)}
    for r in prows:
        skaters["home"][r["start_s"]:r["end_s"]] = r["home_skaters"] + 1 - r["home_goalie_in"]
        skaters["visitor"][r["start_s"]:r["end_s"]] = r["visitor_skaters"] + 1 - r["visitor_goalie_in"]
    for side in ("home", "visitor"):
        for goalie in (False, True):
            grp = [s for s in ps if s["side"] == side and (s["player_id"] in goalie_side) == goalie]
            blanks = []
            for s in grp:
                if s["clock_in_s"] is None:
                    st = mmss(json.loads(s["extra"]).get("start_time")) if s.get("extra") else None
                    if st is not None and 0 < st <= L:
                        blanks.append((L - st, s))
            if not blanks:
                continue
            cap = np.ones(L, int) if goalie else skaters[side]
            count = np.zeros(L + 1, int)
            for s in grp:
                if s["clock_in_s"] is not None:
                    a, b = max(0, L - int(s["clock_in_s"])), min(L, L - int(s["clock_out_s"]))
                    if b > a:
                        count[a] += 1
                        count[b] -= 1
            count = np.cumsum(count)[:L]
            for a, s in sorted(blanks, key=lambda x: x[0]):
                nxt = min((L - int(o["clock_in_s"]) for o in grp if o is not s and o["player_id"] == s["player_id"]
                           and o["clock_in_s"] is not None and L - int(o["clock_in_s"]) > a), default=L)
                if not goalie:
                    nxt = min(nxt, a + MAX_INFERRED_SKATER_S)
                over = np.flatnonzero(count[a:nxt] + 1 > cap[a:nxt])
                b = a + (int(over[0]) if len(over) else nxt - a)
                s["end_inferred"] = 1
                if b > a:
                    s["clock_in_s"], s["clock_out_s"], s["duration_s"] = L - a, L - b, b - a
                    count[a:b] += 1


# --- goalie presence ----------------------------------------------------------
def goalie_in_net(changes, L, max_pull=MAX_PULL_S):
    """Boolean array over the period's seconds: goalie in net. changes: one
    side's goalie_change events in this period as (elapsed_t, goalie_in) where
    goalie_in is True for a goalie_in event and False for a goalie_out."""
    in_net = np.ones(L, bool)
    state, last = True, 0
    for t, is_in in changes:
        t = min(L, max(0, int(t)))
        in_net[last:t] = state
        state, last = is_in, t
    in_net[last:] = state
    i = 0
    while i < L:  # undo implausibly long pulls (goalie_out never followed by goalie_in)
        if in_net[i]:
            i += 1
            continue
        j = i
        while j < L and not in_net[j]:
            j += 1
        if j - i > max_pull:
            in_net[i:j] = True
        i = j
    return in_net


def runs(mask):
    """[start, end) runs of True in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def refine_goalie(in_net, gcov, gshare, extra, anchors, saves, L, fix=GOALIE_FIX):
    """Add goalie pulls the goalie_change events missed (they almost never invent
    one: 2 of 17,038 saved shots fall in an event-based pull).
      shift_gaps:   a gap in this side's goalie shifts, no longer than MAX_PULL_S,
                    during which the team has an extra skater recorded for at least
                    MIN_EXTRA_S. Only in periods where goalie shifts cover at least
                    half the period (whole periods of goalie shifts are missing in
                    some games, and those must not read as pulls). A gap that starts
                    at the period start is a late-recorded first shift, not a pull.
      goal_anchors: at an empty-net goal against this side, or a goal by this side
                    with 6 skaters in its plus list, the goalie is out. The pull is
                    taken to start where the goalie-shift gap or the extra-skater
                    run around the goal starts, else 1 s before the goal.
    An inferred pull never covers a saved shot against this side (a save proves the
    goalie was in): it starts after the last such save.
    anchors: elapsed goal times (seconds) at which this side's goalie was out.
    saves: elapsed times of saved shots on goal against this side."""
    out = ~in_net
    gap = ~gcov
    after_saves = lambda a, b: max([a] + [int(t) for t in saves if a < t <= b])  # in-play: save at t covers second t-1
    if fix.get("shift_gaps") and gshare >= 0.5:
        for a, b in runs(gap):
            if a == 0:
                continue
            a = after_saves(a, b)
            if 0 < b - a <= MAX_PULL_S and any(e - s >= MIN_EXTRA_S for s, e in runs(extra[a:b])):
                out[a:b] = True
    if fix.get("goal_anchors"):
        for t in anchors:
            k = min(L, int(t)) - 1
            if k < 0 or out[k]:
                continue
            start = k
            for mask, ok in ((gap, gshare >= 0.5), (extra, True)):
                if ok and mask[k]:
                    s0 = next(a for a, b in runs(mask) if a <= k < b)
                    s0 = after_saves(s0, k)
                    if k - s0 <= MAX_PULL_S:
                        start = min(start, s0)
            out[start:k + 1] = True
    return ~out


# --- per game -----------------------------------------------------------------
def game_strength(game_id, season_type, home_team_id, events, shifts, goalie_side, jailbreak=JAILBREAK):
    """events: chronological dicts (event, period, period_time_s, team_id,
    penalty_minutes, is_power_play, served_by, goalie_in, goalie_out). shifts:
    dicts (player_id, side, period, clock_in_s, clock_out_s). goalie_side: goalie
    player_id -> roster side. Returns (strength_rows, check_rows)."""
    periods = sorted({int(e.get("period")) for e in events if e.get("period")} |
                     {int(s["period"]) for s in shifts if s["period"]})
    if season_type != "playoffs":
        periods = [p for p in periods if p <= 4]  # period 5 = shootout
    if not periods:
        return [], []
    periods = list(range(1, max(periods) + 1))    # a period with no events/shifts still happened
    plen = {p: period_len(p, season_type) for p in range(1, max(periods) + 1)}
    offset = {p: sum(plen[q] for q in range(1, p)) for p in plen}
    side_of = lambda team: "home" if team == home_team_id else "visitor"

    items, minors_at, gchanges, anchors, saves = [], {}, {}, {}, {}
    for e in events:
        if e.get("event") == "shot" and e.get("is_goal") == 0 and e.get("period") in plen                 and e.get("period_time_s") is not None and e.get("team_id") is not None:
            dfn = "visitor" if side_of(e.get("team_id")) == "home" else "home"
            saves.setdefault((dfn, e["period"]), []).append(e["period_time_s"])
        if e.get("event") == "goal" and e.get("period") in plen and e.get("period_time_s") is not None                 and e.get("team_id") is not None:
            scorer, other = side_of(e.get("team_id")), "visitor" if side_of(e.get("team_id")) == "home" else "home"
            if e.get("is_empty_net") == 1:
                anchors.setdefault((other, e["period"]), []).append(e["period_time_s"])
            if e.get("plus_n") == 6:
                anchors.setdefault((scorer, e["period"]), []).append(e["period_time_s"])
        if e.get("event") == "goalie_change" and e.get("period") in plen and e.get("period_time_s") is not None:
            gid = e.get("goalie_in") or e.get("goalie_out")
            side = goalie_side.get(gid) or side_of(e.get("team_id"))
            gchanges.setdefault((side, e["period"]), []).append((e["period_time_s"], e.get("goalie_in") is not None))
            continue
        if e.get("period") not in plen or e.get("period_time_s") is None:
            continue
        t = offset[e.get("period")] + e.get("period_time_s")
        if e.get("event") == "penalty" and e.get("is_power_play") == 1 and e.get("penalty_minutes") in (2, 5):
            item = {"t": t, "kind": "pen", "side": side_of(e.get("team_id")),
                    "secs": int(e.get("penalty_minutes") * 60), "minor": e.get("penalty_minutes") == 2,
                    "player": e.get("served_by")}
            # Feed quirk: a double minor is sometimes logged as two minors, the
            # second timed exactly when the first would expire. Re-time it to the
            # infraction and queue it right behind the first, so that it starts
            # when the first ends, including when a goal ends the first early.
            key = (item["side"], item["player"], e.get("penalty_description"))
            first = minors_at.get(key)
            if item["minor"] and item["player"] is not None and first is not None and first["t"] + 120 == t:
                item["t"], item["double_minor_part"] = first["t"], 2
                items.insert(items.index(first) + 1, item)
                continue
            if item["minor"]:
                minors_at[key] = item
            items.append(item)
        elif e.get("event") == "goal" and e.get("team_id") is not None:
            items.append({"t": t, "kind": "goal", "side": side_of(e.get("team_id"))})
    end_t = offset[max(periods)] + plen[max(periods)]
    changes = box_counts(items, end_t, jailbreak)
    if GOALIE_FIX.get("early_return"):
        # a goalie_in logged up to 3 s BEFORE an empty-net goal against that side
        # belongs right after the goal (the goalie came back for the faceoff)
        for (side, p), ch in gchanges.items():
            en = [t for t in anchors.get((side, p), [])]
            gchanges[(side, p)] = [(next((g for g in en if 0 < g - t <= 3), t) if is_in else t, is_in)
                                   for t, is_in in ch]

    rows, checks = [], []
    for p in periods:
        L, off = plen[p], offset[p]
        ot3 = p >= 4 and season_type != "playoffs"
        ps = [s for s in shifts if s["period"] == p]
        goalies = {side: goalie_in_net(gchanges.get((side, p), []), L) for side in ("home", "visitor")}
        # manpower per second, and what the shifts show, to refine goalie state
        man = {"home": np.zeros(L, int), "visitor": np.zeros(L, int)}
        for i, (t, nh, nv) in enumerate(changes):
            a = max(0, t - off); b = min(L, (changes[i + 1][0] - off) if i + 1 < len(changes) else L)
            if b > a:
                man["home"][a:b], man["visitor"][a:b] = skaters(nh, nv, ot3)
        for side in ("home", "visitor"):
            cnt = np.zeros(L + 1, int); gcov = np.zeros(L, bool)
            for s in ps:
                if s["side"] != side or s["clock_in_s"] is None:
                    continue
                a, b = max(0, L - int(s["clock_in_s"])), min(L, L - int(s["clock_out_s"]))
                if b <= a:
                    continue
                if s["player_id"] in goalie_side:
                    gcov[a:b] = True
                else:
                    cnt[a] += 1
                    cnt[b] -= 1
            extra = np.cumsum(cnt)[:L] >= man[side] + 1
            goalies[side] = refine_goalie(goalies[side], gcov, gcov.mean(), extra,
                                          anchors.get((side, p), []), saves.get((side, p), []), L)
        cuts = {0, L} | {t - off for t, _, _ in changes if off < t < off + L}
        for side in goalies:
            cuts |= {int(k) + 1 for k in np.flatnonzero(np.diff(goalies[side]))}
        cuts = sorted(c for c in cuts if 0 <= c <= L)
        for a, b in zip(cuts, cuts[1:]):
            if b <= a:
                continue
            nh, nv = next((h, v) for t, h, v in reversed(changes) if t <= off + a)
            hs, vs = skaters(nh, nv, ot3)
            hg, vg = (int(goalies[s][a:b].all()) for s in ("home", "visitor"))
            row = {"game_id": game_id, "period": p, "start_s": a, "end_s": b,
                   "home_skaters": hs, "visitor_skaters": vs, "home_goalie_in": hg, "visitor_goalie_in": vg,
                   "manpower": f"{hs}v{vs}", "state": f"{hs + 1 - hg}v{vs + 1 - vg}"}
            last = rows[-1] if rows else None
            if last and last["period"] == p and last["end_s"] == a and all(
                    last[k] == row[k] for k in ("home_skaters", "visitor_skaters", "home_goalie_in", "visitor_goalie_in")):
                last["end_s"] = b
            else:
                rows.append(row)

        fill_blank_ends(ps, [r for r in rows if r["period"] == p], goalie_side, L)

        # per-second check: SKATERS on the ice per the shifts vs. the expected
        # skaters (manpower + 1 extra attacker when the goalie is pulled). Goalie
        # shifts are left out: they are missing for whole periods in many games,
        # which would swamp the check (goalie presence comes from events anyway).
        expected = {"home": np.zeros(L, int), "visitor": np.zeros(L, int)}
        for r in (r for r in rows if r["period"] == p):
            expected["home"][r["start_s"]:r["end_s"]] = r["home_skaters"] + 1 - r["home_goalie_in"]
            expected["visitor"][r["start_s"]:r["end_s"]] = r["visitor_skaters"] + 1 - r["visitor_goalie_in"]
        for side in ("home", "visitor"):
            diff = np.zeros(L + 1, int)
            for s in ps:
                if s["side"] != side or s["clock_in_s"] is None or s["player_id"] in goalie_side:
                    continue
                a, b = max(0, L - s["clock_in_s"]), min(L, L - s["clock_out_s"])
                if b > a:
                    diff[a] += 1
                    diff[b] -= 1
            obs = np.cumsum(diff)[:L]
            gap = obs - expected[side]
            checks.append({"game_id": game_id, "period": p, "side": side, "seconds": L,
                           "sec_ok": int((gap == 0).sum()), "sec_under": int((gap < 0).sum()),
                           "sec_over": int((gap > 0).sum()),
                           "player_sec_missing": int(-gap[gap < 0].sum()),
                           "player_sec_extra": int(gap[gap > 0].sum())})
    for r in rows:
        r["duration_s"] = r["end_s"] - r["start_s"]
    return rows, checks
