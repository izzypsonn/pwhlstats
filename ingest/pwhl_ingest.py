#!/usr/bin/env python3
"""
Incrementally fetch PWHL play-by-play events and shifts and store the raw JSON
as Parquet. These files are the source of truth; normalize.py builds the clean
tables from them with no API calls.

Data comes from the undocumented HockeyTech feeds that fastRhockey uses:
  - seasons  : feed=modulekit,     view=seasons
  - schedule : feed=modulekit,     view=schedule
  - events   : feed=statviewfeed,  view=gameCenterPlayByPlay  (JSONP)
  - shifts   : feed=modulekit,     view=gameshifts
  - rosters  : feed=modulekit,     view=roster  (per team-season: shoots/catches, portrait URL)

Files written (all under data/raw/):
  seasons.parquet                    one row per season, raw fields as text
  schedule/season=N.parquet          one row per scheduled game, raw fields as text
  payloads/season=N.parquet          game_id, feed ('events'|'shifts'), fetched_at, payload (JSON text)
  rosters/season=N.parquet           team_id, fetched_at, payload (roster JSON text)
  failures.csv                       games that could not be fetched/validated yet

A game counts as loaded only when BOTH feeds returned a payload that passes
validation; otherwise nothing is stored and it is retried on the next run
(up to MAX_ATTEMPTS runs, then skipped until --retry-failed). Files are only
rewritten when their contents change, so a run with no new games leaves the
repo untouched.

Usage:
  python ingest/pwhl_ingest.py                      # all seasons (first run = backfill)
  python ingest/pwhl_ingest.py --seasons 8 9        # specific HockeyTech season_ids
  python ingest/pwhl_ingest.py --max-games 20       # cap API work for one run
  python ingest/pwhl_ingest.py --refresh-current    # also re-fetch every game of active seasons
                                                    # (weekly, to pick up late league corrections)
  python ingest/pwhl_ingest.py --import-sqlite pwhl.db   # one-off: import old raw_payloads
"""
import argparse
import json
import logging
import os
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BASE = "https://lscluster.hockeytech.com/feed/index.php"
KEY = "446521baf8c38984"  # public PWHL client key
COMMON = {"key": KEY, "client_code": "pwhl", "lang": "en", "fmt": "json"}
MAX_ATTEMPTS = 5  # runs a failing game is retried before it is skipped

log = logging.getLogger("pwhl")
session = requests.Session()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def parse_json(text):
    """Parse JSON, stripping a JSONP wrapper such as cb({...}); if present."""
    text = text.strip()
    text = re.sub(r"^[A-Za-z_$][\w.$]*\(", "", text)
    text = re.sub(r"\)\s*;?\s*$", "", text)
    return json.loads(text)


def fetch(params, tries=4, timeout=30):
    """GET the feed and return parsed JSON. Retries with exponential backoff
    (5, 10, 20 s), honouring Retry-After on 429/503."""
    last = None
    for attempt in range(1, tries + 1):
        wait = 5 * 2 ** (attempt - 1)
        try:
            r = session.get(BASE, params={**COMMON, **params}, timeout=timeout)
            if r.status_code in (429, 503) and r.headers.get("Retry-After", "").isdigit():
                wait = max(wait, int(r.headers["Retry-After"]))
            r.raise_for_status()
            try:
                return parse_json(r.text)
            except ValueError:
                log.warning("not JSON from %s; body starts: %r", r.url, r.text[:200])
                raise
        except (requests.RequestException, ValueError) as e:
            last = e
            if attempt < tries:
                log.warning("fetch failed (%s/%s), retrying in %ss: %s", attempt, tries, wait, e)
                time.sleep(wait)
    raise RuntimeError(f"giving up on {params}: {last}")


def fetch_events(game_id):
    # fastRhockey sends a JSONP callback (not fmt=json) for this feed
    return fetch({"feed": "statviewfeed", "view": "gameCenterPlayByPlay",
                  "game_id": game_id, "league_id": "",
                  "fmt": None, "callback": "angular.callbacks._0"})


def fetch_shifts(game_id):
    return fetch({"feed": "modulekit", "view": "gameshifts", "game_id": game_id})


def fetch_seasons():
    return fetch({"feed": "modulekit", "view": "seasons"})


def fetch_schedule(season_id):
    return fetch({"feed": "modulekit", "view": "schedule", "season_id": season_id})


def fetch_roster(season_id, team_id):
    return fetch({"feed": "modulekit", "view": "roster", "season_id": season_id, "team_id": team_id})


# --------------------------------------------------------------------------
# Parsing helpers (normalize.py imports list_of_dicts)
# --------------------------------------------------------------------------
def list_of_dicts(obj, path=""):
    """Yield (path, rows) for each list of dicts in obj. Lists nested inside a
    yielded list's rows are left in place. A one-row wrapper such as
    [{"Events": [...]}] is looked through, not yielded."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            wrapper = len(obj) == 1 and any(True for _ in list_of_dicts(obj[0], path + "[0]"))
            if not wrapper:
                yield path, obj
                return
        for i, x in enumerate(obj):
            yield from list_of_dicts(x, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from list_of_dicts(v, f"{path}.{k}" if path else k)


def rows_frame(payload):
    """Flatten every list-of-dicts in a schedule/seasons payload into one
    DataFrame of text columns (nested values kept as JSON text)."""
    rows = [r for _, rs in list_of_dicts(payload) for r in rs]
    df = pd.DataFrame(rows)
    for c in df.columns:
        df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v))
                          else json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                          else str(v)).astype("string")
    return df


def n_events(payload):
    """Number of play-by-play events in an events payload (0 = unusable)."""
    for _, rows in list_of_dicts(payload):
        if rows and "event" in {str(k).lower() for k in rows[0]}:
            return len(rows)
    return 0


def roster_players(payload):
    """Player rows of a roster payload (staff rows have no player_id)."""
    try:
        return [p for p in payload["SiteKit"]["Roster"] if isinstance(p, dict) and p.get("player_id")]
    except (KeyError, TypeError):
        return []


def n_shifts(payload):
    """Number of shifts in a shifts payload; 0 unless both teams have players."""
    try:
        gs = payload["SiteKit"]["Gameshifts"]
        if not (gs.get("home") and gs.get("visitor")):
            return 0
        return sum(len(p.get("shifts") or []) for side in ("home", "visitor") for p in gs[side])
    except (KeyError, TypeError, AttributeError):
        return 0


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def payload_path(season):
    return RAW / "payloads" / f"season={season}.parquet"


def schedule_path(season):
    return RAW / "schedule" / f"season={season}.parquet"


def roster_path(season):
    return RAW / "rosters" / f"season={season}.parquet"


ROSTER_COLS = ["team_id", "fetched_at", "payload"]


def read_rosters(season):
    p = roster_path(season)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame({"team_id": pd.Series(dtype="int64"), "fetched_at": pd.Series(dtype="string"),
                         "payload": pd.Series(dtype="string")})


def update_rosters(scheds, payloads, end_dates, today, now, sleep):
    """Fetch team rosters (player bio: shoots/catches, portrait URL) for every
    team-season with stored games. Rosters of seasons that ended more than 7 days
    ago are fetched once; active seasons are re-fetched every run (signings,
    call-ups). A team's row keeps its old fetched_at when the payload is unchanged.
    Failures keep the stored copy and are retried next run."""
    fetched = changed = 0
    for s, d in scheds.items():
        if not stored_ids(payloads.get(s, pd.DataFrame(columns=PAYLOAD_COLS))):
            continue
        end = end_dates.get(s)
        active = end is None or pd.isna(end) or end.date() >= today - timedelta(days=7)
        teams = sorted({int(t) for c in ("home_team", "visiting_team") for t in finals(d)[c].dropna()})
        old = read_rosters(s)
        rows = {int(r.team_id): r._asdict() for r in old.itertuples(index=False)}
        for t in teams:
            if t in rows and not active:
                continue
            try:
                text = json.dumps(fetch_roster(s, t))
                if not roster_players(json.loads(text)):
                    raise ValueError("no players in roster")
                fetched += 1
                if t not in rows or rows[t]["payload"] != text:
                    rows[t] = {"team_id": t, "fetched_at": now, "payload": text}
            except Exception as e:
                log.error("roster season %s team %s failed: %s", s, t, e)
            time.sleep(sleep)
        df = pd.DataFrame([rows[t] for t in sorted(rows)], columns=ROSTER_COLS)
        if rows:
            df = df.astype({"team_id": "int64", "fetched_at": "string", "payload": "string"})
            changed += write_if_changed(df, roster_path(s))
    log.info("rosters: %d fetched, %d season files changed", fetched, changed)


def write_if_changed(df, path):
    """Write Parquet atomically, but only if the contents differ from what is
    on disk (keeps the git history free of no-op data commits)."""
    df = df.reset_index(drop=True)
    if path.exists():
        try:
            if pd.read_parquet(path).reset_index(drop=True).equals(df):
                return False
        except Exception:  # unreadable/corrupt file: overwrite it
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)
    return True


PAYLOAD_COLS = ["game_id", "feed", "fetched_at", "payload"]


def read_payloads(season):
    p = payload_path(season)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame({"game_id": pd.Series(dtype="int64"), "feed": pd.Series(dtype="string"),
                         "fetched_at": pd.Series(dtype="string"), "payload": pd.Series(dtype="string")})


def stored_ids(df):
    """Games with both feeds stored."""
    feeds = df.groupby("game_id")["feed"].nunique()
    return set(feeds[feeds == 2].index.astype(int))


def upsert_game(df, game_id, ev_text, sh_text, fetched_at):
    """Replace one game's two payload rows. Keeps the old fetched_at when the
    payload is unchanged. Returns (new_df, changed)."""
    old = df[df.game_id == game_id].set_index("feed")
    new_rows, changed = [], False
    for feed, text in (("events", ev_text), ("shifts", sh_text)):
        if feed in old.index and old.at[feed, "payload"] == text:
            new_rows.append({"game_id": game_id, "feed": feed,
                             "fetched_at": old.at[feed, "fetched_at"], "payload": text})
        else:
            changed = True
            new_rows.append({"game_id": game_id, "feed": feed, "fetched_at": fetched_at, "payload": text})
    if not changed:
        return df, False
    out = pd.concat([df[df.game_id != game_id], pd.DataFrame(new_rows)], ignore_index=True)
    out = out.astype({"game_id": "int64", "feed": "string", "fetched_at": "string", "payload": "string"})
    return out.sort_values(["game_id", "feed"]).reset_index(drop=True)[PAYLOAD_COLS], True


def read_failures():
    p = RAW / "failures.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype={"game_id": int, "attempts": int, "last_error": str, "last_attempt": str})
    return {r.game_id: r._asdict() for r in df.itertuples(index=False)}


def write_failures(fails):
    p = RAW / "failures.csv"
    if not fails:
        if p.exists():
            p.unlink()
        return
    df = pd.DataFrame(sorted(fails.values(), key=lambda r: r["game_id"]),
                      columns=["game_id", "attempts", "last_error", "last_attempt"])
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator="\n")


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def finals(sched):
    """Final games from a schedule frame. Checked against all 11 seasons
    (2023-24 .. 2026-27): `final == "1"` and game_status starting with "Final"
    agree for every game."""
    is_final = sched["final"].eq("1") | sched["game_status"].str.lower().str.startswith("final")
    return sched[is_final.fillna(False)]


def season_complete(sched, have, end_date, today):
    """True if a finished season has every game final and stored: no need to
    hit the schedule endpoint for it again."""
    if sched is None or end_date is None or end_date >= today - timedelta(days=7):
        return False
    ids = set(sched["game_id"].astype(int))
    return ids == set(finals(sched)["game_id"].astype(int)) and ids <= have


def import_sqlite(db_path, game_season, payloads):
    """One-off migration: copy raw_payloads from the old SQLite store."""
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT game_id, feed, fetched_at, payload FROM raw_payloads").fetchall()
    con.close()
    by_game = {}
    for g, feed, at, text in rows:
        by_game.setdefault(int(g), {})[feed] = (at, text)
    n = 0
    for g, feeds in sorted(by_game.items()):
        s = game_season.get(g)
        if s not in payloads or g in stored_ids(payloads[s]) or set(feeds) != {"events", "shifts"}:
            continue
        ev, sh = (json.loads(feeds[f][1]) for f in ("events", "shifts"))
        if not (n_events(ev) and n_shifts(sh)):
            continue
        payloads[s], _ = upsert_game(payloads[s], g, json.dumps(ev), json.dumps(sh), feeds["events"][0])
        n += 1
    log.info("imported %d games from %s", n, db_path)
    return n


def run(seasons=None, sleep=1.0, refresh_days=2, max_games=None, retry_failed=False,
        force_schedule=False, import_db=None, refresh_current=False):
    today = date.today()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # seasons -----------------------------------------------------------
    try:
        seasons_df = rows_frame(fetch_seasons())
        seasons_df = seasons_df.drop_duplicates("season_id").sort_values(
            "season_id", key=lambda s: s.astype(int)).reset_index(drop=True)
        write_if_changed(seasons_df, RAW / "seasons.parquet")
    except Exception as e:
        if not (RAW / "seasons.parquet").exists():
            raise
        log.error("seasons fetch failed (%s); using stored seasons.parquet", e)
        seasons_df = pd.read_parquet(RAW / "seasons.parquet")
    time.sleep(sleep)
    end_dates = {r.season_id: pd.to_datetime(r.end_date, errors="coerce")
                 for r in seasons_df.itertuples()}
    ids = [str(s) for s in (seasons or seasons_df["season_id"])]

    # schedules ---------------------------------------------------------
    payloads = {s: read_payloads(s) for s in ids}
    scheds = {}
    for s in ids:
        old = pd.read_parquet(schedule_path(s)) if schedule_path(s).exists() else None
        end = end_dates.get(s)
        end = None if end is None or pd.isna(end) else end.date()
        if not force_schedule and season_complete(old, stored_ids(payloads[s]), end, today):
            scheds[s] = old
            continue
        try:
            sched = rows_frame(fetch_schedule(s))
            if not sched.empty:
                sched = sched.sort_values("game_id", key=lambda c: c.astype(int)).reset_index(drop=True)
                write_if_changed(sched, schedule_path(s))
            scheds[s] = sched
        except Exception as e:
            log.error("schedule for season %s failed (%s); using stored copy if any", s, e)
            scheds[s] = old
        time.sleep(sleep)
    scheds = {s: d for s, d in scheds.items() if d is not None and not d.empty}
    game_season = {int(g): s for s, d in scheds.items() for g in d["game_id"]}

    dirty = set()
    if import_db:
        before = {s: len(df) for s, df in payloads.items()}
        import_sqlite(import_db, game_season, payloads)
        dirty |= {s for s in payloads if len(payloads[s]) != before[s]}

    # work list ---------------------------------------------------------
    fails = read_failures()
    todo, skipped = [], []
    for s, d in scheds.items():
        have = stored_ids(payloads[s])
        end = end_dates.get(s)
        active = end is not None and not pd.isna(end) and end.date() >= today - timedelta(days=7)
        for r in finals(d).itertuples():
            g = int(r.game_id)
            played = pd.to_datetime(r.date_played, errors="coerce")
            recent = refresh_days and not pd.isna(played) and played.date() >= today - timedelta(days=refresh_days)
            if g in have and not recent and not (refresh_current and active):
                continue
            if g not in have and fails.get(g, {}).get("attempts", 0) >= MAX_ATTEMPTS and not retry_failed:
                skipped.append(g)
                continue
            todo.append((g, s, g in have))
    todo.sort()
    if max_games is not None:
        todo = todo[:max_games]
    log.info("%d games to fetch (%d new, %d re-checks); %d skipped after %d failed attempts",
             len(todo), sum(not t[2] for t in todo), sum(t[2] for t in todo), len(skipped), MAX_ATTEMPTS)

    # fetch -------------------------------------------------------------
    def flush():
        for s in sorted(dirty):
            write_if_changed(payloads[s], payload_path(s))
        dirty.clear()
        write_failures(fails)

    added = updated = failed = 0
    for i, (g, s, refresh) in enumerate(todo, 1):
        try:
            ev = fetch_events(g)
            time.sleep(sleep)
            sh = fetch_shifts(g)
            ne, ns = n_events(ev), n_shifts(sh)
            if not (ne and ns):
                raise ValueError(f"incomplete data: {ne} events, {ns} shifts")
            payloads[s], changed = upsert_game(payloads[s], g, json.dumps(ev), json.dumps(sh), now)
            if changed:
                dirty.add(s)
                updated += refresh
                added += not refresh
            fails.pop(g, None)
            log.info("[%d/%d] game %s (season %s): %d events, %d shifts%s", i, len(todo), g, s, ne, ns,
                     "" if changed else " (unchanged)")
        except Exception as e:  # nothing stored for this game; retried next run
            failed += 1
            if not refresh:
                f = fails.get(g, {"game_id": g, "attempts": 0})
                fails[g] = {**f, "attempts": f["attempts"] + 1,
                            "last_error": str(e)[:200], "last_attempt": today.isoformat()}
            log.error("game %s failed: %s", g, e)
        time.sleep(sleep)
        if i % 25 == 0:  # keep progress if the run is interrupted
            flush()
    flush()
    update_rosters(scheds, payloads, end_dates, today, now, sleep)

    total = sum(len(stored_ids(df)) for df in payloads.values())
    msg = (f"PWHL ingest: {added} new games, {updated} updated, {failed} failed, "
           f"{len(skipped)} skipped; {total} games stored")
    log.info(msg)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(f"### {msg}\n")
            if fails:
                fh.write("\n| game | attempts | last error |\n|---|---|---|\n")
                for f in sorted(fails.values(), key=lambda r: r["game_id"]):
                    fh.write(f"| {f['game_id']} | {f['attempts']} | {f['last_error']} |\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="*", help="HockeyTech season_id values (default: all)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--refresh-days", type=int, default=2,
                    help="re-fetch games played in the last N days (late corrections); 0 = off")
    ap.add_argument("--max-games", type=int, help="fetch at most N games this run")
    ap.add_argument("--retry-failed", action="store_true",
                    help=f"also retry games that already failed {MAX_ATTEMPTS} times")
    ap.add_argument("--force-schedule", action="store_true",
                    help="re-fetch schedules even for completed seasons")
    ap.add_argument("--import-sqlite", metavar="DB", help="import raw_payloads from an old pwhl.db")
    ap.add_argument("--refresh-current", action="store_true",
                    help="re-fetch every stored game of seasons that ended less than 7 days ago or are "
                         "still running (late corrections); unchanged games leave the files untouched")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(a.seasons, a.sleep, a.refresh_days, a.max_games, a.retry_failed, a.force_schedule, a.import_sqlite,
        a.refresh_current)
