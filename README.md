# pwhlstats

PWHL play-by-play and shift data (from the HockeyTech feeds that fastRhockey uses),
stored as Parquet and published as an [Observable Framework](https://observablehq.com/framework/) site.

## Pipeline

```
HockeyTech API ──pwhl_ingest.py──▶ data/raw/      (raw JSON as Parquet; source of truth)
                                     │
                    normalize.py ────┤──▶ data/clean/   (typed tables as Parquet; committed)
                                     └──▶ pwhl_clean.db (same tables + views in SQLite; local only)
                                              │
                    src/data/*.py loaders ◀───┘  (DuckDB over data/clean) ──▶ site
```

- `data/raw/payloads/season=N.parquet`: one row per game and feed (`events`, `shifts`) with the raw JSON.
  Rebuilding everything downstream needs no API calls.
- `data/raw/schedule/season=N.parquet`, `data/raw/seasons.parquet`: raw schedule and season rows.
- `data/raw/failures.csv`: games that couldn't be fetched yet (retried up to 5 runs).
- `data/clean/<table>/season=N.parquet` for per-game tables (events, event_players, event_on_ice,
  shifts, game_rosters, strength); `data/clean/{games,seasons,players,teams}.parquet`;
  `data/clean/quality/` for the shift-quality tables (shift_coverage, game_shift_quality,
  strength_check, game_strength_quality).
- `strength`: skaters per team and goalie in/out for every stretch of every game, from penalties
  (with the PWHL jailbreak rule) and goalie-change events. Rules and the boundary convention for
  joining events to it are documented in [ingest/strength.py](ingest/strength.py).

RAPM (`python rapm/fit.py`, run in CI): xG-based regularized adjusted plus-minus per season year
(regular season + playoffs, no preseason): 5v5 offence / defence → `data/rapm/rapm_5v5_xg.parquet`, and 5v4
power-play offence / penalty-kill defence → `data/rapm/rapm_pp_xg.parquet` (each with a `.meta.json`). Stints are
built from shifts + strength + `data/xg/shot_xg.parquet` in [rapm/stints.py](rapm/stints.py); model and
outputs are documented in [rapm/fit.py](rapm/fit.py). Ratings are multi-season estimates (earlier seasons
weighted by a half-life chosen on completed seasons); files are rewritten only when results change.

The workflow `.github/workflows/update.yml` runs daily (and on manual dispatch): ingest → normalize →
xG scoring → RAPM → tests → commit `data/` → build → deploy to GitHub Pages. On Mondays it also re-fetches every game of the
current season to pick up late corrections. Pushes to `main` rebuild and deploy without scraping.

Tests (`python -m pytest -q`): unit tests for the strength, ingest and parsing rules, plus data checks on
`data/raw` and `data/clean` (completeness, keys, strength tiling and agreement with goals, shift quality).
In CI a failing test stops the run before any data is committed or deployed.

## Local use

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install

python ingest/pwhl_ingest.py        # fetch new final games (incremental)
python ingest/normalize.py          # rebuild data/clean and pwhl_clean.db
python query.py "SELECT * FROM game_shift_quality LIMIT 5"
npm run dev                         # preview at http://localhost:3000
```

Framework caches loader output in `src/.observablehq/cache` and only reruns a loader when the
loader file itself changes, not when `data/` changes. After new data, run `npm run clean` before
previewing locally. CI always starts with an empty cache.
