"""Framework data loader: time on ice per player and season, from the clean
Parquet export (data/clean). Writes CSV to stdout.

TOI here is the sum of shift lengths (all strengths, regulation + overtime).
Shifts with a blank end time use the inferred end from normalize.py (or count
as zero if none could be inferred); games flagged games.exclude_shifts are left out.
"""
import sys
from pathlib import Path

import duckdb

CLEAN = (Path(__file__).resolve().parents[2] / "data" / "clean").as_posix()

sql = f"""
WITH games AS (
    SELECT * FROM read_parquet('{CLEAN}/games.parquet') WHERE exclude_shifts = 0
), shifts AS (
    SELECT s.game_id, s.player_id, s.side, s.duration_s
    FROM read_parquet('{CLEAN}/shifts/*.parquet', hive_partitioning = false) s
    JOIN games USING (game_id)
    WHERE s.duration_s IS NOT NULL
), per_game AS (
    SELECT game_id, player_id, side, SUM(duration_s) AS toi_s FROM shifts GROUP BY ALL
), rosters AS (
    SELECT game_id, player_id, position
    FROM read_parquet('{CLEAN}/game_rosters/*.parquet', hive_partitioning = false)
)
SELECT g.season_id, se.season_name, se.season_type,
       pg.player_id, p.first_name || ' ' || p.last_name AS player,
       mode(t.abbreviation) AS team,
       CASE WHEN mode(r.position) IS NULL THEN NULL
            WHEN mode(r.position) = 'G' THEN 'G'
            WHEN mode(r.position) IN ('D', 'LD', 'RD') THEN 'D' ELSE 'F' END AS pos,
       COUNT(*) AS gp,
       ROUND(SUM(pg.toi_s) / 60.0, 1) AS toi_min,
       ROUND(SUM(pg.toi_s) / 60.0 / COUNT(*), 2) AS toi_per_gp
FROM per_game pg
JOIN games g USING (game_id)
JOIN read_parquet('{CLEAN}/seasons.parquet') se USING (season_id)
LEFT JOIN read_parquet('{CLEAN}/players.parquet') p USING (player_id)
LEFT JOIN rosters r USING (game_id, player_id)
LEFT JOIN read_parquet('{CLEAN}/teams.parquet') t
       ON t.team_id = CASE pg.side WHEN 'home' THEN g.home_team_id ELSE g.visiting_team_id END
GROUP BY ALL
ORDER BY g.season_id, toi_min DESC
"""
# write UTF-8 bytes: on Windows a piped stdout would otherwise be cp1252
sys.stdout.buffer.write(duckdb.sql(sql).df().to_csv(index=False, lineterminator="\n").encode("utf-8"))
