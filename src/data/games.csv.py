"""Framework data loader: one row per scheduled game (from data/clean/games.parquet)."""
import sys
from pathlib import Path

import duckdb

CLEAN = (Path(__file__).resolve().parents[2] / "data" / "clean").as_posix()
sql = f"""
SELECT g.game_id, g.season_id, s.season_name, s.season_type, g.date_played,
       h.abbreviation AS home, v.abbreviation AS visitor, g.home_goals, g.visiting_goals,
       g.game_status, g.has_pbp, g.exclude_shifts
FROM read_parquet('{CLEAN}/games.parquet') g
JOIN read_parquet('{CLEAN}/seasons.parquet') s USING (season_id)
LEFT JOIN read_parquet('{CLEAN}/teams.parquet') h ON h.team_id = g.home_team_id
LEFT JOIN read_parquet('{CLEAN}/teams.parquet') v ON v.team_id = g.visiting_team_id
ORDER BY g.game_id
"""
# write UTF-8 bytes: on Windows a piped stdout would otherwise be cp1252
sys.stdout.buffer.write(duckdb.sql(sql).df().to_csv(index=False, lineterminator="\n").encode("utf-8"))
