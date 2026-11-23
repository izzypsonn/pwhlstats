---
toc: false
---

# PWHL stats

Play-by-play and shift data for the PWHL, scraped from the league's HockeyTech feeds and
updated daily. The groundwork for expected-goals (xG) and RAPM models.

```js
const games = FileAttachment("data/games.csv").csv({typed: true});
```

```js
const played = games.filter((d) => d.has_pbp);
const last = d3.max(played, (d) => d.date_played);
```

<div class="grid grid-cols-3">
  <div class="card"><h2>Games with play-by-play</h2><span class="big">${played.length.toLocaleString("en-US")}</span></div>
  <div class="card"><h2>Seasons</h2><span class="big">${new Set(played.map((d) => d.season_id)).size}</span></div>
  <div class="card"><h2>Most recent game</h2><span class="big">${last ? d3.utcFormat("%b %-d, %Y")(last) : "—"}</span></div>
</div>

- [Time on ice](./toi): TOI per player and season, from shift data.

Data notes: shift data has known gaps (some shifts have no end time, and a few games have broken shift
files). Games flagged as unusable are left out of shift-based stats.
