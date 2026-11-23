---
title: Time on ice
---

# Time on ice

Total time on ice per player and season: the sum of shift lengths at all strengths, regulation and
overtime. When a shift has no recorded end time, it is ended at the first moment the team would
otherwise have too many players on the ice.

```js
const toi = FileAttachment("data/toi.csv").csv({typed: true});
```

```js
const seasonNames = [...new Map(toi.map((d) => [d.season_id, d.season_name]))]
  .sort((a, b) => b[0] - a[0]);
const season = view(Inputs.select(new Map(seasonNames), {label: "Season"}));
const pos = view(Inputs.radio(["All", "F", "D", "G"], {label: "Position", value: "F"}));
const minGp = view(Inputs.range([1, 30], {label: "Min. games", step: 1, value: 5}));
```

```js
const rows = toi.filter((d) => d.season_id === season && (pos === "All" || d.pos === pos) && d.gp >= minGp);
const top = d3.sort(rows, (d) => -d.toi_per_gp).slice(0, 25);
```

<div class="card">
  <h2>Top 25 by time on ice per game</h2>
  ${resize((width) => Plot.plot({
    width,
    height: Math.max(120, top.length * 22 + 40),
    marginLeft: 170,
    x: {label: "Minutes per game", grid: true},
    y: {label: null},
    marks: [
      Plot.barX(top, {
        x: "toi_per_gp", y: (d) => `${d.player} (${d.team})`, sort: {y: "-x"},
        fill: "var(--theme-foreground-focus)", rx: 2, insetTop: 1, insetBottom: 1,
        tip: {format: {y: true, x: (v) => v.toFixed(1)}},
        channels: {GP: "gp", "Total minutes": "toi_min"}
      }),
      Plot.ruleX([0])
    ]
  }))}
</div>

```js
Inputs.table(rows, {
  columns: ["player", "team", "pos", "gp", "toi_min", "toi_per_gp"],
  header: {player: "Player", team: "Team", pos: "Pos", gp: "GP", toi_min: "TOI (min)", toi_per_gp: "TOI / GP"},
  sort: "toi_per_gp", reverse: true, rows: 25
})
```
