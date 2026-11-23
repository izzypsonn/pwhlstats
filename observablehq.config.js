// See https://observablehq.com/framework/config for documentation.
import {existsSync} from "node:fs";

// Python data loaders run as `<interpreter> <loader file>` from the folder where
// `npm run build` / `npm run dev` was started (the project root). Prefer the
// project's .venv so duckdb/pandas are found; on Windows there is usually no
// real `python3`, and in CI (no .venv) setup-python provides `python3`.
const venv = process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python";
const python = existsSync(venv) ? venv : process.platform === "win32" ? "python" : "python3";

export default {
  title: "PWHL stats",

  pages: [
    {name: "Time on ice", path: "/toi"}
  ],

  head: '<link rel="icon" href="observable.png" type="image/png" sizes="32x32">',
  root: "src",

  // GitHub Pages project site: served under /pwhlstats/. Framework writes relative
  // links everywhere, so `base` is only used by the 404 page.
  base: "/pwhlstats/",

  interpreters: {".py": [python]},

  footer: "Data: PWHL / HockeyTech feeds. Built with Observable Framework.",
};
