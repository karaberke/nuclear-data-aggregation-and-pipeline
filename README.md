# Nuclear Data Cross-Section Explorer

Look up and compare evaluated neutron cross sections from the OECD NEA **JANIS**
tool, in a browser.

Everything runs as **one program in one container**: a Python backend that calls
JANIS and does the maths, and a Plotly Dash web interface served by that same
program. There is no separate frontend service and no reverse proxy to set up.

---

## Quick start

You need Docker. Everything else, including the JANIS runtime, is in the
repository (see [What you need first](#what-you-need-first)).

```bash
# 1. Build and start it
docker compose up --build

# 2. Check it is alive  (in another terminal)
curl http://localhost:8080/api/health

# 3. Open it
#    interface -> http://localhost:8080
#    API docs  -> http://localhost:8080/docs

# 4. Stop it
docker compose down
```

A healthy response looks like this:

```json
{"status":"ok","single_process":true,"pid":1,
 "query_budget_seconds":180,"max_concurrent_queries":2}
```

If you want to change how it behaves, jump to [Configuring it](#configuring-it).
The setting most people want is `JANIS_MAX_CONCURRENT_QUERIES` — how many people
can run a query at the same time.

---

## What you need first

**To run it with Docker (recommended):** Docker, with the Compose plugin. That
is all.

The JANIS runtime in `backend/Janis_all_jars/` 

The JANIS folder holds `Janis.jar` *and*
every `.jar` its manifest refers to:

```text
backend/
└── Janis_all_jars/
    ├── Janis.jar
    ├── nea-janis-domain.jar
    ├── nea-jdbc.jar
    └── ...the rest of the JANIS dependencies
```


The jars are needed for any *live* query, Docker or not — the path
`backend/Janis_all_jars/` is hardcoded in `backend/jar_runner.py` with no
environment override, and JANIS is run with that directory as its working
directory so the manifest `Class-Path` resolves.

**To run it without Docker:**

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A Java runtime that works with JANIS 4.1

No Node.js, npm, or build step. The interface is written in Python.

> JANIS 4.1 was built for Java 8. If it prints its version and then exits
> immediately, that is usually a Java compatibility problem or a missing
> `.jar` from the manifest.

---

## Running it

### With Docker — the normal way

```bash
docker compose up --build      # build then start, logs in your terminal
docker compose up -d --build   # same, but in the background
```

`--build` is only needed the first time, or after you change the code. Plain
`docker compose up` reuses the image you already built.

The app is on **http://localhost:8080**.

To watch the logs when it is running in the background:

```bash
docker compose logs -f
```

### Without Docker — for development

```bash
uv sync --frozen                            # install dependencies, once
uv run uvicorn backend.main:app --reload
```

Now it is on **http://localhost:8000** instead. `--reload` restarts the app
automatically when you edit a file, which is handy while working on the
interface.

This path needs Java. The JANIS jars are already in the repository.

### Building the image only

If you want to build without starting anything:

```bash
docker compose build
```

Or build the image directly with Docker, giving it your own name:

```bash
docker build -f backend/Dockerfile -t nuclear-data:1.0 .
```

Note the `.` at the end — the build must run from the repository root, because
it needs both `backend/`.

### Deploying it to a server

1. Copy the repository to the server, **including** `backend/Janis_all_jars/`.
2. Create a `.env` file with your settings (see below).
3. Start it in the background:

   ```bash
   docker compose up -d --build
   ```

`restart: unless-stopped` is already set, so the container comes back by itself
after a reboot or a crash.

**Two things to get right before real users arrive:**

- **Do not run more than one copy.** The app must stay at one container with one
  worker; see [Deployment limits](#deployment-limits). It refuses to start if
  you configure more.
- **Check your timeout chain.** A query can legitimately take minutes. If there
  is a load balancer or ingress in front of this app, either raise its read
  timeout or lower `JANIS_QUERY_BUDGET_SECONDS` below it. Details in
  [Deployment limits](#deployment-limits).

---

## Configuring it

Everything is set with environment variables. There are three ways to do it.

### 1. A `.env` file — best for a server

`docker-compose.yml` already reads every setting from the environment, so you
just create a `.env` file next to it:

```bash
cat > .env <<'EOF'
JANIS_MAX_CONCURRENT_QUERIES=4
JANIS_QUERY_BUDGET_SECONDS=90
QUERY_CACHE_MAX_MB=512
EOF
```

Check it was picked up before starting:

```bash
docker compose config      # prints the settings it will actually use
docker compose up -d
```

> `.env` is not currently in `.gitignore`. Add it if your settings should not
> be committed.

### 2. Inline, for a one-off

```bash
JANIS_MAX_CONCURRENT_QUERIES=1 docker compose up
```

### 3. Inline, running locally

```bash
JANIS_MAX_CONCURRENT_QUERIES=4 SERIES_CACHE_SIZE=64 \
  uv run uvicorn backend.main:app --reload
```

### Checking what is actually in effect

Never guess — ask the app:

```bash
curl http://localhost:8080/api/health
```

It also prints one line at startup:

```text
Deployment limits: single worker (pid=1); JANIS_MAX_CONCURRENT_QUERIES=4
concurrent JANIS queries; query budget=180s ...
```

### The settings

| Variable | Default | What it does |
| --- | --- | --- |
| `JANIS_MAX_CONCURRENT_QUERIES` | `2` | **How many people can run a query at the same time.** Each slot is one Java process, so this is mainly a memory decision. People over the limit wait their turn. |
| `JANIS_QUERY_BUDGET_SECONDS` | `180` | How long one query may take in total, *including* time spent waiting for a free slot. After this it gives up with a clear message. Keep it below any proxy timeout in front of the app. |
| `SERIES_CACHE_SIZE` | `16` | How many JANIS results to keep in memory. Bigger = fewer repeat JANIS calls. |
| `QUERY_CACHE_TTL_SECONDS` | `1800` | How long a user's loaded data stays cached (30 minutes). |
| `QUERY_CACHE_MAX_ENTRIES` | `32` | Cache size limit, by number of results. |
| `QUERY_CACHE_MAX_MB` | `256` | Cache size limit, by memory. Both limits apply. |
| `ENABLE_DASH` | `1` | Set to `0` to turn off the web interface and serve only the API. |
| `PLOTLY_FORCE_SVG` | unset | Set to `1` if charts do not appear on locked-down browsers or virtual desktops (no WebGL). |
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` for more detail in the logs. |
| `DASH_URL_PREFIX` | `/` | Where the interface is mounted. |
| `DASH_REQUESTS_PREFIX` | = `DASH_URL_PREFIX` | Only needed if a proxy rewrites the path. |
| `QUERY_CACHE_BACKEND` | `memory` | Only `memory` exists. Any other value stops the app on purpose. |
| `ALLOW_UNSAFE_MULTIWORKER` | unset | Do not set this. See [Deployment limits](#deployment-limits). |
| `SKIP_PERF_TESTS` | unset | Set to `1` to skip the speed test on a slow machine. |

`JANIS_QUEUE_DEPTH` and `JANIS_MAX_CONCURRENCY` are old names. They still work,
but log a warning telling you to use `JANIS_MAX_CONCURRENT_QUERIES`.

### How many people can use it at once?

Two different answers, because two different things are happening:

- **Running a new query** (pressing *Run comparison*) calls JANIS, which is
  slow — seconds to minutes. This is what `JANIS_MAX_CONCURRENT_QUERIES`
  limits. Extra people wait in line.
- **Exploring data already loaded** (changing filters, switching charts, moving
  sliders) does not call JANIS at all. Measured at 9–54 ms per action, so dozens
  of people can do this comfortably at the same time.

Start at `2`, then raise it only if your server has the memory for another
JVM per slot and you see people queueing.

### Things you cannot change with a setting

These are in the code and need an edit:

| What | Where |
| --- | --- |
| Max 5 series per comparison | `backend/charts.py` (`MAX_QUERY_SERIES`) |
| 60-second limit per JANIS call | `backend/jar_runner.py` (`JANIS_TIMEOUT_SECONDS`) |
| Which databases and MT codes are offered | `backend/jar_runner.py` (`DESIRED_DATABASES`, `DESIRED_DATASETS`) |
| KDE and binning constants | `backend/services/analytics.py` |
| Which quantities may be compared as a ratio | `backend/services/analytics.py` (`check_comparable`) |

---

## Stopping it

```bash
docker compose down       # stop and remove the container
docker compose stop       # just stop it; `docker compose start` resumes
```

Running locally with uvicorn instead? Press `Ctrl+C`.

`docker compose down` does **not** delete your image or any data. To do that,
see below.

---

## Removing the container and the image

Names used by this project:

- container: `nuclear-backend`
- image: `aa056-nucleardata-backend`

**The easy way** — remove the container and the image Compose built:

```bash
docker compose down --rmi local
```

**Also remove volumes** (this project has none by default, but harmless):

```bash
docker compose down --rmi local --volumes
```

**By hand**, if you built the image yourself with `docker build`:

```bash
docker rm -f nuclear-backend             # remove the container
docker rmi aa056-nucleardata-backend     # remove the image
```

**Check what is left:**

```bash
docker ps -a --filter name=nuclear       # containers
docker images | grep nucleardata         # images
```

**Reclaim disk space from old build layers:**

```bash
docker builder prune       # cached build layers
docker image prune         # untagged leftover images
```

> `docker system prune -a` also works but deletes unused images across *all*
> your projects. Be sure that is what you want.

---

## Testing it

```bash
uv run python -m unittest discover -s tests -v      # everything (278 tests)
```

Useful variations:

```bash
# one file
uv run python -m unittest tests.test_parity_analytics

# one test
uv run python -m unittest tests.test_backend.QueryValidationTests.test_more_than_five_series_are_rejected

# skip the speed test on a slow or busy machine
SKIP_PERF_TESTS=1 uv run python -m unittest discover -s tests
```

The tests need no Docker, no browser, no JANIS, and no network — JANIS is
stubbed out. They use Python's built-in `unittest`; there is no pytest.

What the main files cover:

| File | What it protects |
| --- | --- |
| `test_backend.py` | Query rules, parsing, eV→MeV, caching |
| `test_parity_analytics.py` | The maths, pinned to golden fixtures from the original TypeScript |
| `test_analytics_*.py` | Filtering, binning, KDE, ratio/difference |
| `test_analytics_comparison.py` | Ratio / difference: grid merging, crossings, undefined values |
| `test_dash_smoke.py` | The app starts and every element the UI needs exists |
| `test_dash_callbacks.py` | Callback helpers and input validation |
| `test_dash_integration.py` | Real browser wire protocol — catches bugs direct calls miss |
| `test_concurrency.py` | Several users at once; the app stays responsive |
| `test_deployment_guards.py` | The one-worker rule is enforced |
| `test_cache.py`, `test_exports.py`, `test_mt_labels.py`, `test_multigroup.py` | Caching, CSV output, labels, group structures |

If you add a new Dash callback, it **must** be wrapped with `@offload` (see
`backend/dash_ui/callbacks.py`). Without it the app quietly handles one request
at a time. `test_concurrency.py` is what catches that.

---

## Using the app

### Comparison modes

Pick a mode first, then your data:

| Mode | Database | Isotope | Dataset |
| --- | --- | --- | --- |
| Single series | One | One | One |
| Compare databases | 2–5 | One common to all | One common to all |
| Compare isotopes | One | 2–5 | One common to all |
| Compare datasets | One | One | 2–5 |

Only the dimension you are comparing becomes a multi-select. The other two stay
single. The dropdowns only offer options valid for *every* item selected, so you
cannot build a comparison that has no data. Switching modes clears your
selections.

The first three charts work in every mode. **Ratio / Difference** needs two
evaluations of the same nuclide and reaction, so it works in **Compare
databases** only — see that section.

Choose **Cross section** for values only, or **Cross section ± std. deviation**
to add uncertainty bars to the raw line chart. Then press **Run comparison**.

That is the only action that calls JANIS. Everything afterwards — filters,
sliders, switching charts — reuses the data already loaded.

Click a series pill to hide or show it. Hidden series are excluded from the
chart, the binning, the KDE, the ratio comparison, and the CSV export.

### Raw line

The unmodified energy/cross-section points. Uncertainties appear as bars when
requested. A missing uncertainty stays missing — it is never turned into zero.

### Binned bars

Builds one group structure shared by all visible series. An *N*-group structure
has *N+1* edges: group *i* spans `[edgeᵢ, edgeᵢ₊₁)`, except the last, which
includes its upper edge. The bar centre shown in the tooltip is for display only
(arithmetic mean on a linear axis, geometric on log) and is never used to assign
points to groups or to integrate.

**Group structure** chooses how the edges are made:

- **Automatic** — the slider (10–100 groups, default 40), plus optional
  minimum/maximum energy. Left blank, each bound follows the visible data. A log
  x axis gives log spacing.
- **Standard structure** — a bundled JANIS/NEA preset used exactly as published
  (WIMS69, 14MeV129, STAYSL140, VITAMIN-J175, SAND-II640, SAND-II725).
- **Custom edges** — type an ascending, comma- or space-separated list. The box
  always shows the edges currently in use, so you can adjust one number without
  retyping everything. Clear it to return to Automatic.

**Bar value** chooses what each group reports:

- **Mean** — the plain average of whichever raw points land in the group. This
  depends on how densely that region happens to be sampled, not on the shape of
  the curve.
- **Group Average / Density** — the energy-averaged cross section
  `∫σ(E)dE / ΔE`, obtained by trapezoidal integration across each series' own
  points in physical energy, always, even on a log axis. **Still in barns**, not
  a probability density. It is defined for any group that overlaps a series'
  data, even one with no sample inside it, as long as it lies between two real
  points.

  Because of that, `Σ averageᵢ × widthᵢ` equals `∫σ(E)dE` over the overlap of
  your edges and the series' data range — exactly, for uniform, standard, or
  custom edges. You only recover the whole series integral if the edges span its
  full range. And since integration happens in physical energy, "preserves the
  integral" means the *numbers* are preserved; on a log x axis the on-screen bar
  width is proportional to `log(edgeᵢ₊₁) − log(edgeᵢ)`, not `ΔE`, so the visible
  areas are not preserved there.

**Coverage** is how much of a group's width the series actually has data for.
Groups with partial coverage are drawn with diagonal hatching and lower opacity,
so a gap in the data is never hidden. The exact percentage is in the tooltip and
the export.

MT codes are labelled with their ENDF-6 reaction name (`MT 102 — (n,γ) Radiative
capture`). A code outside the bundled lookup is shown as a plain `MT <number>`
rather than a guessed name.

**Export graph CSV** writes one row per group — including partial and empty ones
— with both values, the raw integral, coverage, the MT label, and which
structure produced the edges.

Flux-weighted multigroup averaging is **not** implemented; every value here uses
an unweighted spectrum.

### KDE

Estimates the distribution of cross-section values per series using Gaussian
kernels and Silverman's bandwidth. Each curve is normalised on its own. The
slider multiplies the automatic bandwidth from `0.25×` to `4.0×`.

It evaluates 200 positions and samples at most 5,000 values per series, chosen
deterministically so the same data always gives the same curve. Log-scale KDE is
computed in log space with the Jacobian correction.

### Ratio / Difference

Compares one evaluation against another as a function of energy, so a
disagreement you cannot see on two overlaid curves becomes obvious. Two
libraries that look identical on a log axis can differ by 40%; this chart shows
that as a line 40% below the baseline.

**Metric** chooses what the line reports:

- **Ratio** — `σ_comparison / σ_reference`, drawn around a baseline of **1**.
  Above the line the comparison is larger, below it the reference is. It is
  dimensionless.
- **Percent Difference vs Reference** — `100 × (σ_comparison − σ_reference) /
  σ_reference`, drawn around **0%**. Same reading, signed: `+20%` means the
  comparison is a fifth larger. A ratio of 2.00 is `+100%`; 0.80 is `−20%`.

This is the reference-relative percent difference. The *symmetric* form
`200(A−B)/(|A|+|B|)` is deliberately not offered, because it answers a different
question and the two are easy to confuse.

**Which way round.** The calculation always divides **comparison ÷ reference**,
and the caption next to the controls spells out the pair currently plotted —
`JEFF-3.3 / ENDF/B-VIII.0`. Getting this backwards inverts the reading of every
point, which is why it is stated on screen rather than implied.

Load two series and the first becomes the reference, the second the comparison.
**Swap** exchanges them; the line flips about the baseline and the ratio becomes
its reciprocal. With more than two loaded, pick one **Reference** and any number
of **Comparison** series — you get one line per comparison, each in that series'
own colour.

**It only works in Compare databases mode.** A ratio is only meaningful between
two evaluations of *the same thing*, so the nuclide and the reaction channel
must match. Compare isotopes varies the nuclide and Compare datasets varies the
MT channel — exactly the two things that have to be equal — so those modes are
refused with a message naming what differs, rather than plotting a number that
looks fine and means nothing.

**Different energy grids are handled properly.** Two evaluations almost never
share a grid, and values are never paired up by position in the file. Both
curves are restricted to the energy range they have in common, their grids are
merged into one, and each is interpolated onto it (piecewise-linear, the same
rule the binned integration uses). Nothing is evaluated outside the shared
range, so no value is ever extrapolated.

**Crossings are exact.** Wherever the two curves swap places, the exact energy
is solved for and added to the grid, so the comparison line meets its baseline
precisely there. Multiple crossings all survive — comparing Co59 capture between
ENDF/B-VIII.0 and JEFF-4.0 finds several thousand across the resonance region,
and none is smoothed away. The count is reported under the chart.

**Where the comparison is undefined**, the line breaks. A ratio needs a nonzero
reference, so at energies where the reference is zero, missing, or has a
discontinuity, the result is left blank rather than being reported as infinity
or — worse — as zero, which would read as "100% smaller". Both original cross
sections are still shown in the export, with a reason. Those points are counted
under the chart.

**Axes.** The x-axis follows the usual Linear/Log control. The y-axis is
**always linear and cannot be changed**: percent difference is signed, and a log
axis would silently drop every energy at which the reference is the larger
evaluation. Large ratios are never clipped — the baseline is always kept in
view, and you can zoom freely.

The **Min/Max cross section** filters do not apply to this chart. They remove
points by value, which can empty out the middle of a curve and make the
comparison span energies where one evaluation has no data at all. The energy
filters work normally, and the notice under the chart says when value filters
were ignored.

**Export graph CSV** writes one row per grid point: energy, both series names,
both cross sections, ratio, percent difference, which series is larger, whether
the point is valid, why not if it is not, and whether it is a crossing. Rows
where the ratio is undefined are kept with the calculated fields blank.

### Filters and scales

Optional minimum and maximum for energy (MeV) and cross section (barns). Ranges
include their endpoints, and the minimum must be below the maximum. Filters
commit when you leave the box or press Enter.

Each chart remembers its own axes, both defaulting to **Log**:

- **Linear** keeps zero and negative values.
- **Log** drops non-positive values and tells you how many it dropped.

The one exception is **Ratio / Difference**, whose y-axis is fixed to Linear and
whose Y-scale control is therefore disabled. The Min/Max **cross section**
filters are also ignored there; see that section for why.

**Reset controls** clears the filters, returns every chart to its default axes,
and restores 40 groups, `1.00×` bandwidth, and the **Ratio** metric.

---

## API

The interface uses in-process function calls, not these endpoints — but they are
a supported way to script the app.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health, plus the settings in effect |
| `GET` | `/api/databases` | Databases available from JANIS |
| `GET` | `/api/isotopes?database=...` | Isotopes common to every database given |
| `GET` | `/api/datasets?database=...&isotope=...` | Datasets common to every pair |
| `GET` | `/api/reaction-types` | `xs` and `xs_stddev` |
| `POST` | `/api/cross-sections/query` | Fetch one or more series |
| `GET` | `/api/table` | Raw JANIS table lines |

Interactive docs: **http://localhost:8080/docs** (or `:8000` running locally).

### Fetching data

`POST /api/cross-sections/query`

```json
{
  "databases": ["TENDL-2019", "ENDF/B-VIII.0"],
  "isotopes": ["Co59"],
  "datasets": ["MT102"],
  "field": "SIG",
  "reaction_type": "xs_stddev"
}
```

The same rule as the interface: all three arrays must be non-empty and unique,
at most one may hold multiple values, and the result may not exceed five series.

```json
{
  "series": [
    {
      "key": "TENDL-2019|Co59|MT102",
      "database": "TENDL-2019",
      "isotope": "Co59",
      "dataset": "MT102",
      "points": [
        {
          "energy_MeV": 0.001,
          "cross_section_barns": 2.4,
          "cross_section_stddev_barns": null
        }
      ]
    }
  ]
}
```

Every requested series must have cross-section data. Standard deviations are
`null` where JANIS reports none at that energy.

### Asking for intersections

Repeat a parameter to intersect:

```bash
curl --get http://localhost:8080/api/isotopes \
  --data-urlencode "database=TENDL-2019" \
  --data-urlencode "database=ENDF/B-VIII.0"
```

That returns only isotopes present in both databases.

JANIS reports energy in eV; these endpoints convert to MeV. Cross sections and
standard deviations stay in barns.

`/api/cross-section`, `/api/cross-section/export`, and `/api/fields` were
removed at cutover. Use the batch route above; `field` is always `SIG`.

---

## Deployment limits

Two limits are accepted here, and both are **enforced in code**, not just
documented — see `backend/deployment.py` and `tests/test_deployment_guards.py`.

### 1. One container, one worker

The JANIS gates and the caches live inside a single process. Run two workers or
two replicas and you get twice the Java processes you asked for — exactly the
resource exhaustion the limit exists to prevent.

Enforced four ways: `--workers 1` in the Dockerfile, a start-up check that
refuses to boot if `WEB_CONCURRENCY` > 1, `deploy.replicas: 1` in Compose, and a
`single_process` flag on `/api/health`.

To scale beyond one container you would have to replace **both** the cache
(`services/cache.py::get_cache`, already behind a `DataCache` Protocol) **and**
the JANIS gate (a distributed lock, or a dedicated single-replica worker).
Doing one without the other is worse than doing neither.
`ALLOW_UNSAFE_MULTIWORKER=1` disables the guard once that work is genuinely
done.

### 2. Long queries and proxy timeouts

One comparison can take minutes — up to 10 JANIS calls for a 5-series
`xs_stddev` query, each capped at 60 seconds.

This repository ships **no proxy of its own**, so every read timeout in your
deployment belongs to something else:

| Layer | Setting |
| --- | --- |
| nginx / ingress-nginx | `proxy_read_timeout` / `nginx.ingress.kubernetes.io/proxy-read-timeout` |
| AWS ALB | `idle_timeout.timeout_seconds` |
| GCP backend service | `timeoutSec` |
| Azure Application Gateway | `requestTimeout` |
| Traefik | `respondingTimeouts.readTimeout` |
| Envoy | `route.timeout` |

**Cloudflare caps origin responses at 100 s on non-Enterprise plans (error 524),
and that cap cannot be raised.**

If an outer proxy gives up first, the browser sees a 504 or 524 while JANIS
keeps running — holding a slot with nobody left to receive the answer, so the
next person queues behind a dead request.

**So `JANIS_QUERY_BUDGET_SECONDS` is the lever, not the proxy.** Set it below the
smallest timeout in your chain and the app fails fast with a message a user can
act on. Treat the 180-second default as a placeholder until you know your real
ingress timeout.

### A note on concurrency

Every Dash callback is wrapped with `@offload` so its work happens on a worker
thread. This is load-bearing: Dash's FastAPI backend runs a plain callback
directly on the event loop, which would make the whole app handle one request at
a time — measured at three concurrent queries taking 9.0 s instead of 3.0 s,
with `/api/health` unable to answer at all until they finished. `test_concurrency.py`
guards this.

---

## Project structure

```text
.
├── backend/
│   ├── main.py                 # API routes; mounts the Dash UI last
│   ├── charts.py               # Batch query API, parsing, parsed-table cache
│   ├── jar_runner.py           # Runs JANIS, holds the concurrency semaphore
│   ├── deployment.py           # The enforced deployment limits and settings
│   ├── services/               # Analysis, independent of any web framework
│   │   ├── analytics.py        # Filtering, binning, integration, KDE, ratios
│   │   ├── cache.py            # Session cache behind a DataCache Protocol
│   │   ├── query_store.py      # Query handles, the queue gate, the budget
│   │   ├── exports.py          # CSV generation
│   │   ├── errors.py           # Errors that do not depend on HTTP
│   │   ├── multigroup.py       # Bundled JANIS/NEA group structures
│   │   └── mt_labels.py        # ENDF-6 MT reaction labels
│   ├── dash_ui/                # The web interface
│   │   ├── app.py              # create_dash(server) — native ASGI backend
│   │   ├── layout.py           # Page structure and element ids
│   │   ├── callbacks.py        # Callback graph (all @offload; no numpy/math)
│   │   ├── inputs.py           # Input coercion and validation
│   │   ├── presenters.py       # Service results -> display shapes
│   │   ├── figures.py          # Plotly figures
│   │   ├── components.py       # Reusable layout pieces
│   │   └── assets/             # app.css, favicon
│   └── Janis_all_jars/         # JANIS runtime (committed)
├── tests/
│   └── fixtures/parity/        # Golden fixtures pinning the maths
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

---

## Troubleshooting

**`janis error`** — Usually Java. JANIS 4.1 was built for Java 8, so a much
newer JRE is the first thing to check; then that the NEA data source is
reachable. The jars themselves ship with the repository, so a missing one means
something removed it.

**"No common options"** — The databases or isotopes you picked share no valid
value. Choose a different combination.

**A query fails with "exceeded the query budget"** — It genuinely took too long.
Use fewer series, narrow the selection, or raise `JANIS_QUERY_BUDGET_SECONDS`
(but keep it under your proxy timeout).

**"Waited … for a free query slot"** — Everyone else is querying. Raise
`JANIS_MAX_CONCURRENT_QUERIES` if the server has memory to spare.

**504 or 524 on a long query** — A proxy gave up before the app did. Lower
`JANIS_QUERY_BUDGET_SECONDS` below that proxy's timeout.

**The chart goes empty** — Your filters exclude everything, or you are on a Log
axis with non-positive values. Press **Reset controls**, or switch to Linear.

**Charts do not appear at all (virtual desktop, locked-down browser)** — Set
`PLOTLY_FORCE_SVG=1`.

**The page sits on "Loading..."** — The browser is fetching Dash's files from
the wrong path. Check `DASH_REQUESTS_PREFIX` matches the public URL.


**The app feels like it handles one person at a time** — Check every callback in
`backend/dash_ui/callbacks.py` has `@offload`, then run
`uv run python -m unittest tests.test_concurrency`.

**Port 8080 already in use** — Change the left-hand number in the `ports:` line
of `docker-compose.yml`, e.g. `"9090:8000"`.
