# CTMC Entropy Production Explorer — Build Plan

> **Status:** Living document for implementing agents
> **Owner:** the project author
> **Last updated:** 2026-09-07 (design locked after grilling session)

---

## 1. Purpose

This document is the authoritative build spec for turning a research
internship's Numba-JIT CTMC simulator and entropy-production estimators into a
**polished, interactive web app** — a software-engineering portfolio piece.

**Audience (decided):** the author is hunting **SWE roles**, not research
roles. A recruiter should open the app, play with it for 60 seconds, and
conclude *"this person can code, and has real research experience."* The
engineering depth — real simulation core, typed API, async story (V2),
tests, CI, Docker, interactive visualization — is the talking point in
interviews. The physics is the *domain*, not the point.

**Definition of done (decided):** **MVP is shipped.** The author is applying
to jobs now; a live working demo comes first, stretch goals strictly after.

**We are not rebuilding the science.** The estimators and Gillespie simulator
are correct and proven. We wrap them in an API + frontend.

---

## 2. Project summary

The author has a Python module (`ctmc_simulator.py`, preserved verbatim in
`docs/source/reference_ctmc_simulator.py`) containing:

| Asset | What it does | Where to port |
|---|---|---|
| `simulate_trajectory_core`, `precompute_transition_data`, `simulate_single_trajectory` | Gillespie CTMC simulation with metastate coarse-graining | Copy JIT functions **unchanged**; refactor the wrapper into a pure function |
| `kth_order_estimator` | Generic EPR estimator from k+1-length sequence statistics | Port to `estimators.py` |
| `repeated_transitions_estimator`, `analyze_transitions` | EPR estimator exact for unicyclic systems | Port to `estimators.py` |
| `thermodynamic_uncertainty_relation_estimator`, `count_transitions_fast`, `find_snippet_boundaries_fast` | TUR lower-bound estimator (hard-coded A/B/C topology) | Copy JIT helpers unchanged; port wrapper into `estimators.py`. **A general-TUR is a user-led stretch goal, not agent work** (see Phase 6). |
| `generate_rate_matrix_for_parallel_tracks`, `get_true_EPR_for_parallel_tracks` | 6-state parallel-tracks model with **closed-form exact EPR** — the flagship demo asset | Port to `models/parallel_tracks.py` |
| `mathematica_to_numpy_array`, `save_*_to_text`, `process_simulation_results` | CLI/notebook plumbing | **Do not port** |
| print statements throughout | Debug output | **Remove**; raise on error instead |

### Key design constraint

**Embed whatever was a `print("Warning: ...")` and continue into a proper
`ValueError` raise.** The original code degrades gracefully on bad input; the
web app must not. Validation must be explicit and fail loudly (HTTP 422).

---

## 3. Architecture

```
┌─────────────────────┐        HTTPS/JSON        ┌──────────────────────────┐
│  React + TS frontend │ ───────────────────────▶ │  FastAPI backend          │
│  (Vercel, free)      │ ◀─────────────────────── │  (Render free web service│
│  Recharts + KaTeX +   │                          │   + GH Actions keep-warm) │
│  React Flow           │                          └──────────────────────────┘
└─────────────────────┘                                    │
                                                      ctmc_core/ (ported,
                                                      refactored simulator +
                                                      estimators, Numba JIT)
```

**Hosting (decided after 2026 pricing research):**
- **Backend:** Render free web service (0.1 CPU / 512 MB). It **spins down
  after 15 idle minutes** and takes ~1 min to wake, with an ephemeral
  filesystem and 750 instance-hours/month (**pooled per Render *workspace*,
  not per service** — the keep-warm math (~744/750h) assumes this workspace
  hosts exactly one web service; a second free service there would break it).
  Mitigation: a **GitHub Actions keep-warm ping every 5 minutes** to
  `/api/health`. Caveat: GitHub auto-disables scheduled workflows after 60
  days of repo inactivity; **the author must touch the repo at least every 60
  days** (committed to).
- **Numba JIT reality:** JIT compiles lazily on first call, and Render can
  restart the service at any point (ephemeral disk → JIT cache wiped). So the
  backend **warm-ups at process startup** (P2-T4) and the frontend shows a
  distinct **"waking up the simulator…"** state via `/api/health`.
- **Keep-warm margin:** GitHub Actions' `schedule:` trigger is best-effort and
  can slip several minutes under platform load, so a 10-minute ping is not a
  safe margin against a 15-minute idle window. Ping every 5 minutes (P5-T3).
  Occasional cold starts still happen and are covered by the waking-up UX.
- **Frontend:** Vercel free (Hobby). Static SPA, non-commercial portfolio use
  is fine under the free terms.
- **No database in MVP.** A Neon free Postgres job-store is **V2 only, after
  MVP** (see §5 V2) and requires **no hosting change** — the backend just
  connects over the network with a connection string.

---

## 4. Tech stack (decided)

- **Backend:** Python 3.11, FastAPI, Uvicorn, NumPy, Numba, Pydantic v2,
  pytest, slowapi (per-IP rate limiting) + a global concurrency semaphore
- **Frontend:** React + TypeScript (Vite), Tailwind CSS, shadcn/ui, Recharts,
  react-katex, **React Flow (`@xyflow/react`)** for the Markov-chain diagram
- **CI:** GitHub Actions — backend pytest + frontend `tsc`/build on every push
- **Deployment:** backend → Render free web service + GH Actions keep-warm
  ping + Docker; frontend → Vercel free
- **V2 database:** Neon free Postgres (no hosting change)
- **No user accounts.** Stateless playground in MVP.

---

## 5. Feature scope

### MVP (must ship)

1. **Parallel Tracks demo** (flagship): sliders/inputs for `α, β, u₁, w₁, u₂,
   w₂`. Run simulation → system diagram (React Flow), trajectory viz,
   per-metastate occupancy, and an EPR chart comparing the **k-th order
   estimator across a sweep of k (k = 1–4, one simulation run)** against the
   closed-form true EPR as a horizontal reference line.
2. **Custom rate matrix mode**: N×N matrix input (N ≤ 12 capped) + metastate
   groups + run; k-th order EPR estimate + trajectory/occupancy viz. The
   matrix editor **auto-generates a drawing of the Markov chain** the matrix
   represents (shared React Flow component). Server validates the matrix
   (columns sum to zero) and the metastate groups (every state in exactly one
   group) and returns a clear 422 otherwise.
3. **Trajectory visualization**: state-vs-time step plot (fine-grained,
   decimated server-side to ~2,000 points), bar chart of total time / visit
   count per metastate.
4. **"How it works" page**: recruiter-readable explanation of CTMCs, entropy
   production, and each estimator. **Content split (decided):** equations are
   sourced **directly from the existing docstrings** (verbatim, no
   re-derivation); the **narrative prose and research citations are written by
   the author** (who knows the physics and has the papers). The agent builds
   the scaffold + KaTeX rendering only.
5. **README**: architecture diagram, live demo link, "why I built this"
   framing tied to the internship + SWE engineering story.

### Stretch (only after MVP deployed and working; ordered)

6. Repeated-transitions estimator UI for the tracks demo.
7. Convergence-vs-length chart (re-run simulation at several trajectory
   lengths; plot estimator error vs. length on log-log).
8. **Shareable permalinks via URL query params** (no persistence needed).
9. **General-TUR estimator — THE LAST stretch goal, author-implemented.** The
   current TUR is hard-coded to a 3-metastate A/B/C topology; a general TUR
   requires a derivation the implementing agent **cannot** produce. The author
   must bring the math; this item exists as the "what to work on when
   everything else is done" slot. The agent may only wire up whatever formula
   the author supplies.

### V2 — after stretch (persistence, the DB interview story)

10. **Neon free Postgres job-store**: simulations become **async jobs**
    (POST → job ID → poll for result), with run history and **durable
    ID-based permalinks**. This is the deliberate engineering addition that
    justifies a database (long-running stochastic sims shouldn't block HTTP),
    plus migrations in CI. No hosting change.

### Explicitly NOT in scope

- Save-to-file endpoints / the original `save_*` /
  `process_simulation_results` file-I/O functions
- User accounts
- Any estimate of a DB "just because" — see V2 for the justified version

---

## 6. Backend API design

All endpoints return JSON. No endpoint writes to disk (MVP).

### `POST /api/simulate`

Request:
```json
{
  "rate_matrix": [[...]],
  "metastate_groups": {"track1": [0,2,4], "track2": [1,3,5]},
  "max_length": 5000,
  "initial_state": 0,
  "estimator": {"type": "kth_order", "k_values": [1,2,3,4]}
}
```

Response:
```json
{
  "trajectory_preview": {"states": [...], "times": [...]},
  "waiting_times_summary": {"track1": {"total_time": 12.3, "visits": 41, "avg": 0.3}, ...},
  "epr_estimates": {"1": 0.42, "2": 0.44, "3": 0.44},
  "true_epr": 0.446,
  "meta": {"n_transitions": 5000, "final_time": 118.2, "compute_ms": 340}
}
```

**Load protection:** slowapi per-IP rate limit **plus a global concurrency
semaphore** (max 2–3 simultaneous simulations; beyond that respond HTTP 503
"simulators busy, retry"). A 0.1 CPU box queues badly when several *different*
IPs hit `/api/simulate` at once (shared link → recruiter forwards → Discord),
which per-IP limits cannot stop.

**Caps (server-side, enforced with 422 + clear message):**
- `max_length` ≤ 20,000
- `n_states` ≤ 12 (deliberate — keeps single runs fast on the free tier)

**Validation (all 422 with a clear message, not warnings):**
- Rate matrix must be square; columns must sum to zero (report offending
  state index). Use the existing check in `simulate_single_trajectory`, but
  raise instead of print.
- All entries must be **finite** (reject NaN/Inf with the offending entry).
- All **off-diagonal** entries must be **non-negative**.
- Diagonal (exit-rate) entries must be strictly positive per state. The JIT
  core's "negative exit rate → print warning and break" branch (see
  `simulate_trajectory_core`) must become an **explicit raise**: a
  numerically-adjacent-but-column-balanced matrix must return a clear 422, not
  a silently truncated/empty trajectory.
- Metastate groups must cover **every** state exactly once (duplicate or
  missing assignments are rejected).

**Decimation:** `trajectory_preview` must be decimated server-side (max
~2,000 points via uniform stride or LTTB downsampling), not shipped raw. The
estimators always run on the full-resolution trajectory.

### `GET /api/presets/parallel-tracks`

Returns default `α, β, u₁, w₁, u₂, w₂` values and the generated rate matrix so
the frontend doesn't hardcode simulation parameters.

### `GET /api/health`

Returns 200 once Numba warm-up has completed. Used by the frontend to show a
"waking up the simulator…" state, by the keep-warm ping, and by Render's
health check.

---

## 7. Frontend structure

```
/                     landing page: what this is, link into the playground
/playground/tracks     Parallel Tracks demo (flagship)
/playground/custom     Custom rate matrix mode
/how-it-works          math explainer with KaTeX
```

- Landing page reads like a **portfolio case study**, not a tool homepage:
  1–2 sentences on the internship context, what problem entropy production
  estimation solves, then a clear CTA into the tracks demo. **Copy flow
  (decided): the agent drafts, the author rewrites in their own voice.**
- **Loading/compute states matter more than in most demos** (JIT warm-up +
  Render spin-down): show a distinct "waking up" state (from `/api/health`)
  vs. a per-run "simulating…" spinner. A cold first hit is expected and
  handled gracefully, never a broken page.
- **`RateMatrixGraph`** (React Flow): auto-drawn directed graph of the rate
  matrix — nodes = states colored by metastate group, edge arrows labeled
  with their rate, pan/zoom/drag. Reused on `/playground/custom` (dynamic)
  and `/playground/tracks` (the 6-state system diagram).
- Charts (all Recharts): staircase trajectory plot, metastate occupancy bar
  chart, EPR comparison chart (estimated points vs. true-EPR reference line).
- **Responsive:** landing and how-it-works are fully mobile-responsive.
  `/playground/*` renders a graceful **"best viewed on desktop"** fallback on
  narrow viewports (<768px) rather than a broken layout (recruiters use
  laptops).

---

## 8. Task breakdown

Work through phases in order. **Each task = a separate commit.**

### Phase 0 — Repo & scaffolding
- [ ] P0-T1: Create monorepo with `/backend` and `/frontend` directories, root
      README stub, `.gitignore`, license. _(Done — repo exists on GitHub:
      `skibkitty/ctmc-entropy-explorer`.)_
- [ ] P0-T2: Scaffold FastAPI app in `/backend` with `/api/health` returning
      `{"status": "cold"}` initially.
- [ ] P0-T3: Scaffold Vite + React + TS app in `/frontend` with Tailwind and
      shadcn/ui installed and configured.
- [ ] P0-T4: Set up CORS on the backend for the frontend's dev and prod
      origins.
- [ ] P0-T5: **GitHub Actions CI**: backend `pytest` + frontend
      `tsc`/`build` on push/PR. (SWE-signal; runs from the first commit.)

### Phase 1 — Port and refactor simulation core
- [ ] P1-T1: Copy JIT functions (`simulate_trajectory_core`,
      `precompute_transition_data`, `find_snippet_boundaries_fast`,
      `count_transitions_fast`) into `backend/ctmc_core/simulation.py`
      **unchanged**.
- [ ] P1-T2: Refactor `simulate_single_trajectory` into a pure function that
      returns a structured result object and raises `ValueError` on invalid
      input (no printing/warn-and-continue). The JIT core's "negative exit
      rate → print and break" path must become an explicit raise (non-positive
      or non-finite exit rate → error).
- [ ] P1-T3: Port `kth_order_estimator`, `repeated_transitions_estimator`,
      `analyze_transitions`, `thermodynamic_uncertainty_relation_estimator`
      into `backend/ctmc_core/estimators.py`, same treatment (no prints, raise
      on bad input, return values not side effects). TUR wrapper is ported for
      completeness; it is **not** exposed in the MVP or stretch beyond item 9
      (user-led).
- [ ] P1-T4: Port `generate_rate_matrix_for_parallel_tracks` and
      `get_true_EPR_for_parallel_tracks` into
      `backend/ctmc_core/models/parallel_tracks.py`.
- [ ] P1-T5: Port trajectory decimation as a new function (uniform-stride or
      LTTB) for `trajectory_preview`.
- [ ] P1-T6: pytest unit tests:
      (a) parallel-tracks EPR converges to true EPR as trajectory grows, ≥2
      parameter sets;
      (b) rate matrix normalization rejects malformed matrix;
      (c) `kth_order_estimator` returns 0.0 on degenerate trajectory without
      raising.

### Phase 2 — Backend API
- [ ] P2-T1: Pydantic request/response models per Section 6.
- [ ] P2-T2: Implement `POST /api/simulate` wiring Phase 1 functions, with
      input caps and full matrix validation (square, finite, off-diagonal ≥ 0,
      columns sum to zero) plus group validation.
- [ ] P2-T3: Implement `GET /api/presets/parallel-tracks`.
- [ ] P2-T4: Numba warm-up on app startup (run one tiny simulation through
      every JIT'd function); flip `/api/health` → `{"status": "ready"}` once
      complete.
- [ ] P2-T5: Rate limiting (slowapi per-IP) **and a global concurrency
      semaphore** (max 2–3 concurrent sims → 503 "busy, retry") on
      `/api/simulate`.
- [ ] P2-T6: FastAPI TestClient integration tests for all endpoints, including
      422 error paths.

### Phase 3 — Frontend core
- [ ] P3-T1: Landing page with case-study framing (draft; author rewrites).
- [ ] P3-T2: Reusable rate-matrix editor component (grid input + inline
      validation).
- [ ] P3-T3: **`RateMatrixGraph`** component (React Flow): nodes colored by
      metastate, rate labels on edges, auto-layout, pan/zoom/drag.
- [ ] P3-T4: Trajectory staircase chart component (Recharts).
- [ ] P3-T5: Metastate occupancy bar chart component.
- [ ] P3-T6: EPR comparison chart component (estimate vs. true-EPR reference
      line, or small table when no ground truth).
- [ ] P3-T7: API client with typed request/response models matching Section 6,
      plus loading/error/cold-start ("waking up") states.
- [ ] P3-T8: React error boundary around the playground results panel — a
      malformed response or a chart edge case blanks only the result area,
      never the whole page mid-demo.

### Phase 4 — Playground pages
- [ ] P4-T1: `/playground/tracks` — sliders seeded from
      `/api/presets/parallel-tracks`, "Run simulation" button, system diagram
      via `RateMatrixGraph`, results via Phase 3 chart components.
- [ ] P4-T2: `/playground/custom` — matrix editor + `RateMatrixGraph` preview
      + per-state metastate group assignment (dropdown, full coverage enforced)
      + run + results.
- [ ] P4-T3: `/how-it-works` with KaTeX equations sourced from the docstrings
      (scaffold only; author supplies prose + citations).

### Phase 5 — Polish & deploy
- [ ] P5-T1: Responsive pass; "best viewed on desktop" fallback for
      `/playground/*` on narrow viewports.
- [ ] P5-T2: **Dockerfile** for the backend (linux/amd64); deploy to **Render
      free web service**.
- [ ] P5-T3: **GitHub Actions keep-warm workflow** pinging `/api/health`
      **every 5 minutes** (cron `*/5 * * * *`). Render's window is 15 idle
      minutes and the `schedule:` trigger is best-effort, so every-5-min is
      the margin — 10-min is not safe. Occasional cold starts still happen and
      are covered by the waking-up UX. Note the 60-day repo-activity caveat
      for the author.
- [ ] P5-T4: Deploy frontend to **Vercel** (free), pointed at the deployed
      backend.
- [ ] P5-T5: Root README: what it is, screenshot/GIF, architecture diagram,
      live link, local dev instructions, "ported from internship research
      code" + SWE engineering story.
- [ ] P5-T6: Smoke-test the full deployed flow end-to-end (cold start →
      health → run each demo → check chart rendering → wake-from-sleep).
- [ ] P5-T7: **Link-preview polish** — OG/Twitter meta tags (title,
      description, image) in the landing page `<head>`. This is a link sent
      directly in emails/LinkedIn/Discord; a bare URL with no preview card
      looks unfinished.
- [ ] P5-T8: Lightweight privacy-respecting analytics (Vercel Analytics —
      included free on Hobby, or Plausible). Gives an interview-ready data
      point ("N people ran the tracks demo, here's the parameter range they
      explored") instead of just asserting the project got attention.

### Phase 6 — Stretch goals (only after MVP deployed; ordered)
- [ ] P6-T1: Repeated-transitions estimator UI for the tracks demo.
- [ ] P6-T2: Convergence-vs-length chart (log-log, estimator error vs.
      trajectory length).
- [ ] P6-T3: Shareable permalinks via URL query params.
- [ ] P6-T4: **General-TUR estimator — LAST. Author-implemented.** The agent
      cannot derive it; the author supplies the math, the agent wires it up.

### Phase 7 — V2 (persistence, the DB interview story)
- [ ] P7-T1: Neon free Postgres; migrations setup (alembic or similar) in CI.
- [ ] P7-T2: Async job-store: `POST /api/jobs` → job ID → poll/result;
      run history endpoint.
- [ ] P7-T3: Durable ID-based permalinks (replaces/extends query-param links).
- [ ] P7-T4: README architecture update documenting the async design decision
      (why a DB now: long sims shouldn't block HTTP).

---

## 9. Design decisions — settled in the grilling session (2026-09-07)

These were the plan's open questions; all have explicit answers. **Do not
relitigate them.**

1. **Backend vs. full client-side port:** ✗ closed — real Python backend
   (Section 3). No TS reimplementation of the estimators.
2. **Hosting:** ✗ closed — Render free web service + GH Actions keep-warm
   ping; frontend on Vercel free. Fly.io ruled out (no free tier for new
   accounts as of 2026). Author must touch the repo within every 60 days to
   keep the ping alive.
3. **Domain/branding:** generic host subdomains; no custom domain.
4. **Custom rate matrix state cap:** 12 states, confirmed.
5. **Database:** V2 only — Neon free Postgres job-store, after MVP/stretch.
   No hosting change. (Answer to "does it need a hosting change?": only if
   file-backed SQLite; Neon doesn't.)
6. **Audience framing:** SWE recruiters; engineering depth is the product.
7. **New feature:** auto-drawn Markov chain (`RateMatrixGraph`, React Flow)
   on both playground pages.
8. **TUR:** fixed-preset stretch dropped; general-TUR is the last stretch
   goal and is **author-implemented**.
9. **MVP meta:** GH Actions CI + Dockerfile + meaningful tests all in MVP.
10. **Content split:** equations from docstrings; author writes prose +
    citations; landing copy drafted by agent, rewritten by author.
11. **Estimator sweep:** k = 1–4 single-run k-sweep in MVP; convergence-vs-
    length stays as stretch.
12. **Custom-mode metastate assignment:** per-state dropdown, full coverage
    required (422 otherwise).
13. **Keep-warm interval:** every 5 minutes (`*/5` cron). Render's idle window
    is 15 min and GH Actions `schedule:` is best-effort, so 10-min is not a
    safe margin; occasional cold starts remain and are covered by the
    waking-up UX.
14. **Load protection:** per-IP rate limit **plus** a global concurrency
    semaphore (2–3 concurrent sims → 503). A shared link hits the box from
    many IPs at once; per-IP limits can't stop that queueing a 0.1 CPU.
15. **Matrix validation depth:** finite entries, off-diagonal ≥ 0, exit rates
    > 0 — the JIT core's "negative exit rate → break" branch becomes an
    explicit 422, never a truncated trajectory.
16. **Render workspace budget:** 750 free instance-hours are pooled per
    workspace, not per service — keep-warm math assumes this workspace hosts
    only this one web service.

---

## 10. Porting notes (operational guidance for the implementing agent)

### Source-of-truth mapping

The authoritative source file is `docs/source/reference_ctmc_simulator.py`
(preserved verbatim). Use it as the reference. Trust it over any paraphrase in
this document.

### Function placement (target tree)

```
backend/
├── ctmc_core/
│   ├── simulation.py          # JIT core + refactored simulate_single_trajectory
│   ├── estimators.py          # kth_order, repeated_transitions, TUR
│   ├── decimation.py          # new: uniform-stride / LTTB downsampling
│   └── models/
│       └── parallel_tracks.py # rate matrix + true EPR
├── api/                       # FastAPI app, routes, schemas
├── tests/                     # pytest + TestClient integration tests
└── main.py                    # app entrypoint, warm-up logic
```

### Ground rules for the port

1. **JIT functions copied verbatim.** `simulate_trajectory_core`,
   `precompute_transition_data`, `find_snippet_boundaries_fast`,
   `count_transitions_fast` are correct and Numba-optimized. Do not
   "improve" them unless a test proves a bug.
2. **No print statements in core.** All debug/informational prints are
   removed or converted to logging (or just dropped). All warn-and-continue
   becomes raise/422.
3. **Pure functions.** Estimators return values, never mutate outside state,
   never write to disk.
4. **Equations for the "How it works" page come from the docstrings** — copy
   the LaTeX/math semantics from the existing docstrings verbatim. Do not
   re-derive the physics. Prose + citations are the author's.
5. **Validation is centralized in Pydantic + a matrix validator**, so the API
   and the core agree on what's valid.
6. **TUR is ported but not surfaced.** Export the wrapper (P1-T3) for
   completeness and tests only; nothing in the MVP UI or stretch exposes it
   except the author-led general-TUR goal (P6-T4).

### Config / environment variables

- `CORS_ORIGINS` — comma-separated list of allowed origins (dev + prod).
- `RATE_LIMIT` — slowapi limit string for `POST /api/simulate`.
- `MAX_LENGTH_CAP` (default 20000) and `MAX_N_STATES_CAP` (default 12) —
  overridable via env for tests.