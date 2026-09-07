# CTMC Entropy Production Explorer — Build Plan

> **Status:** Living document for implementing agents
> **Owner:** the project author
> **Last updated:** 2026-09-07

---

## 1. Purpose

This document is the authoritative build spec for turning a research
internship's Numba-JIT CTMC simulator and entropy-production estimators into a
**polished, interactive web app**. The goal is a portfolio piece: a recruiter
should be able to open it, play with it for 60 seconds, and see that real
research/simulation code was productionized into something usable.

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
| `thermodynamic_uncertainty_relation_estimator`, `count_transitions_fast`, `find_snippet_boundaries_fast` | TUR lower-bound estimator (hard-coded A/B/C topology) | Copy JIT helpers unchanged; port wrapper |
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
│  (Vercel/Netlify)    │ ◀─────────────────────── │  (Fly.io / Render, always-│
│  Recharts + KaTeX     │                          │  on Docker container)     │
└─────────────────────┘                          └──────────────────────────┘
                                                          │
                                                    ctmc_core/ (ported,
                                                    refactored simulator +
                                                    estimators, Numba JIT)
```

**Critical constraint: do NOT deploy the backend as a serverless function.**
Numba JIT compiles lazily on first call and takes real time. A cold-started
serverless function would pay that cost on every scale-to-zero cycle. Use a
small **always-on container** (Fly.io free tier / Render free web service) and
**warm up the JIT cache at process startup** (Phase 2, task P2-T4).

---

## 4. Tech stack (defaults)

- **Backend:** Python 3.11, FastAPI, Uvicorn, NumPy, Numba, Pydantic v2,
  pytest, slowapi (rate limiting)
- **Frontend:** React + TypeScript (Vite), Tailwind CSS, shadcn/ui, Recharts,
  react-katex
- **Deployment:** backend → Fly.io/Render (Docker, always-on); frontend →
  Vercel
- **No database. No user accounts.** Stateless playground — every request is
  self-contained (rate matrix + params in, results out).

---

## 5. Feature scope

### MVP (must ship)

1. **Parallel Tracks demo** (flagship): sliders/inputs for `α, β, u₁, w₁, u₂,
   w₂`. Run simulation → trajectory viz, per-metastate occupancy, and a chart
   comparing k-th order estimator (sweep of k, or vs. trajectory length)
   against the closed-form true EPR.
2. **Custom rate matrix mode**: N×N matrix input (N capped) + metastate
   groupings, run simulation, get k-th order EPR estimate + basic
   trajectory/occupancy viz. Server validates the matrix (columns sum to
   zero) and returns a clear error otherwise.
3. **Trajectory visualization**: state-vs-time step plot (fine-grained,
   decimated), bar chart of total time / visit count per metastate.
4. **"How it works" page**: recruiter-readable explanation of CTMCs, entropy
   production, and each estimator, with KaTeX equations sourced **directly
   from the existing docstrings** — do not invent or paraphrase the physics.
5. **README**: architecture diagram, live demo link, "why I built this"
   framing tied to the internship.

### Stretch (only after MVP deployed and working)

6. Repeated-transitions estimator UI for tracks demo.
7. TUR estimator for fixed 3-metastate A/B/C preset.
8. Convergence-vs-length chart (re-run simulation at several trajectory
   lengths; plot estimator error vs. length on log-log).
9. Shareable permalinks (encode inputs in URL query string).

### Explicitly NOT in scope

- Persistence / save-to-file endpoints
- User accounts
- The original `save_*` / `process_simulation_results` file-I/O functions

---

## 6. Backend API design

All endpoints return JSON. No endpoint writes to disk.

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

**Caps (server-side, enforced with 422 + clear message):**
- `max_length` ≤ 20,000
- `n_states` ≤ 12

**Validation:** rate matrix must be square and columns must sum to zero. On
failure return 422 with the offending state index. (Use the existing check in
`simulate_single_trajectory`, but raise instead of print.)

**Decimation:** `trajectory_preview` must be decimated server-side (max
~2,000 points via uniform stride or LTTB downsampling), not shipped raw.

### `GET /api/presets/parallel-tracks`

Returns default `α, β, u₁, w₁, u₂, w₂` values and the generated rate matrix so
the frontend doesn't hardcode simulation parameters.

### `GET /api/health`

Returns 200 once Numba warm-up has completed. Used by the frontend to show a
"waking up the simulator…" state and by the host's health check.

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
  estimation solves, then a clear CTA into the tracks demo.
- **Loading/compute states matter more than in most demos** (JIT warm-up):
  show a distinct "waking up" state (from `/api/health`) vs. a per-run
  "simulating…" spinner.
- Charts (all Recharts): staircase trajectory plot, metastate occupancy bar
  chart, EPR comparison chart (estimated vs. true as horizontal reference
  line).

---

## 8. Task breakdown

Work through phases in order. **Each task = a separate commit.**

### Phase 0 — Repo & scaffolding
- [ ] P0-T1: Create monorepo with `/backend` and `/frontend` directories, root
      README stub, `.gitignore`, license.
- [ ] P0-T2: Scaffold FastAPI app in `/backend` with `/api/health` returning
      `{"status": "cold"}` initially.
- [ ] P0-T3: Scaffold Vite + React + TS app in `/frontend` with Tailwind and
      shadcn/ui installed and configured.
- [ ] P0-T4: Set up CORS on the backend for the frontend's dev and prod
      origins.

### Phase 1 — Port and refactor simulation core
- [ ] P1-T1: Copy JIT functions (`simulate_trajectory_core`,
      `precompute_transition_data`, `find_snippet_boundaries_fast`,
      `count_transitions_fast`) into `backend/ctmc_core/simulation.py`
      **unchanged**.
- [ ] P1-T2: Refactor `simulate_single_trajectory` into a pure function that
      returns a structured result object and raises `ValueError` on invalid
      input (no printing/warn-and-continue).
- [ ] P1-T3: Port `kth_order_estimator`, `repeated_transitions_estimator`,
      `analyze_transitions`, `thermodynamic_uncertainty_relation_estimator`
      into `backend/ctmc_core/estimators.py`, same treatment (no prints, raise
      on bad input, return values not side effects).
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
      input caps and matrix validation.
- [ ] P2-T3: Implement `GET /api/presets/parallel-tracks`.
- [ ] P2-T4: Numba warm-up on app startup (run one tiny simulation through
      every JIT'd function); flip `/api/health` → `{"status": "ready"}` once
      complete.
- [ ] P2-T5: Rate limiting (slowapi) on `/api/simulate`.
- [ ] P2-T6: FastAPI TestClient integration tests for all endpoints, including
      422 error paths.

### Phase 3 — Frontend core
- [ ] P3-T1: Landing page with case-study framing.
- [ ] P3-T2: Reusable rate-matrix editor component (grid input + inline
      validation).
- [ ] P3-T3: Trajectory staircase chart component (Recharts).
- [ ] P3-T4: Metastate occupancy bar chart component.
- [ ] P3-T5: EPR comparison chart component (estimate vs. true-EPR reference
      line, or small table when no ground truth).
- [ ] P3-T6: API client with typed request/response models matching Section 6,
      plus loading/error/cold-start states.

### Phase 4 — Playground pages
- [ ] P4-T1: `/playground/tracks` — sliders seeded from
      `/api/presets/parallel-tracks`, "Run simulation" button, results via
      Phase 3 components.
- [ ] P4-T2: `/playground/custom` — matrix editor + metastate group assignment
      + run + results.
- [ ] P4-T3: `/how-it-works` with KaTeX equations sourced from the docstrings.

### Phase 5 — Polish & deploy
- [ ] P5-T1: Responsive pass (laptop minimum; graceful "best viewed on
      desktop" fallback for `/playground/*` on narrow viewports).
- [ ] P5-T2: Dockerize backend; deploy to Fly.io/Render as always-on service.
- [ ] P5-T3: Deploy frontend to Vercel, pointed at deployed backend.
- [ ] P5-T4: Root README: what it is, screenshot/GIF, architecture diagram,
      live link, local dev instructions, "ported from internship research
      code" note.
- [ ] P5-T5: Smoke-test full deployed flow end-to-end (cold start → health
      check → run each demo → check chart rendering).

### Phase 6 — Stretch goals (only if time remains)
- [ ] P6-T1: Repeated-transitions estimator UI for tracks demo.
- [ ] P6-T2: Fixed-topology TUR estimator preset.
- [ ] P6-T3: Convergence-vs-length chart.
- [ ] P6-T4: Shareable permalinks via URL query params.

---

## 9. Open questions — ask the author before proceeding if unclear

1. **Backend vs. full client-side port**: this plan assumes a real Python
   backend (Section 3). A zero-hosting-cost fully-static site (estimators
   reimplemented in TypeScript) is a materially different plan. Confirm before
   Phase 1.
2. **Hosting accounts**: does the author already have Fly.io/Render and Vercel
   accounts, or should the agent default to whichever is easiest to set up?
3. **Domain/branding**: generic host subdomain, or under a personal domain?
4. **Custom rate matrix state cap**: confirmed default of 12 states acceptable,
   or is a specific larger showcase system needed?

If none of these block progress, **proceed with the defaults** stated
throughout this document rather than pausing.

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
   re-derive the physics.
5. **Validation is centralized in Pydantic + a matrix validator**, so the API
   and the core agree on what's valid.

### Config / environment variables

- `CORS_ORIGINS` — comma-separated list of allowed origins (dev + prod).
- `RATE_LIMIT` — slowapi limit string for `POST /api/simulate`.
- `MAX_LENGTH_CAP` (default 20000) and `MAX_N_STATES_CAP` (default 12) —
  overridable via env for tests.
