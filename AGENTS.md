# AGENTS.md — for implementing agents

You are implementing a portfolio web app from an existing research codebase.
Read this file first, then `docs/BUILD_PLAN.md` (the authoritative build spec),
then work through the phases in order.

## Critical facts to keep in mind

1. **The science is done and correct.** Do not rebuild, "improve," or
   re-derive the physics in `ctmc_simulator.py`. Your job is to productionize
   it. The authoritative source is `docs/source/reference_ctmc_simulator.py`
   — trust it over any paraphrase.

2. **Three JIT functions must be copied verbatim** (correct as-is, kept
   unchanged):
   - `precompute_transition_data`
   - `find_snippet_boundaries_fast`
   - `count_transitions_fast`
   Do not touch these unless a test proves a bug. They run under Numba
   `@jit(nopython=True)`.
   `simulate_trajectory_core` is the exception: it was refactored per issue #6 —
   its `print("Warning: …")`/`break` branch is now an explicit `ValueError`
   raise. The authoritative original remains in
   `docs/source/reference_ctmc_simulator.py`.

3. **No serverless backend.** Numba JIT compiles lazily on first call and
   takes real time. Deploy the backend to **Render free web service** (it spins
   down after 15 idle minutes — keep it warm with the GitHub Actions ping in
   P5-T3, scheduled every 5 minutes) and warm up the JIT cache at process startup. The frontend must show
   a "waking up the simulator…" state via `/api/health`.

4. **No prints, no warn-and-continue in core code.** The original module has
   `print("Warning: ...")` and continues. In the web app these must become
   `ValueError` raises surfaced as HTTP 422 with clear messages. No file-I/O
   (`save_*`, `process_simulation_results`, pickle) in the app.

5. **Push to things don't exist yet.** The backend and frontend are scaffolded
   directories only. Implementation tasks (P0–P6 in the build plan) are not
   done. Work through them in order; each task is one commit.

6. **The audience is SWE recruiters**, not researchers. Engineering depth
   (typed API, tests, CI, Docker, interactive viz, the async V2 story) is the
   product; the physics is the domain, not the point.

7. **Defaults locked in a grilling session** (don't relitigate — see
   `docs/BUILD_PLAN.md` §9 for the full list):
   - Python FastAPI backend + React/TS frontend; Vercel free for the frontend
   - `max_length` cap 20,000; `n_states` cap 12
   - `trajectory_preview` decimated to ~2,000 points server-side
   - Single-run **k-sweep (k = 1–4)** on the tracks demo
   - **React Flow (`@xyflow/react`)** `RateMatrixGraph` on both playground pages
   - Custom-mode metastate assignment: per-state dropdown, full coverage (422)
   - GH Actions CI (pytest + tsc/build), real Dockerfile, meaningful tests, all in MVP
   - **No database in MVP**; Neon free Postgres job-store is V2 (no hosting change)
   - **TUR is NOT exposed** in MVP/stretch; the general-TUR goal is the LAST
     stretch item and is **author-implemented** (agent can't derive it)
   - "How it works" equations come verbatim from docstrings; **prose and
     citations are the author's**; landing copy drafted by agent → author rewrite
   - Per-IP rate limit **plus** global concurrency semaphore (2–3 concurrent
     sims → 503 "busy"); matrix validation includes finite entries and
     off-diagonal ≥ 0, and the JIT core's "negative exit rate → break" branch
     becomes an explicit raise (422)
   - Keep-warm ping every 5 minutes (Render idle window is 15 min; GH Actions
     `schedule:` is best-effort)
   - MIT license

## Commands

- Backend: `python -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt`
- Backend tests: `pytest backend/` (run from repo root or `backend/` as the
  project structure stabilizes — follow the layout in BUILD_PLAN Section 8)
- Frontend: `cd frontend && npm install && npm run dev`
- Frontend typecheck: `npm run build` (runs `tsc -b` via Vite)

## Directory layout (target, after Phase 1)

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

## When you're stuck

- The build plan (`docs/BUILD_PLAN.md`) is the source of truth for scope,
  API shape, and defaults.
- Open questions for the project author are in BUILD_PLAN Section 9. Only ask
  if a default is unworkable.
- Commit each completed task separately with a clear message.

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues; use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical labels (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
