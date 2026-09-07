# AGENTS.md — for implementing agents

You are implementing a portfolio web app from an existing research codebase.
Read this file first, then `docs/BUILD_PLAN.md` (the authoritative build spec),
then work through the phases in order.

## Critical facts to keep in mind

1. **The science is done and correct.** Do not rebuild, "improve," or
   re-derive the physics in `ctmc_simulator.py`. Your job is to productionize
   it. The authoritative source is `docs/source/reference_ctmc_simulator.py`
   — trust it over any paraphrase.

2. **The JIT functions are correct and must be copied verbatim:**
   - `simulate_trajectory_core`
   - `precompute_transition_data`
   - `find_snippet_boundaries_fast`
   - `count_transitions_fast`
   Do not touch these unless a test proves a bug. They run under Numba
   `@jit(nopython=True)`.

3. **No serverless backend.** Numba JIT compiles lazily on first call and
   takes real time. Deploy as an always-on container (Fly.io/Render) and warm
   up the JIT cache at process startup.

4. **No prints, no warn-and-continue in core code.** The original module has
   `print("Warning: ...")` and continues. In the web app these must become
   `ValueError` raises surfaced as HTTP 422 with clear messages. No file-I/O
   (`save_*`, `process_simulation_results`, pickle) in the app.

5. **Push to things don't exist yet.** The backend and frontend are scaffolded
   directories only. Implementation tasks (P0–P6 in the build plan) are not
   done. Work through them in order; each task is one commit.

6. **Defaults already chosen** (don't relitigate unless a Section 9 question
   is directly relevant and blocking):
   - Python backend + TS/React frontend
   - `max_length` cap 20,000; `n_states` cap 12
   - `trajectory_preview` decimated to ~2,000 points server-side
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
