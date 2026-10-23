# Phase 2 merge notes — overlap with Phase 1 fixes

Branch `feat/phase-2-sim-api` was cut from the tip of `feat/phase-1-sim-core`
(`f423f40`). The Phase 1 issue-fix work (on `feat/phase-1-sim-core`) and Phase 2
land side by side. This file records the seams where the two branches touch the
same files, so the eventual merge/rebase is low-friction.

## File-by-file

| File | Phase 1 fixes touch it? | Phase 2 touched it? | Friction |
|---|---|---|---|
| `backend/main.py` | No (Phase 1 never edits it) | Yes — refactored into a `create_app()` factory; added lifespan + semaphore + slowapi wiring. **Keep the module-level `app = create_app()` singleton** (tests + uvicorn import it). | None |
| `backend/tests/test_api.py` | No — new file | New | None |
| `backend/api/*` (config, schemas, service, routes, warmup) | No — new package | New | None |
| `backend/tests/test_app.py` | **Also edited by Phase 1** (`88c2f56` moved `parse_cors_origins`; tests import it from `backend.main`) | Still passes untouched; `parse_cors_origins` is re-exported from `backend.main`. | Low — same file, both sides edit it |
| `backend/tests/test_simulation.py`, `test_estimators.py`, `test_parallel_tracks.py`, `test_decimation.py` | Phase 1 issue fixes (e.g. #13, #15, #17, #20 addressed there) | Not touched | None |
| `backend/requirements.txt` | **Issue #4 (lock file) is destined here** | Not yet touched by Phase 2 (no new runtime deps — slowapi was already pinned in P0) | Low — one line coexists fine |
| `docs/BUILD_PLAN.md`, `AGENTS.md` | Phase 1 docs tweaks | Not touched by Phase 2 | None |

## Recommended merge sequence

1. Finish Phase 1 fixes on `feat/phase-1-sim-core`; keep that branch green.
2. `git checkout feat/phase-2-sim-api`
3. `git rebase feat/phase-1-sim-core` — on conflict, only `backend/tests/test_app.py`
   (and possibly `backend/requirements.txt` if #4 lands first) should need
   attention; resolve by keeping both sides' changes.
4. Merge Phase 2 into main (or into Phase 1 then main).

## Notes

- Phase 2 assumes the module-level `create_app()` singleton name `app` stays
  exported from `backend.main` (P0/P1 tests import `from backend.main import app`).
- The semaphore capacity, caps, and rate limit all read `backend.api.config.settings`,
  so merging cannot silently change runtime defaults.
- No Phase 2 commit touches `ctmc_core/` (the physics port) — Phase 1 remains
  the sole owner of that directory, which keeps the rebase clean.