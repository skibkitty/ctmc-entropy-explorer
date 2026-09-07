# CTMC Entropy Production Explorer

An interactive web playground that visualizes entropy production in
continuous-time Markov chains (CTMCs), driven by Numba-JIT research simulation
code ported from a research internship.

**Status: scaffolding. Implementation follows `docs/BUILD_PLAN.md`.**

## Layout

```
backend/   FastAPI + NumPy/Numba simulation core + estimators
frontend/  React + TypeScript (Vite) playground
docs/      BUILD_PLAN.md (authoritative spec) + reference source
```

## Background

This is the blueprint phase. See `docs/BUILD_PLAN.md` for the full build plan:
architecture, API design, task breakdown (Phases 0–6), and open questions.
