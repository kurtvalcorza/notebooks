---
description: Scaffold a deterministic CI gate (notebook/JSON validation + unit tests) for this repo
argument-hint: "[subdir to focus on, e.g. aiaas-benchmarking] (optional)"
---

You are setting up a **deterministic CI gate** for this repository so that pull
requests are checked by something reproducible — not only by an AI reviewer.

Scope hint (optional): `$ARGUMENTS`
If empty, infer the primary code/notebook area from the repo layout.

## Principles
- CI is the backbone; AI review is a supplement on top. This command builds the
  backbone.
- Only add checks that are **fast, deterministic, and free** (no GPU, no large
  downloads, no network). Heavy/real runs stay out of CI.
- Prefer testing **pure logic** (parsers, aggregators, cost/format math) with
  small in-repo fixtures over trying to execute notebooks.

## Steps
1. **Detect the stack.** Inspect the repo (and the scope hint if given): is it
   Python, JS/TS, Go, …? Are there Jupyter notebooks (`*.ipynb`)? List the
   importable modules / scripts that contain real logic worth unit-testing.

2. **Add artifact validation** appropriate to what exists:
   - Notebooks: a check that every `*.ipynb` is valid JSON and that each code
     cell `ast`-parses (Python) — catches corrupt/broken notebooks.
   - Plain JSON/config: a JSON-validity check.
   - Source: the language's parse/compile/lint step (e.g. `python -m compileall`
     / `ruff`, `tsc --noEmit`, `go vet`).

3. **Add fixture-based unit tests** for the pure logic you found. For each
   data-consuming script, write a tiny fixture input (one per supported
   schema/case) and assert the output (columns/keys/values). Put them under
   `tests/` with a couple of `test_*.py` (or the stack's idiom). Keep fixtures
   inline or in `tests/fixtures/`.

4. **Write the workflow** at `.github/workflows/ci.yml`:
   - Trigger: `pull_request` (and `push` to the default branch), ideally
     `paths:`-filtered to the relevant subdir.
   - Set up the runtime, install only the lightweight deps the tests need,
     run the validation step, then run the tests.
   - No secrets, no GPU, no network-dependent steps.

5. **Verify locally** before committing: run the same validation + tests you put
   in CI and confirm they pass (and that a deliberately broken input fails).

6. **Commit on a feature branch, push, open a draft PR.** Do NOT push to the
   default branch. In the PR body, list exactly what the gate checks and what it
   deliberately does NOT (the heavy/real runs).

## Portability note
Keep the workflow and tests generic enough to copy into sibling repos: isolate
repo-specific bits (paths, module names, fixtures) so adapting it elsewhere is a
small edit. Mention in the PR body what to change when reusing it.
