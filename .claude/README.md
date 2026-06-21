# Claude Code project commands

Custom slash commands for this repo. Type `/<name>` in Claude Code to run one.

| Command | What it does |
|---------|--------------|
| `/setup-bench-ci` | Scaffolds a deterministic CI gate — notebook/JSON validation + fixture-based unit tests for the repo's pure logic — and opens a draft PR. |
| `/codex-loop` | Runs the AI code-review loop on a PR: trigger `@codex review`, verify each finding against the code, fix + push, re-trigger, stop when clean (or on usage limit). |

## Reusing these in other repos
These are **project** commands (live in `.claude/commands/`, shared with anyone
who clones this repo). To make them available everywhere:

- **Per repo / for a team:** copy `.claude/commands/` (and, if you want this
  reference along with it, `.claude/README.md`) into the other repo and commit it.
- **For yourself across all repos:** copy the `*.md` files into
  `~/.claude/commands/` on your machine — then `/setup-bench-ci` and `/codex-loop`
  work in every repo you open locally. (Note: on the ephemeral *remote* Claude
  Code environment, `~/.claude` does not persist between sessions; keep them in a
  repo for durability.)

The commands are written to auto-detect the stack, so they adapt to a non-notebook
repo with little or no editing.
