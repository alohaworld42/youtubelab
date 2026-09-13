---
name: code-review
description: >-
  Evaluates code for correctness, readability, test gaps, and security. Use when
  the user asks to review code, a diff, a pull request, or a changeset before
  merging. Produces a prioritized list of findings with concrete fixes.
---

# Code Review

Review the target code (working diff, a named file, or a PR) across four
dimensions and report actionable findings.

## Scope

1. If no target is given, review the current uncommitted diff:
   `git diff HEAD` (plus `git diff --staged`). If the tree is clean, review the
   most recent commit: `git show HEAD`.
2. If the user names a file, function, or PR, review that.

## Review dimensions

Evaluate every changed hunk against all four dimensions:

- **Correctness** — logic errors, off-by-one, null/undefined handling, wrong
  async/await, unhandled promise rejections, incorrect error propagation, race
  conditions, resource leaks (unclosed files/handles/connections), and behavior
  that contradicts the surrounding code or stated intent.
- **Readability** — unclear names, dead code, duplicated logic that could reuse
  an existing helper, over-complex control flow, missing or misleading comments,
  inconsistent style vs. the surrounding file.
- **Test gaps** — new branches, error paths, and edge cases with no test
  coverage. Name the specific untested case, not "add more tests."
- **Security** — injection (shell/SQL/template), unsanitized external input,
  secrets or API keys in source, unsafe deserialization, missing authz checks,
  path traversal, SSRF, and dependency risks introduced by the change.

## Output

Report findings ranked most-severe first. For each finding give:

- **File:line** and a one-sentence statement of the defect.
- **Failure scenario** — concrete inputs/state that trigger the wrong behavior.
- **Suggested fix** — the smallest change that resolves it.

If nothing substantive is wrong, say so plainly and note only genuine nits.
Do not pad the list. A short review of real problems beats a long review of
speculation.
