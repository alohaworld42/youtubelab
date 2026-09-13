---
name: summarize-changes
description: >-
  Pulls a live diff of the Git working tree and summarizes it, flagging risky
  updates. Use when the user asks what changed, to summarize the diff, review
  pending changes before a commit or PR, or runs /summarize-changes.
---

# Dynamic Diff Summarization

Produce a concise, accurate summary of the current changes straight from Git —
never from memory of what you think you changed.

## 1. Pull the live diff

Gather the real state of the tree:

- `git status --short` — untracked and modified files.
- `git diff` — unstaged changes.
- `git diff --staged` — staged changes.
- If the working tree is clean, summarize the last commit: `git show HEAD`.

Base the summary only on what these commands actually return.

## 2. Summarize

- Give a one-line overall description of the change set.
- Group changes by area/file and describe *what* changed and *why* (intent),
  not a line-by-line restatement of the diff.
- Note added/removed files, renames, and deletions.

## 3. Flag risky updates

Call out, in a dedicated **Risks** section, anything that deserves a second look
before committing:

- Secrets, API keys, tokens, or credentials added to tracked files.
- Deletion or weakening of tests, validation, or auth checks.
- Changes to security-sensitive code (auth, crypto, input handling, shell/SQL).
- Broad or unexpected changes — large auto-formatting churn, generated files,
  lockfile changes, edits far outside the stated scope.
- Debug leftovers — `console.log`, `print`, `TODO`, commented-out code,
  hardcoded local paths.
- Breaking changes to a public API or config.

If there are no risky updates, say so explicitly.

## Output

Keep it tight: overall summary, per-area changes, then the Risks section. This
is a pre-commit sanity check, not a full code review — if deep correctness
analysis is needed, hand off to the `code-review` skill.
