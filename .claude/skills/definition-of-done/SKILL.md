---
name: definition-of-done
description: >-
  Builds an execution plan, chunks the work into ordered tasks, and defines exact
  verification steps BEFORE executing. Use at the start of any non-trivial
  implementation task, or when the user asks to plan work, scope a feature, or
  define what "done" means.
---

# Definition of Done (DoD)

Before writing any implementation code, produce a plan that makes "done"
unambiguous. Only then execute.

## 1. Restate the goal

In one or two sentences, state what the finished change must do, from the user's
point of view. Note any constraints (compatibility, performance, files that must
not change).

## 2. Build the execution plan

Break the work into ordered, independently verifiable chunks. Each chunk should
be small enough to complete and check on its own. For each chunk record:

- **What** — the concrete change (file(s), function(s), behavior).
- **Why** — how it advances the goal.
- **Depends on** — earlier chunks it requires.

Use the task tools (TaskCreate/TaskUpdate) to track chunks so progress is
visible. Mark each `in_progress` when started and `completed` when its
verification passes.

## 3. Define exact verification steps

For the change as a whole and for each risky chunk, write the *precise* command
or observation that proves it works — before implementing. Examples:

- `pytest tests/test_generator.py -q` passes.
- `npm run build` exits 0 with no type errors.
- Running `<command>` produces `<observable output>`.
- The new endpoint returns `200` with the expected JSON shape.

Prefer checks that can actually be run in this environment. If a check can't be
automated, state the manual observation that confirms success.

## 4. Execute

Work the chunks in dependency order. After each, run its verification step and
only then move on. If a step fails, fix and re-verify — do not proceed on a red
check.

## 5. Confirm done

The task is done only when **every** verification step has been run and passes.
Report each check and its result plainly. If a step was skipped or a check
still fails, say so explicitly — do not claim completion.
