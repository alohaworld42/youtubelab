---
name: comprehensive-testing
description: >-
  Generates unit tests that cover happy paths, error boundaries, and edge cases
  for a given function, module, or change. Use when the user asks to write, add,
  or improve tests, or to raise coverage for a specific piece of code.
---

# Comprehensive Testing

Write unit tests that exercise the target code thoroughly — not just the obvious
success case.

## Steps

1. **Locate the test setup.** Detect the project's test framework and
   conventions (e.g. `pytest`, `jest`/`vitest`, `go test`, `JUnit`) by checking
   the manifest and existing test files. Mirror the existing naming, directory
   layout, and assertion style. Do not introduce a new framework.
2. **Read the code under test** and enumerate its behaviors before writing any
   test.
3. **Cover three categories explicitly:**
   - **Happy path** — typical, valid inputs produce the expected result.
   - **Error boundaries** — invalid input, wrong types, out-of-range values,
     empty/`null`/`undefined`, thrown exceptions and rejected promises. Assert
     the specific error, not just "it fails."
   - **Edge cases** — empty collections, single-element collections, boundary
     values (0, -1, max), unicode/whitespace strings, duplicate keys,
     concurrency, and any branch that is easy to miss.
4. **One behavior per test.** Give each test a name that states the case
   (e.g. `returns empty list when input is empty`).
5. **Keep tests deterministic** — no reliance on real time, network, randomness,
   or ordering unless controlled via mocks/fixtures.

## Verification

After writing, run the test suite and confirm the new tests pass. Report the
command used and the result. If a new test reveals a real bug in the code under
test, surface it rather than weakening the test to make it pass.

## Output

List the cases covered, grouped by category, and note any behavior you could not
test and why.
