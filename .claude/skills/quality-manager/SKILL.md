---
name: quality-manager
description: Run a channel quality audit — the quality-manager agent sweeps review logs and final cuts, finds defects the gates missed, mines cross-episode patterns, researches fixes online, and files tracked workflow improvements. Use after finalizing an episode, on a schedule, or when quality feels off. Args: optional episode base name (default = newest final) or "all"; optional depth "quick" (default) or "full".
---

# /quality-manager — channel quality audit

Orchestrate one audit cycle of the ZubiBop pipeline. The heavy lifting belongs to the
`quality-manager` agent (`.claude/agents/quality-manager.md`); this skill wires it up.

## Steps

1. **Resolve scope from args.** No args → newest episode in `output/Final/` at depth
   `quick`. An episode base name → that episode. `all` or depth `full` → every final plus
   staging rejects. If `output/Final/` is empty, say so and audit the newest draft instead.
2. **Spawn the `quality-manager` agent** with: the resolved scope (explicit episode base
   names and their paths), the depth, and the instruction to run its full audit cycle and
   update `docs/quality/DEFECT_BACKLOG.md`, `IMPROVEMENT_LOG.md`, and `SOURCES.md` itself.
   Run it in the foreground — the audit result is the point of the command.
3. **Relay the report.** Give the user the agent's summary: top findings by severity, the
   2–3 recommended workflow changes, and the verification status of previously logged
   improvements. Any S1 finding goes first.
4. **Apply approved changes.** Changes the agent classified as clearly safe are already
   applied (via worker delegation) — list them. For the judgment-call proposals, present
   them and wait for the user's pick; don't apply those unprompted.
5. **Commit** the ledger updates and any applied workflow changes as one commit
   (`QM audit: <date> — <headline finding>`), push per repo habit.

## Rules

- Never let an audit modify or delete anything in `output/` — media is channel inventory;
  new contact sheets and probe artifacts for the audit go next to the episode's existing
  `review/` files or in the scratchpad.
- The audit is read-mostly: workflow changes flow through the agent's improvement rules
  (one change per defect class, hypothesis + metric logged), never as drive-by edits.
- If the agent reports an S1 compliance finding (IP lookalike), surface it prominently and
  recommend holding uploads of the affected episode until the user decides.
