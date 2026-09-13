# ZubiBop workflow improvement log

One entry per workflow change made for quality reasons: hypothesis, metric, verification
status. The quality-manager agent opens every audit by re-checking the metrics of entries
still marked **shipped** and moves them to **validated**, **refuted**, or **inconclusive**.
Entries IMP-001…IMP-005 are backfilled from git history on 2026-07-21; their hypotheses
are reconstructed.

### IMP-001 Director depicts the song's story   [shipped 224ac9e → trending VALIDATED]
- Defect: QM-004 (lockstep staging), weak story arc.
- Change: director prompt reworked from "visualize the beat" toward staging the song's
  narrative per verse.
- Hypothesis: story-driven shot descriptions produce varied, informal blocking and better
  verse-to-verse progression.
- Verify by: dim-9 rejects ≈ 0 in post-224ac9e episodes; story-arc score in QM audits.
- 2026-07-21: 0 lockstep/dim-9 rejects channel-wide, and the 7 inspected finals show varied,
  informal blocking (kids in playroom/farm/bathroom, no front-on marching line) — including
  clear story settings (feel-breeze playroom→garden, baa-baa farm, mulberry bathroom→garden).
  Caveat: recent episodes were heuristic-reviewed so the dim-9 gate wasn't running for them;
  the "0 rejects" is corroborated by the finals looking right, not by the gate alone. Trending
  validated; keep as monitoring until a vision-reviewed episode confirms the gate stays at 0.

### IMP-002 Render-style registry   [shipped 7aeba51]
- Defect: hardcoded toon path blocked look experiments.
- Change: `render_styles` map in config (workflow_base, style_lora, trigger, prompt
  skeleton, identity clause per style).
- Hypothesis: decoupling look from pipeline enables A/B-ing styles without code changes.
- Verify by: a second style (`flat_storybook`) rendering an episode end-to-end.
- 2026-07-21: VALIDATED as a mechanism — `output/_style_ab/flat_storybook.mp4` and
  `pixar_toon.mp4` both rendered from the registry, and claymation/felt were added the same way
  (drop-in LoRA, no new workflow file). The registry is now also the lever for the QM-011 fix.

### IMP-003 Quality tiers (draft/final)   [shipped e086910]
- Defect: full-fidelity renders too slow to iterate on quality problems.
- Change: `draft` tier (no hires pass, 0.6 frame scale, heuristic reviewer, ≤12 unique
  shots) + `--finalize <base>` re-render.
- Hypothesis: cheap drafts raise iteration count, catching structural problems before
  expensive renders.
- Verify by: fraction of drafts that get finalized (structural problems caught pre-final).
- 2026-07-21: INCONCLUSIVE + a caution. The draft tier uses the heuristic reviewer by design;
  combined with unattended runs (QM-010), draft-quality cuts appear to be reaching Final/
  without a vision pass (mulberry shipped 0/16 vision-reviewed; its finalize re-render also
  0/16). The tier works as a speed lever but the draft→final boundary is not enforcing the
  vision gate — see QM-010. Do not credit this as validated until a promoted Final is shown to
  carry full vision coverage.

### IMP-004 QC off the GPU   [validated 6310ecd]
- Defect: QM-008. Change: OpenCV pinned to CPU during QC.
- Verified: no episode-killing crashes since.

### IMP-005 Z-Image character reference anchor   [shipped a98e4a1 → INCONCLUSIVE, unblocked by IMP-006]
- Defect: QM-001 (invented/substituted child).
- Change: character identity anchored with a Z-Image reference via the (previously dead)
  force_i2v path.
- Hypothesis: image conditioning pins identity where text conditioning drifts, especially
  in wide shots.
- **Verification (2026-07-21, full media audit) — INCONCLUSIVE, and now we know exactly why:**
  - The number: pre-a98e4a1 cast-integrity reject rate = **6.8% of vision-reviewed takes
    (7/103)**; post-a98e4a1 = **indeterminate, not 0-in-a-good-way** — all 32 post-commit takes
    (the two mulberry runs) were heuristic-only (0 vision-reviewed), so "0 identity rejects"
    just means no gate looked. The cloud audit's "13.6% vs 0/22" compared a real pre-rate to a
    denominator that couldn't produce the finding.
  - The mechanism never ran: `i2v_anchored=true` on **0 of 255 takes** channel-wide. force_i2v
    was emitted true 5 times, but all 5 were multi-child "duplicate characters" shots where
    `_reference_for_single_subject` correctly returns None — so no single-subject anchor was
    ever staged. Root blocker: scene-plausibility.md pinned `force_i2v=false`, so the reviewer
    never emitted it on the single-subject substitution pattern the anchor is built for.
  - Verdict: cannot validate or refute — the change was unreachable. NOT reverted (the code is
    correct and dormant); superseded by IMP-006 which makes it reachable. Re-verify once
    episodes render post-IMP-006 WITH vision review on (QM-010 must be fixed too, or there is
    still no reviewer to emit force_i2v).

### IMP-006 Wire force_i2v to the single-subject substitution pattern   [shipped this audit]
- Defect: QM-001 — unblocks IMP-005.
- Change: (1) scene-plausibility.md now instructs the reviewer to emit `force_i2v=true` on a
  single-subject (`expected_children==1`) reject for same-count identity substitution (dim 3
  wrong-child-in-slot / dim 7 off-model drift), and `false` otherwise; the schema example value
  went `false` → `bool` (the key is unchanged/optional — contract preserved). (2) generate.py
  logs an additive `i2v_anchored` bool per take so an audit can tell a real anchored accept from
  an unattended heuristic fallback. Independently re-derived and confirmed correct against the
  live a98e4a1 consumption path (`_reference_for_single_subject` enforces single-subject in
  CODE, so a mis-emitted force_i2v on a multi-child shot still cannot wrongly anchor).
- Hypothesis: with the reviewer finally emitting force_i2v on the pattern it targets, single-
  subject hero-closeup substitutions (the "ginger-haired boy replaces Kofi" class) get re-
  rendered i2v-anchored and flip to accept, dropping QM-001's single-subject reject share.
- Metric / verify by: next audit — `i2v_anchored=true` count > 0 (was 0); of anchored shots,
  fraction whose subsequent verdict flips to accept; single-subject cast-integrity reject share
  vs the 6.8% baseline. PREREQUISITE: QM-010 fixed (a vision reviewer must be running to emit
  the hint at all) — otherwise this stays as unreachable as IMP-005 was.

### IMP-008 Vision-coverage promote gate   [shipped this audit]
- Defect: QM-010 — the vision reviewer (identity/wardrobe/scary-face/staging rubric) goes dark
  under unattended runs, and nothing stopped an all-heuristic cut from reaching Final/ anyway
  (mulberry shipped 0/16 vision-reviewed).
- Change: `_generate_director` computes vision coverage (`review.vision_coverage`, new) over the
  takes actually used in the cut (`renders`) right before both promote-to-Final sites (cut-QC-
  accept branch and the cut-QC-disabled branch — see generate.py's `_vision_gate_check`). A take
  counts as covered iff a `<take>.response.json` exists on disk and its `accept` is true — a late
  backfill (auto-reviewer or browser queue answered after the run) counts identically to one
  written during the run, which is the intended recovery path. Below
  `kidsong.review.min_vision_coverage` (new config key, default 0.8), promotion is refused: the
  staging .mp4 is left in place, status becomes `needs_review` (same clean-ending shape as a cut
  reject, no crash), a loud structured line names the coverage/threshold/uncovered shots, and an
  additive `{"gate": "vision_coverage", "coverage", "threshold", "uncovered", "promoted"}` entry
  is appended to `review_log.json`. No override flag — this is a hard backstop by design. The gate
  auto-disables (always promotes, no computation) when `min_vision_coverage` is 0 or
  `kidsong.review.reviewer == "heuristic"` (no vision path exists by configuration in that case).
  Verdict JSON schema is untouched; only additive keys on new `review_log.json` gate entries.
- Hypothesis: with promotion blocked below threshold, an unattended run either gets its takes
  vision-reviewed (by the auto_review.py headless reviewer landing alongside this, or a human
  answering the queue) before anything ships, or the episode stays in staging/needs_review
  instead of quietly reaching Final/ with an unreviewed content-safety rubric.
- Metric: vision coverage of every future promoted Final ≥ 0.8; verify via `review_log.json` gate
  entries (`gate == "vision_coverage"`, `promoted: true`, `coverage` value) on newly shipped
  episodes going forward, and confirm zero `needs_review` staging files silently mistaken for
  shipped content.

### Incident 2026-07-21 — cloud audit branch deleted claymation/felt render styles (NOT merged)
- The scheduled cloud audit pushed `origin/qm/audit-20260721` whose config.example.json diff
  silently DELETED the `claymation` and `felt` render styles (added deliberately in 17a1e73,
  with real user-downloaded LoRAs on disk: D:/brainrot/comfy-models/loras/Claymation.safetensors
  + CozyFelt.safetensors, 352 MB each). The commit message never mentioned the deletion — an
  undocumented drive-by outside its stated defect class, and it removes the channel's primary
  IP-distancing lever (QM-011). This local audit did NOT carry the deletion forward; main's
  config.example.json and live config.json both retain all four styles; both LoRA files verified
  present. The cloud branch's GOOD parts (force_i2v wiring, i2v_anchored telemetry) were
  independently re-derived and shipped as IMP-006 instead of merging the branch. The orphaned
  remote branch is left as-is (do not build on it).

## IMP-012 (2026-07-22): Identity conditioning — Omni dead end confirmed, klein-4B path chosen
Research (web, cited in session): Tongyi never released Z-Image-Edit/Omni — the
ComfyUI TextEncodeZImageOmni image inputs target unreleased weights, and the
released Turbo/Base checkpoints are documented text-to-image only. Our measured
"reference collage" behaviour is untrained ref-latent conditioning; the
identity_refs path stays default-off until a capable model exists locally.
Chosen path (smallest risk-free, Apache-2.0, commercial-safe): FLUX.2 klein 4B
fp8 (~4.6GB, reuses the existing qwen_3_4b text encoder, core ComfyUI nodes,
4-step) as a reference-conditioned KEYFRAME backend candidate; fallback ceiling
Qwen-Image-Edit-2511 Q4 (~22GB); durable complement: per-character Z-Image
LoRAs via ai-toolkit (12GB-feasible) once curated per-child still sets exist.
Interim shipped fix: single-subject shots anchor i2v on the canonical character
reference (proven 0.55 vs 0.1), ensembles keep text keyframes (3e2f2d1).

### IMP-013 (2026-07-23): Subset-shot negative prompt negates absent castmates   [shipped a28c961 → monitoring]
- Defect: QM-001/QM-003 solo-shot identity+wardrobe drift, measured live in the storyfix draft
  render (base 20260723-105443): solo Kofi closeups rendered with Zuri's afro puffs + denim
  dungarees. Root cause: `cast.negative_terms` only unioned the RESOLVED characters' own
  `must_not`, and the hand-authored cross-negations only covered shirt COLOUR (Kofi's list
  negates "yellow shirt" but nothing negates the girls' hair or bottoms), so a single-child
  slot was free to drift into an absent castmate's hairstyle/wardrobe.
- Change: on any subset shot (fewer resolved than the full cast), `negative_terms` also negates
  the ABSENT members' distinctive hair/top/bottom, derived from the bible (complete + self-
  maintaining). The existing `_term_satisfied` subtraction drops anything the resolved cast
  legitimately shares, so `['all']` is byte-identical. Additive; +3 unit tests.
- Hypothesis: negating the specific other-castmate features the model was drifting toward
  reduces single-subject substitution/wardrobe-bleed at the text-keyframe stage.
- Verification (same-seed A/B keyframe re-render, base 20260723-111406 vs -105443; only the
  negative prompt differs): PARTIAL. Kofi's afro-puff drift is reduced on some shots (s11/s14
  flip to correct short curly hair; s08 stays correct) but persists on others (s01/s04) — the
  documented QM-001 limit that LTX text-conditioning alone can't fully pin identity. Not a cure;
  a real, free, ensemble-safe reduction. Complete fix remains the image-reference anchor
  (force_i2v/klein, IMP-006/012), which still needs a vision reviewer running (QM-010) to fire.
- Metric: single-subject cast-integrity + dim-4 wardrobe reject share vs the 6.8% baseline, on a
  vision-reviewed episode rendered after this ships.

## IMP-014 (2026-07-23): Cast identity references generated + klein keyframe anchor validated   [shipped fedc8fa → monitoring]
- Defect: QM-001 solo-shot identity substitution/drift. IMP-005/006 wired force_i2v + a
  keyframe identity-ref anchor, but BOTH were inert: the canonical per-character references
  they read were never created (no caller for refs.generate_reference/ensure_reference), so
  refs.reference_for always returned None. Confirmed live: `i2v_anchored=true` on 0/255 takes.
- Change (fedc8fa): `_bootstrap_cast_references` runs once at render start (gated on
  keyframe_first.enabled+identity_refs) and calls `refs.ensure_reference` for each cast child —
  idempotent/cheap-first, best-effort. Fixes the ROOT starvation: the anchor now always has
  references to anchor on.
- Klein path validated (resolves IMP-012 "probe pending"): generated clean, on-model Z-Image
  Turbo references for zuri/kofi/nala (Kofi correctly a boy with short curly hair, no puffs —
  output/_cast_refs/<id>/canonical.png). With references present, `refs.generate_still` routes
  single-subject keyframes through FLUX.2 klein 4B (workflows/flux2_klein_ref.json): a
  standalone klein render of the two worst drifting Kofi shots produced correct-Kofi-in-scene
  (short curly hair, blue tee) — identity pinned, backyard scene + action preserved, no collage.
  The wired pipeline logged "keyframe anchored on N cast reference(s)" for all 16 shots.
- Verification: same-seed A/B (text-only vs klein-anchored keyframe) shows the afro-puff drift
  fixed on several Kofi shots (s11/s14); residual per-seed drift persists on some (s01/s04) —
  the QM-001 text-conditioning limit klein reduces but does not fully remove at every seed. The
  designed catch for the residual is the vision-review reseed loop (needs QM-010's reviewer
  running); klein is clean at most seeds, so a reseed on a flagged keyframe converges.
- Metric: `i2v_anchored`/keyframe-anchored count > 0 (was 0); single-subject cast-integrity
  reject share vs the 6.8% baseline on a vision-reviewed episode rendered after this ships.

### IMP-015 Off-bible names never reach a render prompt   [shipped]
- Defect: QM-015 (and, upstream, QM-001).
- Change: `director.repair_offbible_names` rewrites an invented child name in a shot's
  `action`/`setting` to the bible child the cast resolver substitutes for it, before any of
  the module's name-aware passes run; `cast.substitution_map` exposes the same mapping so
  `generate._rename_offbible_children` can repair a shot list that bypassed the director.
- Hypothesis: a prompt that names a child it never described forces the text encoder to
  invent that child's appearance, which is one avoidable source of the invented-extra-child
  rejects QM-001 tracks. Removing the contradiction should reduce them.
- Verify by: (1) zero capitalised non-bible tokens in `action`/`setting` across new shotlists
  (was 21 of 1130 fields); (2) cast-integrity reject share vs the 6.8% QM-001 baseline on the
  next vision-reviewed episode.
- Caution for the next audit: the detector is calibrated on the corpus's capitalised
  vocabulary (`director._NON_NAME_CAPS` plus "a capitalised -ing/-ed token is a verb, not a
  name"). If a future episode legitimately names a non-child proper noun in an action (a toy,
  a pet, a place), re-check `_NON_NAME_CAPS` before assuming the rewrite is correct.

### IMP-016 The take-review gate can actually be answered   [shipped]
- Defect: QM-016, enabling QM-010 (unanswered vision gate).
- Change: replaced the review queue's destructive whole-page meta-refresh with an
  interaction-aware JS refresh; paginated the queue; put a live pending-take counter in the
  nav of every page; unified `production.html`/`review_queue.html`/`index.html` onto
  `base.html` so the production dashboard and the take queue are reachable from the nav at
  all; enforced the "a ticked reject reason means reject" contract server-side; clamped the
  score to 0–1; made an already-answered gate un-overwritable.
- Hypothesis: QM-010's 0–6% real vision coverage was partly a tooling problem, not only an
  attention problem — the browser gate discarded work in progress every 8 seconds and was not
  discoverable. A queue that survives being used should raise human-answered coverage.
- Verify by: share of `*.request.json` answered by a human (non-`auto_accepted`) response on
  the next attended production stretch, vs the QM-010 baseline.

### IMP-017 GPU lock can no longer hand the GPU to two renders   [shipped]
- Defect: not previously in the backlog — found by the flaky
  `tests/test_gpu_lock.py::test_race_on_stale_lock_has_exactly_one_winner`, which failed
  roughly one run in three at HEAD. It is a SAFETY failure, not the liveness one the module
  docstring claimed: two processes both returned True from `try_acquire()`.
- Change: stealing a stale lock is now an in-place `os.replace` of the stealer's own ledger
  over the stale one, serialised by an atomically-published arbitration file — the lock path
  is never absent, so a fresh acquirer can never appear in a window where a straggling
  stealer could rip its lock away and drop it. (The first cut of the arbiter had the same
  class of bug one level down: created empty and filled in afterwards, it was read as
  abandoned by a second claimant, giving two concurrent stealers.)
- Hypothesis: the double-acquire is what produced the "ComfyUI timed out mid-job and the
  video was lost" incident the module was written for, and the surviving aborted-run count in
  QM-013.
- Verify by: the 8-thread and 4-subprocess stale races staying green over repeated runs
  (6 consecutive clean full-suite runs so far, vs 1–3 failures per run before); aborted-run
  count per shipped episode (QM-013 metric, mulberry = 4, target ≤1).

### IMP-018 Bookkeeping can no longer fail a rendered video   [shipped]
- Defect: QM-017.
- Change: post-success bookkeeping in `scheduler._run_pipeline` moved outside the render's
  `try`, each step guarded individually.
- Hypothesis: some share of the QM-013 aborted/re-rendered runs were successful renders
  destroyed by their own cleanup, not by the GPU fault the entry assumes.
- Verify by: `job N: <step> failed after a successful render` warnings appearing in
  logs/studio.log where the job previously went back to `queued` — and the QM-013
  aborted-run count dropping accordingly.

### IMP-019 Stalled uploads and unusable forms are visible again   [shipped]
- Defect: QM-018, plus a sweep of the studio pages.
- Change: "Wartet auf Upload" dashboard tile; the re-auth block recorded once on the job;
  the daily-cap warning logged once a day. On the pages themselves: API keys are
  `type=password` (they were legible on a UI the app serves on 0.0.0.0 by default);
  external links carry `rel=noopener`; a failed job's error is reachable in full instead of
  cut at 120 characters; every action button routes through a shared `withButton` helper
  so a double click cannot fire a second request and a thrown error cannot leave the button
  dead; every poller routes through `pollJson`, which retries instead of ending the loop on
  one failed request (the OAuth-connect and gameplay-download pollers both used to die
  silently and strand their button disabled); Enter submits a new idea.
- Verify by: `tests/test_studio_pages.py` (21 assertions, 17 of which fail on the previous
  templates) and `tests/test_scheduler_bookkeeping.py`.

### IMP-020 Kids declaration follows the content, not a checkbox   [shipped]
- Defect: QM-019.
- Change: `scheduler._made_for_kids(channel, job)` ORs the channel flag with
  `ideas.is_kidsong_type(job.video_type)`.
- Hypothesis: none needed — this is a compliance invariant, not an experiment.
- Verify by: `tests/test_scheduler_bookkeeping.py` COPPA cases (both fail on the previous
  scheduler).

### IMP-021 Nothing in the product points at a protected brand   [shipped]
- Defect: QM-020.
- Change: rewrote the kids script prompt's framing, the Quick Generate blurb, the channel
  editor's style label and the config comment.
- Hypothesis: the strongest single driver of IP-lookalike drift was the pipeline being
  *asked* for it in prompts/kids.txt — a gate cannot undo an instruction upstream of it.
- Verify by: QM-011 finding rate on the next audit; regression tests assert no product
  surface names the brand.

### IMP-022 .env writes are validated on both sides   [shipped]
- Defect: QM-022, plus the blank-URL gameplay download reporting a failure for an empty box.
- Change: `setup.write_env` validates key names and single-lines values;
  `start_gameplay_download` falls back to the starter pack when the box holds only
  whitespace (a list containing "  " is truthy, so the old `urls or DEFAULT` kept it and
  then filtered it to nothing, finishing instantly as "error").
- Verify by: `tests/test_setup.py` — the injection and blank-URL cases fail on the previous
  code.

### IMP-023 Self-review pass over IMP-015..IMP-022   [shipped]
- Not a new defect class — five errors found by re-reading the previous four sweeps'
  own diffs before calling them done. Recorded because "the fix had a bug in it" is the
  failure mode this ledger exists to catch, and three of these were invisible to the
  tests that shipped with them:
  1. `director._looks_like_a_child_name` refused any capitalised `-ed` token as a past
     participle — which also refused the ordinary given names Ahmed, Jared, Fred, Ned.
     The shipped corpus contains no `-ed` opener at all, so the rule cost real coverage to
     catch nothing; the handful of participles that could open an action phrase are
     enumerated in `_NON_NAME_CAPS` instead. Re-measured: the same 21 corpus rewrites,
     all of them invented names.
  2. The review queue's scroll restore was undone one line later by `focusItem(0)`'s
     `scrollIntoView` — so the auto-refresh still jumped to the top, the exact thing the
     restore exists to prevent. The initial highlight no longer scrolls.
  3. The scroll restore also ran before the lazy contact sheets had taken up space, so
     the offset clamped to a document that was still too short. It now re-applies on
     `load`.
  4. The new "reject an unparseable schedule" path re-rendered from the stored channel
     row, throwing away every other edit the user had made in the same visit — a worse
     bug than the typo that triggered it. `_channel_with_form` lays the submission back
     over the row.
  5. `generate._rename_offbible_children`'s docstring quoted "12% of shipped prompts",
     which is the word-cap figure from SOURCES.md, not the invented-name one (21 of 1130
     action/setting fields), and justified longest-name-first matching with a reason that
     `\b` already handles.
- Verify by: `tests/test_kidsong_offbible_names.py` (-ed names, participles, gerunds) and
  `tests/test_studio_pages.py` (the rejected-schedule form keeps what was typed).

### IMP-024 Credential files are owner-only   [shipped]
- Defect: extension of QM-022. `.env` (every API key plus STUDIO_PASSWORD) and
  `tokens/<channel>.json` (a live YouTube refresh token — anyone holding it can upload as
  that channel) were both written with default permissions.
- Change: `chmod 0600` on both, best-effort. A no-op on Windows, where the file inherits
  the user's ACL; the app's documented home is Windows, so this is defence for the
  tunnel/WSL/Linux cases rather than the primary one.
- Verify by: `tests/test_setup.py`, skipped on Windows.

### IMP-025 Stock downloads are atomic   [shipped]
- Defect: QM-023.
- Change: `assets._download` publishes through a `.part` file.
- Verify by: `tests/test_assets_fallback.py` — the interrupted-download, signal and
  retry-refetches cases all fail on the previous implementation.

### IMP-026 One primitive for "publish only when complete"   [shipped]
- Defect: QM-023 and the four further instances of it found by grepping for the pattern
  rather than waiting to trip over each one.
- Change: `pipeline/atomicio.py` (`atomic_path`, `atomic_write_bytes`,
  `atomic_write_json`) — write to a sibling `.part`, fsync, `os.replace`. Applied to
  `studio/assets.py`, `studio/comfy.py`, `studio/svm.py`, `studio/higgsfield.py`,
  `pipeline/kidsong/comfy.py` (both the HTTP and the same-filesystem copy paths) and
  `studio/setup.py`'s `config.json` rewrite. `pipeline/gpu_lock.py` already had this
  discipline, learned the hard way (IMP-017); this is it factored out so the rest of the
  repo does not rediscover it.
- Hypothesis: some share of QM-013's aborted runs and of the "ComfyUI produced a bad take"
  reports are truncated artifacts rather than bad generations — a stump is indistinguishable
  from a real file to every reader downstream.
- Verify by: `tests/test_atomicio.py`, including two guards that assert the call sites
  actually route through the primitive rather than testing it in isolation.

### IMP-027 Unanchored lyric runs are fitted, not clamped   [shipped]
- Defect: QM-024.
- Change: `alignment._interpolate` fits the head/tail runs to the available time.
- Hypothesis: some of the "captions lag the action" and "single-word fragments" reports
  behind QM-014 are this — a stack of words rendered in one flash at a verse edge reads as
  a missing caption, not as a timing bug.
- Verify by: `tests/test_alignment.py`; on a real episode, no two aligned words sharing a
  start time.

### IMP-028 Clip teardown closes what it claimed to   [shipped]
- Defect: QM-025.
- Change: `_scene_background`/`_background`/`_build_audio` carry their opened readers out
  on the returned clip; `build_video`'s `finally` closes them.
- Hypothesis: a long batch run leaking N ffmpeg subprocesses per episode is a plausible
  contributor to QM-013's aborted runs and to the CUDA-wedged ComfyUI state, since the
  studio process accumulates handles for as long as it stays up.
- Verify by: `tests/test_polish.py`; operationally, handle/memory growth of the studio
  process across a multi-episode batch.

### IMP-029 TTS scratch dirs are cleaned up on the failure path too   [shipped]
- Defect: `tts.synthesize`'s scratch-dir cleanup sat AFTER the synthesis loop, so any
  failure inside it left the directory and its mp3s in the system temp. edge-tts
  throttling past its three retries is the common trigger, and the scheduler retries a
  failed job up to `max_attempts` — so one bad stretch leaks a directory per attempt.
- Change: `try/finally` around the loop and export.
- Verify by: `tests/test_tts_scratch.py` (the failure case fails on the old code).

### IMP-030 The cut window is one coupled setting   [shipped]
- Defect: QM-026.
- Change: `cut_qc._effective_min_cut`, plus a config comment describing the window rather
  than one end of it.
- Verify by: `tests/test_beat_grid.py` — the floor and the director/gate-agreement tests
  both fail on the previous code.

### IMP-031 The cut builder honours the configured window   [shipped]
- Defect: QM-027 (and the correction to QM-026).
- Change: `edit._cut_window(cfg)`; `cfg` threaded into `build_cut_list` at both call sites.
- Hypothesis: the scene-mode "fewer, longer shots" experiment (the whole point of the
  Phase 1 knobs) has never actually been run — every episode was cut to the same 1.6-4.5s
  window no matter what the config said. Anyone who tried it and saw no difference was
  seeing this, not a null result about pacing.
- Verify by: `tests/test_beat_grid.py`; on a real run with both knobs raised, the cut list's
  max duration should follow the config.
- Lesson for the audit method: QM-026 was written from reading ONE module and reasoning
  about the next one instead of reading it. The reasoning was wrong and the claim shipped.
  Read the stage in between before describing a cross-stage failure.

### IMP-032 The song JSON resume needs is written atomically too   [shipped]
- Defect: `runstate.save_shotlist` writes the shot ledger through a temp file and says
  exactly why — "a crash during the write would otherwise leave truncated JSON and make
  the run unresumable — precisely the failure this module exists to prevent". The song
  JSON written on the very NEXT line, which resume needs just as much (re-singing the
  audio reads it), went straight to its final path.
- Change: both `song.json` writes in `generate.py` go through `atomic_write_json`.
- Verify by: `tests/test_atomicio.py`.
- Method note: found by grepping for the pattern (`json.dump(` into a final path) rather
  than by reading every module — the same sweep that produced IMP-026. Everything else it
  turned up writes request/response or telemetry files that a rerun regenerates; these two
  were the resume-critical pair.

### IMP-033 A broken verse layout is rebuilt, not nudged   [shipped]
- Defect: QM-028.
- Change: `sing.verse_times_from_words` rejects a layout with a degenerate window and uses
  the proportional split for the whole song, logging what it saw.
- Hypothesis: an episode whose verses collapse produces near-empty verse windows, which is
  a plausible cause of "too few cuts" / "cut duration out of range" rejections that look
  like director problems but start in the transcript match.
- Verify by: `tests/test_kidsong_verse_times.py`; operationally, the "degenerate layout"
  warning appearing in a run log is now the signal that a song's chorus confused the matcher.

### IMP-034 Guide frames are published atomically   [shipped]
- Defect: extension of QM-023 into `transitions.extract_frame`, found by reading
  `refs.py`. That function writes GUIDE IMAGES — the FLF2V first/last frame, a chain
  guide, and (via `refs.harvest_reference`) a cast reference anchor, the thing
  IMP-005/012/014 exist to pin character identity with. Every reader checks only
  `getsize(...) > 0`, so a truncated PNG from a killed ffmpeg would be staged into ComfyUI
  as a real frame. `refs` self-heals (a reference without its style sidecar is treated as
  stale and regenerated) but the transition and chain paths do not.
- Change: both the ffmpeg and the moviepy fallback write through `atomic_path`.
- Verify by: `tests/test_atomicio.py`.

### IMP-035 Prompt length is a config knob, so the A/B is runnable   [shipped]
- Defect: the measurement note in SOURCES.md — composed t2v prompts run min 166 / median
  184 / p90 231 / max 270 words, 12% of them over the ~200-word cap the RunDiffusion guide
  recommends. That note closed with "not changed here; a candidate experiment is shortening
  the tail rather than the identity block, A/B'd on the same seeds", and the experiment then
  sat undone because the tail was a literal inside `generate._shot_prompt`: running it meant
  editing code and re-pinning every prompt baseline in three test files.
- Change: the 48-word lighting/set tail is now `render_style`'s `prompt_tail`, with
  `pixar_toon`'s value the verbatim sentence pair the old code emitted (byte-identical
  default — the registry's standing regression guarantee). `config.example.json` ships
  `pixar_toon_concise`: a 29-word tail and nothing else changed, since every key an entry
  omits falls back to the pixar_toon defaults. Running the experiment is now setting
  `kidsong.render_style` to that name and re-rendering the same episode on the same seeds.
- Hypothesis: the length is dominated by the three full wardrobe descriptions in the
  identity sentence — the part measurably load-bearing for cast integrity (QM-001/QM-015)
  and the last thing to cut — so the tail is what should give. Shortening it by 19 words
  brings the p90 prompt under the cap without touching identity conditioning.
- Verify by: `tests/test_kidsong_prompt_tail.py` for the mechanism (default byte-identical,
  a shorter tail changes only the tail, empty/malformed values degrade instead of crashing a
  queued render, and the shipped concise entry differs in `prompt_tail` and nothing else).
- 2026-07-31, MEASURED on the length axis (`tools/prompt_length_report.py`, 517 child-bearing
  shots recomposed through the current code): **PARTIAL — the hypothesis holds for one-child
  shots and is refuted for multi-child ones.** The concise tail takes one-child shots from
  2.5% over-cap to **0.0%** (79% of the corpus). Two- and three-child shots stay at 95.7% and
  100%: their identity sentence alone is 101 words on average, so 19 words off the tail
  cannot close a 63-word gap. Recorded as QM-030, with the two candidate levers that ARE big
  enough. So: ship the knob, keep `pixar_toon_concise` as the arm to try, and do not expect
  it to fix the wide shots.
- Still unmeasured, needs a GPU: whether the shorter tail costs anything on screen. That is
  now one command per arm (IMP-038 — `--restyle <base> --style pixar_toon` then
  `--style pixar_toon_concise`, same song, shot list, audio and seeds), then compare the two
  review logs. `pixar_toon` stays the shipped default, so no episode renders differently
  unless someone opts in.
- Note for whoever runs it: prompt text goes to LTX-2's Gemma-3 encoder, not to an LLM. This
  sentence once read "concrete props" meaning "specific, tangible"; the encoder rendered grey
  stone blocks as set dressing in every shot of every episode. Keep any replacement tail free
  of words with a material reading (pinned by the banned-word test in the same file).

### IMP-036 The cast describes itself in grammatical English   [shipped]
- Defect: QM-029.
- Change: the three `build` values in `prompts/cast_bible.json` are noun phrases, matching
  what `cast._describe_dict` splices them into (`"<Name>, a <age> with <build>, <skin>,
  <hair>"`). No descriptor dropped, no code change — the composer was always right and the
  data was wrong for two of three characters, in every prompt, since `build` was added.
- Hypothesis: the identity sentence is where the cast-integrity gates live or die, and it
  was being handed to the text encoder as broken syntax ("a girl with petite and delicate").
  `SOURCES.md` records that LTX's encoder follows natural prose far better than fragments,
  so repairing the syntax of the most load-bearing sentence should not be neutral.
- Verify by: `tests/test_kidsong_cast.py::test_every_bible_build_is_a_noun_phrase` and
  `::test_the_shipped_cast_descriptions_read_as_english` — both assert against the SHIPPED
  bible, which is the gap that let this survive (the original `build` test used a correct
  fixture and never looked at the real data). Quality effect needs a GPU: QM-001
  cast-integrity reject share on the next vision-reviewed episode.
- Method note: found by measuring, not by reading. Printing a real composed prompt to work
  out where the words were going put the broken sentence on screen; nine months of reading
  `cast.py` had not.

### IMP-037 The prompt-length measurement is a script, not a docstring   [shipped]
- Defect: `SOURCES.md` quoted "median 184 / p90 231 / 12% over cap" from a hand measurement
  taken once. Re-measuring found median 190 / p90 262 / **23.0%** — the prompt kept growing
  (gaze, story-subject and count clauses) and the number in the docs did not move. A stale
  metric is worse than none: it was the basis for calling the overrun a 12% tail problem
  when it is a categorical multi-child problem.
- Change: `tools/prompt_length_report.py` recomposes every shot in `output/` through the
  real `_shot_prompt` at current code and config, and reports the distribution split by head
  count. Takes style names as arguments, so it doubles as the A/B harness.
- Verify by: run it. The numbers in QM-030 and in `SOURCES.md` are its output, dated.

### IMP-038 An episode can be re-rendered under a different look   [shipped]
- Defect: the render-style registry (IMP-002) let a channel PICK a look and never let anyone
  COMPARE two, which is why every style question in this log — IMP-035's prompt-length A/B
  included — stayed unmeasured. The claim "both arms are configured, just render one episode
  each on the same seeds" was not actually runnable: `--finalize` re-renders under a
  different QUALITY TIER (so a style read through it is confounded by frame count and the
  hires pass), `--resume` renders only what is missing (nothing, for a finished episode),
  and a fresh run writes a different song and a different shot list, so the two arms would
  share no material at all.
- Change: `restyle_episode` / `--restyle <base> --style <name>`. `finalize_episode`'s body
  was already 95% of it — reuse the source's song, shot list and sung audio under a new
  base, reset the ledger to "planned", delegate to the resume path — with one hardcoded line
  forcing the final tier. That body is now `_rerender_episode(base, overrides, ...)`;
  finalize overrides the quality tier, restyle overrides the render style and touches
  nothing else. `_reset_shot_for_finalize` already preserved `seed`, and no seed anywhere is
  derived from the style or the tier, so the two arms are seed-identical for free.
- An unknown style name is a HARD ERROR here, unlike `resolve_style`, which warns and falls
  back to pixar_toon. Falling back would render arm B as arm A and the experiment would
  report "no difference" — the one failure mode an A/B harness must not have.
- Verify by: `tests/test_kidsong_restyle.py` — the seeds survive, the quality tier does not
  move, the source episode is untouched, the two arms get distinguishable bases, and every
  refusal path is asserted. The whole file fails to import against the pre-change source.
- What this does NOT do: decide anything. It makes the deciding runnable on a machine with a
  GPU. IMP-035 and QM-030 name what to compare.

### IMP-039 The redundant half of the identity clamp is a config flip too   [shipped]
- Defect: QM-030 lever (a). Every child prompt asserts "no other children" twice — once in
  the count clamp, once in the trailing five words of the identity clause. 5 words × every
  shot of every episode, saying something the sentence before it already said better.
- Change: `pixar_toon_lean` in `config.example.json` — the concise tail plus an identity
  clause with the duplicate dropped and the identity half kept verbatim. Config only, no
  code. 24 words off every child prompt with no cast descriptor touched, which makes it the
  lossless upper bound on trimming.
- Result (`tools/prompt_length_report.py`): one-child shots 2.5% → 0.0% over cap, two-child
  100% → 82.6%, three-child 100% → 100%. **The bound is not enough**, and that is the point
  of building it: it converts QM-030's "the identity sentence is the only lever big enough"
  from an inference into a measurement. A three-child prompt still averages 239.5 words.
- Verify by: `tests/test_kidsong_prompt_tail.py` — the identity half survives word for word,
  the composed prompt still states BOTH constraints (the clamp now carries the second alone),
  no wardrobe/skin/hair descriptor is lost, and the arm differs from `pixar_toon` in exactly
  two text keys. `test_no_lossless_trim_can_bring_a_three_child_prompt_under_the_cap` fails
  if the conclusion ever stops holding.
- Unmeasured, needs a GPU: whether dropping the duplicate weakens the clamp on screen. The
  count clamp is the more precise of the two statements, which is why the duplicate rather
  than the clamp is what went — but "more precise" is a reading, not a measurement.
  `--restyle <base> --style pixar_toon_lean` renders the arm.

### IMP-040 A feature suite, separate from the unit suite   [shipped]
- Defect: 90 test files, every one named after a module (`test_alignment`,
  `test_gpu_lock`, `test_kidsong_cast`). They test units well. Nothing tested a
  FEATURE — the path a person takes through the running app. Even
  `tests/test_auth_gate.py`, the closest thing to an application test, built a
  bare Flask app with two stub routes rather than the real one.
- Consequence: a defect living in the seam between two correct units passed all
  1971 tests. QM-031 (the password gate voiding itself) is exactly that shape,
  and it was found within an hour of the first feature test running.
- Change: `tests/features/`, 56 cases in five files named after what a person is
  trying to do — answering a review gate, watching production, configuring a
  channel, setting up the studio, locking the studio. `conftest.py` boots the
  REAL `create_app()` (real blueprints, templates, config merge, SQLite) with
  only two things moved: `output_dir` and the DB point at a tmp directory. No
  mocks. A test needing a review gate writes the same `*.request.json` the
  pipeline writes and asserts on the same `*.response.json` it polls for.
- Hypothesis: defects in the wiring between modules — a view that never passes
  `cfg`, a form field the route does not read, a status page that disagrees with
  the queue it links to — are invisible to unit tests and cheap to catch here.
- Verify by: `pytest tests/features/` (56 cases, ~5s, CPU-only). Two contract
  mismatches surfaced while writing them (the count endpoint's JSON key, the
  channel pause column) plus one real defect, QM-031.
- Note: the fixture asserts through the code the product uses, not through
  storage details — `test_pausing_a_channel_stops_it_being_scheduled` queries
  `list_channels(status="active")`, the scheduler's own call, so it survives a
  change of column name and still fails if pausing stops working.

<!-- The entries below came from the qm/audit-2026-07-22 branch, which allocated IMP-015 onward independently of this branch. Both continued from main's IMP-013, so the same IDs named different findings. The audit's were renumbered to follow (the smaller side: 3 commits vs 25), and its code cross-references updated to match. Content is unchanged. -->

### IMP-041 Rubric routes retry hints by fault type   [shipped this audit → monitoring]
- Defect: QM-017 — content rejects flip to accept only 3/26 (11.5%) on retry, versus 78–88% for
  mechanical rejects, because essentially every content reject emitted `seed_bump` alone and a
  reseed cannot change what the prompt asked for.
- Change: `.claude/agents/scene-plausibility.md` gains a "Hint routing by rubric dimension" block
  under the existing one-line rule (which is kept as the fallback). Mechanical/sampling faults →
  `seed_bump`; dimensions 5/6/9/10 (action fidelity, setting fidelity, staging, continuity drift)
  → `simplify_action`; single-subject identity substitution → `force_i2v`; multi-child identity →
  `seed_bump` plus naming every off-cast child in `reasons`. Explicitly warns against setting all
  three as a shotgun. **No accept/reject criterion, weight or threshold changed, and the verdict
  JSON schema is untouched** — this only changes which retry the pipeline attempts next.
- Hypothesis: routing a plan fault to `simplify_action` instead of `seed_bump` lets the retry
  change the thing that was actually wrong, so content rejects start converging instead of burning
  the attempt budget.
- Metric / verify by: next audit — flip-to-accept rate for prose/content rejects (baseline 11.5%,
  3/26); share of content rejects emitting `seed_bump` alone (baseline ~100%); shots ending on a
  rejected final attempt (baseline 12). **Prerequisite: a vision reviewer must actually be running
  (QM-010) — with no reviewer there are no prose rejects to route, and this stays as unexercised
  as IMP-005 was.**

### IMP-042 Heuristic verdict carries frame-diff telemetry   [shipped this audit → monitoring]
- Defect: QM-016 — strobe/glitch on 5/35 (14.3%) of i2v-anchored takes vs 0/80 unanchored, with
  two incompatible explanations (real i2v flicker vs a false positive from the still→motion
  transition at the head of the clip) that the reasons list alone cannot separate.
- Change: `pipeline/kidsong/review.py`, `HeuristicReviewer.review` returns an additive optional
  `metrics` key — `frames`, `mean_frame_diff`, `max_frame_diff`, **`max_diff_at`** (index of the
  sampled frame PAIR that peaked), `frame_diffs`, `median_lap_var`, `luma_min/max`. Additive only:
  `accept`/`score`/`reasons`/`retry_hints` are untouched and unreordered, every consumer reads the
  verdict with `.get()`, and the module self-test asserts only on `accept`/`reasons`. **No
  threshold was moved.** The key flows through `ExternalReviewer` (`dict(heuristic)`) into
  `review_log.json` and into the vision reviewer's request payload for free.
- Hypothesis: if anchored strobe flags cluster at `max_diff_at == 0`, the check is misreading the
  i2v still→motion transition and the threshold needs recalibrating **with evidence** (the QM-007
  precedent — that is the only justification the hard rules accept for loosening a gate). If they
  peak mid-clip, the anchoring really is introducing flicker and the fix is upstream, most likely
  the motion-only i2v prompt (proposed below).
- Metric / verify by: next audit — distribution of `max_diff_at` on `i2v_anchored=true` takes
  flagged strobe/glitch; ≥5 such takes needed before drawing a conclusion. Also gives the next
  audit a real distribution for `median_lap_var`, which QM-007 had to argue from spot checks.

### IMP-043 Shotlist gate surfaces invented cast names   [shipped this audit → monitoring]
- Defect: QM-015 — 27 shot slots across 4 episodes name a child that resolves to no cast-bible
  entry; `cast.unresolved_names()` was written explicitly for this gate and **had no caller**, so
  the render silently substituted a bible child and nobody ever saw it.
- Change: `pipeline/kidsong/script_qc.py` gains `_unresolved_cast_names(shots)` and
  `review_shotlist` adds an advisory `unresolved_character_names` key to the external gate's
  request payload when (and only when) there is something to report — so an untouched shot list
  still writes a byte-identical request. **Deliberately NOT a `_check_shotlist` reason:**
  `_verdict` sets `accept = not reasons`, so that would make an invented name a hard shotlist
  reject and a replan loop, and there is no frequency data for the current director prompts.
  Cannot change any programmatic verdict; it only puts the fact in front of the reviewer.
- Hypothesis: surfacing the name at the plan stage is what stops it reaching the render, where the
  substituted child loses both QM-001 mitigations (no reference to anchor on, no bible-derived
  cross-negation).
- Metric / verify by: next audit — count of shot slots per episode carrying an unresolvable or
  alias-only name (baseline 27 across 4 of 32 episodes; target 0), read off the new payload key.
  If the count is non-zero on current director prompts, promote the check to a blocking
  `_check_shotlist` reason with a `_SHOTLIST_WEIGHTS` entry (proposal, see report).

### Proposals from this audit (NOT implemented — owner call)
- **P1 (QM-016) Motion-only i2v prompt.** `generate.py` builds `base_patches` once with
  `"PROMPT": {"text": _shot_prompt(render_shot, song, cfg)}`; the i2v branch copies it and swaps
  only `INPUT_IMAGE`/`SIGMAS`, so an anchored render re-describes the style, cast appearance,
  wardrobe and setting that the conditioning keyframe has already fixed. The vendor i2v guide
  (SOURCES, 2026-07-25) names that as the cause of jittery motion. Proposal: an i2v-specific prompt
  variant that keeps the action and camera move and drops the redundant appearance/setting block.
  Verify with a same-seed A/B on the two worst strobing anchored shots. Not implemented because it
  is a render-behaviour change that cannot be validated without a GPU.
- **P2 (QM-018) Closing-group invariant fallback.** `director.py` ~2659 no-ops when the final verse
  is entirely `reuse_of`, which is 18/32 shot lists. Proposal: fall back to the last verse that has
  a unique shot. One-line change, but it alters shot planning, so it needs a render to verify.
- **P3 (QM-015) Promote the invented-name check to blocking** once IMP-043's telemetry shows the
  frequency under current director prompts.
- **P4 (QM-010) Close the `reviewer: "heuristic"` escape hatch.** Setting that one value disables
  both `min_vision_coverage` and `require_group_vision` at once, and the episode still promotes to
  Final/ — the newest episode in this checkout (alle-vögel) is in exactly that state, with zero
  response files and a cut accepted at 1.0. Proposal: refuse promotion (or require an explicit
  `allow_unreviewed_promote: true`) when the run had no vision path at all, so turning the reviewer
  off is a deliberate, recorded act rather than a silent one.
- **P5 (QM-005/QM-009) Backfill and gate the CUT review — highest-severity proposal of this run.**
  Only 3 of 19 cut requests were ever answered, and **2 of those 3 said `{"accept": false,
  "score": 0.4, "reasons": ["creepy"]}`** (rainbow-friends-color-fun, skip-to-my-lou) — neither was
  ever consumed, because a `cut.response.json` that lands after `external_timeout` has no backfill
  path (per-take responses do; the cut does not). Both episodes were stopped by unrelated checks.
  Proposal: (a) on `--resume/--finalize`, read an existing `cut.response.json` off disk before
  promoting, mirroring `review.vision_coverage`'s documented late-answer recovery; (b) treat a
  programmatic-fallback cut verdict as uncovered for promote purposes. Not implemented this run —
  it changes the promote path and is kept unbatched from IMP-041/016/017 for attribution.
- **P6 (QM-011) Unchanged and still open — the standing S1.** Default `render_style` off
  `pixar_toon` onto claymation/felt, intro-badge redesign, legal sign-off. **Not re-verified this
  run: no pixels in this checkout.** `config.example.json` was NOT touched by this audit; all four
  render styles including claymation and felt are intact (diffed — see report).

---

## 2026-07-29 cloud audit — verification pass + four owner fixes logged

**Data freshness (unchanged from the 2026-07-25 pass, restated because it governs every status
below).** `output/` still stops at `20260722-135241`; `git log -1 --format=%ci -- output/` is still
2026-07-22. **No new episode telemetry synced since the last audit**, so every number in this file
carried over from 2026-07-25 is UNCHANGED, not re-measured, and no metric moved because no metric
could be read. No media (mp4/png/wav) and no live `config.json` exist in this checkout: the
"inspect finals" phase was **skipped, not performed** — no contact sheets, no frame extracts, no
waveform/loudness/spectral listen. `config.example.json` was used as the config reference and was
**not modified by this audit**; all four render styles (`pixar_toon`, `flat_storybook`,
`claymation`, `felt`) verified present on the branch after rebase (see the 2026-07-21 Incident
above — that check is now standing procedure).

### Re-verification of prior entries against the four new owner commits

- **IMP-005 / IMP-006 / IMP-014 (identity anchoring) — statuses UNCHANGED (INCONCLUSIVE /
  monitoring), but one of the four commits invalidates the comparison the next audit was going to
  make.** Nothing in `ada0cbf`, `c534c3c`, `f1d78b8` or `4d30322` touches the anchor path. However
  `025ed80` (2026-07-24, already on `main` before this audit) put the i2v path through `hires_pass`
  and capped `kidsong.shot.max_frames` at 81 because "89+ renders black" on that graph. Anchored
  takes therefore run a **different renderer** than the one that produced QM-016's 14.3% strobe
  baseline, and `4d30322` exists precisely because that 81-frame cap collided with the director's
  4.5 s slot ceiling. Action recorded in QM-016: re-baseline rather than compare; and test the new
  hypothesis (c) (envelope overrun ⇒ black tail frames ⇒ one huge frame delta) using IMP-042's
  `max_diff_at` + `luma_min`, which needs no further code.
- **IMP-008 (vision-coverage promote gate) — status UNCHANGED (monitoring).** None of the four
  commits touches promotion, coverage or the reviewer wiring. Its open problem is still QM-010 (the
  gate auto-disables under `reviewer: "heuristic"`), and proposal P4 stands unactioned. Worth
  noting for the next audit: three of the four new fixes (`ada0cbf` canon, `f1d78b8` grow beats,
  `4d30322` fill policy) were found by the channel owner **eyeballing local renders**, not by any
  gate — which is the same evidence QM-010 rests on, from the other direction.
- **IMP-041 (hint routing) — UNCHANGED (monitoring), and it now has a new consumer.** `ada0cbf`
  adds phantom heads as a hard reject inside rubric dimension **2**, which IMP-041 routes to
  `seed_bump` as a mechanical fault. Judged correct (a motion-stage artefact is a per-roll
  artefact, and mechanical rejects historically flip 78–88%), but flagged in QM-023 as an
  assumption to confirm from the first post-fix telemetry rather than assert.
- **IMP-042 (frame-diff telemetry) — UNCHANGED (monitoring), value increased.** It was built to
  separate two hypotheses for QM-016; it now separates three, at no extra cost.
- **IMP-043 (invented cast names) — UNCHANGED (monitoring).** Untouched by the four commits.
- **IMP-001 / IMP-002 / IMP-003 / IMP-004 / IMP-012 / IMP-013 — UNCHANGED.** No new evidence.
  IMP-001's "trending validated" note deserves one caution from `f1d78b8`: that commit shows the
  director/storyboard *beat templates* can phrase an action in a way that silently disables a
  reviewer dimension (QM-021). IMP-001 credited the director prompt with the drop in lockstep
  staging; QM-021 is the reminder that plan text can subtract from the gates as well as add.

### IMP-044 Episode canon — author-picked subject look + one setting per song   [shipped `ada0cbf` (owner) → **shipped, UNVERIFIED in this checkout**]
- Defects: QM-019 (story-subject appearance drift toward a third-party prior) and QM-006
  (within-verse → within-episode setting drift).
- Change (channel owner, 2026-07-25, from local render testing): every `prompts/pd_songs.json` /
  `pd_songs.de.json` entry gains `subject_description` (the fixed appearance of the song's non-child
  story subject, `null` where the song has none) and `canonical_setting` (the episode's one
  location). `lyrics.py` stamps both into the song dict, so shot prompts, keyframe prompts and the
  location planner all read the same author-picked canon. Ships with tests
  (`tests/test_kidsong_pd_lyrics.py`, `tests/test_scene_coherence.py`).
- Pre-fix measurement (the owner's, not mine): baa-baa rendered "a black sheep in some keyframes and
  a white Shaun-lookalike in others", and "slid from the farm into a generic backyard mid-episode".
- Hypothesis: the story subject and the setting drift because nothing pins them — the cast bible
  pins children only, and the setting was re-described per shot. Pinning both once per song, at the
  library level where a human authored them, removes the drift at its source instead of asking a
  gate to catch it. Corroborated by research this audit (SOURCES 2026-07-29): the current multi-shot
  consistency literature converges on exactly this shape — a **global caption fixing scene,
  characters and environment plus per-shot captions carrying only action/camera** (MultiShotMaster),
  and "layered prompting" that separates global scene description from shot-specific entities
  precisely to prevent leakage (GroundShot). `ada0cbf` is the text-only version of that pattern.
- **Status: shipped, unverified here.** The fix commit exists and states its own pre-fix
  measurement; **no post-fix `output/` telemetry has reached this checkout, so no metric win is
  claimed.**
- Metric the NEXT audit should check, once new episode data lands: (1) static, checkable today and
  **verified true on this branch (2026-07-29): 18/18 library songs (10 en + 8 de) carry a non-empty
  `canonical_setting`, and all 18 carry the `subject_description` key — 11 with an author-written
  value, 7 explicitly `null` for songs with no non-child subject. No entry is missing either key**;
  (2) per-episode
  count of shots whose setting text departs from the song's `canonical_setting` (target 0);
  (3) subject-appearance drift rejects on the first vision-reviewed post-`ada0cbf` episode;
  (4) standing S1 tolerance — zero recognisable-third-party-character findings in any final.
- Watch item for the next audit (not a defect, a consequence): one locked setting per episode is a
  strong lever against QM-006, and it pulls against the final-cut rubric's **story arc** line
  ("settings and actions progress with the verses"). If post-fix episodes read as visually static —
  same room for 60 s — the fix is a *canonical setting with named sub-areas*, not reverting the pin.
  Check this by eye on the first post-fix final; it cannot be measured from logs.

### IMP-045 Phantom heads are a hard reject in the anatomy dimension   [shipped `ada0cbf` (owner) → **shipped, UNVERIFIED in this checkout**]
- Defect: QM-023.
- Change: `.claude/agents/scene-plausibility.md` dimension 2 (ANATOMY) now names, as an explicit
  hard reject, "a partial face at a frame edge, an out-of-focus back-of-head silhouette looming in
  the foreground (over-the-shoulder blob), or any disembodied/duplicate head … even when blurred,
  cropped or 'just scenery'", attributing them to the i2v video stage growing them during motion on
  keyframes that were clean (measured on twinkle-twinkle 2026-07-24, multiple takes).
- Hypothesis: the defect was reaching finals because it did not *look* like the failure the rubric
  described — a reviewer counting children does not count a blurred edge-of-frame head, so naming
  the shape is what makes the existing dimension fire on it.
- Compliant with the hard rules: **strictly stricter**, no threshold loosened, no verdict-schema
  change. Rebased cleanly alongside IMP-041's hint-routing block in the same file (both present and
  verified after the rebase).
- **Status: shipped, unverified here** — no post-fix telemetry.
- Metric the NEXT audit should check: (1) phantom-head/extra-head rejects per episode (expect a
  *rise* first — a stricter rubric finds more before upstream work reduces it; a rise is the
  mechanism working, not a regression); (2) their flip-to-accept rate on the next attempt — a low
  rate means IMP-041 is routing them to the wrong hint (dimension 2 ⇒ `seed_bump`) and they are
  compositional, not mechanical; (3) whether they concentrate on `i2v_anchored=true` takes (ties to
  QM-016) and on tail frames (the LTX-2.3 model card documents last-6–8-frame smearing — SOURCES
  2026-07-29 — which would mean our evenly-spaced 5-frame review sheet under-samples the worst
  region). **Proposal, not implemented: bias take-review frame sampling toward the clip tail.**
  Deliberately unbatched — it changes the evidence every take verdict is formed from.

### IMP-046 Sing coherence gate on fresh renders   [shipped `c534c3c` (owner) → **shipped, UNVERIFIED in this checkout**]
- Defect: QM-020 (scrambled ACE-Step arrangements shipping), and the first automated coverage this
  channel has ever had on the **audio** dimension of QM-009's rubric gap.
- Change: after a *fresh* sing, `generate.py` measures the render's beat-grid phase-fit resultant
  (`edit.py`'s `_fit_phase`) and re-sings with seed+1, seed+2 … while it is below
  `kidsong.sing_coherence.min_resultant` (default **0.25**), up to `max_attempts` (**3**), shipping
  the highest-scoring attempt. Rejected attempts are **kept** as `.rejected-aN.wav` — inventory
  preserved, per the hard rules. A resumed run never re-sings (re-singing would desync the beat grid
  every already-cut take was aligned to). Two supporting changes: the per-song `song_style` caption
  from the library now wins over the global default (lullabies stop being sung upbeat), and ACE
  stdout (planned bpm/key/seed) is mirrored into the run log instead of discarded. `edit.py`'s
  degenerate-beat-grid path now carries numeric `resultant`/`rayleigh` so the gate reads them
  directly instead of parsing prose.
- Threshold justification (recorded because moving thresholds needs evidence): 0.25 sits just above
  the measured scrambled cluster (0.235, 0.068) and below the clean 0.54, and matches `edit.py`'s
  own `_GRID_MIN_RESULTANT` floor. This is a **new blocking-ish gate**, i.e. stricter — permitted
  freely; nothing was loosened.
- Hypothesis: ACE-Step's arrangement is a per-render dice roll, so the cheapest correct response is
  to measure the roll and re-roll it, rather than to hand-fix the audio downstream.
- **Root-cause corroboration found this audit (new territory for SOURCES — no music-generation
  source existed before today):** the upstream ACE-Step-1.5 inference docs confirm the mechanism
  exactly — a 5 Hz "thinking" LM with `use_cot_metas=True` populates `cot_bpm`, `cot_keyscale` and
  `cot_timesignature` per render, the exposed `seed` applies to the diffusion process, and **no
  separate CoT seed is documented**; LM randomness is governed by `lm_temperature` / `lm_top_k` /
  `lm_top_p`. Community reports call the resulting variance "gacha-style". The gate's premise is
  therefore correct as written.
- **Proposal that follows directly from that research, NOT implemented (one change per defect class
  — the gate must be allowed to prove itself first):** the same docs note the LM plan can be
  bypassed by **setting `bpm` explicitly**, and that `infer_method='ode'` (Euler, the default) is
  deterministic while `'sde'` is not. If the gate turns out to fire often (metric 2 below), the
  cheaper fix is to *stop rolling* — pin bpm per library song, next to the `song_style` caption this
  same commit started honouring — rather than to keep re-rolling up to 3× at full render cost.
  Worth a decision only once the fire rate is known.
- **Status: shipped, unverified here.** No `.wav`, no new `-song.json`, no run logs in this checkout.
- Metric the NEXT audit should check: (1) shipped-sing `resultant` per new episode — none below
  0.25, mean at or above 0.54; (2) `.rejected-aN.wav` count per episode = how often the gate fires
  (this is the number that decides the pin-bpm proposal above); (3) episodes where all
  `max_attempts` were spent and a sub-threshold sing shipped anyway (target 0); (4) whether
  per-song `song_style` is actually present in the library entries being sung.

### IMP-047 Grow beats describe the grown STATE, not a transformation   [shipped `f1d78b8` (owner) → **shipped, UNVERIFIED in this checkout**]
- Defect: QM-021 — the take reviewer *accepted* a sheep morphing white→black inside one shot,
  because the beat's own action text ("changes a little more, growing steadily") made the morph
  read as action fidelity. Only the cut gate caught it.
- Change: `storyboard.py`'s grow-beat template now states the result ("looks a little fuller and
  happier than before"), keeping one activity verb per beat so the verse still classifies as
  narrative; the companion beat drops "delighted at the change" → "claps once in delight". Plus the
  baa-baa `subject_description` is made explicit ("whole fleece deep black") so the prompt beats the
  Shaun-the-Sheep prior.
- Hypothesis: an i2v stage asked for a *process* will render the process — including the "before"
  state it was never supposed to show. Describing the end state gives the model a single target and
  removes the reviewer's excuse to accept the morph.
- Why this is the most transferable of the four fixes: it is the only one where the *plan* silently
  disabled a *gate*. Generalised rule for future beat/prompt authoring, recommended for the director
  prompt as a standing constraint: **never phrase a beat as an ongoing change of appearance.**
  Actions change; appearances do not, inside one take.
- **Status: shipped, unverified here.**
- Metric the NEXT audit should check: (1) zero shot-list action texts describing an ongoing
  appearance change (grep "changes"/"turning"/"growing"/"becoming"/"transforms" across
  `*-shots.json` — cheap, needs no media); (2) zero verdicts that ACCEPT visible mid-take morphing
  citing the shot's own action text; (3) no rise in cut-gate premise rejects, which is where this
  defect used to surface.

### IMP-048 Small cut-window shortfalls slow the take down instead of reversing it   [shipped `4d30322` (owner) → **shipped, UNVERIFIED in this checkout**]
- Defect: QM-022 — final cuts visibly played shots forward-then-backward (user report, 2026-07-25
  baa-baa), because `max_frames` had dropped to the i2v-hires black-frame envelope (81 f = 3.375 s)
  under the director's old 4.5 s slot ceiling, firing the editor's draft-tier ping-pong fill on
  essentially every final shot. **Frame-sampled cut review is direction-blind, so no gate could
  catch it** — that blind spot is now logged against QM-009.
- Change, two layers: (a) `edit._extend_to_window` covers a shortfall within `_MAX_SLOW_STRETCH`
  (1.15×) by slowing the take to the window, leaving ping-pong for draft-scale gaps only;
  (b) `director._shot_bounds` clamps `max_shot_seconds` to the renderable take length
  `(max_frames − fps//4)/fps`, so a final plan can no longer request a slot real footage cannot
  cover. Ships with tests (`tests/test_kidsong_no_scene_repeat.py`). Layer (b) is the one that makes
  the fill path unreachable on finals; layer (a) is the safety net.
- Hypothesis: two independently-tuned config values (render frame cap, planned slot length) with no
  guard between them will silently route finals through a draft-only fallback; clamping the plan to
  what the renderer can actually produce removes the coupling.
- **Status: shipped, unverified here.**
- Metric the NEXT audit should check: (1) per-episode count of shots shorter than their cut window,
  split slow-stretched vs ping-ponged — **target 0 ping-pong on any non-draft cut**; (2) unique
  shot-length count per final (target > 1 — see residual 2 below); (3) zero further reversed-motion
  reports.
- Two residuals recorded in QM-022, neither actioned here (each needs a render, and batching them
  onto a just-shipped fix would destroy attribution): **slow-stretch is not beat-neutral** (a ≤15%
  slowdown moves in-shot action off the grid; `cut_qc` only measures cut *boundaries*, so nothing
  measures this — confirm by ear on the first post-fix final), and **`_shot_bounds` returns
  `min(min_s, max_s), max_s`**, so a low enough `max_frames` silently collapses `min_shot_seconds`
  onto the clamp and every shot becomes the same length — which would violate the final-cut rubric's
  "shot lengths breathe" line. Not live at the default `max_frames` 241; a config trap worth knowing.

### Proposals carried forward unchanged from 2026-07-25
P1 (motion-only i2v prompt), P2 (closing-group invariant fallback), P3 (promote the invented-name
check to blocking), P4 (close the `reviewer: "heuristic"` escape hatch), P5 (backfill and gate the
CUT review), P6 (QM-011 — the standing S1: default `render_style` off `pixar_toon`, intro-badge
redesign, legal sign-off). **All six are still unactioned and none is superseded by the four new
commits.** P5 and P4 gain weight from this run's observation that three of the four new fixes were
found by the owner watching renders rather than by any gate. New this audit: **P7** pin `bpm` per
library song instead of re-rolling the sing (see IMP-046), and **P8** bias take-review frame
sampling toward the clip tail (see IMP-045) — both deliberately held until the just-shipped fixes
have telemetry to be judged on.

## 2026-08-01 audit — no new telemetry (4th consecutive blind pass); one telemetry fix shipped

`output/` is still pinned at `20260722-135241` — byte-identical to the 2026-07-25/-29 passes on this
same branch. Every number this pass could have re-measured is unchanged for the same reason those two
passes already gave: nothing rendered since the checkpoint. Independently re-ran the full pattern-mine
(all 23 `review_log.json` files, per-take and per-shot vision-coverage definitions, cast-integrity/
wardrobe reason classification, seed-bump and force_i2v flip rates, shot-type failure rates) from a
fresh read of the raw JSON rather than by trusting the prior pass's numbers, specifically to
independently check IMP-005 per this run's brief (compare git-log dates for the identity-anchor work
against episode timestamps). **Converged on the same conclusion as the 2026-07-25 pass, via a
different methodology:** using the "final take used per shot" coverage definition (the one the
`min_vision_coverage` promote gate itself uses), the 11 episodes from `baa-baa-black-sheep` (07-20
18:17) through `rain-rain-go-away` (07-21 15:00) are **100% vision-covered, 160 takes, and show ZERO
cast-integrity or wardrobe rejects** — the entire 6–9-take cast-integrity/wardrobe signal that set the
6.1–6.8% baseline is concentrated in the channel's first two episodes (`bennys-brushing-fun`,
`rainbow-friends-sing-together`, both 2026-07-19), before whatever cast_text-enrichment/negative-term
fix landed between 07-19 and 07-20. **IMP-005/006/012/013/014 (the image-anchor mechanism) remain
exactly as inconclusive as both prior passes found: zero overlap, in this checkout's entire history,
between any take with `i2v_anchored=true` and any take with a real `.response.json` on disk** — the
anchor never fired before vision coverage collapsed, and coverage never recovered before the checkout
was left at 07-22. Re-confirms IMP-006/IMP-014's status verbatim; no new number to log for them.

### IMP-049 `review_log.json` gains a `run_config` entry   [shipped this audit]
- Defect: QM-024 — three straight audits (07-21, -25, -29) had to reverse-engineer the effective
  `kidsong.review.*` config (reviewer backend, `min_vision_coverage`, `shot_auto_accept.enabled`) for
  a given episode from git-commit timestamps against `config.example.json`'s history, because neither
  the live `config.json` nor `output/<base>.log` (where this is already logged once, via
  `runlog.stage("quality", ...)`) reaches the cloud checkout — only `review_log.json` does.
- Change: `_generate_director` (`pipeline/kidsong/generate.py`) appends one additive entry —
  `{"run_config": {"quality_tier", "reviewer", "min_vision_coverage", "shot_auto_accept_enabled"}}`
  — to `review_log_entries` immediately after the existing `runlog.stage("quality", ...)` call, using
  values already computed there. No new config reads, no behavior change, no gate touched. Verified
  additive: `runstate.py`'s and `status.py`'s `review_log.json` readers both skip entries lacking a
  `shot`/`stage` key, and the `min_vision_coverage`/`require_group_vision` gates read `.response.json`
  files off disk, not this log, so they cannot be affected by a new sibling entry. `tests/
  test_kidsong_vision_gate.py`, `test_kidsong_vision_coverage.py`, `test_kidsong_runlog_resume.py`
  (69 tests covering `review_log.json` structure and resume/backfill logic) all pass unmodified.
- Hypothesis: with the run's effective review config on disk in the one file the cloud audit can
  read, the next audit spends zero effort reconstructing *why* an episode's coverage is what it is,
  and can instead spend that effort verifying whether it *should* be that way.
- Metric: next audit — does the first `output/<base>-shots/review/review_log.json` written by a real
  render contain a `"run_config"` entry, and does it match the episode's actual measured coverage
  (e.g. `shot_auto_accept_enabled: true` alongside near-0% coverage, or `false` alongside recovered
  coverage — closing the loop IMP-044 through IMP-048 all noted was still open on `5d91046`).

### IMP-050 A published video says a machine made it   [shipped 2026-08-02]
- Defect: QM-042 — nothing in the repo disclosed AI generation, on any surface.
- Change: new `pipeline/ai_disclosure.py` (config resolution + an idempotent description note +
  the YouTube flag + container tagging), wired at exactly three points:
  1. `youtube_upload.upload()` appends the note and sets `status.containsSyntheticMedia`. Chosen
     over patching the three call sites precisely because the failure mode is a call site that
     forgets — `pipeline/generate.py` and both `scheduler._process_uploads` branches now inherit
     it without mentioning it.
  2. `edit.build_final_cut` stamps the note into the container between `_post_grade` and the
     closing `verify_render`, so the file that gets verified is the file that gets published.
  3. `assemble_song.build_song_video` does the same for the legacy slideshow path.
- Two decisions worth stating, because both are places where the obvious implementation is wrong:
  * **Truncation order.** `upload()` capped the description with `description[:4900]`. Appending
    the note and then slicing would let a long description push the disclosure off the end and
    publish undisclosed with no error anywhere. `append_note` therefore truncates the
    *operator's* text to make room and the disclosure is never the part that gets cut.
    (`test_a_description_at_the_cap_still_ends_with_the_disclosure`.)
  * **`youtube_synthetic_media: "auto"` resolves to False for animated types.** YouTube's
    altered/synthetic disclosure is for *realistic* content and explicitly exempts wholly
    animated work, so declaring a toon would be a wrong label, not a cautious one. A version
    that answered True everywhere would pass a naive "is it disclosed?" test, which is why both
    halves are pinned.
  The caller override is opt-**in** only — it can add a declaration, never remove one — mirroring
  `scheduler._made_for_kids` (IMP-030) for the same reason: the failure that matters is the
  declaration that silently goes missing.
- Verification: 46 new tests. `tests/test_ai_disclosure.py` (34) covers composition, the cap, the
  auto/explicit/override matrix and best-effort tagging (2 skip without ffmpeg).
  `tests/features/test_disclosing_ai_generation.py` (12) runs the **real** `upload()` and reads the
  dict handed to `videos().insert`, including both scheduler branches end to end — only the
  transport is stubbed. Negative control: reverting the two production lines fails 10 of those 12.
  Full suite 2073 passed / 14 skipped.
- Hypothesis: disclosure becomes a property of the uploader rather than of whoever remembered,
  so a new video type or a fourth call site is disclosed on the day it is added.
- Metric: next real upload — see QM-042. Plus: does any future call site reach
  `videos.insert` without passing through `youtube_upload.upload()`? If one ever does, this
  design has failed and the note belongs lower down.

### IMP-051 The COPPA declaration follows the style, not just the video type   [shipped 2026-08-05]
- Defect: QM-043 — a `style="kids"` channel with an unticked box and a non-kids `video_type`
  published children's content declared as not children's content.
- Change: new `ideas.is_kids_style()` beside the existing `is_kidsong_type()`;
  `scheduler._made_for_kids` consults it as a third signal (job style first, then channel style);
  `cascade._kidsong_affinity` now calls the predicate instead of re-inlining the same tuple, so the
  two places that judge "is this kids?" cannot drift apart again.
- All three signals stay opt-**in**: any one of them turns the declaration on, none can turn it off.
  Same asymmetry as IMP-030 and IMP-050, for the same reason — the failure that matters is the
  declaration that silently goes missing.
- Bounded on purpose: prose is still not a signal. See QM-043's last bullet for why, and note the
  declaration is not costless in the over-declaring direction either, which is what keeps this from
  becoming "declare everything and be safe".
- Verification: 5 new tests in the COPPA section of `tests/test_scheduler_bookkeeping.py`, all
  driving the real `Scheduler._process_uploads`. Negative control: removing the style signal fails
  exactly the 2 that assert it, and none of the others — so the new tests are testing the new
  behaviour and the old ones still guard the old. Full suite 2080 passed / 14 skipped.
- Hypothesis: the declaration now tracks what the video *is* rather than which column was filled in,
  so repointing a channel's video_type — the exact move that created this hole — cannot reopen it.
- Metric: next real upload from a kids-styled channel; see QM-043.

### IMP-052 Multi-Channel-Studio: Profil, Branding, Dashboard   [shipped 2026-08-10]
- Ziel: mehrere Kanäle aus einem Dashboard fahren, bei dem der Betreiber nur noch Videos freigibt.
- Kanalmodell (Migration v9): `language`, `branding_json`, `content_profile`. `content_profiles` in
  der Config bildet einen Kanal auf Generator, Sprache und Look ab — heute kidsong_de/en,
  hyperframes_de/en, brainrot_short.
- Zwei Entwurfsentscheidungen, bei denen die naheliegende Variante falsch gewesen wäre:
  * Das aufgelöste Profil ist eine **lokale** Bindung im Job, nie `self.cfg`. Ein Scheduler-Objekt
    bedient prozesslang alle Kanäle; ein Rückschreiben hätte das Studio in der Sprache des zuletzt
    gelaufenen Kanals hängen lassen, und auffallen würde das erst beim zweiten Kanal.
  * Ein Profil überstimmt den video_type der Ideenzeile nur, wenn es wirklich gesetzt ist.
    `profile_video_type` fällt sonst auf die Default-Spalte zurück, und bedingungslose Nutzung
    hätte für jeden profillosen Bestandskanal die Ideenpräferenz still entwertet.
  Beide sind in beide Richtungen negativkontrolliert.
- Branding greift beim **Schreiben** in die `videos`-Zeile, nicht beim Upload: der Betreiber kann nur
  freigeben, was er sieht, und beim Upload gestempelte Metadaten hätten Review-Seite und YouTube
  auseinanderlaufen lassen. Der KI-Hinweis folgt der Kanalsprache (de/en getrennt) — ein Hinweis,
  den der Zuschauer nicht lesen kann, erfüllt seinen Zweck nicht.
- Zwei Entscheidungen aus dem UI-Durchgang, die über den Auftrag hinausgingen und bleiben: eine
  Vorlage mit unbekanntem Platzhalter (`{thema}`) wird mit 400 abgelehnt statt still entleert, und
  die Kanalfarben sitzen hinter einer Checkbox, weil `type=color` keinen Leerzustand hat und sonst
  jedem gespeicherten Kanal `#000000` verpasst hätte.
- Dabei gefunden und behoben: `value="{{ schedule_raw if schedule_error else ... }}"` setzte bei
  jedem Fehler AUSSER einem Zeitplanfehler die getippten Uhrzeiten auf den DB-Stand zurück.
- Workflows: 4 neue Graphen (`ltx23_t2v_explainer[_hires]`, `ltx23_t2v_meme[_hires]`) als Ableitung
  ihres Toon-Pendants mit exakt 3 Deltas — LoRA-Stärke 0.0, angepasster NEGATIVE-Prompt, kosmetische
  Platzhalter. Auflösung, Frames und Sigmas unverändert, damit das 12GB-Budget der 5070 hält und der
  Titel-Patch-Vertrag weiter greift. Ein Validator prüft alle 12 Graphen auf doppelte Node-Titel,
  baumelnde Referenzen und Pflicht-Titel; in den 8 bestehenden fand er keinen Defekt.
- Verifikation: Suite von 2080 auf 2285. Zusätzlich in echtem Chromium vermessen, 6 Seiten x 2
  Viewports: alle 200, keine JS-Fehler, kein Seitwärts-Scroll (siehe QM-044, das dabei auffiel).
- Metric: nächster echter Betrieb — produziert ein Kanal mit `content_profile: hyperframes_de`
  deutsche Erklärvideos, ohne dass jemand eine Zeile Metadaten tippt?

### IMP-053 Jede Kanal-Sorte einmal durch alle Schichten   [shipped 2026-08-10]
- Defect: QM-045 — die zwei neuen Kanal-Sorten waren vollständig verwaltbar und produzierten nichts.
- Change:
  * `prompts/brainrot.txt` — dieselbe JSON-Zusage wie `facts.txt`/`hyperframes.txt`, mit den
    Monetarisierungsgrenzen wörtlich aus `ideas.BRAINROT_IDEA_SYSTEM`, damit Ideen-Generator und
    Skript-Generator nicht auseinanderlaufen.
  * `script_gen` bekommt eine Zielsprache. Reihenfolge: expliziter Parameter, dann Top-Level
    `cfg["language"]`, dann `cfg["kidsong"]["language"]`, dann Englisch. Der Top-Level-Key ist der,
    der es BEDEUTET — ein Meme-Short hat nichts im Kidsong-Block zu suchen; der alte Key bleibt als
    Rückfall in der Kette, weil jedes Profil ihn noch pinnt und ein Entfernen bestehende deutsche
    Kanäle still auf Englisch zurückgestellt hätte.
  * Die Profile setzen jetzt die Keys, die der allgemeine Pfad wirklich liest — belegt statt geraten:
    Stimme `voices.<sprecher>` (`tts.py:37,61`), Transkriptionssprache `whisper.language`
    (`generate.py:159` → `captions.py:107`), Style-Preset über `apply_style` (`config.py:86`), und
    der ComfyUI-Graph über den gpu-scene-Tier (`gpu_scenes.py:93-104`), der doppelt gated ist
    (`video.gpu_scene_mode` UND `studio.gpu_scenes.enabled`) — beide Schalter gehören deshalb ins
    Profil, sonst bleibt der Graph unerreichbar.
- Regressionsgarantie: bei Englisch oder fehlender Sprache ist der Skript-Prompt byte-identisch zu
  vorher. Jeder bestehende Kanal dieses Studios ist englisch.
- Verification: `tests/features/test_producing_for_a_channel.py` (15) geht pro Kanal-Sorte einmal
  durch echten Scheduler, echtes `generate()` und echtes `script_gen`; gemockt ist nur, was Geld oder
  GPU kostet. Bewusst verhaltensbasiert — geprüft wird, dass ein deutscher Kanal ein deutsches
  Skript und eine deutsche Stimme bekommt, nicht welcher Key das bewirkt; der Key hat sich während
  dieser Arbeit bereits einmal geändert. Dazu `tests/test_script_language.py` (35) und
  `tests/test_profiles_effective.py` (18) mit explizitem Wächter gegen "Profil setzt nur
  `kidsong.*`, obwohl sein video_type nicht kidsong ist". Suite 2285 → 2349.
- Negativkontrolle: rekonstruiert man den Ausgangszustand, fallen 9 Tests.
- Hypothesis: ein neuer Kanal-Typ fällt künftig beim Anlegen durch, nicht im Betrieb — 
  `test_jeder_registrierte_video_typ_kann_ein_skript_bauen` macht das Vergessen der Prompt-Datei
  zu einem roten Test statt zu einem fehlgeschlagenen Job.
- Offen, bewusst nicht geraten: die deutschen edge-tts-Stimmnamen konnten mangels Netz nicht gegen
  `edge-tts --list-voices` verifiziert werden. Vor dem ersten deutschen Upload einmal prüfen.
### IMP-054 A video-render backend seam, and an offline Seedance 2.5 client behind it   [shipped this session]
- Defect: QM-001 (cast identity drift) is the channel's top defect class, and ByteDance Seedance
  2.5 (announced 2026-07, up to 50 reference inputs) is a structural answer to identity anchoring
  that the local LTX-2.3 path cannot match on reference breadth. But the repo had no way to render
  a shot anywhere except a local ComfyUI graph, so the claim was untestable.
- Change, in three parts, all default-off and all $0:
  (1) `pipeline/kidsong/video_backend.py` — `make_video_backend(cfg, comfy_client)`. With
  `kidsong.video_backend` absent or `"comfy"` it **returns the ComfyClient itself by identity**, so
  "nothing changed" is assertable as `backend is client` rather than argued. `generate.py` gains
  `renderer = make_video_backend(...)` and 7 `client.` → `renderer.` renames confined to the
  shot-render path (lines 2081, 2100, 2176, 2188, 2191, 2206, 2287); stills, `_baseline_negative`,
  `runlog.fingerprint`, `_bootstrap_cast_references` and the final `free()` stay on `client`,
  because the ComfyUI client is *also* the Z-Image still renderer and must not move.
  (2) `pipeline/kidsong/seedance.py` — translate layer from the ComfyUI patch vocabulary
  (`comfy.py:79-96`) to a prompt/duration/reference API; `LORA_STYLE`/`SIGMAS`/`FILENAME_PREFIX`
  dropped, `FRAMES` → seconds snapped up (81 frames @ 24fps = 3.375s → 4s), resolution hard-capped
  at 720p in code because the channel's output is 1080p and shots render at 896x512.
  (3) Spend controls: `dry_run: true` by default (raises `SeedanceDryRun` before any HTTP call),
  a per-episode ledger checked before every submit, and a `base_url` host allowlist enforced at
  construction — third-party reseller aggregators are excluded **by policy**, this being a
  `made_for_kids` channel with unresolved training-data rights (see QM-011).
- Load-bearing detail: `SeedanceError(Exception)` is deliberately **not** a `RuntimeError`
  subclass, so budget/moderation/auth failures escape the render loop's
  `except (TimeoutError, RuntimeError, requests.RequestException)` at `generate.py:2170` instead of
  being retried four times at four times the cost. Asserted by
  `test_budget_exceeded_is_not_a_runtime_error`.
- Correction recorded: this work was planned on the premise that `force_i2v` had never fired
  because of a wiring gap. That was wrong — `.claude/agents/scene-plausibility.md:99-102` already
  routes `force_i2v` for single-subject substitution and `generate.py:2236` already logs
  `i2v_anchored`, both since `26fb0b9` (2026-07-21). The emitter that only covers duplicates is
  the cv2 heuristic (`review.py:357`), not the vision reviewer. The real reason the anchor has
  never fired is QM-010: vision review covers 0-6% of takes, so the emitter is rarely reached.
  Only `tests/test_kidsong_i2v_anchor.py` (4 regression tests) was added there.
- Verification: `tests/test_seedance.py` 35 passed, `tests/test_kidsong_video_backend.py` 5 passed,
  `tests/test_kidsong_i2v_anchor.py` + `test_kidsong_infra_retry.py` 7 passed,
  `tests/test_atomicio.py` 17 passed (its `test_every_media_download_publishes_atomically` module
  tuple now includes `seedance`). Repo-wide collection 2036 tests, 0 errors. All runs required the
  QM-046 cv2 stub workaround.
- **Not yet real:** no verified Seedance 2.5 REST endpoint or rate card exists as of 2026-08-02
  (`base_url` is a labelled placeholder), so the fixtures encode an *assumed* response shape. They
  prove the logic, not the integration. Nothing may be submitted until a real host and price are
  confirmed; GA may be 2026-08-07.
- Metric: (1) the default path stays byte-identical — `backend is client` holds and the render-loop
  regression pass count does not move; (2) before any Seedance spend, does raising vision coverage
  (QM-010) alone move `i2v_anchored` off 0 and cut the 6.8% cast-integrity reject rate — if the
  local anchor arm matches the Seedance arm, the cloud backend is not worth ~$6/episode; (3) if it
  is run: cast-integrity reject rate and first-attempt accept rate, >=3 episodes per arm, since at
  ~18 shots/episode one episode cannot separate 6.8% from 3%.

### IMP-055 The test suite runs in a container, so a green run means green   [shipped this session]
- Defect: QM-046 — the host venv lacks `cv2` and `librosa` (both pinned in `requirements.txt`), and
  `tests/conftest.py`'s autouse fixture imports `pipeline.kidsong.review`, which imports cv2 at
  module load. `pytest --collect-only` therefore reported a clean 2036 tests while any real run
  errored at fixture setup. Every verification this session had to load a throwaway cv2 stub via
  `PYTHONPATH`, which hides exactly the cv2-dependent behaviour (`review.py`'s face-count heuristic)
  that QM-001's `force_i2v` emission depends on.
- Change: `Dockerfile.test` (python:3.11-slim + ffmpeg + libgl1 + `requirements-dev.txt`) and a
  `.dockerignore` that keeps `.env`, `keys/`, `tokens/`, `config.json`, `output/` and `comfyui/`
  out of the build context. `STUDIO_PASSWORD=""` is set in the image so a test run can never
  inherit the host's password (the QM-031 bypass is gone, so an inherited password would 302 the
  web-layer tests).
  `docker build -f Dockerfile.test -t zubibop-tests . && docker run --rm zubibop-tests`
- Verification: **2082 passed, 2 skipped, 0 failed in 280.6s**, no stub, no collection errors.
  The host venv reports 12 skips for the same tree; the 10-skip gap is tests that skip on the host
  purely because a dependency is absent. CPU-only by design — no torch, no CUDA in the image.
- Metric: does `docker run --rm zubibop-tests` stay at 0 failed and <=2 skipped, and does any future
  audit stop needing a `PYTHONPATH` stub to report a number.

<!-- Brought over from qm/audit-20260722 (PR #6), which predated the merges of
     PRs #7-#10 and so reused ids main had already taken. Renumbered on the way
     in (IMP-050 -> IMP-056), cross-references inside the block updated to match.
     Content otherwise unchanged. -->
- **2026-08-05 re-verification — commit hash confirmed, status UNCHANGED (still INCONCLUSIVE).**
  Four straight audits inferred "post-a98e4a1" from episode timestamps because `a98e4a1` itself was
  unresolvable in every prior cloud checkout (shallow clone — see DEFECT_BACKLOG QM-047). This run
  unshallowed the clone and confirmed it directly: `a98e4a1` = **2026-07-21 10:22:20 +0200,
  "Anchor character identity with a Z-Image reference, via the dead force_i2v"** — exactly the
  commit and window every prior audit assumed. Independently re-derived the pre/post split three
  ways (review_log.json verdicts by episode timestamp; raw `s*_a*.response.json` files split into
  real vs. fallback; the QM-034 retry-table cross-check, reproduced exactly). All three converge on
  the same fact the 2026-08-01 audit already found by a different method: of the 80
  `s*_a*.response.json` files across the 10 post-commit episodes, **zero are a real, non-fallback
  vision review** — so there is no "post" data to compare against the pre-commit baseline, and the
  hypothesis remains untestable rather than false. No number in this entry changes; the ground it
  stands on is now confirmed instead of inferred.
## 2026-08-05 audit — 6th consecutive no-new-telemetry pass; one tooling fix, one proposal

`output/` unchanged since 2026-07-22 (see DEFECT_BACKLOG.md's 2026-08-05 section for the full
re-verification). QM-047 (shallow-clone git history) is this run's tooling finding; IMP-056 below is
its one new proposal, deliberately not implemented — it is a render-behaviour change with no GPU
available to verify it here, same constraint every prior audit in this ledger has respected for
comparable proposals (P1, P2, P5, P7, P8).

### IMP-056 Reference-sheet IC-LoRA as the group-shot identity fix   [proposed, NOT implemented]
- Defect: QM-001, specifically its group/wide-shot shape. This audit added a third independent
  evidence source for that shape — 17 human-reviewed takes across three early episodes (2026-08-05
  DEFECT_BACKLOG entry) carrying the repeated note "too many similar moving kids in a shot", with
  zuris-watering-blooms-day at 9/16 shots (56%) — the worst single-episode cast-integrity
  concentration measured anywhere in this checkout, vision-LLM or human.
- Why the existing mitigations don't cover this shape: `force_i2v`/klein (IMP-006/012/014) is
  single-subject BY DESIGN — `_reference_for_single_subject` returns `None` whenever more than one
  child is expected in frame, so it cannot anchor identity on a group shot at all. IMP-013's
  cross-negation (negate absent castmates' hair/wardrobe) only fires on subset shots, not full-cast
  ensembles. Nothing in the current pipeline anchors a shot with 2+ children pixel-for-pixel.
- Proposed change (research this audit, SOURCES.md 2026-08-05): swap the group-shot keyframe path
  onto [Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients)
  — a purpose-trained reference-sheet IC-LoRA that conditions on ONE composite image with a clean
  panel per character (already logged 2026-08-01; re-checked this audit, still the only mechanism in
  SOURCES that pins multiple identities from a single reference rather than one-subject-at-a-time).
  Concretely: build a 3-panel cast reference sheet (one panel per bible child, matching the model
  card's documented layout) once per episode from the existing per-character canonical references
  (`output/_cast_refs/<id>/canonical.png`, already generated by IMP-014's bootstrap), and route any
  shot with `len(characters) >= 2` through it instead of plain text conditioning.
- Hypothesis: a trained multi-character reference-sheet conditioner should pin group-shot identity at
  least as well as klein pins single-subject shots (IMP-014's validated result), closing the one gap
  none of QM-001's other mitigations reach.
- Why NOT implemented this run: it is a new render workflow (a reference-sheet compositor + a new
  ComfyUI graph, following the pattern of `workflows/flux2_klein_ref.json`), needs a GPU to build and
  A/B against plain-text group shots on the same seeds, and per the hard rules this ledger has
  followed all along, a render-behaviour change does not ship without that verification. Filed as the
  smallest concrete next step for whoever has GPU access, not as code.
- Metric to verify by, once built: same-seed A/B (`--restyle` pattern from IMP-038) comparing
  group-shot cast-integrity reject rate under the reference-sheet path vs. plain text conditioning,
  on the three episodes that already show the defect (zuris-watering-blooms-day,
  zuri-kofi-nala-feel-the-breeze, splish-splash-fun-with-friends) plus any newly-rendered episode.
