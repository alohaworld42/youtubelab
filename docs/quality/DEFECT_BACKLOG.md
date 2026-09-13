# ZubiBop defect backlog

Maintained by the `quality-manager` agent. One entry per defect *class* (not per bad
frame). Statuses: **open** (no mitigation), **mitigated** (a gate catches it, upstream
cause remains), **monitoring** (fix shipped, metric not yet confirmed), **resolved**
(metric confirmed over ≥3 episodes). Seeded 2026-07-21 from `review_log.json` history,
the scene-plausibility rubric's measured-failure notes, and git history.

### QM-001 Invented/substituted light-skinned child   [S2] [monitoring]
- Evidence: 2026-07-21 full sweep — ≥7 cast-integrity rejects across 103 vision-reviewed
  takes (6.8%; a LOWER bound — the regex misses "matches no cast member", "cast bleed",
  "crowd shot: five-plus children" phrasings). Verbatim reasons confirm the pattern and its
  two shapes: (a) hero-closeup single-subject substitution — "the middle child is a white
  ginger-haired boy — Kofi…", "center child is a non-cast white ginger-haired boy again (same
  failure as s08)", "non-cast white ginger-haired boy … becoming the hero closeup in the final
  beat"; (b) wide/group cast bleed — "crowd shot: five-plus children, none matching the
  three-kid cast specs", "characters match no cast member (teal outfit girl…)". Shape (a) is
  exactly the single-subject case force_i2v is meant to anchor.
- Stage: cast/identity (render).
- Root cause (best current theory): LTX drifts to training-set priors where per-character
  detail is few pixels (wide shots) or where the director hands a plural action to a
  single-subject slot; text conditioning alone doesn't pin identity.
- Fix / mitigation: scene-plausibility hard rejects (head-count + field-by-field identity);
  cast_text enrichment; Z-Image reference anchoring (a98e4a1, IMP-005) — but see IMP-005: the
  anchor mechanism **never actually executed on any episode** (force_i2v fired on 0 takes; the
  5 force_i2v=true emissions were all multi-child "duplicate characters" shots where
  `_reference_for_single_subject` returns None by design). IMP-006 (this audit) wires the
  reviewer to emit force_i2v on the single-subject substitution pattern so the anchor can
  finally fire, and logs `i2v_anchored` telemetry to prove it did.
- Metric: (1) cast-integrity reject share among *vision-reviewed* takes (baseline 6.8%);
  (2) count of takes with `i2v_anchored=true` (baseline 0 — must go >0 before the anchor can
  be credited with anything); (3) among anchored shots, did the next verdict flip to accept.

- 2026-07-24 storymode render (staging="story", keyframe-precedence + prop gate live,
  full vision review of all 16 takes by the session reviewer): **13/16 takes clean on
  first render — zero invented children in 13, zero clone-collapse anywhere, prop
  present in every prop shot, 0 strobe rejects (was 100% of solo takes)**. The residual
  class is ANIMATION-TIME morphing on 3 takes: s09/s14 solo Kofi morphs into a
  puff-haired Zuri-hybrid mid-take (s14 additionally grew a green-shirt non-cast child
  at frame edge), s13's circle dance smears identities during fast motion. All three
  vision-rejected + re-rendered (seed bump). New sub-pattern for the next audit:
  fast/rotational group motion and mid-take costume-hair blending are now the LAST
  remaining identity vector — keyframes are clean, frame-0 conditioning holds, drift
  begins mid-take. Candidate upstream levers: shorter high-motion takes, motion-damped
  action phrasing for group dance beats, i2v strength/schedule experiments.

### QM-002 Hand-vs-face skin desaturation   [S2] [mitigated]
- Evidence: hands gripping props render grayer/paler than the same character's face;
  called out as a recurring easy-to-miss defect in the scene-plausibility rubric (dim 2).
- Stage: render-prompt / sampler.
- Root cause (best current theory): unknown — possibly LoRA/skin-tone interaction at small
  spatial scales. Needs research (LTX prompt-adherence guidance, negative prompts).
- Fix / mitigation: caught at the shot gate only. No upstream fix yet.
- Metric: count of dim-2 skin-tone reasons in review logs per episode.

### QM-003 Wardrobe drift (colour and silhouette)   [S2] [mitigated]
- Evidence: 100% of measured wardrobe failures were closeup/medium shots; both wrong
  garment colour and wrong garment type observed.
- Stage: cast/identity (render).
- Fix / mitigation: rubric dim 4 shot-type-aware policy (hard at closeup/medium,
  silhouette always hard); cast bible pins top/bottom per character. Upstream: open —
  candidate is IC-LoRA-style identity conditioning (see SOURCES).
- Metric: dim-4 reject share per episode.
- 2026-07-23 evidence (storyfix draft render, claymation, heuristic-reviewed so the
  wardrobe gate did NOT run): keyframe contact sheet of 16 shots shows the **group/"all"
  shots render the three cast members correctly and distinctly** (Zuri yellow-scrunchie
  puffs + denim dungarees, Kofi boy blue tee + grey shorts, Nala braids + yellow pinafore),
  but **single-child closeups drift toward an averaged template** — several shots labelled
  Kofi or Zuri render a generic "child in blue dungarees with yellow-scrunchie afro puffs",
  bleeding the two kids' wardrobe/hair together (Kofi should not have puffs; Zuri's top is
  yellow, not the blue dungarees). No invented non-cast child and no in-frame duplicate
  (QM-001's worse shape absent). Consistent with the theory that solo-slot text conditioning
  under-pins per-character identity where the group frame's mutual contrast does the work —
  the exact case force_i2v single-subject anchoring (IMP-005/006) targets, still unexercised
  because this draft ran heuristic-only (QM-010).

- 2026-07-21 pattern-mining, recovered from the closed `qm/audit-20260721` branch (PR #2, never
  merged; its code half — the `force_i2v` rubric wiring and `i2v_anchored` telemetry — reached
  main by another route in `26fb0b9`, but this measurement did not). Reject rate by `shot_type`,
  counting **any** reject before the final accept, so this is all-cause and NOT wardrobe-only:
  **wide 28.6% (8/28), medium 25.6% (22/86), closeup 17.1% (12/70)**.
  Note the tension with this entry's own first Evidence line: all-cause rejects are worst on WIDE
  shots, while the *wardrobe* failures measured here were 100% closeup/medium and the 2026-07-23
  contact sheet shows group shots rendering the cast correctly while single-child closeups average
  together. Both can hold — wide shots fail most often overall (QM-001's group/identity pattern),
  wardrobe-specific drift concentrates in closeups — but any future claim that one shot type is
  'the problem' has to say which metric it means.
- 2026-07-21 research note (same branch): the cast bible already pins concrete garment nouns plus
  colour ("yellow pinafore dress over a white t-shirt", not "yellow dress"), which matches current
  best practice, so no further prompt-text change was proposed. The open upstream candidate stays
  IC-LoRA-style identity conditioning, with the same single-identity ceiling noted under QM-001.

### QM-004 "Scary marching kids" lockstep staging   [S2] [monitoring]
- Evidence: feel-the-breeze episode s00_a0.mp4 / s03_a0.mp4 — all three toddlers in an
  even front-on line walking at camera in step; reported by the channel owner.
- Stage: director (shot phrasing) + render.
- Fix / mitigation: rubric dim 9 hard reject; director prompt reworked to depict the
  song's story rather than generic group choreography (commit 224ac9e, IMP-001).
- Metric: dim-9 rejects per episode since 224ac9e (expect ~0); spot-check wides in finals.

### QM-005 Frightening/uncanny faces   [S1 if it reaches a final] [mitigated]
- Evidence: dead/glassy eyes, unnatural smiles, doll-like faces in otherwise-clean takes;
  2B model era was worse ("creepy/incoherent", LTX2_5070_NEXT_STEPS.md) but the class
  persists at low rate.
- Stage: render-prompt / sampler.
- Fix / mitigation: rubric dim 8 hard reject at the shot gate. Upstream open: strengthen
  negative prompt terms in the render-style prompt skeletons; research before changing.
- Metric: dim-8 reject share; zero tolerance in finals (any hit = S1).
- 2026-07-21 audit: 0 scary/uncanny faces in the 7 inspected finals (12-frame contact sheet
  each). Caveat — see QM-010: most of those finals were heuristic-reviewed, and the blur/
  black/static heuristic CANNOT see dim-8, so "0 in finals" is only as strong as a 12-frame
  human spot-check, not a gate guarantee.

### QM-006 Within-verse continuity mismatches   [S3] [open]
- Evidence: rubric dim 10 exists because location/lighting/set drift between shots of the
  same verse recurs; usually a shot-list fault, so re-seeding doesn't fix it.
- Stage: shotlist (director).
- Fix / mitigation: soft-flag only today. Proposal candidate: director-side per-verse
  continuity block (one locked setting/lighting description reused by all shots of a verse).
- Metric: dim-10 flag count per episode.

### QM-007 Blur heuristic false-rejects on toon style   [—] [resolved]
- Evidence: low Laplacian variance is *normal* for the flat toon look; the heuristic
  blurry check was rejecting fine takes.
- Fix: check demoted to advisory-only in review.py. Kept here as the calibration example:
  loosening a gate requires exactly this kind of false-positive evidence.

### QM-008 OpenCV-on-GPU crashes during QC   [—] [resolved]
- Evidence: whole episodes died when cv2 touched the GPU mid-render (commit 6310ecd).
- Fix: QC pinned to CPU. Watch for regressions if OpenCV is upgraded.

### QM-009 Unwatched dimensions (coverage gap)   [S3] [open]
- Evidence: no gate currently judges cross-shot grade consistency, word-level lyric-visual
  sync, episode pacing/arc, mix quality beyond cut_qc's spectral checks, or IP-lookalike
  distance. These are exactly the quality-manager's final-cut rubric — coverage exists only
  when an audit runs.
- Fix / mitigation: run /quality-manager after each finalized episode until any of these
  proves frequent enough to deserve an automated check.
- Metric: findings per audit in these dimensions.
- 2026-07-21 audit result: grade/look consistency = good (coherent within-episode; twinkle's
  night/lantern palette is intentional); audio loudness = good (all 7 finals mean −16.0 to
  −17.0 dB, 1 dB spread — consistent). Sync/pacing/IP findings spun out to QM-011/QM-014 below.

### QM-010 Vision gate goes dark under unattended runs   [S2] [open]
- Evidence: 2026-07-21 sweep — of 255 takes, only 40.4% were vision-reviewed; 59.6% carry
  "unattended mode: external review skipped" or "external review timed out — heuristic verdict
  used" (logs: "unattended" ×1290, "external review timed out" ×17). The collapse is temporal:
  the early episodes (bennys, bubbles, rainbow-sing, rainbow-color-fun) are 100% vision-
  reviewed; from splish-splash onward episodes flip to mostly/entirely heuristic — feel-breeze
  3/19 vision, twinkle 2/17, hickory 2/18, baa-baa ×3 ≈3/19, **mulberry (shipped) 0/16**,
  mulberry (finalize re-render) 0/16. So the most recent shipped/finalized episodes passed on
  the blur/black/static heuristic ALONE — the only gate that sees identity (QM-001), wardrobe
  (QM-003), scary faces (QM-005/dim-8), lockstep staging (QM-004/dim-9), wholesomeness and IP
  never ran on them.
- Stage: edit/compliance (review orchestration) — architectural: the vision reviewer is a live
  Claude scene-plausibility agent watching review_dir; headless/unattended runs have no agent,
  so requests time out to the heuristic fallback. `--finalize` did NOT re-enable it (the 115254
  finalize re-render is also 0/16 vision).
- Root cause (best current theory): unattended generation is the default operating mode; there
  is no gate preventing an all-heuristic cut from being promoted to Final/.
- Fix / mitigation: PROPOSED (owner call, see report) — either (a) a cut/promote guard that
  refuses (or loudly flags) promotion to Final/ when vision coverage < threshold, or (b) an
  automated contact-sheet→vision-LLM pass at the cut stage that runs even unattended. Additive
  telemetry (vision_coverage per episode) is the safe first step. NOT implemented this run
  (different defect class from IMP-006; kept unbatched for attribution).
- Metric: % of accepted takes per shipped episode that were vision-reviewed (baseline: recent
  finals ≈ 0–15%). Target: 100% vision coverage on anything promoted to Final/.
- 2026-07-21: promote gate shipped (min_vision_coverage, default 0.8) — see IMPROVEMENT_LOG IMP-008.

### QM-011 CoComelon-adjacent visual identity (intro + house style)   [S1] [open]
- Evidence: 2026-07-21 pixel inspection of the intro asset + all 7 finals. The intro badge
  (`assets/intro/zubibop_intro.mp4`) is a rounded smiling character face in a roundel with a
  rainbow, a sun-with-a-face, musical notes, rolling green hills and a multicolour bubble
  wordmark — the CoComelon logo formula. The episode house style (default `render_style =
  pixar_toon`) renders chibi big-glossy-eyed toddlers in primary-colour overalls in pastel
  playroom/garden/**bathroom** sets; the mulberry tooth-brushing set (white pedestal sink,
  round bead-framed mirror, pastel tiles) is CoComelon's single most iconic scene type. This
  reads as *adjacent*, which the rubric names an S1 ship-blocker.
- Stage: compliance (branding + render-style default).
- Root cause: the default look was chosen to match the dominant glossy-3D-CGI preschool
  aesthetic — which IS the CoComelon/Moonbug house style. Per SOURCES (Moonbug v. Babybus,
  affirmed 9th Cir.), Moonbug litigates *overall look* = character design + settings + trade
  dress, and wins; the cherubic-3D-toddler design is held copyrightable.
- Fix / mitigation: PROPOSED (owner call — do NOT unilaterally reskin the channel). The repo
  already ships the intended mitigation: `render_style` claymation & felt were added (17a1e73)
  explicitly as "the strongest visual differentiator from the many glossy-3D-CGI Cocomelon-
  alike channels." Recommend (1) switch the default render_style off pixar_toon to claymation
  or felt; (2) redesign the intro badge away from the roundel+rainbow+sun-face+notes formula;
  (3) legal review before scaling uploads. NB the orphaned cloud branch tried to DELETE
  claymation/felt — that would remove this very mitigation (see IMPROVEMENT_LOG incident).
- Metric: default render_style ≠ pixar_toon on shipped episodes; a documented intro redesign;
  an owner/legal sign-off recorded here.

### QM-012 Orphan Final sidecars / missing final mp4s   [S2] [open]
- Evidence: `output/Final/` holds 7 playable mp4s but 3 base names exist there ONLY as
  191-byte `<base>.mp4.intro.json` sidecars with no video: brushing-smiles-v3,
  rainbow-friends-color-fun, zuris-watering-blooms-day. The mp4s exist NOWHERE in the tree
  (searched repo + D:/brainrot). brushing-smiles was cut-REJECTED (cut gate 0.5/0.55) and 79%
  of its takes were rejects, so it correctly never shipped — its 38 MB `.staging.mp4` is that
  rejected cut and the Final sidecar is an orphan (i.e. it is NOT a shipped episode, despite
  being listed as one). zuris-watering's cut was ACCEPTED (1.0) and promotion is a *move*, yet
  no Final mp4 remains → promoted then removed/relocated. rainbow's `.staging.mp4` is a
  suspicious 1.0 MB for a 59.8 s video (likely a failed/corrupt encode).
- Broken-encode evidence (ffprobe, 2026-07-21): brushing-smiles staging = 38 MB / 58.5 s /
  1754 frames (a real, full cut — legitimately cut-rejected). BUT rainbow-color-fun staging =
  1.0 MB / 60.0 s / **no decodable video stream**, and splish-splash staging = 1.1 MB / 60.0 s /
  **6 frames total**. Two of the three "rejects" are degenerate/corrupt encodes, not content
  rejects — the assemble/encode step emitted a technically-broken container that the cut gate
  then (correctly) refused. So rainbow-color-fun most likely never shipped a playable final
  despite its orphan Final sidecar; its "shipped after re-cut" status is doubtful.
- Stage: edit (assemble/encode) + promote + ops.
- Root cause (theory): intro.json sidecars are written independently of successful promotion,
  so a cut-rejected or externally-archived episode leaves an orphan sidecar; combined with
  uploads/cleanup this makes Final/ an unreliable ledger of what actually shipped.
- Fix / mitigation: PROPOSED — write the `.mp4.intro.json` sidecar only after the mp4 lands in
  Final/, and/or a `--verify-finals` check that every sidecar has a sibling mp4. Owner: confirm
  whether zuris/rainbow were uploaded-then-locally-cleaned (respecting "never delete inventory").
- Metric: count of Final/ sidecars without a sibling mp4 (baseline 3 → target 0).

### QM-013 Render infra instability (ComfyUI timeout / GPU OOM)   [S3] [open]
- Evidence: the "here-we-go-round-the-mulberry-bush" song took 6 base attempts before shipping
  (224041, 231612, 232604, 103126 all aborted — two produced only s00, two produced nothing,
  none has a review_log or any QC reject → they died mid-render, not at a gate; then 110932
  shipped, 115254 re-rendered). studio.log is saturated with "RuntimeError: ComfyUI timed out"
  and "gpu_scenes: scene render failed (node error: OOM) — aborting GPU scene mode." This is
  wasted GPU compute, not a shipped-content quality defect.
- Stage: sampler/render infra (ops).
- Fix / mitigation: ops matter (ComfyUI timeout ceiling, VRAM headroom / batch or resolution
  caps for GPU scene mode). Out of QM's prompt/rubric remit — flagged for the owner.
- Metric: aborted-run count per shipped episode (mulberry = 4; target ≤1).
- 2026-07-21 partial fix (comfy.py ensure_up): a CUDA-wedged ComfyUI (system_stats
  returns HTTP 500, so `_alive()` is False, yet it keeps port 8188 bound) made the
  autostart path launch a second server that hit "port already in use" → exit 1 →
  every queued render failed (observed: jobs 18/19/20). ensure_up now detects a
  wedged-but-port-holding server (`_port_in_use()`) and clears it (`stop()` if we
  own it, else `_kill_port_owner()`) BEFORE launching, so a dead CUDA context
  self-heals on the next render instead of failing the batch. The root CUDA fault
  (driver/GPU state) is still upstream/open; this stops it from cascading.

### QM-014 Karaoke caption sync/duplication   [S3] [open]
- Evidence: baa-baa-black-sheep (181720) contact sheet shows the caption "name Clap your"
  held across two consecutive shots (frames 5–6), and captions render as fragmented
  word-groups ("name Clap your" / "hands and sing") that break the lyric line oddly; other
  finals show single-word karaoke fragments ("THE" "HIGH" "YOUR" "BEAT") that can lag the
  action. Cosmetic, not a comprehension-blocker.
- Stage: edit (caption/word-timing).
- Root cause (theory): word-level caption timing derived from forced-alignment can repeat or
  mis-chunk a line across a shot boundary.
- Fix / mitigation: PROPOSED — de-duplicate identical adjacent caption chunks; prefer phrase-
  level chunks aligned to beats. Low priority.
- Metric: adjacent-duplicate caption count per episode (baa-baa = 1; target 0).

### QM-015 Off-bible child name reaches the render prompt   [S2] [monitoring]
- Evidence: measured 2026-07-31 across every shotlist in `output/`. `generate._shot_prompt`
  renders `shot["action"]` verbatim, and five LLM-invented names survived into it
  (Amira ×15, Maya ×3, Kiara ×2, Jaden, Jaxson — 21 action/setting fields of 1130). The
  shipped Benny's-Brushing prompt for s00 reads: *"Exactly three children are on screen:
  Zuri, Kofi and Nala. … A wide shot: **Amira** and Kofi standing in front of bathroom sink
  with Zuri washing her face behind them."* Four names against a head count of three, and
  the fourth has no description attached — so the text encoder has to invent a child's
  appearance from the name alone.
- Stage: director (shot list) → render prompt.
- Root cause (confirmed, not theory): `_normalize_shots` scrubbed off-bible names out of a
  shot's `characters` list (`names_allowed` = `cast.names()`), but never out of the free-text
  `action`. Every other name-aware pass — `_cast_names_in`, `implied_child_count`,
  `_closeup_subject`, `sanitize_action` — matches against `names_allowed`, so an invented
  name was invisible to all of them and passed through untouched. This is the
  invented-extra-child defect of QM-001 being handed to the model in its own prompt, and it
  is the exact "prompts must not contradict themselves" failure the LTX guides in SOURCES.md
  warn about.
- Fix: `director.repair_offbible_names` rewrites the name to the bible child it stands in
  for (alias table, else stable hash — deterministic across shots and reruns), running before
  every name-aware pass so they all see a real name; `generate._rename_offbible_children`
  does the same for a shot list that never went through the director. Two invented names in
  one action can no longer collapse onto the same child, and an invented name can never
  become a child the sentence already mentions. Detection is calibrated against the corpus's
  actual capitalised vocabulary (`_NON_NAME_CAPS` + a participle rule); pinned by
  `tests/test_kidsong_offbible_names.py`.
- Metric: capitalised non-bible tokens in shipped `action`/`setting` fields (baseline 21 of
  1130; target 0), and the QM-001 cast-integrity reject share on the next vision-reviewed
  episode rendered after this ships.

### QM-016 Take-review gate was un-fillable in the browser   [S2] [resolved]
- Evidence: `templates/review_queue.html` carried `<meta http-equiv="refresh" content="8">`.
  Reviewing one take — read the shot spec, scan the contact sheet, tick reasons, type a note
  — takes longer than 8 seconds, so the page reloaded mid-review and discarded every ticked
  box, the typed note, the scroll position and any playing cut video. The queue was also
  reachable only from body text on two pages (never from the nav), and rendered every pending
  item into one document.
- Stage: review tooling (not the model) — but it is the direct enabler of QM-010: a gate
  nobody can answer is a gate that does not run.
- Fix: auto-refresh is now a JS timer that pauses while a form is dirty, an input is focused,
  media is playing or the tab is backgrounded, and restores the scroll position; the queue is
  paginated; a pending-take counter sits in the nav on every page; ticking a reject reason
  now really does mean reject (it used to ship the take with the reject reasons attached as
  invisible noise); a score outside 0–1 is clamped; an already-answered gate is never
  silently overwritten. Pinned by `tests/test_review_submit.py`.
- Metric: real (non-auto-accepted) vision coverage per episode — the QM-010 metric this
  unblocks.

### QM-017 A finished render thrown away by its own bookkeeping   [S1] [resolved]
- Evidence: `scheduler._run_pipeline` ran its post-success bookkeeping —
  `set_idea_status`, `touch_asset`, `assets.prune_cache` — INSIDE the render's own
  `try`, between `set_job_status(ready)` and the success log line. Anything raising in
  there was caught by the render's `except`, which flipped the finished job back to
  `queued` (or `failed` at max attempts) and re-rendered it. A kidsong render is ~1 hour
  of GPU, discarded because a cache file could not be deleted.
- Stage: scheduler (ops), not the model.
- Fix: bookkeeping moved out of the try, each step individually guarded and logged as
  "the video is fine, this is bookkeeping only". Pinned by
  `tests/test_scheduler_bookkeeping.py`, which also pins that a REAL render failure still
  requeues.
- Metric: aborted/re-rendered-after-success count per batch (QM-013's metric; this removes
  one of its causes).

### QM-018 Jobs that stop moving with nothing on screen   [S3] [resolved]
- Evidence: a job in `approved` (rendered, human-approved, waiting for the uploader
  thread) appeared in NO view — not the review queue, not the failed list, not any
  dashboard counter. The uploader skipped it every tick with `continue` and no record when
  its channel was `needs_reauth`, so a video blocked on an expired YouTube token simply
  vanished from the operator's point of view. The daily-cap path was the same shape,
  except it also logged the identical warning once per 15s tick (~5700 lines a day).
- Stage: studio UI / scheduler.
- Fix: a "Wartet auf Upload" tile (`approved` + `uploading`) on the dashboard and in
  `/api/status`; the re-auth block is written once to the job's event trail with what to do
  about it; the cap warning is logged once per day.
- Metric: none needed — the state is now visible by construction.

### QM-019 Kids uploads declared not-for-kids   [S1] [resolved]
- Evidence: `scheduler._process_uploads_locked` passed
  `made_for_kids=bool(channel.get("made_for_kids"))` — the channel checkbox alone. A
  kidsong/kids job queued against a channel whose box was unchecked (via `/quick` with a
  channel picked, or a channel someone edited) uploaded a toddler sing-along with
  `selfDeclaredMadeForKids: false`. The kidsong CLI path
  (`pipeline/kidsong/generate.py`) already forces True; the studio's own uploader — the
  one the scheduler actually uses — was the single place that did not, against this repo's
  own stated rule ("Kidsong uploads: always declare made_for_kids=True", AGENTS.md).
- Stage: upload (studio scheduler).
- Fix: `scheduler._made_for_kids(channel, job)` — the channel flag is an opt-IN that can
  add the declaration, never an opt-out that can remove it from a kids video type. Pinned
  by `tests/test_scheduler_bookkeeping.py`.
- Metric: none — it is now structurally impossible for a kids video type to upload
  undeclared.

### QM-020 Product surfaces pointed at the brand we must not resemble   [S1] [resolved]
- Evidence: `prompts/kids.txt` opened with "in the spirit of CoComelon / nursery-rhyme
  channels" — an instruction sent to the LLM on every kids script. The Quick Generate tile
  blurb read "CoComelon-style sing-along" and the channel editor labelled the kids style
  "CoComelon-artig". QM-011 and the quality-manager rubric treat look-alike drift toward
  CoComelon as ship-blocking, and `SOURCES.md` records Moonbug winning $23.4M on an
  overall-look theory covering "characters, settings, song titles, lyrics and images" — so
  the pipeline was being told to aim at the exact target its own gates reject.
- Stage: prompt + UI copy.
- Fix: the kids prompt now demands originality explicitly and cites why; the tile blurb and
  the style label describe our own look. `config.example.json`'s kids-style comment reworded
  the same way. Pinned by `tests/test_studio_pages.py` (no product surface may name the
  brand; the prompt's instruction block must say "original only").
- Metric: QM-011's IP-lookalike finding rate on the next audit.

### QM-021 A schedule the worker silently ignores   [S3] [resolved]
- Evidence: `scheduler.due_slots` skips any slot it cannot parse. The channel editor
  accepted and saved anything, so "9am" or "18.00" produced a channel that looked scheduled,
  reported its schedule back in the form, and never produced a video — with no error
  anywhere.
- Stage: studio UI.
- Fix: `channels.normalize_schedule` accepts what people actually type ("9:00", "09.00"),
  normalises and de-duplicates it, and refuses the save with a message naming the entries it
  could not parse. A test asserts the two halves agree: every slot the form accepts must be
  one `due_slots` recognises.

### QM-022 A browser field could write arbitrary settings into .env   [S2] [resolved]
- Evidence: `setup.write_env` wrote `KEY=VALUE` lines with no validation of either side.
  `.env` is one setting per line, so a value containing a newline does not store a
  multi-line value — it writes further lines that `read_env` and
  `pipeline.config.load_dotenv` then parse as more settings. Every value reaching it comes
  from a browser field on the setup page, and the views only `.strip()`, which catches a
  trailing newline and nothing in the middle. Pasting
  `gsk_real⏎STUDIO_PASSWORD=hunter2` into the API-key box set the studio's login password.
- Stage: setup UI / secret storage.
- Fix: keys must match `[A-Za-z_][A-Za-z0-9_]*`; values are collapsed to a single line
  (`\r`, `\n`, `\r\n`, U+2028/2029, NUL) before writing; `.env` is chmod 0600 on POSIX,
  since it holds every API key and the studio password. Pinned by `tests/test_setup.py`.
- Metric: none — the injection is structurally impossible now.

### QM-023 A partial stock download cached as a finished clip   [S2] [resolved]
- Evidence: `assets._download` streamed straight into the destination path. A dropped
  connection, a full disk or a killed process therefore left a TRUNCATED mp4 at the final
  path — and every caller guards with `if not os.path.exists(dest)`, so that stump was
  reused as the background clip on every subsequent render, with `record_asset` storing
  its partial size as the real one. A corrupt background that survives restarts is much
  worse than a failed download the next run retries.
- Stage: asset sourcing (studio).
- Fix: write to a `.part` sibling and `os.replace` into place only after the stream
  completes; remove the part file on failure. Caught with `BaseException`, not
  `Exception` — a KeyboardInterrupt/SIGTERM mid-download is exactly the case that
  produced the stump. Same atomic-publication discipline as `pipeline/gpu_lock.py`.
  Pinned by `tests/test_assets_fallback.py` (three cases, all failing on the old code).
- Metric: none — a partial file can no longer reach the destination path.
- 2026-07-31 follow-up: the same pattern was found in FOUR more writers and fixed with a
  shared primitive (`pipeline/atomicio.py`, IMP-026). The worst was
  `kidsong/comfy.py::_retrieve`, which writes the RENDERED TAKE — a truncated mp4 there
  goes straight to the contact sheet, the vision reviewer and the editor, none of which
  can tell it from a complete one. `studio/setup.py` rewrote the whole of `config.json`
  in place to change one string, so a crash mid-write left the studio unable to start.
  `studio/comfy.py`, `studio/svm.py` and `studio/higgsfield.py` had it too; higgsfield's
  additionally swallowed the exception and returned None while leaving the stump behind.

### QM-024 Caption words piled onto one instant at a song's edges   [S3] [resolved]
- Evidence: `alignment._interpolate` extrapolated the unanchored run before the first
  anchor by walking backward one `DEFAULT_WORD_SECONDS` at a time and clamping at 0. Every
  word that did not fit landed on the SAME instant: twelve lyric words with the first
  anchor at 0.5s produced seven timed `0.000–0.080` — one 2-frame flash of overlapping
  captions instead of seven words. The tail had the mirror image whenever `audio_end` sat
  at or before the last anchor's end (the whole tail got the same zero-length slot,
  floored to identical 0.08s boxes). Whisper regularly drops the quiet opening of a song
  over a music bed, so the head case is the common one.
- Stage: edit (caption timing).
- Root cause: clamping a fixed-rate walk instead of fitting the run to the room available.
- Fix: lay the unanchored run out at the nominal rate when it fits, otherwise compress it
  evenly into the room there is; a tail with no room left falls back to the nominal rate
  rather than honouring an impossible bound, since the editor clips to the real duration
  anyway. Pinned by `tests/test_alignment.py` (three cases failing on the old code).
- Metric: distinct start times per aligned word (was: up to 7 sharing one); related to
  QM-014's caption-quality class, though that entry's duplication has a different cause.

### QM-025 An ffmpeg reader leaked per scene, per render   [S2] [resolved]
- Evidence: `assemble.build_video`'s teardown closed `(final, audio, bg, *avatars,
  *captions)` with a comment explaining that leaked handles "pile up across repeated
  generations in the long-running Flask server" — but it released almost nothing.
  MoviePy's `CompositeVideoClip.close()` deliberately leaves member clips alone ("it
  remains the job of whoever created it") and the base `Clip.close()` is a documented
  no-op, so closing a composite/concatenation never reaches the `VideoFileClip`s
  underneath it. `_scene_background` opens one per scene, each holding an ffmpeg
  subprocess and a pipe; a 16-scene render leaked 16, once per video, for the life of the
  server process. `CompositeAudioClip` has the same non-recursive close, so the voice and
  music readers leaked too.
- Stage: assemble (ops).
- Fix: the opened readers are carried out on the returned clip (`source_clips`, the same
  mechanism as `whip_times`) and closed in the teardown, which now also covers the motion,
  flash and overlay clips it previously skipped. Pinned by `tests/test_polish.py` — on the
  old code the test reports three of four readers leaked.
- Metric: none directly measurable from the artifacts; watch for the studio's memory/handle
  growth over a long batch, which QM-013's aborted runs may partly reflect.

### QM-026 Lowering a documented knob deadlocked the cut gate   [S2] [resolved]
- Evidence: `cut_qc._check_cuts` resolved its cut-duration CEILING from
  `kidsong.review.max_cut_seconds` — with a docstring explaining that the director's bound
  and the gate's bound have to move together — while the FLOOR stayed a hardcoded
  `_MIN_CUT = 1.4`. So the coupling only held in one direction. Lowering
  `kidsong.director.min_shot_seconds` (a documented knob: snappier cutting) made the
  director plan shots outside the window the gate accepts.
- **CORRECTION (2026-07-31, same day).** This entry originally claimed the mismatch made
  the episode "fail its own QC gate and never reach Final/". That is WRONG and was written
  without checking the stage in between. `edit.build_cut_list` merges any segment under
  its own floor before the gate ever sees it, so short slots never reach the gate as short
  cuts and no deadlock occurs. The floor fix below is still right — the two windows should
  be symmetric — but the failure it was justified by does not exist. Reading `edit.py`
  afterwards found the real defect, QM-027.
- Stage: cut QC gate.
- Fix: `_effective_min_cut(cfg)` reading `kidsong.review.min_cut_seconds`, symmetric with
  the ceiling and defaulting to `_MIN_CUT` so an unset key is byte-identical.
  `config.example.json` now documents the window as one coupled thing rather than
  explaining the coupling for only one end of it. Pinned by `tests/test_beat_grid.py`,
  including an invariant test asserting the director's planning bounds and the gate's
  accepted window agree.
- Metric: none — the asymmetry is gone by construction.

### QM-027 The "fewer, longer shots" knob never reached the picture   [S2] [resolved]
- Evidence: a cut-duration window exists in THREE places —
  `director._shot_bounds` (plans the slots, config-driven),
  `edit.build_cut_list` (merges/splits them into the actual cuts), and
  `cut_qc._check_cuts` (gates the result, config-driven). The middle one, the stage that
  actually decides how long a cut is, took no `cfg` at all and used fixed constants
  (1.6/4.5). So the documented workflow in `config.example.json` — "raise
  max_shot_seconds ... also raise kidsong.review.max_cut_seconds below to match" — could
  not work. Measured directly: four 8.0s shot slots came out as **eight 4.0s cuts**. Both
  knobs moved the plan and neither moved the picture.
- Stage: edit (cut list).
- Root cause: the window was made configurable at the two ends and not in the middle. Note
  this also means QM-026's stated failure mode was never real — see the correction on that
  entry; the merge floor here is why short slots never reached the gate.
- Fix: `edit._cut_window(cfg)` resolves the builder's window from the same
  `kidsong.review.min/max_cut_seconds`, inset by the margin the two windows have always
  kept (`_GATE_MARGIN_MIN/MAX`) so an emitted cut never lands exactly on the gate boundary.
  Absent config is byte-identical. `cfg` threaded through both `generate.py` call sites.
  Pinned by `tests/test_beat_grid.py`, including an end-to-end invariant: what the builder
  emits must pass the gate, for the default config and for a moved one.
- Metric: with both knobs raised, max cut duration in the shipped cut list should equal the
  configured ceiling (was: pinned at 4.5s regardless).

### QM-028 One late verse match collapsed every verse after it   [S2] [resolved]
- Evidence: `sing.verse_times_from_words` anchors each verse by fuzzy-matching its first
  line in the sung transcript, then enforces monotonic boundaries with a +1ms nudge. The
  matcher takes the BEST overlap in the remaining stream, not the earliest good one, so a
  repeated chorus line scoring higher in the outro than in its own verse anchors that verse
  far too late — and every later verse whose proportional fallback lands before that point
  is squeezed to a 1ms window. Reproduced on a 60s song: windows of
  **48.000s / 0.001s / 0.001s / 11.998s**. The director then tiles no shots into the
  collapsed verses and the cut gate rejects the result.
- Stage: sing (verse timing) → director → cut.
- Fix: after the fill, a layout containing a window shorter than `_MIN_VERSE_SECONDS` is
  treated as broken rather than slightly wrong, and the whole song falls back to the
  proportional split — the fallback that already exists for a verse that could not be
  matched at all. A healthy anchoring is untouched. Pinned by
  `tests/test_kidsong_verse_times.py` (6 of 11 cases fail on the old code).
- Metric: minimum verse-window duration per episode (was: could be 0.001s; now >= 1.0s or
  the proportional split).

### QM-029 The identity sentence is not grammatical English   [S2] [fixed]
- Evidence: measured 2026-07-31 by recomposing every shot in `output/` through the current
  `generate._shot_prompt` (`tools/prompt_length_report.py` runs the same walk). Two of the
  three cast bible characters produced a broken identity sentence in EVERY prompt they
  appeared in — 517 of 517 child-bearing shots across the whole corpus name at least one of
  them:
  > *"Nala, a 3-year-old girl **with petite and delicate**, a little shorter and slighter,
  > deep brown skin, neat cornrow braids, wearing …"*
  > *"Kofi, a 4-year-old boy **with noticeably tall and sturdy for a preschooler, with
  > slightly broader shoulders**, medium-deep brown skin, …"*
- Stage: cast bible (data) → render prompt.
- Root cause (confirmed): `cast._describe_dict` composes `"{name}, a {age} with {traits}"`
  where `traits = build, skin, hair`. `with` takes a NOUN phrase, and `skin`/`hair` are
  noun phrases — but `build` (added later) was written as an ADJECTIVE phrase for Kofi and
  Nala. The unit test that shipped with `build` used the correct noun-phrase form
  (`"a taller, sturdier build"`), so the contract was understood but never enforced against
  the shipped bible. Kofi's value also nested a second `with` clause inside the first, so
  the attribute list parses ambiguously.
- Why it matters beyond tidiness: this is the identity sentence — the sentence QM-001
  (invented/off-cast children) and QM-015 are about, and the one the LTX guides in
  `SOURCES.md` single out ("follows natural film-style prose far better than keyword
  lists"). The pipeline spends 42–113 words per prompt on it, more than any other component,
  and two thirds of the cast were describing themselves in broken syntax.
- Fix: the three `build` values are noun phrases (`"a petite, delicate build, a little
  shorter and slighter"`), with no nested `with`. No descriptor was dropped — the same
  attributes reach the encoder, in a sentence it can parse. `prompts/cast_bible.json` now
  documents the contract inline (`_build_note`), and two tests enforce it against the
  SHIPPED bible rather than a fixture.
- Metric: `tests/test_kidsong_cast.py::test_every_bible_build_is_a_noun_phrase` (a
  determiner opens every build, no nested `with`). Quality effect is unmeasured — it needs
  a GPU render; the QM-001 cast-integrity reject share on the next vision-reviewed episode
  is the number to watch.

### QM-030 Multi-child prompts cannot be brought under the word cap by trimming the tail   [S3] [monitoring]
- Evidence: `tools/prompt_length_report.py`, 517 child-bearing shots, current code and
  config. **23.0% over the ~200-word cap** — not the 12% quoted in `SOURCES.md` since the
  last audit; the figure drifted as gaze/subject/count clauses were added, and nothing
  re-measured it. Distribution: median 190, p90 262, max 273.
- The split is not gradual, it is categorical:
  | children on screen | shots | mean words | over cap |
  |---|---|---|---|
  | 1 | 408 | 186.9 | 2.5% |
  | 2 | 23 | 230.2 | 100% |
  | 3 | 86 | 263.5 | 100% |
- Root cause: component decomposition of the over-cap shots gives identity sentence 101.3
  words (mean), tail 49, action+setting 31.2, camera/look/glue 31.8, identity clause 15,
  count clamp 10.7, skeleton+trigger 12. The identity sentence is ~38 words per child and
  is the ONLY component that scales with head count. Every other component is fixed.
- Consequence for IMP-035: the `pixar_toon_concise` tail (30 words instead of 49) takes
  one-child shots from 2.5% over-cap to **0.0%** — a real win over 79% of the corpus — and
  leaves multi-child shots at 100%/95.7%. Trimming the tail cannot fix them; 19 words
  against a 63-word excess.
- Lever (a), BUILT AND MEASURED: the count clamp ("No other children appear anywhere in the
  frame, foreground or background.") and the identity clause ("Every child on screen is one
  of these Black toddlers — no other children appear.") assert the same constraint twice.
  The clause's unique content is the IDENTITY half; its trailing 5 words are a verbatim
  duplicate of the clamp, which states it more precisely. `pixar_toon_lean` in
  `config.example.json` drops exactly those 5 words and takes the concise tail — 24 words
  off every child prompt, **not one cast descriptor touched**. That is the LOSSLESS UPPER
  BOUND, and it is not enough:
  | arm | 1 child | 2 children | 3 children | overall over-cap |
  |---|---|---|---|---|
  | `pixar_toon` | 2.5% | 100% | 100% | 23.0% |
  | `pixar_toon_concise` | 0.0% | 95.7% | 100% | 20.9% |
  | `pixar_toon_lean` | 0.0% | 82.6% | 100% | 20.3% |
  A three-child prompt still averages 239.5 words. Closing the last ~40 means cutting the
  wardrobe block itself — roughly one child's entire description — which is the measured
  mitigation for QM-001/QM-015. **So: the cap is unreachable for three-child shots by any
  text trim that keeps the cast description intact.** That is a finding, not a to-do; the
  remaining choices are to accept the overrun on group shots, stage fewer children per shot,
  or lever (b).
- Lever (b), NOT ATTEMPTED: on the keyframe-first / reference-anchored path the first frame
  already carries the correct children (IMP-005/012/014), and `SOURCES.md` records reference
  images as the most reliable consistency fix with reused descriptors only second — so the
  wardrobe text may be redundant specifically on anchored shots. Unlike (a) this is a code
  change to `_identity_block`, it is conditional on how a shot was rendered, and it cuts
  cast conditioning rather than duplicated boilerplate. It needs frames before it needs an
  implementation.
- Metric: re-run `tools/prompt_length_report.py`; target is the multi-child over-cap share,
  currently 100%. On real frames, one command per arm: `--restyle <base> --style
  pixar_toon` / `--style pixar_toon_lean` (IMP-038) — same song, shot list, audio and seeds.
- Pinned by `tests/test_kidsong_prompt_tail.py`, including
  `test_no_lossless_trim_can_bring_a_three_child_prompt_under_the_cap`, which fails if that
  conclusion ever stops being true.

### QM-031 The studio password was void whenever Flask's TESTING flag was set   [S1] [fixed]
- Evidence: with `STUDIO_PASSWORD=hunter2` configured and `app.config["TESTING"]=True`,
  `GET /dashboard` returned **200**. Reproduced against the real `create_app()`:
  the gate was installed, `auth._password()` returned `'hunter2'` at request time,
  and every page, every JSON endpoint and `/artifact/<path>` (which serves every
  rendered frame and take) answered anonymously. A POST to
  `/kidsong-review/submit` from a logged-out client wrote a real accept verdict.
- Stage: studio / auth.
- Root cause (confirmed): `studio.auth.install_auth._gate` opened with
  `if current_app.config.get("TESTING"): return None`. The comment gave the
  reason — web-layer tests build the real app and the developer's `.env` may
  carry a password, so an always-on gate would 302 them — and asserted
  "production never does" set TESTING. That is a convention, not a guarantee,
  and this gate exists specifically so the studio can be exposed through a
  tunnel with stored YouTube OAuth tokens behind it.
- Second cost: it made the control **untestable**. Using the test client turned
  the gate off, which is why `tests/test_auth_gate.py` asserts against a stub
  Flask app with two fake routes rather than the real one — the bypass created
  the coverage gap that hid it.
- Fix: the bypass is gone. `install_auth` already returns early when no password
  is set, so a test that does not want the gate simply does not set one;
  `tests/conftest.py` clears `STUDIO_PASSWORD` for the suite, which also makes
  the six test files that build the real app independent of the developer's
  environment instead of quietly depending on it.
- Metric: `tests/features/test_locking_the_studio.py` enumerates the app's LIVE
  route table and asserts every GET redirects to /login — so a blueprint route
  added later is covered the day it is added. 6 of its 9 cases fail against the
  pre-fix code (verified by restoring the bypass).

<!-- The entries below came from the qm/audit-2026-07-22 branch, which allocated QM-015 onward independently of this branch. Both continued from main's QM-014, so the same IDs named different findings. The audit's were renumbered to follow (the smaller side: 3 commits vs 25), and its code cross-references updated to match. Content is unchanged. -->

### QM-032 Invented (non-bible) child names survive from script into the shot list   [S2] [mitigated]
- Evidence: 2026-07-25 sweep of all 32 shot lists in `output/`. Character names that resolve to no
  cast-bible entry appear on **27 shot slots across 4 episodes**: kiara ×9, jaden ×6, nia ×5,
  jaxson ×4, ava ×3 (bubbles-and-smiles, counting-friends-fun-day, plus alias-rescued maya ×13,
  luna ×8, amira ×6 in rainbow-friends-sing / rainbow-color-fun / bennys-brushing). It reaches the
  planning artefacts, not just the shot list: `20260720-092939-…-song.json` has
  `"scene": "Zuri waves her arms up in the air while Maya and Kofi clap."` and the shot list then
  carries `characters: ["Maya"]`.
- Stage: director / shotlist (with a compliance edge — the channel's cast is its brand).
- Root cause: the lyrics/director LLM invents a fourth child. `cast._resolve_with_fallback`
  substitutes a deterministic bible child at RENDER time and logs a warning, so the render is
  correct-ish and the fault is invisible. `cast.unresolved_names()` exists with a docstring saying
  it is "exposed so the script/shotlist QC gate can SURFACE an invented character name instead of
  letting the render silently paper over it" — but it **had no caller anywhere in the repo**. A
  guardrail that is never invoked is the IMP-005 failure mode repeating.
- Why it matters beyond tidiness: a name that is not in the bible cannot be identity-anchored
  (`refs.reference_for` returns None for an unknown id) and cannot be cross-negated (IMP-013
  derives the absent-castmate negatives from the bible), so an invented-name shot loses exactly the
  two mitigations built for QM-001/QM-003. It also risks a lyric-visual mismatch if the invented
  name is ever sung.
- Fix / mitigation: IMP-017 (this audit) wires `unresolved_names` into the shotlist gate's external
  request payload as an advisory `unresolved_character_names` key — additive, non-blocking, cannot
  change the programmatic verdict. Promoting it to a `_check_shotlist` reason (which WOULD be a
  hard reject, since `_verdict` sets `accept = not reasons`) is proposed but not implemented: no
  frequency data exists for the current director prompts, and a replan loop is the wrong thing to
  ship blind. Last observed occurrence is 2026-07-20, so the director may already have improved.
- Metric: count of shot slots per episode naming an unresolvable/alias-only character (baseline 27
  across 4 of 32 episodes; target 0). Read it off the new `unresolved_character_names` payload key.

### QM-033 Reference-anchored (i2v) takes strobe-reject ~14% vs 0% unanchored   [S2] [open]
- Evidence: 2026-07-25 sweep, restricted to the 4 episodes carrying `i2v_anchored` telemetry
  (118 takes): **strobe/glitch flagged on 5/35 (14.3%) of `i2v_anchored=true` takes and 0/80
  (0.0%) of `i2v_anchored=false` takes.** Channel-wide, strobe/glitch occurs in only 4 episodes,
  and **every one of the 5 strobe flags in the anchored era landed on an anchored take** — where
  anchored takes are 35/118 (30%) of that era, so ~0.3^5 ≈ 0.24% by chance. Confounded by episode
  (little-gardeners and alle-vögel are 100% anchored), so treat as a strong signal, not proof.
- The consequential case: sunflower-family-tree **s07 is the only reviewer-driven `force_i2v`
  escalation in the entire corpus.** s07_a0 was rejected "duplicate characters" with
  `force_i2v=true`; a1 and a2 then rendered `i2v_anchored=true` and **both were strobe-rejected**;
  the shot exhausted its budget and shipped its slot covered by the UNAPPROVED s07_a0
  (`shots.json: status=rejected, take=s07_a0.mp4`) because `review.require_approved` defaults off.
  So the first real exercise of the QM-001 identity anchor cost 2 renders, fixed nothing, and put
  a QC-rejected take in a promoted episode.
- Stage: sampler / render-prompt (i2v path).
- Root cause — two live hypotheses, deliberately not yet chosen between:
  (a) **Prompt redundancy.** `generate.py` builds `base_patches` once with
  `"PROMPT": {"text": _shot_prompt(render_shot, song, cfg)}` and the i2v branch copies it verbatim,
  swapping only `INPUT_IMAGE` and `SIGMAS`. So the anchored render sends the full t2v prompt —
  style skeleton, identity block, wardrobe, setting, camera move, action — re-describing everything
  the conditioning keyframe has already fixed. The vendor i2v guide (SOURCES, added 2026-07-25)
  names exactly this as the cause of *jittery motion*: "your source image already contains the
  visual information… direct what happens next", "simplify to a single, clear motion vector".
  (b) **Heuristic false positive.** i2v starts from a *still*; the still→motion transition
  concentrates change in the first sampled frame pair, which can trip `max_diff > 60` with nothing
  actually wrong — the QM-007 blurry story repeating on a different metric. The vendor/community
  material describes this transition as a known discontinuity. Our i2v sigma schedule
  (`SIGMAS_OFFICIAL_9STEP`) is already inside the vendor-recommended 8–12-step band, so the
  schedule is unlikely to be the cause.
- Fix / mitigation: IMP-016 (this audit) adds additive `metrics` telemetry to the heuristic verdict
  — `max_frame_diff`, `max_diff_at` (which sampled frame PAIR peaked), `frame_diffs`,
  `median_lap_var`, luma range. `max_diff_at == 0` on anchored takes ⇒ hypothesis (b) and the
  threshold gets recalibrated **with evidence, as the hard rules require**; a mid-clip peak ⇒
  hypothesis (a) and the fix is the motion-only i2v prompt (proposed, not implemented — it is a
  render-behaviour change that needs a same-seed A/B). **No threshold was moved this audit.**
- Metric: (1) strobe/glitch share of `i2v_anchored=true` takes (baseline 14.3%) vs
  `i2v_anchored=false` (baseline 0.0%); (2) distribution of `max_diff_at` on anchored strobe
  flags; (3) count of shots exhausting their attempt budget on an anchored retry (baseline 1).
- **2026-07-29 — the 14.3% baseline is about to become non-comparable, and a third hypothesis
  appears.** Commit `025ed80` (2026-07-24, i.e. *after* every episode in this checkout) made the i2v
  path follow `hires_pass`, so anchored takes now run a 2× refine that they did not run when the
  14.3% was measured. The same commit bisected that graph's envelope and set
  `kidsong.shot.max_frames` to **81** with the note "81 (3.4 s) verified, **89+ renders black** …
  the i2v refine tightens the envelope" (the t2v graph alone was good to 121). Consequences:
  (c) **new hypothesis** — on the *current* code a strobe/glitch flag on an anchored take can simply
  be the i2v-hires black-frame envelope being exceeded: one or more black tail frames produce
  exactly one enormous frame-pair delta. IMP-016's `max_diff_at` separates all three cleanly —
  peak at index 0 ⇒ (b) still→motion transition, mid-clip ⇒ (a) real prompt-redundancy jitter,
  peak at the LAST pair (plus `luma_min` ≈ 0) ⇒ (c) envelope overrun. `luma_min/max` is already in
  the telemetry, so no code change is needed to test this.
  **And a measurement caveat:** hypothesis (c) cannot explain the historical 5/35, because the i2v
  refine was not running on those renders. So the next audit must NOT compare a post-2026-07-24
  anchored strobe rate against the 14.3% baseline as if it were the same experiment — the renderer
  changed underneath it. Re-baseline on the first vision-reviewed episode rendered after `025ed80`.

### QM-034 Content rejects almost never recover on retry (retry-hint misrouting)   [S2] [mitigated]
- Evidence: 2026-07-25 retry-efficacy analysis over all 389 takes, grouped by what the rejected
  take was actually rejected FOR:
  | reject class | retried | flipped to accept | exhausted |
  |---|---|---|---|
  | black/blank + static (dead render) | 28 | 22 (**78.6%**) | 0 |
  | duplicate characters | 8 | 7 (**87.5%**) | 0 |
  | strobe/glitch | 7 | 4 (57.1%) | 1 |
  | reviewer prose (content: identity, setting, wardrobe, action) | 26 | 3 (**11.5%**) | 11 |
  By emitted hint: `seed_bump` alone flips **7/33 (21.2%)**; `seed_bump+simplify_action` flips
  22/28 (78.6%); `force_i2v` (±seed_bump) flips 7/8 (87.5%). **12 shots channel-wide ended with a
  rejected final attempt.**
- Stage: review rubric / retry policy (not the renderer).
- Root cause: the reviewer's hint rule was one line — "set seed_bump=true on any reject unless the
  flaw is clearly prompt-driven" — and in practice essentially every content reject carried
  `seed_bump` alone. A new seed re-rolls the sample; it does not change what the prompt asked for.
  The verbatim logs make this concrete: bennys-brushing s05 was rejected "wrong setting: kids
  brush teeth seated at a dining table with plates (should be bathroom/sink)" with `seed_bump`,
  reseeded twice, and came back "no bathroom set again — bare room". The reviewer itself had
  already written the diagnosis in a1's own reason line: "this shot's generic action text ('The
  kids brush their teeth') carries no setting anchor". Rubric dimension 10 has always said in so
  many words that re-seeding will not fix a plan fault; the hint rule did not encode it.
- Fix / mitigation: IMP-015 (this audit) — the scene-plausibility rubric now routes hints by fault
  type: mechanical → `seed_bump`; dimensions 5/6/9/10 (action, setting, staging, continuity) →
  `simplify_action`; single-subject identity substitution → `force_i2v`; multi-child identity →
  `seed_bump` plus naming every off-cast child. **No accept/reject threshold changed**; only which
  retry the pipeline attempts. Corroborated by the evaluation-guided-regeneration literature added
  to SOURCES: the productive loop revises the prompt on critique, it does not resample it.
- Metric: flip-to-accept rate for prose/content rejects (baseline **11.5%**, 3/26); share of
  content rejects emitting `seed_bump` alone (baseline ~100%); shots ending on a rejected final
  attempt (baseline 12/389 takes across 23 episodes).

### QM-035 Story-mode closing-group invariant silently no-ops on half the catalogue   [S3] [open]
- Evidence: `director.py` (~line 2659 after the 2026-07-25 rebase; was ~2644) enforces the episode's second bookend with
  `finale = [s for s in shots if s["verse"] == last_verse and not s.get("reuse_of")]` and then
  `if finale: closer = finale[-1]` — so when the final verse contains **no** unique shot, the
  invariant no-ops silently. Measured across all 32 shot lists: **18/32 episodes (12 of the 23
  with review dirs) have a final verse composed entirely of `reuse_of` shots** — baa-baa ×3,
  twinkle, hickory, mulberry ×5, rain-rain ×2, are-you-sleeping, skip-to-my-lou, colors-on-our-
  table, pat-a-cake, zuri-had-a-little-lamb. This is structural, not accidental: the final verse is
  usually a chorus repeat, and the director tiles chorus repeats entirely out of reuse shots
  pointing back at the first chorus (`s16→s08`, `s17→s09`, …).
- Stage: shotlist (director).
- Root cause: the invariant assumes the last verse owns at least one unique shot. Under the
  `max_unique_shots` budget (16) a repeated final chorus is 100% reuse, so the guarantee "the red
  thread resolves with everyone together" is exactly absent on the episodes that end on a chorus
  repeat — i.e. most of them. NB none of the 23 audited episodes ran story mode (it postdates this
  telemetry), so this is a static/structural finding about how the shipped invariant will behave,
  not a measured render defect. It should be confirmed on the first story-mode episode.
- Fix / mitigation: PROPOSED, not implemented (shot-planning behaviour, needs a render to verify):
  fall back to the last verse that HAS a unique shot — or force the last *rendered* (non-reuse)
  shot of the episode to be the closing group shot — instead of no-oping. Keep it a one-line
  fallback so the change stays attributable.
- Metric: fraction of episodes whose last non-reuse shot stages `characters == ["all"]` (baseline:
  unenforced on 12/23; target 100% under story mode).

---

## 2026-07-29 cloud audit — defects logged from git history (no new telemetry)

**Scope note that applies to QM-036…QM-040.** This checkout's `output/` still stops at
`20260722-135241` — byte-identical to what the 2026-07-25 audit above pattern-mined, and
`git log -1 --format=%ci -- output/` is still 2026-07-22. **No new episode telemetry synced, so
every number restated from the previous audit is UNCHANGED, not re-measured**, and no fresh
sweep was run. What *is* new is four channel-owner commits on `main` (2026-07-25, `ada0cbf`,
`c534c3c`, `f1d78b8`, `4d30322`) that each document a defect measured on LOCAL renders which do
not exist here. The entries below record those defect classes so they are tracked like any other,
with their own metrics — but the evidence is **the fix commits' own pre-fix measurements, taken by
the channel owner, not observations of mine.** No pixels, no audio, no contact sheets and no
`config.json` are in this checkout; the "inspect finals" phase was **skipped, not performed**, for
the fifth consecutive cloud audit.

### QM-036 Story-subject appearance drifts to a third-party character prior   [S2] [mitigated]
- Evidence (owner-measured, 2026-07-25 local renders, commits `ada0cbf` + `f1d78b8`): the baa-baa
  episode rendered "a black sheep in some keyframes and a white Shaun-lookalike in others". The
  commit names the mechanism outright — "the Shaun-the-Sheep prior (white fleece, black face)
  otherwise wins over a bare 'BLACK sheep'". Not verifiable in this checkout (no frames).
- Stage: render-prompt (subject description) + shotlist/director, with a **compliance edge**.
- Root cause (best current theory): the non-child story subject had no pinned appearance anywhere.
  Cast children are pinned by `cast_bible.json`; the *subject* of the song (sheep, lamb, star, bus)
  was described only by whatever noun phrase the lyrics/storyboard produced. A two-word prompt token
  ("black sheep") is weaker than the training-set prior attached to the most famous cartoon sheep,
  so the model resolves toward the recognisable character design.
- **Compliance note (escalation path to S1):** this is the QM-011 failure mode with a different
  target — a *model prior* pulling a render toward a recognisable third-party character. Shaun the
  Sheep is an Aardman-owned character design; the originality rubric makes "any character drifting
  toward a recognisable third-party design" a hard S1 finding. Filed S2 here because I have no
  frames and cannot judge how close the render actually read; **if a phenotype-level lookalike is
  confirmed in any final, it is an S1 ship-blocker, not a continuity defect.** The generalised risk
  is the same for every library song whose subject has a famous cartoon incarnation (sheep, bus,
  spider, duckling) — the fix below is what keeps ZubiBop's subjects author-picked instead of
  prior-picked, which is exactly the QM-011 "original, not adjacent" argument.
- Fix / mitigation: IMP-018 — `subject_description` per `pd_songs` entry (author-picked, explicit;
  baa-baa's now reads "whole fleece deep black"), stamped into shot and keyframe prompts.
- Metric for the next audit: (1) every `pd_songs` entry with a non-child story subject carries a
  non-null `subject_description` (static check, verifiable without media — **currently true, see
  IMP-018**); (2) subject-appearance drift rejects per episode on the first vision-reviewed episode
  rendered after `ada0cbf` (baseline: unmeasured here); (3) zero recognisable-third-party-character
  findings in any QM final inspection (standing S1 tolerance: zero).

### QM-037 ACE-Step re-plans the arrangement per render — scrambled sings shipped   [S2] [mitigated]
- Evidence (owner-measured 2026-07-24 on baa-baa and twinkle, commit `c534c3c`): identical inputs
  produced beat-phase resultants of **0.54 (steady, 99.38 bpm — sang clean)**, **0.235 (129 bpm)**
  and **0.068 (86 bpm)** — the latter two "audibly scrambled". The song is the spine of the whole
  episode: the beat grid derived from a scrambled sing drives verse timing, the cut list and every
  shot boundary, so a bad roll degrades edit rhythm channel-wide, not just the audio.
- Stage: audio (sing).
- Root cause (confirmed by the vendor docs, researched this audit — see SOURCES 2026-07-29): ACE-Step
  runs a 5 Hz "thinking" LM with `use_cot_metas=True` that plans `cot_bpm` / `cot_keyscale` /
  `cot_timesignature` **per render**, and the exposed `seed` pins only the DiT/diffusion noise —
  the upstream ACE-Step-1.5 inference docs document **no separate CoT seed**; LM randomness is
  governed by `lm_temperature` / `lm_top_k` / `lm_top_p`. So a scrambled arrangement is a dice roll,
  not a deterministic bug, and re-rolling is a legitimate response. Community write-ups describe the
  same behaviour as "gacha-style" output variance.
- Fix / mitigation: IMP-020 — coherence gate on fresh sings (`min_resultant` 0.25, `max_attempts` 3),
  re-sing with seed+n, ship the highest-scoring take, keep rejects as `.rejected-aN.wav` (inventory
  preserved, per the hard rules). Resume never re-sings, by design — re-singing would desync the
  beat grid every already-cut take was aligned to.
- Metric for the next audit: (1) distribution of shipped-sing `resultant` across new episodes —
  target: none below 0.25, mean at or above the 0.54 clean baseline; (2) count of `.rejected-aN.wav`
  files per episode (= how often the gate fires; if it is firing on most episodes the *retry* is
  masking a cheaper upstream fix, see the proposal in IMP-020); (3) count of episodes where all
  `max_attempts` failed and a sub-threshold sing shipped anyway (target 0 — this is the gate's
  documented escape valve and needs to stay rare).

### QM-038 Take reviewer ACCEPTED mid-take morphing because the plan asked for it   [S2] [resolved]
- Evidence (owner-measured, commit `f1d78b8`): s06 of the 2026-07-25 baa-baa render showed a sheep
  that **starts white and turns black within one shot**. The scene-plausibility reviewer *accepted*
  it — scoring it as action fidelity, because the beat's own action text said the subject "changes a
  little more, growing steadily". Only the downstream cut gate caught it, against the episode premise.
- Stage: director/storyboard (beat phrasing) — surfacing as a review-rubric failure.
- Root cause: a rubric-vs-plan interaction, and the more interesting defect of the two. Dimension 2
  hard-rejects "morphing between frames", but the reviewer weighs the take against *the shot's own
  action text*; when the plan explicitly requests an ongoing transformation, the two rules collide
  and action fidelity wins. **A gate cannot reject what its own instructions asked for.** This is the
  general lesson: a plan-stage phrasing bug can silently disable a rubric dimension, and no amount of
  reviewer strictness fixes it — the fix has to be upstream in the plan.
- Relationship to QM-001: this is the same family as QM-001's recorded residual sub-pattern
  ("ANIMATION-TIME morphing… drift begins mid-take", 2026-07-24 storymode note), but a distinct root
  cause. QM-001's morphing is *unrequested* identity drift under motion; this one was *requested* by
  the beat text. Fixing this does not close QM-001's residual — keep both open in the next audit's
  sweep and separate them by whether the shot's action text names a change.
- Fix / mitigation: IMP-021 — grow beats now describe the grown STATE ("looks a little fuller and
  happier than before"), never a transformation in progress. Marked **resolved** at the plan level:
  the offending phrasing is gone from `storyboard.py` and the class cannot recur from that template.
  It stays worth a sweep because any *new* beat template that describes a process re-opens it.
- Metric for the next audit: (1) grep every shot list for action text describing an ongoing change
  ("changes", "turning", "growing", "becoming", "transforms") — target 0 on episodes rendered after
  `f1d78b8`; (2) mid-take morph rejects that cite the shot's own action text as justification for an
  ACCEPT (target 0, read from verdict reasons).

### QM-039 Final cuts visibly played shots backward (ping-pong fill)   [S2] [resolved]
- Evidence (user report, 2026-07-25 baa-baa, commit `4d30322`): scenes played forward-then-backward
  in a shipped-quality cut. Trigger: `kidsong.shot.max_frames` was lowered to the i2v-hires
  black-frame envelope (81 f = 3.375 s) while the director's slot ceiling was still 4.5 s, so nearly
  every final-quality shot came up short of its cut window and hit `edit._extend_to_window`'s
  ping-pong fill — a path built for ~40%-short *draft* takes.
- Stage: edit (fill policy) + director (slot planning), with a **review blind spot**.
- Root cause: two config values with an implicit coupling and no guard between them (render cap vs
  planned slot length), plus a fallback that was correct for its original tier and wrong for the
  other. **Why no gate caught it is the durable finding: frame-sampled cut review is
  direction-blind.** Every review artefact in this pipeline — contact sheets, the 5-frame take
  review, the cut request — is a set of stills. A reversed clip and a forward clip sample to the
  same frames in a different order, so no still-based reviewer, human or model, can see it. Any
  future defect that lives purely in frame *ordering* (reversal, stutter, freeze, duplicated
  segment) is equally invisible. That is a structural gap, not an oversight — logged against QM-009.
- Fix / mitigation: IMP-022 — (a) `edit._extend_to_window` covers a shortfall within
  `_MAX_SLOW_STRETCH` (1.15×) by *slowing the take to the window* instead of reversing it, keeping
  ping-pong for draft-scale gaps only; (b) `director._shot_bounds` clamps `max_shot_seconds` to the
  renderable take length `(max_frames − fps//4) / fps`, so a final plan can no longer request a slot
  real footage cannot cover. Belt-and-braces, and the clamp is the one that makes the fill path
  unreachable on finals.
- Two residuals this audit will NOT fix (unbatched, need a render to judge):
  1. **Slow-stretch is not beat-neutral.** Slowing a take up to 1.15× moves the *internal* motion off
     the grid — a clap or a skip that landed on the beat now lands up to 15% late within its slot.
     The commit's own reasoning ("invisible for toddler motion") is about smoothness, not sync, and
     `cut_qc` measures cut *boundaries* against the beat grid, not in-shot action timing, so this too
     is unmeasured. It is very likely the right trade against visible reversal; it should still be
     confirmed by ear on the first episode that exercises it.
  2. **`_shot_bounds` returns `min(min_s, max_s), max_s`** — deliberate (it keeps the pair ordered
     when the clamp bites), but it means a low enough `max_frames` silently collapses
     `min_shot_seconds` onto the clamp and every shot becomes the same length. That directly
     contradicts the final-cut rubric's "shot lengths breathe / no metronomic same-length runs".
     Not a bug today (the default `max_frames` 241 never binds); a config trap worth a log line.
- Metric for the next audit: (1) count of shots per episode whose take is shorter than its cut
  window — split into slow-stretched vs ping-ponged (target: 0 ping-pong on any non-draft cut);
  (2) unique shot-length count per final (target: not 1 — the metronome check for residual 2);
  (3) any user/QM report of reversed motion (target 0).

### QM-040 Phantom heads grown by the i2v stage on clean keyframes   [S2] [mitigated]
- Evidence (owner-measured, twinkle-twinkle 2026-07-24, "multiple takes", commit `ada0cbf`): partial
  faces at frame edges, out-of-focus back-of-head silhouettes looming in the foreground
  (over-the-shoulder blobs), and disembodied/duplicate heads — **appearing during the i2v video stage
  on keyframes that were clean**. Not verifiable here (no frames).
- Stage: render (i2v motion stage).
- Root cause (best current theory): the same class of failure as QM-001's invented extra child, but
  produced downstream of the keyframe rather than in it, and in a form the old rubric let through —
  a blurred edge-of-frame head reads as "scenery", not as an extra character, so a reviewer counting
  *children* would not count it. The i2v stage is inventing crowd/composition detail during motion.
- Fix / mitigation: IMP-019 — the scene-plausibility anatomy dimension (2) now names phantom heads an
  explicit **hard reject**, "even when blurred, cropped or 'just scenery'". Strictly stricter, which
  the hard rules permit freely; no threshold loosened, no schema change.
- Interaction to watch (found this audit): the new hard reject lands in dimension **2 (anatomy)**,
  which IMP-015's hint-routing block classes as a *mechanical/sampling* fault → `seed_bump`. For a
  phantom head that is the right routing (it is a per-roll motion-stage artefact, and mechanical
  rejects historically recover 78–88% on reseed), but it is worth confirming rather than assuming:
  if phantom-head rejects turn out to reseed poorly, the fault is compositional and belongs on
  `simplify_action`, or upstream on the i2v prompt (which is already the open P1 proposal).
- Metric for the next audit: (1) phantom-head/extra-head rejects per episode and their flip-to-accept
  rate on the following attempt (a low flip rate ⇒ re-route the hint); (2) whether they concentrate
  on `i2v_anchored=true` takes (would tie this to QM-033) and on the **tail frames** of the clip —
  the LTX-2.3 model card documents "last 6–8 frames may smear" as a known distilled-model limitation
  (SOURCES 2026-07-29), so a 5-frame evenly-spaced contact sheet may be systematically under-sampling
  the region where motion-stage artefacts are worst. **Proposal (not implemented): weight take-review
  frame sampling toward the last ~20% of the clip.** That is a review-artefact change that alters
  every take verdict, so it is kept unbatched and out of this audit.

### QM-041 Cloud audits have been reconstructing run config from git timestamps, not reading it   [S2] [mitigated]
- Evidence: this is the **fourth consecutive scheduled cloud audit** on this checkout (2026-07-21,
  -25, -29, and this one, 2026-08-01) and the **fourth in a row with zero new episode telemetry** —
  `output/` is still pinned at `20260722-135241` (confirmed again this run: `git log -1 --format=%ci
  -- output/` unchanged). Every prior pass had to infer *why* a given episode shows 0% vision
  coverage (deliberate `reviewer: "heuristic"` dev run vs. the `shot_auto_accept` prefilter silently
  eating the request queue vs. an unattended timeout) by cross-referencing episode-directory
  timestamps against `config.example.json`'s git history, because the run's actual effective
  `kidsong.review.*` config lives only in the gitignored `config.json` on the owner's machine —
  `output/<base>.log` (where `runlog.stage("quality", ...)` already prints tier/reviewer) is
  *also* gitignored (not in the `output/*` allow-list in `.gitignore`) and never reaches this
  checkout either. Independently re-derived this audit: `shot_auto_accept.enabled` went `true` at
  `a1bd0ee` (2026-07-21 17:58) and `false` at `5d91046` (2026-07-24 21:19); the five 0%-coverage
  episodes in this checkout (sunflower-family-tree 18:09 through alle-vögel 07-22 13:52) all fall
  inside that window — same conclusion the 2026-07-29 addendum already reached from the same
  evidence, now confirmed by a second, independent pass.
- Stage: edit/compliance (review orchestration) — telemetry, not a QC gate.
- Root cause: `review_log.json` is the one file in the allow-list built for the cloud audit to read,
  but it never recorded the config the run actually used — only the *outcomes* under that config.
- Fix / mitigation: IMP-023 (this audit) — log one additive `{"run_config": {quality_tier, reviewer,
  min_vision_coverage, shot_auto_accept_enabled}}` entry into `review_log_entries` at the top of
  `_generate_director`, right where the existing (gitignored-only) `runlog.stage("quality", ...)`
  line already computes the same values. Zero new computation — this only widens where an
  already-computed fact gets written. Additive entry shape (siblings: existing `{"stage": ...}` and
  `{"gate": ...}` entries); does not touch the take verdict schema
  (`accept`/`score`/`reasons`/`retry_hints`) and every existing reader of `review_log.json` already
  guards on `entry.get("shot")`/`entry.get("stage")` being present, so an entry with neither is
  inert to them (verified: `runstate.py`, `status.py` both skip on missing keys; the vision-coverage
  gate reads `.response.json` files off disk, not this log).
- Metric: from the next episode that actually renders, does a `"run_config"` entry appear as the
  first item in `review_log.json`, and does its `reviewer`/`shot_auto_accept_enabled` value correctly
  explain that episode's measured vision-coverage percentage without cross-referencing git history.

### QM-042 Every published video was undisclosed AI output   [S2] [monitoring]
- Evidence: measured 2026-08-02 against the shipped tree, before any fix. The body sent to
  `videos.insert` (`pipeline/youtube_upload.py`) carried exactly two status fields,
  `privacyStatus` and `selfDeclaredMadeForKids`; `status.containsSyntheticMedia` — the YouTube
  Data API field for the altered-or-synthetic declaration, available since 2024-10-30 — was
  never set on any code path. The description reached YouTube untouched apart from the
  `#Shorts` tag appended at `youtube_upload.py:80`. The finished MP4 carried no container
  tags: neither `edit.build_final_cut` nor `assemble_song.build_song_video` passed any
  `-metadata`. A grep of the whole tree for disclosure vocabulary
  (`ai.generated|synthetic|disclosure|Kennzeichnung`) returned 16 hits, none of them a
  disclosure. Three upload call sites (`studio/scheduler.py` ×2, `pipeline/generate.py`), zero
  disclosures.
- Stage: publish (uploader) — compliance, not a picture-quality gate.
- Root cause: the uploader was written against the minimum field set that makes an upload
  succeed. Nothing fails, is logged, or is displayed when a disclosure is absent, so the
  omission had no feedback path at all — the same shape as QM-031 (an auth bypass that
  survived because it made the control untestable).
- Scope, honestly: the binding duty is narrower than "we generate video, therefore we must
  label". AI Act **Art. 50(2)** (machine-readable marking) binds the *provider* of the
  generative system — the LTX-2/ACE-Step/Gemma authors — not an operator rendering with it,
  and carries a further grace period to 2026-12-02 for systems already on the market.
  **Art. 50(4)** binds the deployer (us) but is scoped to *deep fakes* per Art. 3(60):
  content resembling real persons/places/events that would falsely appear authentic. Per the
  Commission's final Art. 50 guidelines a realistic depiction of a fictitious-but-natural
  looking person counts; plainly unrealistic imagery does not. A stylised toon of invented
  toddlers most likely falls outside, and the "evidently artistic/fictional" carve-out would
  in any case reduce the duty to a disclosure that does not spoil the work. Filed as S2 rather
  than S1 for that reason — and filed at all because (a) the boundary is a judgement call
  about 3D-rendered children rather than a fact, (b) `pipeline/generate.py` ships stock-footage
  videos through the same uploader where the reading is different, and (c) YouTube's own rule
  applies regardless of the AI Act, with removal/demonetisation as the penalty.
- Fix / mitigation: IMP-050 (this pass). Disclosure is applied at the single seam every caller
  passes through, so a fourth call site cannot forget it.
- Metric: next real upload — does the returned video resource carry
  `status.containsSyntheticMedia` matching the video type (False for `kidsong`, True for
  stock-footage types), and does the published description end with the configured note?

### QM-043 A kids-styled channel uploaded children's content declared not-for-kids   [S2] [monitoring]
- Evidence: measured 2026-08-05 against `origin/main` by driving the real `scheduler._made_for_kids`.
  A channel with `style="kids"` and the made-for-kids box unticked, publishing a job whose
  `video_type` is `facts` or `conversation`, resolved to `made_for_kids=False`. The declaration
  consulted exactly two things — `channels.made_for_kids` and `is_kidsong_type(job.video_type)` —
  and ignored `style` on both the channel and the job, even though `style="kids"` is what makes the
  output child-directed in the first place: `studio/svm.py:83-89` selects a kids voice, "happy"
  music at high volume and a pink caption background from it, and `studio/views/api.py:30` forces
  the style on for the kids video type.
- Reachability is documented in this repo, not hypothetical: `studio/ideas.py:122` records that the
  real channel KinderKrippenLernLieder "used to be a 'facts' channel". A kids channel still carrying
  `video_type=facts` is exactly the failing shape. `studio/cascade.py:82` already scored the same
  style as a kids signal worth +3 — the uploader was the single place that did not.
- Stage: publish (uploader) — legal declaration, not a QC gate.
- Root cause: the same shape as QM-031 and QM-042. The declaration was derived from the columns
  that happened to be at hand rather than from what YouTube's rule actually asks, and nothing
  fails, logs or displays when it comes out wrong, so there was no feedback path.
- Against the rule as written: YouTube classifies content as made for kids when children are the
  primary audience **and** when characters, songs, activities or child-directed language appeal to
  children as part of a mixed audience. The first half was covered by `is_kidsong_type`; the second
  was not covered at all. Creators are responsible for the classification and cannot defer to
  YouTube's automated systems.
- Fix / mitigation: IMP-051 (this pass) — style becomes a third opt-in signal.
- Deliberately still NOT inferred: channel name/description wording. `cascade` may guess from prose
  because a wrong guess there only mis-ranks a candidate row; a wrong COPPA declaration is a legal
  filing, and it is not free in the other direction either (a made-for-kids video loses personalised
  ads, comments and end screens). Style and video type are explicit operator choices about the
  output; prose is not. Pinned by `test_channel_prose_alone_never_triggers_the_declaration`.
- Metric: next real upload from a kids-styled channel — does the returned video resource carry
  `status.selfDeclaredMadeForKids: true` regardless of its `video_type`?

### QM-044 Die Startseite scrollte auf dem Telefon seitwärts   [S3] [monitoring]
- Evidence: gemessen 2026-08-10 in echtem Chromium, 390x844 (iPhone 12/13/14). `document.
  documentElement.scrollWidth` = **400** bei 390px Fenster auf `/`, und dasselbe auf `/history`.
  Ursache war die Job-Tabelle: eine Zelle der Spalte "Fehler" enthielt einen `FileNotFoundError`
  mit vollem Pfad, also einen langen, kaum umbrechbaren String. `table{width:100%}` in
  `templates/base.html` ist gegenüber so einem Inhalt nur eine Bitte, und keine der 9 Tabellen in
  den 5 Templates stand in einem `overflow-x`-Container.
- Stage: UI (Studio) — Bedienbarkeit, kein Datenfehler.
- Root cause: die Zellbreite entsteht erst beim Rendern, aus Daten, die es zur Testzeit nicht gibt.
  `tests/test_ui_reachability.py` prüft bewusst Template-*Form* statt Pixel, weil die Hauptsuite
  CPU-only bleiben muss — und genau deshalb konnte diese Klasse dort nicht auffallen. Der Defekt war
  seit der Browser-Messung vom 2026-08-01 vorhanden und wurde damals nicht erfasst, weil nur
  Seitenhöhe und Fold-Position gemessen wurden, nicht die Breite.
- Fix / mitigation: IMP-052 — `.table-wrap` in `base.html`, alle 9 Tabellen darin.
- Metric: `tests/test_ui_no_horizontal_scroll.py` misst `scrollWidth <= viewport` auf 5 Seiten in
  2 Viewports. Negativkontrolle: neutralisiert man die CSS-Regel, fallen `/` und `/history` auf dem
  Telefon und sonst nichts.

### QM-045 Die zwei neuen Kanal-Sorten produzierten nichts   [S1] [monitoring]
- Evidence: gemessen 2026-08-10 auf `main`, unmittelbar nach dem Merge von PR #8, das
  `content_profile`, `language` und die Typen `hyperframes`/`brainrot` einführte. Zwei getrennte
  Totalausfälle, beide bei grüner Suite:
  1. `python -c "from pipeline.script_gen import _load_prompt; _load_prompt('brainrot', os.getcwd())"`
     → `FileNotFoundError: No prompt template for video type 'brainrot'`. `brainrot` war ein
     registrierter VIDEO_TYPE, für den `_enqueue_due_slots` Jobs anlegte; **jeder** dieser Jobs wäre
     sofort gestorben.
  2. Die `content_profiles` für `hyperframes_de/en` und `brainrot_short` setzten ausschliesslich Keys
     unter `kidsong`. `pipeline/generate.generate()` zweigt aber nur bei `video_type == "kidsong"` in
     die kidsong-Pipeline ab (`generate.py:106`); diese Typen laufen durch den allgemeinen Pfad, der
     `kidsong.*` nie liest. Die eigens gebauten Looks `explainer_clean` und `meme_pop` griffen
     folglich **nie**, ebensowenig Sprache, Stimme oder Transkriptionssprache.
- Stage: Produktion (Job-Anlage bis Skript) — Totalausfall, kein Qualitätsmangel.
- Root cause: **jede Schicht wurde gegen ihre Nachbarin geprüft, keine Anfrage je durch alle
  Schichten.** Registry, Profil-Auflösung, Branding und Dashboard hatten je eigene, grüne Tests;
  keiner davon stellte die Frage, ob am Ende ein Video entsteht. Dieselbe Form wie QM-031 (eine
  Kontrolle, die sich selbst untestbar machte) und QM-042 (eine Deklaration, die nirgends gesetzt
  wurde): niemand fragte nach dem Ergebnis, nur nach den Teilen.
- Fix / mitigation: IMP-053 — `prompts/brainrot.txt`, Sprachsteuerung im allgemeinen Pfad, Profile
  auf die Keys umgestellt, die dieser Pfad wirklich liest, plus `tests/features/
  test_producing_for_a_channel.py` als durchgehender Beweis pro Kanal-Sorte.
- Metric: Rekonstruiert man den Defekt (Overrides zurück auf reine `kidsong.*`-Keys), fallen
  9 Tests. Nächster echter Betrieb: produziert ein `hyperframes_de`-Kanal ein deutsches Video mit
  deutscher Stimme, ohne dass jemand etwas eintippt?
### QM-046 The venv is missing a pinned dependency, so every kidsong test errors locally   [S2] [mitigated]
- Evidence: `venv\Scripts\python` (3.11.9) has no `cv2`, although `opencv-python-headless` is
  pinned in `requirements.txt`. `tests/conftest.py`'s autouse `_clean_review_sessions` fixture
  does `from pipeline.kidsong import review`, and `review.py` does `import cv2` at module load,
  so the failure is at **fixture setup, not collection** — `pytest --collect-only` reports 2036
  tests and 0 errors while an actual run of any kidsong file errors out. Reproduced independently
  by three separate workers this session, and confirmed against an unrelated file
  (`tests/test_config_merge.py`) that fails identically, ruling out any change made this session.
- Stage: dev environment (not the pipeline).
- Root cause: the venv was never re-synced after `opencv-python-headless` entered
  `requirements.txt`. `requirements-dev.txt`'s own note that "pytest alone is not enough" (QM-020)
  covers the same class of gap.
- Why it matters beyond convenience: a green `--collect-only` next to a red run means **local
  test results silently understate the suite**. Every verification run this session had to load a
  throwaway `cv2` stub via `PYTHONPATH` to get an honest number, which is a workaround that hides
  precisely the cv2-dependent behaviour (`review.py`'s face-count heuristic) that QM-001's
  `force_i2v` emission depends on.
- Fix (2026-08-02): `Dockerfile.test` + `.dockerignore` in the repo root — a CPU-only image that
  installs `requirements-dev.txt` for real. Measured there: **2082 passed, 2 skipped, 0 failed,
  0 collection errors in 4:40**, against 2 skips instead of the 12 the host venv reports, because
  the deps the host is missing make those tests skip rather than run. See IMP-055. The host venv
  is still broken; `venv\Scripts\python -m pip install -r requirements.txt` fixes that separately
  and was deliberately not run (modifying the owner's venv was out of scope).
- Metric: `venv\Scripts\python -m pytest tests/test_kidsong_review_prefilter.py -q` passes without
  a stub on `PYTHONPATH`; and the full-suite pass count matches the 2036 collected.

<!-- Brought over from qm/audit-20260722 (PR #6), which predated the merges of
     PRs #7-#10 and so reused ids main had already taken. Renumbered on the way
     in (QM-042 -> QM-047), cross-references inside the block updated to match.
     Content otherwise unchanged. -->
- 2026-08-05 audit — a THIRD, independent evidence source for the group-shot shape of this
  defect, found by scanning every `s*_a*.response.json` for exact-duplicate reason strings
  (a signature of a human batch-rejecting via the browser queue rather than the per-shot
  vision-LLM, whose reasons are always varied prose). 17 takes across three of the channel's
  earliest episodes (all 2026-07-20, all pre-`a98e4a1`) carry the identical canned pair
  `["cast integrity", "To many similar moving kids in a shot"]` or the bare phrase alone —
  zuris-watering-blooms-day **9/16 shots (56%)**, zuri-kofi-nala-feel-the-breeze 6/16,
  splish-splash-fun-with-friends 2/16. The exact string appears nowhere in the codebase (not a
  test fixture), so this reads as a human reviewer's own repeated note, independent of both the
  AI vision-LLM's narrative rejects and QM-001's own theorising — and it names the same failure
  shape a third time: multiple visually-similar moving children in one frame is where identity
  breaks down hardest. zuris-watering-blooms-day's 56% rate is the single worst per-episode
  concentration of any cast-integrity signal in this checkout, vision-LLM or human.
## 2026-08-05 audit — 6th consecutive pass with zero new `output/` telemetry

`output/` is still pinned at `20260722-135241` (confirmed again: `git log -1 --format=%ci -- output/`
unchanged, `find output -name review_log.json | wc -l` still 23). No pixels, audio, contact sheets or
live `config.json` in this checkout; "inspect finals" was skipped, not performed, for the sixth
straight cloud audit. What *is* new this run is a tooling fix that unblocks the git-log-based
verification every prior audit's brief has asked for and every prior audit could not actually do.

### QM-047 Cloud-audit checkouts are shallow clones — every commit hash older than the clone's fetch depth is unresolvable   [S3] [resolved]
- Evidence: this run's brief asked to verify IMP-005 by "compar[ing] git-log dates against the
  episode timestamps encoded in output base names" against commit `a98e4a1`. `git cat-file -t
  a98e4a1` failed (`fatal: Not a valid object name`) — as did every other pre-2026-07-24 hash named
  anywhere in this ledger (`224ac9e`, `6310ecd`, `7aeba51`, `e086910`, `17a1e73`, `a1bd0ee`). `git
  rev-parse --is-shallow-repository` confirmed why: this checkout is a **shallow clone**
  (`git log --oneline | wc -l` = 79 before, spanning only back to 2026-07-21 18:25). Every prior
  audit hit the same wall silently — QM-041's "config.json isn't in the allow-list" diagnosis is real
  but incomplete; even a *config-independent* git-hash check was never actually executable, and
  nothing in the last five audits' write-ups says so. `git fetch --unshallow origin` resolved it
  immediately: 79 → 177 commits, several previously-invisible branches surfaced
  (`qm/audit-20260721`, `claude/video-generation-review-35ggkn`, …), and `a98e4a1` resolved to
  `2026-07-21 10:22:20 +0200 "Anchor character identity with a Z-Image reference, via the dead
  force_i2v"` — confirming the exact commit and timestamp IMP-005 has been reasoning about
  indirectly since 2026-07-21.
- Stage: tooling (cloud-audit environment), not the pipeline.
- Root cause: the cloud session's default `git clone` depth cuts the history at (or shortly before)
  the session's own start, which happens to sit inside the exact window (2026-07-21 through -24) this
  ledger's earliest, most load-bearing hypotheses (IMP-001 through IMP-006) were shipped in.
- Fix: none needed in the repo — `git fetch --unshallow origin` before any git-log-based
  verification is the fix, and it is cheap (a few seconds, this checkout). Recording it here so it
  is standing procedure rather than rediscovered per audit: **run `git fetch --unshallow origin`
  before checking `git cat-file -t <hash>` on any commit named in this ledger.**
- Metric: none needed — mechanical, and now documented. Watch for whether a *future* audit still
  reports a hash as unresolvable after unshallowing; that would mean the hash itself is wrong
  (rewritten history), a different failure this entry does not cover.

### Re-verification: IMP-005 (identity-reject share pre/post `a98e4a1`)   — see IMPROVEMENT_LOG.md IMP-005 for the numbers, still INCONCLUSIVE
With the commit now resolved (QM-047), this audit re-derived the pre/post split three independent
ways — grouping every `review_log.json` verdict by episode timestamp against `a98e4a1`'s
2026-07-21 10:22:20 +0200; cross-checking against every raw `s*_a*.response.json` file on disk,
splitting real (non-fallback) reviews from `auto_accepted`/"unattended mode"/"external review timed
out" fallbacks; and re-deriving the published QM-034 retry-flip-rate table from scratch as a sanity
check (reproduced exactly: `seed_bump` alone 7/33 = 21.2%, `seed_bump+simplify_action` 22/28 = 78.6%,
`force_i2v`(±`seed_bump`) 7/8 = 87.5%, `i2v_anchored=true` 35 takes total, matching QM-033's count
take-for-take). The headline number is unchanged by any of the three methods: of the 80
`s*_a*.response.json` files written across the 10 post-`a98e4a1` episodes, **zero are a real
(non-fallback) vision review** — every single one is `auto_accepted`, "unattended mode", or
"external review timed out". The pre/post comparison the brief asks for cannot be computed as a
real-vs-real comparison because the "post" side has no real data, for the same root cause IMP-005
has recorded for three straight audits: the vision reviewer was not running when the anchor-eligible
episodes rendered (QM-010). Status stays **INCONCLUSIVE** — this run does not move it, but it is now
resting on a confirmed commit and three convergent measurements instead of an inferred one.
