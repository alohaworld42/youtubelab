# Quality research sources

Curated by the quality-manager agent. Every entry: what it is, what to use it for, and the
one thing it taught us. Add new sources with the date; re-check "living" sources (blogs,
model docs) when researching a defect in their area. Seeded 2026-07-21.

## Model-specific (LTX-2.x — our renderer)

- [LTX-2.3 prompt guide](https://ltx.io/blog/ltx-2-3-prompt-guide) — official. Structure
  prompts: subject + action + constraints first, style later; concrete nouns/verbs beat
  vibe words. Use when revising `prompt_skeleton` in render styles.
- [Improving LTX-2.3 prompt adherence](https://ltx.io/blog/how-to-improve-ltx-2-3-prompt-adherence)
  — official. Relevant to QM-002/QM-003/QM-005 (drift/artefact classes).
- [IC-LoRA in LTX-2.3 for character consistency](https://ltx.io/blog/how-to-use-ic-lora-in-ltx-2)
  — official ComfyUI workflow for identity conditioning. Candidate upstream fix for
  QM-001/QM-003 beyond the Z-Image anchor (IMP-005).
- [Audio-to-motion prompting for LTX-2](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2)
  — official. Relevant to lyric-visual sync and dance/action shots.
- [RunDiffusion LTX-2 prompt guide](https://www.rundiffusion.com/ltx-2-prompt-guide) —
  community. Six-part shot-note format (scene anchor; subject+action; camera+lens; style;
  motion/time cues; guardrails); ~200-word cap; one main action per 2–3 s of video.
- [LTX 2.3 prompting techniques](https://cosmo-edge.com/ltx-2-3-prompting-guide/) —
  community. Motion-intent tokens ("steady dolly", "tripod-locked", concrete path lengths)
  for motion coherence; the larger the motion, the more precisely to describe it.

## Evaluation & QC methods

- [A Survey of AI-Generated Video Evaluation](https://arxiv.org/html/2410.19884v2) — map of
  metrics: CLIP-similarity temporal consistency, VBench's 16 dimensions (object/background
  consistency, motion smoothness). Mine for automatable checks before writing a new gate.
- [AIGCBench](https://arxiv.org/pdf/2401.01651) — i2v evaluation; relevant since IMP-005
  moved us toward image-conditioned generation.
- [LLM-based holistic video quality assessment (CVPRW 2025)](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Qi_Towards_Holistic_Visual_Quality_Assessment_of_AI-Generated_Videos_A_LLM-Based_CVPRW_2025_paper.pdf)
  — validates our contact-sheet + vision-LLM gate design; ideas for rubric dimensions.
- [AIGVE-Tool](https://arxiv.org/pdf/2503.14064) and
  [VideoGen-Eval](https://arxiv.org/pdf/2503.23452) — evaluation toolkits/agent systems;
  scan for checks portable to cut_qc.
- [Temporal drift in AI video (iMerit)](https://imerit.ai/resources/blog/solving-temporal-drift-in-ai-generated-video/)
  — production-side taxonomy of drift causes; matches our QM-001 wide-shot drift theory.

## Compliance (originality guardrail)

- [Cocomelon terms of use](https://www.cocomelon.com/terms-of-use) — Moonbug protects the
  CoComelon marks, characters (JJ, Nina, Cody), music, artwork and video content; any
  commercial use of their content requires a licence. Consequence for ZubiBop: characters,
  logo, thumbnails and melodies must keep clear distance from CoComelon (and by extension
  any kids' brand) — melodies only from `prompts/pd_songs.json` public-domain sources; the
  cast, intro and trade dress must stay recognisably our own. IP-lookalike drift is an S1
  finding (see quality-manager agent rubric).
- [Moonbug v. Babybus — $23.4M, affirmed 9th Cir.](https://www.awn.com/news/moonbug-wins-235-million-cocomelon-copyright-case)
  (added 2026-07-21) — Moonbug won $23.4M against Babybus's "Super JoJo"; the Ninth Circuit
  affirmed that the 3D cherubic toddler **JJ is a copyrightable character** and that the
  infringement spanned "characters, **settings**, song titles, lyrics and images." The one
  thing it taught us: the winning infringement theory is *overall look* — character design
  **plus settings** plus trade dress — not a single logo. A glossy-3D chibi-toddler show with
  CoComelon-style bathroom/playroom **sets** and a rounded-smiley-face-in-a-roundel logo sits
  squarely in the zone Moonbug actively and successfully litigates. Directly sharpens QM-011.
- [Kid Party Pros default judgment (2026)](https://news.bloomberglaw.com/ip-law/magistrate-judge-favors-blippi-cocomelon-studio-over-trademark)
  (added 2026-07-21) — Moonbug also pleaded **trade-dress + trademark** (not just copyright)
  and won a partial default judgment. Taught us: trade dress (the look-and-feel of the mark
  and get-up) is a separate, live enforcement lever — our intro badge is exposed on this axis
  independently of the character renders.

## Character-identity conditioning (QM-001 / IMP-006)

- [LTX — Maintaining character consistency in AI video](https://ltx.io/blog/how-to-maintain-character-consistency-in-ai-video)
  (added 2026-07-21) — official. Taught us: identity quality tracks *reference* quality — use a
  clean, frontal, well-lit close-up; **lead every prompt with a per-feature identity anchor
  block** (jawline/nose bridge/eyebrow arch/lip thickness — "same child, same face, same
  outfit"); identity holds well for 5–20 s clips and drifts past ~20 s. Our shots are 1.6–4.5 s,
  so single-clip anchoring (force_i2v/IMP-006) is well inside the reliable range. Backlog
  candidate: promote the cast_bible identity clause to a *leading* per-character anchor block.
- [LTX-Best-Face-ID / VBVR reference conditioning](https://huggingface.co/Alissonerdx/LTX-Best-Face-ID)
  (added 2026-07-21) — community. Taught us the concrete knob: character-sheet reference must be
  exactly **1536×1024** (its trained resolution) and VBVR/identity-LoRA strength ~0.75 to enforce
  consistency. Relevant if we move the Z-Image anchor toward a character-sheet workflow.
- [BachVid (arXiv 2510.21696)](https://arxiv.org/pdf/2510.21696) and
  [EntityBench (arXiv 2605.15199)](https://arxiv.org/pdf/2605.15199) (added 2026-07-21) —
  training-free consistent multi-shot / entity-consistent generation; scan when tackling
  cross-shot (not just within-shot) identity drift, our QM-001 wide-shot failure mode.

## Added 2026-07-31

- [LTX 2.3 prompting techniques (cosmo-edge)](https://cosmo-edge.com/ltx-2-3-prompting-guide/)
  — community, re-read for this pass. Three rules worth holding the pipeline to: describe
  observable physics, not emotions ("tightened jaw, furrowed brows" over "a furious man");
  camera language over aesthetic adjectives ("slow dolly in, turns head toward camera" over
  "epic cinematic masterpiece"); break a complex motion into micro-actions. On consistency it
  is blunt: **reference images are the most reliable fix**, and reused identical descriptors
  across generations are second — which is exactly what the cast bible + Z-Image/klein anchor
  (IMP-005/IMP-012/IMP-014) already implement.
- [AI video prompt guide 2026 (LTX)](https://ltx.io/blog/ai-video-prompt-guide) — official,
  re-read. The line that drove QM-015: a prompt must be **consistent, not conflicting**
  ("harsh noon sunlight casts short, sharp shadows", never "bright noon sunlight with dramatic
  moonlight shadows"). Our prompts were contradicting themselves on the one dimension that
  matters most here — how many children are on screen and who they are.
- Measurement, re-run 2026-07-31 by `tools/prompt_length_report.py` (no external source; run
  the script rather than quoting these numbers — the earlier hand-measured figures had gone
  stale, see IMP-037). Composed t2v prompts across the shipped corpus: **min 166 / median 190
  / p90 262 / max 273 words, 23.0% over** the ~200-word cap the RunDiffusion guide above
  recommends. The overrun is categorical, not a tail: one-child shots average 187 words and
  2.5% over cap, two-child 230 and 100%, three-child 264 and 100%. The identity sentence
  (~38 words per child) is the only component that scales with head count; everything else
  is fixed. Shortening the lighting/set tail — now `render_style`'s `prompt_tail`, with
  `config.example.json` shipping `pixar_toon_concise` as the B arm — clears one-child shots
  entirely (2.5% → 0.0%) and cannot touch the multi-child ones. See QM-030 for the two levers
  that are large enough, and IMP-035. Whether the shorter tail costs anything ON SCREEN is
  still UNMEASURED: it needs a GPU and one episode per arm.

<!-- Research notes from the qm/audit-2026-07-22 branch, merged in alongside this branch's. No IDs here, so nothing needed renumbering. -->
## Image-to-video conditioning (QM-016 — anchored-take strobe)
- [fal — LTX Video 2 Pro image-to-video prompt guide](https://fal.ai/learn/devs/ltx-video-2-pro-image-to-video-prompt-guide)
  (added 2026-07-25) — vendor guide for the i2v path. The one thing it taught us: **for i2v the
  prompt must describe MOTION, not what is already visible.** "Your source image already contains
  the visual information. Your prompt directs what happens next." It names *jittery motion* as the
  artefact caused by "conflicting movement directions", fixed by "simplify to a single, clear
  motion vector", and *subject distortion* as the artefact of physics-violating prompts. Directly
  relevant: `generate.py` sends the SAME `_shot_prompt` (style skeleton + cast identity block +
  wardrobe + setting + camera move + action) to the i2v graph as to t2v — i.e. it re-describes
  everything the conditioning keyframe already fixes. That is the vendor's textbook jitter recipe
  and is the leading hypothesis for QM-016.
- [LTX — How to reduce warble and AI pattern artifacts in LTX-2](https://ltx.io/model/model-blog/how-to-reduce-warble-and-ai-pattern-artifacts-in-ltx-2)
  (added 2026-07-25) — official; page body did not fetch (header overflow), indexed summary only.
  Taught us: warble/flicker is attributed to "missing temporal constraints, misaligned inputs, or
  conflicting instructions", and is called out as *more frequent in image-to-video workflows with
  poor first-frame alignment*. Re-fetch when working QM-016 in depth.
- [ComfyUI — LTX-2 image-to-video (distilled) workflow](https://comfy.org/workflows/video_ltx2_i2v_distilled-63cbc1ad946c/)
  and [LTX-2.3 ComfyUI tutorial](https://docs.comfy.org/tutorials/video/ltx/ltx-2-3)
  (added 2026-07-25) — the distilled i2v pipeline is documented at ~8 steps stage-1 + 4 stage-2 on
  a manual sigma schedule. Taught us: our `SIGMAS_OFFICIAL_9STEP` i2v default is already in the
  vendor-recommended band, so the sigma schedule is NOT the likely cause of QM-016 — which is what
  moves the prompt-redundancy hypothesis to the front. Community note from the same cluster: for
  i2v a higher output fps "makes the transition from the static image to video motion feel less
  jarring", i.e. the still→motion transition at the head of the clip is a known, expected
  discontinuity — the exact thing our `max_diff > 60` strobe check would flag. Hence the
  `max_diff_at` telemetry added this audit (IMP-016) before anyone touches that threshold.
- [Generative Inbetweening / SVD keyframe interpolation](https://svd-keyframe-interpolation.github.io/)
  and the i2v-conditioning discussion around [LTX-Video (arXiv 2501.00103)](https://arxiv.org/pdf/2501.00103)
  (added 2026-07-25) — taught us the underlying trade-off: first-frame latent conditioning that is
  too strong lets the model reconstruct the reference instead of learning how the scene evolves,
  and trainers deliberately *degrade* the conditioning image at high noise so motion must be
  inferred. Relevant if we ever expose an i2v conditioning-strength knob.
- [Prompt-A-Video (arXiv 2412.15156)](https://arxiv.org/pdf/2412.15156) and
  [Identity-Preserving Image-to-Video via Reward-Guided Optimization (arXiv 2510.14255)](https://arxiv.org/html/2510.14255v3)
  (added 2026-07-25) — evaluation-guided regeneration. Taught us the principle behind QM-017: the
  productive loop is reward/critique feedback **revising the prompt**, not resampling the same
  prompt. Our measured 11.5% flip rate on reseeded content rejects is the empirical version of
  that, and is why the reviewer rubric now routes hints by fault type (IMP-015).
- [Each Judge Its Own Yardstick (arXiv 2606.22918)](https://arxiv.org/pdf/2606.22918) (added
  2026-08-01) — measures cross-VLM disagreement on physical-plausibility video judging: different
  VLMs apply different evaluation taxonomies rather than converging on one; individual VLMs also
  disagree with human raters (false-accept and false-reject both occur). The one thing it taught
  us: our entire per-take rubric (scene-plausibility.md) is judged by a **single** vision-LLM
  backend, so its blocking verdicts inherit whatever taxonomy that one model happens to apply —
  there is no cross-model or model-vs-human calibration check anywhere in this pipeline. Not
  actionable as a code change today (a second-judge consensus step is a bigger change than any
  currently-open defect justifies), but directly relevant to how much weight future audits should
  put on any single vision-reviewed verdict, and worth revisiting if QM-016/QM-023's strobe/
  phantom-head calls ever look inconsistent between runs on the same take.
- [Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients)
  (added 2026-08-01) — official model card, more concrete than the general IC-LoRA blog post already
  above. Taught us the actual mechanism: a **reference-sheet** IC-LoRA — one composite still image
  with "one clean panel per distinct visual element" (character/prop/location) on a black
  background, no text, converted to a looped static video and fed as the i2v conditioning signal;
  trained at 768×448/121f/24fps; prompts are two-part (`Reference sheet:` / `Generated video:`), no
  trigger word. Directly relevant to QM-001/QM-003: this is a trained, purpose-built alternative to
  our own ad-hoc per-character `_cast_refs` + klein-4B anchor path (IMP-006/012/014), and the model
  card's explicit caveat — "the model only reproduces elements present on the sheet" and "bigger
  panels carry over better" — matches our own klein-path finding that reference quality/prominence
  drives anchor quality. Candidate to A/B against klein once the current anchor path has any
  vision-reviewed telemetry to compare it to (currently none — see QM-024/IMP-023).
## Music generation coherence (QM-020 / IMP-020 — new area, first sources 2026-07-29)
Nothing in this file covered the AUDIO half of the pipeline until now; the sing-coherence gate
(`c534c3c`) is the channel's first automated audio check, so it gets its own section.
- [ACE-Step-1.5 inference docs](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)
  (added 2026-07-29) — upstream, for the model we sing with. The one thing it taught us: **the
  arrangement is planned by a separate LM and is NOT covered by the seed.** With `thinking=True` /
  `use_cot_metas=True` a 5 Hz "chain-of-thought" LM populates `cot_bpm`, `cot_keyscale` and
  `cot_timesignature` per render; the exposed `seed` parameter applies to the diffusion (DiT)
  process and **no separate CoT seed is documented** — LM randomness is governed by
  `lm_temperature` / `lm_top_k` / `lm_top_p`. This independently confirms the premise of IMP-020's
  gate: an identical-input render really can come back at a different bpm, so measure-and-re-roll is
  a legitimate response rather than papering over a deterministic bug. It also names two cheaper
  levers the gate does not use: **`bpm` can be set explicitly, bypassing LM detection**, and
  `infer_method='ode'` (Euler, the default) is deterministic where `'sde'` is not. Those are the
  basis of proposal P7 — pin the tempo per library song instead of paying up to 3 full re-sings —
  to be decided once we know how often the gate actually fires.
- [ACE-Step 1.5 technical report (arXiv 2602.00744)](https://arxiv.org/pdf/2602.00744) and the
  [ACE-Step project page](https://ace-step.github.io/) (added 2026-07-29) — model provenance and the
  claimed quality bar. Taught us the framing rather than a knob: per-render output variance is a
  documented, accepted property of the model family (community write-ups describe it as
  "gacha-style" sensitivity to seed and input duration), so the correct engineering posture is a
  measurable acceptance criterion on the *output*, which is what `min_resultant` is. Note for a
  future audit: nothing here evaluates *beat-phase* coherence specifically — our resultant metric is
  our own, borrowed from `edit.py`'s grid fitter, and is not validated against any published measure.
## Multi-shot / episode-level consistency (QM-006 / QM-019 / IMP-018)
- [GroundShot: visually consistent multi-shot long video generation via entity-grounded shot
  scheduling (arXiv 2606.20799)](https://arxiv.org/html/2606.20799v1) (added 2026-07-29) — three
  things, all of which land on live findings. (1) **"Layered prompting" separates global scene
  description from shot-specific entities, explicitly to prevent leakage** — the same shape as
  `ada0cbf`'s `canonical_setting` + `subject_description`, which is useful corroboration that
  IMP-018 is the standard fix and not an improvisation. (2) **Canonical-reference discipline: "the
  first reliable reference becomes the canonical reference. Later candidates do not overwrite it,"**
  producing a star-shaped consistency structure instead of a drifting chain — directly relevant to
  our `output/_cast_refs/<id>/canonical.png` anchor (IMP-014); worth confirming we never regenerate
  over a good reference. (3) **"A character occupying only a few pixels in a wide establishing shot
  gives a much weaker identity signal than the same character shown in a medium close-up"** — an
  independent statement of QM-001's wide-shot theory, and the motivation for their reordering
  generation so reference-source shots render first. Reported 4.18/5.0 vs 3.62 human identity
  consistency.
- [MultiShotMaster (Kling/Kuaishou, CVPR 2026)](https://studio.aifilms.ai/blog/multishotmaster-multi-shot-video-generation)
  (added 2026-07-29) — community write-up of the released framework. Taught us the concrete prompt
  architecture now converged on for multi-shot work: **a global caption defining scene, characters
  and environment, plus per-shot captions carrying only action, background and camera**, with
  identity held across shots from text alone. That is exactly what `lyrics.py` now stamps. Also
  notes "environment anchoring" (a defined environment persisting across wide and close shots) as a
  first-class model feature in the 2026 generation — i.e. our text-level pin is the portable version
  of something models are starting to support natively.
## Motion-stage artefacts and i2v behaviour (QM-016 / QM-023)
- [LTX-2.3 dual-character LoRA model card](https://huggingface.co/SyFeee/LTX2.3-Dual-Character-en)
  (added 2026-07-29) — community, but the only place we have found documented failure modes for the
  distilled LTX-2.3 graph. Taught us three checkable things: **"last 6–8 frames may smear; trim
  post-hoc if needed"** — a *positional* artefact, which means our evenly-spaced 5-frame take-review
  sheet systematically under-samples the region where motion-stage defects like QM-023's phantom
  heads are worst (basis of proposal P8); **object tracking is lost during high-intensity action**
  ("detachable objects may dematerialize mid-scene") — relevant to the prop gate; and **cross-shot
  identity drift is mitigated by re-pinning the reference image at frame 0 of each new shot**, which
  is what our keyframe anchor does. It explicitly does *not* document extra/duplicate heads, so
  QM-023 stays an unconfirmed-in-the-literature failure mode of ours.
- [Improving Motion in Image-to-Video Models via Adaptive Low-Pass Guidance (arXiv 2506.08456,
  CVPR 2026 highlight)](https://arxiv.org/html/2506.08456) (added 2026-07-29) — the one thing it
  taught us cuts against our own instinct on QM-016: **i2v adaptation of a t2v model systematically
  SUPPRESSES motion**, because the sampling trajectory converges to a "shortcut solution" that
  reconstructs the reference frame; their fix is to condition on a low-pass-filtered reference at
  early timesteps and restore the full one later. Implication for us: an anchored take should be
  *less* dynamic than an unanchored one, so a strobe flag driven by a single large frame-pair delta
  on an otherwise low-motion anchored clip is more consistent with a head-of-clip (or, post-`025ed80`,
  a black-tail-frame) discontinuity than with genuine sustained flicker. Testable today with
  IMP-016's `max_diff_at` + `luma_min` — no new code. If we ever expose an i2v conditioning-strength
  or schedule knob, ALG is the reference method.
- [EntityBench / Background Consistency (BG-Sim)](https://www.emergentmind.com/topics/entitybench)
  (added 2026-07-29; EntityBench itself was already listed under character-identity conditioning) —
  taught us a candidate *automatable* detector for QM-023: BG-Sim scores background fidelity on
  foreground-masked regions and is designed to penalise **hallucinated** background content, which is
  precisely what a phantom head is. Filed as a research lead only — our current gate is a vision-LLM
  reading a contact sheet, and adding a masked-similarity metric is a bigger change than any defect
  here currently justifies.
- [AI Act Art. 50 — transparency obligations](https://artificialintelligenceact.eu/article/50/) ·
  [European Commission FAQ on Art. 50](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act) ·
  [Bird & Bird on the final Art. 50 guidelines](https://www.twobirds.com/en/insights/2026/european-commission-adopts-final-guidelines-on-ai-act-article-50-transparency-obligations-first-impr)
  (added 2026-08-02, for QM-042) — the source of the provider/deployer split this repo's disclosure
  design rests on. Art. 50(2)'s machine-readable marking binds whoever puts the *generative system*
  on the market (LTX-2, ACE-Step, Gemma), not an operator rendering with it; Art. 50(4)'s disclosure
  binds the deployer but only for "deep fakes" as defined in Art. 3(60). The final guidelines draw
  the line at realism, not at synthesis: a realistic depiction of a fictitious person is in scope,
  plainly unrealistic imagery is not. Also the source of the "evidently artistic/fictional" carve-out
  that makes a description note sufficient and a burnt-in overlay unnecessary.
- [Art. 50 was not postponed by the Digital Omnibus](https://www.aiactblog.nl/en/posts/article-50-transparency-deadline-2-august-2026)
  (added 2026-08-02) — worth pinning because the omnibus *did* move the Annex III high-risk deadlines
  to 2027-12-02, and it is easy to assume the transparency duties moved with them. They did not: they
  applied from 2026-08-02. Systems placed on the market before that date get until 2026-12-02 for the
  Art. 50(2) marking specifically.
- [YouTube: disclosing altered or synthetic content](https://support.google.com/youtube/answer/14328491) ·
  [YouTube Data API revision history](https://developers.google.com/youtube/v3/revision_history)
  (added 2026-08-02, for QM-042/IMP-050) — the platform rule, which applies independently of the AI
  Act and carries removal/demonetisation as its penalty. Two things it taught us: the API field is
  `status.containsSyntheticMedia` (added 2024-10-30, settable on `videos.insert`/`update`), and the
  disclosure is scoped to *realistic* altered content — wholly animated work is explicitly out of
  scope. That second point is why `youtube_synthetic_media: "auto"` resolves to False for `kidsong`
  rather than declaring everything: a wrong label is its own defect.
- [YouTube: Bestimmen, ob Inhalte "speziell für Kinder" sind](https://support.google.com/youtube/answer/9528076?hl=de)
  (added 2026-08-05, for QM-043) — the COPPA/FTC classification rule, and the source of the criterion
  this repo was missing. The declaration is required of creators worldwide, and it turns on two
  things, not one: whether children are the **primary** audience, and whether characters, songs,
  activities or child-directed language appeal to children as part of a **mixed** audience. The
  uploader covered only the first (via `is_kidsong_type`). The page also states that creators are
  legally responsible for the classification and cannot rely on YouTube's automated systems — which
  is why this is derived from explicit operator choices (style, video type) rather than guessed from
  prose, and why it is asserted rather than assumed.

## Added 2026-08-05 (6th consecutive cloud audit, still no new `output/` telemetry)
- [OneStory: Coherent Multi-Shot Video Generation with Adaptive Memory (arXiv 2512.07802)](https://arxiv.org/abs/2512.07802)
  — reformulates multi-shot generation as *next-shot synthesis*: an "adaptive conditioner" does
  importance-guided patchification on the MOST INFORMATIVE frames selected from prior shots (not
  simply the literal last frame) and feeds that as compact conditioning for the next shot. Directly
  relevant to our `shot_chaining_enabled` mechanism (AGENTS.md — shot N+1 conditions on shot N's
  accepted last frame): OneStory's frame-selection module is a strictly smarter version of "use the
  last frame" and is a candidate refinement if chaining is ever revisited, but not an immediate
  action — no chaining telemetry exists in this checkout to motivate it yet.
- [VideoMemory: Toward Consistent Video Generation via Memory Integration (arXiv 2601.03655)](https://arxiv.org/abs/2601.03655)
  — a 54-case multi-shot consistency benchmark (character/prop/background-persistent scenarios) built
  around explicit memory integration across shots. Filed as a research lead for QM-006/QM-001's
  cross-shot (not just within-shot) drift class — same category as EntityBench/GroundShot, not yet
  actionable here.
- Re-checked [Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients)
  (already logged 2026-08-01): still the most concrete candidate fix for QM-001's *group*-shot cast
  bleed specifically, because it is the one mechanism in SOURCES that pins **multiple** characters at
  once from a single reference sheet — `force_i2v`/klein (IMP-006/012/014) is single-subject by
  design (`_reference_for_single_subject` returns `None` on multi-child shots) and cannot anchor a
  group shot at all. See IMP-056 below.
