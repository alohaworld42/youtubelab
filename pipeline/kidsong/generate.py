"""
kidsong.generate — Orchestrate a full children's-song video.

Two modes (config kidsong.mode):
  "director" (default): LLM lyrics -> ACE-Step sings -> Whisper words -> beat
      grid -> director shot list -> ComfyUI LTX-2.3 t2v renders with QC review
      loop -> beat-aligned edit -> [YouTube upload]
  "classic": the previous verse-clip pipeline (SDXL keyframes + Wan/LTX i2v,
      Ken Burns fallback). Zero-risk escape hatch.

Use from the web UI (app.py), pipeline.generate, or the command line:
    python -m pipeline.kidsong.generate --topic "counting to five"
    python -m pipeline.kidsong.generate --list-resumable
    python -m pipeline.kidsong.generate --resume 20260720-085055-kidsong-splish-splash

Every run writes a timestamped log to ``output/<base>.log`` and appends any
crash to ``output/kidsong_errors.log`` — see pipeline.kidsong.runlog.
"""
import copy
import json
import logging
import os
import re
import shutil
import sys
import time

import requests

from pipeline.atomicio import atomic_write_json
from pipeline.config import abspath, load_config
from pipeline.generate import _slug
from pipeline.kidsong import runstate
from pipeline.kidsong.render_style import (
    i2v_workflow_name, resolve_style, workflow_name,
)
from pipeline.kidsong.runlog import RunLog

log = logging.getLogger("kidsong.generate")


def generate_kidsong(topic=None, do_upload=False, cfg=None, on_progress=None,
                     resume_base=None, runlog=None):
    """Generate a kidsong video.

    `resume_base` picks up an interrupted run: its song, shot ledger and every
    take already on disk are reused, and only the missing shots are rendered.
    """
    if cfg is None:
        cfg = load_config()

    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # The run owns its own log unless a caller (the CLI, the scheduler) already
    # opened one — so an in-process call from the web UI is logged too.
    own_log = runlog is None
    if own_log:
        # Console output stays the caller's job when on_progress is supplied
        # (the web UI renders progress itself); the file always gets everything.
        runlog = RunLog(out_dir, base=resume_base, stream=on_progress is None)
        runlog.fingerprint(cfg, extra={"topic": topic, "resume": resume_base or "(new run)"})

    def step(msg):
        # One call site, two sinks: the StreamHandler reproduces the exact
        # "  -> …" console line, the FileHandler timestamps it into <base>.log.
        runlog.info("  -> %s", msg)
        if on_progress:
            on_progress(msg)

    try:
        return _run(topic, do_upload, cfg, step, out_dir, resume_base, runlog)
    except BaseException as exc:
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            runlog.exception("Kidsong run FAILED", exc)
        else:
            runlog.warning("Kidsong run interrupted (%s)", type(exc).__name__)
        raise
    finally:
        if own_log:
            runlog.close()


def _run(topic, do_upload, cfg, step, out_dir, resume_base, runlog):
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # 1. Lyrics — or, when resuming, the song persisted by the original run.
    if resume_base:
        song = runstate.load_song(out_dir, resume_base)
        if not song:
            raise FileNotFoundError(
                f"Cannot resume {resume_base}: "
                f"{runstate.song_json_path(out_dir, resume_base)} is missing or unreadable."
            )
        base = resume_base
        runlog.stage("resume", base=base, title=song.get("title"))
        step(f"Resuming run {base} — reusing takes already on disk…")
    else:
        runlog.stage("script")
        step("Writing song lyrics with the LLM…")
        from pipeline.kidsong.lyrics import generate_song

        song = generate_song(topic, cfg)
        base = f"{stamp}-kidsong-{_slug(song['title'])}"

    # Now that <base> exists, move the log next to the run's other artifacts.
    runlog.rebind(base)

    if cfg.get("kidsong", {}).get("mode", "director") == "director":
        try:
            return _generate_director(
                song, base, do_upload, cfg, step, out_dir,
                topic=topic, resume=bool(resume_base), runlog=runlog,
            )
        except BaseException as exc:
            # Contract 3: surface the base this run died on so a caller (the
            # studio scheduler) can requeue the job with resume_base=<base>
            # instead of starting a brand-new episode from scratch on retry.
            # Only set once, so an exception re-raised through nested handlers
            # keeps the base of the run that actually died, not an outer
            # wrapper's.
            if not hasattr(exc, "kidsong_resume_base"):
                exc.kidsong_resume_base = base
            raise

    # 2. The song itself. Preferred path: ACE-Step actually SINGS the lyrics
    # with a full arrangement (no separate music bed needed). Fallback path:
    # edge-tts child voice + procedural music bed.
    voice_path = os.path.join(out_dir, base + ".wav")
    music_path = None
    verse_times = None  # ACE path: derived from Whisper words later (step 5b)

    sung = False
    if cfg.get("kidsong", {}).get("singer", "ace") == "ace":
        step("Singing the song (ACE-Step on GPU)…")
        try:
            from pipeline.kidsong.sing import sing_song

            vocals = sing_song(song, cfg, voice_path)
            sung = True
        except Exception as e:
            step(f"ACE-Step unavailable ({e}) — falling back to edge-tts.")

    if not sung:
        step("Recording the child voice (edge-tts)…")
        from pipeline.kidsong.song_audio import make_music_bed, synthesize_vocals

        vocals = synthesize_vocals(song["verses"], cfg, voice_path)
        verse_times = vocals["verse_times"]

        step("Composing the music bed…")
        music_path = os.path.join(out_dir, base + "-music.wav")
        make_music_bed(min(vocals["duration"], cfg["video"]["max_seconds"]), music_path)

    duration = min(vocals["duration"], cfg["video"]["max_seconds"])

    # 4. Scene images (the GPU part)
    step("Rendering 3D scenes on the GPU (SDXL)…")
    from pipeline.kidsong.scenes import render_scenes

    scenes_dir = os.path.join(out_dir, base + "-scenes")
    scene_pngs = render_scenes(
        [v["scene"] for v in song["verses"]],
        song["characters"],
        scenes_dir,
        cfg,
        on_progress=step,
    )

    # 4b. Animate scenes (optional GPU image-to-video pass)
    if cfg.get("kidsong", {}).get("animate", True):
        step("Animating scenes (image-to-video on GPU)…")
        try:
            from pipeline.kidsong.animate import animate_scenes

            clips = animate_scenes(
                scene_pngs, [v["scene"] for v in song["verses"]], scenes_dir, cfg, on_progress=step
            )
            scene_media = [c or p for c, p in zip(clips, scene_pngs)]
        except Exception as e:
            step(f"Animation unavailable ({e}) — falling back to still images.")
            scene_media = scene_pngs
    else:
        scene_media = scene_pngs

    # 5. Word captions
    step("Aligning word-level captions (Whisper)…")
    from pipeline.captions import get_word_timestamps

    words = get_word_timestamps(
        voice_path,
        cfg,
        uppercase=cfg["captions"].get("uppercase", True),
        lyrics=song["verses"],
        duration=duration,
        language=(cfg.get("kidsong") or {}).get("language", "en"),
    )
    words = [w for w in words if w["start"] < duration]
    for w in words:
        w["end"] = min(w["end"], duration)

    # 5b. ACE path: locate each verse in the sung audio via the aligned words.
    if verse_times is None:
        from pipeline.kidsong.sing import verse_times_from_words

        verse_times = verse_times_from_words(words, song["verses"], duration)

    # 6. Assemble — finished videos go to Final/, intermediates stay in out_dir.
    from pipeline.kidsong.assemble_song import build_song_video

    final_dir = os.path.join(out_dir, "Final")
    os.makedirs(final_dir, exist_ok=True)
    # Never write over a finished episode, even if a caller reuses a <base>.
    video_path = runstate.reserve_output_path(
        os.path.join(final_dir, base + ".mp4"), companions=(".intro.json",)
    )
    if os.path.basename(video_path) != base + ".mp4":
        step(f"  NOTE: Final/{base}.mp4 exists — writing {os.path.basename(video_path)}.")
    build_song_video(
        cfg,
        voice_path,
        music_path,
        scene_media,
        verse_times,
        words,
        duration,
        video_path,
        on_progress=step,
    )

    result = {
        "video_path": video_path,
        "title": song["title"],
        "description": song["description"],
        "tags": song["tags"],
        "duration": duration,
    }

    # 7. Upload — children's content must be declared made-for-kids on YouTube.
    if do_upload:
        step("Uploading to YouTube (declared made-for-kids)…")
        from pipeline.youtube_upload import upload

        up_cfg = copy.deepcopy(cfg)
        up_cfg["youtube"]["made_for_kids"] = True
        up = upload(video_path, song["title"], song["description"], song["tags"], up_cfg)
        result["youtube"] = up

    # tidy: drop the intermediate wavs (scene PNGs are kept for reuse/review)
    for p in (voice_path, music_path) if music_path else (voice_path,):
        try:
            os.remove(p)
        except OSError:
            pass

    step("Done.")
    return result


def _wav_duration(path):
    """Seconds of audio in a PCM wav — stdlib only, so resume stays import-cheap."""
    import wave

    with wave.open(path, "rb") as wf:
        rate = wf.getframerate() or 1
        return wf.getnframes() / float(rate)


def _cast_for(shot, song):
    """The identity clause for one shot's characters, backed by the versioned
    cast bible (pipeline.kidsong.cast) so a given child's look is pinned
    across shots and episodes.

    Only describes the characters actually IN the shot — describing the whole
    cast in a closeup invites the model to draw everyone (crowding). If the
    cast bible is unavailable or raises (missing/corrupt prompts/cast_bible.json,
    import failure, ...) this falls back to the OLD free-text parsing of
    song["characters"] and LOGS A WARNING — the previous version of this
    function fell back to that same parsing silently, which hid real bugs.
    """
    return _cast_for_with_count(shot, song)[0]


def _cast_for_with_count(shot, song):
    """`(identity_clause, described_count_or_None)`.

    The count is how many children the clause ACTUALLY describes, so
    `_shot_prompt` can never state a head count that disagrees with its own
    cast description. That disagreement is exactly what produced the legacy
    arm's self-contradictory "Exactly one children are on screen:" followed by
    three described people, and the bible arm's "Exactly three children" on a
    specified closeup of one.
    """
    names = shot.get("characters") or ["all"]
    shot_type = shot.get("shot_type")
    try:
        from pipeline.kidsong import cast

        return (
            cast.cast_sentence(names, shot_type=shot_type),
            cast.expected_child_count(names, shot_type=shot_type),
        )
    except Exception as exc:
        log.warning(
            "Shot %s: cast bible unavailable (%s) — falling back to free-text "
            "parsing of song['characters'].",
            shot.get("id"), exc,
        )
        return _cast_for_legacy_with_count(shot, song)


def _cast_for_legacy(shot, song):
    """Pre-cast-bible free-text parser of song["characters"]. Kept only as the
    crash-proof fallback `_cast_for` uses when the cast bible can't be read."""
    return _cast_for_legacy_with_count(shot, song)[0]


def _cast_for_legacy_with_count(shot, song):
    """`(clause, described_count_or_None)`. The count is None whenever this
    falls back to the whole free-text blob, because that blob describes an
    unknown number of people — and stating a head count we can't back up is how
    "Exactly one children are on screen: <three people>" happened."""
    full = str(song.get("characters", "")).rstrip(".")
    names = shot.get("characters") or ["all"]
    if names == ["all"] or "all" in names:
        return full, None
    # Drop the "Three adorable ... toddlers:" preamble before splitting into
    # per-character fragments, or it leaks into single-character prompts.
    body = full.split(":", 1)[1] if ":" in full else full
    fragments = [f.strip(" ;,") for f in re.split(r"[;.]", body)]
    picked = [f for f in fragments if any(n.lower() in f.lower() for n in names)]
    if not picked:
        # Nothing matched the shot's names — we are about to describe EVERYONE
        # in the blob, so we do not know (and must not claim) a head count.
        return full, None
    label = "One adorable Black toddler" if len(picked) == 1 else "Adorable Black toddlers"
    return f"{label}: " + "; ".join(picked), len(picked)


_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _cast_body(cast_sentence):
    """Strip the cast bible's own leading count label ('One adorable Black
    toddler: ', 'Three adorable Black toddlers: ', ...) from a cast_sentence,
    leaving just the character description(s). `_shot_prompt` states its own
    explicit head count ("Exactly three children are on screen: ..."), so
    without this the two would stack into a redundant "Exactly three
    children are on screen: Three adorable Black toddlers: ..."."""
    if ": " in cast_sentence:
        return cast_sentence.split(": ", 1)[1]
    return cast_sentence


def _expected_count(names, shot_type=None):
    """How many children the cast bible says `names` resolves to, or None when
    that can't be determined (cast bible unavailable AND the shot's own
    ["all"]/None doesn't tell us the ensemble size either) — callers must not
    state an exact head count they can't back up."""
    try:
        from pipeline.kidsong import cast

        return cast.expected_child_count(names, shot_type=shot_type)
    except Exception:
        if not names or names == ["all"] or "all" in names:
            return None
        return max(1, len(names))


def _prompt_subject(cast_sentence, shot):
    """The name the render prompt actually puts on screen for a one-child shot.

    Not always the shot's own `characters[0]`: when that name is unresolvable
    the cast bible substitutes a real cast member (shot lists carrying "Amira"
    render Nala), and a rewritten action must name the child the prompt
    describes, not the one it dropped."""
    body = _cast_body(cast_sentence).strip()
    head = body.split(",", 1)[0].strip()
    if head and head[:1].isupper() and " " not in head:
        return head
    names = [n for n in (shot.get("characters") or []) if n and n != "all"]
    return names[0] if names else None


def _agree_with_count(action, setting, count, subject, story_subject=None, decreep=False):
    """Force the action/setting prose to agree with the head count this prompt
    is about to assert — belt-and-braces over director.py's source-level fix.

    `_shot_prompt` renders `action` and `setting` verbatim, so a shot list
    written before that fix (or hand-edited, or produced by an LLM overlay that
    slipped through) still reaches the text encoder saying "exactly one child:
    Zuri" and then "The kids ... in Amira, Kofi, and Zuri are rinsing their
    mouths". That self-contradiction is what two GPU A/B rounds measured as the
    dominant remaining defect: the model resolves it by cloning the one
    described child 2-3x. Degrades to the unmodified text if director.py can't
    be imported — a sanitizer problem must never block a render.

    `story_subject` (already sanitized by `_clean_story_subject`, or None) is
    the shot's non-child story subject, e.g. "a tiny round cartoon mouse". It
    is never counted in `count` and never renders as a child, but its own
    prose ("a tiny round cartoon mouse scampers up the tall grandfather
    clock") must survive untouched — `sanitize_action`/`sanitize_setting` only
    ever rewrite text that names a CAST member or uses group language, so a
    subject-led sentence is already safe in practice (the subject is not a
    recognized cast name); this is the explicit belt-and-braces guard so a
    future cast member sharing words with a subject phrase can never cause
    the subject's own sentence to be rewritten into naming a child instead.

    `decreep` (kidsong.decreep.enabled) is threaded straight through to
    `director.sanitize_action`, so a contradiction-repair rewrite can never
    reintroduce a camera-referencing verb phrase behind this feature's back."""
    try:
        from pipeline.kidsong import director
    except Exception as exc:
        log.warning(
            "Could not import director to check prompt head-count consistency "
            "(%s) — action/setting used verbatim.", exc,
        )
        return action, setting

    subj_text = str(story_subject or "").strip().lower()
    action_led_by_subject = bool(subj_text) and str(action or "").strip().lower().startswith(subj_text)
    setting_names_subject = bool(subj_text) and subj_text in str(setting or "").lower()

    clean_setting = setting if setting_names_subject else director.sanitize_setting(setting)
    clean_action = action if action_led_by_subject else director.sanitize_action(
        action, count, subject=subject, decreep=decreep,
    )
    if clean_setting != setting or clean_action != action:
        log.info(
            "Prompt head-count contradiction repaired (count=%s, subject=%s): "
            "action %r -> %r; setting %r -> %r",
            count, subject, action, clean_action, setting, clean_setting,
        )
    return clean_action, clean_setting


def _clean_story_subject(raw):
    """Sanitize a director-supplied `story_subject` field for prompt use.

    Defensive, not trusting: director.py validates this field at plan time,
    but `_shot_prompt` must never crash or render garbage from a bad value —
    a non-string (int, dict, ...), a blank/whitespace string, or an absurdly
    long runaway value are all treated as "no subject" (or truncated) rather
    than rendered as-is. Returns a trimmed non-empty string, or None."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if len(text) > 120:
        text = text[:120].rstrip()
    return text or None


def _canonical_subject_clause(song, story_subject):
    """The text that introduces `story_subject` in a shot's render prompt.

    `song["subject_description"]` (prompts/pd_songs*.json's fixed per-episode
    look for its non-child story subject — colour/material/size relative to
    the children, e.g. "a small round woolly BLACK sheep, knee-high to the
    children, ...", stamped onto the song dict by `lyrics.pd_song_to_dict`) is
    the `identity_clause` equivalent for that subject: the cast bible pins the
    children's look into every prompt via `_cast_body`/`identity_clause`, but
    the story subject used to be introduced by nothing more than its own
    (often terser) noun phrase, e.g. "a round woolly black sheep" — one short
    mention with no colour/size anchor repeated verbatim, which measurably
    drifted (a black sheep rendering white in some keyframes, a star with a
    face in some shots and without in others). When the song carries a
    description, it is used verbatim in place of the bare `story_subject`
    phrase EVERY time the subject is mentioned, so it can never quietly change
    shot to shot. Falls back to the bare `story_subject` for a song without
    the field (every song that predates it, and any library entry the
    catalog marks as having no single recurring subject) — byte-identical.
    """
    if not story_subject:
        return ""
    description = str((song or {}).get("subject_description") or "").strip()
    return description or story_subject


def _gaze_sentence(shot, story_subject, count):
    """The de-creep replacement for "...face the camera as they move" — what
    the child (or children) in THIS shot are doing with their eyes. Gated by
    kidsong.decreep.enabled (see `_shot_prompt`); flag off never calls this.

    Every branch is a POSITIVE statement of where the eyes ARE — never write
    "not at the camera" or similar. The Gemma-3 text encoder driving LTX-2
    reads negation unreliably, so a negated camera cue is liable to render as
    a toddler looking AT the camera: precisely the defect this removes.

    `shot["gaze"]` (str, <=80 chars once stripped, no newline) is a director-
    supplied direction — "at each other", "down at the blocks" — rendered as
    "They look {gaze}." ("The child looks {gaze}." for a one-child shot). Any
    other value, or a missing/invalid one (junk falls back exactly like
    "absent"), resolves from what the shot already tells us: `story_subject`
    (the mouse/lamb/sheep/boat the verse is about) wins when present, else a
    generic ensemble/solo default. The literal value "at the camera" is
    special-cased to a warmer sentence than the generic template would
    produce verbatim — this is the one shot per episode
    `director._fallback_planner` deliberately lets play to the camera (a
    hook/refrain verse's opening beat), not the pervasive blank stare.
    """
    singular = count == 1

    raw_gaze = shot.get("gaze")
    gaze = None
    if isinstance(raw_gaze, str):
        text = raw_gaze.strip()
        if text and len(text) <= 80 and "\n" not in text and "\r" not in text:
            gaze = text

    if gaze == "at the camera":
        return (
            "The child faces the camera with a warm natural smile, blinking "
            "naturally." if singular else
            "They face the camera with warm natural smiles, blinking naturally."
        )
    if gaze:
        return f"The child looks {gaze}." if singular else f"They look {gaze}."

    if story_subject:
        return f"Their eyes are on {story_subject}."
    if singular:
        return "The child looks at what their hands are doing."
    return "The children look at each other as they play."


def _rename_offbible_children(action, setting, shot):
    """Rewrite an invented child name inside `action`/`setting` to the bible
    child the cast resolver actually put on screen for it.

    `cast._resolve_with_fallback` already substitutes an off-bible name in
    `shot["characters"]` (Amira -> Nala) so the identity sentence describes a
    real, consistently-drawn character. The shot's free text was never repaired
    to match, so shipped prompts read like this (measured 2026-07-31: 21 of the
    1130 action/setting fields across every shotlist in output/ named a child
    the cast bible has never heard of):

        "Exactly three children are on screen: Zuri, Kofi and Nala. …
         A wide shot: Amira and Kofi standing in front of the sink …"

    Four names, a head count of three, and one of them with no description
    attached — so the text encoder invents a fourth child's appearance from
    the name alone. That is the invented-extra-child failure this pipeline
    already spends a keyframe pass, a head-count clamp and a negative-term
    list fighting, handed to the model in its own prompt. Both official and
    community LTX prompt guidance is emphatic that a prompt must not contradict
    itself; this removes the self-contradiction at the source.

    Matching is word-boundary and case-insensitive, so a name never fires
    inside a longer word ("Ava" in "Avalon"). Names are applied longest-first
    so a multi-word entry cannot be pre-empted by a shorter one nested in it.
    Best-effort: if the cast bible can't be read the text is returned
    untouched, exactly as before.
    """
    names = shot.get("characters") or []
    if not names or not (action or setting):
        return action, setting
    try:
        from pipeline.kidsong import cast

        mapping = cast.substitution_map(names, shot_type=shot.get("shot_type"))
    except Exception:
        return action, setting
    if not mapping:
        return action, setting

    def _rewrite(text):
        if not text:
            return text
        for invented in sorted(mapping, key=len, reverse=True):
            replacement = mapping[invented]
            if not invented.strip() or invented.strip() == replacement:
                continue
            text = re.sub(
                r"\b%s\b" % re.escape(invented.strip()), replacement, text,
                flags=re.IGNORECASE,
            )
        return text

    return _rewrite(action), _rewrite(setting)


def _identity_block(shot, song, cfg):
    """The shared identity/head-count/story-subject sentences — and the
    action/setting text they were contradiction-repaired against — used by
    BOTH `_shot_prompt` (the t2v prompt) and `_keyframe_prompt` (the Z-Image
    keyframe-first still). Factored out so the cast-bible lookup, the
    `_agree_with_count` repair and the exact-head-count wording live in
    exactly one place: this is precisely the sentence pair (identity + "no
    other children" clamp) that measurably fixed identity/clone breaks, so
    the video prompt and the still prompt driving its first frame must never
    disagree on who is on screen.

    Callers must already be past `_shot_prompt`'s two early-return branches
    (insert shots; subject-only shots with no `characters`) before calling
    this — it always resolves `shot.get("characters")` through the cast
    bible, which treats a missing/empty list as "all" (see
    `_cast_for_with_count`) and would misreport a head count for either of
    those shot kinds.

    Returns `(identity_sentence, subject_sentence, count_clamp, action,
    setting, count)`. `action`/`setting` are the REPAIRED text (not the raw
    `shot["action"]`/`shot["setting"]`), already stripped of a trailing
    period, ready for a caller's own action sentence. `subject_sentence`
    introduces the story subject (if any) as its own sentence, meant to sit
    AFTER the identity sentence and BEFORE the "no other children" clamp, so
    the ordering makes unambiguous that the clamp is scoped to children — the
    subject was already named as a non-child a sentence earlier.
    """
    shot_type = shot.get("shot_type", "medium")
    decreep_cfg = (cfg.get("kidsong", {}) or {}).get("decreep", {}) or {}
    decreep_enabled = bool(decreep_cfg.get("enabled", False))
    story_subject = _clean_story_subject(shot.get("story_subject"))

    cast_sentence, count = _cast_for_with_count(shot, song)
    action = str(shot.get("action", "")).rstrip(".")
    setting = str(shot.get("setting", "")).rstrip(".")
    action, setting = _rename_offbible_children(action, setting, shot)
    action, setting = _agree_with_count(
        action, setting, count, _prompt_subject(cast_sentence, shot),
        story_subject=story_subject, decreep=decreep_enabled,
    )
    action = action.rstrip(".")
    setting = setting.rstrip(".")

    if shot_type in ("closeup", "medium") and count == 1:
        identity_sentence = f"A {shot_type} shot of exactly one child: {_cast_body(cast_sentence)}."
        count_clamp = "No other children are visible in frame."
    elif count is not None:
        count_word = _COUNT_WORDS.get(count, str(count))
        noun = "child is" if count == 1 else "children are"
        identity_sentence = (
            f"Exactly {count_word} {noun} on screen: {_cast_body(cast_sentence)}."
        )
        count_clamp = "No other children appear anywhere in the frame, foreground or background."
    else:
        # Count unknown (cast bible unavailable and the shot didn't pin it
        # down itself) — state identity without a head count we can't back up.
        identity_sentence = f"{cast_sentence}."
        count_clamp = ""

    subject_text = _canonical_subject_clause(song, story_subject)
    subject_sentence = (
        f"{subject_text[:1].upper()}{subject_text[1:]} is also in the shot."
        if subject_text else ""
    )
    return identity_sentence, subject_sentence, count_clamp, action, setting, count


def _shot_prompt(shot, song, cfg):
    """Compose the t2v prompt for one shot — the director's control surface.

    Subject-first: the cast/identity clause comes BEFORE the action/setting,
    so the text encoder weights identity early instead of reading it as an
    unanchored appositive after the action. Closeups/medium shots that resolve
    to exactly one character say so explicitly ("...of exactly one child...");
    wide/group (or multi-character) shots state the EXACT head count via
    cast.expected_child_count, targeting the measured failure mode (invented
    extra kids, 67% of identity breaks, concentrated in wide/group shots).

    LTX-2's Gemma text encoder follows natural film-style prose far better
    than keyword lists, so this reads like a shot description, not tags.

    `shot["story_subject"]` (str or None) is the non-child story subject
    visible in this shot — the mouse/lamb/sheep/boat the song is actually
    about — added as its own clause so the text encoder treats it as a real
    on-screen subject rather than incidental words buried in `action`. It is
    NEVER a child and is never folded into the head-count machinery below. A
    shot can legitimately have `characters == []` (or None) when only the
    subject is on screen; that case skips the child-identity machinery
    entirely (see the branch below) rather than resolving the empty list to
    the whole cast, which is what `_cast_for_with_count` would otherwise do.
    """
    # The look — trigger token, prose skeleton and identity clamp — comes from
    # the active render style (pipeline.kidsong.render_style), defaulting to the
    # pixar_toon values so this stays byte-identical to the pre-registry prompt.
    style = resolve_style(cfg)
    trigger = style["style_trigger"]
    skeleton = style["prompt_skeleton"]
    identity_clause = style["identity_clause"]
    # str()/or "": a style is operator-edited config, and a tail that is not a
    # string (or is None/False) must degrade to "no tail" rather than crash the
    # join below with the render already queued.
    prompt_tail = str(style["prompt_tail"] or "")
    shot_type = shot.get("shot_type", "medium")
    camera = shot.get("camera", "static")
    # An empty trigger (e.g. flat_storybook) must not leave a leading ", ".
    style_open = f"{trigger}, {skeleton}" if trigger else skeleton

    # kidsong.decreep.enabled (default False): swaps the blanket "face the
    # camera as they move" sentence (and the contradiction-repair path below)
    # for a per-shot gaze sentence -- see `_gaze_sentence` and the tail of
    # this function. Resolved once, here, so the insert/subject-only branches
    # right below are untouched -- neither ever renders a gaze sentence.
    decreep_cfg = (cfg.get("kidsong", {}) or {}).get("decreep", {}) or {}
    decreep_enabled = bool(decreep_cfg.get("enabled", False))

    # Insert shots are object closeups — no children on screen at all.
    if shot_type == "insert":
        action = str(shot.get("action", "a colorful toothbrush")).rstrip(".")
        setting = str(shot.get("setting", "")).rstrip(".")
        return (
            f"{style_open} "
            f"An insert closeup of an object, no people: {action}, in {setting}. "
            "The camera holds steady. Bold happy colors, warm key light with soft "
            "shadows, a lovingly dressed set kept clear around the object, soft "
            "rounded shapes."
        )

    story_subject = _clean_story_subject(shot.get("story_subject"))

    # Subject-only shots ("the mouse alone climbing the clock") have NO
    # children on screen — modeled on the insert branch above, which already
    # handles "no people". `characters` is [] or None here, so falling
    # through to `_cast_for_with_count` would resolve that to the WHOLE cast
    # bible ensemble (its `["all"]` fallback) and assert a head count of
    # children that is not on screen at all — this branch must come first.
    if not shot.get("characters") and story_subject:
        action = str(shot.get("action", "")).rstrip(".")
        setting = str(shot.get("setting", "")).rstrip(".")
        subject_text = _canonical_subject_clause(song, story_subject)
        subject_cap = subject_text[:1].upper() + subject_text[1:]
        cam_no_people = (
            "The camera holds steady."
            if not camera or camera == "static"
            else f"The camera moves in a {camera}."
        )
        return " ".join(part for part in (
            style_open,
            f"{subject_cap} is the subject of this {shot_type} shot: {action}, in {setting}.",
            "No people are anywhere in the frame, foreground or background.",
            cam_no_people,
            "Bold happy colors, warm key light with soft shadows, a lovingly "
            "dressed set kept clear around the subject, soft rounded shapes.",
        ) if part)

    identity_sentence, subject_sentence, count_clamp, action, setting, count = _identity_block(
        shot, song, cfg
    )

    cam_sentence = (
        "The camera holds steady at the children's eye level."
        if not camera or camera == "static"
        else f"The camera moves in a {camera} at the children's eye level."
    )

    # action_sentence's own "A {shot_type} shot: " prefix (or lack of it) is
    # NOT part of _identity_block's shared middle — _keyframe_prompt builds
    # its own frozen-moment action tail instead — so this mirrors the exact
    # same condition _identity_block branched on, using the count it returned.
    if shot_type in ("closeup", "medium") and count == 1:
        action_sentence = f"{action}, in {setting}."
    else:
        action_sentence = f"A {shot_type} shot: {action}, in {setting}."

    if decreep_enabled:
        look_sentence = (
            "The children have soft rounded shapes and big expressive eyes. "
            f"{_gaze_sentence(shot, story_subject, count)}"
        )
    else:
        # Legacy sentence — byte-identical pin, see tests/test_kidsong_shot_prompt.py
        # and tests/test_kidsong_render_style.py. Appending "face the camera as they
        # move" to EVERY child shot is the verified root cause of the pipeline's
        # signature defect (rendered toddlers blankly staring into the lens);
        # kidsong.decreep.enabled switches to `_gaze_sentence` instead (see above).
        look_sentence = (
            "The children have soft rounded shapes and big expressive eyes and face "
            "the camera as they move."
        )

    return " ".join(part for part in (
        style_open,
        identity_sentence,
        subject_sentence,
        action_sentence,
        count_clamp,
        # The identity clamp — measurably helped, must not regress. Sourced from
        # the active render style; pixar_toon's value is the verbatim sentence.
        identity_clause,
        cam_sentence,
        look_sentence,
        # The lighting/set tail, from the active render style. pixar_toon's
        # value is the verbatim sentence pair this used to hardcode, so the
        # default path stays byte-identical; a style may shorten it (or set it
        # to "") to buy back prompt words — see `render_style.prompt_tail` for
        # the measured length budget and the "concrete props" warning about
        # words the Gemma-3 encoder reads as materials.
        prompt_tail,
    ) if part)


# ------------------------------------------------------- keyframe-first ---
def _keyframe_prompt(shot, song, cfg):
    """Z-Image still prompt for one shot's keyframe-first FIRST FRAME.

    Keyframe-first (`kidsong.keyframe_first.enabled`; see the render loop in
    `_generate_director`) exists because LTX's t2v graph has to GUESS how
    many children are in a shot from text alone — and that guess is exactly
    where clone-collapse (the same child rendered 2-3x) comes from. Z-Image
    is a STILL model: it reads `_identity_block`'s exact-head-count sentence
    far more reliably than a video model does, so rendering the shot's first
    frame with Z-Image — with the correct children already baked in — and
    then animating it with the existing i2v graph fixes the count before LTX
    ever has to guess at it.

    Composition: the active render style's prose skeleton WITHOUT the
    `style_trigger` token (same reasoning as `refs._reference_prompt` — the
    token activates an LTX-specific LoRA and is meaningless to Z-Image's Qwen
    text encoder) + `_identity_block`'s identity/head-count/story-subject
    sentences (the SAME sentences `_shot_prompt` renders, so the video's
    first frame and its own text prompt never disagree on who is on screen)
    + the shot's action rephrased as a FROZEN moment (no camera-move
    phrasing at all — a still has no camera move to describe) + the setting
    + a gaze sentence (see `_gaze_sentence`).

    Group/wide `["all"]` shots are INCLUDED, not just closeups: the exact
    head-count sentence is precisely where Z-Image beats t2v on clones, and a
    wide multi-child shot is where clone-collapse is worst.

    Insert shots (object closeups, no children at all) and subject-only
    shots (`characters` empty, only a non-child `story_subject` on screen)
    skip `_identity_block` entirely, mirroring `_shot_prompt`'s own two
    early-return branches — `_cast_for_with_count` treats a missing/empty
    `characters` as "all" and would otherwise bake a wrong head count into a
    keyframe that has no children in it at all.
    """
    style = resolve_style(cfg)
    skeleton = style["prompt_skeleton"]
    shot_type = shot.get("shot_type", "medium")
    action = str(shot.get("action", "")).rstrip(".")
    setting = str(shot.get("setting", "")).rstrip(".")
    frozen = ", held mid-motion, a single crisp frame, no motion blur"

    if shot_type == "insert":
        action = action or "a colorful toothbrush"
        return (
            f"{skeleton} An insert closeup of an object, no people: "
            f"{action}{frozen}, in {setting}. Bold happy colors, warm key "
            "light with soft shadows, a lovingly dressed set kept clear "
            "around the object, soft rounded shapes."
        )

    story_subject = _clean_story_subject(shot.get("story_subject"))
    if not shot.get("characters") and story_subject:
        subject_text = _canonical_subject_clause(song, story_subject)
        subject_cap = subject_text[:1].upper() + subject_text[1:]
        return " ".join(part for part in (
            skeleton,
            f"{subject_cap} is the subject of this {shot_type} shot: "
            f"{action}{frozen}, in {setting}.",
            "No people are anywhere in the frame, foreground or background.",
            "Bold happy colors, warm key light with soft shadows, a lovingly "
            "dressed set kept clear around the subject, soft rounded shapes.",
        ) if part)

    identity_sentence, subject_sentence, count_clamp, action, setting, count = (
        _identity_block(shot, song, cfg)
    )
    action_sentence = f"{action}{frozen}, in {setting}."
    return " ".join(part for part in (
        skeleton,
        identity_sentence,
        subject_sentence,
        count_clamp,
        action_sentence,
        _gaze_sentence(shot, story_subject, count),
    ) if part)


def _prop_emphasis(prop_text):
    """A prompt clause demanding the shot's interaction object be visibly in
    frame, for re-rendering a keyframe whose prop check failed. Appended only
    after a measured prop miss (the base render stays on the plain prompt),
    mirroring `_identity_emphasis`. Returns "" for an empty prop (-> no-op)."""
    prop_text = str(prop_text or "").strip()
    if not prop_text:
        return ""
    return (
        f" {prop_text} is clearly visible in the frame, held in the "
        "child's hands, fully inside the picture."
    )


def _identity_emphasis(char_id):
    """A prompt clause that reinforces `char_id`'s OWN hair and rules out the
    other cast children's hairstyles, for re-rendering a keyframe that drifted.

    The measured drift is a solo child inheriting a castmate's hair (Kofi ->
    Zuri's afro puffs). A plain klein re-render is stochastic and usually lands
    clean on its own, but a stubborn shot needs the text pushed too: naming the
    target's exact hair (which the reference already carries) and the specific
    OTHER hairstyles to avoid raised a stuck solo Kofi keyframe to 8/8 on-model
    in a live probe. Derived from the bible so it stays correct as the cast
    grows. Returns "" when the character or its hair is unknown (-> no-op)."""
    try:
        from pipeline.kidsong import cast

        c = cast.character(char_id)
        hair = str((c or {}).get("hair") or "").strip()
        if not hair:
            return ""
        others = [
            str((o or {}).get("hair") or "").strip()
            for o in cast.load_bible().get("characters", [])
            if (o or {}).get("id") != char_id and str((o or {}).get("hair") or "").strip()
        ]
        clause = f" The child has exactly {hair}, matching the reference image"
        if others:
            clause += " — not " + ", not ".join(others)
        return clause + "."
    except Exception:
        return ""


def _render_keyframe_on_model(client, cfg, shot, song, kf_path, w, h, ref_images, runlog):
    """Render one shot's keyframe-first still, and — for a single-subject shot
    anchored on a reference — RE-RENDER it until the child on screen matches the
    intended cast member, keeping the best attempt.

    The klein reference-anchored keyframe is a stochastic 4-step sample: most
    renders land on-model, a minority still drift the child into a castmate's
    hair/wardrobe (measured: solo Kofi keyframes drifting into Zuri's afro
    puffs). Because the drift is stochastic, a fresh render at a bumped seed
    usually lands clean, so this loops render -> `identity_check.identity_margin`
    -> re-render while the margin is below `identity_min_margin`, up to
    `identity_retries` times, and keeps the highest-margin still it saw.

    Best-effort and byte-identical when the retry can't apply: a group/insert
    keyframe (no single identity to anchor), the feature off
    (`identity_retries <= 0`), or no detector available (CLIP/refs missing ->
    margin is None) all fall through to exactly one render. Never raises past
    the caller's own guard."""
    from pipeline.kidsong import refs

    prompt = _keyframe_prompt(shot, song, cfg)
    base_seed = int(shot.get("seed") or 0)

    refs.generate_still(client, cfg, prompt, w, h, kf_path, base_seed, ref_images=ref_images)
    if ref_images:
        runlog.info("Shot %s: keyframe anchored on %d cast reference(s).",
                    shot["id"], len(ref_images))

    kf = ((cfg.get("kidsong", {}) or {}).get("keyframe_first", {}) or {}) if isinstance(cfg, dict) else {}
    try:
        retries = int(kf.get("identity_retries", 3))
    except (TypeError, ValueError):
        retries = 3
    try:
        min_margin = float(kf.get("identity_min_margin", 0.0))
    except (TypeError, ValueError):
        min_margin = 0.0
    # PROP GATE (kidsong.keyframe_first.prop_gate, default on; inert without a
    # shot["prop"] or without CLIP): a shot whose story hinges on an
    # interaction OBJECT ("Zuri passes the umbrella…") must show that object
    # in its keyframe BEFORE the shot animates — i2v carries frame 0 forward,
    # so a prop missing here is a prop missing for the whole take (the
    # measured vanishing-umbrella defect).
    prop_gate = bool(kf.get("prop_gate", True))
    try:
        prop_min_prob = float(kf.get("prop_min_prob", 0.6))
    except (TypeError, ValueError):
        prop_min_prob = 0.6
    prop_text = str(shot.get("prop") or "").strip() if prop_gate else ""

    if retries <= 0 or (not ref_images and not prop_text):
        return

    # Only a single-subject shot has ONE identity to verify (the same resolver
    # the prompt and the ref-images list use); group/insert keyframes run the
    # prop check alone.
    char_id = None
    try:
        from pipeline.kidsong import cast, identity_check

        if ref_images:
            resolved = cast._resolve_with_fallback(shot.get("characters"), shot.get("shot_type"))
            if len(resolved) == 1:
                char_id = resolved[0].get("id")
    except Exception:
        return

    def _score(path):
        mg = identity_check.identity_margin(path, char_id, cfg) if char_id else None
        pp = identity_check.prop_presence(path, prop_text, cfg) if prop_text else None
        identity_ok = mg is None or mg >= min_margin
        prop_ok = pp is None or pp >= prop_min_prob
        # Identity outranks prop (never sacrificed): ordering key is
        # (identity_ok, prop_ok, margin, prop_prob), None scored lowest.
        key = (
            identity_ok, prop_ok,
            mg if mg is not None else float("-inf"),
            pp if pp is not None else float("-inf"),
        )
        return mg, pp, identity_ok and prop_ok, key

    margin, prop_prob, ok, best_key = _score(kf_path)
    if margin is None and prop_prob is None:
        return                     # no detector at all — leave the render as-is
    if ok:
        return

    # Re-renders push the TEXT as well as the seed: name the child's own hair
    # (identity drift) and demand the object in-frame (prop miss) so a stubborn
    # failure converges. The base render (attempt 0 above) stays on the plain
    # prompt, so the common clean path is unchanged.
    retry_prompt = prompt
    if char_id and margin is not None and margin < min_margin:
        retry_prompt += _identity_emphasis(char_id)
    if prop_text and prop_prob is not None and prop_prob < prop_min_prob:
        retry_prompt += _prop_emphasis(prop_text)

    for attempt in range(1, retries + 1):
        tmp = f"{kf_path}.try{attempt}.png"
        try:
            refs.generate_still(client, cfg, retry_prompt, w, h, tmp, base_seed + attempt,
                                ref_images=ref_images)
        except Exception as exc:
            runlog.warning("Shot %s: keyframe re-render %d failed (%s) — keeping best.",
                           shot["id"], attempt, exc)
            break
        mg, pp, cand_ok, cand_key = _score(tmp)
        if mg is None and pp is None:
            _safe_remove(tmp)
            break
        if cand_key > best_key:
            best_key = cand_key
            margin, prop_prob = mg, pp
            os.replace(tmp, kf_path)   # promote the better still
        else:
            _safe_remove(tmp)
        if best_key[0] and best_key[1]:
            runlog.info(
                "Shot %s: keyframe on-model after %d re-render(s) "
                "(margin %s, prop %s).", shot["id"], attempt,
                "%+.3f" % margin if margin is not None else "n/a",
                "%.2f" % prop_prob if prop_prob is not None else "n/a",
            )
            return

    if not (best_key[0] and best_key[1]):
        runlog.warning(
            "Shot %s: keyframe still failing gates after %d retries "
            "(margin %s, prop %s) — keeping the closest match.",
            shot["id"], retries,
            "%+.3f" % margin if margin is not None else "n/a",
            "%.2f" % prop_prob if prop_prob is not None else "n/a",
        )


def _safe_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _bootstrap_cast_references(client, cfg, runlog):
    """Make sure every cast child has a canonical identity reference on disk
    before the keyframe render loop runs. Never raises.

    The identity anchor (``keyframe_first.identity_refs`` klein keyframes and
    ``reference_anchor`` i2v anchoring) reads each child's reference PNG via
    ``refs.reference_for``; when none exists the anchor silently no-ops. That
    is *exactly* how the feature shipped ENABLED yet never once fired
    (IMP-005/006: the references were never created, so there was nothing to
    anchor on). ``refs.ensure_reference`` is idempotent and cheap-first — an
    existing or harvested reference is returned untouched, only a genuinely
    missing one costs a one-time Z-Image Turbo still — so on every run after the
    first this is a no-op.

    Best-effort by contract: any failure (no client, ComfyUI hiccup, bible
    problem) is swallowed and the keyframe simply renders text-only, exactly as
    before. Gated on the same switches the anchor itself reads, so a config that
    wants no references does no work here."""
    ks = (cfg.get("kidsong", {}) or {}) if isinstance(cfg, dict) else {}
    kf = (ks.get("keyframe_first", {}) or {})
    # Only the klein KEYFRAME anchor (keyframe_first + identity_refs) needs the
    # references pre-rendered here; it is the one consumer that reads them for
    # EVERY single-subject shot up front. reference_anchor's i2v path only fires
    # on a reviewer's force_i2v and can lazily harvest, so it does not warrant
    # generating three stills at the top of every render.
    if not (bool(kf.get("enabled", False)) and bool(kf.get("identity_refs", False))):
        return
    try:
        from pipeline.kidsong import cast as _cast, refs as _refs

        for _c in _cast.load_bible().get("characters", []):
            cid = (_c or {}).get("id")
            if not cid:
                continue
            if _refs.ensure_reference(client, cfg, cid) and runlog is not None:
                runlog.stage("cast_reference", char=cid)
    except Exception as exc:  # pragma: no cover - defensive net
        if runlog is not None:
            runlog.warning(
                "Cast reference bootstrap failed (%s) — keyframes render "
                "text-only this run.", exc,
            )


def _keyframe_ref_images(shot, cfg):
    """The shot's cast members' canonical reference PNGs, for Z-Image's
    identity-anchor inputs (see refs.generate_still's `ref_images`).

    Resolves the shot's `characters` through the SAME cast resolver the
    prompt uses (aliases and all) and collects each child's persisted
    reference (refs.reference_for). Best-effort by contract: any resolution
    problem, a subject-only/insert shot, or simply no references on disk
    returns [] — the keyframe then renders text-only exactly as before, so
    this can never block or degrade an episode. Measured motivation (job 22):
    text alone let "Zuri" drift visibly lighter-skinned in one keyframe; a
    pixel-level anchor pins the identity.
    """
    try:
        # Default OFF (kidsong.keyframe_first.identity_refs): three live probes
        # (2026-07-22, output/_anchor_validation/) showed the z_image_turbo
        # checkpoint treats TextEncodeZImageOmni's image inputs as a spatial
        # COLLAGE source, not identity guidance — refs render as side-by-side
        # panels and the identity does not follow (a Kofi ref produced a
        # Zuri-styled child, three probes in a row, explicit "same boy from
        # image 1" phrasing included). The wiring stays (tested, and a future
        # Omni-capable checkpoint may honor it); text-only keyframes are the
        # proven default (the _ab_decreep arm B ran text-only and held count
        # and wardrobe cleanly).
        ks_local = (cfg.get("kidsong", {}) or {}) if isinstance(cfg, dict) else {}
        if not (ks_local.get("keyframe_first", {}) or {}).get("identity_refs", False):
            return []

        from pipeline.kidsong import cast, refs as refs_mod

        characters = shot.get("characters")
        if characters == []:  # subject-only: no children to anchor
            return []
        resolved = cast._resolve_with_fallback(characters, shot.get("shot_type"))
        paths = []
        for c in resolved[:3]:
            p = refs_mod.reference_for(cfg, c.get("id"))
            if p:
                paths.append(p)
        return paths
    except Exception:
        return []


def _keyframe_path(shots_dir, shot, w, h):
    """Where a shot's Z-Image keyframe-first still lives on disk:
    ``<shots_dir>/keyframes/<shot id>_<w>x<h>.png``. The dimensions are part
    of the filename so a `kidsong.shot.width`/`.height` change never reuses a
    keyframe rendered at the wrong size. A pure path calculation only — an
    existing non-empty file at this path is reused untouched (resume-safe by
    construction, see the render loop's batch phase); runstate.py is never
    involved.
    """
    return os.path.join(shots_dir, "keyframes", f"{shot['id']}_{w}x{h}.png")


def _fit_keyframe(path, w, h, suffix="_fit"):
    """Cover-fit a conditioning PNG into exactly `(w, h)`, writing
    `<path-without-ext><suffix>.png` beside it and returning that path.

    Two callers: (1) a keyframe whose dims drifted from the shot's current
    render size (default suffix, byte-identical to before the param existed);
    (2) the reference-card fallback/force_i2v paths, which pass a dims-keyed
    suffix — a raw 768x1024 PORTRAIT card staged into an 896x512 landscape
    i2v latent gets stretched by the graph (measured: oversized foggy face on
    the take's first frames), so the card must be cover-fit to the exact shot
    dims first, and the dims-keyed name keeps fitted copies from colliding
    when shot dims change between runs.
    """
    from PIL import Image, ImageOps

    stem, ext = os.path.splitext(path)
    fitted = f"{stem}{suffix}{ext}"
    with Image.open(path) as im:
        ImageOps.fit(im, (w, h)).save(fitted)
    return fitted


def _fit_reference_card(path, w, h):
    """The reference-card variant of `_fit_keyframe`: cover-fit `path` to the
    exact shot dims under a dims-keyed name, returning the original path (a
    best-effort no-op) when fitting fails — a fallback anchor must never kill
    the attempt over an unreadable PNG."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            if im.size == (w, h):
                return path
        return _fit_keyframe(path, w, h, suffix=f"_fit_{w}x{h}")
    except Exception as exc:
        log = logging.getLogger("kidsong")
        log.warning("could not fit reference card %s to %dx%d (%s) — staging raw.",
                    path, w, h, exc)
        return path


def _shot_dims(cfg):
    """`(width, height)` for one rendered shot, from `kidsong.shot` in config.

    THE CONFIG IS AUTHORITATIVE. This exists because it did not used to be:
    `workflows/*.json` carried a hardcoded 512x896 (the old vertical Shorts
    format) and nothing ever patched WIDTH/HEIGHT, so `kidsong.shot.width` /
    `.height` were dead keys and every shot rendered PORTRAIT. The editor then
    cover-cropped that portrait frame into the 16:9 output and threw away ~69%
    of the picture height — which is precisely the "heads and feet cut off"
    defect. Defaults here are the landscape channel format, so a missing config
    key can never resurrect the portrait render.

    LTX requires both dimensions to be a multiple of 32 (ComfyUI's
    EmptyLTXVLatentVideo declares step=32 and builds its latent as
    `height // 32, width // 32`, so a non-multiple is silently truncated to a
    latent that decodes at the wrong size). We round DOWN to the nearest
    multiple of 32 rather than raise: a config typo must not block a render,
    and the rounded value is still a valid frame.
    """
    shot_cfg = (cfg.get("kidsong", {}) or {}).get("shot", {}) or {}
    width = int(shot_cfg.get("width", 896))
    height = int(shot_cfg.get("height", 512))

    def snap(value, name):
        snapped = max(64, (value // 32) * 32)
        if snapped != value:
            log.warning(
                "kidsong.shot.%s = %s is not a multiple of 32 (LTX requires it) "
                "— rounding down to %s.", name, value, snapped,
            )
        return snapped

    return snap(width, "width"), snap(height, "height")


def _baseline_negative(client, workflow_name):
    """The workflow's authored baseline NEGATIVE text, read straight off the
    graph before any patch is applied — per-shot negatives are additive on top
    of this, never a replacement for the hardened baseline another agent
    maintains in workflows/*.json."""
    try:
        wf = client.load_workflow(workflow_name)
        for node in wf.values():
            if node.get("_meta", {}).get("title") == "NEGATIVE":
                return str(node.get("inputs", {}).get("text", ""))
    except Exception as exc:
        log.warning(
            "Could not read the baseline NEGATIVE text from %s (%s) — "
            "shots will render with no negative prompt patched.",
            workflow_name, exc,
        )
    return ""


def _shot_negative(shot, cfg, baseline):
    """`baseline` (the workflow's authored negative) plus this shot's
    character-specific negatives (e.g. a Kofi shot negates 'yellow shirt',
    a Nala shot negates 'purple dress'), deduped case-insensitively against
    what's already in the baseline. Degrades to the plain baseline if the
    cast bible is unavailable — a lookup problem here must never block a
    render."""
    try:
        from pipeline.kidsong import cast

        terms = cast.negative_terms(
            shot.get("characters"), shot_type=shot.get("shot_type")
        )
    except Exception as exc:
        log.warning(
            "Shot %s: cast bible unavailable for negatives (%s) — using the "
            "baseline negative only.",
            shot.get("id"), exc,
        )
        terms = []

    existing_lower = baseline.lower()
    extra = [t for t in terms if t and t.lower() not in existing_lower]
    if not extra:
        return baseline
    if not baseline:
        return ", ".join(extra)
    sep = " " if baseline.rstrip().endswith(",") else ", "
    return f"{baseline}{sep}{', '.join(extra)}"


def _retry_seed(characters, base_seed, attempt_index, shot_type=None):
    """Seed for a retry attempt: stays within the character's seed family
    (cast.seed_for) instead of drifting away from it, so a shot that needs a
    second take still varies its seed per attempt while remaining anchored to
    the character it's rendering. `attempt_index` is 1 for the first retry, 2
    for the second, etc. Falls back to the old `+= 977` behaviour (relative to
    `base_seed`) if the cast bible is unavailable — still deterministic, just
    not character-keyed."""
    try:
        from pipeline.kidsong import cast

        return cast.seed_for(
            characters, base_seed + 977 * attempt_index, shot_type=shot_type
        )
    except Exception as exc:
        log.warning(
            "cast bible unavailable for retry seeding (%s) — falling back to "
            "base_seed + 977 * attempt.",
            exc,
        )
        return base_seed + 977 * attempt_index


# ----------------------------------------------------- reference anchoring ---
def _reference_for_single_subject(shot, cfg):
    """The canonical reference PNG anchoring a single-subject shot's one child,
    or ``None``.

    Phase 3 (pixel anchoring): the reviewer's dead ``force_i2v`` hint — emitted
    when it sees duplicate/off-model children — is finally consumed by
    re-rendering the shot image-to-video FROM this reference. That only makes
    sense when the shot has exactly ONE correct child to anchor: a single still
    can pin ONE identity, so ONLY single-subject shots escalate. A wide with
    three kids has no single correct image to feed and is deliberately left to
    the existing t2v retry path.

    Returns ``None`` — leaving ``force_i2v`` inert exactly as before — unless
    ALL of:
      * ``kidsong.reference_anchor.enabled`` is on (default true);
      * the shot resolves to exactly one cast child
        (``cast.expected_child_count == 1``, same resolver the prompt uses);
      * a reference image for that child actually exists on disk.

    So with the feature off, or with no reference rendered yet, behaviour is
    byte-identical to today. Never raises — a lookup problem must not block a
    render; it just means no escalation.
    """
    ks = (cfg.get("kidsong", {}) or {}) if isinstance(cfg, dict) else {}
    if not (ks.get("reference_anchor", {}) or {}).get("enabled", True):
        return None
    try:
        from pipeline.kidsong import cast, refs

        names = shot.get("characters")
        shot_type = shot.get("shot_type")
        if cast.expected_child_count(names, shot_type=shot_type) != 1:
            return None
        resolved = cast._resolve_with_fallback(names, shot_type=shot_type)
        if len(resolved) != 1:
            return None
        return refs.reference_for(cfg, resolved[0]["id"])
    except Exception as exc:
        log.warning(
            "Shot %s: reference lookup failed (%s) — staying on t2v.",
            shot.get("id"), exc,
        )
        return None


# --------------------------------------------------------------- quality tiers ---
#: The knobs a quality tier bundles, expressed as OVERRIDES on the base kidsong
#: config: a tier only changes the keys it names, everything else is inherited.
#:
#:  * "draft" trades fidelity for throughput — no 2x hires refine pass, fewer
#:    frames per shot, fewer unique shots, and automatic heuristic review (no
#:    blocking vision-review waits).
#:  * "final" forces the full-fidelity hires pass and otherwise keeps whatever
#:    the base config already says (frames, shot count, reviewer) — "current".
#:
#: These are the shipped defaults; ``kidsong.quality_tiers`` in config overrides
#: them per tier and per key. ``kidsong.quality`` picks the active tier; the
#: literal "custom" (or an absent/unknown value) means "use the individual keys
#: exactly as-is" — the regression-safe path that is byte-identical to today.
_QUALITY_TIER_DEFAULTS = {
    "draft": {
        "hires_pass": False,
        "frames_scale": 0.6,
        "max_unique_shots": 12,
        "reviewer": "heuristic",
    },
    "final": {
        "hires_pass": True,
    },
}

#: kidsong.quality values meaning "use the individual keys as-is" (no tier).
_QUALITY_PASSTHROUGH = ("custom",)


def _quality_overrides(cfg):
    """`(tier_name, overrides)` for the active ``kidsong.quality``.

    ``overrides`` is the set of knob overrides the tier imposes on the base
    config (only the keys it changes). For "custom", an absent value, or an
    unknown tier name it is ``{}`` — no overrides, so the effective config is
    the base config unchanged. An unknown name is LOGGED and treated as "custom"
    rather than guessed at, so a config typo can never silently downgrade
    quality. The tier's shipped defaults are overlaid with any per-tier config
    in ``kidsong.quality_tiers`` (config wins, ``_`` comment keys ignored)."""
    ks = (cfg or {}).get("kidsong", {}) if isinstance(cfg, dict) else {}
    ks = ks or {}
    raw = ks.get("quality")
    if raw is None:
        return "custom", {}
    tier = str(raw).strip().lower()
    if tier in _QUALITY_PASSTHROUGH:
        return "custom", {}
    tiers = ks.get("quality_tiers") or {}
    entry = tiers.get(tier)
    if entry is None and tier not in _QUALITY_TIER_DEFAULTS:
        log.warning(
            "kidsong.quality=%r is not a known tier (known: %s) — using the "
            "individual keys as-is (custom).",
            tier, ", ".join(sorted(set(_QUALITY_TIER_DEFAULTS) | set(tiers))) or "none",
        )
        return "custom", {}
    overrides = dict(_QUALITY_TIER_DEFAULTS.get(tier, {}))
    if isinstance(entry, dict):
        overrides.update({k: v for k, v in entry.items() if not str(k).startswith("_")})
    return tier, overrides


def _read_quality_settings(cfg):
    """The effective, resolved values of the tier-controlled knobs in ``cfg``,
    read with the SAME defaults the pipeline itself uses so they mirror what a
    render will actually do (see ``workflow_name``, the per-shot frame math, the
    director planner and ``review.get_reviewer``)."""
    ks = (cfg or {}).get("kidsong", {}) if isinstance(cfg, dict) else {}
    ks = ks or {}
    shot = ks.get("shot", {}) or {}
    review = ks.get("review", {}) or {}
    director = ks.get("director", {}) or {}
    return {
        "hires_pass": bool(shot.get("hires_pass", True)),
        "frames_scale": float(shot.get("frames_scale", 1.0)),
        "max_frames": int(shot.get("max_frames", 241)),
        "max_unique_shots": int(director.get("max_unique_shots", 16)),
        "reviewer": str(review.get("reviewer", review.get("backend", "heuristic"))).lower(),
    }


def resolve_quality(cfg):
    """Resolve ``kidsong.quality`` into ``(effective_cfg, tier_name, settings)``.

    ``settings`` is the resolved value of every tier-controlled knob
    (hires_pass, frames_scale, max_frames, max_unique_shots, reviewer) as the
    render will actually use it. Precedence: an explicit "draft"/"final" tier
    OVERRIDES the individual keys; "custom", an absent value, or an unknown name
    leaves the individual keys exactly as-is.

    REGRESSION GUARANTEE: for the custom/absent/unknown path ``effective_cfg``
    is the SAME object as ``cfg`` (no copy, no mutation), so today's behaviour
    is byte-identical. Only an active tier deep-copies and overlays, and it
    never mutates the caller's config."""
    tier, overrides = _quality_overrides(cfg)
    if not overrides:
        return cfg, tier, _read_quality_settings(cfg)

    eff = copy.deepcopy(cfg)
    ks = eff.setdefault("kidsong", {})
    shot = ks.setdefault("shot", {})
    review = ks.setdefault("review", {})
    director = ks.setdefault("director", {})
    if "hires_pass" in overrides:
        shot["hires_pass"] = overrides["hires_pass"]
    if "frames_scale" in overrides:
        shot["frames_scale"] = overrides["frames_scale"]
    if "max_frames" in overrides:
        shot["max_frames"] = overrides["max_frames"]
    if "max_unique_shots" in overrides:
        director["max_unique_shots"] = overrides["max_unique_shots"]
    if "reviewer" in overrides:
        review["reviewer"] = overrides["reviewer"]
    return eff, tier, _read_quality_settings(eff)


# ------------------------------------------------------- sing coherence gate ---
# ACE-Step's 5Hz "thinking" LM replans bpm/arrangement on every render
# (ace_runner.py: thinking=True, use_cot_metas=True) — the seed only pins the
# DiT noise, not what the LM decides to sing, so a scrambled render is a dice
# roll, not a deterministic bug. `edit.beat_grid`'s phase-fit resultant (the
# circular concentration of the beat grid's phase: 1.0 = rock steady, 0.0 =
# scattered) is a cheap, objective proxy for "did this come out coherent".
# Measured 2026-07-24: resultant 0.54 at 99.38 bpm sang clean; 0.235 at 129bpm
# and 0.068 at 86bpm were audibly scrambled.
_SING_COHERENCE_DEFAULTS = {"min_resultant": 0.25, "max_attempts": 3}


def _sing_coherence_settings(cfg):
    """Resolve kidsong.sing_coherence into (min_resultant, max_attempts)."""
    sc = (cfg.get("kidsong", {}) or {}).get("sing_coherence", {}) or {}
    min_resultant = float(sc.get("min_resultant", _SING_COHERENCE_DEFAULTS["min_resultant"]))
    max_attempts = int(sc.get("max_attempts", _SING_COHERENCE_DEFAULTS["max_attempts"]))
    return min_resultant, max(1, max_attempts)


def _measure_sing_coherence(audio_path, duration, runlog=None):
    """The beat grid's phase-fit resultant for a freshly sung wav, or None if
    it could not be measured.

    Reuses `edit.beat_grid`/`edit.complete_beat_grid` rather than
    re-implementing the phase fit — see edit.py's `_fit_phase`, which is the
    ONE place that number is computed. Always runs fresh (cache=False): every
    re-sing attempt writes to the SAME `audio_path` (sing_song always renders
    to its out_path), so a cached `<audio_path>.beats.json` sidecar would
    describe a DIFFERENT rendition than whatever currently sits on disk —
    beat_grid's cache key is the path alone, not the content.
    """
    from pipeline.kidsong.edit import beat_grid

    try:
        result = beat_grid(audio_path, cache=False, duration=duration)
    except Exception as exc:
        if runlog is not None:
            runlog.warning("Sing coherence: beat_grid failed on %s (%s)", audio_path, exc)
        return None
    resultant = (result.get("grid") or {}).get("resultant")
    return float(resultant) if resultant is not None else None


def _sing_with_coherence_gate(song, cfg, voice_path, runlog, step):
    """Sing the song, re-singing with a bumped seed if the render's beat-phase
    coherence is too scrambled, and keep whichever attempt scored highest.

    Up to `kidsong.sing_coherence.max_attempts` TOTAL attempts (default 3),
    seeded `base_seed`, `base_seed + 1`, ... Stops early the moment an attempt
    clears `kidsong.sing_coherence.min_resultant` (default 0.25) — no point
    spending more GPU time once a good take exists. If NONE clear the bar, the
    best-scoring attempt still ships (loudly logged) rather than blocking the
    episode.

    Every render lands at `voice_path` in turn (sing_song always writes its
    out_path); a render that turns out not to be the final winner is moved
    (never deleted — channel rule: generated media is inventory) to
    `<voice_path>.rejected-a<N>.wav` before the next attempt overwrites
    `voice_path`.
    """
    from pipeline.kidsong.sing import sing_song

    min_resultant, max_attempts = _sing_coherence_settings(cfg)
    base_seed = int((cfg.get("kidsong", {}) or {}).get("seed", 20260717))

    attempts = []  # [{"attempt", "seed", "resultant", "vocals", "parked_path"}]
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            # Park the previous attempt's render (still sitting at voice_path,
            # since nothing has touched it since it was sung) before sing_song
            # overwrites voice_path with this attempt — a plain rename, so
            # nothing is duplicated or lost.
            prev = attempts[-1]
            parked = f"{voice_path}.rejected-a{prev['attempt']}.wav"
            os.replace(voice_path, parked)
            prev["parked_path"] = parked

        seed = base_seed + (attempt - 1)
        attempt_cfg = cfg
        if attempt > 1:
            attempt_cfg = copy.deepcopy(cfg)
            attempt_cfg.setdefault("kidsong", {})["seed"] = seed
        if max_attempts > 1:
            step(f"  Singing attempt {attempt}/{max_attempts} (seed {seed})…")

        vocals = sing_song(song, attempt_cfg, voice_path, runlog=runlog)
        resultant = _measure_sing_coherence(voice_path, vocals.get("duration"), runlog)
        runlog.stage(
            "sing_coherence", attempt=attempt, seed=seed,
            resultant=resultant, min_resultant=min_resultant,
        )
        attempts.append({
            "attempt": attempt, "seed": seed, "resultant": resultant,
            "vocals": vocals, "parked_path": None,
        })
        if resultant is not None and resultant >= min_resultant:
            break

    best = max(attempts, key=lambda a: a["resultant"] if a["resultant"] is not None else -1.0)
    last = attempts[-1]
    if best is not last:
        # The winner is an earlier, already-parked attempt; the LAST attempt
        # (currently sitting at voice_path) lost this round — park it too,
        # then bring the winner's render back to voice_path.
        parked = f"{voice_path}.rejected-a{last['attempt']}.wav"
        os.replace(voice_path, parked)
        last["parked_path"] = parked
        os.replace(best["parked_path"], voice_path)
        best["parked_path"] = None

    if best["resultant"] is None or best["resultant"] < min_resultant:
        runlog.warning(
            "Sing coherence gate: no attempt reached resultant >= %.3f after "
            "%d attempt(s) — shipping the best available (attempt %d, seed "
            "%d, resultant=%s).",
            min_resultant, len(attempts), best["attempt"], best["seed"], best["resultant"],
        )
        step(
            f"  WARNING: sung audio may be scrambled (best resultant "
            f"{best['resultant']}, threshold {min_resultant}) after "
            f"{len(attempts)} attempt(s) — shipping the best available "
            f"(seed {best['seed']})."
        )
    elif len(attempts) > 1:
        step(
            f"  Sing coherence OK on attempt {best['attempt']} "
            f"(resultant={best['resultant']:.3f}, seed={best['seed']})."
        )

    return best["vocals"]


def _generate_director(song, base, do_upload, cfg, step, out_dir, topic=None,
                       resume=False, runlog=None):
    """Director mode: shot-planned, QC-reviewed, beat-cut ComfyUI renders."""
    # Resolve the quality tier FIRST so every downstream knob — workflow
    # selection (hires_pass), per-shot frames, the director's shot count and the
    # review mode — reads the tier's effective config. custom/absent returns
    # `cfg` unchanged (same object), so that path stays byte-identical to today.
    cfg, quality_tier, quality_settings = resolve_quality(cfg)
    ks = cfg.get("kidsong", {})
    review_log_entries = []  # {"stage": "script"|"shotlist"|"cut", "attempt", "verdict"}
    if runlog is None:
        runlog = RunLog(out_dir, base=base, stream=False)
    # Make the run's quality visible in output/<base>.log from the first line.
    runlog.stage(
        "quality", tier=quality_tier,
        hires_pass=quality_settings["hires_pass"],
        frames_scale=quality_settings["frames_scale"],
        max_frames=quality_settings["max_frames"],
        max_unique_shots=quality_settings["max_unique_shots"],
        reviewer=quality_settings["reviewer"],
    )
    # QM: output/<base>.log (where the "quality" stage line above lands) is
    # gitignored and never reaches the cloud quality-manager audit, but
    # review_log.json does (it's the one telemetry file explicitly allow-
    # listed for that purpose). Without this entry, an auditor working only
    # from the git checkout has no way to tell a real 0%-vision-coverage
    # defect apart from a deliberate heuristic-only dev run, short of
    # reverse-engineering commit timestamps against config.example.json
    # history — which is exactly how this gap was found (2026-08-01 audit).
    _review_cfg = ks.get("review", {}) or {}
    review_log_entries.append({
        "run_config": {
            "quality_tier": quality_tier,
            "reviewer": quality_settings["reviewer"],
            "min_vision_coverage": _review_cfg.get("min_vision_coverage"),
            "shot_auto_accept_enabled": bool(
                (_review_cfg.get("shot_auto_accept") or {}).get("enabled")
            ),
        }
    })
    step(
        f"Quality tier '{quality_tier}': hires_pass={quality_settings['hires_pass']}, "
        f"frames_scale={quality_settings['frames_scale']}, "
        f"max_frames={quality_settings['max_frames']}, "
        f"max_unique_shots={quality_settings['max_unique_shots']}, "
        f"review={quality_settings['reviewer']}."
    )

    # 1b. Script QC — catch a bad song before any GPU work happens (cheap, CPU-only).
    # Skipped on resume: the script was already vetted and its verdict is spent
    # GPU-free work we do not need to repeat, and regenerating it here would
    # invalidate every take already on disk.
    if not resume and ks.get("review", {}).get("script", True):
        runlog.stage("script_qc")
        from pipeline.kidsong.lyrics import generate_song
        from pipeline.kidsong.script_qc import review_script

        step("QC: reviewing the song script…")
        v_script = review_script(song, cfg)
        review_log_entries.append({"stage": "script", "attempt": 0, "verdict": v_script})
        if not v_script.get("accept") and v_script.get("retry_hints", {}).get("regenerate"):
            step(f"  Script rejected ({', '.join(v_script.get('reasons', []))}) — regenerating…")
            song2 = generate_song(topic, cfg)
            v_script2 = review_script(song2, cfg)
            review_log_entries.append({"stage": "script", "attempt": 1, "verdict": v_script2})
            if v_script2.get("score", 0.0) >= v_script.get("score", 0.0):
                song, v_script = song2, v_script2
            if not v_script.get("accept"):
                step(
                    f"  Script still rejected after retry "
                    f"({', '.join(v_script.get('reasons', []))}) — proceeding with best available."
                )

    # 2. The song (ACE-Step sings; edge-tts fallback).
    voice_path = os.path.join(out_dir, base + ".wav")
    sung = False
    # A resumed run must reuse the *same* audio: the shot ledger's start/end
    # times and the beat grid are derived from it, so re-singing would desync
    # every take already rendered.
    reuse_audio = resume and os.path.exists(voice_path) and os.path.getsize(voice_path) > 0
    if reuse_audio:
        runlog.stage("sing", reused=voice_path)
        step("Reusing the sung audio from the interrupted run…")
        sung = True
        vocals = {"duration": _wav_duration(voice_path)}
    elif resume:
        # A run that reached Final/ deletes its wav on the way out, so "no
        # audio" usually means "already finished", not "corrupt".
        if os.path.exists(os.path.join(out_dir, "Final", base + ".mp4")):
            raise FileNotFoundError(
                f"Cannot resume {base}: this run already completed — its video "
                f"is in Final/{base}.mp4 and its working audio was cleaned up. "
                "There is nothing left to render."
            )
        raise FileNotFoundError(
            f"Cannot resume {base}: the sung audio {voice_path} is missing. "
            "The run's shot timings and beat grid are derived from it; without "
            "it the existing takes cannot be cut."
        )
    if not sung and ks.get("singer", "ace") == "ace":
        runlog.stage("sing", singer="ace")
        # Free ComfyUI's resident weights first: a previous video's renders
        # leave the server holding models, and ACE-Step's audio fidelity
        # degrades measurably when squeezed (tiled VAE decode at ~4.7GB free).
        # Singing happens before any rendering, so the GPU should be all ours.
        try:
            from pipeline.kidsong.comfy import ComfyClient

            ComfyClient(cfg).free()
        except Exception:
            pass  # server down/not started yet — nothing to free
        step("Singing the song (ACE-Step on GPU)…")
        try:
            vocals = _sing_with_coherence_gate(song, cfg, voice_path, runlog, step)
            sung = True
        except Exception as e:
            step(f"ACE-Step unavailable ({e}) — falling back to edge-tts.")
    verse_times = None
    if not sung:
        step("Recording the child voice (edge-tts)…")
        from pipeline.kidsong.song_audio import synthesize_vocals

        vocals = synthesize_vocals(song["verses"], cfg, voice_path)
        verse_times = vocals["verse_times"]
    duration = min(vocals["duration"], cfg["video"]["max_seconds"])

    # 3. Words, verse times, beat grid (all CPU).
    runlog.stage("whisper", audio=voice_path, duration=round(duration, 3))
    step("Aligning word-level captions (Whisper)…")
    from pipeline.captions import get_word_timestamps

    words = get_word_timestamps(
        voice_path,
        cfg,
        uppercase=cfg["captions"].get("uppercase", True),
        lyrics=song["verses"],
        duration=duration,
        language=(cfg.get("kidsong") or {}).get("language", "en"),
    )
    words = [w for w in words if w["start"] < duration]
    for w in words:
        w["end"] = min(w["end"], duration)
    if verse_times is None:
        from pipeline.kidsong.sing import verse_times_from_words

        verse_times = verse_times_from_words(words, song["verses"], duration)

    runlog.stage("beats")
    step("Detecting the beat grid…")
    from pipeline.kidsong.edit import assemble, beat_grid, build_cut_list

    beats = beat_grid(voice_path)

    # 4. Direction: plan the shot list. On resume the ledger on disk IS the
    # plan — re-planning would renumber shots and orphan every existing take.
    resumed_shotlist = runstate.load_shotlist(out_dir, base) if resume else None
    if resumed_shotlist and resumed_shotlist.get("shots"):
        shotlist = resumed_shotlist
        recovered = runstate.recover_takes(shotlist, runstate.shots_dir_path(out_dir, base))
        runlog.stage(
            "shotlist", source="resumed",
            shots=len(shotlist["shots"]), adopted_from_disk=recovered,
        )
        if recovered:
            step(f"Adopted {recovered} take(s) already on disk from the interrupted run.")
    else:
        if resume:
            raise FileNotFoundError(
                f"Cannot resume {base}: "
                f"{runstate.shots_json_path(out_dir, base)} is missing or has no shots."
            )
        runlog.stage("shotlist", source="planned")
        # Storyboard brain (kidsong.storyboard.enabled, default off): attach the
        # gated per-verse story beats BEFORE planning so `_fallback_planner`
        # consumes them (see director._storyboard_beats_for_verse) — and before
        # the song.json write below, so a resumed run replans from the same
        # board instead of rolling new beats. Resume never rebuilds: the loaded
        # song.json already carries the board (or legitimately doesn't).
        # build_storyboard never raises (its own LLM/gate/fallback chain), but
        # this stays belt-and-braces — a storyboard problem must degrade to
        # today's planner, never kill the episode.
        if (ks.get("storyboard", {}) or {}).get("enabled", False) and not song.get("storyboard"):
            step("Storyboarding: building the per-verse action arc…")
            try:
                from pipeline.kidsong.storyboard import build_storyboard

                song["storyboard"] = build_storyboard(song, cfg)
                runlog.stage("storyboard", verses=len(song["storyboard"].get("verses", [])))
            except Exception as exc:  # pragma: no cover - defensive net
                runlog.warning("Storyboard build failed (%s) — planning without it.", exc)
        step("Directing: planning the shot list…")
        from pipeline.kidsong.director import plan_shots

        shotlist = plan_shots(song, verse_times, beats, cfg)

    # 4b. Shotlist QC — catch a broken plan before ComfyUI renders a single frame.
    if not resume and ks.get("review", {}).get("script", True):
        from pipeline.kidsong.script_qc import review_shotlist

        step("QC: reviewing the shot list…")
        v_shotlist = review_shotlist(shotlist, song, cfg)
        review_log_entries.append({"stage": "shotlist", "attempt": 0, "verdict": v_shotlist})
        if not v_shotlist.get("accept") and v_shotlist.get("retry_hints", {}).get("replan"):
            step(f"  Shot list rejected ({', '.join(v_shotlist.get('reasons', []))}) — replanning…")
            replan_cfg = copy.deepcopy(cfg)
            replan_cfg.setdefault("kidsong", {})["seed"] = int(ks.get("seed", 20260717)) + 1
            shotlist2 = plan_shots(song, verse_times, beats, replan_cfg)
            v_shotlist2 = review_shotlist(shotlist2, song, cfg)
            review_log_entries.append({"stage": "shotlist", "attempt": 1, "verdict": v_shotlist2})
            if v_shotlist2.get("score", 0.0) >= v_shotlist.get("score", 0.0):
                shotlist, v_shotlist = shotlist2, v_shotlist2
            if not v_shotlist.get("accept"):
                step(
                    f"  Shot list still rejected after replan "
                    f"({', '.join(v_shotlist.get('reasons', []))}) — proceeding with best available."
                )

    # The shot list is the run's resume ledger, rewritten after every shot —
    # save_shotlist writes it atomically so a crash mid-write cannot leave
    # truncated JSON and make the run unresumable.
    runstate.save_shotlist(out_dir, base, shotlist)
    # Persist the full song (lyrics/cast) too — repairs like re-singing the
    # audio need it, and gate request files get overwritten per run.
    #
    # Atomically, for the reason runstate.save_shotlist states one line above:
    # a crash during the write leaves truncated JSON and makes the run
    # unresumable. The shotlist next to it has been written atomically since
    # runstate existed; this file, which resume needs just as much, was not.
    atomic_write_json(
        runstate.song_json_path(out_dir, base), song, indent=2, ensure_ascii=False
    )

    # 5. Render unique shots via ComfyUI, with the producer's QC loop.
    from pipeline.kidsong.comfy import ComfyClient, ComfyUnreachableError
    from pipeline.kidsong.review import (
        contact_sheet, get_reviewer, review_shots_log, vision_coverage,
    )
    from pipeline.kidsong.video_backend import make_video_backend

    shots_dir = os.path.join(out_dir, base + "-shots")
    review_dir = os.path.join(shots_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    reviewer = get_reviewer(cfg)
    max_retries = int(ks.get("review", {}).get("max_retries_per_shot", 2))
    max_attempts = int(ks.get("review", {}).get("max_attempts_per_shot",
                                                runstate.DEFAULT_MAX_ATTEMPTS))
    # Contract 1 (infra-retry): a ComfyUI/network hiccup (timeout, dropped
    # connection, a dead/wedged server) is not a QC verdict on the take -- it
    # never even finished rendering one. Retrying it must NOT burn a QC
    # attempt (max_retries/max_attempts above); it gets its own small budget,
    # bounded so a genuinely dead server still eventually aborts the shot
    # instead of retrying forever.
    comfy_infra_retries = int(ks.get("review", {}).get("comfy_infra_retries", 3))
    # The render style decides the look: its workflow_base (+ _hires when
    # kidsong.shot.hires_pass) selects the graph, and its style_lora/style_strength
    # patch the LORA_STYLE node per render below. Unknown/absent style names fall
    # back to pixar_toon (logged) inside resolve_style — never a crash.
    render_style = resolve_style(cfg)
    workflow = workflow_name(cfg, render_style)

    # Phase 3 reference anchoring: a shot the reviewer rejects for duplicate/
    # off-model children (force_i2v) is re-rendered image-to-video FROM the
    # cast member's canonical reference — but ONLY when it is a single-subject
    # shot and a reference exists (see _reference_for_single_subject). The i2v
    # graph is a DIFFERENT file from the t2v one: _apply_patches raises KeyError
    # on a missing title, so the workflow name and the INPUT_IMAGE/SIGMAS patches
    # must switch together (they do, below). Like the t2v graph, the i2v graph
    # follows kidsong.shot.hires_pass — with keyframe_first/reference_anchor on,
    # EVERY shot renders i2v, so without this the configured 2x refine silently
    # never ran and episodes shipped at the soft 896x512 base resolution.
    i2v_workflow = i2v_workflow_name(cfg)
    # i2v conditions only frame 0, so it MAY survive the channel's 3-step
    # distilled schedule — but that is UNVERIFIED until the GPU proof, so we
    # default to the same official 9-step schedule flf2v needed (3 steps
    # collapse the interior there). Override with kidsong.shot.i2v_sigmas; the
    # SIGMAS node is titled in the i2v graph, patched as a dict below.
    from pipeline.kidsong import transitions as _transitions

    i2v_sigmas = str(
        (ks.get("shot", {}) or {}).get("i2v_sigmas")
        or _transitions.SIGMAS_OFFICIAL_9STEP
    )

    renders = {}
    log_entries = []
    review_log_path = os.path.join(review_dir, "review_log.json")

    def _flush_review_log():
        """Persist buffered per-take verdicts and clear the buffer.

        Called after every shot (and on the way out of a crash) so the verdict
        history survives a hard kill — and so the next resume can read it back
        via runstate.load_take_verdicts to pick the best take.
        """
        if not log_entries:
            return
        try:
            review_shots_log(review_log_path, list(log_entries))
        except OSError as exc:
            runlog.warning("Could not write %s: %s", review_log_path, exc)
        else:
            log_entries.clear()

    unique = [s for s in shotlist["shots"] if not s.get("reuse_of")]

    # Work out what is actually left to do BEFORE touching the GPU, so a resume
    # with nothing pending never starts ComfyUI at all.
    pending = []
    for shot in unique:
        if runstate.shot_is_done(shot, shots_dir):
            renders[shot["id"]] = shot["take"]
            runlog.info("Shot %s: reusing existing take %s", shot["id"], shot["take"])
        elif runstate.shot_is_exhausted(shot, max_attempts):
            # Bounded failure: a shot that has already burned its attempts is
            # never retried again, so a resume loop cannot run forever.
            runlog.warning(
                "Shot %s: exhausted after %s attempt(s) — not retrying (%s)",
                shot["id"], shot.get("attempts", 0), shot.get("failure_reason", "no reason recorded"),
            )
            # It is exhausted, not satisfied — but if QC-rejected media exists
            # it still covers the slot, which beats duplicating another shot.
            fallback = shot.get("take")
            if fallback and os.path.isfile(fallback) and os.path.getsize(fallback) > 0:
                renders[shot["id"]] = fallback
                runlog.warning(
                    "Shot %s: covering its slot with UNAPPROVED take %s (verdict=%s)",
                    shot["id"], fallback, shot.get("verdict", "unknown"),
                )
        else:
            pending.append(shot)

    runlog.stage(
        "render", total=len(unique), reusing=len(renders),
        pending=len(pending), workflow=workflow,
    )
    if renders:
        step(f"Reusing {len(renders)} take(s); {len(pending)} shot(s) still to render.")

    client = ComfyClient(cfg)
    # The VIDEO backend for the shot-render loop below only — stills (the
    # baseline-negative read, the run fingerprint, cast reference bootstrap,
    # and keyframe-first renders, all just below) stay on `client` no matter
    # what kidsong.render_backend says. Default/absent config returns
    # `client` itself (see make_video_backend's docstring), so this line
    # changes nothing about the byte-identical default path.
    renderer = make_video_backend(cfg, client)
    # Read once — same workflow graph for every shot this run, so its
    # authored NEGATIVE baseline only needs reading off disk once.
    baseline_negative = _baseline_negative(client, workflow)
    if pending:
        client.ensure_up()
        runlog.fingerprint(
            cfg, comfy_client=client,
            autostarted=getattr(client, "_proc", None) is not None,
        )

        # Bootstrap the cast identity references BEFORE any keyframe renders,
        # so the anchor is never starved (see _bootstrap_cast_references).
        _bootstrap_cast_references(client, cfg, runlog)

        # Keyframe-first rendering (kidsong.keyframe_first.enabled, default
        # False = byte-identical). LTX's t2v graph has to GUESS how many
        # children are in a shot from text alone — exactly where
        # clone-collapse (the same child rendered 2-3x) comes from. Fix:
        # pre-render each pending shot's FIRST FRAME with Z-Image Turbo (a
        # still model that reads _keyframe_prompt's exact head-count
        # sentence far more reliably), ALL AT ONCE here, before the per-shot
        # render loop below ever stages one onto the i2v graph — see the
        # keyframe-staging block inside that loop, right before the
        # reference_anchor.always block.
        if bool((ks.get("keyframe_first", {}) or {}).get("enabled", False)):
            from pipeline.kidsong import refs

            kf_w, kf_h = _shot_dims(cfg)
            for kf_shot in pending:
                if kf_shot.get("reuse_of"):
                    continue
                kf_path = _keyframe_path(shots_dir, kf_shot, kf_w, kf_h)
                if os.path.isfile(kf_path) and os.path.getsize(kf_path) > 0:
                    continue
                try:
                    ref_images = _keyframe_ref_images(kf_shot, cfg)
                    _render_keyframe_on_model(
                        client, cfg, kf_shot, song, kf_path, kf_w, kf_h,
                        ref_images, runlog,
                    )
                except Exception as exc:
                    runlog.warning(
                        "Shot %s: keyframe-first still failed (%s) — this "
                        "shot will render t2v instead.", kf_shot["id"], exc,
                    )
            # Z-Image must fully unload before the 22B LTX model loads —
            # both resident at once will not fit in 12GB of VRAM.
            client.free()

    in_flight = None  # the shot being rendered when a crash lands, if any
    try:
        for i, shot in enumerate(pending):
            step(f"Rendering shot {shot['id']} ({i + 1}/{len(pending)}) via ComfyUI…")
            in_flight = shot
            shot_started = time.time()
            config_base_seed = int(ks.get("seed", 20260717))
            seed = int(shot.get("seed", config_base_seed))
            shot_negative = _shot_negative(shot, cfg, baseline_negative)
            original_action = shot.get("action", "")
            best = None  # (score, path)
            # Never overwrite media: on resume, takes continue at the next free
            # index rather than clobbering a0 (channel-inventory rule).
            first_attempt = runstate.next_attempt_index(shots_dir, shot["id"])
            attempts_before = int(shot.get("attempts", 0) or 0)

            # Render exactly as many frames as the shot's timeline slot needs
            # (LTX wants 8n+1) so cuts never loop-restart mid-shot. Capped at
            # max_frames (241 ≈ 10 s, the model's long-clip limit).
            shot_cfg = ks.get("shot", {})
            shot_w, shot_h = _shot_dims(cfg)
            fps = int(shot_cfg.get("fps", 24))
            max_frames = int(shot_cfg.get("max_frames", 241))
            # A quality tier can render fewer frames per shot for a faster draft
            # (frames_scale < 1). Default 1.0 is a no-op, so this stays
            # byte-identical for the custom/absent/final path.
            frames_scale = float(shot_cfg.get("frames_scale", 1.0))
            slot = float(shot.get("end", 0)) - float(shot.get("start", 0))
            frames = int(slot * fps) + fps // 4  # small handle for the editor
            if frames_scale != 1.0:
                frames = int(frames * frames_scale)
            frames = max(49, min(max_frames, (frames // 8) * 8 + 1))
            # `shot` stays the ledger entry we write status back into; the
            # prompt may be rendered from a simplified copy on retry.
            render_shot = shot
            accepted = False
            attempt = first_attempt
            # Reference-anchor escalation state for THIS shot (Phase 3). Set
            # once a rejected verdict carries force_i2v AND a single-subject
            # reference is staged; from then on the shot's remaining attempts
            # render image-to-video from that reference.
            escalate_i2v = False
            i2v_staged = None
            # The LOCAL SOURCE file behind i2v_staged (keyframe or reference
            # PNG). The infra-retry loop below calls client.free() between
            # retries, which deletes every staged input — retrying the same
            # patches then 400s on LoadImage ("Invalid image file", measured
            # live on job 22's first attempt). Keeping the source lets the
            # retry RE-STAGE before re-rendering.
            i2v_source = None
            # Keyframe-first (kidsong.keyframe_first.enabled, default off).
            # The batch phase above already rendered this shot's first frame
            # with Z-Image Turbo, if it succeeded (a missing file just means
            # that render failed, or the feature is off — either way this
            # shot falls straight through to t2v below, exactly as before).
            # Staged here at attempt 0, so every retry in the loop below
            # reuses the SAME staged keyframe with the existing seed bump —
            # nothing further down the loop touches i2v_staged/escalate_i2v
            # once they are set, so this needs no changes to the retry loop.
            # KEYFRAME OUTRANKS THE REFERENCE CARD. The scene keyframe is
            # identity-verified (klein refs + the CLIP identity/prop retry
            # loop) AND carries the shot's prop, setting and composition; the
            # canonical card is a bare portrait with none of them. Animating a
            # solo shot from the raw card was the measured root of the
            # appearance-loss / vanishing-prop / hazy-face / strobe-reject
            # cluster (pixar-full episode: EVERY single-child take strobe-
            # rejected while morphing card -> scene), so the card is now only
            # the FALLBACK when no usable keyframe exists — and then it is
            # cover-fit to the exact shot dims first (a raw 768x1024 portrait
            # stretched into the 896x512 latent was the hazy-face source).
            i2v_source_kind = None
            if bool((ks.get("keyframe_first", {}) or {}).get("enabled", False)):
                kf_path = _keyframe_path(shots_dir, shot, shot_w, shot_h)
                if os.path.isfile(kf_path) and os.path.getsize(kf_path) > 0:
                    try:
                        from PIL import Image

                        with Image.open(kf_path) as im:
                            fits = im.size == (shot_w, shot_h)
                        staged_src = kf_path if fits else _fit_keyframe(kf_path, shot_w, shot_h)
                        i2v_staged = renderer.stage_input_image(staged_src)
                        i2v_source = staged_src
                        escalate_i2v = True
                        i2v_source_kind = "keyframe"
                        runlog.info(
                            "Shot %s: keyframe-first — animating from Z-Image "
                            "keyframe %s", shot["id"], staged_src,
                        )
                    except Exception as exc:
                        runlog.warning(
                            "Shot %s: could not stage keyframe %s (%s) — "
                            "trying the reference-card fallback.",
                            shot["id"], kf_path, exc,
                        )
            if not escalate_i2v and (ks.get("reference_anchor", {}) or {}).get("always"):
                single_ref = _reference_for_single_subject(shot, cfg)
                if single_ref:
                    try:
                        fitted_ref = _fit_reference_card(single_ref, shot_w, shot_h)
                        i2v_staged = renderer.stage_input_image(fitted_ref)
                        i2v_source = fitted_ref
                        escalate_i2v = True
                        i2v_source_kind = "reference"
                        runlog.info(
                            "Shot %s: no usable keyframe — anchoring on the "
                            "fitted cast reference %s (fallback).",
                            shot["id"], fitted_ref,
                        )
                    except Exception as exc:
                        runlog.warning(
                            "Shot %s: could not stage reference %s (%s) — "
                            "staying on t2v.", shot["id"], single_ref, exc,
                        )
            # (The former second "proactive reference_anchor.always" block was
            # a near-duplicate of the fallback above and staged the RAW,
            # unfitted card — merged into the single fitted fallback.)
            for n in range(max_retries + 1):
                attempt = first_attempt + n
                out_path = os.path.join(shots_dir, f"{shot['id']}_a{attempt}.mp4")
                attempt_started = time.time()
                runlog.info(
                    "Shot %s attempt %s: rendering %sx%s, %s frames, seed %s -> %s",
                    shot["id"], attempt, shot_w, shot_h, frames, seed, out_path,
                )
                base_patches = {
                    "PROMPT": {"text": _shot_prompt(render_shot, song, cfg)},
                    "NEGATIVE": {"text": shot_negative},
                    "SEED": {"noise_seed": seed},
                    # WIDTH/HEIGHT are patched per render so kidsong.shot
                    # actually governs the frame. Without these the graph's
                    # own baked-in values win — which is how every shot came
                    # out portrait and got cropped to pieces in the 16:9 cut.
                    "WIDTH": {"value": shot_w},
                    "HEIGHT": {"value": shot_h},
                    "FRAMES": {"value": frames},
                    # The active render style's LoRA file + strength. The dict
                    # form writes both inputs on the LoraLoaderModelOnly|LORA_STYLE
                    # node (comfy.ComfyClient._apply_patches). This is what lets a
                    # style neutralize the LoRA (strength 0.0) to reuse the same
                    # graph for a different look with no new workflow file.
                    "LORA_STYLE": {
                        "lora_name": render_style["style_lora"],
                        "strength_model": render_style["style_strength"],
                    },
                    "FILENAME_PREFIX": {"filename_prefix": f"kidsong/{base}/{shot['id']}"},
                }
                # Reference anchoring: once force_i2v fired for this shot and a
                # single-subject reference was staged, render image-to-video FROM
                # that reference (the i2v graph is a different file, so the
                # workflow name and the INPUT_IMAGE/SIGMAS patches switch
                # together). Otherwise this is the exact same single t2v render as
                # before — byte-identical.
                if escalate_i2v and i2v_staged:
                    active_workflow = i2v_workflow
                    active_patches = dict(base_patches)
                    active_patches["INPUT_IMAGE"] = {"image": i2v_staged}
                    active_patches["SIGMAS"] = {"sigmas": i2v_sigmas}
                    runlog.info(
                        "Shot %s attempt %s: image-to-video (%s) anchored on the "
                        "cast reference", shot["id"], attempt, i2v_workflow,
                    )
                else:
                    active_workflow = workflow
                    active_patches = base_patches
                # Infra-retry loop (Contract 1): a ComfyUI hang/dead-server/
                # network blip aborts only the render, not the whole episode.
                # This retries the SAME attempt (out_path, seed, patches all
                # unchanged) up to comfy_infra_retries times BEFORE falling
                # through to the QC review below -- n / max_retries_per_shot
                # is never touched here, so an infra hiccup never costs the
                # shot a QC re-roll.
                infra_attempt = 0
                infra_backoff = (5, 15, 45)
                while True:
                    try:
                        renderer.render(active_workflow, active_patches, out_path)
                        break
                    except (TimeoutError, RuntimeError, requests.RequestException) as exc:
                        if infra_attempt >= comfy_infra_retries:
                            raise
                        wait_s = infra_backoff[min(infra_attempt, len(infra_backoff) - 1)]
                        infra_attempt += 1
                        runlog.info(
                            "Shot %s attempt %s: infra failure (%s) — retry %d/%d "
                            "after VRAM free", shot["id"], attempt, exc,
                            infra_attempt, comfy_infra_retries,
                        )
                        renderer.free()
                        if isinstance(exc, ComfyUnreachableError):
                            try:
                                renderer.restart_if_hung()
                            except Exception as restart_exc:
                                runlog.warning(
                                    "Shot %s attempt %s: restart_if_hung failed "
                                    "(%s) — still retrying the render.",
                                    shot["id"], attempt, restart_exc,
                                )
                        # client.free() above just deleted every staged input —
                        # an i2v render's retry would 400 on LoadImage with the
                        # now-missing name ("Invalid image file", measured live
                        # on job 22). Re-stage from the kept local source and
                        # rewrite the patch before retrying. A re-stage failure
                        # degrades this retry to t2v rather than looping 400s.
                        if escalate_i2v and i2v_source:
                            try:
                                i2v_staged = renderer.stage_input_image(i2v_source)
                                active_patches["INPUT_IMAGE"] = {"image": i2v_staged}
                                runlog.info(
                                    "Shot %s attempt %s: re-staged %s after the "
                                    "VRAM free.", shot["id"], attempt, i2v_source,
                                )
                            except Exception as restage_exc:
                                runlog.warning(
                                    "Shot %s attempt %s: could not re-stage %s "
                                    "(%s) — degrading this retry to t2v.",
                                    shot["id"], attempt, i2v_source, restage_exc,
                                )
                                escalate_i2v = False
                                active_workflow = workflow
                                active_patches = dict(base_patches)
                        time.sleep(wait_s)
                runlog.info(
                    "Shot %s attempt %s: render finished in %.1fs",
                    shot["id"], attempt, time.time() - attempt_started,
                )
                try:
                    # External reviewer: requests/sheets go into review_dir
                    # (watched by the supervising session's monitor).
                    verdict = reviewer.review(render_shot, out_path, out_dir=review_dir)
                except TypeError:
                    verdict = reviewer.review(render_shot, out_path)
                contact_sheet(render_shot, out_path, review_dir)
                log_entries.append(
                    {
                        "shot": shot["id"],
                        "attempt": attempt,
                        "seed": seed,
                        "verdict": verdict,
                        # Telemetry only (not part of the verdict contract): lets an
                        # audit tell whether an i2v-anchored take's accept came from
                        # a real vision verdict or an unattended heuristic fallback
                        # — the anchor mechanism (IMP-005) is otherwise unverifiable
                        # from review_log.json alone.
                        "i2v_anchored": bool(escalate_i2v and i2v_staged),
                        # What the animation was conditioned on: "keyframe"
                        # (the verified scene still) vs "reference" (the
                        # fitted card fallback) vs None (t2v). Additive —
                        # audits read this to attribute drift per source.
                        "i2v_source_kind": i2v_source_kind,
                    }
                )
                score = float(verdict.get("score", 0.0))
                runlog.info(
                    "Shot %s attempt %s: verdict accept=%s score=%.3f%s",
                    shot["id"], attempt, bool(verdict.get("accept")), score,
                    " reasons=" + "; ".join(verdict.get("reasons", []))
                    if verdict.get("reasons") else "",
                )
                if best is None or score > best[0]:
                    best = (score, out_path)
                if verdict.get("accept"):
                    accepted = True
                    break
                step(
                    f"  Producer rejected {shot['id']} "
                    f"({', '.join(verdict.get('reasons', []))}) — retrying…"
                )
                seed = _retry_seed(
                    shot.get("characters"), config_base_seed, n + 1,
                    shot_type=shot.get("shot_type"),
                )
                if verdict.get("retry_hints", {}).get("simplify_action"):
                    render_shot = dict(shot, action=original_action.split(",")[0])
                # Consume force_i2v (previously read by nobody): the reviewer saw
                # duplicate/off-model children. When this is a single-subject shot
                # with a reference on disk, stage that reference ONCE and anchor
                # the remaining attempts to it via i2v. Staged copies are cleaned
                # up by client.free() in the loop's finally. With the feature off
                # or no reference present, _reference_for_single_subject returns
                # None and force_i2v stays inert — byte-identical to before.
                if (verdict.get("retry_hints", {}).get("force_i2v")
                        and not escalate_i2v):
                    reference = _reference_for_single_subject(shot, cfg)
                    if reference:
                        try:
                            reference = _fit_reference_card(reference, shot_w, shot_h)
                            i2v_staged = renderer.stage_input_image(reference)
                            i2v_source = reference
                            escalate_i2v = True
                            i2v_source_kind = "reference"
                            step(
                                f"  {shot['id']}: duplicate/off-model children — "
                                "re-rendering image-to-video anchored on the cast "
                                "reference."
                            )
                            runlog.info(
                                "Shot %s: force_i2v honored — anchoring on "
                                "reference %s", shot["id"], reference,
                            )
                        except Exception as exc:
                            runlog.warning(
                                "Shot %s: could not stage reference %s (%s) — "
                                "staying on t2v.", shot["id"], reference, exc,
                            )
                    else:
                        # A single reference anchors ONE child; wide/group shots
                        # have no single correct image, and the feature-off /
                        # no-reference paths leave force_i2v inert exactly as
                        # before. Those stay on the existing t2v retry path.
                        runlog.info(
                            "Shot %s: force_i2v emitted but no single-subject "
                            "reference is available — staying on t2v.",
                            shot["id"],
                        )
            # The best take always covers the slot so the episode is never
            # blocked — but "usable for the cut" is NOT "accepted by QC", and
            # only the latter may satisfy the shot in the ledger.
            renders[shot["id"]] = best[1]

            # Commit this shot to the ledger immediately: if the very next shot
            # kills the run, everything up to here resumes without re-rendering.
            total_attempts = attempts_before + (attempt - first_attempt) + 1
            runstate.mark_rendered(
                shot, best[1], score=best[0], attempts=total_attempts,
                verdict=runstate.VERDICT_ACCEPTED if accepted else runstate.VERDICT_REJECTED,
            )
            if not accepted:
                # Rejected: the shot stays pending so a resume renders it again,
                # unless it has burned its attempt budget — then it is failed,
                # not silently "rendered", and no resume retries it forever.
                if runstate.shot_is_exhausted(shot, max_attempts):
                    runstate.mark_failed(
                        shot,
                        f"QC rejected all {total_attempts} take(s); "
                        f"best score {best[0]:.3f} used to cover the slot",
                        attempts=total_attempts,
                    )
                    runlog.warning(
                        "Shot %s REJECTED on every one of %s attempt(s) — marked FAILED. "
                        "Covering its slot with the best take %s (score %.3f); "
                        "this shot ships UNAPPROVED footage.",
                        shot["id"], total_attempts, best[1], best[0],
                    )
                    step(
                        f"  WARNING: {shot['id']} was rejected on every attempt and has "
                        f"exhausted its budget — the cut uses its best take anyway."
                    )
                else:
                    runlog.warning(
                        "Shot %s REJECTED on all %s attempt(s) so far — staying PENDING "
                        "for the next resume. Best take %s (score %.3f) covers the slot "
                        "in this cut only.",
                        shot["id"], total_attempts, best[1], best[0],
                    )
                    step(
                        f"  WARNING: {shot['id']} was rejected on every attempt — "
                        f"a resume will re-render it."
                    )
            runstate.save_shotlist(out_dir, base, shotlist)
            # Flush the verdicts for this shot now. Buffering them to the end of
            # the loop meant a hard kill (or a run still in flight) lost the
            # entire verdict history — several runs on disk have contact sheets
            # but no review_log.json for exactly this reason.
            _flush_review_log()
            runlog.info(
                "Shot %s DONE in %.1fs — take=%s accepted=%s attempts=%s status=%s",
                shot["id"], time.time() - shot_started, best[1], accepted,
                total_attempts, shot.get("status"),
            )
            in_flight = None
    except BaseException as exc:
        # Charge the failure to the shot we died on, so a shot that reliably
        # kills the run is eventually marked failed instead of being retried by
        # every future resume forever.
        if in_flight is not None:
            burned = int(in_flight.get("attempts", 0) or 0) + 1
            if burned >= max_attempts:
                runstate.mark_failed(
                    in_flight, f"{type(exc).__name__}: {exc}", attempts=burned
                )
                runlog.error(
                    "Shot %s marked FAILED after %s attempt(s) — resume will skip it.",
                    in_flight.get("id"), burned,
                )
            else:
                in_flight["attempts"] = burned
                in_flight["failure_reason"] = f"{type(exc).__name__}: {exc}"[:500]
        # Record where we died so --list-resumable can show it, then let the
        # caller's handler log the traceback.
        runlog.error(
            "Render loop aborted: %s: %s — %s/%s shot(s) usable, ledger saved.",
            type(exc).__name__, exc, len(renders), len(unique),
        )
        try:
            runstate.save_shotlist(out_dir, base, shotlist)
        except OSError:
            pass
        raise
    finally:
        client.free()
        _flush_review_log()

    if not renders:
        raise RuntimeError(
            f"No usable takes for {base}: every shot either failed or was "
            "exhausted, so there is nothing to cut."
        )

    # Chorus/reuse slots point at their source shot's accepted render.
    for s in shotlist["shots"]:
        if s.get("reuse_of"):
            renders[s["id"]] = renders.get(s["reuse_of"]) or next(iter(renders.values()))

    # A shot that exhausted its attempts has no take of its own. Rather than
    # fail the whole run, cover its slot with another render and say so loudly —
    # a video with one repeated shot is reviewable, a crash here is not.
    covered = next(iter(renders.values()))
    for s in shotlist["shots"]:
        if s["id"] not in renders:
            renders[s["id"]] = covered
            runlog.warning(
                "Shot %s has no usable take (status=%s) — covering its slot with %s",
                s["id"], s.get("status"), covered,
            )
            step(f"  WARNING: {s['id']} never rendered — its slot reuses another shot.")

    # 6. Edit: beat-aligned cut with captions. Assembled first to a STAGING path;
    # only promoted to Final/ once (or if) the cut QC gate accepts it —
    # intermediates (shots, review sheets, a rejected cut's staging file) stay
    # next to them in out_dir.
    runlog.stage("cut", shots=len(shotlist["shots"]), beats=len(beats or []))
    step("Editing: beat-aligned cut…")
    cut_list = build_cut_list(shotlist, beats, duration, cfg)

    # Optional real lip-sync (kidsong.lipsync.enabled, default FALSE). Runs AFTER
    # build_cut_list because a shot's audio slice is defined by its FINAL cut
    # window; on success it swaps the closeup/medium cuts' takes for lip-synced
    # ones (edit.assemble then uses them). Off = byte-identical to today: apply()
    # returns cut_list/renders unchanged and never imports the heavy worker. Any
    # failure inside is caught and the un-synced takes ship — lip-sync must never
    # block an episode. Writes only under output/_lipsync/<base>/ (a NEW dir).
    def _maybe_lipsync(cl):
        if not (ks.get("lipsync", {}) or {}).get("enabled", False):
            return cl
        nonlocal renders
        from pipeline.kidsong import lipsync

        runlog.stage("lipsync", shots=len(shotlist["shots"]))
        step("Lip-syncing prominent-face shots (Wav2Lip)…")
        build_dir = os.path.join(out_dir, "_lipsync", base)
        try:
            cl, renders = lipsync.apply(
                cfg, cl, renders, shotlist, voice_path, build_dir, on_progress=step,
            )
        except Exception as exc:
            runlog.warning("Lip-sync step failed (%s) — shipping un-synced takes.", exc)
            step(f"  Lip-sync failed ({exc}) — shipping un-synced takes.")
        return cl

    cut_list = _maybe_lipsync(cut_list)
    final_dir = os.path.join(out_dir, "Final")
    os.makedirs(final_dir, exist_ok=True)

    def _promote(src):
        """Move `src` into Final/ without ever replacing an existing episode.

        A resume re-cuts the same <base>, so the obvious target is precisely
        the name an earlier — possibly already approved and uploaded — cut
        occupies. Generated media is channel inventory: mint the next version
        instead, and tell the caller which name was actually written.
        """
        target = runstate.reserve_output_path(
            os.path.join(final_dir, base + ".mp4"),
            # prepend_intro writes <video>.intro.json next to the episode.
            companions=(".intro.json",),
        )
        if os.path.basename(target) != base + ".mp4":
            runlog.warning(
                "Final/%s.mp4 already exists — promoting to %s instead so the "
                "existing episode is left untouched.",
                base, os.path.basename(target),
            )
            step(f"  NOTE: Final/{base}.mp4 exists — writing {os.path.basename(target)}.")
        os.replace(src, target)
        return target

    def _vision_gate_check():
        """QM-010 promote gate: refuse to ship a cut whose takes were mostly
        never looked at by the vision reviewer (identity/wardrobe/scary-face/
        staging rubric) — the heuristic fallback (`review.HeuristicReviewer`)
        cannot see any of that, so an unattended run defaulting to it is not
        a safe substitute for QC. Returns True iff promotion may proceed.

        Disabled (always True, no computation) when
        `kidsong.review.min_vision_coverage` is 0 (opt-out) or the run's
        reviewer is configured as "heuristic" outright — in that case there
        IS no vision path by configuration, so gating on it would refuse
        every episode. Coverage is computed over the takes actually used in
        this cut (`renders`, one entry per shot including reuse_of/covered
        fallbacks), not merely the takes that were rendered.

        A gate entry (additive, `review_log.json`-shaped) is always appended
        to `review_log_entries` when the check actually runs, so the outcome
        is visible on disk even for a passing run.
        """
        review_cfg = ks.get("review", {}) or {}
        threshold = float(review_cfg.get("min_vision_coverage", 0.8))
        backend = str(
            review_cfg.get("reviewer", review_cfg.get("backend", "heuristic"))
        ).lower()
        if threshold <= 0 or backend == "heuristic":
            return True

        accepted_takes = [
            os.path.splitext(os.path.basename(p))[0] for p in renders.values()
        ]
        coverage, uncovered = vision_coverage(review_dir, accepted_takes)
        review_log_entries.append({
            "gate": "vision_coverage",
            "coverage": coverage,
            "threshold": threshold,
            "uncovered": uncovered,
            "promoted": coverage >= threshold,
        })
        if coverage >= threshold:
            return True

        runlog.error(
            "QC GATE: %s vision coverage %.1f%% is below the required %.1f%% -- "
            "NOT promoting to Final/. Uncovered shot/take(s) (%d): %s. Recovery: "
            "run the auto-reviewer or answer the pending review queue (responses "
            "backfill onto disk), then `--resume %s` or `--finalize %s` to "
            "recompute coverage from disk and promote.",
            base, coverage * 100, threshold * 100, len(uncovered),
            ", ".join(uncovered) or "(none)", base, base,
        )
        step(
            f"  QC GATE: vision coverage {coverage:.0%} is below the required "
            f"{threshold:.0%} — NOT promoting. Uncovered: "
            f"{', '.join(uncovered) or '(none)'}. Answer the review queue, then "
            f"--resume/--finalize {base} to recompute and promote."
        )
        return False

    def _group_vision_gate_check():
        """R2's other half: multi-child takes must never ship sight-unseen.

        The heuristic gate cannot see clone-collapse, invented non-cast
        children or identity at all — and those defects CONCENTRATE in group
        shots (measured: every group take of the pixar-full episode shipped
        score-1.0 with zero inspection). So every take used in the cut whose
        shot stages >1 expected children needs an ACCEPTING vision response
        on disk (`vision_coverage` on the group subset — backfilled responses
        count, same recovery path as the coverage gate). Config
        `kidsong.review.require_group_vision` (default ON); skipped when the
        reviewer is configured "heuristic" (no vision path exists then, same
        reasoning as `_vision_gate_check`). With story staging there are only
        ~2-3 group shots per episode, so this is a small, focused ask.
        """
        review_cfg = ks.get("review", {}) or {}
        if not bool(review_cfg.get("require_group_vision", True)):
            return True
        backend = str(
            review_cfg.get("reviewer", review_cfg.get("backend", "heuristic"))
        ).lower()
        if backend == "heuristic":
            return True

        by_id = {s["id"]: s for s in shotlist["shots"]}
        group_takes = []
        for shot_id, path in renders.items():
            shot = by_id.get(shot_id) or {}
            try:
                from pipeline.kidsong import cast as _cast

                expected = _cast.expected_child_count(
                    shot.get("characters"), shot.get("shot_type")
                )
            except Exception:
                expected = None
            if expected is not None and int(expected) > 1:
                group_takes.append(os.path.splitext(os.path.basename(path))[0])

        coverage, uncovered = vision_coverage(review_dir, group_takes)
        review_log_entries.append({
            "gate": "group_vision",
            "group_takes": len(group_takes),
            "uncovered": uncovered,
            "promoted": not uncovered,
        })
        if not uncovered:
            return True
        runlog.error(
            "QC GATE: %d group take(s) were never vision-approved (%s) -- NOT "
            "promoting to Final/. Recovery: answer the review queue (or run the "
            "auto-reviewer), then `--resume %s`/`--finalize %s`.",
            len(uncovered), ", ".join(uncovered), base, base,
        )
        step(
            f"  QC GATE: {len(uncovered)} group take(s) unreviewed "
            f"({', '.join(uncovered)}) — NOT promoting. Answer the queue, then "
            f"--resume/--finalize {base}."
        )
        return False

    def _approval_gate_check():
        """`kidsong.review.require_approved` (default OFF): when on, a cut
        containing a slot covered by a take QC never accepted (an exhausted
        shot's best REJECTED take — generate's "best take covers the slot"
        behavior) is assembled for review but NOT promoted. The owner's
        quality bar: rejects must never silently ship in a Final.
        """
        review_cfg = ks.get("review", {}) or {}
        if not bool(review_cfg.get("require_approved", False)):
            return True
        by_id = {s["id"]: s for s in shotlist["shots"]}
        unapproved = sorted(
            shot_id for shot_id in renders
            if str((by_id.get(shot_id) or {}).get("verdict") or "")
            != runstate.VERDICT_ACCEPTED
            and not (by_id.get(shot_id) or {}).get("reuse_of")
        )
        review_log_entries.append({
            "gate": "require_approved",
            "unapproved": unapproved,
            "promoted": not unapproved,
        })
        if not unapproved:
            return True
        runlog.error(
            "QC GATE: %d shot slot(s) are covered by UNAPPROVED takes (%s) -- "
            "NOT promoting to Final/ (kidsong.review.require_approved). "
            "Recovery: `--resume %s` re-renders rejected non-exhausted shots; "
            "exhausted ones need a manual take swap or a config change.",
            len(unapproved), ", ".join(unapproved), base,
        )
        step(
            f"  QC GATE: {len(unapproved)} slot(s) ship unapproved takes "
            f"({', '.join(unapproved)}) — NOT promoting (require_approved)."
        )
        return False

    def _promote_gates_pass():
        """All promote gates, evaluated WITHOUT short-circuiting so every
        gate's entry lands in review_log.json even when an earlier one already
        failed — an audit must see the full picture, not the first refusal."""
        results = [
            _vision_gate_check(),
            _group_vision_gate_check(),
            _approval_gate_check(),
        ]
        return all(results)

    # Staging is generated media too: a previous attempt's rejected cut is kept
    # for review, so this never writes over one.
    staging_path = runstate.reserve_output_path(os.path.join(out_dir, base + ".staging.mp4"))
    assemble(cfg, cut_list, renders, voice_path, words, duration, staging_path, on_progress=step)

    video_path = staging_path
    status = "approved"

    # 6b. Cut QC — verify the assembled edit (container, beat-synced cuts,
    # musical audio, no black seams at the cut seams) before it's ever
    # promoted to Final/.
    if ks.get("review", {}).get("cut", True):
        from pipeline.kidsong.cut_qc import review_cut

        runlog.stage("cut_qc")
        step("QC: reviewing the assembled cut…")
        cv = review_cut(staging_path, cut_list, beats, words, duration, cfg, review_dir)
        review_log_entries.append({"stage": "cut", "attempt": 0, "verdict": cv})
        runlog.info(
            "Cut QC verdict: accept=%s score=%s reasons=%s",
            bool(cv.get("accept")), cv.get("score"), "; ".join(cv.get("reasons", [])),
        )

        if not cv.get("accept") and cv.get("retry_hints", {}).get("recut"):
            step(
                f"  Cut rejected ({', '.join(cv.get('reasons', []))}) — "
                "recutting on lyric timing (no beat snap)…"
            )
            cut_list = _maybe_lipsync(build_cut_list(shotlist, None, duration, cfg))
            # The rejected first cut stays on disk for review, so the recut gets
            # its own staging name rather than overwriting the evidence.
            staging_path = runstate.reserve_output_path(
                os.path.join(out_dir, base + ".staging.mp4")
            )
            assemble(cfg, cut_list, renders, voice_path, words, duration, staging_path, on_progress=step)
            cv = review_cut(staging_path, cut_list, beats, words, duration, cfg, review_dir)
            review_log_entries.append({"stage": "cut", "attempt": 1, "verdict": cv})

        if cv.get("accept") and _promote_gates_pass():
            final_path = _promote(staging_path)
            step("Adding the channel branding intro…")
            from pipeline.kidsong.edit import prepend_intro

            intro_result = prepend_intro(final_path, cfg, on_progress=step)
            runlog.stage("intro", **intro_result)
            video_path = final_path
            status = "approved"
            runlog.stage("promote", path=final_path)
        elif not cv.get("accept"):
            step(
                f"  Cut still rejected after recut attempt "
                f"({', '.join(cv.get('reasons', []))}) — leaving in staging for review."
            )
            video_path = staging_path
            status = "needs_review"
            runlog.stage("promote", skipped="cut rejected", staging=staging_path)
        else:
            # Cut QC accepted; a promote gate refused (vision coverage,
            # group-vision, or require_approved — each already logged loudly
            # and appended to review_log_entries). Same outcome shape as a
            # cut reject: staging file stays put, run ends cleanly.
            video_path = staging_path
            status = "needs_review"
            runlog.stage("promote", skipped="promote gate", staging=staging_path)
    elif _promote_gates_pass():
        final_path = _promote(staging_path)
        step("Adding the channel branding intro…")
        from pipeline.kidsong.edit import prepend_intro

        intro_result = prepend_intro(final_path, cfg, on_progress=step)
        runlog.stage("intro", **intro_result)
        video_path = final_path
        runlog.stage("promote", path=final_path, qc="disabled")
    else:
        video_path = staging_path
        status = "needs_review"
        runlog.stage(
            "promote", skipped="promote gate", staging=staging_path,
            qc="disabled",
        )

    if review_log_entries:
        review_shots_log(review_log_path, review_log_entries)

    result = {
        "video_path": video_path,
        "status": status,
        "title": song["title"],
        "description": song["description"],
        "tags": song["tags"],
        "duration": duration,
        "shots": len(shotlist["shots"]),
        "unique_renders": len(unique),
    }

    if do_upload and status == "approved":
        step("Uploading to YouTube (declared made-for-kids)…")
        from pipeline.youtube_upload import upload

        up_cfg = copy.deepcopy(cfg)
        up_cfg["youtube"]["made_for_kids"] = True
        # Wide 16:9 kidsong videos are regular uploads, not Shorts.
        up_cfg["youtube"]["append_shorts_tag"] = False
        result["youtube"] = upload(
            video_path, song["title"], song["description"], song["tags"], up_cfg
        )
    elif do_upload:
        step("Skipping YouTube upload — the cut needs review before it can ship.")

    # Keep the sung wav when the cut wasn't approved — repairs need it. Also
    # keep it when any shot is still unsatisfied (QC rejected every take but it
    # has attempts left): a resume re-renders that shot, and the ledger's shot
    # timings and beat grid are derived from this exact wav, so deleting it
    # would make the re-render impossible.
    unsatisfied = [
        s["id"] for s in shotlist["shots"]
        if not s.get("reuse_of")
        and not runstate.shot_is_done(s, shots_dir)
        and not runstate.shot_is_exhausted(s, max_attempts)
    ]
    if unsatisfied:
        runlog.warning(
            "Keeping %s: %s shot(s) still unapproved (%s) — resume can re-render them.",
            voice_path, len(unsatisfied), ", ".join(unsatisfied),
        )
        step(
            f"  NOTE: {len(unsatisfied)} shot(s) ship unapproved "
            f"({', '.join(unsatisfied)}); resume to re-render them."
        )
    # A draft episode KEEPS its working audio so a later `--finalize` can reuse
    # the exact same sung take when it re-renders at final quality (its shot
    # timings and beat grid are derived from this wav). Every other tier keeps
    # today's cleanup exactly, so the default/custom/final path is unchanged.
    keep_audio_for_finalize = quality_tier == "draft"
    if (result.get("status") != "needs_review" and not unsatisfied
            and not keep_audio_for_finalize):
        try:
            os.remove(voice_path)
        except OSError:
            pass
    elif keep_audio_for_finalize and result.get("status") == "approved" and not unsatisfied:
        runlog.info(
            "Draft kept its audio %s so `--finalize %s` can re-render at final "
            "quality against the same sung take.", voice_path, base,
        )

    step("Done.")
    return result


# ------------------------------------------------------- on-request finalize ---
#: Fields on a shot ledger entry that describe a rendered TAKE (as opposed to
#: the plan). Cleared when a draft's shot list is reused for a final re-render so
#: the render loop redraws every shot instead of adopting the draft's take.
_TAKE_LEDGER_FIELDS = (
    "status", "take", "verdict", "score", "attempts", "failure_reason",
    "last_verdict",
)


def _reset_shot_for_finalize(shot):
    """A copy of `shot` with its render-ledger fields cleared back to "planned",
    keeping the structural plan (id, verse, timing, shot_type, characters,
    action, setting, camera, reuse_of, seed, story_subject, lyric_span, …). The
    finalize render must NOT adopt the draft's takes, so every shot must look
    unrendered on the new base's ledger."""
    fresh = {k: v for k, v in shot.items() if k not in _TAKE_LEDGER_FIELDS}
    fresh["status"] = runstate.STATUS_PLANNED
    return fresh


def _style_slug(name):
    """A filesystem-safe fragment of a style name for the re-render base."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-") or "style"


def _reserve_rerender_base(out_dir, source_base, suffix):
    """A fresh, non-colliding ``<base>-<suffix>`` for a re-render, then
    ``-<suffix>-v2``, ``-<suffix>-v3``, … A base is "taken" if its ledger, shots
    dir, working audio OR a promoted ``Final/`` output already exists, so the
    source — and any earlier re-render of it — is never touched or overwritten."""
    def _taken(b):
        return (
            os.path.exists(runstate.shots_json_path(out_dir, b))
            or os.path.exists(runstate.shots_dir_path(out_dir, b))
            or os.path.exists(runstate.wav_path(out_dir, b))
            or os.path.exists(os.path.join(out_dir, "Final", b + ".mp4"))
        )

    candidate = f"{source_base}-{suffix}"
    if not _taken(candidate):
        return candidate
    for n in range(2, runstate.MAX_VERSIONS + 1):
        c = f"{source_base}-{suffix}-v{n}"
        if not _taken(c):
            return c
    raise RuntimeError(
        f"Refusing to re-render {source_base}: every -{suffix} base up to "
        f"-{suffix}-v{runstate.MAX_VERSIONS} already exists."
    )


def _reserve_final_base(out_dir, draft_base):
    """``<draft>-final``, ``-final-v2``, … — see `_reserve_rerender_base`."""
    return _reserve_rerender_base(out_dir, draft_base, "final")


def _rerender_episode(source_base, overrides, suffix, what, do_upload=False,
                      cfg=None, on_progress=None, runlog=None, stage="finalize"):
    """Re-render an existing episode under `overrides`, on a NEW base.

    The shared body of `finalize_episode` (overrides the quality tier) and
    `restyle_episode` (overrides the render style). It reuses the source's song,
    shot list and sung audio — so the timing, the beat grid, the shot plan AND
    **every shot's seed** are identical — resets the ledger to "planned" so
    nothing adopts the old takes, and delegates to the resume path. The source's
    ledger, takes and ``Final/`` output are never touched.

    Keeping the seeds is what makes a re-render a controlled comparison rather
    than a second roll of the dice: `_reset_shot_for_finalize` clears the take
    ledger and keeps the structural plan, `seed` included, and no seed anywhere
    in the pipeline is derived from the quality tier or the render style.

    SCOPE — a full re-render reusing song + shot list, NOT a selective one. The
    shot ledger records no per-take quality/style flag, so the set of shots
    needing a redraw is exactly all of them. Bounding — the per-shot attempt cap
    and the "cover the slot with the best take, never fail the episode"
    guarantee — is inherited unchanged from the resume path this delegates to.
    """
    if cfg is None:
        cfg = load_config()
    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    own_log = runlog is None
    if own_log:
        runlog = RunLog(out_dir, base=None, stream=on_progress is None)
        runlog.fingerprint(cfg, extra={stage: source_base})

    def step(msg):
        runlog.info("  -> %s", msg)
        if on_progress:
            on_progress(msg)

    try:
        song = runstate.load_song(out_dir, source_base)
        shotlist = runstate.load_shotlist(out_dir, source_base)
        if not song:
            raise FileNotFoundError(
                f"Cannot {stage} {source_base}: "
                f"{runstate.song_json_path(out_dir, source_base)} is missing or unreadable."
            )
        if not (isinstance(shotlist, dict) and shotlist.get("shots")):
            raise FileNotFoundError(
                f"Cannot {stage} {source_base}: "
                f"{runstate.shots_json_path(out_dir, source_base)} is missing or has no shots."
            )
        source_wav = runstate.wav_path(out_dir, source_base)
        if not (os.path.exists(source_wav) and os.path.getsize(source_wav) > 0):
            raise FileNotFoundError(
                f"Cannot {stage} {source_base}: its working audio {source_wav} is "
                f"gone. A re-render runs against the episode's ORIGINAL sung audio "
                "so the shot timings and beat grid still line up; a draft keeps "
                "that wav for exactly this, so a missing one means the episode was "
                "not rendered as a draft (or the wav was removed by hand)."
            )

        new_base = _reserve_rerender_base(out_dir, source_base, suffix)
        runlog.stage(
            stage, source=source_base, new_base=new_base,
            shots=len(shotlist["shots"]),
        )
        step(f"Re-rendering {source_base} -> {new_base} ({what})…")

        # Reuse the song + shot list + audio under the NEW base. The ledger is
        # reset to "planned" so the render redraws every shot instead of
        # adopting the source's takes (which stay put under the source base —
        # the new base's shots dir starts empty, so recover_takes finds nothing
        # to adopt). Nothing under the source base is written.
        atomic_write_json(
            runstate.song_json_path(out_dir, new_base), song,
            indent=2, ensure_ascii=False,
        )
        fresh = {k: v for k, v in shotlist.items() if k != "shots"}
        fresh["shots"] = [_reset_shot_for_finalize(s) for s in shotlist["shots"]]
        runstate.save_shotlist(out_dir, new_base, fresh)
        shutil.copyfile(source_wav, runstate.wav_path(out_dir, new_base))

        new_cfg = copy.deepcopy(cfg)
        new_cfg.setdefault("kidsong", {}).update(overrides)

        # Delegate to the resume path: audio present -> no re-singing; shot list
        # present -> no re-planning; empty takes dir -> every shot re-rendered
        # under the overrides; promotion + versioning never overwrite anything.
        return generate_kidsong(
            do_upload=do_upload, cfg=new_cfg, on_progress=on_progress,
            resume_base=new_base, runlog=runlog,
        )
    except BaseException as exc:
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            runlog.exception(f"{stage.capitalize()} FAILED", exc)
        raise
    finally:
        if own_log:
            runlog.close()


def finalize_episode(draft_base, do_upload=False, cfg=None, on_progress=None,
                     runlog=None):
    """Re-render an already-rendered DRAFT episode at final quality, on request.

    "gemischt — schneller Entwurf, dann teure Endfassung auf Zuruf": drafts are
    the fast throughput default; when a draft is worth keeping this produces the
    expensive final version. Reuses the draft's song, shot list, sung audio and
    per-shot seeds; forces the ``final`` quality tier; re-renders every shot at
    full fidelity under ``<draft>-final``. See `_rerender_episode`.
    """
    return _rerender_episode(
        draft_base, {"quality": "final"}, "final", "re-render at final quality",
        do_upload=do_upload, cfg=cfg, on_progress=on_progress, runlog=runlog,
        stage="finalize",
    )


def restyle_episode(source_base, style_name, do_upload=False, cfg=None,
                    on_progress=None, runlog=None):
    """Re-render an existing episode under a DIFFERENT render style.

    This is the A/B harness for the render-style registry, and the reason it
    exists: before it, there was no way to see two looks on the same material.
    `--finalize` changes the quality tier (so a style comparison through it is
    confounded by frame count and the hires pass), `--resume` renders only what
    is missing (nothing, for a finished episode), and a fresh run writes a
    different song and a different shot list. Every knob that could vary is held
    still — same lyrics, same sung audio, same beat grid, same shot plan, same
    per-shot seeds — and only the style moves.

    The concrete experiment it was built for is the prompt-length A/B recorded
    in IMP-035/QM-030::

        python -m pipeline.kidsong.generate --restyle <base> --style pixar_toon
        python -m pipeline.kidsong.generate --restyle <base> --style pixar_toon_concise

    then compare the two episodes' review logs. An unknown style name is a hard
    error here even though `resolve_style` merely warns and falls back: falling
    back would silently render arm B as arm A, and an A/B that quietly compares
    a style against itself is worse than one that refuses to start.
    """
    if cfg is None:
        cfg = load_config()
    style_name = str(style_name or "").strip()
    known = (cfg.get("kidsong", {}) or {}).get("render_styles") or {}
    if not style_name:
        raise ValueError("--restyle needs --style NAME (the render style to render under).")
    if style_name not in known:
        raise ValueError(
            f"Unknown render style {style_name!r}. Defined in kidsong.render_styles: "
            f"{', '.join(sorted(k for k in known if not k.startswith('_'))) or 'none'}."
        )
    return _rerender_episode(
        source_base, {"render_style": style_name}, f"style-{_style_slug(style_name)}",
        f"re-render in render style {style_name}",
        do_upload=do_upload, cfg=cfg, on_progress=on_progress, runlog=runlog,
        stage="restyle",
    )


def _build_parser():
    """The CLI argument parser — shared by ``__main__`` and the tests so the
    accepted flags (and their dispatch) can be exercised without the process /
    GPU-lock plumbing."""
    import argparse

    p = argparse.ArgumentParser(description="Generate a kids' song video")
    p.add_argument("--topic", default=None, help="song topic, e.g. 'counting to five'")
    p.add_argument("--upload", action="store_true")
    p.add_argument(
        "--resume", default=None, metavar="BASE",
        help="resume an interrupted run: reuse its takes and render only what is missing",
    )
    p.add_argument(
        "--finalize", default=None, metavar="BASE",
        help="re-render an already-rendered DRAFT episode at final quality under a "
             "new base (reuses its song, shot list and audio; never touches the draft)",
    )
    p.add_argument(
        "--restyle", default=None, metavar="BASE",
        help="re-render an existing episode under a different render style "
             "(needs --style): same song, shot list, audio and per-shot seeds, "
             "so the two looks are directly comparable. The A/B harness.",
    )
    p.add_argument(
        "--style", default=None, metavar="NAME",
        help="the kidsong.render_styles entry --restyle renders under, "
             "e.g. pixar_toon_concise",
    )
    p.add_argument(
        "--list-resumable", action="store_true",
        help="list interrupted runs that --resume can pick up, then exit",
    )
    return p


def _dispatch(args, cfg, runlog):
    """Route parsed CLI args to the right generator. Extracted so the finalize /
    resume / new-run decision is unit-testable without the GPU lock and log
    plumbing that wraps it in ``__main__``."""
    if args.restyle:
        return restyle_episode(
            args.restyle, getattr(args, "style", None), do_upload=args.upload,
            cfg=cfg, runlog=runlog,
        )
    if args.finalize:
        return finalize_episode(
            args.finalize, do_upload=args.upload, cfg=cfg, runlog=runlog,
        )
    return generate_kidsong(
        args.topic, args.upload, cfg=cfg, runlog=runlog, resume_base=args.resume,
    )


if __name__ == "__main__":
    args = _build_parser().parse_args()

    cfg = load_config()
    _out_dir = abspath(cfg, cfg["paths"]["output_dir"])

    # Read-only inventory: no GPU, no lock, safe to run while a render is going.
    if args.list_resumable:
        print(runstate.format_resumable(runstate.find_resumable(_out_dir)))
        raise SystemExit(0)

    # Logging is set up BEFORE the lock wait, so even a run that dies waiting
    # for the GPU leaves a log behind.
    _runlog = RunLog(_out_dir, base=args.resume)
    from pipeline.kidsong.runlog import install_excepthook

    install_excepthook(_runlog)
    _runlog.fingerprint(
        cfg,
        extra={
            "argv": " ".join(sys.argv[1:]),
            "topic": args.topic,
            "resume": args.resume or "(new run)",
            "finalize": args.finalize or "(no)",
            "restyle": f"{args.restyle} as {args.style}" if args.restyle else "(no)",
        },
    )

    # Single-GPU mutual exclusion: the studio scheduler (running inside
    # app.py, if it's up) can be rendering a kidsong job on its own thread at
    # the same moment a batch script starts this CLI invocation in its own
    # process — that collision has already cost a render (ComfyUI timed out
    # mid-job). We choose to WAIT for the lock rather than fail immediately:
    # this entry point is normally driven by an unattended batch runner
    # (output/batch_queue.ps1) that expects to run a whole topic list
    # sequentially without a human restarting it after every collision, and
    # batch_queue.ps1 already waits out same-CLI collisions the same way —
    # this just extends that to the scheduler too. A generous bounded
    # timeout (gpu_lock.cli_wait_timeout_seconds, default 2h) is still the
    # loud-failure fallback in case the lock is genuinely stuck in a way the
    # stale-pid check can't detect, so this never hangs forever silently.
    from pipeline.gpu_lock import get_lock

    lock = get_lock(cfg)
    lock_cfg = cfg.get("gpu_lock") or {}
    wait_timeout = lock_cfg.get("cli_wait_timeout_seconds")
    wait_timeout = float(wait_timeout) if wait_timeout else None
    poll_seconds = float(lock_cfg.get("poll_seconds", 2.0))
    # Print (not log) less often than we poll, so a long wait doesn't spam
    # the console/batch log every couple of seconds.
    _last_notice = [0.0]

    def _on_wait(holder_pid):
        now = time.time()
        if now - _last_notice[0] < 30:
            return
        _last_notice[0] = now
        _runlog.info(
            "Waiting for the GPU lock (held by pid %s) — another render is using it…",
            holder_pid,
        )

    try:
        _runlog.stage("gpu_lock", action="acquire")
        lock.acquire(timeout=wait_timeout, poll_seconds=poll_seconds, on_wait=_on_wait)
        _runlog.stage("gpu_lock", action="acquired")
    except TimeoutError as e:
        msg = (
            f"ERROR: {e}\n"
            "Another process has held the GPU lock the whole wait — if it's not "
            "actually rendering anymore, delete the lock file and re-run."
        )
        _runlog.exception("GPU lock acquisition timed out", e)
        _runlog.close()
        raise SystemExit(msg) from e

    # Any crash from here on is written to <base>.log AND appended to
    # output/kidsong_errors.log before we exit nonzero — a run can no longer
    # vanish without an error message.
    try:
        out = _dispatch(args, cfg, _runlog)
    except BaseException as exc:
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            _runlog.exception("Kidsong CLI run FAILED", exc)
            base_hint = _runlog.base or "(no base yet)"
            print(
                f"\nERROR: the run failed. Full traceback in:\n"
                f"  {_runlog.path}\n"
                f"  {os.path.join(_out_dir, 'kidsong_errors.log')}\n"
                f"Resume what was rendered with:\n"
                f"  python -m pipeline.kidsong.generate --resume {base_hint}",
                file=sys.stderr,
            )
        _runlog.close()
        raise SystemExit(1) from exc
    finally:
        lock.release()

    _runlog.info("Result: %s", json.dumps(out, default=str))
    _runlog.close()
    print("\nResult:")
    for k, v in out.items():
        print(f"  {k}: {v}")
