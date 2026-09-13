"""
kidsong.sing — Sing a whole children's song with ACE-Step 1.5.

This replaces the TTS-vocals + procedural music-bed path (song_audio.py) with a
single, fully-sung track: ACE-Step 1.5 turns a style/genre *caption* plus tagged
*lyrics* into one complete song (vocals + instruments) in one shot.

ACE-Step runs OUT OF PROCESS. Driving it in-process from the repo venv (Python
3.13 / torch 2.11 / transformers 5.x, with a meta-device monkey-patch) produced
corrupted, noise-like audio. The verified-good recipe only reproduces in the
official pinned environment (Python 3.12 / torch 2.7.1+cu128 / transformers
4.57.6 / ACE-Step 1.5). So `sing_song` writes a small job JSON and shells out to
`ace_runner.py` under that environment's Python (cfg["kidsong"]["ace_python"]),
then converts the wav ACE-Step saved to a 44.1 kHz stereo WAV with ffmpeg.

Because the heavy work happens in a child process that exits when it is done, the
GPU memory is released automatically — no in-process teardown, and this module
imports neither torch nor acestep. The child mirrors the ground-truth recipe:
acestep's own get_gpu_config tier defaults, the turbo DiT, inference_steps=8, and
the MANDATORY turbo shift=3.0 (see ace_runner.py).

ACE-Step downloads its weights into a *checkpoints directory* (not the HF cache),
resolved from the ACESTEP_CHECKPOINTS_DIR env var; we point both that and HF_HOME
at D: (via the subprocess env) so nothing lands on the space-constrained C: drive.
"""
import json
import logging
import os
import re
import subprocess

log = logging.getLogger("kidsong.sing")

# A verse window shorter than this is not a slightly-wrong anchor, it is a
# broken layout — see the fallback at the end of verse_times_from_words.
_MIN_VERSE_SECONDS = 1.0



# ------------------------------------------------------------------ settings ---
_DEFAULTS = {
    # Style/genre prompt (ACE-Step "caption"): describes the sound, not the words.
    "song_style": (
        "children's nursery rhyme, cheerful upbeat kids song, cute female child "
        "voice, simple melody, playful, bright"
    ),
    "song_seconds": 60,
    "seed": 20260717,
    "hf_home": "D:/brainrot/hf",
    # Where ACE-Step caches its model weights (DiT, VAE, text encoder, 5Hz LM).
    "acestep_checkpoints": "D:/brainrot/acestep",
    # 5Hz LM checkpoint id, or null/None to let acestep's get_gpu_config pick the
    # tier default (the verified-good path: 0.6B on this 12 GB card). Passed
    # straight through to ace_runner.py.
    "lm_model": None,
    # Python interpreter of the official isolated ACE-Step environment.
    "ace_python": "D:/brainrot/ace-venv/Scripts/python.exe",
}

_TARGET_SR = 44100  # contract: write a 44.1 kHz WAV
_ACE_TIMEOUT_S = 900  # subprocess wall-clock cap

# Stdout lines worth keeping: ace_runner.py prints the GPU tier, the 5Hz LM's
# planned bpm/key (when it can extract them), the seed and the generation
# time — the only per-render record of what the LM actually planned (see the
# module docstring: thinking=True/use_cot_metas=True lets it replan on every
# render, seed only pins the DiT noise).
_ACE_STDOUT_LOG_RE = re.compile(r"(?i)\b(bpm|key|seed|lm_model|gen_time|tier)\b")


def _mirror_ace_stdout(stdout, runlog):
    """Mirror the ACE-Step subprocess's informative stdout lines into the run
    log instead of silently discarding them once the subprocess exits.

    Without this, the only trace of what the 5Hz LM planned for THIS render
    (bpm, key, seed, tier/backend) lived in a child process's stdout that the
    parent captured and then threw away on success — undiagnosable after the
    fact when a render turns out scrambled (see edit.beat_grid's resultant
    metric, which is what actually flags a scrambled render)."""
    if runlog is None or not hasattr(runlog, "info"):
        return
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line and _ACE_STDOUT_LOG_RE.search(line):
            runlog.info("  ace: %s", line)


# --------------------------------------------------------------- lyric format ---
def _clean_line(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def build_lyrics(verses):
    """Render the lyrics dict's verses into ACE-Step's tagged lyric format.

    ACE-Step organizes lyrics with bracketed section tags (see the repo's
    examples/text2music/*.json), one line per line, blank line between sections:

        [Verse 1]
        Clap, clap, clap your hands
        Clap them with me now

        [Chorus]
        ...

    A verse whose lines exactly repeat an earlier verse is tagged [Chorus] so the
    model treats the recurring refrain as a hook; everything else is [Verse N].
    """
    blocks = []
    seen = {}
    verse_no = 0
    for v in verses:
        lines = [_clean_line(l) for l in (v.get("lines") or []) if _clean_line(l)]
        if not lines:
            continue
        key = " / ".join(l.lower() for l in lines)
        if key in seen:
            tag = "[Chorus]"
        else:
            verse_no += 1
            tag = f"[Verse {verse_no}]"
            seen[key] = tag
        blocks.append(tag + "\n" + "\n".join(lines))
    return "\n\n".join(blocks) if blocks else "[Verse 1]\nLa la la la la"


def _resolve_caption(song, settings):
    """The ACE-Step style/genre caption: the per-song caption wins.

    `pipeline.kidsong.lyrics.generate_song` stamps a mood-matched caption onto
    `song["song_style"]` (e.g. a lullaby gets a gentle/sleepy caption, an
    upbeat romp gets a bouncy one). Before this, `sing_song` read ONLY the
    global `kidsong.song_style` config value, so every song — lullaby or
    romp — sang with the channel's one default caption; measured 2026-07-24: a
    Twinkle-Twinkle lullaby sung with the cheerful-upbeat default. The global
    value remains the fallback for songs that don't carry their own (older
    song.json files, or library songs that never got a per-song caption)."""
    song_style = (song or {}).get("song_style")
    if song_style:
        return str(song_style)
    return str(settings.get("song_style", _DEFAULTS["song_style"]))


# ------------------------------------------------------------------- public ---
def sing_song(song, cfg, out_path, runlog=None):
    """Generate one fully-sung children's song with ACE-Step 1.5.

    song:     lyrics dict from pipeline.kidsong.lyrics
              ({"title", "verses": [{"lines": [...], "scene": ...}, ...]}).
    cfg:      loaded config dict (reads cfg["kidsong"]).
    out_path: destination .wav (written as 44.1 kHz).
    runlog:   optional pipeline.kidsong.runlog.RunLog. When present the
              audio-cleanup decision is written to the run log; callers that
              have no logger (including the existing positional 3-argument
              callers in generate.py) are unaffected.

    Runs ACE-Step out of process in its official isolated environment (see the
    module docstring and ace_runner.py). Raises RuntimeError on any failure so
    the orchestrator can fall back to the TTS path.

    The raw render is then passed through pipeline.kidsong.audio_clean, which
    strips ACE-Step's junk lead-in and trailing dead air and levels the track.
    The returned `duration` is the duration of the CLEANED file — downstream the
    beat grid, the verse timing and the cut list are all derived from it, so it
    has to describe the audio that actually ships.

    Returns {"audio_path": out_path, "duration": <float seconds>,
             "clean": <audio_clean metadata dict>}.
    """
    settings = cfg.get("kidsong", {})

    def opt(key):
        return settings.get(key, _DEFAULTS[key])

    seed = int(opt("seed"))
    duration = float(opt("song_seconds"))
    caption = _resolve_caption(song, settings)
    # lm_model may legitimately be None/null -> ace_runner uses the tier default.
    lm_model = settings.get("lm_model", _DEFAULTS["lm_model"])

    hf_home = opt("hf_home")
    checkpoints_dir = os.path.abspath(opt("acestep_checkpoints"))
    ace_python = opt("ace_python")

    lyrics = build_lyrics(song.get("verses") or [])

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw_dir = os.path.join(os.path.dirname(out_path), "_acestep_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ---- 1. Write the job JSON the isolated runner consumes. ----
    job = {
        "caption": caption,
        "lyrics": lyrics,
        "duration": duration,
        "seed": seed,
        "lm_model": lm_model,   # null -> ace_runner picks the gpu_config default
        "out_dir": raw_dir,
    }
    job_path = os.path.join(raw_dir, "job.json")
    with open(job_path, "w", encoding="utf-8") as fh:
        json.dump(job, fh)

    ace_runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ace_runner.py")
    if not os.path.exists(ace_python):
        raise RuntimeError(
            f"ACE-Step environment Python not found at {ace_python}. "
            "Set cfg['kidsong']['ace_python']. Falling back to TTS vocals."
        )

    # ---- 2. Run ACE-Step in its official env. The child owns (and, on exit,
    # frees) the GPU; PYTHONUTF8=1 keeps its ASCII/Unicode stdout clean. ----
    env = {
        **os.environ,
        "ACESTEP_CHECKPOINTS_DIR": checkpoints_dir,
        "HF_HOME": hf_home,
        "PYTHONUTF8": "1",
    }

    print(f"Singing '{song.get('title', 'song')}' (~{duration:.0f}s, seed {seed}) "
          f"via ACE-Step subprocess…")
    try:
        proc = subprocess.run(
            [ace_python, ace_runner_path, job_path],
            env=env,
            capture_output=True,
            text=True,
            # The child runs with PYTHONUTF8=1, so decode its stdout/stderr as
            # UTF-8 (not the parent's cp1252 default) and never raise on a stray
            # byte — otherwise the stderr tail we report on failure is lost.
            encoding="utf-8",
            errors="replace",
            timeout=_ACE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ACE-Step subprocess timed out after {_ACE_TIMEOUT_S}s. "
            "Falling back to TTS vocals."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not launch ACE-Step subprocess ({type(exc).__name__}: {exc}). "
            "Falling back to TTS vocals."
        ) from exc

    # Mirror the child's bpm/key/seed/tier lines into the run log BEFORE the
    # success/failure branch below — a failed render's partial planning info
    # is exactly as useful for post-mortems as a successful one's.
    _mirror_ace_stdout(proc.stdout, runlog)

    result_path = os.path.join(raw_dir, "result.json")
    if proc.returncode != 0 or not os.path.exists(result_path):
        tail = "\n".join((proc.stderr or "").splitlines()[-30:])
        raise RuntimeError(
            f"ACE-Step subprocess failed (exit {proc.returncode}). "
            f"Last stderr:\n{tail}"
        )

    with open(result_path, "r", encoding="utf-8") as fh:
        res = json.load(fh)
    raw_path = res.get("audio_path")
    if not raw_path or not os.path.exists(raw_path):
        tail = "\n".join((proc.stderr or "").splitlines()[-30:])
        raise RuntimeError(
            "ACE-Step subprocess reported success but the wav is missing "
            f"({raw_path!r}). Last stderr:\n{tail}"
        )

    # ---- 3. Convert ACE-Step's saved render to a 44.1 kHz stereo WAV. ----
    # Trust ACE-Step's OWN wav and let ffmpeg do the resample — battle-tested,
    # no hand-rolled tensor/layout handling to get subtly wrong.
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", raw_path,
         "-ar", str(_TARGET_SR), "-ac", "2", out_path],
        check=True,
    )

    # ---- 4. Strip ACE-Step's junk lead-in / trailing dead air and level it. ----
    # This runs on the converted 44.1 kHz wav (not the raw render) so the
    # trimmer sees exactly the audio that would otherwise have shipped, and it
    # runs BEFORE the duration is read: every downstream timing decision must
    # be based on the cleaned file.
    from pipeline.kidsong.audio_clean import clean_song

    clean_meta = clean_song(out_path, cfg, runlog=runlog)

    import wave

    with wave.open(out_path, "rb") as wf:
        final_duration = wf.getnframes() / float(wf.getframerate())

    msg = f"Wrote {out_path} ({final_duration:.2f}s @ {_TARGET_SR} Hz)"
    if runlog is not None and hasattr(runlog, "info"):
        runlog.info("  %s", msg)
    else:
        print(f"  {msg}")

    # The raw ACE-Step renders are duplicates of out_path — don't litter.
    import shutil

    shutil.rmtree(raw_dir, ignore_errors=True)
    return {"audio_path": out_path, "duration": final_duration, "clean": clean_meta}


# --------------------------------------------------- verse alignment (Whisper) ---
_PUNCT_RE = re.compile(r"[^\w']+", re.UNICODE)


def _norm_token(text):
    return _PUNCT_RE.sub("", str(text).lower()).strip("'")


def _line_tokens(line):
    return [t for t in (_norm_token(w) for w in str(line).split()) if t]


def _word_text(w):
    return w.get("word", w.get("text", ""))


def _find_first_line(words, target_tokens, search_from, min_overlap=0.5):
    """Slide a window over `words` (from `search_from`) looking for the verse's
    first line. Returns (index, start_time) of the best window whose token
    overlap with `target_tokens` is >= min_overlap, else None.

    Sung vocals transcribe imperfectly, so we match on set overlap (order- and
    duplicate-insensitive) rather than an exact substring.
    """
    target = set(target_tokens)
    if not target or not words:
        return None
    win = max(1, len(target_tokens))
    best = None
    for j in range(search_from, len(words)):
        window = set(
            _norm_token(_word_text(words[k]))
            for k in range(j, min(j + win, len(words)))
        )
        window.discard("")
        if not window:
            continue
        overlap = len(window & target) / len(target)
        if overlap >= min_overlap and (best is None or overlap > best[2]):
            best = (j, float(words[j].get("start", 0.0)), overlap)
            if overlap >= 0.99:
                break
    if best is None:
        return None
    return best[0], best[1]


def verse_times_from_words(words, verses, duration):
    """Map Whisper word timings of the SUNG audio to per-verse (start, end) windows.

    words:    [{"word"/"text": str, "start": float, "end": float}, ...]
    verses:   the song's verses (each with "lines").
    duration: total audio length in seconds.

    Locates each verse by fuzzy-matching its first line in the transcript stream;
    verses that can't be matched (or that would break monotonicity) fall back to a
    proportional split of `duration` by verse line counts. Returns contiguous
    (start, end) tuples covering [0, duration].
    """
    duration = float(duration)
    n = len(verses)
    if n == 0:
        return []

    # Proportional fallback boundaries, weighted by each verse's line count.
    counts = [max(1, len(v.get("lines") or [])) for v in verses]
    total = float(sum(counts))
    prop_starts = [0.0]
    acc = 0
    for c in counts[:-1]:
        acc += c
        prop_starts.append(duration * acc / total)

    words = words or []
    starts = [None] * n
    starts[0] = 0.0  # first verse always anchors the start of the track

    search_from = 0
    for i in range(1, n):
        lines = verses[i].get("lines") or []
        target = _line_tokens(lines[0]) if lines else []
        match = _find_first_line(words, target, search_from) if target else None
        if match is not None and match[1] > (starts[i - 1] or 0.0):
            starts[i] = match[1]
            search_from = match[0] + 1
        else:
            starts[i] = None  # defer to proportional / monotonic fill below

    # Fill gaps and enforce a strictly increasing, in-bounds boundary sequence.
    for i in range(1, n):
        s = starts[i]
        if s is None or s <= starts[i - 1] or s >= duration:
            s = prop_starts[i]
        s = min(max(s, starts[i - 1] + 1e-3), duration)
        starts[i] = s

    # One late false match collapses everything after it. The monotonic nudge
    # above is +1e-3 per verse, so if verse i matched far too late (a repeated
    # chorus line scoring higher in the outro than in its own verse — the fuzzy
    # match takes the best overlap in the remaining stream, not the earliest
    # good one), every later verse whose proportional fallback lands BEFORE
    # that point is squeezed to a 1ms window. Measured: a 60s song came out as
    # 48.0s / 0.001s / 0.001s / 12.0s. The director then tiles ~no shots into
    # those verses and the cut gate rejects the result.
    #
    # A layout with a degenerate window is not a slightly-wrong anchor, it is a
    # broken one — so distrust the anchors wholesale and use the proportional
    # split, which is exactly the fallback that already exists for a verse that
    # could not be matched at all.
    windows = [(starts[i], starts[i + 1] if i + 1 < n else duration) for i in range(n)]
    if n > 1 and any(b - a < _MIN_VERSE_SECONDS for a, b in windows):
        log.warning(
            "kidsong sing: verse anchoring produced a degenerate layout (%s) — "
            "falling back to the proportional split for the whole song",
            ", ".join(f"{b - a:.3f}s" for a, b in windows),
        )
        starts = list(prop_starts)
        windows = [(starts[i], starts[i + 1] if i + 1 < n else duration) for i in range(n)]

    return windows


# --------------------------------------------------------------------- main ---
if __name__ == "__main__":
    from pipeline.config import load_config

    cfg = load_config()
    out_dir = os.path.join(cfg["_root"], "output", "kidsong_sing_test")
    os.makedirs(out_dir, exist_ok=True)
    song_path = os.path.join(out_dir, "song.wav")

    test_song = {
        "title": "Wiggle and Giggle",
        "verses": [
            {
                "lines": [
                    "Wiggle your fingers, wiggle your toes",
                    "Wiggle your nose wherever it goes",
                    "Wiggle up high and wiggle down low",
                ],
                "scene": "toddlers wiggling in a sunny playroom",
            },
            {
                "lines": [
                    "Giggle and giggle, hee hee hee",
                    "Giggle with you and giggle with me",
                    "Giggle so happy, wild and free",
                ],
                "scene": "toddlers giggling together under a rainbow",
            },
        ],
    }

    print("=== sing_song ===")
    result = sing_song(test_song, cfg, song_path)
    print("result:", result)

    # 3-second sample from the middle for a quick listen + a loudness readout.
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(song_path)
        print("full dBFS:", seg.dBFS)
        mid = max(0, len(seg) // 2 - 1500)
        sample = seg[mid:mid + 3000]
        sample_path = os.path.join(out_dir, "sample.wav")
        sample.export(sample_path, format="wav")
        print("sample:", sample_path, "sample dBFS:", sample.dBFS)
    except Exception as exc:
        print("sample extraction skipped:", exc)

    print("\n=== verse_times_from_words (matched) ===")
    fake_words = []
    t = 0.0
    # Verse 1 first line, some filler, then verse 2 first line — as Whisper might
    # transcribe the sung audio (imperfectly).
    stream = ("wiggle your fingers wiggle your toes la la la "
              "giggle and giggle hee hee hee la la").split()
    for w in stream:
        fake_words.append({"word": w, "start": t, "end": t + 0.4})
        t += 0.4
    print(verse_times_from_words(fake_words, test_song["verses"], t))

    print("=== verse_times_from_words (fallback, no usable transcript) ===")
    print(verse_times_from_words([], test_song["verses"], 60.0))
