"""
captions.py — Get exact per-word timestamps from the voice track using faster-whisper.

We transcribe the audio we just generated (so the caption text always matches what's
actually spoken) with word_timestamps=True. The model is cached between runs.

Returns a flat list of words:
  [{"word": "octopuses", "start": 1.22, "end": 1.74}, ...]
"""
import os
import re

_MODEL = None
_MODEL_KEY = None


def _add_nvidia_dll_dirs():
    """ctranslate2 on Windows needs cuBLAS/cuDNN DLLs on the DLL search path.
    The pip packages nvidia-cublas-cu12 / nvidia-cudnn-cu12 ship them under
    site-packages/nvidia/*/bin — register those directories if present."""
    if os.name != "nt":
        return
    try:
        import nvidia

        for base in nvidia.__path__:
            for sub in os.listdir(base):
                bin_dir = os.path.join(base, sub, "bin")
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)
    except Exception:
        pass


def _load(size, device, compute_type):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device=device, compute_type=compute_type)


def _get_model(cfg, force_cpu=False, size=None):
    global _MODEL, _MODEL_KEY
    w = cfg["whisper"]
    size = size or w["model_size"]
    device = "cpu" if force_cpu else w["device"]
    compute = "int8" if force_cpu else w["compute_type"]
    key = (size, device, compute)
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    _add_nvidia_dll_dirs()
    try:
        _MODEL = _load(size, device, compute)
    except Exception as e:  # GPU/cuDNN not available -> fall back to CPU
        print(f"[captions] GPU load failed ({e}); falling back to CPU int8.")
        _MODEL = _load(size, "cpu", "int8")
        key = (size, "cpu", "int8")
    _MODEL_KEY = key
    return _MODEL


def _clean(word):
    w = word.strip()
    # drop surrounding punctuation but keep apostrophes inside words
    w = re.sub(r"^[^\w']+|[^\w']+$", "", w)
    return w


def _transcribe(model, audio_path, vad, language):
    segments, _info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=vad,
        language=language,
        beam_size=5,
    )
    words = []
    for seg in segments:  # generator — must be drained by the caller's try block
        for w in seg.words or []:
            txt = _clean(w.word)
            if txt:
                words.append(
                    {"word": txt, "start": float(w.start), "end": float(w.end)}
                )
    return words


def _vad_default(cfg):
    """VAD is OFF unless config opts in — see transcribe_words."""
    return bool(cfg["whisper"].get("vad_filter", False))


def transcribe_words(audio_path, cfg, vad=None, language=None, size=None):
    """Open transcription -> [{"word", "start", "end"}, ...] (original casing).

    vad defaults to whisper.vad_filter in config (OFF), because Silero VAD is
    trained on speech and classifies sung audio over a music bed as non-speech —
    it silently returned ZERO words for every song this channel shipped.

    language defaults to `whisper.language` in config, or "en" when unset, so a
    German channel (kidsong.language="de", threaded into whisper.language) is
    transcribed as German instead of forcing an English decode on German vocals.
    """
    global _MODEL, _MODEL_KEY
    if vad is None:
        vad = _vad_default(cfg)
    if language is None:
        language = (cfg.get("whisper") or {}).get("language", "en")
    model = _get_model(cfg, size=size)
    try:
        return _transcribe(model, audio_path, vad, language)
    except Exception as e:
        # GPU model can load fine but crash on first real encode (missing
        # cuBLAS/cuDNN DLLs) — retry once on CPU instead of dying.
        print(f"[captions] GPU transcribe failed ({e}); retrying on CPU int8.")
        _MODEL = None
        _MODEL_KEY = None
        return _transcribe(_get_model(cfg, force_cpu=True, size=size), audio_path, vad, language)


def get_word_timestamps(audio_path, cfg, uppercase=True, lyrics=None, duration=None,
                        language=None):
    """Per-word caption timings for `audio_path`.

    When `lyrics` (a list of known lyric lines, or a song's verses) is supplied
    we align that known text onto the transcript's clock instead of trusting the
    transcript — far more reliable on singing. Degrades in a fixed order and
    prints which path it took, so a caption failure can never again pass silently:

        forced alignment to known lyrics  ->  raw transcript  ->  [] (caller's
        proportional split)

    language selects the Whisper decode language; None resolves to
    `whisper.language` in config (default "en"). The kidsong pipeline passes its
    `kidsong.language` here so German vocals are transcribed as German for both
    verse alignment and on-screen captions.
    """
    if language is None:
        language = (cfg.get("whisper") or {}).get("language", "en")
    # Singing needs a bigger model: `small` silently drops the whole first half
    # of a song (quiet verses over a music bed); `medium` recovers them.
    size = cfg["whisper"].get("song_model_size", "medium") if lyrics else None
    words = transcribe_words(audio_path, cfg, language=language, size=size)
    if not words:
        # VAD or a bad decode ate everything — retry once with VAD explicitly off.
        print("[captions] transcript empty; retrying with vad_filter=False.")
        words = transcribe_words(audio_path, cfg, vad=False, language=language, size=size)

    if lyrics:
        from pipeline import alignment

        lines = lyrics
        if lines and isinstance(lines[0], dict):
            # with_verse: keeps the verse index attached to each line so the
            # aligned words carry it through to caption grouping, which must
            # not build a group that spans a verse boundary.
            lines = alignment.verse_lines(lyrics, with_verse=True)
        try:
            aligned = alignment.align_to_lyrics(words, lines, audio_end=duration)
        except Exception as e:
            print(f"[captions] lyric alignment errored ({e}); using raw transcript.")
            aligned = None
        if aligned:
            print(f"[captions] path=aligned-to-lyrics words={len(aligned)}")
            words = aligned
        else:
            print(
                f"[captions] path=raw-transcript (lyric alignment rejected) "
                f"words={len(words)}"
            )
    else:
        print(f"[captions] path=raw-transcript words={len(words)}")

    if not words:
        print(
            "[captions] WARNING: no words recovered — captions will be EMPTY and "
            "verse times will fall back to a proportional split."
        )
    if uppercase:
        for w in words:
            w["word"] = w["word"].upper()
    return words


if __name__ == "__main__":
    import sys

    from pipeline.config import load_config

    cfg = load_config()
    path = sys.argv[1] if len(sys.argv) > 1 else "output/_tts_test.wav"
    for word in get_word_timestamps(path, cfg):
        print(word)
