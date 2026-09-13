"""
kidsong.director — Turn a song into a professional shot list for the text-to-video renderer.

Given the song, the per-verse (start, end) timestamps (seconds), and (optionally) the
song's beat grid, this produces a kids'-TV-grammar shot list: contiguous ~3s shots per
verse, a wide establishing opener, sparing camera moves, and chorus verses that reuse an
earlier verse's shots instead of inventing new footage.

Chain of attempts (first success wins; every exception is swallowed and we fall through
to the next option, so this never crashes the pipeline) — the same shape as
pipeline.kidsong.lyrics.generate_song:
  1. Whatever backend is configured in config.json -> llm.backend (groq/ollama/gemini),
     reusing the same call helpers as script_gen.py.
  2. A plain Ollama /api/chat call against llm.ollama_host (skipped if step 1 already
     was an Ollama call — no point retrying the identical request).
  3. An OpenAI-compatible /v1/chat/completions call against the same host, useful when
     a local llama-server (not real Ollama) is listening on that port instead and only
     speaks the OpenAI API shape. The model id is discovered via GET /v1/models.
  4. A deterministic fallback planner.

The LLM (when reachable) only ever influences *choices* inside each shot — shot_type,
camera, action, characters. Every hard constraint (contiguous tiling, duration bounds,
beat snapping, the wide establishing opener, chorus reuse, the unique-shot cap, seeding)
is enforced in `_normalize_shots` regardless of what the LLM returns, on top of the same
deterministic skeleton `_fallback_planner` builds when the LLM is unreachable.

Returns strict JSON shaped like:
  {
    "shots": [
      {"id": "s03", "verse": 1, "lyric_span": [0, 1], "start": 12.41, "end": 15.62,
       "shot_type": "wide|medium|closeup", "characters": ["Zuri"], "action": "...",
       "camera": "static|slow push-in|slow pull-back|gentle pan left|gentle pan right",
       "setting": "...", "reuse_of": null, "seed": 20260717, "status": "planned"},
      ...
    ]
  }
"""
import json
import logging
import math
import os
import re

from pipeline.script_gen import (
    _call_gemini,
    _call_groq,
    _call_ollama,
    _extract_json,
    _load_prompt,
)

log = logging.getLogger("kidsong.director")

SYSTEM_PROMPT = (
    "You are a professional director for a wholesome preschool YouTube channel, turning "
    "song lyrics into a detailed shot list using kids'-TV camera grammar (clear wides, "
    "warm mediums, playful close-ups, sparing camera moves). You ALWAYS reply with a "
    "single valid JSON object and nothing else."
)

_VALID_SHOT_TYPES = ("wide", "medium", "closeup", "insert")
_VALID_CAMERAS = (
    "static", "slow push-in", "slow pull-back", "gentle pan left", "gentle pan right",
)

_DIRECTOR_DEFAULTS = {
    # Calibrated against the most-viewed real preschool video (60 shots,
    # median 2.54s, p10 1.33s, p90 4.56s, ~21 cuts/min).
    "target_shot_seconds": 2.8,
    "max_unique_shots": 16,
    "reuse_chorus_shots": True,
}
_MIN_SHOT_SECONDS = 1.4
_MAX_SHOT_SECONDS = 4.5
_BEAT_SNAP_TOLERANCE = 0.45

# Scene-mode Phase 1: opt-in, config-driven per-shot duration ceiling/floor.
# `_MIN_SHOT_SECONDS` / `_MAX_SHOT_SECONDS` above remain the hard-coded
# DEFAULTS (existing callers/tests that never pass min_s/max_s, and any cfg
# that never sets these keys, get byte-identical behaviour); `_shot_bounds`
# is the single place that resolves the effective bounds from cfg for the
# planning core, so raising kidsong.director.max_shot_seconds (Phase 1's
# "fewer, longer shots" knob) actually reaches `_num_shots_for`/`_tile`.
_CAMERA_PALETTE = (
    "static", "slow push-in", "slow pull-back", "gentle pan left", "gentle pan right",
)


def _shot_bounds(cfg):
    """(min_seconds, max_seconds) for per-shot duration, resolved from
    kidsong.director.min_shot_seconds / .max_shot_seconds, defaulting to the
    module constants when absent -- so an unset cfg is byte-identical to
    today's fixed bounds.

    max_seconds is additionally clamped to the RENDERABLE take length,
    (kidsong.shot.max_frames - editor handle) / fps: a slot longer than the
    longest take the render cap allows cannot be covered by real footage and
    forces the editor's fill path (slow-stretch, or the visible ping-pong) --
    measured 2026-07-25 when max_frames dropped to the i2v-hires black-frame
    envelope (81f) under the old 4.5s ceiling and cuts started playing
    forward-then-backward. Absent shot config keeps today's numbers
    (max_frames default 241 never binds)."""
    ks = ((cfg or {}).get("kidsong", {}) or {})
    director_cfg = ks.get("director", {}) or {}
    min_s = float(director_cfg.get("min_shot_seconds", _MIN_SHOT_SECONDS))
    max_s = float(director_cfg.get("max_shot_seconds", _MAX_SHOT_SECONDS))
    shot_cfg = ks.get("shot", {}) or {}
    fps = int(shot_cfg.get("fps", 24)) or 24
    max_frames = int(shot_cfg.get("max_frames", 241))
    renderable_s = (max_frames - fps // 4) / float(fps)
    if renderable_s < max_s:
        _warn_clamped_ceiling(max_s, renderable_s, max_frames, fps)
        max_s = renderable_s
    return min(min_s, max_s), max_s


_CLAMP_WARNED = set()


def _warn_clamped_ceiling(configured_s, renderable_s, max_frames, fps):
    """Say out loud that kidsong.director.max_shot_seconds is not what's in effect.

    Without this the knob silently does nothing: the shipped config asks for
    4.5s shots while shot.max_frames=81 caps a take at 3.13s, so raising
    max_shot_seconds changes neither the shot count nor the shot length and
    the operator gets no hint why. Deduped per (configured, effective) pair so
    a batch run logs it once, not once per episode."""
    key = (round(configured_s, 3), round(renderable_s, 3))
    if key in _CLAMP_WARNED:
        return
    _CLAMP_WARNED.add(key)
    log.warning(
        "kidsong.director.max_shot_seconds=%.2fs is capped to %.2fs by "
        "kidsong.shot.max_frames=%d @ %dfps (minus the editor handle). "
        "Raise max_frames to make longer shots reachable.",
        configured_s, renderable_s, max_frames, fps,
    )


# De-creep (kidsong.decreep.*, default disabled): the verified root cause of
# the channel's signature defect -- every rendered toddler blankly staring
# into the lens -- is a handful of camera-directed phrases baked into this
# module's and generate.py's prompt text (see `_gaze_sentence`, the
# `_PERFORMANCE_BEATS`/`_ACTION_VERBS[_SINGULAR]` overlays below, and the
# hook-verse designation in `_fallback_planner`). `enabled` gates all of it
# behind one switch; a cfg that never sets kidsong.decreep resolves to these
# defaults and is byte-identical to today. `hook_camera_budget` bounds how
# many shots per episode `_fallback_planner` may deliberately point at the
# camera -- a hook/refrain verse playing to the lens is an intentional
# performance beat, not the pervasive stare this feature removes.
_DECREEP_DEFAULTS = {
    "enabled": False,
    "hook_camera_budget": 1,
}


def _decreep_cfg(cfg):
    """The effective kidsong.decreep.* settings, every key defaulted so a cfg
    that predates this feature (or never sets the block at all) resolves to
    `_DECREEP_DEFAULTS` exactly -- the flag-off behaviour pinned by
    tests/test_kidsong_shot_prompt.py and tests/test_kidsong_render_style.py."""
    raw = ((cfg or {}).get("kidsong", {}) or {}).get("decreep", {}) or {}
    return {
        "enabled": bool(raw.get("enabled", _DECREEP_DEFAULTS["enabled"])),
        "hook_camera_budget": int(
            raw.get("hook_camera_budget", _DECREEP_DEFAULTS["hook_camera_budget"])
        ),
    }


# Common capitalized words that show up in a "characters" description sentence but
# aren't names (counts, skin/hair descriptors, articles, pronouns...). Anything left
# over after stripping these from the capitalized tokens is treated as a cast name.
_NAME_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "in", "on", "at", "of", "to", "for",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "black", "white", "brown", "tan", "toddlers", "toddler", "kids", "children",
    "girl", "girls", "boy", "boys", "she", "he", "they", "his", "her", "their",
    "is", "are", "was", "were", "this", "that", "little", "small", "big", "all",
}


# ---------------------------------------------------------------- backends ----
def _post_ollama_chat(system, user, model, host):
    """Plain Ollama /api/chat call, independent of which backend plan_shots already tried."""
    import requests

    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.7},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _post_openai_compatible_chat(system, user, host):
    """Fallback for a local llama-server exposing an OpenAI-compatible API instead of
    (or in addition to) Ollama's native one. Discovers the model id via GET /v1/models."""
    import requests

    models_resp = requests.get(f"{host}/v1/models", timeout=600)
    models_resp.raise_for_status()
    models = models_resp.json().get("data") or []
    if not models:
        raise RuntimeError(f"No models reported by {host}/v1/models")
    model = models[0]["id"]

    resp = requests.post(
        f"{host}/v1/chat/completions",
        json={
            "model": model,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ----------------------------------------------------------------- public -----
def plan_shots(song, verse_times, beats, cfg=None):
    if cfg is None:
        from pipeline.config import load_config

        cfg = load_config()
    root = cfg["_root"]
    template = _load_prompt("kidsong_director", root)

    bpm = beats.get("bpm") if isinstance(beats, dict) else None
    user = (
        template
        .replace("{{SONG_JSON}}", json.dumps(song, ensure_ascii=False))
        .replace("{{VERSE_TIMES}}", json.dumps(list(verse_times)))
        .replace("{{BPM}}", str(bpm) if bpm else "unknown")
    )

    llm = cfg.get("llm", {}) or {}
    backend = llm.get("backend")
    host = llm.get("ollama_host", "http://localhost:11434")

    data = None

    # 1. Whatever backend config.json points at.
    try:
        if backend == "groq":
            raw = _call_groq(SYSTEM_PROMPT, user, llm["groq_model"])
        elif backend == "ollama":
            raw = _call_ollama(SYSTEM_PROMPT, user, llm["ollama_model"], host)
        elif backend == "gemini":
            raw = _call_gemini(SYSTEM_PROMPT, user, llm["gemini_model"])
        else:
            raise ValueError(f"Unknown llm.backend: {backend}")
        data = _extract_json(raw)
    except Exception:
        data = None

    # 2. Plain Ollama /api/chat, unless step 1 already made that exact call.
    if data is None and backend != "ollama":
        try:
            raw = _post_ollama_chat(SYSTEM_PROMPT, user, llm.get("ollama_model", ""), host)
            data = _extract_json(raw)
        except Exception:
            data = None

    # 3. OpenAI-compatible endpoint (e.g. a local llama-server sharing the Ollama port).
    if data is None:
        try:
            raw = _post_openai_compatible_chat(SYSTEM_PROMPT, user, host)
            data = _extract_json(raw)
        except Exception:
            data = None

    # 4. `_normalize_shots` builds the deterministic skeleton itself (via
    # `_fallback_planner`) and only lets `data` influence per-shot choices, so a
    # missing/broken LLM response (data stays None) degrades gracefully.
    shots = _normalize_shots(data, song, verse_times, beats, cfg)

    # Scene-mode Phase 1: opt-in camera-variety post-pass (default False ->
    # byte-identical shot list; see `_enforce_camera_variety`).
    director_cfg = ((cfg or {}).get("kidsong", {}) or {}).get("director", {}) or {}
    if bool(director_cfg.get("enforce_camera_variety", False)):
        shots = _enforce_camera_variety(shots)

    return {"shots": shots}


# --------------------------------------------------------------- small utils ---
def _clean_line(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _char_hash(characters):
    """Small stable hash (NOT Python's randomized str hash()) so seeds are
    identical across runs — sorted so same-character-set shots share a seed."""
    key = ",".join(sorted(str(c) for c in characters))
    h = 0
    for ch in key:
        h = (h * 31 + ord(ch)) % 100000
    return h


def _seed_for_characters(characters, base_seed):
    """Deterministic per-character seed via the cast bible (pipeline.kidsong.cast),
    so a given child keeps the SAME seed family across shots AND across
    episodes — the cross-episode consistency lever. Falls back to the old
    `base_seed + _char_hash(...)` scheme (still deterministic, just not
    character-keyed) if the cast bible is unavailable, so a bible problem is
    never a reason to fail shot planning."""
    try:
        from pipeline.kidsong import cast

        return cast.seed_for(characters, base_seed)
    except Exception:
        return base_seed + _char_hash(characters)


def _cast_name_pool(song):
    """The legal cast-name pool for shot planning: the cast bible's fixed
    names (`pipeline.kidsong.cast.names()`) — the SOURCE OF TRUTH for which
    names are even real — plus 'all'. Falls back to the old free-text parse
    of the song's "characters" sentence only if the bible itself can't be
    read, so a bible problem is never a reason to fail shot planning outright.

    Before this, `_normalize_shots` built its `names_allowed` pool from
    `_parse_character_names(song["characters"])` — the LLM's own FREE-TEXT
    field — so the bible was never consulted as the legal name pool and an
    LLM-invented name (typo, hallucination, a name from a different episode)
    could enter a shot's "characters" list just by appearing in that sentence.
    """
    try:
        from pipeline.kidsong import cast

        bible_names = cast.names()
        if bible_names:
            return set(bible_names) | {"all"}
    except Exception as exc:
        log.warning(
            "director: cast bible unavailable for the shot-planning name pool "
            "(%s) — falling back to free-text parsing of song['characters']",
            exc,
        )
    return set(_parse_character_names(song.get("characters", ""))) | {"all"}


# Capitalised words that appear in real shot actions/settings and are NOT a
# child's name. Calibrated against the capitalised vocabulary of every shipped
# shotlist in output/, which is small and highly structured: the cast names,
# a handful of function words/quantifiers/adjectives (below), gerunds that open
# an action phrase ("Watering the sunflowers", "Smiling at Kofi" — covered by
# the participle rule in `_looks_like_a_child_name`, not by this list), and
# five LLM-invented child names (Amira, Maya, Kiara, Jaden, Jaxson) which are
# exactly what we DO want to rewrite. Generalised a little past what the corpus
# contains so a new episode's prose doesn't trip it.
_NON_NAME_CAPS = frozenset({
    # determiners / pronouns / connectives
    "The", "They", "Them", "Their", "Then", "There", "These", "Those", "This",
    "That", "And", "But", "With", "Her", "His", "Its", "She", "One", "Both",
    "All", "Each", "Every", "Everyone", "Everybody", "Together", "While", "When",
    # quantities
    "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    # people words
    "Kid", "Kids", "Child", "Children", "Toddler", "Toddlers", "Friend",
    "Friends", "Boy", "Boys", "Girl", "Girls", "Baby", "Mom", "Mommy", "Dad",
    "Daddy", "Grandma", "Grandpa", "Teacher", "Doctor", "Mr", "Mrs", "Miss", "Ms",
    # colours / qualities / common scene nouns
    "Black", "Brown", "White", "Blue", "Green", "Yellow", "Orange", "Purple",
    "Pink", "Rainbow", "Bright", "Warm", "Sunny", "Cozy", "Happy", "Big",
    "Little", "Small", "Soft", "Colorful", "Colourful", "Sun", "Moon", "Star",
    "Stars", "Sky", "Music", "Morning", "Afternoon", "Evening", "Night",
    # calendar / languages
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "English", "German",
    # past participles that could open an action phrase. Enumerated rather than
    # matched by an "-ed" suffix rule: a blanket -ed rule would also refuse to
    # repair the perfectly ordinary invented names Ahmed, Jared, Fred, Ned,
    # Mohamed — and the shipped corpus contains no -ed opener at all, so the
    # rule was costing real coverage to catch nothing.
    "Dressed", "Seated", "Wrapped", "Surrounded", "Gathered", "Excited",
    "Curled", "Perched", "Covered", "Lined", "Painted", "Framed", "Tucked",
})
_CAP_WORD_RE = re.compile(r"\b([A-Z][a-z]{2,11})\b")


def _looks_like_a_child_name(word, allowed, subject_words):
    """True when a capitalised token in shot prose is a personal name that the
    cast bible does not know.

    Shot actions are written either as a gerund phrase ("Watering the
    sunflowers", "Smiling at Kofi") or as "Name verb-s", so a capitalised token
    ending in -ing is a verb form, never a name — that one rule covers every
    sentence-opening non-name in the shipped corpus and is why this does not
    need a position heuristic (an invented name is very often the FIRST word of
    the action, which is precisely where a position guard would miss it).

    Past participles are handled by `_NON_NAME_CAPS` instead of a matching -ed
    rule, because -ed is a common ending for real given names (Ahmed, Jared,
    Fred) and the corpus contains no -ed opener to justify the trade."""
    if word in allowed or word in _NON_NAME_CAPS or word in subject_words:
        return False
    if word.endswith("ing"):
        return False
    return True


def repair_offbible_names(text, names_allowed, story_subject=None):
    """Replace a child's name that is NOT in the cast bible with the bible child
    it stands in for. Returns the repaired text (unchanged if nothing matched).

    Why this exists: `_normalize_shots` already scrubs off-bible names out of a
    shot's `characters` list, but the LLM writes the same invented name into the
    free-text `action` — and generate.py renders `action` VERBATIM into the
    prompt. Shipped episodes therefore contained prompts like

        "Exactly three children are on screen: Zuri, Kofi and Nala. …
         A wide shot: Amira and Kofi standing in front of the sink …"

    — four names against a head count of three, one of them with no description
    attached, so the text encoder had to invent a fourth child's appearance out
    of the name alone. That is the invented-extra-child defect being fed to the
    model in its own prompt, and both the official and the community LTX
    prompting guidance is explicit that a prompt must never contradict itself.

    Every other name-aware pass in this module (`_cast_names_in`,
    `implied_child_count`, `_closeup_subject`, `sanitize_action`) matches
    against `names_allowed`, so an invented name is invisible to all of them —
    it has to be resolved to a real name before they run.

    The substitution is `cast.substitute_for`: alias-table first, else a stable
    hash, so the same invented name maps to the same bible child on every shot
    and every rerun. Best-effort — an unreadable cast bible returns the text
    untouched."""
    text = str(text or "")
    if not text:
        return text
    allowed = {str(n) for n in (names_allowed or ())}
    subject_words = set(re.findall(r"[A-Za-z]+", str(story_subject or "")))

    try:
        from pipeline.kidsong import cast
    except Exception:
        return text

    # Children this text already puts on screen. Two invented names must not
    # collapse onto the same child — "Maya clapping for Jaden" became "Nala
    # clapping for Nala", which is worse prose than the bug it fixed — and an
    # invented name must not become somebody the sentence already mentions.
    taken = set()
    for word in set(_CAP_WORD_RE.findall(text)):
        if word not in allowed:
            continue
        try:
            known = cast.character(word)
        except Exception:
            known = None
        if known:
            taken.add(known["id"])
    assigned = {}

    def _sub(match):
        word = match.group(1)
        if not _looks_like_a_child_name(word, allowed, subject_words):
            return word
        if word in assigned:  # same invented name twice in one action
            return assigned[word]
        try:
            replacement = cast.substitute_for(word, taken_ids=taken)
        except Exception:
            return word
        if not replacement:
            return word
        taken.add(replacement["id"])
        assigned[word] = replacement["name"]
        log.info(
            "director: action/setting named %r, which is not in the cast bible "
            "— rewritten to %s so the prompt cannot describe a child it never "
            "introduced", word, replacement["name"],
        )
        return replacement["name"]

    return _CAP_WORD_RE.sub(_sub, text)


def _first_named_in_action(action, names_allowed):
    """Cast names actually mentioned in `action`, ordered by where they first
    appear (left to right). Used to find a closeup's grammatical subject —
    "Zuri and Kofi giggle at Nala" names Zuri first, so a closeup ending up
    with that action is a shot ABOUT Zuri, not an alphabetical pick."""
    action_lower = str(action or "").lower()
    hits = []
    for n in names_allowed:
        if n == "all":
            continue
        pos = action_lower.find(n.lower())
        if pos != -1:
            hits.append((pos, n))
    hits.sort()
    return [n for _, n in hits]


def _closeup_subject(merged, skeleton_shot, names_allowed):
    """Pick exactly ONE cast name for a closeup shot that currently names more
    than one character (or 'all') — BUG 3. Preference order:
      1. the grammatical subject: the first cast name mentioned (left to
         right) in the shot's own action text (see `_first_named_in_action`);
      2. the deterministic skeleton's own single-character assignment for
         this shot (never 'all' outside the opening wide shot);
      3. the first non-'all' name already on `merged`;
      4. the first known cast name — so this always returns something.
    """
    mentioned = _first_named_in_action(merged.get("action"), names_allowed)
    if mentioned:
        return mentioned[0]

    skeleton_chars = skeleton_shot.get("characters") or []
    if skeleton_chars and skeleton_chars != ["all"]:
        return skeleton_chars[0]

    for c in merged.get("characters") or []:
        if c != "all":
            return c

    ordered = sorted(n for n in names_allowed if n != "all")
    return ordered[0] if ordered else "all"


def _simplify_closeup_action(action, subject, decreep=False):
    """Rewrite a closeup's action text after narrowing its `characters` to
    `subject`, for a shot whose original action still names one of the OTHER
    children.

    Narrowing `characters` alone isn't enough — generate.py's `_shot_prompt`
    renders `action` verbatim into the render prompt, so an action that still
    says "Zuri and Kofi giggle at Nala" pushes the model towards a crowd
    regardless of what `characters` says. Surgically editing the LLM's prose
    (stripping just the other names out of an arbitrary sentence) reliably
    produces grammatically broken leftovers — "Zuri and  giggle at " — so
    instead this looks for a known action verb already in the sentence
    (the same stems `_line_action` recognizes) and restates it as a clean
    singular action for `subject`; if no known verb is found, it falls back
    to a generic-but-visible singular action. Always grammatical, and always
    mentions only `subject`.

    `decreep` (kidsong.decreep.enabled) selects the verb table via
    `_action_verb_table` — see that helper for why this reads through it
    instead of `_ACTION_VERBS_SINGULAR` directly.
    """
    action_lower = str(action or "").lower()
    tokens = [t.strip(".,!?'\"") for t in action_lower.split()]
    table = _action_verb_table(singular=True, decreep=decreep)
    for t in tokens:
        for verb, direction in table.items():
            if t.startswith(verb):
                return f"{subject} {direction}"
    return f"{subject} sings along with a happy face, moving to the beat"


def _parse_character_names(characters):
    """Heuristically pull cast names out of the song's one-sentence "characters"
    description: capitalized tokens, minus common non-name descriptor words."""
    tokens = re.findall(r"\b[A-Z][a-zA-Z']*\b", str(characters or ""))
    names = []
    seen = set()
    for t in tokens:
        low = t.lower()
        if low in _NAME_STOPWORDS or low in seen:
            continue
        seen.add(low)
        names.append(t)
    return names


def _detect_chorus(verses):
    """Verse index -> source verse index, for verses whose lines exactly repeat an
    earlier verse (same normalized-lines keying as sing.build_lyrics' [Chorus] tag)."""
    chorus_source = {}
    seen_keys = {}
    for i, v in enumerate(verses):
        lines = [_clean_line(l) for l in (v.get("lines") or []) if _clean_line(l)]
        key = " / ".join(l.lower() for l in lines)
        if not key:
            continue
        if key in seen_keys:
            chorus_source[i] = seen_keys[key]
        else:
            seen_keys[key] = i
    return chorus_source


def _num_shots_for(duration, target, min_s=_MIN_SHOT_SECONDS, max_s=_MAX_SHOT_SECONDS):
    """Shot count whose resulting per-shot duration sits in [min_s, max_s]s, as close
    to `target` as possible. Degrades to 1 shot for verses too short to split at all.

    `min_s`/`max_s` are NEW optional kwargs (scene-mode Phase 1) defaulting to the
    module constants, so every existing caller that omits them is byte-identical to
    before; the planning core resolves cfg-driven bounds via `_shot_bounds` and passes
    them through explicitly."""
    if duration <= 0:
        return 1
    lo = max(1, math.ceil(duration / max_s - 1e-9))
    hi = max(1, math.floor(duration / min_s + 1e-9))
    if hi < lo:
        return 1
    target_n = max(1, round(duration / target))
    return min(hi, max(lo, target_n))


def _tile(vs, ve, n, beat_times, min_s=_MIN_SHOT_SECONDS, max_s=_MAX_SHOT_SECONDS):
    """n contiguous (start, end) pairs covering [vs, ve). Internal boundaries snap to
    the nearest beat within +/-0.45s *only* if every resulting shot still lands in
    [min_s, max_s]s — otherwise the even, unsnapped tiling is kept (the duration bound
    is the hard constraint; beat-snapping is best-effort).

    `min_s`/`max_s` are NEW optional kwargs (scene-mode Phase 1) defaulting to the
    module constants, so every existing caller that omits them is byte-identical to
    before."""
    step = (ve - vs) / n
    raw_cuts = [vs + k * step for k in range(n + 1)]
    raw_cuts[-1] = ve  # avoid float drift off the verse's real end

    cuts = raw_cuts
    if beat_times:
        snapped = list(raw_cuts)
        for k in range(1, n):
            best = min(beat_times, key=lambda b: abs(b - raw_cuts[k]))
            if abs(best - raw_cuts[k]) <= _BEAT_SNAP_TOLERANCE:
                snapped[k] = best
        durations = [snapped[k + 1] - snapped[k] for k in range(n)]
        if all(d > 0 for d in durations) and all(
            min_s - 1e-6 <= d <= max_s + 1e-6 for d in durations
        ):
            cuts = snapped

    return [(cuts[k], cuts[k + 1]) for k in range(n)]


def _line_breaks(num_lines, n):
    """n contiguous [start, end) line-index spans covering every line of the verse."""
    num_lines = max(1, num_lines)
    breaks = [round(k * num_lines / n) for k in range(n + 1)]
    breaks[0] = 0
    breaks[-1] = num_lines
    for k in range(1, len(breaks)):
        if breaks[k] < breaks[k - 1]:
            breaks[k] = breaks[k - 1]
    return [(breaks[k], breaks[k + 1]) for k in range(n)]


# The generic outdoor catch-all, named so `_match_location` can specifically
# exclude it (see `allow_generic_outdoor` below). It sits LAST among the real
# location rows precisely because it is the weakest signal in the table: its
# keyword list ("grass", "tree", "flower", "sunshine", "sky", "wind"...) is
# common incidental scenery vocabulary that shows up in almost any outdoor
# scene sentence, including ones that also name a much more specific place.
_GENERIC_OUTDOOR_LOCATION = (
    "a sunny backyard with green grass, a wooden fence, bright flowerbeds, a "
    "red tricycle and a leafy tree"
)

# Map scene keywords to renderable LOCATIONS. Verse "scene" text from the LLM
# is usually an action sentence ("Zuri is applying toothpaste…"), and feeding
# that after "in …" in the render prompt produces garbled, placeless prompts —
# the model then invents dining rooms and extra kids. Settings must be places.
#
# The old table had a `("pool", "splash", "water", "wash")` row mapping to "a
# bright bathroom with a bubbly tub behind them, a rubber duck on the tub edge".
# That single row was measured causing two separate defects in
# output/20260720-110129-kidsong-zuris-watering-blooms-day-shots:
#   * CONTENT RULE BREACH — a song about watering GARDEN FLOWERS matched on the
#     word "water" and rendered shots s05-s09 inside a bathroom with a bathtub
#     and rubber ducks (see s08_a0.mp4). The channel forbids bath/tub/pool.
#   * CONTINUITY BREAK — verse 0 was the garden, verse 1 became the bathroom,
#     verse 2 snapped back to the garden. Whiplash mid-song.
# Water play is now an OUTDOOR activity (puddles, watering cans, a garden hose),
# which is both compliant and continuous with the garden songs that trigger it.
#
# ORDER MATTERS: `_match_location` returns the FIRST row whose keywords hit, so
# every SPECIFIC place (a named room, or one of the PD corpus's own farm/boat/
# star rows below) must be listed BEFORE `_GENERIC_OUTDOOR_LOCATION`. Measured
# regression this fixes (output/20260724-181038-kidsong-baa-baa-...): the PD
# block used to sit AFTER the generic backyard row (despite its own comment
# claiming otherwise — it was appended at the end instead), so a farm verse
# that merely mentioned incidental outdoor words alongside its farm words
# ("wool... on the grass") matched the generic row FIRST and the episode
# silently reclassified from its farm into "a sunny backyard ... red tricycle"
# from verse 1 onward — a farm song that starts on the farm and then teleports
# to a random backyard mid-song.
_LOCATION_KEYWORDS = [
    (("toothbrush", "toothpaste", "teeth", "tooth", "brushing", "mirror", "sink"),
     "a bright cheerful bathroom with a white pedestal sink, a round mirror with "
     "a colorful frame, a hanging pendant light and colorful toiletry bottles"),
    (("bed", "sleep", "night", "pajama", "nap", "bedtime"),
     "a cozy bedroom with a star-patterned blanket, a shelf of plush toys, a "
     "small night lamp and a crescent-moon wall decal"),
    (("kitchen", "breakfast", "food", "eat", "snack", "table"),
     "a sunny cheerful kitchen with a fruit bowl, checkered curtains, a small "
     "table with stools and colorful cups on open shelves"),
    (("school", "classroom", "class", "circle time"),
     "a colorful preschool classroom with an alphabet wall poster, cubby "
     "shelves with toys, a round rug and paper crafts on the wall"),
    # --- public-domain nursery-rhyme locations -------------------------------
    # The PD corpus (Twinkle Twinkle, Row Row Row Your Boat, Baa Baa Black
    # Sheep, Zuri Had a Little Lamb) carries NO per-verse "scene" at all, so the
    # song-level fallback in `plan_verse_locations` is the only thing choosing
    # the place. Without these rows every nursery rhyme landed in the generic
    # playroom while its lyrics described a stream, a farm or the night sky —
    # which is precisely the indoor-action-in-an-outdoor-setting contradiction
    # that rendered a room with a grass floor, and it would have made the
    # replan loop unable to converge (the fallback would re-propose a playroom
    # every time). Listed BEFORE the generic backyard row (see the ORDER
    # MATTERS note above) because that row's keywords ("grass", "sky", ...)
    # would otherwise steal a farm/boat/star scene that merely mentions them
    # in passing.
    (("star", "stars", "moon", "twinkle", "night sky"),
     "a cozy backyard at night with a deep blue starry sky, a soft picnic "
     "blanket on the grass and warm lantern light"),
    (("boat", "stream", "river", "pond", "lake", "shore", "sail", "oar", "row"),
     "a sunny grassy riverbank with a little wooden toy rowboat on a calm "
     "shallow stream, green reeds and lily pads"),
    (("farm", "sheep", "lamb", "barn", "hay", "pony", "chicken", "wool"),
     "a sunny little farmyard with a red barn, a wooden gate, round hay bales "
     "and a green pasture"),
    (("backyard", "garden", "outside", "outdoor", "grass", "park", "leaves",
      "flower", "plant", "tree", "water", "watering", "puddle", "splash",
      "bubble", "breeze", "wind", "sunshine", "sky"),
     _GENERIC_OUTDOOR_LOCATION),
]

_DEFAULT_LOCATION = "a bright cheerful playroom with colorful toys and a soft round rug"

# Places the channel's content rules forbid outright. Matched on WORD
# boundaries so "bathroom" (a legal location — the sink, for tooth brushing)
# is not caught by the "bath" term.
_FORBIDDEN_LOCATION_TERMS = (
    "tub", "bathtub", "bath", "baths", "bathing", "pool", "paddling pool",
    "swimming", "swim", "potty", "toilet", "shower", "jacuzzi", "hot tub",
)


# Indoor / outdoor markers, used to catch a shot whose ACTION happens somewhere
# other than its SETTING. Measured instance: feel-the-breeze shot s00 paired the
# action "play in a bright green grassy field with colorful flowers" with the
# setting "a bright cheerful playroom with colorful toys". The renderer resolved
# that contradiction literally — s00_a0.mp4 shows an interior room, complete
# with wall trim and a framed picture, whose FLOOR IS GRASS with flowers growing
# out of it. A physically impossible space, produced by a prompt that asserted
# both halves in one breath.
_INDOOR_MARKERS = (
    "playroom", "bedroom", "kitchen", "classroom", "bathroom", "living room",
    "hallway", "indoors", "indoor", "rug", "carpet", "sofa", "couch",
    "cot", "crib", "sink", "cupboard",
)
_OUTDOOR_MARKERS = (
    "backyard", "garden", "field", "park", "playground", "meadow", "lawn",
    "grass", "grassy", "outside", "outdoors", "outdoor", "sky", "sunshine",
    "breeze", "hedge", "fence", "flowerbed", "treehouse", "sandpit",
)


def location_class(text):
    """'indoor', 'outdoor', or None when `text` does not commit to either."""
    indoor = bool(_matched_terms(text, _INDOOR_MARKERS))
    outdoor = bool(_matched_terms(text, _OUTDOOR_MARKERS))
    if indoor and not outdoor:
        return "indoor"
    if outdoor and not indoor:
        return "outdoor"
    return None


def _location_keyword_hit(text, keyword):
    """Whole-word match for a location keyword plus its common inflections.

    Bare substring matching (what this used to do) is unsafe for the short
    nursery-rhyme keywords: "row" appears inside "brown", "arrow" and "grow",
    and "oar" inside "board" and "roar", so a song could be teleported to a
    riverbank by the word "tomorrow". Requiring a word boundary at the START
    kills those, and allowing an optional s/es/ing suffix keeps plurals and
    gerunds working ("flower" still matches "flowers", "water" still matches
    "watering") — while "star" no longer matches "start".
    """
    return bool(
        re.search(r"\b" + re.escape(keyword) + r"(?:s|es|ing)?\b", text)
    )


def _match_location(scene, allow_generic_outdoor=True):
    """The location a scene sentence names, or None when it names none.

    Distinct from `_location_from`, which substitutes the generic playroom for
    "no match" — the caller building a whole song's location plan needs to tell
    "this verse says nothing about where it is" apart from "this verse is
    explicitly in a playroom", so that a silent verse can INHERIT its
    neighbour's location instead of teleporting indoors.

    `allow_generic_outdoor=False` (used by `plan_verse_locations` once a song
    supplies its own `canonical_setting`) skips `_GENERIC_OUTDOOR_LOCATION`
    entirely, so a match against ONLY that weak catch-all is treated the same
    as no match at all — the episode's own canonical setting is a far
    stronger signal than "this scene happens to mention grass".
    """
    text = str(scene or "").lower()
    for keywords, location in _LOCATION_KEYWORDS:
        if not allow_generic_outdoor and location == _GENERIC_OUTDOOR_LOCATION:
            continue
        if any(_location_keyword_hit(text, k) for k in keywords):
            return location
    return None


def _location_from(scene):
    """Reduce a scene sentence to a concrete, renderable location phrase."""
    return _match_location(scene) or _DEFAULT_LOCATION


def forbidden_location_terms(setting):
    """Forbidden-place terms present in `setting` (word-boundary matched)."""
    text = str(setting or "").lower()
    return [
        term for term in _FORBIDDEN_LOCATION_TERMS
        if re.search(r"\b" + re.escape(term) + r"\b", text)
    ]


def plan_verse_locations(song):
    """One location per verse, chosen for CONTINUITY rather than per-verse in
    isolation.

    Measured failure this replaces (output/20260720-170942-...-feel-the-breeze):
    every verse called `_location_from(verse["scene"])` independently, so a
    verse whose scene named no location fell back to the generic playroom while
    its neighbours resolved outdoors. "Feel the Breeze" — a song entirely about
    going outside — planned verse 0 in a PLAYROOM, verse 1 in a BACKYARD and
    verse 2 back in the PLAYROOM. Shot s00 then rendered an indoor room with a
    grass floor and flowers growing out of it (s00_a0.mp4), because its action
    said "grassy field" while its setting said "playroom".

    Two rules fix that:
      1. INHERITANCE — a verse whose scene names no location takes the previous
         verse's location; the first verse falls back to the location implied by
         the song as a whole (title + description + every scene + every lyric),
         not to the generic playroom.
      2. NO PING-PONG — a verse may not return to a location the song already
         left. Real preschool songs move forward through places (bathroom →
         kitchen → backyard); A → B → A reads as a continuity error.

    A THIRD rule applies when the song carries `song["canonical_setting"]`
    (prompts/pd_songs*.json's one fixed place for the whole episode, stamped
    onto the song dict by `lyrics.pd_song_to_dict`):
      3. CANONICAL DEFAULT — `canonical_setting` replaces the whole-song
         keyword scan as the fallback for a verse that names nothing (INSTEAD
         of `song_default`, so verse 0 of a library song no longer depends on
         `_match_location` guessing well off the concatenated title/scene/
         lyric text), AND a verse match against ONLY the generic outdoor
         catch-all no longer counts as "naming a location" (see
         `_match_location`'s `allow_generic_outdoor`) — that catch-all is too
         weak a signal to override an author-picked canonical setting. A verse
         whose scene names a genuinely SPECIFIC place (a bathroom, a
         schoolroom, the farm/boat/star rows) still overrides, same as always
         — this only closes the generic-catch-all loophole.
    """
    verses = (song or {}).get("verses") or []

    canonical_setting = str((song or {}).get("canonical_setting") or "").strip()
    if canonical_setting:
        song_default = canonical_setting
    else:
        song_text = " ".join(
            [str((song or {}).get("title") or ""), str((song or {}).get("description") or "")]
            + [str(v.get("scene") or "") for v in verses if isinstance(v, dict)]
            + [
                str(line)
                for v in verses if isinstance(v, dict)
                for line in (v.get("lines") or [])
            ]
        )
        song_default = _match_location(song_text) or _DEFAULT_LOCATION

    allow_generic_outdoor = not canonical_setting

    locations = []
    for v in verses:
        scene = v.get("scene") if isinstance(v, dict) else None
        matched = _match_location(scene, allow_generic_outdoor=allow_generic_outdoor)
        if matched is None:
            matched = locations[-1] if locations else song_default
        locations.append(matched)

    # -- rule 2: never go back to a place the song has already left --
    for i in range(1, len(locations)):
        if locations[i] != locations[i - 1] and locations[i] in locations[:i - 1]:
            locations[i] = locations[i - 1]

    return locations


# ------------------------------------------------------ story subjects ------
# The SUBJECT of a nursery rhyme is very often not a child: a mouse runs up the
# clock, a lamb follows Zuri to school, a sheep gives away three bags of wool, a
# boat goes down the stream. Before this table those subjects could not exist in
# a shot at all — `characters` is validated against the cast bible
# (`_cast_name_pool`), so "a tiny cartoon mouse" was dropped, and the verse's
# `scene` hint (which names the subject explicitly, e.g. "A tall friendly
# grandfather clock stands in a sunny playroom with a tiny round cartoon mouse
# at its base") was consumed ONLY for location keyword matching. The song's
# protagonist was read and thrown away, every episode, which is the measured
# root of "there is no story".
#
# `story_subject` is a separate, first-class shot field precisely so it never
# collides with head-count logic: it is NEVER a child and must never be counted
# as one. Phrases are deliberately toon-safe and friendly — nothing frightening
# and no predators, per the channel's content rules.
_STORY_SUBJECTS = (
    (("mouse", "mice"), "a tiny round cartoon mouse"),
    (("lamb", "lambs"), "a fluffy white cartoon lamb"),
    (("sheep",), "a round woolly black cartoon sheep"),
    (("wool", "fleece"), "three fat sacks of soft wool"),
    (("rowboat", "boat", "boats"), "a little wooden toy rowboat"),
    (("duck", "ducks", "duckling", "ducklings"), "a small cheerful cartoon duck"),
    (("star", "stars"), "a big friendly twinkling star"),
    (("clock", "clocks"), "a tall friendly grandfather clock"),
    (("cow", "cows"), "a gentle spotted cartoon cow"),
    (("cat", "cats", "kitten"), "a soft round cartoon cat"),
    (("dog", "dogs", "puppy"), "a happy little cartoon puppy"),
    (("bird", "birds", "robin"), "a bright little cartoon bird"),
    (("butterfly", "butterflies"), "a colorful cartoon butterfly"),
    (("bee", "bees"), "a friendly round cartoon bee"),
    (("fish", "fishes"), "a shiny orange cartoon fish"),
    (("spider", "spiders"), "a friendly little cartoon spider"),
    (("train", "trains"), "a bright little toy train"),
    (("bus", "buses"), "a cheerful yellow toy bus"),
    (("kite", "kites"), "a bright diamond kite"),
    (("teddy", "bear", "bears"), "a soft brown teddy bear"),
    (("ball", "balls"), "a big bright bouncy ball"),
    # --- learning-mode: letters, numbers, shapes ---------------------------
    # Learning episodes (kidsong.content_mode="learning", see lyrics.py /
    # generate.py) always set an explicit PER-VERSE `story_subject` on the
    # verse dict itself (prompts/learning_songs.json), which `_fallback_planner`
    # honors directly and which never needs this table at all. These generic
    # canned phrases exist purely as a FALLBACK so `extract_story_subject` /
    # `song_story_subject` still resolve something sensible for a verse that
    # names a letter/number/shape but carries no explicit override (a future
    # learning song, or an LLM-authored verse) — the same safety net the
    # animal/object rows above give the public-domain corpus.
    (("letter", "letters"), "a big colorful letter"),
    (("number", "numbers"), "a big colorful number"),
    (("circle", "circles"), "a big red circle"),
    (("square", "squares"), "a big blue square"),
    (("triangle", "triangles"), "a big yellow triangle"),
    (("heart", "hearts"), "a big pink heart"),
    (("shape", "shapes"), "a big colorful shape"),
)

# Adjectival noun phrases already present in a scene hint are preferred over the
# canned phrasing above — the library's hints are art direction written for this
# channel and are richer than anything this table can synthesise.
_SUBJECT_PHRASE_RE_CACHE = {}


def _subject_phrase_in(text, keyword):
    """A noun phrase for `keyword` lifted verbatim from `text` (with up to four
    leading adjectives), or None. Lets "a tiny round cartoon mouse at its base"
    yield "a tiny round cartoon mouse" rather than the generic fallback."""
    rx = _SUBJECT_PHRASE_RE_CACHE.get(keyword)
    if rx is None:
        # Only ADJECTIVES may sit between the article and the noun. Allowing any
        # word produced prepositional garbage lifted straight out of the hint —
        # measured: "the grass around the lamb", "the stream as the little boat",
        # "the clock case". Those then became the shot's `story_subject`, so the
        # render prompt introduced "The grass around the lamb is also in the
        # shot" instead of the lamb.
        rx = re.compile(
            r"\b(?:a|an|the)\s+(?:[a-z]+\s+){0,4}?" + re.escape(keyword) + r"\b",
            re.I,
        )
        _SUBJECT_PHRASE_RE_CACHE[keyword] = rx
    m = rx.search(str(text or ""))
    if not m:
        return None
    phrase = _clean_line(m.group(0))
    # Reject anything carrying a preposition or conjunction — a real subject
    # phrase is "a tiny round cartoon mouse", never "the grass around the lamb".
    if _matched_terms(phrase, (
        "of", "around", "as", "in", "on", "at", "with", "near", "beside",
        "behind", "under", "over", "by", "from", "and", "to",
    )):
        return None
    # `story_subject` is a noun phrase meant to be embedded mid-sentence
    # ("Nala smiles and points at {story_subject}"), never a sentence on its
    # own -- lower-case a leading article that is capitalized only because the
    # scene hint happened to start its sentence there ("The little mouse
    # scampers..." -> "the little mouse"), so it never reads "...points at
    # The little mouse".
    # (Guard against an all-caps acronym like "USA" by only lower-casing when
    # the very next character is NOT also an uppercase letter -- "A round
    # woolly black sheep" has a space there, so it lower-cases; a real
    # acronym would have a second capital and is left alone.)
    if phrase[:1].isupper() and not (len(phrase) > 1 and phrase[1].isalpha() and phrase[1].isupper()):
        phrase = phrase[0].lower() + phrase[1:]
    return phrase


def extract_story_subject(scene, lines=()):
    """The non-child story subject a verse's scene hint / lyric names, as a
    short renderable noun phrase, or None when the verse has none.

    Never returns a cast child, and never returns anything frightening — the
    phrases come from `_STORY_SUBJECTS`, which is curated toon-safe.

    The scene hint is consulted FIRST because it is the song library's own art
    direction and usually carries a better phrasing than the bare lyric noun.
    """
    scene_text = str(scene or "")
    lyric_text = " ".join(str(l) for l in (lines or []))

    for source in (scene_text, lyric_text):
        lowered = source.lower()
        for keywords, canned in _STORY_SUBJECTS:
            for kw in keywords:
                if not re.search(r"\b" + re.escape(kw) + r"\b", lowered):
                    continue
                # Prefer the hint's own phrasing when it reads as a noun phrase.
                phrase = _subject_phrase_in(source, kw)
                if phrase and len(phrase) <= 60:
                    return phrase
                return canned
    return None


def song_story_subject(song):
    """ONE hero subject for the whole song, or None.

    Deliberately song-level, not verse-level. Extracting per verse made the
    subject DRIFT: Hickory Dickory Dock resolved "a tiny round cartoon mouse"
    for verse 0, "The little mouse" for verse 1 and "the big clock" for verse 3
    — three different heroes in one 40-second song, and a clock that inherited
    the mouse's verbs ("the big clock tiptoes along quietly"). A song has one
    protagonist; the shot list must agree with itself about who it is. This also
    gives the through-line the director prompt asks for by name (the "hero prop"
    a toddler can track from scene to scene).

    Picked by frequency across the verse scene hints and lyrics, tie-broken by
    first appearance, so the recurring character wins over incidental scenery.
    """
    verses = (song or {}).get("verses") or []
    counts = {}
    order = {}
    for v in verses:
        if not isinstance(v, dict):
            continue
        found = extract_story_subject(v.get("scene"), v.get("lines") or [])
        if not found:
            continue
        # Key on the HEAD NOUN so "The little mouse" and "a tiny round cartoon
        # mouse" are recognised as the same character rather than competing.
        heads = _subject_head_nouns(found)
        key = " ".join(sorted(heads)) or found.lower()
        counts[key] = counts.get(key, 0) + 1
        order.setdefault(key, len(order))
        # Keep the richest phrasing seen for this character.
        best = _BEST_PHRASE.get(key)
        if best is None or len(found) > len(best):
            _BEST_PHRASE[key] = found

    if not counts:
        # Nothing per-verse — try the song as a whole (title + description).
        return extract_story_subject(
            " ".join([
                str((song or {}).get("title") or ""),
                str((song or {}).get("description") or ""),
            ])
        )

    winner = min(counts, key=lambda k: (-counts[k], order[k]))
    return _BEST_PHRASE.get(winner)


# Richest phrasing seen per character head-noun during `song_story_subject`.
_BEST_PHRASE = {}


def subject_clause(line, story_subject):
    """The clause of `line` that is actually ABOUT `story_subject`.

    Nursery-rhyme lines routinely pack two actors into one line: "The clock
    struck one, the mouse ran down". Handing that whole line to the beat planner
    with the mouse as the actor produced, verbatim, "The little mouse has just
    struck the hour with a happy chime" — the mouse performing the CLOCK's verb.
    Narrowing to the clause containing the subject's head noun gives "the mouse
    ran down", which restates correctly.
    """
    text = _clean_line(line)
    if not text or not story_subject:
        return text
    heads = _subject_head_nouns(story_subject)
    if not heads:
        return text
    clauses = [c for c in re.split(r"\s*[,;]\s*|\s+\band\b\s+", text) if c.strip()]
    if len(clauses) < 2:
        return text
    for clause in clauses:
        lowered = clause.lower()
        if any(re.search(r"\b" + re.escape(h) + r"\b", lowered) for h in heads):
            return _clean_line(clause)
    return text


def valid_story_subject(value):
    """True if `value` is usable as a shot's `story_subject`: a short, non-empty
    string that names no cast child and carries no scary or martial staging."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or len(text) > 120:
        return False
    if _cast_names_in(text):
        return False
    return not scary_terms(text) and not martial_terms(text)


_ACTION_VERBS = {
    "clap": "clap their hands to the beat",
    "stomp": "stomp their feet happily",
    "wave": "wave with both hands",
    "jump": "jump up and down with joy",
    "brush": "brush their teeth with big smiles",
    "wash": "wash up with soapy bubbles",
    "dance": "dance and bounce to the music",
    "spin": "spin around giggling",
    "count": "count along on their fingers",
    "splash": "splash and giggle",
    "hug": "share a big group hug",
    "sing": "sing along with happy faces",
    "smile": "smile brightly at each other",
    "share": "share and play together nicely",
    # Prop-driven actions — the reference builds songs around held objects.
    "hold": "hold up the hero prop and show it to the camera",
    "play": "play together with a colorful toy",
    "point": "point excitedly at something just off frame",
    "roll": "roll a bright ball between them",
    "stack": "stack colorful blocks into a tower",
    "pour": "pour water from a little cup",
}

# Singular-subject counterparts of _ACTION_VERBS, keyed by the same verb stems,
# for a shot whose "characters" names exactly one child (BUG 4: pairing a
# single-character shot with _ACTION_VERBS' plural prose produced a direct
# self-contradiction in the render prompt — "A closeup shot of exactly one
# child: Kofi, ... . The kids sing along ..."). Group-only actions ("hug",
# "share") get a plausible solo rephrase rather than a nonsensical singular of
# a group act.
_ACTION_VERBS_SINGULAR = {
    "clap": "claps their hands to the beat",
    "stomp": "stomps their feet happily",
    "wave": "waves with both hands",
    "jump": "jumps up and down with joy",
    "brush": "brushes their teeth with a big smile",
    "wash": "washes up with soapy bubbles",
    "dance": "dances and bounces to the music",
    "spin": "spins around giggling",
    "count": "counts along on their fingers",
    "splash": "splashes and giggles",
    "hug": "hugs their favorite toy happily",
    "sing": "sings along with a happy face",
    "smile": "smiles brightly at the camera",
    "share": "shows off a favorite toy happily",
    "hold": "holds up the hero prop and shows it to the camera",
    "play": "plays happily with a colorful toy",
    "point": "points excitedly at something just off frame",
    "roll": "rolls a bright ball across the floor",
    "stack": "stacks colorful blocks into a tower",
    "pour": "pours water from a little cup",
}


# De-creep verb overlay (kidsong.decreep.enabled, default False -> every
# _ACTION_VERBS[_SINGULAR] lookup below stays on the base tables above,
# byte-identical to today). Overrides ONLY the entries that reference the
# camera/lens ("smiles... at the camera", "...shows it to the camera") --
# _ACTION_VERBS["hold"] and _ACTION_VERBS_SINGULAR["smile"]/["hold"] above --
# with an equally concrete, equally positive direction that points at
# something actually in the scene instead of the lens. Deliberately small and
# additive (see `_action_verb_table`): _ACTION_VERBS["smile"] ("smile
# brightly at each other") already has no camera language and is untouched,
# and every verb not listed here keeps its base-table phrasing either way.
_ACTION_VERBS_SINGULAR_V2 = {
    "smile": "smiles brightly at their friends",
    "hold": "holds up the hero prop proudly for the others to see",
}
_ACTION_VERBS_V2 = {
    "hold": "hold up the hero prop proudly for the others to see",
}


def _action_verb_table(singular, decreep=False):
    """The verb -> stage-direction table a lookup should read from: the base
    `_ACTION_VERBS_SINGULAR`/`_ACTION_VERBS`, with the de-creep overlay merged
    on top (never mutating the base dict) when `decreep` is True.

    Centralizing this is what "applied at lookup time" means in practice --
    `_simplify_closeup_action`, `_line_action` and
    `reduce_scene_to_shared_activity` all ask this function for their table
    instead of reading the module-level dicts directly, so the overlay is
    defined exactly once rather than forking each base table.
    """
    base = _ACTION_VERBS_SINGULAR if singular else _ACTION_VERBS
    if not decreep:
        return base
    overlay = _ACTION_VERBS_SINGULAR_V2 if singular else _ACTION_VERBS_V2
    return {**base, **overlay}


# Performance beats appended to an action that would otherwise repeat the
# PREVIOUS shot's action verbatim. These vary the ENERGY and the framing, never
# the activity — "one verse, one activity" stays intact while the shot stops
# being a carbon copy of its neighbour.
#
# Measured need (output/20260720-...-feel-the-breeze-shots.json): verse 2 ran
# 11 shots, of which s08-s15 carried the byte-identical action "<name> sings
# along with a happy face" while `characters` cycled Zuri → Kofi → Nala → Zuri
# → … Rendered, that strict rotation of interchangeable children performing one
# identical motion is exactly the regimented, roll-call feel the channel is
# trying to avoid.
_PERFORMANCE_BEATS = (
    "looking straight at the camera",
    "bouncing gently on the spot",
    "swaying from side to side",
    "tilting their head with a giggle",
    "leaning in a little closer",
    "clapping once on the beat",
)

# De-creep pool (kidsong.decreep.enabled): the SAME kind of energy/framing
# variety as `_PERFORMANCE_BEATS` above, with zero camera/lens language --
# "looking straight at the camera" is exactly the pervasive-stare defect this
# feature exists to remove, so it has no place in a repeated-action beat
# either. Selected by `vary_repeated_actions`'s `decreep` argument.
_PERFORMANCE_BEATS_V2 = (
    "leaning in to look closer",
    "bouncing gently on the spot",
    "swaying side to side with the melody",
    "turning to grin at a friend",
    "reaching toward it with both hands",
    "clapping once on the beat",
)


# Words that carry no activity information — dropped when reducing an action to
# its activity signature, so that singular/plural and article differences
# ("Zuri sings along with A happy face" vs "The kids sing along with happy
# faceS") do not read as two different activities.
_SIGNATURE_STOPWORDS = {
    "a", "an", "the", "their", "his", "her", "they", "them", "with", "to",
    "and", "of", "at", "in", "on", "up", "kid", "kids", "child", "children",
    "toddler", "toddlers", "friend", "friends", "one", "another", "each",
    "other", "others", "together", "big", "little", "happy",
}


def _activity_signature(action):
    """`action` reduced to the ACTIVITY it describes.

    Three normalisations, each answering a way that two shots of the SAME
    activity would otherwise compare unequal:
      1. the performer's name is removed — "Zuri sings along" and "Kofi sings
         along" are one activity performed by different children, which is
         exactly the roll-call pattern and must compare equal;
      2. everything after the first comma is dropped — that is where
         `vary_repeated_actions` appends its performance beat, and a beat varies
         the ENERGY of an activity, it is not a new activity;
      3. articles, group nouns and plural inflection are normalised away, so
         "sings along with a happy face" and "sing along with happy faces" agree.
    """
    text = str(action or "").lower()
    for name in _DEFAULT_NAME_POOL:
        text = re.sub(r"\b" + re.escape(name.lower()) + r"(?:'s)?\b", "", text)

    # Take the first clause that still says something once the names are gone.
    # Splitting before removing names would take "Zuri" as the whole first
    # clause of "Zuri, Kofi and Nala play in a grassy field" and reduce the
    # entire action to an empty signature — which silently excluded that shot
    # from the per-verse activity count and let a genuinely mixed verse pass.
    clauses = [c for c in text.split(",") if re.search(r"[a-z]", c)]
    text = clauses[0] if clauses else ""

    tokens = []
    for token in re.findall(r"[a-z]+", text):
        # Crude de-inflection: strip ONE trailing "s" ("faces"->"face",
        # "hands"->"hand", "sings"->"sing"). Deliberately not an "es" rule —
        # stripping "es" turns "faces" into "fac" while leaving "face"
        # untouched, which would make the two spellings of the SAME word
        # compare unequal, the exact failure this normalisation exists to stop.
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token in _SIGNATURE_STOPWORDS:
            continue
        tokens.append(token)
    return " ".join(tokens)


# Verbs that are performance BEATS or reactions, not activities. Every
# preschool verse contains them regardless of what the verse is about — an
# establishing shot where the kids "stand" and "smile", a sing-along beat, a
# reaction shot where someone "watches" or "giggles". Counting them as
# activities made a perfectly coherent verse look scattered: the fallback
# song's verse 0 is ONE clapping activity, phrased by the LLM as "the kids
# stand in a sunny backyard with big smiles" / "the kids clasp hands together
# and clap" / "Zuri claps her hands with a big smile" / "the kids continue
# clapping and smiling" — four phrasings, one activity.
_FILLER_VERBS = {
    "sing", "smile", "stand", "sit", "look", "watch", "giggle", "laugh",
    "cheer", "nod", "sway", "bounce", "lean", "tilt", "continue", "face",
    "gather", "wait", "listen", "enjoy", "beam", "grin",
}


def activity_verbs(action):
    """The non-filler activity verbs `action` describes.

    This is the unit the per-verse coherence check counts, because the ACTIVITY
    of a shot is carried by its verb, not by its full phrasing. Comparing whole
    phrases treats paraphrase ("claps her hands" vs "continue clapping") as two
    different activities, which is a false positive on exactly the well-formed
    verses the channel wants.
    """
    text = str(action or "").lower()
    found = set()
    for token in re.findall(r"[a-z]+", text):
        for verb in _ACTIVITY_DETECT_VERBS:
            if token.startswith(verb) and verb not in _FILLER_VERBS:
                found.add(verb)
                break
    return found


def activity_signature(action):
    """Public alias of `_activity_signature` — the QC gate compares a verse's
    shots with it, and must use the SAME notion of "these two shots depict the
    same activity" as the planner that varies them."""
    return _activity_signature(action)


def verse_activity_progression(shots):
    """Classify how a verse's shots vary in ACTIVITY across the verse.

    This exists because "one verse, one activity" — added to stop genuinely
    incoherent crowded frames — was written too broadly and started rejecting
    the thing the channel actually needs: a song telling its story. The two
    cases must be told apart, because they look similar to a naive verb count
    and are opposites in quality:

      * SIMULTANEOUS mixing (bad, still rejected elsewhere): two or more
        children handed two or more DIFFERENT activities to perform in the SAME
        beat — "Zuri holds a bubble wand, Kofi stomps in a puddle, Nala dances
        with arms outstretched". A 2-3s shot cannot stage that, so the renderer
        crowds the frame. Detected on the verse's scene text by
        `verse_mixes_activities` / `_check_verse_scenes`, which stay unchanged.

      * TEMPORAL progression (good, REQUIRED): the verse's shots move through a
        small arc — setup, the event the lyric describes, a reaction. "The clock
        strikes one" / "a tiny round cartoon mouse scampers down the clock case"
        / "Zuri points after it and giggles". Three shots, ONE activity each,
        one shared location, one continuous moment. Counting distinct verbs
        across the verse (the old `<=2` cap) rejected exactly this.

    Returns:
      'uniform'   — <=1 distinct non-filler activity verb across the verse.
      'narrative' — a legitimate arc: one shared setting, at most ONE non-filler
                    activity verb per individual shot, <=4 distinct verbs total.
      'scattered' — no through-line; genuinely incoherent.
    """
    originals = [s for s in (shots or []) if not (s or {}).get("reuse_of")]
    if not originals:
        return "uniform"

    # The story-subject phrase is a NOUN and must not be scanned for verbs:
    # "a little wooden toy rowboat" contains "row", so `activity_verbs`
    # (prefix-matched, to catch inflections) counted a phantom "row" activity in
    # every shot that merely NAMED the boat — including a child's "points at the
    # rowboat" — which broke one-verb-per-shot and mislabelled a clean narrative
    # arc as scattered. Strip the subject phrase from the action first.
    verbs_per_shot = []
    for s in originals:
        text = str(s.get("action") or "")
        subj = s.get("story_subject")
        if subj:
            text = re.sub(re.escape(str(subj)), " ", text, flags=re.I)
        verbs_per_shot.append(activity_verbs(text))
    all_verbs = set()
    for v in verbs_per_shot:
        all_verbs |= v
    if len(all_verbs) <= 1:
        return "uniform"

    settings = {
        str(s.get("setting") or "").strip()
        for s in originals
        if str(s.get("setting") or "").strip()
    }
    # One activity per SHOT is the coherence rule that must survive; several
    # activities across the VERSE is the story.
    one_verb_per_shot = all(len(v) <= 1 for v in verbs_per_shot)
    if len(settings) <= 1 and one_verb_per_shot and len(all_verbs) <= 4:
        return "narrative"
    return "scattered"


def vary_repeated_actions(shots, decreep=False):
    """Give any shot whose action repeats the previous shot's ACTIVITY (within
    the same verse) a different performance beat. Mutates and returns `shots`.

    `decreep` (kidsong.decreep.enabled, default False) selects the beat pool:
    off keeps `_PERFORMANCE_BEATS` (byte-identical to today, including its
    "looking straight at the camera" entry); on switches to
    `_PERFORMANCE_BEATS_V2`, which carries the same energy/framing variety
    with zero camera or lens language, so a de-creeped episode's repeated-
    action shots never reintroduce a camera cue through this side door."""
    pool = _PERFORMANCE_BEATS_V2 if decreep else _PERFORMANCE_BEATS
    # Compare against the previous shot's ORIGINAL action, not its already-varied
    # one — otherwise a run of N identical actions only gets varied on alternate
    # shots, because the varied text no longer equals the next raw duplicate.
    last_original_by_verse = {}
    beat_index = 0
    for shot in shots:
        if shot.get("reuse_of"):
            continue
        verse = shot.get("verse")
        action = str(shot.get("action") or "").strip()
        signature = _activity_signature(action)
        previous = last_original_by_verse.get(verse)
        if previous is not None and signature and signature == previous:
            beat = pool[beat_index % len(pool)]
            beat_index += 1
            shot["action"] = f"{action.rstrip('.')}, {beat}"
        last_original_by_verse[verse] = signature
    return shots


def _line_action(lines, span, subject=None, decreep=False):
    """Deterministic fallback action: a grammatical, VISIBLE action inferred
    from the shot's lyric line. Naively gluing 'The kids' onto raw lyric text
    produced word salad ('The kids to keep them clean and shining'), so we map
    the line's first known action verb to a canned stage direction and fall
    back to a generic sing-along beat when no verb matches.

    `subject` is the shot's single cast name (e.g. "Kofi"), or None/"all" for
    an ensemble shot — it selects singular vs. plural phrasing so a
    single-character shot never gets "The kids ..." prose (BUG 4).

    `decreep` (kidsong.decreep.enabled) selects the verb table via
    `_action_verb_table`, so a matched "smile"/"hold" verb never renders a
    camera-referencing direction under the flag."""
    singular = bool(subject) and subject != "all"
    fallback = (
        f"{subject} sings along with a happy face, moving to the beat"
        if singular else "The kids sing along with happy faces, moving to the beat"
    )
    if not lines:
        return (
            f"{subject} sings along with a happy face" if singular
            else "The kids sing along with happy faces"
        )
    idx = min(span[0], len(lines) - 1)
    tokens = [t.strip(".,!?'\"").lower() for t in lines[idx].split()]
    table = _action_verb_table(singular, decreep=decreep)
    for t in tokens:
        for verb, direction in table.items():
            if t.startswith(verb):
                return f"{subject} {direction}" if singular else f"The kids {direction}"
    return fallback


_SUBJECT_HEAD_STOPWORDS = {
    "a", "an", "the", "little", "tiny", "big", "small", "soft", "round",
    "bright", "friendly", "happy", "cheerful", "cartoon", "toy", "fat",
    "fluffy", "woolly", "white", "black", "brown", "orange", "yellow",
    "spotted", "shiny", "gentle", "tall", "colorful", "three", "sacks", "of",
}


def _subject_head_nouns(story_subject):
    """The distinctive nouns of a story-subject phrase — "a tiny round cartoon
    mouse" -> {"mouse"}. Used to tell whether an action is ABOUT the subject."""
    return {
        t for t in re.findall(r"[a-z]+", str(story_subject or "").lower())
        if t not in _SUBJECT_HEAD_STOPWORDS and len(t) > 2
    }


def action_is_subject_only(action, story_subject, names_allowed=None):
    """True when `action` describes the STORY SUBJECT doing something and puts
    no child on screen at all — "The mouse scampers down the clock case".

    Such a shot must carry `characters: []`, not a child, or generate.py's
    `_shot_prompt` asserts "a closeup of exactly one child: Nala" in the same
    breath as an action about a mouse. Measured verbatim from a live plan for
    Hickory Dickory Dock: shot s05 paired `characters: ["Nala"]` with the action
    "The mouse scampers down the clock case." The renderer resolves that
    contradiction by drawing the child and dropping the mouse — which is exactly
    how the song's protagonist kept vanishing from its own episode.
    """
    text = str(action or "").strip()
    if not text or not story_subject:
        return False
    if _cast_names_in(text, names_allowed) or _mentions_group(text):
        return False
    heads = _subject_head_nouns(story_subject)
    if not heads:
        return False
    lowered = text.lower()
    if not any(re.search(r"\b" + re.escape(h) + r"\b", lowered) for h in heads):
        return False
    # "child"/"kid" wording without a name still puts a child on screen.
    return not _matched_terms(
        text, ("child", "children", "kid", "kids", "toddler", "toddlers", "girl", "boy")
    )


def is_placeholder_action(action):
    """True when `action` is the LLM echoing the prompt's example JSON rather
    than writing a shot.

    Measured, verbatim, from a live llama3.1 plan for Hickory Dickory Dock: shots
    s03 and s06 came back with the action "..." — the literal ellipsis from the
    "action": "..." line of the response template in prompts/kidsong_director.txt.
    The same defect is present in the shipped Twinkle Twinkle shot list (verse 1,
    shot s04). Nothing downstream caught it: `_normalize_shots` only tested the
    action for truthiness, so "..." is truthy and overwrote the skeleton's real
    action, and generate.py then rendered a prompt whose entire action clause was
    three dots. Treat these as ABSENT so the deterministic skeleton's action wins.
    """
    text = str(action or "").strip()
    if not text:
        return True
    if not re.search(r"[a-zA-Z]", text):  # "...", "-", "???"
        return True
    # Bare template leftovers ("TODO", "action here", "<action>").
    stripped = text.strip(".<>[]{}\"' \t").lower()
    return stripped in {"", "todo", "tbd", "action", "action here", "n/a", "none"}


# ------------------------------------------- lyric-derived stage directions ---
# THE flattening bug. `_line_action` maps a lyric's first verb through
# `_ACTION_VERBS` — a 21-entry table of copy-along preschool verbs (clap, stomp,
# wave, brush, ...). Nursery rhymes are not made of those verbs. "The mouse ran
# up the clock" matches NOTHING in that table, so it fell through to the generic
# "The kids sing along with happy faces, moving to the beat", and so did almost
# every other narrative line in the public-domain corpus. Measured on the
# deterministic planner: Hickory Dickory Dock produced 15 shots of which 13 were
# that one sentence, and four different songs produced the identical verb set
# {clap}. The song's own events never reached the screen.
#
# The fix is to stop looking the lyric up in a table of OUR verbs and instead
# restate the LYRIC ITSELF as a stage direction: normalise its tense, swap the
# bare story noun for the full renderable subject phrase, and use that. The
# result is specific to the song by construction, because it is the song's line.
_PAST_TO_PRESENT = {
    "ran": "runs", "was": "is", "were": "are", "went": "goes",
    "struck": "strikes", "followed": "follows", "made": "makes",
    "had": "has", "laughed": "laughs", "played": "plays",
    "clapped": "claps", "waved": "waves", "sang": "sings", "gave": "gives",
    "said": "says", "saw": "sees", "came": "comes", "took": "takes",
    "sat": "sits", "stood": "stands", "fell": "falls", "grew": "grows",
    "flew": "flies", "swam": "swims", "rang": "rings", "ate": "eats",
    "slept": "sleeps", "woke": "wakes", "hid": "hides", "found": "finds",
    "brought": "brings", "kept": "keeps", "held": "holds", "began": "begins",
}

# Nonsense / refrain tokens. A line built mostly of these ("Hickory, dickory,
# dock", "Merrily, merrily, merrily, merrily", "Baa, baa, black sheep") carries
# no depictable event and must NOT be restated literally as an action — it is a
# chant, and the right staging for it is the subject simply being present, or
# the children singing along.
_CHANT_TOKENS = {
    "hickory", "dickory", "dock", "baa", "merrily", "la", "tra", "tick",
    "tock", "twinkle", "ee", "aye", "oh", "hey", "ho", "fa", "doo", "dum",
    "diddle", "hush", "rock", "row",
}

# Lines that are an instruction to the AUDIENCE ("clap along with me", "sing
# with me", "count to three") are the SING-ALONG hook. Repetition is correct
# there — that is what a hook is for — so these keep the copy-along staging.
_HOOK_MARKERS = (
    "clap along", "sing with me", "sing along", "clap your hands",
    "count to three", "with me", "one, two, three", "join in", "everybody",
)


def _is_chant_line(line):
    """True when a lyric line is mostly nonsense/refrain syllables."""
    tokens = [t for t in re.findall(r"[a-z]+", str(line or "").lower())]
    if not tokens:
        return True
    chant = sum(1 for t in tokens if t in _CHANT_TOKENS)
    return chant >= max(1, len(tokens) // 2)


def _is_hook_line(line):
    """True when a lyric line addresses the audience — the sing-along hook."""
    return bool(_matched_terms(line, _HOOK_MARKERS))


def _to_present_tense(text):
    """Past -> present, so a stage direction reads as something happening now."""
    out = []
    for token in str(text or "").split():
        bare = token.strip(".,!?;:'\"").lower()
        mapped = _PAST_TO_PRESENT.get(bare)
        if mapped is None and bare.endswith("ed") and len(bare) > 4:
            stem = bare[:-2]
            mapped = (stem + "s") if not stem.endswith("e") else (stem + "s")
        out.append(mapped if mapped else token)
    return " ".join(out)


def lyric_stage_direction(line, story_subject=None, subject=None):
    """Restate one lyric LINE as a concrete, visible stage direction.

    Returns None when the line carries no depictable event (a chant or a
    sing-along instruction), so the caller can fall back to hook staging.

    `story_subject` is the song's non-child protagonist; when the line is ABOUT
    it, the bare noun is swapped for the full renderable phrase, so "The mouse
    ran up the clock" becomes "A tiny round cartoon mouse runs up the clock".
    """
    text = _clean_line(line)
    if not text or _is_chant_line(text) or _is_hook_line(text):
        return None

    heads = _subject_head_nouns(story_subject) if story_subject else set()
    lowered = text.lower()
    about_subject = any(
        re.search(r"\b" + re.escape(h) + r"\b", lowered) for h in heads
    )

    direction = _to_present_tense(text)

    if about_subject and story_subject:
        # Swap the FIRST bare mention of the subject noun (with any leading
        # article) for the full renderable phrase.
        for head in sorted(heads, key=len, reverse=True):
            rx = re.compile(r"\b(?:a|an|the)\s+" + re.escape(head) + r"\b", re.I)
            if rx.search(direction):
                direction = rx.sub(story_subject, direction, count=1)
                break
            rx2 = re.compile(r"\b" + re.escape(head) + r"\b", re.I)
            if rx2.search(direction):
                direction = rx2.sub(story_subject, direction, count=1)
                break
        return _clean_line(direction).rstrip(".")

    # A line about the children: keep the lyric's own event, but anchor it to
    # this shot's actual subject so the prose agrees with the head count.
    if subject and subject != "all":
        # Replace a leading cast name or pronoun with this shot's child.
        direction = re.sub(
            r"^(?:she|he|they|we)\b", subject, direction, count=1, flags=re.I
        )
        if not _cast_names_in(direction):
            direction = f"{subject}: {direction}"
        return _clean_line(direction).rstrip(".")
    return _clean_line(direction).rstrip(".")


def _subject_presence_action(story_subject, location=None):
    """Staging for a chant line when the song has a story subject: the subject
    is simply THERE, doing something gentle and characteristic."""
    return (
        f"{story_subject} sits quietly in view, looking around with big friendly "
        f"eyes and gently bobbing to the music"
    )



# ------------------------------------------------- head-count consistency ---
# Two GPU A/B rounds (output/_ab_cast_prompts/, output/_ab_cast_prompts_v2/)
# measured wardrobe fidelity improving 5/5 with the cast bible while head-count
# control improved 0/5 — the failure mode merely changed shape, from "invents a
# different extra child" to "clones the correct child 2-3x" (shot s10 rendered
# three identical Zuris in a circle).
#
# The cause is not the negatives; it is that the shot's own action/setting prose
# contradicts the head-count sentence built from the SAME shot a few words
# earlier. Measured, verbatim, from the current code:
#
#   s14  "A closeup shot of exactly one child: Zuri, ... .
#         The kids smile brightly at each other, in
#         Amira, Kofi, and Zuri are rinsing their mouths ..."
#   s10  "A closeup shot of exactly one child: Zuri, ... .
#         The kids sing along ..., in
#         The three friends are brushing their teeth together ..."
#
# One child and three children asserted in one breath. Duplicating the described
# child is an entirely reasonable resolution for a text encoder handed that, and
# it is what the renders do. The fix is to make the prose agree with the count.
#
# Group language that puts 2+ children on screen regardless of any name.
_GROUP_LANGUAGE = (
    "the three friends", "three friends", "the friends", "their friends",
    "all three", "the three", "the kids", "the children", "the toddlers",
    "the trio", "the group", "the others", "everyone", "everybody",
    "each other", "one another", "both children", "both kids",
    "together", "as a group", "in a circle", "side by side",
)

# Settings are PLACES. A setting that predicates something of people ("... are
# rinsing their mouths") is an action sentence that leaked into the location
# slot, and generate.py renders it verbatim after "in ...".
# Matched as whole words (see `setting_is_a_place`). The previous version
# matched bare substrings including " play", which fired on "a bright cheerful
# playROOM with colorful toys" — `_location_from`'s OWN default location. Every
# shot carrying the default setting was therefore reported by the QC gate as
# "setting is not a place", 11 times per episode on the feel-the-breeze shot
# list, which is noise that buries the real findings and can trigger pointless
# replans. Stems keep a trailing marker so inflections still match as words.
_PEOPLE_IN_SETTING = (
    "are", "is", "sing", "sings", "singing", "brush", "brushes", "brushing",
    "play", "plays", "playing", "wash", "washes", "washing", "smile", "smiles",
    "smiling", "laugh", "laughs", "laughing", "stand", "stands", "standing",
    "sit", "sits", "sitting", "dance", "dances", "dancing", "hold", "holds",
    "holding", "look", "looks", "looking", "their", "they", "he", "she",
    "his", "her",
)


# The fixed channel cast, used when no per-song name pool is supplied.
_DEFAULT_NAME_POOL = ("Zuri", "Kofi", "Nala")

# ------------------------------------------------ martial / scary staging ---
# Preschool staging is loose, playful and small-group. Language that puts
# children into ROWS, FORMATIONS or LOCKSTEP renders as a little parade — the
# "scary marching kids" complaint. Confirmed in the renders: s00_a0.mp4 and
# s03_a0.mp4 of the feel-the-breeze episode both show all three toddlers in an
# evenly-spaced straight line, front-on, striding toward camera in unison with
# flat expressions. Nothing in the shot text asked for that staging, but
# nothing forbade it either, and a bare ensemble action ("the three friends
# dance and play") leaves the model to pick a default arrangement — it picks
# a row. These terms are blocked in shot text and pushed into the NEGATIVE.
_MARTIAL_LANGUAGE = (
    "march", "marches", "marching", "parade", "parading", "formation",
    "formations", "in a row", "in rows", "rows of", "straight line",
    "single file", "lockstep", "in step", "in unison", "unison",
    "synchronised", "synchronized", "line up", "lines up", "lined up",
    "salute", "salutes", "saluting", "uniform", "uniforms", "troop",
    "troops", "soldier", "soldiers", "army", "drill", "drills",
    "ranks", "regiment", "military", "battalion", "cadet", "cadets",
)

# Nothing frightening: this is a channel for toddlers.
_SCARY_LANGUAGE = (
    "scary", "scared", "frightening", "frightened", "terrifying", "terrified",
    "creepy", "eerie", "spooky", "sinister", "menacing", "monster", "monsters",
    "ghost", "ghosts", "zombie", "witch", "demon", "nightmare", "horror",
    "screaming", "shrieking", "crying", "sobbing", "weeping", "angry",
    "shouting", "yelling", "growling", "snarling", "chase", "chases",
    "chasing", "weapon", "weapons", "knife", "gun", "blood", "danger",
    "dangerous", "storm", "lightning", "thunder", "darkness", "shadowy",
    "abandoned", "alone in the dark", "lost and alone", "trapped",
)


def _matched_terms(text, terms):
    """Terms from `terms` present in `text`, matched on word boundaries so
    'uniform' does not fire on 'uniformly' and 'drill' does not fire on
    'drilled into the wall'. Multi-word terms are matched verbatim."""
    lowered = str(text or "").lower()
    hits = []
    for term in terms:
        if re.search(r"\b" + re.escape(term) + r"\b", lowered):
            hits.append(term)
    return hits


def martial_terms(text):
    """Martial / lockstep staging terms present in `text`."""
    return _matched_terms(text, _MARTIAL_LANGUAGE)


def scary_terms(text):
    """Frightening or distressing terms present in `text`."""
    return _matched_terms(text, _SCARY_LANGUAGE)


# --------------------------------------------- one verse, one activity ------
# A verse's shots must all read as the SAME continuous moment. The dominant
# source of crowded, incoherent frames is a verse "scene" that hands a
# DIFFERENT action to each child, e.g. (verbatim, from
# output/20260720-170942-...-feel-the-breeze-song.json verse 0):
#
#   "Zuri holds a bubble wand, Kofi stomps in a puddle, Nala dances with
#    arms outstretched"
#
# Three children, three unrelated activities, to be satisfied inside a 2-3s
# shot. The renderer cannot stage that coherently, so it crowds the frame and
# the shots stop matching each other. `reduce_scene_to_shared_activity`
# collapses such a scene to ONE activity the whole cast shares; individual
# children then differ only in FRAMING, never in what they are doing.
_SHARED_ACTIVITY_BY_PROP = (
    (("bubble", "bubbles", "bubble wand"), "blow bubbles and watch them float up"),
    (("puddle", "puddles"), "splash gently in a shallow puddle"),
    (("watering can", "watering cans", "flower", "flowers", "plant", "plants"),
     "water the flowers with little watering cans"),
    (("ball", "balls"), "roll a bright ball to each other"),
    (("block", "blocks"), "stack colorful blocks into a tower"),
    (("leaf", "leaves"), "toss bright autumn leaves into the air"),
    (("toothbrush", "toothbrushes", "teeth"), "brush their teeth with big smiles"),
    (("drum", "drums", "shaker", "shakers", "bell", "bells"),
     "play a simple rhythm on little instruments"),
    (("kite", "kites"), "hold a bright kite up in the breeze"),
    (("ribbon", "ribbons", "scarf", "scarves"), "wave bright ribbons in the air"),
)

# Clause separators that signal "and now a DIFFERENT child does a DIFFERENT
# thing" rather than one continuous action.
_CLAUSE_SPLIT = re.compile(r"\s*(?:,|;|\band\b|\bwhile\b|\bas\b|\bthen\b)\s*", re.I)

# DETECTION vocabulary for "this clause gives a child an activity". Wider than
# `_ACTION_VERBS` (which is the much smaller set of activities we can RESTATE as
# canned stage directions): a scene mixes activities whenever its clauses assign
# different children different DOING words, whether or not this module happens
# to know how to rewrite that particular verb. Without the extra verbs, the real
# scene "Zuri claps, Kofi twirls, Nala laughs as they spin around each other"
# went undetected, because only "clap" was in the narrow set.
_ACTIVITY_DETECT_VERBS = set(_ACTION_VERBS) | {
    "twirl", "laugh", "giggle", "hop", "skip", "wiggle", "run", "walk",
    "sway", "bounce", "cheer", "reach", "stretch", "nod", "kick", "climb",
    "crawl", "blow", "water", "pick", "carry", "push", "pull", "throw",
    "catch", "paint", "draw", "build", "sweep", "sprinkle", "tiptoe",
    "shake", "swing", "peek", "wander", "stroll", "march", "stand", "sit",
    "kneel", "crouch", "lean", "watch", "look", "supervise", "help",
}


def scene_activity_clauses(scene, names_allowed=None):
    """(name, verb) pairs for clauses of `scene` that assign an activity to a
    specific named child. Used to detect a verse that mixes activities."""
    pairs = []
    for clause in _CLAUSE_SPLIT.split(str(scene or "")):
        names = _cast_names_in(clause, names_allowed)
        if not names:
            continue
        tokens = [t.strip(".,!?'\"").lower() for t in clause.split()]
        for t in tokens:
            for verb in sorted(_ACTIVITY_DETECT_VERBS, key=len, reverse=True):
                if t.startswith(verb):
                    pairs.append((names[0], verb))
                    break
            else:
                continue
            break
    return pairs


def verse_mixes_activities(scene, names_allowed=None):
    """True when `scene` gives two or more DIFFERENT children two or more
    DIFFERENT activities — the structural fault behind crowded frames."""
    pairs = scene_activity_clauses(scene, names_allowed)
    return len({n for n, _ in pairs}) >= 2 and len({v for _, v in pairs}) >= 2


def reduce_scene_to_shared_activity(scene, names_allowed=None, decreep=False):
    """Collapse a multi-activity scene into ONE activity the cast shares.

    Prefers an activity built around a PROP the scene already names (so the
    bubble wand in "Zuri holds a bubble wand, Kofi stomps in a puddle, Nala
    dances" survives as "The kids blow bubbles and watch them float up"), and
    falls back to the first recognised action verb restated in the plural.
    Scenes that do not mix activities are returned unchanged.

    `decreep` (kidsong.decreep.enabled) selects the verb table via
    `_action_verb_table` for that plural-verb fallback.
    """
    text = str(scene or "").strip()
    if not text or not verse_mixes_activities(text, names_allowed):
        return text

    lowered = text.lower()
    for props, activity in _SHARED_ACTIVITY_BY_PROP:
        if any(re.search(r"\b" + re.escape(p) + r"\b", lowered) for p in props):
            return f"The kids {activity} together"

    # The first clause's verb, restated in the plural — but only when it is one
    # of the verbs we can actually phrase as a stage direction. The detection
    # vocabulary is deliberately wider than the rewrite vocabulary, so a verb we
    # merely RECOGNISED ("supervise", "twirl") has no canned plural form and
    # falls through to the shared sing-along beat rather than raising.
    table = _action_verb_table(singular=False, decreep=decreep)
    for _name, verb in scene_activity_clauses(text, names_allowed):
        direction = table.get(verb)
        if direction:
            return f"The kids {direction}"
    return "The kids sing along with happy faces, moving to the beat"


def _mentions_group(text):
    """True if `text` uses language that implies more than one child on screen."""
    lowered = " %s " % str(text or "").lower()
    return any(g in lowered for g in _GROUP_LANGUAGE)


def _cast_names_in(text, names_allowed=None):
    """Cast names mentioned in `text`. Falls back to the fixed channel cast plus
    the off-cast names shot lists are known to carry, so this stays useful when
    no name pool is supplied (e.g. the QC gate, which has no `song`)."""
    pool = [n for n in (names_allowed or _DEFAULT_NAME_POOL) if n and n != "all"]
    lowered = str(text or "").lower()
    return [n for n in pool if n.lower() in lowered]


def implied_child_count(text, names_allowed=None):
    """Lower bound on how many distinct children `text` puts on screen.

    Distinct cast names counted directly; group language floors the count at 2
    even when it names nobody ("The kids smile at each other"). Returns 0 when
    the text implies no particular number, which must never be read as "zero
    children" — only as "this text does not constrain the head count".
    """
    named = len(set(_cast_names_in(text, names_allowed)))
    if _mentions_group(text):
        return max(2, named)
    return named


def setting_is_a_place(setting):
    """True if `setting` reads as a location rather than an action sentence."""
    text = str(setting or "")
    if not text.strip():
        return True
    if _cast_names_in(text) or _mentions_group(text):
        return False
    return not _matched_terms(text, _PEOPLE_IN_SETTING)


def sanitize_setting(setting):
    """Reduce a setting that describes PEOPLE to the location it takes place in.

    `_location_from` already maps scene keywords to a concrete, renderable place
    and is the documented contract for this field; it just was never applied to
    settings arriving from an LLM overlay or an older shot list. Keyword mapping
    keeps the room ("... brushing their teeth ... in the mirror" -> the bathroom
    location), so the shot stays in the place the director intended while the
    people vanish from the location slot.
    """
    text = str(setting or "").strip()
    if not text or setting_is_a_place(text):
        return text
    return _location_from(text)


def sanitize_staging(action, subject=None, decreep=False):
    """Replace martial/lockstep or frightening staging with loose, playful
    staging. Preschool blocking is a small, informal cluster of children — never
    a row, a formation or anything a toddler could find alarming.

    `decreep` (kidsong.decreep.enabled) is threaded through to
    `_simplify_closeup_action`'s verb table."""
    text = str(action or "").strip()
    if not text:
        return text
    if not martial_terms(text) and not scary_terms(text):
        return text
    if subject and subject != "all":
        return _simplify_closeup_action(text, subject, decreep=decreep)
    return "The kids play together in a loose, happy cluster, moving to the beat"


def sanitize_location(setting, fallback=None):
    """Replace a setting naming a forbidden place (tub/bath/pool/potty) with a
    compliant one. `fallback` is normally the verse's planned location."""
    text = str(setting or "").strip()
    if not text or not forbidden_location_terms(text):
        return text
    return fallback or _DEFAULT_LOCATION


def sanitize_action(action, count, subject=None, names_allowed=None, decreep=False):
    """Make `action` agree with the head count the render prompt will state.

    `count` is the number of children the prompt asserts (generate.py's
    `_shot_prompt` computes it from the cast bible); None means "unconstrained",
    in which case the action is left alone. When the prompt says one child, an
    action naming or implying more is rewritten as a clean singular action for
    `subject` via `_simplify_closeup_action`, which restates the sentence's own
    verb rather than trying to excise names from arbitrary prose (that reliably
    produced fragments like "Zuri and  giggle at ").

    `decreep` (kidsong.decreep.enabled) is threaded through to
    `_simplify_closeup_action`'s verb table.
    """
    text = str(action or "").strip()
    if count is None or count != 1 or not text:
        return text
    others = [n for n in _cast_names_in(text, names_allowed) if n != subject]
    if not others and not _mentions_group(text):
        return text
    return _simplify_closeup_action(text, subject or "The child", decreep=decreep)


# --------------------------------------------------------- lyric-derived beats ---
# `_ACTION_VERBS` (21 entries, above) is a stage-direction RESTATEMENT table for
# the handful of verbs common to every preschool verse (clap, wave, dance...);
# it is deliberately small because every entry needs a plural AND a singular
# canned restatement. It is nowhere near enough vocabulary to tell "The mouse
# ran up the clock" apart from "The clock struck one, the mouse ran down" --
# both reduce to the same generic "sings along with a happy face" fallback,
# which is the measured defect this module exists to fix (see the module
# docstring). `_LYRIC_VERBS` is the much wider vocabulary nursery rhymes
# actually use, each mapped to two short, concrete, VISIBLE stage directions:
# one for when the STORY SUBJECT performs the beat, one for when a CHILD does.
_LYRIC_VERBS = {
    "run":      ("{subject} runs briskly across the scene", "{name} runs and giggles"),
    "ran":      ("{subject} scampers busily through the scene", "{name} looks up, startled and giggling"),
    "climb":    ("{subject} climbs up high", "{name} reaches up to help, giggling"),
    "scamper":  ("{subject} scampers along in a hurry", "{name} watches it scamper by, delighted"),
    "strike":   ("{subject} strikes the hour with a happy chime", "{name} claps once at the cheerful chime"),
    "struck":   ("{subject} has just struck the hour with a happy chime", "{name} looks up, startled and giggling"),
    "follow":   ("{subject} follows close behind", "{name} follows along happily"),
    "followed": ("{subject} follows right along", "{name} follows close behind, smiling"),
    "go":       ("{subject} goes on ahead", "{name} goes along too, smiling"),
    "went":     ("{subject} went off exploring", "{name} skips along after, smiling"),
    # Avoid the token "water" in these templates: `activity_verbs` treats
    # "water" as the verb (as in "water the flowers"), so "through the water"
    # was miscounted as a second activity and tipped a legitimate one-verb-per-
    # shot boat verse into a false "verse mixes activities" rejection.
    "row":      ("{subject} rows gently along the stream", "{name} rows with both hands on the oar"),
    "sail":     ("{subject} sails smoothly along", "{name} waves as they sail along"),
    "drift":    ("{subject} drifts gently along", "{name} leans back, drifting along happily"),
    "wave":     ("{subject} gives a friendly wave", "{name} waves with a big smile"),
    "tiptoe":   ("{subject} tiptoes along quietly", "{name} tiptoes along, finger to lips"),
    "creep":    ("{subject} creeps along slowly", "{name} creeps along on tiptoe, grinning"),
    "peek":     ("{subject} peeks out shyly", "{name} peeks around with a giggle"),
    "laugh":    ("{subject} seems to laugh along", "{name} laughs with delight"),
    "play":     ("{subject} plays happily nearby", "{name} plays happily nearby"),
    "give":     ("{subject} is given over with a smile", "{name} gives it over with a smile"),
    "gives":    ("{subject} is given over with a smile", "{name} gives it over with a smile"),
    "fill":     ("{subject} is filled right up", "{name} fills it up carefully"),
    "carry":    ("{subject} is carried along carefully", "{name} carries it along carefully"),
    "shine":    ("{subject} shines brightly overhead", "{name} looks up at the shine, smiling"),
    "twinkle":  ("{subject} twinkles softly overhead", "{name} points up at the twinkling light"),
    "sparkle":  ("{subject} sparkles softly overhead", "{name} points up, eyes sparkling with wonder"),
    "watch":    ("{subject} watches quietly", "{name} watches closely with a smile"),
    "sleep":    ("{subject} settles down to sleep", "{name} rests their head, eyes closing sleepily"),
    "wake":     ("{subject} wakes up with a little stretch", "{name} wakes up with a big yawn and a smile"),
    "eat":      ("{subject} nibbles happily", "{name} takes a happy bite"),
    "drink":    ("{subject} takes a little drink", "{name} takes a happy sip"),
    "jump":     ("{subject} hops up playfully", "{name} jumps up and down with joy"),
    "hop":      ("{subject} hops along happily", "{name} hops along happily"),
    "fly":      ("{subject} flies up into the air", "{name} looks up, following it with delight"),
    "swim":     ("{subject} swims along happily", "{name} claps along at the water's edge"),
    "ring":     ("{subject} gives a cheerful ring", "{name} claps at the cheerful sound"),
    "knock":    ("{subject} gives a friendly knock", "{name} knocks along playfully"),
    "open":     ("{subject} opens up wide", "{name} opens it up with both hands"),
    "close":    ("{subject} closes up gently", "{name} closes it up carefully"),
    "sit":      ("{subject} sits calmly nearby", "{name} sits and smiles"),
    "stand":    ("{subject} stands proudly in place", "{name} stands with a big smile"),
    "walk":     ("{subject} walks along steadily", "{name} walks along, smiling"),
    "look":     ("{subject} seems to look right back", "{name} looks around, eyes wide with wonder"),
    "point":    ("{subject} seems to point the way", "{name} points excitedly"),
    "count":    ("{subject} is counted with delight", "{name} counts along on their fingers"),
    "clap":     ("{subject} claps along in spirit", "{name} claps their hands to the beat"),
    "sing":     ("{subject} hums along happily", "{name} sings along with a happy face"),
    "skip":     ("{subject} skips along in a happy circle", "{name} skips along happily"),
    "spin":     ("{subject} spins around gleefully", "{name} spins around giggling"),
    "dance":    ("{subject} bobs and dances to the music", "{name} dances and bounces to the music"),
    "wiggle":   ("{subject} wiggles happily", "{name} wiggles along to the music"),
    "glow":     ("{subject} glows warmly", "{name} looks up at the warm glow, smiling"),
    "float":    ("{subject} floats gently along", "{name} watches it float by, delighted"),
    "bloom":    ("{subject} blooms bright and open", "{name} cups it gently, beaming"),
    "grow":     ("{subject} grows taller with a happy stretch", "{name} reaches up beside it, smiling"),
    "splash":   ("{subject} makes a little splash", "{name} splashes and giggles"),
    "bounce":   ("{subject} bounces along cheerfully", "{name} bounces along happily"),
    "nod":      ("{subject} nods along to the beat", "{name} nods along with a big smile"),
    "wander":   ("{subject} wanders along gently", "{name} wanders along, looking around with wonder"),
}
# Widen the shared "this clause carries an activity" detection vocabulary the
# same way (`verse_activity_progression` / `verse_mixes_activities` count
# through it) so a beat's verb is recognised as an activity there too, not
# just here.
_ACTIVITY_DETECT_VERBS |= set(_LYRIC_VERBS)

# Verbs a non-child STORY SUBJECT can plausibly perform -- an animal, a clock,
# a boat, a star. The remaining `_LYRIC_VERBS` entries (clap, sing, count,
# point, look, laugh, ring, knock, open, close) read as things a CHILD does in
# these songs, even when a story subject is present in the shot.
_SUBJECT_CAPABLE_VERBS = {
    "run", "ran", "climb", "scamper", "follow", "followed", "go", "went",
    "row", "sail", "drift", "wave", "tiptoe", "creep", "peek", "fly", "swim",
    "eat", "drink", "jump", "hop", "play", "watch", "sleep", "wake", "sit",
    "stand", "walk", "shine", "twinkle", "sparkle", "give", "gives", "fill",
    "carry", "strike", "struck",
}

_LYRIC_VERB_ORDER = sorted(_LYRIC_VERBS, key=len, reverse=True)


def _find_lyric_verb(text):
    """The first `_LYRIC_VERBS` stem `text`'s own words match, in reading
    order -- so "The clock struck one, the mouse ran down" resolves to
    "struck", the earlier event in the line, not "ran"."""
    tokens = [t.strip(".,!?'\"").lower() for t in str(text or "").split()]
    for t in tokens:
        for verb in _LYRIC_VERB_ORDER:
            if t.startswith(verb):
                return verb
    return None


_HOOK_PHRASES = (
    "clap along", "sing with me", "sing along", "come sing", "clap your hands",
    "one, two, three", "one two three", "count to three",
)


def _is_hook_verse(lines):
    """True for a hook/refrain verse -- short, or built around a sing/clap/
    count-along cue. Hook verses REPEAT on purpose (the whole point of a
    refrain); `plan_verse_beats` must not force a setup/event/reaction arc
    onto one -- repetition there is correct, not a defect."""
    text = " ".join(str(l) for l in lines).lower()
    if any(p in text for p in _HOOK_PHRASES):
        return True
    return sum(len(str(l).split()) for l in lines) <= 6


def _beat_action(line, subj, singular, story_subject, role):
    """One shot's action, derived from `line` -- THAT SHOT's own lyric text,
    not the verse as a whole (this is what makes consecutive shots of a
    two-line verse read as different moments)."""
    # When the subject is present, resolve the verb from the CLAUSE that is
    # about the subject, not the whole line — "The clock struck one, the mouse
    # ran down" must give the mouse "ran", not the clock's "struck". Without
    # this the subject performed another actor's verb ("The little mouse has
    # just struck the hour").
    verb_source = subject_clause(line, story_subject) if story_subject else line
    verb = _find_lyric_verb(verb_source)
    # If the subject's own clause carried no depictable verb, fall back to the
    # whole line — "Twinkle, twinkle, little star" narrows to "little star"
    # (no verb), but the line's verb "twinkle" is exactly the star's action.
    if verb is None and story_subject:
        verb = _find_lyric_verb(line)

    if not singular:
        # Ensemble ("all") shots only occur as the very first shot of the
        # whole song, in the current character-rotation scheme.
        if story_subject and role == "setup":
            # If this establishing line gives the subject a characteristic
            # action (the star twinkles, the lamb follows), show it happening
            # WHILE the cast watches — otherwise the subject's one signature
            # beat is lost to the mandatory wide, which is how Twinkle's star
            # ended up never twinkling on screen.
            if verb and verb in _SUBJECT_CAPABLE_VERBS:
                subject_tpl, _ = _LYRIC_VERBS[verb]
                return (
                    f"The kids watch happily as {subject_tpl.format(subject=story_subject)}"
                )
            return f"The kids gather happily and look at {story_subject}"
        return "The kids sing along with happy faces, moving to the beat"

    if not verb:
        if story_subject and role == "setup":
            return f"{subj} looks over at {story_subject}, sitting quietly nearby"
        return f"{subj} sings along with a happy face, moving to the beat"

    subject_tpl, child_tpl = _LYRIC_VERBS[verb]
    # The last beat of an arc is always a CHILD reaction, regardless of which
    # verb its own line carries -- "setup -> event -> reaction" means the
    # reaction is a child's response to what just happened, not a restatement
    # of the event itself.
    if story_subject and role == "reaction":
        return f"{subj} smiles and points at {story_subject}"
    if story_subject and verb in _SUBJECT_CAPABLE_VERBS:
        return subject_tpl.format(subject=story_subject)
    return child_tpl.format(name=subj)


def _storyboard_beats_for_verse(song, verse_index):
    """Validated storyboard beats for `verse_index`, or None.

    `song["storyboard"]` is attached upstream by generate.py when
    kidsong.storyboard.enabled (see pipeline/kidsong/storyboard.py for the
    schema and the gate that already vetted it). This reader stays defensive
    anyway — a hand-edited song.json or a partial board must degrade to the
    planner's own beats, never crash the plan: absent/malformed board or
    verse entry -> None; beats without a usable action string are dropped.
    """
    board = (song or {}).get("storyboard") if isinstance(song, dict) else None
    if not isinstance(board, dict):
        return None
    for entry in board.get("verses") or []:
        if not isinstance(entry, dict):
            continue
        try:
            if int(entry.get("verse", -1)) != verse_index:
                continue
        except (TypeError, ValueError):
            continue
        beats = [
            b for b in (entry.get("beats") or [])
            if isinstance(b, dict) and str(b.get("action") or "").strip()
        ]
        return beats or None
    return None


def _storyboard_beat_action(beat, subject_name, partner_name=None):
    """Render one storyboard beat's action for the shot's planned subject.

    Beats carry the literal "{name}" placeholder (never concrete cast names —
    storyboard.py's gate enforces it; the constant lives there as
    _NAME_PLACEHOLDER, spelled out here to avoid a circular import — the
    storyboard module already imports this one). Substitution:

      performer "child"         -> the planner's own rotated subject. A shot
                                   with NO planned subject (the establishing
                                   wide, or a castless song) returns None so
                                   the planner's canonical establishing/
                                   ensemble action stands — "The kids spots
                                   the tree" grammar can never ship.
      performer "all"           -> "The kids" ("{name}"-led boards stay
                                   grammatical because group beats are written
                                   as group sentences; the substitution is a
                                   safety net, not a grammar engine).
      performer "story_subject" -> the action verbatim (it describes the
                                   subject, carries no "{name}"); the existing
                                   `action_is_subject_only` derivation then
                                   decides whether the shot is subject-only,
                                   keeping ONE definition of that rule.

    A placeholder action ("...", template leftovers) returns None — same
    treatment as an LLM echo, see `is_placeholder_action`.

    `partner_name` (story-casting): a role="partner" beat additionally carries
    the "{partner}" placeholder for the second planned child. Without a
    partner name such a beat cannot render ("passes the toy gently to
    {partner}" would ship a literal placeholder) and returns None — graceful
    degradation to the planner's own action, same as every other unrenderable
    beat.
    """
    action = str((beat or {}).get("action") or "").strip()
    if not action or is_placeholder_action(action):
        return None
    performer = str((beat or {}).get("performer") or "child").strip().lower()
    if performer == "story_subject":
        return action
    if performer == "all":
        return action.replace("{name}", "The kids")
    if not subject_name:
        return None
    action = action.replace("{name}", subject_name)
    if "{partner}" in action:
        if not partner_name:
            return None
        action = action.replace("{partner}", partner_name)
    return action


def plan_verse_beats(
    lines, line_spans, scene, story_subject, subject_names, location, decreep=False,
):
    """One action string per shot of a verse, forming a small arc (setup ->
    event -> reaction) derived from what the LYRIC actually says.

    Each shot's action comes from its OWN `line_spans[j]` slice of `lines`,
    not the verse as a whole, so "The mouse ran up the clock" and "The clock
    struck one, the mouse ran down" -- different shots of different verses --
    produce different actions instead of both collapsing to the same generic
    sing-along fallback. When `story_subject` is set, it is the actor for
    event beats ("a tiny round cartoon mouse scampers up..."); children appear
    as observers/reactors on the surrounding beats. Hook/refrain verses
    (`_is_hook_verse`) and single-shot verses keep the simpler,
    repetition-friendly phrasing from `_line_action` instead -- repetition is
    correct there, not a defect to fix.

    `scene` and `location` are accepted for interface completeness (mirrors
    the verse-level context every other planning helper in this module takes)
    and to leave room for future, richer per-verb templates; the current
    templates deliberately do not restate the location inside `action` text,
    since `generate.py`'s render prompt already appends the shot's `setting`
    separately and doing it twice produced duplicated place names in the
    prompt.

    `decreep` (kidsong.decreep.enabled) is threaded through to `_line_action`
    for the hook/single-shot branch below (the narrative setup/event/reaction
    branch uses `_beat_action`'s own `_LYRIC_VERBS` table, which carries no
    camera language and needs no overlay).
    """
    n = len(line_spans)
    if n == 0:
        return []
    lines = list(lines) or ["La la la"]

    if _is_hook_verse(lines) or n < 2:
        actions = []
        for j, span in enumerate(line_spans):
            subj = subject_names[j] if j < len(subject_names) else None
            singular = bool(subj) and subj != "all"
            actions.append(
                _line_action(lines, span, subject=subj if singular else None, decreep=decreep)
            )
        return actions

    actions = []
    for j, span in enumerate(line_spans):
        subj = subject_names[j] if j < len(subject_names) else None
        singular = bool(subj) and subj != "all"
        role = "setup" if j == 0 else ("reaction" if j == n - 1 else "event")
        idx = min(span[0], len(lines) - 1)
        line = lines[idx]
        actions.append(_beat_action(line, subj, singular, story_subject, role))
    return actions


def _learning_insert_action(story_subject):
    """Action text for a LEARNING-MODE insert shot: the taught item alone,
    named plainly so a toddler can see it large and clear. Deterministic
    (no randomness, no LLM) and always verb-ish — `script_qc._is_verbish`
    requires a recognizable verb, and "shines" (an `_LYRIC_VERBS` entry,
    already a known activity) reliably satisfies it. Must say nothing that
    `implied_child_count` reads as a person or a group ("kids"/"children"/...)
    — this shot's `characters` is `[]`, and the action text has to agree with
    that zero, the same head-count-consistency contract every other shot
    honors (see `_check_head_count_consistency` in script_qc)."""
    subject = str(story_subject or "the item").strip()
    return f"{subject} shines bright and clear, ready to be learned"


# --------------------------------------------------------- deterministic planner ---
def _fallback_planner(song, verse_times, beats, cfg):
    """Deterministic shot list: per verse, a wide->medium->closeup cycle with actions
    derived verbatim from the lyric lines. Used outright when the LLM is unreachable,
    and as the skeleton `_normalize_shots` overlays LLM choices onto otherwise."""
    settings = (cfg or {}).get("kidsong", {}) or {}
    director_cfg = settings.get("director", {}) or {}

    def dopt(key):
        return director_cfg.get(key, _DIRECTOR_DEFAULTS[key])

    # Scene-mode Phase 1: cfg-resolved bounds (kidsong.director.min_shot_seconds /
    # .max_shot_seconds), defaulting to the module constants -- see `_shot_bounds`.
    # Clamping `target` against these (not the fixed module constants) means a
    # raised max_shot_seconds actually lets a raised target_shot_seconds through.
    min_shot_s, max_shot_s = _shot_bounds(cfg)
    target = _clamp(float(dopt("target_shot_seconds")), min_shot_s, max_shot_s)
    max_unique = int(dopt("max_unique_shots"))
    reuse_chorus = bool(dopt("reuse_chorus_shots"))
    base_seed = int(settings.get("seed", 20260717))
    # LEARNING BIAS (kidsong.content_mode="learning", default "song"): when a
    # verse teaches one item (its own explicit `story_subject`, see below),
    # bias that verse's shots to show the item large and clear before the
    # kids react to it. Reading this straight off `settings` (not threaded as
    # a separate kwarg) keeps `plan_shots`/`_fallback_planner`'s signature
    # unchanged, so a caller that never sets content_mode gets byte-identical
    # output to before this existed.
    learning_mode = str(settings.get("content_mode", "song")).strip().lower() == "learning"

    # De-creep (kidsong.decreep.enabled, default False -> every use of
    # `decreep` below is False and this function's output is byte-identical
    # to before this existed). See `_decreep_cfg`.
    decreep_cfg = _decreep_cfg(cfg)
    decreep_enabled = decreep_cfg["enabled"]

    verses = song.get("verses") or []
    names = _parse_character_names(song.get("characters", ""))
    # The song's single hero subject (a mouse, a lamb, a boat), resolved once so
    # every verse agrees on who the protagonist is.
    song_subject = song_story_subject(song)

    # STORY-DRIVEN CASTING (kidsong.staging="story", default "ensemble" =
    # byte-identical): the storyboard owns WHO is in each beat — one
    # protagonist child carries the arc through every verse (the red thread in
    # casting form), a partner appears only in the board's role="partner"
    # beat, "all" only at gathering moments. The mechanical round-robin
    # (names[j % len(names)] below) is exactly the anti-pattern this replaces:
    # characters changing shot-to-shot for no story reason. Requires an
    # attached board — without one there is nothing to cast FROM, so the
    # rotation stays (logged), byte-identical.
    story_mode = (
        str(settings.get("staging", "ensemble")).strip().lower() == "story"
    )
    protagonist_name, partner_name = None, None
    if story_mode:
        if isinstance(song.get("storyboard"), dict) and song["storyboard"].get("verses"):
            try:
                from pipeline.kidsong import cast as _cast

                prot_id = _cast.protagonist_for(song.get("title"), cfg)
                partner_id = _cast.partner_for(prot_id, cfg)
                protagonist_name = (_cast.character(prot_id) or {}).get("name")
                partner_name = (_cast.character(partner_id) or {}).get("name")
            except Exception:
                log.exception("story casting: could not resolve protagonist/partner")
        if not protagonist_name:
            story_mode = False
            log.warning(
                "kidsong.staging='story' but no usable storyboard/cast — "
                "falling back to the ensemble rotation for this plan."
            )
    beat_times = []
    if isinstance(beats, dict):
        beat_times = [float(b) for b in (beats.get("beat_times") or []) if isinstance(b, (int, float))]

    chorus_source = _detect_chorus(verses) if reuse_chorus else {}

    # The FINAL verse is the narrative finale. When a storyboard drives the arc
    # (kidsong.storyboard.enabled), its last verse is the distinct "celebrate"
    # climax — so never let chorus reuse collapse the last verse onto an earlier
    # identical-lyric verse (nursery choruses routinely recur AS the closing
    # verse). Collapsing it replayed a middle verse's footage as the ending and,
    # when the finale needed more shots than its source, cycled `src_ids` into a
    # within-verse repeat — measured on "Rain, Rain, Go Away", which ended by
    # reusing its clap verse with s16 == s20. Reprises EARLIER in the song still
    # reuse; only the finale is protected, and only when the board actually has
    # distinct beats to render there (else behaviour is byte-identical).
    if chorus_source and verses and _storyboard_beats_for_verse(song, len(verses) - 1):
        chorus_source.pop(len(verses) - 1, None)

    # One location per verse, chosen with continuity across the whole song
    # rather than per verse in isolation (see `plan_verse_locations`).
    verse_locations = plan_verse_locations(song)

    # ---- initial per-(non-chorus)-verse shot counts, then merge down to the cap ----
    counts = {}
    durations = {}
    for i, vt in enumerate(verse_times):
        if i >= len(verses) or i in chorus_source:
            continue
        vs, ve = vt
        d = max(0.0, float(ve) - float(vs))
        durations[i] = d
        counts[i] = _num_shots_for(d, target, min_shot_s, max_shot_s)

    def total_unique():
        return sum(counts.values())

    while total_unique() > max_unique:
        candidates = [
            i for i in counts
            if counts[i] > 1 and durations[i] / (counts[i] - 1) <= max_shot_s + 1e-9
        ]
        if not candidates:
            break  # can't reduce further without breaking the duration bound
        # Merge the shortest adjacent same-verse shots first (lowest current per-shot
        # duration); tie-break by verse index for determinism.
        i = min(candidates, key=lambda k: (durations[k] / counts[k], k))
        counts[i] -= 1

    # ---- build the actual shots, verse by verse, in song order ----
    shots = []
    id_counter = 0
    verse_shot_ids = {}

    for i, v in enumerate(verses):
        if i >= len(verse_times):
            break
        vs, ve = verse_times[i]
        vs, ve = float(vs), float(ve)
        lines = [_clean_line(l) for l in (v.get("lines") or []) if _clean_line(l)] or ["La la la"]
        # ONE VERSE, ONE ACTIVITY: a scene handing each child a different thing
        # to do is collapsed to a single shared activity before any shot text is
        # derived from it.
        raw_scene = str(v.get("scene") or "").strip()
        # An EXPLICIT per-verse story_subject (learning songs set one on every
        # verse, see lyrics.pd_song_to_dict / prompts/learning_songs.json)
        # always wins: it overrides both the song-level hero and the
        # extract_story_subject gate below, because a learning verse's taught
        # item (letter B, the number 3, a square) is verse-specific by
        # design — unlike a nursery rhyme's one stable protagonist. Verses
        # that carry no such field are completely unaffected: `explicit_subject`
        # is falsy and this falls through to the existing behaviour unchanged.
        explicit_subject = v.get("story_subject") if isinstance(v, dict) else None
        explicit_subject_active = bool(explicit_subject) and valid_story_subject(explicit_subject)
        if explicit_subject_active:
            story_subject = _clean_line(explicit_subject)
        else:
            # ONE hero subject for the whole song (computed once, above the loop) —
            # NOT re-extracted per verse, which made the hero drift from "a tiny
            # round cartoon mouse" to "the big clock" mid-song. A verse only carries
            # the subject if the subject actually belongs in it (its scene hint or a
            # lyric line names it); otherwise its shots are pure children (the
            # sing-along hook verses).
            verse_names_subject = extract_story_subject(raw_scene, lines) is not None
            story_subject = song_subject if verse_names_subject else None
        scene = reduce_scene_to_shared_activity(raw_scene, decreep=decreep_enabled)
        location = verse_locations[i] if i < len(verse_locations) else _location_from(scene)

        if i in chorus_source:
            src = chorus_source[i]
            src_ids = verse_shot_ids.get(src, [])
            d = max(0.0, ve - vs)
            n = _num_shots_for(d, target, min_shot_s, max_shot_s) if d > 0 else 1
            time_cuts = _tile(vs, ve, n, beat_times, min_shot_s, max_shot_s)
            line_cuts = _line_breaks(len(lines), n)

            this_ids = []
            for j in range(n):
                sid = f"s{id_counter:02d}"
                id_counter += 1
                ref_shot = None
                if src_ids:
                    ref_id = src_ids[j % len(src_ids)]
                    ref_shot = next((s for s in shots if s["id"] == ref_id), None)
                shot = {
                    "id": sid,
                    "verse": i,
                    "lyric_span": list(line_cuts[j]),
                    "start": round(time_cuts[j][0], 3),
                    "end": round(time_cuts[j][1], 3),
                    "shot_type": ref_shot["shot_type"] if ref_shot else "wide",
                    "characters": list(ref_shot["characters"]) if ref_shot else ["all"],
                    "action": (
                        ref_shot["action"] if ref_shot
                        else _line_action(lines, line_cuts[j], decreep=decreep_enabled)
                    ),
                    "camera": "static",
                    "setting": ref_shot["setting"] if ref_shot else location,
                    "reuse_of": ref_shot["id"] if ref_shot else (src_ids[0] if src_ids else None),
                    "seed": ref_shot["seed"] if ref_shot else base_seed,
                    "story_subject": (
                        ref_shot.get("story_subject") if ref_shot else story_subject
                    ),
                    "status": "planned",
                }
                shots.append(shot)
                this_ids.append(sid)
            verse_shot_ids[i] = this_ids
            continue

        n = counts.get(i, 1)
        time_cuts = _tile(vs, ve, n, beat_times, min_shot_s, max_shot_s)
        line_cuts = _line_breaks(len(lines), n)

        # Pre-assign each shot's child subject, then derive all of the verse's
        # actions in one pass — `plan_verse_beats` needs the whole verse to shape
        # an arc (its reaction beat is the LAST shot of a narrative verse).
        planned_subjects = []
        for j in range(n):
            if i == 0 and j == 0:
                planned_subjects.append(None)  # the establishing wide is the cast
            elif story_mode:
                # The protagonist carries every body slot — the board's beats
                # (partner/all roles) override staging per-shot below.
                planned_subjects.append(protagonist_name)
            elif names:
                planned_subjects.append(names[j % len(names)])
            else:
                planned_subjects.append(None)
        verse_actions = plan_verse_beats(
            lines, line_cuts, scene, story_subject, planned_subjects, location,
            decreep=decreep_enabled,
        )
        # STORYBOARD consumption. When generate.py attached a gated board to
        # the song (kidsong.storyboard.enabled), its beats REPLACE the canned
        # per-shot actions above — this is the seam where "sings along with a
        # happy face" filler dies. Beat j drives shot j, cycling when a verse
        # has more shots than beats. Per-shot graceful degradation: a beat
        # that can't render for this shot (placeholder action, or a "child"
        # beat on the subject-less establishing wide) leaves that one shot on
        # the planner's own action. Gazes ride along (validated the same way
        # `_normalize_shots` validates an LLM gaze; "at the camera" is NEVER
        # taken from a board — the decreep designation pass below remains the
        # single owner of the camera budget, so designation + storyboard can
        # never exceed it between them). Absent/malformed board: `sb_beats`
        # is None and every line below is a no-op — byte-identical.
        verse_gazes = [None] * n
        verse_from_storyboard = [False] * n
        verse_sb_ensemble = [False] * n
        verse_sb_partner = [False] * n
        verse_props = [None] * n
        sb_beats = _storyboard_beats_for_verse(song, i)
        if sb_beats:
            # No storyboard action may appear TWICE in one verse. The old
            # plain `j % len(sb_beats)` cycling repeated beats verbatim once a
            # verse had more shots than the board had beats — measured live: a
            # 7-shot celebrate verse over a 4-beat pool planned "The kids join
            # hands and circle …" twice AND "wave … one last time as the song
            # ends" twice (an absurdity — there is only one last time), which
            # rendered as near-identical scenes and read as the cut looping.
            # Each slot now takes the first STILL-UNUSED beat (scanning from
            # its natural cycle position); when every beat's rendered action
            # is already on screen this verse, the slot simply keeps the
            # planner's own action — the same graceful degradation as an
            # unrenderable beat, and the planner's own variety machinery
            # (performer rotation + vary_repeated_actions) takes it from there.
            used_sb_actions = set()
            for j in range(n):
                if i == 0 and j == 0:
                    # The establishing wide is a hard invariant: it shows the
                    # whole cast with its canonical establishing action. No
                    # board beat may replace it — a story_subject beat here
                    # would describe only the mouse while `characters` says
                    # three kids, exactly the action/head-count contradiction
                    # the planner exists to prevent.
                    continue
                beat, sb_action = None, None
                for probe in range(len(sb_beats)):
                    cand = sb_beats[(j + probe) % len(sb_beats)]
                    cand_action = _storyboard_beat_action(
                        cand, planned_subjects[j], partner_name
                    )
                    if cand_action and cand_action not in used_sb_actions:
                        beat, sb_action = cand, cand_action
                        break
                if not sb_action:
                    continue
                used_sb_actions.add(sb_action)
                verse_actions[j] = sb_action
                verse_from_storyboard[j] = True
                # Structured prop passthrough: the beat's interaction object
                # rides into the shot so the keyframe prop-presence gate knows
                # what must be visible before the shot animates. Board-less
                # plans never set the key — byte-identical ledgers.
                raw_prop = beat.get("prop")
                verse_props[j] = (_clean_line(raw_prop) or None) if raw_prop else None
                # A role="partner" beat is the story's planned two-child
                # moment: the shot stages BOTH the protagonist and the
                # partner (never a solo shot whose text implies an invisible
                # friend — the measured non-cast-child invasion vector).
                verse_sb_partner[j] = (
                    story_mode
                    and str(beat.get("role") or "").strip().lower() == "partner"
                    and bool(partner_name)
                )
                # An ensemble beat must STAGE as an ensemble: with the slot's
                # single rotated child in `characters` and "The kids ..." in
                # the action, the head-count repair machinery downstream sees
                # a contradiction and flattens the beat back to canned filler
                # (measured live: "The kids join hands and circle the tree"
                # became "sings along with a happy face" through exactly that
                # path). The shot branch below reads this flag to set
                # characters=["all"] (and widen a closeup slot to medium — a
                # group activity has no single-face framing).
                verse_sb_ensemble[j] = (
                    str(beat.get("performer") or "").strip().lower() == "all"
                )
                raw_gaze = str(beat.get("gaze") or "").strip()
                if (
                    raw_gaze
                    and len(raw_gaze) <= 80
                    and not is_placeholder_action(raw_gaze)
                    and "camera" not in raw_gaze.lower()
                    and "lens" not in raw_gaze.lower()
                ):
                    verse_gazes[j] = raw_gaze
        # A beat whose action is ABOUT the story subject and names no child is a
        # subject-only shot (the mouse alone on the clock). Deriving this from
        # the action text keeps one definition of the rule — the same predicate
        # `_normalize_shots` applies to an LLM-supplied action.
        verse_beats = [
            (
                act,
                bool(story_subject)
                and not (i == 0 and j == 0)
                and action_is_subject_only(act, story_subject),
            )
            for j, act in enumerate(verse_actions)
        ]

        # LEARNING BIAS placement: reserve this verse's FIRST available shot
        # (never shot 0 of the whole song — that establishing wide must keep
        # showing the cast, a hard invariant enforced again below and by
        # script_qc's "first shot not wide" gate) as an INSERT shot naming the
        # taught item, and its LAST shot (if a distinct one exists) as a
        # reaction shot. Both stay None — a complete no-op — unless learning
        # mode is on AND this verse actually carries an explicit subject, so a
        # normal song's shot list is byte-identical to before this existed.
        insert_idx = None
        reaction_idx = None
        if learning_mode and explicit_subject_active and not (i == 0 and n == 1):
            insert_idx = 1 if (i == 0 and n > 1) else 0
            if n - 1 > insert_idx:
                reaction_idx = n - 1

        this_ids = []
        non_static_used = 0
        for j in range(n):
            sid = f"s{id_counter:02d}"
            id_counter += 1

            beat_action, beat_subject_only = verse_beats[j]

            if insert_idx is not None and j == insert_idx:
                # The taught item, shown large and clear, no people —
                # generate.py's `_shot_prompt` renders shot_type=="insert" as
                # "An insert closeup of an object, no people: {action}, in
                # {setting}.", which is the reliable path for a legible
                # letter/number/shape (video diffusion renders on-screen text
                # poorly; a clean insert of the item is what a toddler needs).
                shot_type = "insert"
                characters = []
                beat_action = _learning_insert_action(story_subject)
            elif reaction_idx is not None and j == reaction_idx:
                # A child reacting to the item just shown — the same
                # "smiles and points at {story_subject}" beat `_beat_action`
                # already uses for a narrative verse's reaction shot, applied
                # directly here so it does not depend on `_is_hook_verse`
                # classification (this channel's hook lines routinely match
                # `_HOOK_PHRASES`, which would otherwise route the verse
                # through the simpler `_line_action` path and skip it).
                shot_type = ("medium", "closeup")[j % 2]
                reactor = names[j % len(names)] if names else None
                if reactor:
                    characters = [reactor]
                    beat_action = f"{reactor} smiles and points at {story_subject}"
                else:
                    characters = ["all"]
                    beat_action = f"The kids smile and point at {story_subject}"
            elif i == 0 and j == 0:
                shot_type = "wide"
                characters = ["all"]
            else:
                # Reference analysis: the show's body is ~50/50 medium/closeup;
                # wides are essentially opener-only. Alternate M/C.
                shot_type = ("medium", "closeup")[j % 2]
                if beat_subject_only:
                    # This beat belongs to the story subject alone (the mouse on
                    # the clock). No child on screen — see `action_is_subject_only`.
                    characters = []
                elif verse_from_storyboard[j] and verse_sb_ensemble[j]:
                    # A storyboard ensemble beat ("The kids join hands…") stages
                    # the whole cast; a closeup slot widens to medium — a group
                    # activity has no single-face framing, and a closeup that
                    # says "the kids" would be flattened right back to filler by
                    # the closeup-narrowing repair in `_normalize_shots`.
                    if shot_type == "closeup":
                        shot_type = "medium"
                    characters = ["all"]
                elif verse_from_storyboard[j] and verse_sb_partner[j]:
                    # The board's planned two-child moment: protagonist AND
                    # partner, explicitly cast (mirror of the ensemble widening
                    # above — a two-child handover has no single-face framing,
                    # so never closeup).
                    if shot_type == "closeup":
                        shot_type = "medium"
                    characters = [protagonist_name, partner_name]
                elif story_mode:
                    # The red thread: the protagonist carries every body slot
                    # (no rotation — characters changing shot-to-shot for no
                    # story reason is the anti-pattern story mode removes).
                    characters = [protagonist_name]
                elif names:
                    characters = [names[j % len(names)]]
                else:
                    characters = ["all"]

            # Reference analysis: ~90% of real shots have gentle continuous
            # motion; true statics are bumpers. Default to a slow drift and
            # alternate direction for variety.
            if j % 3 == 0:
                camera = "slow push-in"
            elif j % 3 == 1:
                camera = "gentle pan left" if (i + j) % 2 == 0 else "gentle pan right"
            else:
                camera = "slow pull-back"
            non_static_used += 1

            seed = _seed_for_characters(characters, base_seed)

            shot = {
                "id": sid,
                "verse": i,
                "lyric_span": list(line_cuts[j]),
                "start": round(time_cuts[j][0], 3),
                "end": round(time_cuts[j][1], 3),
                "shot_type": shot_type,
                "characters": characters,
                "action": beat_action,
                "camera": camera,
                "setting": location,
                "reuse_of": None,
                "seed": seed,
                "story_subject": story_subject,
                "status": "planned",
            }
            # Storyboard provenance — only when this shot's action really is
            # the board's beat (the learning insert/reaction branches above
            # overwrite `beat_action`, so those shots stay unmarked and keep
            # full LLM-overlay behaviour). `action_source` shields the beat
            # from the LLM action overlay in `_normalize_shots`; `gaze` feeds
            # generate.py's `_gaze_sentence`.
            if (
                verse_from_storyboard[j]
                and not (insert_idx is not None and j == insert_idx)
                and not (reaction_idx is not None and j == reaction_idx)
            ):
                shot["action_source"] = "storyboard"
                if verse_gazes[j]:
                    shot["gaze"] = verse_gazes[j]
                if verse_props[j]:
                    # The beat's interaction object, structured — the keyframe
                    # prop-presence gate reads this to verify the object is
                    # actually in frame before the shot animates.
                    shot["prop"] = verse_props[j]
            shots.append(shot)
            this_ids.append(sid)
        verse_shot_ids[i] = this_ids

    # CLOSING-GROUP invariant (story mode only): the episode's second bookend.
    # The opening wide is a hard invariant above; story mode adds its mirror —
    # the LAST non-reuse shot of the final verse stages the whole cast, so the
    # red thread resolves with everyone together no matter how the board's
    # celebrate beats happened to land on the verse's slots. Ensemble mode
    # never runs this (byte-identical).
    if story_mode and shots:
        last_verse = max(s["verse"] for s in shots)
        finale = [s for s in shots if s["verse"] == last_verse and not s.get("reuse_of")]
        if finale:
            closer = finale[-1]
            if closer.get("characters") != ["all"]:
                closer["characters"] = ["all"]
                if closer.get("shot_type") == "closeup":
                    closer["shot_type"] = "medium"
                action = str(closer.get("action") or "")
                # A protagonist-solo action on a forced group shot is the exact
                # action/head-count contradiction the planner exists to prevent
                # — swap in a still-unused celebrate group beat, else the
                # canonical celebration line.
                if protagonist_name and protagonist_name in action:
                    replacement = None
                    for cand in _storyboard_beats_for_verse(song, last_verse) or []:
                        if str(cand.get("performer") or "").strip().lower() != "all":
                            continue
                        cand_action = _storyboard_beat_action(cand, None, partner_name)
                        if cand_action and all(
                            cand_action != str(s.get("action")) for s in shots
                        ):
                            replacement = (cand_action, cand)
                            break
                    if replacement:
                        closer["action"] = replacement[0]
                        raw_gaze = str(replacement[1].get("gaze") or "").strip()
                        if raw_gaze and "camera" not in raw_gaze.lower():
                            closer["gaze"] = raw_gaze
                    else:
                        closer["action"] = "The kids wave goodbye together, all smiles"
                        closer.pop("gaze", None)
                closer["seed"] = _seed_for_characters(closer["characters"], base_seed)

    # De-creep step: designate this episode's ONE "plays to the camera" beat
    # (kidsong.decreep.enabled, default False -> this block never runs and no
    # shot ever gets a "gaze" key -- the flag-off shot list is byte-identical
    # to before this existed). A hook/refrain verse ("clap along", "sing with
    # me", or just a short repeated line -- see `_is_hook_verse`) is the one
    # place a real preschool show DOES play directly to the lens; capping this
    # to `hook_camera_budget` (default 1) shots, and ONLY on a hook verse,
    # keeps direct address a deliberate performance beat instead of the
    # pervasive stare this feature exists to remove. Recomputes each verse's
    # `lines` with the exact same expression used earlier in this function,
    # rather than threading hook status out of the main loop above, so this
    # stays a single, self-contained pass over the finished shot list.
    if decreep_enabled:
        budget = decreep_cfg["hook_camera_budget"]
        for i, v in enumerate(verses):
            if budget <= 0:
                break
            if i >= len(verse_times):
                break
            v_lines = [_clean_line(l) for l in (v.get("lines") or []) if _clean_line(l)] or ["La la la"]
            if not _is_hook_verse(v_lines):
                continue
            first_shot = next((s for s in shots if s["verse"] == i), None)
            if first_shot is None:
                continue
            first_shot["gaze"] = "at the camera"
            budget -= 1

    return vary_repeated_actions(shots, decreep=decreep_enabled)


# ------------------------------------------------------------- normalization ---
def _enforce_camera_variety(shots):
    """Scene-mode Phase 1 camera-variety post-pass (opt-in via
    kidsong.director.enforce_camera_variety, default False -> never called).

    Rewrites the `camera` field so that no two ADJACENT non-reuse shots share
    the same camera move, cycling deterministically through `_CAMERA_PALETTE`.
    Only ever touches `camera` -- every other field (start/end/duration/
    characters/setting/story_subject/reuse_of/seed/shot_type/action) is left
    exactly as `_normalize_shots` produced it. The first shot (the wide
    establishing opener) keeps "static" if it already is one -- that shot is
    the cast intro and is deliberately exempt from the variety cycle. A shot
    with `reuse_of` set inherits its SOURCE shot's (already-rewritten) camera
    rather than diverging on its own -- a reused shot is meant to look like a
    repeat of the shot it's reusing, not a fresh choice.
    """
    if not shots:
        return shots

    out = [dict(s) for s in shots]
    by_id = {s["id"]: s for s in out}

    palette_idx = 0
    last_camera = None
    for i, shot in enumerate(out):
        reuse_of = shot.get("reuse_of")
        if reuse_of and reuse_of in by_id:
            shot["camera"] = by_id[reuse_of]["camera"]
            continue
        if i == 0 and shot.get("camera") == "static":
            last_camera = "static"
            continue
        camera = _CAMERA_PALETTE[palette_idx % len(_CAMERA_PALETTE)]
        palette_idx += 1
        if camera == last_camera:
            camera = _CAMERA_PALETTE[palette_idx % len(_CAMERA_PALETTE)]
            palette_idx += 1
        shot["camera"] = camera
        last_camera = camera
    return out


def _normalize_shots(raw, song, verse_times, beats, cfg):
    """Coerce raw LLM output into the hard-constrained shot list. The deterministic
    skeleton from `_fallback_planner` always defines structure (ids, verse, timing,
    lyric_span, reuse_of, seed, status); the LLM (if `raw` is usable) only gets to
    influence shot_type / camera / characters / action, and even those overlays are
    re-validated against the kids-TV-grammar rules afterwards."""
    skeleton = _fallback_planner(song, verse_times, beats, cfg)

    raw_shots = raw.get("shots") if isinstance(raw, dict) else None
    if not isinstance(raw_shots, (list, tuple)) or not raw_shots:
        return skeleton

    names_allowed = _cast_name_pool(song)

    by_verse = {}
    for rs in raw_shots:
        if not isinstance(rs, dict):
            continue
        try:
            v = int(rs.get("verse"))
        except Exception:
            continue
        by_verse.setdefault(v, []).append(rs)

    base_seed = int(((cfg or {}).get("kidsong", {}) or {}).get("seed", 20260717))

    # De-creep (kidsong.decreep.enabled, default False -> every use of
    # `decreep_enabled` below is False and this function's output is
    # byte-identical to before this existed). See `_decreep_cfg`.
    decreep_enabled = _decreep_cfg(cfg)["enabled"]

    result = []
    verse_type_history = {}
    verse_camera_count = {}
    verse_position = {}

    for shot in skeleton:
        v = shot["verse"]
        idx_in_verse = verse_position.get(v, 0)
        verse_position[v] = idx_in_verse + 1

        pool = by_verse.get(v) or []
        overlay = pool[idx_in_verse] if idx_in_verse < len(pool) else None

        merged = dict(shot)

        if overlay is not None and shot["reuse_of"] is None:
            st = str(overlay.get("shot_type") or "").strip().lower()
            if st in _VALID_SHOT_TYPES:
                merged["shot_type"] = st

            cam = str(overlay.get("camera") or "").strip().lower()
            if cam in _VALID_CAMERAS:
                merged["camera"] = cam

            chars = overlay.get("characters")
            if isinstance(chars, (list, tuple)):
                cleaned = [str(c).strip() for c in chars if str(c).strip()]
                cleaned = [c for c in cleaned if c.lower() == "all" or c in names_allowed]
                # An explicit `[]` is a legitimate overlay for a subject-only shot
                # ("a tiny round cartoon mouse scampers down the clock case", no
                # child on screen) and must survive -- only a list that had names
                # but lost ALL of them to validation (garbage/off-cast names)
                # falls back to the skeleton's own characters.
                if cleaned or len(chars) == 0:
                    merged["characters"] = cleaned

            # A story subject the LLM proposed, validated (short, names no cast
            # child, nothing scary/martial). Falls back to the skeleton's, which
            # was derived from the verse's own scene hint.
            raw_subject = overlay.get("story_subject")
            if valid_story_subject(raw_subject):
                merged["story_subject"] = _clean_line(raw_subject)

            # De-creep step 6 (kidsong.decreep.enabled, default False -> this
            # block never runs and a stray "gaze" key from an LLM response is
            # silently ignored, same as any other field this overlay does not
            # recognize -- `merged` never carries a "gaze" key when the flag
            # is off). The LLM may propose a per-shot gaze direction for
            # generate.py's `_gaze_sentence`. "at the camera" is the one value
            # that can reintroduce the pervasive blank-stare defect this
            # feature removes, so it is honored ONLY when the skeleton above
            # already earmarked this exact shot as the episode's single
            # hook-verse camera beat (`_fallback_planner`'s budgeted
            # designation) -- never merely because the LLM asked for it. Any
            # other gaze text is free-form direction, not a camera cue, and
            # passes straight through.
            if decreep_enabled:
                raw_gaze = overlay.get("gaze")
                if (
                    isinstance(raw_gaze, str)
                    and raw_gaze.strip()
                    and len(raw_gaze.strip()) <= 80
                    and "\n" not in raw_gaze and "\r" not in raw_gaze
                    # The LLM echoes the response template's "gaze": "..." the
                    # same way it echoes "action": "..." (see
                    # is_placeholder_action) — a literal-ellipsis gaze then
                    # renders as the nonsense sentence "They look ...". Treat
                    # placeholders as absent so the renderer's default eyeline
                    # (story subject / each other / own hands) wins. Measured
                    # live: plan-A/B arm B shots s00/s05 carried gaze "...".
                    and not is_placeholder_action(raw_gaze)
                ):
                    gaze = raw_gaze.strip()
                    if gaze == "at the camera" and merged.get("gaze") != "at the camera":
                        log.info(
                            "Shot %s: LLM proposed gaze 'at the camera' without the "
                            "hook-verse camera designation — dropped to avoid "
                            "reintroducing the blank-stare default",
                            merged.get("id"),
                        )
                    else:
                        merged["gaze"] = gaze

            action = str(overlay.get("action") or "").strip()
            # `is_placeholder_action` catches the LLM echoing the response
            # template ("..."), which is truthy and used to silently replace the
            # skeleton's real action with three dots.
            if action and is_placeholder_action(action):
                log.info(
                    "Shot %s: LLM returned a placeholder action (%r) — keeping "
                    "the deterministic skeleton's action instead",
                    merged.get("id"), action,
                )
                action = ""
            # A storyboard-derived beat outranks the LLM overlay: the board was
            # already gated (real verb, non-martial, coherent arc, topic-tied —
            # see storyboard._storyboard_verdict), which is strictly more than
            # this ungated overlay can promise. shot_type/camera/characters/gaze
            # overlays above/below keep their normal behaviour — only the ACTION
            # is shielded.
            if action and merged.get("action_source") == "storyboard":
                log.info(
                    "Shot %s: keeping the gated storyboard beat over the LLM "
                    "action overlay (%r)", merged.get("id"), action[:60],
                )
                action = ""
            if action:
                # A name the LLM invented has to become a real bible child
                # BEFORE any of the name-aware passes below run: they all match
                # against `names_allowed`, so an off-bible name is invisible to
                # every one of them and survives verbatim into the render
                # prompt. Doing it here also lets the realignment right below
                # count that child correctly.
                action = repair_offbible_names(
                    action, names_allowed, merged.get("story_subject")
                )
                merged["action"] = action
                # LLMs often invert subject and observer ("closeup of Zuri" with
                # action "Amira and Kofi smiling at Zuri"). The action is the
                # ground truth of what's on screen — realign `characters` to the
                # cast names the action actually mentions.
                mentioned = [
                    n for n in names_allowed
                    if n != "all" and n.lower() in action.lower()
                ]
                if mentioned:
                    lowered = action.lower()
                    if any(w in lowered for w in ("three friends", "all three", "the kids", "everyone")):
                        merged["characters"] = ["all"]
                    else:
                        merged["characters"] = sorted(mentioned)

        # -- never 3x the same shot_type consecutively within a verse --
        hist = verse_type_history.setdefault(v, [])
        if len(hist) >= 2 and hist[-1] == hist[-2] == merged["shot_type"]:
            merged["shot_type"] = shot["shot_type"]  # revert to the skeleton's cycled value
        hist.append(merged["shot_type"])

        # -- the very first shot of verse 1 is always a wide establishing shot --
        if v == 0 and idx_in_verse == 0:
            merged["shot_type"] = "wide"

        # Gentle continuous motion is the norm (reference: ~90% of real shots
        # move). Only guard against every shot in a verse using the SAME move.
        count = verse_camera_count.get(v, 0)
        if merged["camera"] == "static" and idx_in_verse > 0:
            merged["camera"] = "slow push-in" if count % 2 == 0 else "gentle pan right"
        if merged["camera"] != "static":
            verse_camera_count[v] = count + 1

        # -- a shot whose action is ABOUT the story subject and names no child
        # -- carries no children at all. Done before the closeup narrowing,
        # -- which would otherwise force a child onto a mouse's shot. The verse-0
        # -- establishing wide is exempt: it must show the cast.
        if (
            merged.get("story_subject")
            and not (v == 0 and idx_in_verse == 0)
            and action_is_subject_only(
                merged.get("action"), merged.get("story_subject"), names_allowed
            )
        ):
            if merged.get("characters"):
                log.info(
                    "Shot %s: action is about the story subject (%r) and names no "
                    "child — rendering it as a subject-only shot (characters %r -> [])",
                    merged.get("id"), merged.get("story_subject"),
                    merged.get("characters"),
                )
            merged["characters"] = []

        subject_only = not merged.get("characters")

        # -- BUG 3: a closeup must name exactly ONE character. The action-
        # realignment above (and a raw LLM overlay's own "characters" list)
        # can otherwise expand a closeup to the whole cast, or leave it
        # naming "all" — narrow to a single, deliberately-chosen subject and
        # scrub the other children's names out of the action text, since
        # generate.py's render prompt uses `action` verbatim.
        if merged["shot_type"] == "closeup" and not subject_only:
            chars = merged.get("characters")
            if not isinstance(chars, list) or len(chars) != 1 or chars[0] == "all":
                before = chars
                subject = _closeup_subject(merged, shot, names_allowed)
                dropped = [
                    n for n in names_allowed
                    if n != "all" and n != subject
                    and n.lower() in str(merged.get("action") or "").lower()
                ]
                if dropped:
                    merged["action"] = _simplify_closeup_action(
                        merged.get("action", ""), subject, decreep=decreep_enabled
                    )
                merged["characters"] = [subject]
                log.info(
                    "Shot %s: closeup narrowed from %r to a single character (%s)",
                    merged.get("id"), before, subject,
                )

        # -- head-count consistency: the action/setting prose must not put more
        # children on screen than this shot's own `characters` list does. The
        # closeup narrowing above only rewrites the action when it had to CHANGE
        # `characters` first, so a shot that already named exactly one child kept
        # whatever group prose the LLM wrote ("exactly one child: Zuri" + "The
        # kids smile brightly at each other") — the self-contradiction the A/B
        # rounds measured. Settings are reduced to places unconditionally.
        chars = merged.get("characters") or []
        single = len(chars) == 1 and chars[0] != "all"
        subject = chars[0] if single else None

        # -- martial/lockstep or frightening staging is never renderable here --
        before_staging = merged.get("action")
        merged["action"] = sanitize_staging(before_staging, subject, decreep=decreep_enabled)
        if merged["action"] != before_staging:
            log.info(
                "Shot %s: action used martial or frightening staging, restated "
                "as loose playful staging (%r -> %r)",
                merged.get("id"), before_staging, merged["action"],
            )

        # -- forbidden places (tub/bath/pool/potty) fall back to the verse's
        # -- planned, compliant location from the skeleton --
        before_location = merged.get("setting")
        merged["setting"] = sanitize_location(before_location, shot.get("setting"))
        if merged["setting"] != before_location:
            log.info(
                "Shot %s: setting named a forbidden place, replaced (%r -> %r)",
                merged.get("id"), before_location, merged["setting"],
            )

        before_setting = merged.get("setting")
        merged["setting"] = sanitize_setting(before_setting)
        if merged["setting"] != before_setting:
            log.info(
                "Shot %s: setting described people, reduced to a location (%r -> %r)",
                merged.get("id"), before_setting, merged["setting"],
            )

        if single:
            before_action = merged.get("action")
            merged["action"] = sanitize_action(
                before_action, 1, subject=subject, names_allowed=names_allowed,
                decreep=decreep_enabled,
            )
            if merged["action"] != before_action:
                log.info(
                    "Shot %s: action implied a group on a single-character shot, "
                    "restated for %s (%r -> %r)",
                    merged.get("id"), subject, before_action, merged["action"],
                )

        # -- reseed if the LLM changed the character set for this shot --
        if merged["characters"] != shot["characters"]:
            merged["seed"] = _seed_for_characters(merged["characters"], base_seed)

        result.append(merged)

    return vary_repeated_actions(result, decreep=decreep_enabled)


# ------------------------------------------------------------------- output ---
def save_shotlist(shotlist, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(shotlist, f, indent=2, ensure_ascii=False)
    return path


# --------------------------------------------------------------------- main ---
if __name__ == "__main__":
    from pipeline.config import load_config
    from pipeline.kidsong.lyrics import FALLBACK_SONG

    cfg = load_config()
    song = json.loads(json.dumps(FALLBACK_SONG))  # cheap deep copy
    verse_times = [(0, 15), (15, 30), (30, 45), (45, 60)]
    beats = {"bpm": 96, "beat_times": [i * 0.625 for i in range(97)]}

    result = plan_shots(song, verse_times, beats, cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    shots = result["shots"]

    # -- shots tile each verse's [start, end) contiguously --
    by_verse = {}
    for s in shots:
        by_verse.setdefault(s["verse"], []).append(s)
    for v, vshots in by_verse.items():
        vshots = sorted(vshots, key=lambda s: s["start"])
        for a, b in zip(vshots, vshots[1:]):
            assert abs(b["start"] - a["end"]) < 0.01, f"gap in verse {v}: {a} / {b}"

    # -- every shot duration is within [2.0, 4.5]s --
    for s in shots:
        d = s["end"] - s["start"]
        assert 2.0 - 1e-6 <= d <= 4.5 + 1e-6, f"bad duration {d:.3f}s in shot {s['id']}"

    # -- chorus reuse populated, if the fallback song happens to have a repeated verse --
    lines_seen = set()
    has_chorus = False
    for v in song["verses"]:
        key = " / ".join(l.strip().lower() for l in v["lines"])
        if key in lines_seen:
            has_chorus = True
        lines_seen.add(key)
    if has_chorus:
        assert any(s["reuse_of"] for s in shots), "expected reuse_of on a chorus verse's shots"
    else:
        print("(no repeated verse in FALLBACK_SONG - chorus-reuse path not exercised here)")

    # -- first shot of verse 1 is a wide establishing shot --
    first = min(shots, key=lambda s: (s["verse"], s["start"]))
    assert first["shot_type"] == "wide", f"first shot should be wide, got {first['shot_type']}"

    # -- seeds are deterministic across two runs --
    result2 = plan_shots(song, verse_times, beats, cfg)
    seeds1 = [s["seed"] for s in shots]
    seeds2 = [s["seed"] for s in result2["shots"]]
    assert seeds1 == seeds2, "seeds were not deterministic across two runs"

    print(f"\nAll asserts passed. {len(shots)} shots "
          f"({sum(1 for s in shots if s['reuse_of'] is None)} unique).")

    out_path = os.path.join(cfg["_root"], "output", "kidsong_director_test", "shotlist.json")
    save_shotlist(result, out_path)
    print(f"Saved shotlist to {out_path}")
