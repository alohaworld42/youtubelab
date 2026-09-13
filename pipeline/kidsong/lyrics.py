"""
kidsong.lyrics — Supply the lyrics for one kids' sing-along episode.

SOURCES, selected by config `kidsong.lyrics_source`:

  "auto" (DEFAULT) — write a TOPICAL song about the actual job topic with the
      configured LLM, then run it through script_qc.review_script. If it passes,
      ship it; if it fails, retry up to `kidsong.lyrics_llm_attempts` times
      feeding the rejection reasons back into the prompt; if it still cannot
      produce a passing song, fall back to the best public-domain library match
      so a bad song never ships AND the episode still renders. This is the
      default because it restores topic->song coherence — a "balance bike" topic
      yields an actual balance-bike song — now that the lyric-craft gate
      (added to fix the LLM's old unsingable output: no rhyme, lurching meter,
      the cast list narrated in every line) can reject garbage safely.
  "public_domain" — pick a curated public-domain nursery rhyme from
      prompts/pd_songs.json. Only the WORDS are public domain; ACE-Step still
      generates a brand-new melody per episode, so nothing here touches an
      existing tune, arrangement or recording. Topic-mapped, then least-recently-
      used rotation. Never touches the LLM.
  "llm" — write an original song with the configured LLM (see the attempt chain
      below), gated by the same lyric-craft checks as "auto" but with NO library
      fallback: the best LLM attempt ships even if it does not fully pass.

LLM attempt chain (first success wins; every exception is swallowed and we fall
through to the next option, so this never crashes the pipeline):
  1. Whatever backend is configured in config.json -> llm.backend (groq/ollama/gemini),
     reusing the same call helpers as script_gen.py.
  2. A plain Ollama /api/chat call against llm.ollama_host (skipped if step 1 already
     was an Ollama call — no point retrying the identical request).
  3. An OpenAI-compatible /v1/chat/completions call against the same host, useful when
     a local llama-server (not real Ollama) is listening on that port instead and only
     speaks the OpenAI API shape. The model id is discovered via GET /v1/models.
  4. A built-in fallback song.

Returns strict JSON shaped like:
  {
    "title": "...",
    "description": "...",
    "tags": ["...", ...],
    "characters": "...",           # one sentence, reused verbatim in every image prompt
    "verses": [
      {"lines": ["...", ...], "scene": "..."},
      ...
    ]
  }
"""
import copy
import difflib
import glob
import json
import logging
import os
import re

from pipeline.script_gen import (
    _call_gemini,
    _call_groq,
    _call_ollama,
    _extract_json,
    _load_prompt,
)

log = logging.getLogger("kidsong.lyrics")

# Config key + its values. "auto" is the default.
LYRICS_SOURCE_PD = "public_domain"
LYRICS_SOURCE_LLM = "llm"
LYRICS_SOURCE_AUTO = "auto"
DEFAULT_LYRICS_SOURCE = LYRICS_SOURCE_AUTO

# How many times the "auto"/"llm" paths ask the LLM for a song, feeding each
# rejection's reasons back into the prompt, before giving up. Overridable via
# `kidsong.lyrics_llm_attempts`. 3 is enough for the craft gate's actionable
# reasons to be applied without stalling a run when the LLM simply can't sing.
DEFAULT_LYRICS_LLM_ATTEMPTS = 3

# `kidsong.lyrics_topical_floor`: in "auto" mode, when no LLM attempt fully
# passes the craft gate, the best attempt is still shipped (in preference to a
# perfectly-crafted but OFF-TOPIC library song) as long as its craft score is at
# or above this floor. 0.6 empirically separates a topical song with MINOR flaws
# (has the hook and mostly rhymes — one off verse; scores ~0.65) from truly
# unsingable output (no hook, no rhyme, lurching meter; scores <= ~0.45), which
# still falls through to the library. Coherence is the channel's top priority,
# so a slightly-imperfect on-topic song beats a perfect off-topic one. Set to
# >= 1.0 to disable (strict gate: only a fully-passing song ships).
DEFAULT_TOPICAL_FLOOR = 0.6

# `kidsong.content_mode`: "song" (default) sings a story/nursery-rhyme song via
# the lyrics_source dispatch above; "learning" instead sings an educational
# song (letters/numbers/shapes — one item taught per verse) from
# prompts/learning_songs.json. Orthogonal to lyrics_source: learning mode
# never touches the LLM or the pd_songs library, mirroring the public-domain
# mode's own "never silently fall through to the LLM" guarantee.
CONTENT_MODE_SONG = "song"
CONTENT_MODE_LEARNING = "learning"
DEFAULT_CONTENT_MODE = CONTENT_MODE_SONG

PD_LIBRARY_FILENAME = "pd_songs.json"
LEARNING_LIBRARY_FILENAME = "learning_songs.json"

# How close a job's topic has to be to one of a library song's declared topics
# before we call it a match rather than falling back to plain rotation.
_TOPIC_MATCH_THRESHOLD = 0.75

SYSTEM_PROMPT = (
    "You are a songwriter for a wholesome preschool YouTube channel. You write short, "
    "original, repetitive sing-along nursery rhymes starring adorable Black children. "
    "You never copy or reference existing shows, brands, or songs. You ALWAYS reply with "
    "a single valid JSON object and nothing else."
)

def _fallback_characters():
    """The bible's own cast_sentence(), so the fallback song never drifts from
    the canonical Zuri/Kofi/Nala cast (it used to hardcode a stale "Amara").
    Imported lazily and never hard-depended on: if the cast bible module isn't
    importable/loadable for any reason, a hardcoded literal matching the
    bible's current cast keeps this module import-safe on its own."""
    try:
        from pipeline.kidsong import cast

        return cast.cast_sentence(None)
    except Exception:
        return (
            "Three adorable Black toddlers: Zuri, a girl with round afro puffs and a "
            "sunny yellow t-shirt; Kofi, a boy with short dark curly hair and a blue "
            "t-shirt; and Nala, a girl with neat cornrow braids and a yellow pinafore "
            "dress over a white t-shirt."
        )


FALLBACK_SONG = {
    "title": "Clap Your Hands - A Happy Sing-Along Song for Toddlers",
    "description": (
        "Join Zuri, Kofi and Nala for a joyful clapping song all about having fun "
        "together! #kidssong #nurseryrhyme #toddlersongs #singalong #kidsmusic"
    ),
    "tags": [
        "kids song", "nursery rhyme", "toddler song", "sing along", "clapping song",
        "preschool", "kids music", "children song", "baby song", "learning song",
    ],
    "characters": _fallback_characters(),
    # Every verse ends on the same hook line: that repeated refrain is what
    # makes a nursery rhyme singable, and script_qc's `no hook` gate requires it.
    "verses": [
        {
            "lines": ["Clap, clap, clap your hands", "Clap them with me now", "Clap, clap, clap your hands",
                      "Sing and clap along with me"],
            "scene": "The three kids stand in a sunny backyard, clapping their hands together with big smiles.",
        },
        {
            "lines": ["Stomp, stomp, stomp your feet", "Stomp them to the beat", "Stomp, stomp, stomp your feet",
                      "Sing and clap along with me"],
            "scene": "The kids stomp their feet on a grassy lawn, kicking up little leaves as they giggle.",
        },
        {
            "lines": ["Wave, wave, wave hello", "Wave hello to you and me", "Wave, wave, wave hello",
                      "Sing and clap along with me"],
            "scene": "The kids wave both arms high in the air toward colorful cartoon birds and butterflies.",
        },
        {
            "lines": ["Hug, hug, hug it out", "Hugs for all our friends", "Hug, hug, hug it out",
                      "Sing and clap along with me"],
            "scene": "The three kids gather in a big group hug under a rainbow, surrounded by floating balloons.",
        },
    ],
}


# The learning-mode equivalent of FALLBACK_SONG: used only if
# prompts/learning_songs.json is missing/unreadable/empty while
# kidsong.content_mode="learning", so a packaging problem degrades to this
# built-in song instead of ever falling through to the LLM (see generate_song).
# Verses carry the same `story_subject` / `learning_item` fields the library
# entries do, so the learning shot-planning bias in director.py still has
# something to bite on even on this safety-net path. These three verses are
# the shapes taught by the "learning_shapes_time" library song, kept in sync
# on purpose — both are proven against the full script_qc gate in
# tests/test_kidsong_learning_lyrics.py.
LEARNING_FALLBACK_SONG = {
    "title": "Shapes All Around - A Shapes Learning Sing-Along for Toddlers",
    "description": (
        "Circles, squares and triangles too! Zuri, Kofi and Nala point at every "
        "shape with us. #kidssong #shapesong #shapesfortoddlers #toddlersongs "
        "#singalong #preschool"
    ),
    "tags": [
        "kids song", "learning song", "shapes song", "toddler song", "sing along",
        "preschool", "shapes for toddlers", "educational song", "kids music",
        "circle square triangle",
    ],
    "characters": _fallback_characters(),
    "verses": [
        {
            "lines": ["A circle is round with no end", "Point at the circle, my friend",
                      "Point and clap along with me"],
            "scene": "A big red circle rolls into a bright cheerful playroom as the toddlers point and grin.",
            "story_subject": "a big red circle",
            "learning_item": "circle",
        },
        {
            "lines": ["Four straight sides, a square so neat", "Point at the square, tap your feet",
                      "Point and clap along with me"],
            "scene": "A big blue square sits in a bright cheerful playroom as the toddlers point and tap their feet.",
            "story_subject": "a big blue square",
            "learning_item": "square",
        },
        {
            "lines": ["Three sharp points, a triangle true", "Point at the triangle, yellow hue",
                      "Point and clap along with me"],
            "scene": "A big yellow triangle sits in a bright cheerful playroom as the toddlers point and cheer.",
            "story_subject": "a big yellow triangle",
            "learning_item": "triangle",
        },
    ],
}


# German fallback songs — the de-language counterparts of FALLBACK_SONG /
# LEARNING_FALLBACK_SONG. Selected by `_fallback_song`/`_learning_fallback_song`
# when kidsong.language == "de", so a German channel whose library is missing
# degrades to a GERMAN safety-net song rather than emitting the English one.
# CRITICAL: only the SUNG fields (title/description/tags/lines) are German — the
# `scene`, `story_subject` and `learning_item` fields stay ENGLISH because they
# feed the English-trained image/video model (see director.py / scenes.py); a
# German scene string would leak straight into the visual prompt. `characters`
# is likewise the English cast-bible sentence.
FALLBACK_SONG_DE = {
    "title": "Klatsch in die Hände - Ein fröhliches Mitmachlied für Kleinkinder",
    "description": (
        "Klatscht mit Zuri, Kofi und Nala bei diesem fröhlichen Mitmachlied! "
        "#kinderlied #mitmachlied #kleinkindlieder #kinderlieder #kindermusik"
    ),
    "tags": [
        "kinderlied", "mitmachlied", "kleinkinder lied", "kinderlieder", "klatschlied",
        "kindergarten", "kindermusik", "krippe", "babylieder", "lernlied",
    ],
    "characters": _fallback_characters(),
    "verses": [
        {
            "lines": ["Klatsch, klatsch, klatsch in die Hände", "Klatsch sie fröhlich mit",
                      "Klatsch, klatsch, klatsch in die Hände", "Sing und klatsch mit mir mit"],
            "scene": "The three kids stand in a sunny backyard, clapping their hands together with big smiles.",
        },
        {
            "lines": ["Stampf, stampf, stampf mit den Füßen", "Stampf im Takt dazu",
                      "Stampf, stampf, stampf mit den Füßen", "Sing und klatsch mit mir mit"],
            "scene": "The kids stomp their feet on a grassy lawn, kicking up little leaves as they giggle.",
        },
        {
            "lines": ["Wink, wink, wink mal hallo", "Wink hallo zu dir und mir",
                      "Wink, wink, wink mal hallo", "Sing und klatsch mit mir mit"],
            "scene": "The kids wave both arms high in the air toward colorful cartoon birds and butterflies.",
        },
        {
            "lines": ["Drück, drück, drück dich fest", "Drück die Freunde lieb",
                      "Drück, drück, drück dich fest", "Sing und klatsch mit mir mit"],
            "scene": "The three kids gather in a big group hug under a rainbow, surrounded by floating balloons.",
        },
    ],
}


LEARNING_FALLBACK_SONG_DE = {
    "title": "Formen überall - Ein Formen-Lernlied für Kleinkinder",
    "description": (
        "Kreis, Quadrat und Dreieck! Zeigt mit Zuri, Kofi und Nala auf jede Form. "
        "#kinderlied #formenlied #formenlernen #kleinkindlieder #mitmachlied #kindergarten"
    ),
    "tags": [
        "kinderlied", "lernlied", "formenlied", "kleinkinder lied", "mitmachlied",
        "kindergarten", "formen lernen", "lernvideo", "kindermusik", "kreis quadrat dreieck",
    ],
    "characters": _fallback_characters(),
    "verses": [
        {
            "lines": ["Ein Kreis ist rund, ganz ohne Eck", "Zeig auf den Kreis, mein Schatz",
                      "Zeig und klatsch mit mir mit"],
            "scene": "A big red circle rolls into a bright cheerful playroom as the toddlers point and grin.",
            "story_subject": "a big red circle",
            "learning_item": "circle",
        },
        {
            "lines": ["Vier gerade Seiten, ein Quadrat", "Zeig aufs Quadrat, tapp mit dem Fuß",
                      "Zeig und klatsch mit mir mit"],
            "scene": "A big blue square sits in a bright cheerful playroom as the toddlers point and tap their feet.",
            "story_subject": "a big blue square",
            "learning_item": "square",
        },
        {
            "lines": ["Drei spitze Ecken, ein Dreieck", "Zeig aufs Dreieck, gelb und schön",
                      "Zeig und klatsch mit mir mit"],
            "scene": "A big yellow triangle sits in a bright cheerful playroom as the toddlers point and cheer.",
            "story_subject": "a big yellow triangle",
            "learning_item": "triangle",
        },
    ],
}


def _language(cfg):
    """The SUNG-content language, `kidsong.language`, lowercased. Default 'en'."""
    if not cfg:
        return "en"
    return str((cfg.get("kidsong") or {}).get("language", "en")).strip().lower() or "en"


def _lib_filename(base, cfg):
    """Language-keyed library filename: 'pd_songs.json' -> 'pd_songs.de.json' for
    a non-English language, unchanged for English. So each language keeps its own
    curated library file side by side and English stays byte-identical."""
    lang = _language(cfg)
    if lang == "en":
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}.{lang}{ext}"


def _fallback_song(cfg):
    """The built-in safety-net song for this language (deep-copied so callers can
    mutate freely). German for kidsong.language='de', else the English original."""
    src = FALLBACK_SONG_DE if _language(cfg) == "de" else FALLBACK_SONG
    return json.loads(json.dumps(src))


def _learning_fallback_song(cfg):
    """Learning-mode safety-net song for this language (deep-copied)."""
    src = LEARNING_FALLBACK_SONG_DE if _language(cfg) == "de" else LEARNING_FALLBACK_SONG
    return json.loads(json.dumps(src))


# ------------------------------------------------- public-domain library -----
_PD_CACHE = {}


def _project_root(cfg=None):
    if cfg and cfg.get("_root"):
        return cfg["_root"]
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_pd_library(cfg=None, filename=PD_LIBRARY_FILENAME):
    """Read prompts/<filename> and return its `songs` list (cached per path).

    Returns [] — never raises — if the file is missing or unreadable, so a
    packaging mistake degrades to the LLM/fallback path instead of killing a run.

    `filename` defaults to the public-domain library (pd_songs.json); passing
    `LEARNING_LIBRARY_FILENAME` reads the learning-mode library instead — same
    top-level `{"songs": [...]}` shape, same caching, same "never raises"
    contract, so `load_learning_library` is a one-line wrapper around this.

    The filename is language-keyed from cfg (`_lib_filename`): with
    kidsong.language='de' this reads prompts/pd_songs.de.json (or
    learning_songs.de.json) instead. A missing language file returns [] — the
    caller then degrades to the language-appropriate fallback song, never to the
    English library.
    """
    filename = _lib_filename(filename, cfg)
    path = os.path.join(_project_root(cfg), "prompts", filename)
    if path not in _PD_CACHE:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            songs = [s for s in (data.get("songs") or []) if isinstance(s, dict) and s.get("verses")]
        except (OSError, ValueError) as exc:
            log.warning("Could not load the lyric library %s: %s", path, exc)
            songs = []
        _PD_CACHE[path] = songs
    return _PD_CACHE[path]


def load_learning_library(cfg=None):
    """Read prompts/learning_songs.json and return its `songs` list.

    Thin wrapper around `load_pd_library` — see `LEARNING_LIBRARY_FILENAME`.
    """
    return load_pd_library(cfg, filename=LEARNING_LIBRARY_FILENAME)


def _normalize_topic(topic):
    """studio.ideas.normalize_topic — the studio's own topic normalization, so
    the cascade's topics are compared here exactly the way they are deduped
    there. Imported lazily and guarded: this module must stay importable (and
    usable from the CLI) without the studio package."""
    try:
        from studio.ideas import normalize_topic

        return normalize_topic(topic)
    except Exception:
        t = re.sub(r"[^\w\s]", "", (topic or "").lower())
        return re.sub(r"\s+", " ", t).strip()


def _recent_pd_ids(cfg=None):
    """Library song ids used by recent episodes, MOST RECENT FIRST.

    Read straight off the episode records the pipeline already writes —
    `output/*-song.json`, which generate.py persists for every run and which
    carries the `pd_song_id` this module stamps into the song dict. Read-only:
    nothing here creates, moves or deletes anything under output/.
    """
    try:
        from pipeline.config import abspath

        out_dir = abspath(cfg, cfg["paths"]["output_dir"]) if cfg else None
    except Exception:
        out_dir = None
    if not out_dir:
        out_dir = os.path.join(_project_root(cfg), "output")

    ids = []
    try:
        paths = sorted(
            glob.glob(os.path.join(out_dir, "*-song.json")),
            key=os.path.getmtime,
            reverse=True,
        )
    except OSError:
        return ids
    for p in paths[:60]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                sid = (json.load(f) or {}).get("pd_song_id")
        except (OSError, ValueError):
            continue
        if sid and sid not in ids:
            ids.append(sid)
    return ids


def _topic_score(topic, entry):
    """How well `topic` matches a library entry, 0..1.

    Same normalize-then-fuzzy-ratio shape as studio.ideas.is_duplicate (which
    answers "are these the same subject?" with difflib at a threshold); here we
    keep the ratio itself so the best of several candidates can be picked.
    """
    norm = _normalize_topic(topic)
    if not norm:
        return 0.0
    best = 0.0
    candidates = list(entry.get("topics") or []) + [entry.get("title", ""), entry.get("id", "")]
    for cand in candidates:
        cnorm = _normalize_topic(str(cand).replace("_", " "))
        if not cnorm:
            continue
        ratio = difflib.SequenceMatcher(None, norm, cnorm).ratio()
        # A topic that literally contains (or is contained by) a library keyword
        # is a match even when the lengths differ a lot ("rainy day puddle
        # jumping" vs "rainy day"), which a raw ratio scores down. Weighted by
        # how much of the longer string the shorter one covers, so an exact hit
        # still beats a one-word overlap.
        if norm in cnorm or cnorm in norm:
            shorter, longer = sorted((len(norm), len(cnorm)))
            ratio = max(ratio, 0.75 + 0.25 * (shorter / float(longer or 1)))
        best = max(best, ratio)
    return best


def choose_pd_song(topic=None, cfg=None, songs=None, recent_ids=None):
    """Pick one library entry for this episode. Returns (entry, why) or (None, why).

    Selection is deterministic — same topic + same episode history always give
    the same song — and works in two obvious, logged steps:

      1. TOPIC MAP. If the job has a topic, score every library song against its
         declared `topics` (studio-style normalize + difflib) and take the best
         one at or above _TOPIC_MATCH_THRESHOLD. The one exception is an
         immediate repeat: if that song was also the previous episode's, the
         next-best match is used, or, failing that, rotation takes over.
      2. ROTATION. Otherwise pick the LEAST RECENTLY USED song (never used at
         all beats used-long-ago beats used-last-episode), tie-broken by library
         order. That alone guarantees no back-to-back repeat while the library
         has more than one song.
    """
    songs = load_pd_library(cfg) if songs is None else songs
    if not songs:
        return None, "the public-domain library is empty or unreadable"
    recent_ids = _recent_pd_ids(cfg) if recent_ids is None else list(recent_ids)
    last_used = recent_ids[0] if recent_ids else None

    if topic:
        scored = sorted(
            ((_topic_score(topic, s), i, s) for i, s in enumerate(songs)),
            key=lambda t: (-t[0], t[1]),
        )
        matches = [(sc, s) for sc, _i, s in scored if sc >= _TOPIC_MATCH_THRESHOLD]
        for sc, s in matches:
            if s.get("id") == last_used:
                # NEVER two episodes of the same song in a row — even when this
                # is the topic's ONLY match. Measured: "Big Red Tractor on the
                # Farm" matched baa_baa (farm) right after a run that had just
                # shipped baa_baa, producing two identical songs back to back
                # (jobs 25+26, 2026-07-22). An off-topic-but-fresh rotation
                # pick programs the channel better than an on-topic repeat.
                continue
            return s, (
                f"topic {topic!r} matched library song {s.get('id')!r} "
                f"(score {sc:.2f} >= {_TOPIC_MATCH_THRESHOLD})"
            )
        if matches:
            log.info(
                "Topic %r only matched %r, which was the previous episode — "
                "falling back to rotation.", topic, last_used,
            )

    def staleness(song):
        """How many episodes ago this song was used; a song never used at all
        scores higher than any song that was."""
        try:
            return recent_ids.index(song.get("id"))
        except ValueError:
            return len(recent_ids) + 1

    # Most stale first; ties broken by library order, so the choice is stable.
    ordered = sorted(enumerate(songs), key=lambda t: (-staleness(t[1]), t[0]))
    entry = ordered[0][1]
    if entry.get("id") in recent_ids:
        why = (
            f"rotation picked {entry.get('id')!r} — least recently used "
            f"(last seen {recent_ids.index(entry.get('id')) + 1} episode(s) ago)"
        )
    else:
        why = f"rotation picked {entry.get('id')!r} — not used in any recent episode"
    if topic:
        why = f"topic {topic!r} matched no library song closely enough; " + why
    return entry, why


def choose_learning_song(topic=None, cfg=None, recent_ids=None):
    """`choose_pd_song`, scoped to the learning-mode library. Same topic-match
    + least-recently-used rotation, same recency source (`_recent_pd_ids`
    reads `pd_song_id` off `output/*-song.json`, which `pd_song_to_dict` stamps
    for a learning song exactly like a public-domain one — one rotation
    ledger, shared across both content modes, is simplest and still correct:
    a learning song's id can never collide with a pd song's)."""
    return choose_pd_song(topic, cfg, songs=load_learning_library(cfg), recent_ids=recent_ids)


def pd_song_to_dict(entry):
    """Render a library entry into the exact dict shape the rest of the pipeline
    consumes — `title, description, tags, characters, verses[{lines, scene}]` —
    so sing.build_lyrics, director.plan_shots and sing.verse_times_from_words
    all work unchanged. Two extra keys ride along for provenance:
    `pd_song_id` (which _recent_pd_ids reads back to avoid repeats) and
    `pd_source` (the publication/public-domain justification).

    A verse that carries `story_subject` and/or `learning_item` (the learning
    library — prompts/learning_songs.json — sets both on every verse; the
    public-domain library sets neither) has them PRESERVED onto the output
    verse dict. director._fallback_planner reads `story_subject` straight off
    the verse to pin that verse's taught item. A pd_songs.json verse (no such
    keys) produces the exact same two-key {lines, scene} dict as before this
    existed — this is additive only.
    """
    verses = []
    for v in entry.get("verses") or []:
        lines = [str(l).strip() for l in (v.get("lines") or []) if str(l).strip()]
        if not lines:
            continue
        verse = {"lines": lines, "scene": str(v.get("scene_hint") or v.get("scene") or "").strip()}
        story_subject = str(v.get("story_subject") or "").strip()
        if story_subject:
            verse["story_subject"] = story_subject
        learning_item = str(v.get("learning_item") or "").strip()
        if learning_item:
            verse["learning_item"] = learning_item
        verses.append(verse)

    song = {
        "title": str(entry.get("title") or "").strip(),
        "description": str(entry.get("description") or "").strip(),
        "tags": _normalize_tags(entry.get("tags")),
        "characters": _fallback_characters(),  # always the canonical cast bible
        "verses": verses,
        "pd_song_id": entry.get("id"),
        "pd_source": entry.get("source"),
    }
    # The library carries a per-song ACE-Step style caption (a lullaby and a
    # skipping song should not be sung the same way); sing.py reads
    # cfg["kidsong"]["song_style"], so it is surfaced here for callers that
    # want to apply it and is harmless to those that do not.
    if entry.get("song_style"):
        song["song_style"] = entry["song_style"]

    # EPISODE CANON: the library's fixed per-song look for its non-child story
    # subject (a mouse/lamb/sheep/boat/star — never a child) and its one fixed
    # setting for the whole episode, so the shot/keyframe prompts (generate.py
    # `_shot_prompt`/`_keyframe_prompt`) and the location planner
    # (director.plan_verse_locations) both have a stable, author-picked anchor
    # instead of drifting per shot. Measured failure this fixes: a "baa baa
    # black sheep" episode rendered a black sheep in some keyframes and a
    # white one in others, and its setting quietly slid from the farm into a
    # generic backyard mid-episode (output/20260724-181038-kidsong-baa-baa-...).
    # `subject_description` is legitimately null for a song with no single
    # recurring non-child subject (prompts/pd_songs.json documents which);
    # `canonical_setting` is set for every entry, so it is always a plain
    # string here, defaulting to "" (falsy, treated as absent downstream) only
    # for a library entry that predates this field (e.g. learning_songs.json,
    # which never carries either key).
    subject_description = entry.get("subject_description")
    song["subject_description"] = (
        subject_description.strip()
        if isinstance(subject_description, str) and subject_description.strip()
        else None
    )
    song["canonical_setting"] = str(entry.get("canonical_setting") or "").strip() or None
    return song


def generate_pd_song(topic=None, cfg=None):
    """The public-domain path: choose a library song and return it pipeline-shaped.
    Returns None (logged) when no song can be produced, so callers can fall back."""
    entry, why = choose_pd_song(topic, cfg)
    if entry is None:
        log.warning("Public-domain lyrics unavailable: %s", why)
        return None
    log.info("Public-domain lyrics: %s", why)
    print(f"  Lyrics: public-domain library — {why}")
    return _normalize_song(pd_song_to_dict(entry))


def generate_learning_song(topic=None, cfg=None):
    """The learning-mode path: choose a library song (letters/numbers/shapes)
    and return it pipeline-shaped, exactly like `generate_pd_song`. Returns
    None (logged) when no song can be produced, so `generate_song` can fall
    back to `LEARNING_FALLBACK_SONG` rather than the LLM."""
    entry, why = choose_learning_song(topic, cfg)
    if entry is None:
        log.warning("Learning-mode lyrics unavailable: %s", why)
        return None
    log.info("Learning-mode lyrics: %s", why)
    print(f"  Lyrics: learning library — {why}")
    return _normalize_song(pd_song_to_dict(entry))


# ---------------------------------------------------------------- backends ----
def _post_ollama_chat(system, user, model, host):
    """Plain Ollama /api/chat call, independent of which backend generate_song already tried."""
    import requests

    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.9},
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
            "temperature": 0.9,
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
def generate_song(topic=None, cfg=None):
    """Return this episode's song dict, from whichever source config selects.

    The returned shape is identical for both sources —
    {title, description, tags, characters, verses[{lines, scene}]} — so nothing
    downstream (sing.build_lyrics, director.plan_shots, verse_times_from_words,
    script_qc.review_script) needs to know which one produced it.
    """
    if cfg is None:
        from pipeline.config import load_config

        cfg = load_config()

    content_mode = str(
        (cfg.get("kidsong", {}) or {}).get("content_mode", DEFAULT_CONTENT_MODE)
    ).strip().lower()
    if content_mode == CONTENT_MODE_LEARNING:
        song = generate_learning_song(topic, cfg)
        if song is not None:
            return song
        # Mirrors the public-domain guarantee just below: explicitly asking for
        # a learning episode and finding the library unavailable must not
        # silently switch to the (unrelated, story-song) LLM path.
        log.warning("Falling back to the built-in learning song (kidsong.content_mode=learning).")
        return _normalize_song(_learning_fallback_song(cfg))

    source = str(
        (cfg.get("kidsong", {}) or {}).get("lyrics_source", DEFAULT_LYRICS_SOURCE)
    ).strip().lower()

    if source == LYRICS_SOURCE_AUTO:
        return _generate_auto_song(topic, cfg)

    if source == LYRICS_SOURCE_PD:
        song = generate_pd_song(topic, cfg)
        if song is not None:
            return song
        # Explicitly asked for public-domain words and the library failed:
        # the built-in fallback song is original and safe, whereas silently
        # switching to the LLM would ship exactly the lyrics this mode
        # exists to avoid.
        log.warning("Falling back to the built-in song (lyrics_source=public_domain).")
        return _normalize_song(_fallback_song(cfg))

    # source == "llm" (and any unknown value): pure gated LLM, no library
    # fallback. The best attempt ships even if it did not fully pass, so the
    # episode still renders; generate.py's own script gate is the last net.
    song, verdict = _generate_llm_song_gated(topic, cfg)
    if verdict is not None and verdict.get("accept"):
        log.info("LLM lyrics: topical song passed the craft gate (score %.2f).",
                 verdict.get("score", 0.0))
    elif verdict is not None:
        log.warning(
            "lyrics_source=llm: shipping the best-effort LLM song that did not "
            "fully pass the craft gate: %s", "; ".join(verdict.get("reasons", [])),
        )
    return song if song is not None else _normalize_song(_fallback_song(cfg))


def _topical_floor(cfg):
    """The craft-score floor at or above which a topical LLM song is shipped even
    though it did not fully pass the gate. `kidsong.lyrics_topical_floor`,
    default `DEFAULT_TOPICAL_FLOOR`. Clamped to [0, 1]; a value >= 1.0 disables
    the floor (strict gate — only a fully-passing song ships, else the library)."""
    ks = cfg.get("kidsong", {}) or {}
    try:
        floor = float(ks.get("lyrics_topical_floor", DEFAULT_TOPICAL_FLOOR))
    except (TypeError, ValueError):
        floor = DEFAULT_TOPICAL_FLOOR
    return max(0.0, min(1.0, floor))


def _generate_auto_song(topic, cfg):
    """The default path: a TOPICAL LLM song, gated by script_qc; the library is
    the safety net when the LLM cannot sing.

    Restores topic->song coherence — the song is ABOUT the requested topic, so
    the director then depicts that subject. Preference order:
      1. An LLM song that fully PASSES the craft gate.
      2. Else, the best LLM attempt if it clears the TOPICAL FLOOR
         (`kidsong.lyrics_topical_floor`, default 0.6): a genuinely topical song
         with only MINOR craft flaws (it has the hook and mostly rhymes — just
         one off verse) is shipped in preference to a perfectly-crafted but
         OFF-TOPIC library song, because coherence is the channel's stated top
         priority. Set the floor to >= 1.0 to disable this (strict gate).
      3. Else, the best public-domain library match (the safety net for a truly
         unsingable attempt — no hook, no rhyme, lurching meter), so the episode
         still renders. `_generate_llm_song_gated` already guarantees the song
         here is a real topical attempt, never the canned generic fallback.
    """
    song, verdict = _generate_llm_song_gated(topic, cfg)
    if verdict is not None and verdict.get("accept"):
        why = f"topical LLM song passed the craft gate (score {verdict.get('score', 0.0):.2f})"
        log.info("Auto lyrics: %s", why)
        print(f"  Lyrics: topical LLM song — {why}")
        return song

    floor = _topical_floor(cfg)
    score = (verdict or {}).get("score", 0.0)
    if song is not None and verdict is not None and score >= floor:
        flaws = "; ".join(verdict.get("reasons", [])) or "minor craft flaws"
        why = (
            f"topical LLM song shipped above the floor (score {score:.2f} >= "
            f"{floor:.2f}); coherence preferred over an off-topic library song. "
            f"Minor craft flaws: {flaws}"
        )
        log.info("Auto lyrics: %s", why)
        print(f"  Lyrics: topical LLM song (above floor {floor:.2f}, score {score:.2f})")
        return song

    reasons = "; ".join((verdict or {}).get("reasons", [])) or "no LLM song produced"
    log.info(
        "Auto lyrics: no LLM song passed the gate and the best attempt scored "
        "%.2f < floor %.2f (%s) — falling back to the public-domain library.",
        score, floor, reasons,
    )
    pd = generate_pd_song(topic, cfg)
    if pd is not None:
        print(
            f"  Lyrics: LLM song rejected ({reasons}) — fell back to the "
            f"public-domain library."
        )
        return pd

    # Library unavailable too: ship the best LLM attempt rather than nothing.
    log.warning(
        "Auto lyrics: library unavailable as well — shipping the best-effort "
        "LLM song (%s).", reasons,
    )
    return song if song is not None else _normalize_song(_fallback_song(cfg))


def _generate_llm_song_gated(topic, cfg):
    """Ask the LLM for a song and gate it with script_qc.review_script, retrying
    with the rejection reasons fed back into the prompt.

    Returns `(song, verdict)`:
      * the first attempt whose verdict ACCEPTS, or
      * the highest-scoring attempt if none accepts (so callers can decide
        whether to ship it or fall back), or
      * `(None, None)` only if no attempt could even be produced.

    The gate here is the PROGRAMMATIC craft verdict (see `_craft_verdict`): the
    external/human script review still runs once in generate.py on the final
    chosen song, so this loop must not also block on it per attempt.
    """
    ks = cfg.get("kidsong", {}) or {}
    try:
        attempts = int(ks.get("lyrics_llm_attempts", DEFAULT_LYRICS_LLM_ATTEMPTS))
    except (TypeError, ValueError):
        attempts = DEFAULT_LYRICS_LLM_ATTEMPTS
    attempts = max(1, attempts)

    best_song, best_verdict = None, None
    feedback = None
    for i in range(attempts):
        # First attempt sends no feedback so a plain `(topic, cfg)` stub still
        # works; retries append the previous rejection's reasons to the prompt.
        song = _generate_llm_song(topic, cfg, feedback) if feedback else _generate_llm_song(topic, cfg)
        # No backend produced anything — `song` is the canned fallback (crafted
        # to pass the craft gate). Every remaining attempt hits the same dead
        # backend, so stop now and return "no attempt produced"; auto-mode then
        # falls back to the topic-matched public-domain library.
        if song.pop("_backend_dead", False):
            log.warning(
                "LLM backend unavailable on attempt %d/%d — no topical song "
                "produced; deferring to the library fallback.", i + 1, attempts,
            )
            return None, None
        # The LLM replied but its verses were unusable, so they were swapped for
        # the canned generic verses (topical title kept). Never accept that as a
        # topical song; a retry may produce usable verses, so reject and loop.
        if song.pop("_generic_fallback", False):
            feedback = [
                "your verses were empty, too short, or malformed and had to be "
                "discarded — write 3 to 5 verses, each with 2 to 4 short rhyming "
                "lines ABOUT THE TOPIC, in the exact JSON shape requested"
            ]
            log.info(
                "LLM lyric attempt %d/%d produced unusable verses (swapped to the "
                "canned fallback) — retrying with a shape reminder.", i + 1, attempts,
            )
            continue
        verdict = _craft_verdict(song, cfg)
        if verdict.get("accept"):
            if i:
                log.info("LLM lyrics accepted on attempt %d/%d.", i + 1, attempts)
            return song, verdict
        if best_verdict is None or verdict.get("score", 0.0) > best_verdict.get("score", 0.0):
            best_song, best_verdict = song, verdict
        feedback = list(verdict.get("reasons") or [])
        log.info(
            "LLM lyric attempt %d/%d rejected (score %.2f): %s",
            i + 1, attempts, verdict.get("score", 0.0), "; ".join(feedback),
        )
    return best_song, best_verdict


def _craft_verdict(song, cfg):
    """`script_qc.review_script` restricted to its PROGRAMMATIC craft verdict.

    The auto/llm retry loop needs the deterministic lyric-craft judgement, not
    the external/human gate — that gate is run exactly once, later, in
    generate.py on the final chosen song. If `kidsong.review.reviewer` is
    "external", review the song against a copy with the reviewer forced to
    "heuristic" so this internal check neither writes a request file nor blocks.
    """
    from pipeline.kidsong.script_qc import review_script

    review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
    if str(review_cfg.get("reviewer", "heuristic")).lower() == "external":
        craft_cfg = copy.deepcopy(cfg)
        craft_cfg.setdefault("kidsong", {}).setdefault("review", {})["reviewer"] = "heuristic"
        return review_script(song, craft_cfg)
    return review_script(song, cfg)


def _retry_feedback_block(reasons):
    """Prompt appendix that turns the craft gate's rejection reasons into an
    explicit fix-list for the next LLM attempt, while pinning the topic."""
    bullets = "\n".join(f"- {r}" for r in reasons)
    return (
        "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED by the automatic lyric checker "
        "for the reasons below. Write a NEW version that fixes EVERY one of them, "
        "counting syllables on your fingers before you answer:\n"
        f"{bullets}\n"
        "Keep the song about the SAME topic and the same cast — only fix the "
        "craft problems listed above."
    )


def _generate_llm_song(topic, cfg, feedback=None):
    """Generate one original song with the configured LLM (attempt chain in the
    module docstring). `feedback` is an optional list of the previous attempt's
    craft-gate rejection reasons, appended to the prompt so the model fixes them.

    The returned `characters` is always the canonical cast-bible sentence: the
    prompt tells the model that field is discarded, and every downstream consumer
    expects the pinned cast, so it is overwritten here regardless of what the LLM
    wrote — exactly as the public-domain path already does in `pd_song_to_dict`.
    """
    root = cfg["_root"]
    template = _load_prompt("kidsong", root)
    user = template.replace("{{TOPIC}}", topic or "pick a wholesome topic yourself")
    if feedback:
        user += _retry_feedback_block(feedback)

    llm = cfg["llm"]
    backend = llm["backend"]
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

    # 4. Built-in fallback song — guarantees this function never raises.
    backend_dead = data is None
    if data is None:
        data = json.loads(json.dumps(FALLBACK_SONG))  # cheap deep copy

    song = _normalize_song(data)
    # The cast is fixed and versioned in the bible; the LLM's `characters` line
    # is discarded (see the prompt) so every shot's identity clause stays on
    # the canonical Zuri/Kofi/Nala look.
    song["characters"] = _fallback_characters()

    # A topical song can silently collapse to the GENERIC canned song by two
    # routes, and the canned song is crafted to PASS the craft gate — so either
    # would ship a "Clap Your Hands" body for a "Sunflower Family Tree" topic
    # unless flagged here. Both keys are private (leading underscore) and popped
    # by the gated loop before the song ever ships:
    #   * _backend_dead  — NO backend produced anything (data was None); every
    #     retry hits the same dead backend, so the loop aborts to the library.
    #   * _generic_fallback — the LLM replied but its verses were too broken to
    #     keep, so `_normalize_verses` substituted FALLBACK_SONG's verses
    #     wholesale (the topical TITLE survives, masking it). A retry may do
    #     better, so the loop rejects this attempt and keeps trying.
    if backend_dead:
        song["_backend_dead"] = True
    elif song.get("verses") == FALLBACK_SONG["verses"]:
        song["_generic_fallback"] = True
    return song


def _normalize_song(data):
    """Validate and coerce a raw LLM song dict into the shape the rest of the pipeline
    relies on, the same defensive way script_gen._normalize_script does: LLMs are
    inconsistent, so never trust types, presence, or bounds."""
    if not isinstance(data, dict):
        data = {}

    title = str(data.get("title") or "").strip()
    data["title"] = title or FALLBACK_SONG["title"]

    data["description"] = str(data.get("description") or "").strip() or FALLBACK_SONG["description"]

    data["tags"] = _normalize_tags(data.get("tags"))

    characters = str(data.get("characters") or "").strip()
    data["characters"] = characters or FALLBACK_SONG["characters"]

    data["verses"] = _normalize_verses(data.get("verses"))

    return data


def _normalize_verses(raw):
    """Clamp to 3-5 verses of 2-4 non-empty lines each. Verses with too few usable
    lines are dropped; if fewer than 3 verses survive, fall back to the built-in
    song's verses wholesale rather than shipping a broken/too-short song.

    A verse's `story_subject` / `learning_item` (set by learning-mode verses —
    see `pd_song_to_dict` — and by `LEARNING_FALLBACK_SONG`) ride through
    unchanged when present and non-empty; a verse without them (every
    pd_songs.json / LLM verse today) is completely unaffected, so this stays
    byte-identical for every existing caller.
    """
    verses = []
    if isinstance(raw, (list, tuple)):
        for v in raw:
            if not isinstance(v, dict):
                continue
            lines = v.get("lines")
            if isinstance(lines, str):
                lines = [lines]
            elif not isinstance(lines, (list, tuple)):
                lines = []
            clean_lines = [str(ln).strip() for ln in lines if str(ln).strip()]
            clean_lines = clean_lines[:4]
            if len(clean_lines) < 2:
                continue
            scene = str(v.get("scene") or "").strip()
            if not scene:
                scene = FALLBACK_SONG["verses"][len(verses) % len(FALLBACK_SONG["verses"])]["scene"]
            verse_out = {"lines": clean_lines, "scene": scene}
            story_subject = v.get("story_subject")
            if isinstance(story_subject, str) and story_subject.strip():
                verse_out["story_subject"] = story_subject.strip()
            learning_item = v.get("learning_item")
            if isinstance(learning_item, str) and learning_item.strip():
                verse_out["learning_item"] = learning_item.strip()
            verses.append(verse_out)

    if len(verses) < 3:
        return [dict(v) for v in FALLBACK_SONG["verses"]]
    return verses[:5]


def _normalize_tags(raw):
    """Return a de-duplicated list of clean, non-empty string tags."""
    if isinstance(raw, str):
        raw = re.split(r"[,\n]", raw)
    elif not isinstance(raw, (list, tuple)):
        raw = []
    tags = []
    seen = set()
    for t in raw:
        tag = str(t).strip().lstrip("#").strip()
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append(tag)
    return tags or list(FALLBACK_SONG["tags"])


if __name__ == "__main__":
    import sys

    topic = sys.argv[1] if len(sys.argv) > 1 else None
    out = generate_song(topic)
    print(json.dumps(out, indent=2, ensure_ascii=False))
