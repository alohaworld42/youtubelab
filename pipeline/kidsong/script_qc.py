"""
kidsong.script_qc — Automated quality gate for the song script and its shot list,
run BEFORE any GPU work happens: `review_script` gates the lyrics right after
`lyrics.generate_song` returns, and `review_shotlist` gates `director.plan_shots`'
output before ComfyUI renders a single frame. Catching a bad song or a broken
plan here is nearly free (pure CPU, no renders); catching it after rendering
costs minutes per shot.

Verdict shape matches pipeline.kidsong.review: {"accept": bool, "score": float,
"reasons": [...], "retry_hints": {...}}. Reasons are "<category>: <detail>"
strings; the score is 1 minus the sum of each *category* of failure's weight
(not per-instance — five over-long lines cost the same as one), clamped to
[0, 1].

Optional external gate mirroring review.py's ExternalReviewer: if
kidsong.review.reviewer == "external", the programmatic verdict is written
alongside the song/shotlist to `<review_dir>/script.request.json` (or
`shotlist.request.json`), and this blocks (via review.external_gate_poll)
until a matching `*.response.json` appears or kidsong.review.external_timeout
elapses — at which point the programmatic verdict is returned with a timeout
reason appended, so an unattended run never hangs forever.

That wait is run-scoped: all four gates of a run (script, shotlist, per-shot,
cut) share one `review.ReviewSession`, so once
kidsong.review.unattended_after_timeouts requests have gone unanswered
ANYWHERE in the run, these gates stop waiting entirely — they still write
their request file, but return the programmatic verdict immediately. The
programmatic checks below are completely unaffected by that; only the wait is.
"""
import os
import re

from pipeline.kidsong.review import external_gate_poll


def _language(cfg):
    """The sung-content language, `kidsong.language`, lowercased (default 'en').

    Mirrors `lyrics._language` exactly but is not imported from there — this
    module is deliberately import-light (see the docstrings on
    `_capitalized_names`/`_check_staging`), and this one-liner is cheaper to
    duplicate than to add a cross-module dependency for.

    Every English-specific craft heuristic below (meter, rhyme, the action-verb
    reward, the clumsy-phrasing check, the capitalization-based name detector)
    is gated on `language == "en"` and SKIPPED for anything else — the German
    path (kidsong.language="de") ships only curated, pre-vetted library songs
    (see lyrics.py's FALLBACK_SONG_DE / pd_songs.de.json), so the German craft
    profile is deliberately permissive rather than a hand-built German
    syllable/rhyme model: content-safety checks (brand words) and
    language-agnostic structural checks (verse/line counts, repetition, hook)
    stay active for every language.
    """
    if not cfg:
        return "en"
    return str((cfg.get("kidsong") or {}).get("language", "en")).strip().lower() or "en"


# ------------------------------------------------------------------ script ---
_MIN_VERSES, _MAX_VERSES = 3, 5
_MIN_LINES, _MAX_LINES = 2, 4
_MAX_WORDS_PER_LINE = 9
_MAX_MEAN_WORD_LEN = 6.0

# German compounds ("Kindergarten", "Schmetterling") run longer than English
# words at the same reading level, so the English mean-word-length cap would
# false-reject well-formed German lyrics. Used only when kidsong.language !=
# "en" (see `_check_script`); 9.0 was sized to clear both German fallback
# songs (prompts/pd_songs.de.json's curated library songs are shorter still)
# with headroom, while still catching genuinely dense/complex text.
_MAX_MEAN_WORD_LEN_DE = 9.0

# Imitable actions a toddler watching along can physically copy. Not
# exhaustive — just the deliberately-small built-in list the contract calls
# for; the "..." in the spec is covered by adding a few obvious synonyms.
_ACTION_VERBS = {
    "clap", "stomp", "jump", "wave", "wash", "brush", "dance", "spin", "count",
    # "march" deliberately absent: this list is the set of imitable actions the
    # lyrics gate REWARDS, and marching is exactly the martial staging the shot
    # gate now rejects (see _check_staging). Endorsing it here while rejecting
    # it there would have the lyrics gate steer songs into shot lists that
    # cannot pass.
    "splash", "hug", "sing", "twirl", "hop", "shake", "nod", "point",
    "reach", "stretch", "wiggle", "smile", "cheer", "kick", "run", "walk",
    "skip", "bounce", "tiptoe", "sway", "roll", "crawl", "climb", "giggle",
    "blow", "clean", "sweep", "cook", "paint", "draw", "build", "high-five",
}

# Small blocklist — this channel writes wholly original songs, never riffs on
# an existing show/brand (see lyrics.SYSTEM_PROMPT).
_BRAND_WORDS = {
    "cocomelon", "disney", "sesame", "sesame street", "jj", "blippi", "peppa",
    "peppa pig", "elmo", "mickey", "mickey mouse", "barney", "paw patrol",
    "bluey", "pj masks", "pinkfong", "baby shark",
}

_SCRIPT_WEIGHTS = {
    "title missing": 0.1,
    "verse count out of range": 0.25,
    "line too long": 0.15,
    "vocabulary too complex": 0.15,
    "no repetition": 0.15,
    "not enough action verbs": 0.15,
    "characters sentence invalid": 0.15,
    "brand words present": 0.4,
    # A verse scene handing each child a separate activity (see
    # `_check_verse_scenes`) — the structural cause of crowded, incoherent
    # frames, and only fixable by regenerating the lyrics.
    "verse mixes activities": 0.3,
    # --- lyric craft (see the "lyric craft" section below) ---
    "meter inconsistent": 0.15,
    "no rhyme": 0.2,
    "character names overused": 0.15,
    "no hook": 0.2,
    "clumsy phrasing": 0.15,
}

# ------------------------------------------------------------ lyric craft ---
# The gates above check that a song is *safe and well-formed*; these check that
# it is actually SINGABLE. They exist because the LLM path shipped verses like
#   "We'll use our cups to pour with fun! / Every flower will get some water
#    soon. / Nala's helping now, everyone!"
# — no rhyme, syllable counts swinging 8->11, a cast name in nearly every line,
# and no repeated hook anywhere in the song. Every rule below is calibrated so
# that real public-domain nursery rhymes (prompts/pd_songs.json) pass it.

# Max spread of syllable counts within one verse. 2 is deliberate: the real
# rejected verse above spans 3 (8/11/9), while genuine nursery-rhyme verses
# ("Twinkle, twinkle, little star" / "How I wonder what you are") span 0-2.
_METER_TOLERANCE = 2

# A cast name may appear in at most this fraction of the song's lines. A nursery
# rhyme does not narrate its own cast list; the rejected song above named a
# child in 4 of 9 lines (44%).
_MAX_NAME_LINE_FRACTION = 0.34

# A hook line has to be a real phrase, not an interjection.
_MIN_HOOK_WORDS = 3

# English spelling hides vowel identity, so the rhyme key normalizes the most
# common digraphs to the single letter they sound like. This is a heuristic, not
# a pronouncing dictionary — it is only ever used to answer "does SOME pair of
# lines in this verse rhyme?", and the gate fires only when the answer is no for
# every pair, so an occasional missed rhyme costs nothing.
_VOWEL_CLASSES = {
    "ee": "e", "ea": "e", "ie": "e", "ei": "e",
    "oo": "o", "oa": "o", "oe": "o",
    "ai": "a", "ay": "a",
    "ue": "u", "ui": "u",
    "y": "i",
}

# Pairs English spelling splits but pronunciation does not. <oo> before <l>
# ("wool", "pull") is the same vowel as <u> before <ll> ("full") — without this,
# "have you any wool" / "three bags full" reads as a non-rhyme.
_RHYME_EQUIV = {"ol": "ul"}

_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")

# Repeating one of these inside a line is ordinary English, not a mistake
# ("as fast as you can", "Skip to my Lou, my darling").
_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "as", "at", "by", "for", "from", "in",
    "into", "of", "on", "to", "up", "with", "we", "you", "i", "me", "my", "our",
    "your", "it", "its", "is", "are", "was", "be", "do", "so", "no", "not",
    "all", "this", "that", "there", "here", "he", "she", "they", "them", "his",
    "her", "their", "will", "can", "come", "go",
}


def _letters(word):
    # äöüß kept (not just a-z) so a German word's vowel/consonant identity
    # survives this strip — dropping them silently corrupted every downstream
    # heuristic that calls this (syllable count, rhyme key) into English-only
    # tools that mangled German input ("für" -> "fr"). Harmless for English:
    # no English token contains these characters, so this is unconditional.
    return re.sub(r"[^a-zäöüß]", "", str(word).lower())


def _strip_silent_e(w):
    """Drop a word-final silent <e> ("care" -> "car"), but never the <e> of a
    consonant+<le> syllable ("little", "twinkle") which is pronounced."""
    if not w.endswith("e") or len(w) < 3:
        return w
    if w.endswith("le") and w[-3] not in "aeiouy":
        return w
    if len(_VOWEL_GROUP_RE.findall(w)) <= 1:
        return w
    return w[:-1]


def _syllables(word):
    """Heuristic syllable count: vowel groups, minus a silent final <e> and a
    silent past-tense <-ed> ("followed" = 2, but "waited" = 2 as well).

    Approximate by design (it over-counts "everywhere" at 4). The meter gate
    compares counts of lines against each other with the SAME counter, so a
    consistent bias cancels out; only the spread matters.
    """
    w = _letters(word)
    if not w:
        return 0
    # <-ed> is a syllable only after <t>/<d> (wait-ed, need-ed); elsewhere it is
    # silent (followed, turned, lingered) and must not be counted.
    if w.endswith("ed") and len(w) > 3 and w[-3] not in "aeiouytd":
        w = w[:-2]
    return max(1, len(_VOWEL_GROUP_RE.findall(_strip_silent_e(w))))


def _line_syllables(line):
    return sum(_syllables(w) for w in _words(line))


def _is_chant_line(line):
    """True for a reduplication/vocable line — "Merrily, merrily, merrily,
    merrily", "Ee-i-ee-i-oh", "la la la". These are a real nursery-rhyme device
    and are deliberately off-meter and unrhymed, so the meter and rhyme gates
    skip them instead of demanding the song drop them."""
    words = [w.lower() for w in _words(line)]
    if len(words) < 3:
        return False
    if len(set(words)) <= 2:
        return True
    return all(len(w) <= 2 for w in words)


def _rhyme_key(line):
    """The rime of a line's last word: normalized vowel nucleus + coda.

    "star"/"are" -> "ar"; "high"/"sky" -> "i"; "stream"/"dream" -> "em".
    Returns None when the line has no usable last word.
    """
    words = _words(line)
    if not words:
        return None
    w = _letters(words[-1])
    if not w:
        return None
    w = w.replace("gh", "")          # silent <gh>: high -> hi, night -> nit
    w = _strip_silent_e(w)
    if w.endswith("w") and len(w) > 1:  # silent <w>: snow -> sno, now -> no
        w = w[:-1]
    groups = list(_VOWEL_GROUP_RE.finditer(w))
    if not groups:
        return w
    last = groups[-1]
    nucleus = last.group()
    nucleus = _VOWEL_CLASSES.get(nucleus, nucleus[0])
    coda = re.sub(r"(.)\1+", r"\1", w[last.end():])  # "full" -> "ul"
    key = nucleus + coda
    return _RHYME_EQUIV.get(key, key)


def _verse_has_rhyme(lines):
    """True if any two of the verse's singable lines end on the same rime.
    Two lines ending on the SAME WORD count — repeating a line as its own
    refrain is the oldest rhyme scheme there is."""
    keys = [_rhyme_key(l) for l in lines if not _is_chant_line(l)]
    keys = [k for k in keys if k]
    return len(keys) - len(set(keys)) > 0


def _normalized_line(line):
    return " ".join(w.lower() for w in _words(line))


def _hook_line(verses):
    """The song's hook: a substantial line (>= _MIN_HOOK_WORDS words) that
    recurs in two or more DIFFERENT verses. Bookending one verse with its own
    opening line is already covered by the `no repetition` gate; a hook is what
    makes the song memorable ACROSS verses, and both rejected LLM songs had
    none at all. Returns the line, or None."""
    line_verses = {}
    for i, v in enumerate(verses):
        lines = v.get("lines") if isinstance(v, dict) else None
        for line in lines or []:
            norm = _normalized_line(line)
            if len(norm.split()) >= _MIN_HOOK_WORDS:
                line_verses.setdefault(norm, set()).add(i)
    for norm, idxs in line_verses.items():
        if len(idxs) >= 2:
            return norm
    return None


def _clumsy_repetition(line):
    """Detect "Blow bubbles high up high" — a content word repeated with exactly
    ONE word wedged between it, which reads as broken English rather than as a
    deliberate device.

    Everything nursery rhymes actually do is exempt, or the gate would reject
    the entire public-domain canon:
      * adjacent reduplication — "Clap, clap, clap your hands", "Row, row, row",
        "Merrily, merrily, merrily, merrily";
      * a repeated two-word phrase — "Yes, sir, yes, sir", "Brother John,
        Brother John", "Round and round and clap with me";
      * function words, which recur harmlessly in ordinary English — "as fast
        as you can", "Skip to my Lou, my darling".
    Returns the offending word, or None.
    """
    words = [w.lower() for w in _words(line)]
    for i in range(len(words) - 2):
        w = words[i]
        if w != words[i + 2] or w in _FUNCTION_WORDS:
            continue
        if w == words[i + 1]:
            continue  # "clap clap clap": adjacent reduplication
        if i + 3 < len(words) and words[i + 2] == words[i + 3]:
            continue  # reduplication starting one word later
        if i and words[i - 1 : i + 1] == words[i + 1 : i + 3]:
            continue  # repeated 2-word phrase ending here: "yes sir | yes sir"
        if words[i : i + 2] == words[i + 2 : i + 4]:
            continue  # repeated 2-word phrase starting here: "round and | round and"
        return w
    return None

# --------------------------------------------------------------- shotlist ---
_MIN_SHOT_SECONDS, _MAX_SHOT_SECONDS = 1.6, 5.0
_TILE_GAP_TOLERANCE = 0.05
_UNCLUTTERED_FRACTION = 0.6

_SHOTLIST_WEIGHTS = {
    "no shots": 0.5,
    "verse tiling gap": 0.25,
    "duration out of range": 0.2,
    "first shot not wide": 0.15,
    "3 consecutive same shot_type": 0.15,
    "too many non-static cameras": 0.1,
    "cluttered frame": 0.15,
    "action missing/not verb-ish": 0.15,
    "unresolved reuse_of": 0.2,
    "head count contradiction": 0.3,
    "setting is not a place": 0.15,
    # Staging faults measured in the 2026-07-20 episodes. Weighted heavily:
    # each of these ships a defect the per-take vision review cannot repair by
    # re-seeding, because the fault is in the PLAN, not the sample.
    "forbidden location": 0.4,
    "martial staging": 0.4,
    "frightening content": 0.4,
    "verse mixes activities": 0.3,
    "verse location discontinuity": 0.2,
    "indoor/outdoor contradiction": 0.3,
    "location ping-pong": 0.2,
}

# Formerly: "a verse is one continuous moment, so its shots share ONE
# activity; >2 distinct activity verbs across a verse means it has no
# through-line." That blanket cap rejected legitimate narrative verses (the
# clock strikes, the mouse runs down, a child points and giggles — 3 verbs,
# one location, one continuous moment). Enforcement of verse coherence now
# lives in `director.verse_activity_progression` (uniform / narrative /
# scattered) via `_check_verse_coherence`, which additionally requires a
# narrative verse to share one location and carry at most one activity per
# shot. Left defined, unused, in case anything still imports it.
_MAX_ACTIVITIES_PER_VERSE = 2

_NAME_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "in", "on", "at", "of", "to", "for",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "black", "white", "brown", "tan", "toddlers", "toddler", "kids", "children",
    "girl", "girls", "boy", "boys", "she", "he", "they", "his", "her", "their",
    "is", "are", "was", "were", "this", "that", "little", "small", "big", "all",
}


# ------------------------------------------------------------------ utils ---
def _words(text):
    # äöüßÄÖÜ included so German words tokenize whole ("für" stays "für"
    # instead of splitting/corrupting to "fr") — every check downstream of
    # this (word/line counts, mean word length, repetition, hook detection)
    # inherits the fix for free. Unconditional (not language-gated): English
    # tokens never contain these characters, so English callers see byte-
    # identical output.
    return re.findall(r"[A-Za-zäöüßÄÖÜ']+", str(text))


def _contains_verb(text, verb):
    """Whole-word match for `verb` and its common inflections (claps/clapping/clapped)."""
    pattern = re.compile(r"\b" + re.escape(verb) + r"(e?s|ed|ing)?\b", re.IGNORECASE)
    return bool(pattern.search(text))


def _is_verbish(action):
    if any(_contains_verb(action, v) for v in _ACTION_VERBS):
        return True
    # Lenient fallback: any -ing/-ed/-s-suffixed token reads as "doing something"
    # (catches director._line_action's verbatim-lyric actions like "...song ends").
    return bool(re.search(r"\b[a-zA-Z]{3,}(ing|ed|s)\b", action))


def _capitalized_names(characters):
    """Capitalized tokens minus common non-name descriptor words — same heuristic
    as director._parse_character_names, duplicated here to keep this module
    import-light (no LLM-backend deps pulled in just to check a name count)."""
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


def _cast_names():
    """The bible's canonical cast names (Zuri/Kofi/Nala), used INSTEAD of
    `_capitalized_names` for any non-English language (see `_check_script`).

    German capitalizes every noun ("Sonne", "Hände"), so the capitalization
    heuristic that works for English would misread ordinary German nouns as
    cast names and false-fire the "character names overused" check on lyrics
    that never actually repeat a name. Matching against the KNOWN cast set
    sidesteps that entirely — and is correct regardless of language, because
    `characters` is always the English cast-bible sentence (see lyrics.py:
    `_fallback_characters`), even on a German episode.

    Lazy/guarded import, same reasoning as `_check_staging`'s `director`
    import: this module stays import-light, and a cast-bible load failure
    degrades to a small literal rather than crashing the gate.
    """
    try:
        from pipeline.kidsong import cast

        return list(cast.names())
    except Exception:
        return ["Zuri", "Kofi", "Nala"]


def _has_repetition(verses):
    """True if some 2+-word phrase (case-insensitive) occurs in >= 2 distinct lines
    of the song — whether that's a verse repeating its own opening line as its
    closing line (the common nursery-rhyme A-B-A shape, e.g. FALLBACK_SONG's
    "Clap, clap, clap your hands" bookending verse 1) or a phrase echoed across
    two different verses (a chorus/hook). Per-line dedup means a single line's
    own internal reduplication ("clap, clap, clap") doesn't trivially count."""
    phrase_lines = {}
    line_idx = 0
    for v in verses:
        lines = v.get("lines") if isinstance(v, dict) else None
        for line in lines or []:
            toks = [w.lower() for w in _words(str(line))]
            phrases_here = set()
            for n in (2, 3, 4):
                for j in range(len(toks) - n + 1):
                    phrases_here.add(" ".join(toks[j : j + n]))
            for phrase in phrases_here:
                phrase_lines.setdefault(phrase, set()).add(line_idx)
            line_idx += 1
    return any(len(idxs) >= 2 for idxs in phrase_lines.values())


def _verdict(reasons, weights, hint_key, default_weight=0.2):
    categories = {r.split(":", 1)[0].strip() for r in reasons}
    penalty = sum(weights.get(c, default_weight) for c in categories)
    score = max(0.0, min(1.0, 1.0 - penalty))
    accept = not reasons
    return {
        "accept": accept,
        "score": score,
        "reasons": list(reasons),
        "retry_hints": {hint_key: not accept},
    }


def _review_dir(cfg):
    from pipeline.config import abspath

    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    d = os.path.join(out_dir, "_kidsong_review")
    os.makedirs(d, exist_ok=True)
    return d


def _external_gate(stage, payload, verdict, cfg):
    """Offer `verdict` for external (human/Claude) review and return whatever
    comes back, or `verdict` itself if nobody answers.

    Wait duration is governed by the RUN's shared `ReviewSession` (see the
    "run sessions" section of review.py): once N requests anywhere in the run
    have gone unanswered, this writes its request and returns the programmatic
    verdict immediately instead of idling out `external_timeout`. The request
    file is written either way, so a human coming back later still sees it.
    Nothing here can change a verdict — a rejected script stays rejected.
    """
    review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
    timeout = float(review_cfg.get("external_timeout", 600))
    review_dir = _review_dir(cfg)
    request_path = os.path.join(review_dir, f"{stage}.request.json")
    response_path = os.path.join(review_dir, f"{stage}.response.json")

    full_payload = dict(payload)
    full_payload["programmatic_verdict"] = verdict

    response, skipped = external_gate_poll(
        stage, request_path, response_path, full_payload, timeout, cfg
    )
    if response is None:
        out = dict(verdict)
        out["reasons"] = list(verdict.get("reasons", [])) + [
            "unattended mode: external review skipped — programmatic verdict used"
            if skipped
            else "external review timed out"
        ]
        return out

    response = dict(response)
    response.setdefault("reasons", [])
    response.setdefault("retry_hints", dict(verdict.get("retry_hints", {})))
    response.setdefault("score", verdict.get("score", 0.5))
    response["accept"] = bool(response.get("accept"))
    return response


# ------------------------------------------------------------------ script ---
def _check_script(song, language="en"):
    reasons = []
    song = song if isinstance(song, dict) else {}

    title = str(song.get("title") or "").strip()
    if not title:
        reasons.append("title missing: song has no title")

    verses = song.get("verses") or []
    if not isinstance(verses, (list, tuple)) or not (_MIN_VERSES <= len(verses) <= _MAX_VERSES):
        reasons.append(
            f"verse count out of range: {len(verses) if isinstance(verses, (list, tuple)) else 0} "
            f"verses (need {_MIN_VERSES}-{_MAX_VERSES})"
        )

    all_lines = []
    for i, v in enumerate(verses if isinstance(verses, (list, tuple)) else []):
        lines = v.get("lines") if isinstance(v, dict) else None
        lines = lines if isinstance(lines, (list, tuple)) else []
        if not (_MIN_LINES <= len(lines) <= _MAX_LINES):
            reasons.append(
                f"verse count out of range: verse {i} has {len(lines)} lines "
                f"(need {_MIN_LINES}-{_MAX_LINES})"
            )
        all_lines.extend(str(l) for l in lines)

    for ln in all_lines:
        wc = len(_words(ln))
        if wc > _MAX_WORDS_PER_LINE:
            reasons.append(f"line too long: '{ln}' ({wc} words, max {_MAX_WORDS_PER_LINE})")

    all_words = [w for ln in all_lines for w in _words(ln)]
    if all_words:
        # German compounds run longer than English words at the same reading
        # level (see `_MAX_MEAN_WORD_LEN_DE`), so the cap is language-specific.
        max_mean_word_len = _MAX_MEAN_WORD_LEN if language == "en" else _MAX_MEAN_WORD_LEN_DE
        mean_len = sum(len(w) for w in all_words) / len(all_words)
        if mean_len >= max_mean_word_len:
            reasons.append(
                f"vocabulary too complex: mean word length {mean_len:.1f} chars "
                f"(max {max_mean_word_len})"
            )

    if isinstance(verses, (list, tuple)) and verses and not _has_repetition(verses):
        reasons.append("no repetition: no 2+ word phrase repeats across verses")

    # -- lyric craft: meter, rhyme, hook, phrasing (see the section above) --
    # Meter (`_syllables`/`_METER_TOLERANCE`), rhyme (`_rhyme_key`/
    # `_VOWEL_CLASSES`/`_RHYME_EQUIV`) and clumsy-phrasing (`_FUNCTION_WORDS`)
    # are all built from English spelling/pronunciation tables and have no
    # German equivalent here — building one is out of scope (see `_language`'s
    # docstring: the German path only ships curated library songs, so this
    # profile stays permissive rather than modeling German prosody). Skipped
    # entirely for non-English rather than false-firing on well-formed German
    # (confirmed: German's genitive/dative "mit ... mit"-style endings and
    # longer syllable counts trip these on real, singable German verses).
    for i, v in enumerate(verses if isinstance(verses, (list, tuple)) else []):
        lines = v.get("lines") if isinstance(v, dict) else None
        lines = [str(l) for l in (lines if isinstance(lines, (list, tuple)) else [])]
        singable = [l for l in lines if not _is_chant_line(l)]

        if language == "en" and len(singable) >= 2:
            counts = [_line_syllables(l) for l in singable]
            spread = max(counts) - min(counts)
            if spread > _METER_TOLERANCE:
                reasons.append(
                    f"meter inconsistent: verse {i} lines run {counts} syllables "
                    f"(spread {spread}, max {_METER_TOLERANCE}) — rewrite them to "
                    f"a steady beat of about {counts[0]} syllables each"
                )
            if not _verse_has_rhyme(singable):
                ends = [(_words(l) or [""])[-1] for l in singable]
                reasons.append(
                    f"no rhyme: verse {i} ends on {ends} — no two lines rhyme; "
                    "make at least one pair of line endings rhyme"
                )

        if language == "en":
            for line in lines:
                dup = _clumsy_repetition(line)
                if dup:
                    reasons.append(
                        f"clumsy phrasing: '{line}' repeats '{dup}' one word apart "
                        "('X ... X'), which reads as broken English — rewrite the line"
                    )

    hook = _hook_line(verses if isinstance(verses, (list, tuple)) else [])
    if all_lines and not hook:
        reasons.append(
            "no hook: no line of 3+ words repeats in two different verses — "
            "add a repeated chorus/hook line so the song is singable"
        )

    joined = " ".join(all_lines)
    # _ACTION_VERBS is an English lemma list (clap/stomp/wave/...); a German
    # verb (klatschen/stampfen/winken) will never match it, so this reward
    # would false-reject every German song regardless of how imitable its
    # actions actually are. Skipped for non-English rather than duplicating an
    # English-only vocabulary for a curated library that is already vetted.
    if language == "en":
        found_verbs = sorted(v for v in _ACTION_VERBS if _contains_verb(joined, v))
        if len(found_verbs) < 2:
            reasons.append(f"not enough action verbs: found {found_verbs} (need >= 2)")

    characters = str(song.get("characters") or "").strip()
    if language == "en":
        names = _capitalized_names(characters)
    else:
        # See `_cast_names`: German capitalizes every noun, so the
        # capitalization heuristic misreads ordinary nouns as cast names.
        # Match the known cast set instead — `characters` is always the
        # English cast-bible sentence, even for a German episode.
        names = [
            n for n in _cast_names()
            if re.search(rf"\b{re.escape(n)}\b", characters, re.IGNORECASE)
        ]
    # LLMs occasionally return a stringified dict/list ("{'Kofi': '...', ...}")
    # instead of one natural-language sentence — it would get pasted verbatim
    # into every shot's render prompt and garble every image. Reject on the
    # structural tell (wrapping braces/brackets or a dict-like "'key': 'val'"
    # pattern) regardless of whether enough capitalized names are present.
    looks_like_data = bool(
        re.match(r"^[\[{]", characters) or re.search(r"'\s*:\s*'", characters)
    )
    if not characters or len(names) < 2 or looks_like_data:
        reasons.append(
            f"characters sentence invalid: '{characters[:80]}' has {len(names)} "
            f"capitalized name(s) (need >= 2)"
            + (" and looks like a dict/list, not a sentence" if looks_like_data else "")
        )

    # A nursery rhyme does not narrate its own cast list. Names are checked
    # here (not in the per-verse loop) because the cast comes from the
    # `characters` sentence parsed just above.
    if names and len(all_lines) >= 4:
        named = [
            ln for ln in all_lines
            if any(re.search(rf"\b{re.escape(n)}\b", ln, re.IGNORECASE) for n in names)
        ]
        frac = len(named) / len(all_lines)
        if frac > _MAX_NAME_LINE_FRACTION:
            reasons.append(
                f"character names overused: {len(named)} of {len(all_lines)} lines "
                f"name a character ({frac:.0%}, max {_MAX_NAME_LINE_FRACTION:.0%}) — "
                "the pictures show who is on screen, so the lyrics should not keep "
                "listing names"
            )

    full_text = " ".join(
        [
            title,
            str(song.get("description") or ""),
            characters,
            joined,
            " ".join(str(t) for t in (song.get("tags") or [])),
        ]
    ).lower()
    hit_brands = sorted(b for b in _BRAND_WORDS if b in full_text)
    if hit_brands:
        reasons.append(f"brand words present: {hit_brands}")

    return reasons


def review_script(song, cfg=None):
    if cfg is None:
        from pipeline.config import load_config

        cfg = load_config()

    reasons = _check_script(song, _language(cfg))
    reasons.extend(_check_verse_scenes(song))
    verdict = _verdict(reasons, _SCRIPT_WEIGHTS, "regenerate")

    review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
    if str(review_cfg.get("reviewer", "heuristic")).lower() == "external":
        verdict = _external_gate("script", {"song": song}, verdict, cfg)
    return verdict


# --------------------------------------------------------------- shotlist ---
def _char_count(shot):
    chars = shot.get("characters") or []
    if isinstance(chars, str):
        chars = [chars]
    if list(chars) == ["all"]:
        return 3
    return len(chars)


def _check_shotlist(shots):
    reasons = []
    shots = list(shots or [])
    if not shots:
        return ["no shots: shot list is empty"]

    ids = {s.get("id") for s in shots}
    ordered = sorted(shots, key=lambda s: float(s.get("start", 0) or 0))

    # -- contiguous tiling across the whole timeline --
    for a, b in zip(ordered, ordered[1:]):
        gap = abs(float(b.get("start", 0) or 0) - float(a.get("end", 0) or 0))
        if gap >= _TILE_GAP_TOLERANCE:
            reasons.append(
                f"verse tiling gap: shots {a.get('id')}/{b.get('id')} gap {gap:.3f}s "
                f"(max {_TILE_GAP_TOLERANCE}s)"
            )

    # -- durations --
    for s in ordered:
        d = float(s.get("end", 0) or 0) - float(s.get("start", 0) or 0)
        if not (_MIN_SHOT_SECONDS - 1e-6 <= d <= _MAX_SHOT_SECONDS + 1e-6):
            reasons.append(
                f"duration out of range: shot {s.get('id')} duration {d:.2f}s "
                f"(need {_MIN_SHOT_SECONDS}-{_MAX_SHOT_SECONDS}s)"
            )

    # -- first shot (chronologically) is a wide establishing shot --
    first = ordered[0]
    if str(first.get("shot_type", "")).lower() != "wide":
        reasons.append(f"first shot not wide: shot {first.get('id')} is '{first.get('shot_type')}'")

    # -- no 3 consecutive same shot_type WITHIN a verse (director's own invariant) --
    by_verse = {}
    for s in ordered:
        by_verse.setdefault(s.get("verse"), []).append(s)
    for v, vshots in by_verse.items():
        vshots = sorted(vshots, key=lambda s: float(s.get("start", 0) or 0))
        types = [str(s.get("shot_type", "")).lower() for s in vshots]
        for i in range(len(types) - 2):
            if types[i] and types[i] == types[i + 1] == types[i + 2]:
                reasons.append(
                    f"3 consecutive same shot_type: verse {v} shots "
                    f"{vshots[i].get('id')}/{vshots[i+1].get('id')}/{vshots[i+2].get('id')} "
                    f"all '{types[i]}'"
                )

        # Reference analysis: ~90% of real preschool-TV shots have gentle
        # continuous motion — motion is the norm, not the exception. Only
        # flag a verse where every single shot uses the SAME camera move.
        cams = [str(s.get("camera") or "static").lower() for s in vshots]
        if len(vshots) >= 3 and len(set(cams)) == 1 and cams[0] != "static":
            reasons.append(f"monotonous camera: verse {v} uses '{cams[0]}' on every shot")

    # -- >= 60% of shots have <= 2 characters (uncluttered rule) --
    uncluttered = sum(1 for s in ordered if _char_count(s) <= 2)
    frac = uncluttered / len(ordered)
    if frac < _UNCLUTTERED_FRACTION:
        reasons.append(
            f"cluttered frame: only {frac:.0%} of shots have <= 2 characters "
            f"(need >= {_UNCLUTTERED_FRACTION:.0%})"
        )

    # -- every action non-empty and verb-ish --
    for s in ordered:
        action = str(s.get("action") or "").strip()
        if not action or not _is_verbish(action):
            reasons.append(f"action missing/not verb-ish: shot {s.get('id')} action='{action}'")

    # -- reuse_of ids resolve --
    for s in ordered:
        ref = s.get("reuse_of")
        if ref and ref not in ids:
            reasons.append(f"unresolved reuse_of: shot {s.get('id')} references missing '{ref}'")

    reasons.extend(_check_head_count_consistency(ordered))
    reasons.extend(_check_staging(ordered))
    reasons.extend(_check_verse_coherence(by_verse))

    return reasons


def _check_staging(ordered):
    """Reject shots staged somewhere the channel forbids, staged martially, or
    staged around something frightening.

    All three are PLAN faults, not sampling faults — re-seeding a shot whose
    setting says "a bright bathroom with a bubbly tub" renders another bathtub.
    Measured instances this gate is built from:
      * output/20260720-110129-...-watering-blooms-day-shots.json shots s05-s09
        set a song about watering GARDEN FLOWERS in "a bright bathroom with a
        bubbly tub behind them, a rubber duck on the tub edge". Rendered as a
        full bathtub scene in s08_a0.mp4 — a content-rule breach that shipped.
      * The rendered wides of the feel-the-breeze episode put all three
        toddlers in an evenly-spaced row striding at camera in unison.

    Vocabulary lives in `director` so the gate and the sanitizers cannot drift
    apart — same reasoning as `_check_head_count_consistency`.
    """
    try:
        from pipeline.kidsong import director
    except Exception:
        return []

    reasons = []
    for s in ordered:
        setting = str(s.get("setting") or "").strip()
        forbidden = director.forbidden_location_terms(setting)
        if forbidden:
            reasons.append(
                f"forbidden location: shot {s.get('id')} is set in a place the "
                f"channel does not allow ({', '.join(sorted(forbidden))}) — "
                f"setting={setting!r}. Rewrite the setting as a garden, "
                f"backyard, playroom, kitchen, bedroom or classroom; water play "
                f"belongs outdoors with watering cans or puddles, never a tub, "
                f"bath, pool or potty."
            )

        for field in ("action", "setting"):
            text = str(s.get(field) or "").strip()
            if not text:
                continue

            martial = director.martial_terms(text)
            if martial:
                reasons.append(
                    f"martial staging: shot {s.get('id')} {field} uses "
                    f"martial/lockstep language ({', '.join(sorted(martial))}) — "
                    f"{field}={text!r}. Preschool staging is loose, playful and "
                    f"small-group: no marching, rows, formations, lines, "
                    f"uniforms, saluting or children moving in unison. Stage "
                    f"them as an informal cluster at different distances."
                )

            if field == "action":
                action_where = director.location_class(text)
                setting_where = director.location_class(setting)
                if action_where and setting_where and action_where != setting_where:
                    reasons.append(
                        f"indoor/outdoor contradiction: shot {s.get('id')} has an "
                        f"{action_where} action in an {setting_where} setting — "
                        f"action={text!r}, setting={setting!r}. The renderer "
                        f"satisfies both halves literally and produces an "
                        f"impossible space (a room with a grass floor). Move the "
                        f"action indoors or the setting outdoors so they agree."
                    )

            scary = director.scary_terms(text)
            if scary:
                reasons.append(
                    f"frightening content: shot {s.get('id')} {field} contains "
                    f"language that is not wholesome for toddlers "
                    f"({', '.join(sorted(scary))}) — {field}={text!r}. Every "
                    f"shot must be calm and happy; remove the frightening or "
                    f"distressing element entirely rather than softening it."
                )

    return reasons


def _local_activity_progression(originals, director):
    """Stand-in for `director.verse_activity_progression`, used ONLY while
    that helper has not landed yet — it is being added by a concurrent change
    to director.py, so this module must not go silent (or crash) if it runs
    before that lands. Mirrors the same fixed contract:
      * 'uniform': <=1 distinct non-filler activity verb across the verse.
      * 'narrative': one shared setting, every shot carries at most ONE
        non-filler activity verb, and <=4 distinct verbs across the verse.
      * 'scattered': anything else.
    Prefer `director.verse_activity_progression` (the real, single source of
    truth) the instant it exists — see `_check_verse_coherence` below.
    """
    verbs_per_shot = [
        director.activity_verbs(str(s.get("action") or "")) for s in originals
    ]
    all_verbs = set()
    for v in verbs_per_shot:
        all_verbs |= v
    if len(all_verbs) <= 1:
        return "uniform"

    settings = {
        str(s.get("setting") or "").strip() for s in originals
        if str(s.get("setting") or "").strip()
    }
    one_verb_per_shot = all(len(v) <= 1 for v in verbs_per_shot)
    if len(settings) <= 1 and one_verb_per_shot and len(all_verbs) <= 4:
        return "narrative"
    return "scattered"


def _check_verse_coherence(by_verse):
    """Reject a verse whose shots do not read as ONE continuous moment.

    Two faults:
      * NO THROUGH-LINE — `director.verse_activity_progression` classifies a
        verse's shots as 'uniform' (one shared activity), 'narrative' (a
        legitimate setup -> event -> reaction arc: one shared setting, one
        activity verb per shot, a handful of verbs total — e.g. "the clock
        strikes one" / "a mouse scampers down the clock case" / "Zuri points
        and giggles"), or 'scattered' (no through-line at all). Only
        'scattered' is rejected here. This intentionally does NOT reject a
        narrative verse for spanning 3+ activity verbs — that used to be
        flagged by a flat "<=2 distinct verbs" cap, which rejected real
        storytelling along with genuinely broken verses. The SIMULTANEOUS
        fault (different children handed different activities in the same
        beat, e.g. the feel-the-breeze verse: "Zuri holds a bubble wand, Kofi
        stomps in a puddle, Nala dances with arms outstretched") is caught
        earlier, on the LYRICS scene text, by `_check_verse_scenes` — by the
        time a shot list exists that fault has already been repaired, so this
        gate does not re-derive it from shot verb counts alone.
      * LOCATION DISCONTINUITY — the setting changes MID-verse. A deliberate,
        signposted location change belongs between verses; inside one it just
        reads as a continuity error.
    """
    try:
        from pipeline.kidsong import director
    except Exception:
        return []

    reasons = []
    for verse, vshots in sorted(by_verse.items(), key=lambda kv: (kv[0] is None, kv[0])):
        # Reused (chorus) shots deliberately echo an earlier verse — judging
        # them here would double-report the source verse's own faults.
        originals = [s for s in vshots if not s.get("reuse_of")]
        if not originals:
            continue

        progression_fn = getattr(director, "verse_activity_progression", None)
        progression = (
            progression_fn(originals) if progression_fn is not None
            else _local_activity_progression(originals, director)
        )
        if progression == "scattered":
            verbs = set()
            for s in originals:
                verbs |= director.activity_verbs(str(s.get("action") or ""))
            listed = ", ".join(sorted(verbs))
            reasons.append(
                f"verse mixes activities: verse {verse} does not read as one "
                f"continuous moment ({len(verbs)} activities: {listed}). One "
                f"shot shows ONE activity; a verse's shots may progress "
                f"through a temporal arc (setup, event, reaction) but must "
                f"share one location, and must not show different children "
                f"doing different things at the same time."
            )

        settings = {
            str(s.get("setting") or "").strip()
            for s in originals
            if str(s.get("setting") or "").strip()
        }
        if len(settings) > 1:
            listed = ", ".join(repr(x) for x in sorted(settings))
            reasons.append(
                f"verse location discontinuity: verse {verse} changes location "
                f"between its own shots ({listed}). Every shot in a verse shares "
                f"one location and one lighting state; change location only "
                f"BETWEEN verses."
            )

    reasons.extend(_check_location_ping_pong(by_verse))
    return reasons


def _check_location_ping_pong(by_verse):
    """Reject a song that RETURNS to a location it already left (A -> B -> A).

    Real preschool songs move forward through places. Bouncing back is read by a
    toddler as a continuity error, and it is what the feel-the-breeze episode
    shipped: verse 0 in a playroom, verse 1 in a backyard, verse 2 back in the
    playroom — for a song whose entire lyric is about being outside. The
    per-verse checks above cannot see this, because each verse is internally
    consistent; only the sequence is wrong.
    """
    ordered_verses = sorted(
        (v for v in by_verse if v is not None), key=lambda v: (isinstance(v, str), v)
    )

    sequence = []
    for verse in ordered_verses:
        originals = [s for s in by_verse[verse] if not s.get("reuse_of")]
        settings = [
            str(s.get("setting") or "").strip() for s in originals
            if str(s.get("setting") or "").strip()
        ]
        if not settings:
            continue
        if not sequence or sequence[-1][1] != settings[0]:
            sequence.append((verse, settings[0]))

    reasons = []
    seen = {}
    for position, (verse, setting) in enumerate(sequence):
        if setting in seen:
            reasons.append(
                f"location ping-pong: verse {verse} returns to a location the "
                f"song already left in verse {seen[setting]} — setting={setting!r}. "
                f"Move forward through places (e.g. bedroom -> kitchen -> "
                f"backyard) instead of going back; if the song stays in one "
                f"place, keep EVERY verse there."
            )
        else:
            seen[setting] = verse

    return reasons


def _check_head_count_consistency(ordered):
    """Reject a shot whose action/setting prose puts more children on screen than
    its own `characters` list allows.

    generate.py builds the render prompt's head-count sentence from `characters`
    and then appends `action` and `setting` verbatim, so a shot listing one child
    with an action reading "The kids smile brightly at each other" ships a prompt
    that asserts one child and three children in the same breath. Two GPU A/B
    rounds measured that contradiction as the dominant remaining defect (the
    renderer resolves it by cloning the described child 2-3x), which makes this a
    structural shot-list fault of exactly the kind this gate already catches.

    `director` is imported lazily: this module is deliberately import-light, but
    the group-language vocabulary must have ONE definition or the gate and the
    sanitizer drift apart and the gate starts passing what the sanitizer rewrites.
    """
    try:
        from pipeline.kidsong import director
    except Exception:
        return []

    reasons = []
    for s in ordered:
        allowed = _char_count(s)
        for field in ("action", "setting"):
            text = str(s.get(field) or "").strip()
            if not text:
                continue
            implied = director.implied_child_count(text)
            if implied > allowed:
                reasons.append(
                    f"head count contradiction: shot {s.get('id')} lists "
                    f"{allowed} character(s) {s.get('characters')} but its {field} "
                    f"implies {implied} on screen — {field}={text!r}"
                )

        setting = str(s.get("setting") or "").strip()
        if setting and not director.setting_is_a_place(setting):
            reasons.append(
                f"setting is not a place: shot {s.get('id')} setting describes "
                f"people or action rather than a location — setting={setting!r}"
            )

    return reasons


def _check_verse_scenes(song):
    """Reject a verse whose SCENE text hands different children different
    activities — the precise form of the fault the channel owner reported.

    This runs on the SCRIPT gate, not the shot-list gate, deliberately. The
    scene sentence is a LYRICS artifact, and `director` already collapses a
    mixed scene to one shared activity while planning — so by the time a shot
    list exists the fault is repaired and rejecting the shot list for it would
    be non-actionable: the "replan" retry cannot rewrite the song, so the loop
    could never converge. Reported here instead, the "regenerate" retry hint
    points at the thing that can actually change. The shot-list gate still
    independently rejects a verse whose SHOTS mix activity verbs.

    This is the sharpest of the coherence checks because it looks at the
    planner's INPUT rather than its output: "Zuri holds a bubble wand, Kofi
    stomps in a puddle, Nala dances with arms outstretched" names three
    children doing three unrelated things, and no shot list built from it can
    be coherent. It fires only when 2+ distinct CAST children are given 2+
    distinct verbs, so a verse with one child and an animal ("Zuri had a little
    lamb, he followed her to school") is not caught — that is one shared
    activity with a companion in frame, not a mixed verse.

    Songs whose verses carry no "scene" (the public-domain nursery-rhyme
    corpus) are simply not constrained here.
    """
    try:
        from pipeline.kidsong import director
    except Exception:
        return []

    verses = (song or {}).get("verses") if isinstance(song, dict) else None
    reasons = []
    for i, v in enumerate(verses or []):
        scene = str((v or {}).get("scene") or "").strip() if isinstance(v, dict) else ""
        if not scene or not director.verse_mixes_activities(scene):
            continue
        shared = director.reduce_scene_to_shared_activity(scene)
        reasons.append(
            f"verse mixes activities: verse {i} scene gives different children "
            f"different activities — scene={scene!r}. A verse is ONE continuous "
            f"moment; rewrite it as a single activity the whole cast shares "
            f"(e.g. {shared!r}) and let the shots differ only in framing and in "
            f"which child is on screen."
        )
    return reasons


def _unresolved_cast_names(shots):
    """`[{"shot": id, "names": [...]}, ...]` for every shot naming a character
    that is not in the cast bible (and has no alias mapping to one).

    QM-032. `cast.unresolved_names` was written so "the script/shotlist QC gate
    can SURFACE an invented character name instead of letting the render
    silently paper over it" — but it had no caller, so it never did. At render
    time `cast._resolve_with_fallback` substitutes a deterministic bible child
    and logs a warning nobody reads, which means the shot list, the song's
    scene prose and the episode's planning artefacts keep naming a fourth child
    who does not exist and cannot be identity-anchored (`refs.reference_for`
    returns None for an unknown id) or cross-negated (IMP-013 derives negatives
    from the bible). Measured 2026-07-25: 27 shot slots across 4 episodes named
    kiara / jaxson / nia / jaden / ava, none of which resolve.

    This is DELIBERATELY not a `_check_shotlist` reason: `_verdict` sets
    `accept = not reasons`, so adding one there would make an invented name a
    hard shotlist reject and force a replan loop, and there is no evidence yet
    on how often the director does this under the current prompts. Surfacing it
    in the gate's request payload is the additive first step — it cannot change
    the programmatic verdict, it only puts the fact in front of the reviewer
    answering the shotlist gate. Promote it to a reason once the next audit has
    a frequency number (see DEFECT_BACKLOG QM-032).

    Note this reports alias-rescued names (maya/luna/amira/...) too, because
    `cast.unresolved_names` matches ids and canonical names only. That is the
    intended reading: an alias is a rescue for a name the director should not
    have invented, not a licence to keep inventing it.

    Never raises: a cast-bible load failure degrades to "nothing to report",
    same posture as `_cast_names`.
    """
    try:
        from pipeline.kidsong import cast
    except Exception:
        return []

    found = []
    for s in list(shots or []):
        if not isinstance(s, dict):
            continue
        try:
            names = cast.unresolved_names(s.get("characters"))
        except Exception:
            continue
        if names:
            found.append({"shot": s.get("id"), "names": list(names)})
    return found


def review_shotlist(shotlist, song, cfg=None):
    if cfg is None:
        from pipeline.config import load_config

        cfg = load_config()

    shots = shotlist.get("shots") if isinstance(shotlist, dict) else shotlist
    reasons = _check_shotlist(shots)
    verdict = _verdict(reasons, _SHOTLIST_WEIGHTS, "replan")

    review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
    if str(review_cfg.get("reviewer", "heuristic")).lower() == "external":
        payload = {"song": song, "shotlist": shotlist}
        # Additive advisory (QM-032) — informational only, never part of the
        # programmatic verdict. Omitted entirely when there is nothing to say,
        # so an untouched shotlist writes a byte-identical request.
        off_cast = _unresolved_cast_names(shots)
        if off_cast:
            payload["unresolved_character_names"] = off_cast
        verdict = _external_gate("shotlist", payload, verdict, cfg)
    return verdict


# ------------------------------------------------------------------ test ---
if __name__ == "__main__":
    import copy
    import json

    from pipeline.config import load_config
    from pipeline.kidsong.director import _normalize_shots
    from pipeline.kidsong.lyrics import FALLBACK_SONG

    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("kidsong", {}).setdefault("review", {})["reviewer"] = "heuristic"

    song = json.loads(json.dumps(FALLBACK_SONG))  # cheap deep copy
    verse_times = [(0, 15), (15, 30), (30, 45), (45, 60)]
    beats = {"bpm": 96, "beat_times": [i * 0.625 for i in range(97)]}

    # The DETERMINISTIC planner, not `plan_shots`. `plan_shots` calls the
    # configured LLM at temperature 0.7 and overlays its choices, so asserting
    # that its output passes this gate is not a valid invariant — the gate
    # exists precisely BECAUSE the LLM sometimes plans a bad shot list, and the
    # assertion failed on roughly every run whenever Ollama happened to be up
    # (measured 3/3 failures, with different reasons each time). What IS an
    # invariant, and what this self-test means to check, is that the
    # deterministic fallback skeleton always satisfies the gate.
    shotlist = {"shots": _normalize_shots(None, song, verse_times, beats, cfg)}

    v_script = review_script(song, cfg)
    print("script verdict (fallback song):", v_script)
    assert v_script["accept"] is True, v_script

    v_shotlist = review_shotlist(shotlist, song, cfg)
    print("shotlist verdict (fallback shotlist):", v_shotlist)
    assert v_shotlist["accept"] is True, v_shotlist

    # -- now break the song: overlong lines, no cross-verse repetition, no
    # -- action verbs, and an uninformative characters sentence --
    broken_song = {
        "title": "Broken Test Song",
        "description": "test",
        "tags": ["a"],
        "characters": "A happy toddler plays outside all afternoon long today.",
        "verses": [
            {
                "lines": [
                    "Purple elephants juggle enormous mathematics textbooks beside sleepy mountain lighthouses",
                    "Silver dolphins compose intricate symphonies inside forgotten crystal libraries endlessly",
                ],
                "scene": "x",
            },
            {
                "lines": [
                    "Golden giraffes analyze peculiar geometry puzzles atop wandering desert caravans",
                    "Emerald tigers assemble curious mechanical contraptions within abandoned volcanic laboratories",
                ],
                "scene": "y",
            },
            {
                "lines": [
                    "Crimson falcons decipher ancient philosophical manuscripts across frozen arctic tundras",
                    "Turquoise octopi construct fantastical architectural monuments beneath turbulent oceanic trenches",
                ],
                "scene": "z",
            },
        ],
    }

    v_broken = review_script(broken_song, cfg)
    print("script verdict (broken song):", v_broken)
    assert v_broken["accept"] is False, v_broken
    assert v_broken["retry_hints"].get("regenerate") is True, v_broken
    reasons_joined = " | ".join(v_broken["reasons"])
    assert "line too long" in reasons_joined, v_broken["reasons"]
    assert "no repetition" in reasons_joined, v_broken["reasons"]
    assert "not enough action verbs" in reasons_joined, v_broken["reasons"]
    assert "characters sentence invalid" in reasons_joined, v_broken["reasons"]

    # -- a shotlist with an obviously bad shot (huge gap, tiny duration, no
    # -- action, dangling reuse_of, wrong first shot_type) should reject --
    broken_shots = {
        "shots": [
            {
                "id": "s00",
                "verse": 0,
                "start": 0.0,
                "end": 3.0,
                "shot_type": "medium",  # should be "wide"
                "characters": ["all"],
                "action": "",  # empty
                "camera": "static",
                "reuse_of": None,
            },
            {
                "id": "s01",
                "verse": 0,
                "start": 5.0,  # gap vs previous shot's end (3.0)
                "end": 5.3,  # duration 0.3s, below MIN_SHOT_SECONDS
                "shot_type": "medium",
                "characters": ["all"],
                "action": "stares blankly",
                "camera": "static",
                "reuse_of": "s99",  # unresolved
            },
        ]
    }
    v_bad_shotlist = review_shotlist(broken_shots, song, cfg)
    print("shotlist verdict (broken shotlist):", v_bad_shotlist)
    assert v_bad_shotlist["accept"] is False, v_bad_shotlist
    assert v_bad_shotlist["retry_hints"].get("replan") is True, v_bad_shotlist
    reasons_joined = " | ".join(v_bad_shotlist["reasons"])
    assert "first shot not wide" in reasons_joined, v_bad_shotlist["reasons"]
    assert "duration out of range" in reasons_joined, v_bad_shotlist["reasons"]
    assert "unresolved reuse_of" in reasons_joined, v_bad_shotlist["reasons"]
    assert "verse tiling gap" in reasons_joined, v_bad_shotlist["reasons"]

    print("PASS")
