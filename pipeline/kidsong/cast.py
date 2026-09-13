"""
kidsong.cast — The ZubiBop CAST BIBLE: a fixed, versioned set of recurring characters.

Before this module existed, the "characters" field of every song was one free-text
sentence the LLM invented per song (see kidsong.lyrics' `characters` field). Because
nothing pinned the cast's appearance across songs, rendered takes drifted episode to
episode: a yellow-shirt boy where Kofi should be blue-shirted, a purple-dress girl
where Nala should wear yellow, a pale/light-haired middle child instead of Kofi's
established dark curly hair.

This module makes the cast FIXED: three canonical toddlers (Zuri, Kofi, Nala) loaded
from ``prompts/cast_bible.json``, each with a pinned skin/hair/outfit description, a
stable per-character render seed, and a `must_not` list of looks that must never
appear on them. Callers (lyrics/director/generate/scenes prompts) ask this module for
the canonical appearance sentence instead of trusting whatever the LLM wrote.

Stdlib only — no network, no GPU, safe to import from anywhere in the pipeline
(including request-time Flask code) without pulling in heavy deps.
"""
import hashlib
import json
import logging
import os
import re

log = logging.getLogger("kidsong.cast")

# ------------------------------------------------------------------- paths ---
# pipeline/kidsong/cast.py -> pipeline/kidsong -> pipeline -> <repo root>
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIBLE_PATH = os.path.join(_ROOT, "prompts", "cast_bible.json")

# Channel-wide negative-prompt floor: always included in `negative_terms`, on top
# of whatever the requested characters' own `must_not` lists contribute.
_CHANNEL_FLOOR = [
    "shirtless child",
    "bare chest",
    "undressed",
    "underwear",
    "extra children",
    "crowd of children",
    "duplicate characters",
    "jewellery",
    "earrings",
]

_REQUIRED_TEXT_FIELDS = ("name", "skin", "hair", "top")

_NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}

_bible_cache = None
_bible_cache_path = None
_bible_cache_mtime = None


# ------------------------------------------------------------------ loading ---
def load_bible(path=None):
    """Load + validate the cast bible. Cached. Raises ValueError on a malformed bible.

    The cache is invalidated on the file's mtime, not just its path: a
    long-lived process (the Flask studio server, the scheduler) must see an
    edited bible — a new cast reference_image, a fixed wardrobe — without a
    restart. A same-path load whose mtime is unchanged still hits the cache, so
    the hot path stays a single ``stat``.
    """
    global _bible_cache, _bible_cache_path, _bible_cache_mtime

    load_path = os.path.abspath(path) if path else BIBLE_PATH
    try:
        mtime = os.path.getmtime(load_path)
    except OSError:
        mtime = None
    if (_bible_cache is not None and _bible_cache_path == load_path
            and _bible_cache_mtime == mtime):
        return _bible_cache

    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _validate_bible(data, load_path)

    _bible_cache = data
    _bible_cache_path = load_path
    _bible_cache_mtime = mtime
    return data


def _validate_bible(data, source):
    if not isinstance(data, dict):
        raise ValueError(f"Cast bible at {source} is not a JSON object")

    if not str(data.get("version") or "").strip():
        raise ValueError(f"Cast bible at {source} is missing a non-empty 'version'")

    characters = data.get("characters")
    if not isinstance(characters, (list, tuple)) or len(characters) < 2:
        raise ValueError(
            f"Cast bible at {source} must have at least 2 'characters' entries "
            f"(found {len(characters) if isinstance(characters, (list, tuple)) else 0})"
        )

    seen_ids, seen_names, seen_offsets = set(), set(), set()
    for i, c in enumerate(characters):
        if not isinstance(c, dict):
            raise ValueError(f"Cast bible at {source}: characters[{i}] is not an object")

        for field in _REQUIRED_TEXT_FIELDS:
            if not str(c.get(field) or "").strip():
                raise ValueError(
                    f"Cast bible at {source}: characters[{i}] is missing a non-empty '{field}'"
                )

        cid = str(c.get("id") or "").strip().lower()
        if not cid:
            raise ValueError(f"Cast bible at {source}: characters[{i}] is missing a non-empty 'id'")
        if cid in seen_ids:
            raise ValueError(f"Cast bible at {source}: duplicate character id '{cid}'")
        seen_ids.add(cid)

        name_key = str(c["name"]).strip().lower()
        if name_key in seen_names:
            raise ValueError(f"Cast bible at {source}: duplicate character name '{c['name']}'")
        seen_names.add(name_key)

        offset = c.get("seed_offset")
        if offset is None:
            raise ValueError(f"Cast bible at {source}: characters[{i}] is missing 'seed_offset'")
        if offset in seen_offsets:
            raise ValueError(f"Cast bible at {source}: duplicate seed_offset {offset}")
        seen_offsets.add(offset)


# --------------------------------------------------------------- resolution ---
def _all():
    return list(load_bible()["characters"])


def character(name_or_id):
    """Case-insensitive exact lookup by name or id. Returns None if not a cast member.

    Does NOT substring-match — 'Ama' must not resolve 'Amara'.
    """
    key = str(name_or_id or "").strip().lower()
    if not key:
        return None
    for c in _all():
        if str(c["id"]).strip().lower() == key or str(c["name"]).strip().lower() == key:
            return dict(c)
    return None


def _normalize_names(names):
    """`names` -> (is_ensemble, list_of_raw_names). None or an 'all' entry means
    the whole cast; a bare string is treated as a one-element list."""
    if names is None:
        return True, []
    if isinstance(names, str):
        names = [names]
    names = [str(n) for n in names]
    if any(n.strip().lower() == "all" for n in names):
        return True, names
    return False, names


def _split_names(names):
    """(resolved_chars_in_bible_order, unresolved_raw_names) for a non-ensemble
    name list. Duplicates of an already-resolved character are dropped."""
    resolved, seen_ids, unresolved = [], set(), []
    for n in names:
        c = character(n)
        if c and c["id"] not in seen_ids:
            seen_ids.add(c["id"])
            resolved.append(c)
        elif not c:
            unresolved.append(n)
    return resolved, unresolved


def _resolve_exact(names):
    """Resolve `names` to distinct character dicts, in canonical bible order.

    None, or any list containing 'all' (case-insensitive), resolves to the full
    cast. Unknown names are dropped; may return an empty list (never falls back
    — callers that want a fallback go through `_resolve_with_fallback`).
    """
    is_ensemble, names = _normalize_names(names)
    if is_ensemble:
        return _all()

    resolved, unresolved = _split_names(names)
    if unresolved:
        # Silent fallback is what hid the BUG 2 disagreement between
        # `cast_sentence` (falls back to the ensemble) and the old
        # `expected_child_count` (didn't) — logging every unresolvable name
        # makes a hallucinated/mistyped LLM name visible instead of just
        # quietly changing who's on screen.
        log.warning(
            "cast: unresolvable character name(s) %s (not in the cast bible) — ignored",
            unresolved,
        )
    return resolved


# ------------------------------------------------- unresolvable-name policy ---
# A shot whose named characters aren't in the bible used to expand to the FULL
# ENSEMBLE. That is how a CLOSEUP OF ONE CHILD ended up prompting "Exactly three
# children are on screen: ..." and rendering three — the model obeyed a prompt
# that contradicted the shot. The LLM routinely invents names ("Amira", "Luna"),
# so this was not a rare edge case.
#
# The policy now is: an unresolvable name is SUBSTITUTED, one-for-one, by a
# bible character, and the resulting head count is capped by what the SHOT asked
# for. Never more people than the shot specified.
_SINGLE_SUBJECT_SHOT_TYPES = ("closeup", "medium")


def _target_count(names, shot_type, total):
    """How many children the shot may resolve to. A closeup/medium is a
    single-subject frame by construction, so it caps at one no matter how many
    names the LLM attached to it; anything else caps at the number of names the
    shot actually requested. Never exceeds the cast size."""
    if str(shot_type or "").strip().lower() in _SINGLE_SUBJECT_SHOT_TYPES:
        return 1
    return max(1, min(len(names), total))


def _alias_map():
    """Optional `aliases` block in the bible: invented/legacy name -> character
    id. Lets a recurring hallucination ('Amira', 'Luna') map to the SAME cast
    member every time instead of to a hash-chosen one."""
    raw = load_bible().get("aliases") or {}
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}


def _stable_pick(name, pool):
    """Deterministic choice from `pool`, keyed on the name's text.

    md5 — NOT Python's `hash()`, which is salted per process by PYTHONHASHSEED
    and would hand the same invented name a different child on every run,
    breaking exactly the episode-to-episode consistency the bible exists for.
    """
    digest = hashlib.md5(str(name).strip().lower().encode("utf-8")).hexdigest()
    return pool[int(digest, 16) % len(pool)]


def substitute_for(name, taken_ids=None):
    """The bible character an unresolvable `name` stands in for: the alias-table
    entry if there is one, otherwise a stable-hash pick. `taken_ids` are already
    on screen and won't be picked twice."""
    taken_ids = set(taken_ids or ())
    all_chars = _all()
    pool = [c for c in all_chars if c["id"] not in taken_ids]
    if not pool:
        return None
    target_id = _alias_map().get(str(name).strip().lower())
    for c in pool:
        if c["id"] == target_id:
            return c
    return _stable_pick(name, pool)


def unresolved_names(names):
    """The subset of `names` that is not in the cast bible. Exposed so the
    script/shotlist QC gate can SURFACE an invented character name instead of
    letting the render silently paper over it."""
    is_ensemble, names = _normalize_names(names)
    if is_ensemble:
        return []
    return _split_names(names)[1]


def _resolve_with_fallback(names, shot_type=None):
    """The single resolver behind `cast_sentence`, `negative_terms`, `seed_for`
    AND `expected_child_count`. All four MUST agree on who is on screen, or the
    prompt's head-count sentence, its cast description and its negatives
    contradict each other (and a head-count check rejects a render that
    correctly matched what was described).

    Unresolvable names are substituted one-for-one (see `substitute_for`) and
    the result is capped at `_target_count`, so a closeup that named a
    hallucinated child yields exactly ONE bible character — deterministically
    the same one on every run — rather than the whole ensemble.
    """
    return _resolve_mapped(names, shot_type=shot_type)[0]


def substitution_map(names, shot_type=None):
    """`{invented_name: bible_name}` — who each off-bible name in `names` was
    replaced by, using EXACTLY the substitution `_resolve_with_fallback` applies.

    Exists so a caller can repair the same invented name where it also appears
    in free text (the shot's `action`/`setting`). Without it the identity
    sentence said "Exactly three children are on screen: Zuri, Kofi and Nala"
    while the action right after it still read "Amira and Kofi standing…" — a
    fourth, undescribed name the text encoder then had to invent a child for,
    directly against the head count the sentence before it just asserted.

    Names that were dropped rather than substituted (the shot already had its
    `_target_count` of children) map to a character that IS on screen, so the
    action can never name somebody the identity sentence excluded. Never logs —
    `_resolve_with_fallback` already reported these.
    """
    return _resolve_mapped(names, shot_type=shot_type, log_substitutions=False)[1]


def _resolve_mapped(names, shot_type=None, log_substitutions=True):
    """`(picked_chars, {invented_name: bible_name})` — the shared core of
    `_resolve_with_fallback` and `substitution_map`, so the cast that gets
    described and the names rewritten into the action can never diverge."""
    all_chars = _all()
    is_ensemble, raw = _normalize_names(names)
    if is_ensemble:
        return all_chars, {}

    resolved, unresolved = _split_names(raw)
    if not unresolved:
        # Every name resolved (or the list was empty) — untouched behaviour.
        return (resolved if resolved else all_chars), {}

    target = _target_count(raw, shot_type, len(all_chars))
    taken = {c["id"] for c in resolved}
    picked = list(resolved)
    mapping = {}
    dropped = []
    for n in unresolved:
        if len(picked) >= target:
            dropped.append(n)
            continue
        c = substitute_for(n, taken)
        if c is None:
            dropped.append(n)
            continue
        taken.add(c["id"])
        picked.append(c)
        mapping[str(n)] = c["name"]
        if log_substitutions:
            log.warning(
                "cast: unresolvable character name %r (not in the cast bible) — "
                "substituting %s for a %s shot; fix the shotlist or add an alias "
                "to prompts/cast_bible.json",
                n, c["name"], shot_type or "unspecified",
            )
    picked = picked[:target] or [all_chars[0]]

    order = {c["id"]: i for i, c in enumerate(all_chars)}
    picked = sorted(picked, key=lambda c: order[c["id"]])

    # A name the cap dropped still has to go somewhere: point it at a child who
    # IS on screen (round-robin, deterministic) rather than leave it naming
    # nobody.
    for i, n in enumerate(dropped):
        mapping[str(n)] = picked[i % len(picked)]["name"]

    return picked, mapping


# ------------------------------------------------------------------ prose ---
def _join_and(items):
    """'a, b and c' — no Oxford comma, matching the channel's prompt style."""
    items = [str(i).strip() for i in items if str(i or "").strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _describe_dict(c):
    name = str(c.get("name") or "").strip()
    age = str(c.get("age") or "").strip()
    # Optional per-character body/anatomy (the bible's `build` field). It makes
    # the cast read as three individuals of slightly different sizes — Kofi the
    # tallest, Nala the littlest — instead of three identically-proportioned
    # clones, which is what "all the exact same size" looked like on screen.
    # Absent from a character -> this fragment vanishes and the description is
    # byte-identical to before, so an un-migrated bible is unaffected.
    build = str(c.get("build") or "").strip()
    skin = str(c.get("skin") or "").strip()
    hair = str(c.get("hair") or "").strip()
    garments = _join_and([c.get("top"), c.get("bottom"), c.get("shoes")])

    traits = ", ".join(t for t in (build, skin, hair) if t)
    lead = f"{name}, a {age} with {traits}" if age else f"{name} with {traits}"
    return f"{lead}, wearing {garments}" if garments else lead


def describe(name_or_id):
    """Canonical one-line appearance fragment for prompts. Always includes skin,
    hair and the top garment. No action, no setting, no trailing period."""
    c = character(name_or_id)
    if c is None:
        return ""
    return _describe_dict(c)


def _join_fragments(fragments):
    """'<a>; <b>; and <c>' (or just '<a>' for a single fragment)."""
    fragments = list(fragments)
    if len(fragments) <= 1:
        return fragments[0] if fragments else ""
    return "; ".join(fragments[:-1]) + "; and " + fragments[-1]


def _group_label(n, total, ensemble_label):
    if n == total:
        return ensemble_label
    word = _NUMBER_WORDS.get(n, str(n))
    noun = "toddler" if n == 1 else "toddlers"
    return f"{word} adorable Black {noun}"


def cast_sentence(names=None, shot_type=None):
    """The identity clause for a prompt.

    names=None or ['all'] -> the whole ensemble: '<ensemble_label>: <describe(a)>;
    <describe(b)>; and <describe(c)>'. A single name -> 'One adorable Black
    toddler: <describe(x)>'. Two -> 'Two adorable Black toddlers: <describe(a)>;
    and <describe(b)>'. Unknown names are ignored; if nothing resolves, falls
    back to the full ensemble.
    """
    bible = load_bible()
    resolved = _resolve_with_fallback(names, shot_type)
    total = len(bible["characters"])
    label = _group_label(len(resolved), total, bible.get("ensemble_label", "Adorable Black toddlers"))
    fragments = [_describe_dict(c) for c in resolved]
    return f"{label}: {_join_fragments(fragments)}"


# --------------------------------------------------------------- negatives ---
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
# The bible always phrases a top garment as "... a blue t-shirt", but
# `must_not` lists (and this channel's negative-prompt vocabulary generally)
# say "shirt" — normalize the hyphenated form so "blue shirt" lines up with
# "blue t-shirt" instead of silently never matching.
_TSHIRT_RE = re.compile(r"\bt-shirt\b")


def _normalize_for_match(text):
    text = str(text or "").lower()
    text = _TSHIRT_RE.sub("shirt", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _term_satisfied(term, normalized_positive_text):
    """True if `term` (e.g. 'blue shirt') is already true of the resolved
    characters' own positive appearance — i.e. its words appear as a
    contiguous, whole-word phrase in `normalized_positive_text`.

    Deliberately NOT a naive substring check of the raw term against the raw
    description: that both under-matches (the bible's "a blue t-shirt" never
    literally contains the substring "blue shirt") and would over-match if
    it fell back to checking each word independently anywhere in the text
    (e.g. Nala's "pink sneakers" + unrelated "yellow pinafore dress" would
    wrongly look like "pink dress" is satisfied). Requiring the term's words
    contiguous, in order, word-boundary-bounded, in the *normalized* text
    (t-shirt -> shirt, punctuation stripped) gets both right.
    """
    term_norm = _normalize_for_match(term)
    if not term_norm:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(w) for w in term_norm.split()) + r"\b"
    return re.search(pattern, normalized_positive_text) is not None


def negative_terms(names=None, shot_type=None):
    """Deduped, order-stable union of `must_not` for the given characters (all if
    None), MINUS any term already satisfied by what the resolved characters
    actually wear/look like (see `_term_satisfied`), plus the channel-wide
    floor (shirtless/undressed/jewellery/extra kids/...) which is never
    subtracted.

    Ensemble shots union every cast member's `must_not`, which includes each
    child's own wardrobe negated by the OTHERS (Zuri's list negates Kofi's
    blue shirt so a Kofi-only shot never drifts him into her yellow; Kofi's
    list negates Zuri's yellow shirt the same way). Left unfiltered, an
    `['all']` shot's negative prompt then contradicts its own positive prompt
    — negating the exact wardrobe just requested for whoever else is in
    frame. Subtracting whatever the resolved cast's own `describe()` text
    already confirms is on screen removes exactly that contradiction while
    leaving every OTHER term (a colour/look nobody in this shot actually has,
    like "purple dress" or "red hair") negated as before.
    """
    bible = load_bible()
    resolved = _resolve_with_fallback(names, shot_type)
    resolved_ids = {c["id"] for c in resolved}

    raw_terms = []
    seen = set()
    for c in resolved:
        for t in c.get("must_not") or []:
            t = str(t).strip()
            key = t.lower()
            if t and key not in seen:
                seen.add(key)
                raw_terms.append(t)

    # Cross-character identity guard: on a SUBSET shot (not the full ensemble),
    # negate the ABSENT cast members' distinctive hair/top/bottom so a solo or
    # pair frame cannot drift a child into a castmate who isn't in it. Measured
    # live (2026-07-23 storyfix render): a solo Kofi closeup rendered with
    # Zuri's afro puffs + denim dungarees — his `must_not` negated her yellow
    # SHIRT but nothing negated her hair or bottom, and the hand-authored
    # cross-negations only ever covered shirt colour. Deriving the exclusions
    # from the bible keeps them complete and self-maintaining as the cast grows.
    # The `_term_satisfied` subtraction below then drops any of these the
    # resolved cast legitimately shares, so an ['all'] shot — no absent members
    # — adds nothing and stays byte-identical.
    if len(resolved) < len(bible["characters"]):
        for c in bible["characters"]:
            if c["id"] in resolved_ids:
                continue
            for field in ("hair", "top", "bottom"):
                t = str(c.get(field) or "").strip()
                key = t.lower()
                if t and key not in seen:
                    seen.add(key)
                    raw_terms.append(t)

    positive_text = _normalize_for_match(
        " ".join(_describe_dict(c) for c in resolved)
    )
    terms = [t for t in raw_terms if not _term_satisfied(t, positive_text)]

    seen = {t.lower() for t in terms}
    for t in _CHANNEL_FLOOR:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            terms.append(t)

    return terms


# -------------------------------------------------------------------- seeds ---
def seed_for(names, base_seed, shot_type=None):
    """Deterministic per-character seed. For a SINGLE character this is exactly
    base_seed + that character's seed_offset, so the same child keeps the same
    seed family across shots and across episodes. Multi-character sets combine
    offsets by summing them (order-insensitive) and folding into a fixed range.
    Stable across processes — never uses Python's randomized hash()."""
    resolved = _resolve_with_fallback(names, shot_type)
    combined = sum(int(c["seed_offset"]) for c in resolved) % 100000
    return int(base_seed) + combined


def expected_child_count(names, shot_type=None):
    """['all'] or None -> len(characters); otherwise the number of RESOLVED cast
    names, min 1.

    Goes through the SAME shared resolver (`_resolve_with_fallback`) as
    `cast_sentence`, so the two can never disagree about who's on screen. Before
    this fix, `cast_sentence` fell back to the full ensemble when nothing
    resolved but this used `_resolve_exact` (no fallback, floored at 1) — a
    hallucinated/unknown name like `["Amara"]` produced
    `"A closeup shot of exactly one child: Zuri, ...; Kofi, ...; and Nala, ..."`
    (claims one child, describes three), and poisoned the review payload with
    `expected_children: 1` next to a three-child `cast_text`, hard-rejecting a
    correct render under the head-count rule.
    """
    resolved = _resolve_with_fallback(names, shot_type)
    return max(1, len(resolved))


# --------------------------------------------------------------- references ---
def reference_for(name_or_id):
    """The cast bible's recorded ``reference_image`` path for the character
    ``name_or_id`` resolves to, or ``None``.

    Routed through ``_resolve_with_fallback`` so the SAME alias/substitution
    logic the renderer uses applies here: a hallucinated name ("Amira", "Luna")
    resolves to the pinned cast member and returns THAT child's reference, never
    a mismatched one. Only reads the optional per-character ``reference_image``
    field from the bible (an added, ignored-by-the-validator key); the on-disk
    default location and existence checks live in ``kidsong.refs.reference_for``.
    Returns ``None`` when the character has no recorded path.
    """
    resolved = _resolve_with_fallback([name_or_id])
    if not resolved:
        return None
    ref = resolved[0].get("reference_image")
    ref = str(ref).strip() if ref is not None else ""
    return ref or None


# --------------------------------------------------------------- casting ---
def _cfg_kidsong(cfg):
    """`cfg['kidsong']` as a dict, or `{}` for anything else — the same
    defensive shape check `generate.py`'s keyframe-identity code uses
    (`cfg.get("kidsong", {}) or {}` guarded by `isinstance(cfg, dict)`), so a
    caller passing `None`, `{}`, or a config object that isn't a plain dict
    never raises here."""
    if not isinstance(cfg, dict):
        return {}
    ks = cfg.get("kidsong")
    return ks if isinstance(ks, dict) else {}


def protagonist_for(title, cfg=None):
    """The protagonist cast character id for a song titled `title`.

    (a) `kidsong.protagonist` in `cfg`, if set, is resolved through the same
    case-insensitive `character()` lookup used everywhere else in this module.
    An unresolvable value (typo, retired id, not a cast member at all) logs a
    WARNING and falls through to (b) rather than raising or casting nobody.

    (b) A deterministic pick keyed on the song title via `_stable_pick` — the
    SAME md5-not-hash() machinery `substitute_for` uses for invented character
    names, so the same title always casts the same protagonist across
    processes and runs (Python's hash() is salted per process and would break
    that — see `_stable_pick`'s docstring).

    Never raises: a garbage/empty/None `title` just stringifies (`_stable_pick`
    lower()s+strip()s it), and any other failure falls back to the first
    character in bible order.
    """
    try:
        chars = _all()
    except Exception:
        log.warning("cast: protagonist_for(%r) could not load the cast bible", title)
        return ""
    if not chars:
        return ""

    override = str(_cfg_kidsong(cfg).get("protagonist") or "").strip()
    if override:
        c = character(override)
        if c is not None:
            return c["id"]
        log.warning(
            "cast: kidsong.protagonist %r is not a cast bible character — "
            "falling back to a title-derived pick",
            override,
        )

    try:
        return _stable_pick(title, chars)["id"]
    except Exception:
        log.warning(
            "cast: protagonist_for(%r) failed to pick a protagonist — using "
            "the first bible character",
            title,
        )
        return chars[0]["id"]


def partner_for(protagonist_id, cfg=None):
    """The partner cast character id paired with `protagonist_id`.

    (a) `kidsong.partner` in `cfg`, if set, is resolved through the same
    `character()` lookup as `protagonist_for`'s override, with one extra
    check: a partner that resolves to the SAME character as `protagonist_id`
    is invalid (a protagonist can't costar with themselves) and falls through
    to (b), exactly like an unresolvable value.

    (b) The NEXT character after `protagonist_id` in bible order, wrapping
    around — Zuri's partner is Kofi, Kofi's is Nala, and Nala's (last in the
    bible) wraps back to Zuri.

    Never raises: an unresolvable/unknown `protagonist_id` falls back to the
    second character in bible order.
    """
    try:
        chars = _all()
    except Exception:
        log.warning("cast: partner_for(%r) could not load the cast bible", protagonist_id)
        return ""
    if not chars:
        return ""

    protagonist_key = str(protagonist_id or "").strip().lower()

    override = str(_cfg_kidsong(cfg).get("partner") or "").strip()
    if override:
        c = character(override)
        if c is not None and str(c["id"]).strip().lower() != protagonist_key:
            return c["id"]
        if c is None:
            log.warning(
                "cast: kidsong.partner %r is not a cast bible character — "
                "falling back to the next cast member",
                override,
            )
        else:
            log.warning(
                "cast: kidsong.partner %r is the same as the protagonist — "
                "falling back to the next cast member",
                override,
            )

    ids = [str(c["id"]).strip().lower() for c in chars]
    try:
        idx = ids.index(protagonist_key)
    except ValueError:
        return chars[1 % len(chars)]["id"]
    return chars[(idx + 1) % len(chars)]["id"]


# ------------------------------------------------------------------- misc ---
def names():
    """Canonical display names in bible order."""
    return [c["name"] for c in _all()]


def version():
    return str(load_bible().get("version") or "")


# --------------------------------------------------------------------- main ---
if __name__ == "__main__":
    bible = load_bible()
    print(f"Cast bible v{version()} — {bible['channel']}")
    for n in names():
        print(" ", describe(n))
    print()
    print("Ensemble:", cast_sentence())
    print("Kofi only:", cast_sentence(["Kofi"]))
    print("Kofi+Nala:", cast_sentence(["Kofi", "Nala"]))
    print("Negatives (Kofi):", negative_terms(["Kofi"]))
    print("Seed (Kofi, base=20260717):", seed_for(["Kofi"], 20260717))
    print("Seed (all, base=20260717):", seed_for(["all"], 20260717))
