"""Auto idea generation + dedupe.

The LLM gets the channel niche and a list of recent topics to avoid, and must
return {"ideas": [{"topic", "video_type", "hook"}, ...]} (object form because
Groq's json_object mode rejects bare arrays).
"""
import difflib
import hashlib
import json
import re

from studio import models

FUZZY_THRESHOLD = 0.85

IDEA_SYSTEM = (
    "You generate video ideas for short vertical videos (YouTube Shorts). "
    "You ALWAYS reply with a single valid JSON object and nothing else."
)

IDEA_PROMPT = """Generate {k} fresh video ideas for this YouTube Shorts channel.

Channel niche: {niche}
Channel description: {description}
Default video type: {video_type} (allowed types: {allowed_types})

Avoid anything close to these already-used topics:
{avoid}

Reply as JSON: {{"ideas": [{{"topic": "...", "video_type": "...", "hook": "..."}}]}}
"topic" is a one-line video premise. "video_type" must be one of the allowed types.
"""

# --------------------------------------------------------------- kidsong ----
# The brainrot prompt above asks for viral hooks; a children's-song channel needs
# the opposite. Same JSON contract so the parsing/dedupe path stays shared.
KIDSONG_TYPES = ("kidsong", "kids")

KIDSONG_IDEA_SYSTEM = (
    "You invent song subjects for a wholesome preschool YouTube channel. Every "
    "subject must be original, gentle and appropriate for toddlers. You never copy "
    "or reference existing shows, brands, characters or songs. You ALWAYS reply "
    "with a single valid JSON object and nothing else."
)

KIDSONG_IDEA_PROMPT = """Generate {k} fresh sing-along song subjects for this preschool channel.

Channel niche: {niche}
Channel description: {description}

Avoid anything close to these already-used subjects:
{avoid}

Rules:
- Each "topic" is a simple everyday subject a nursery rhyme can be built around,
  written as a short noun phrase in English (e.g. "brushing teeth",
  "colors of the rainbow", "counting to ten", "sharing toys", "getting dressed").
- Toddler-appropriate only: daily routines, feelings, animals, numbers, letters,
  colors, shapes, weather, healthy habits, kindness, family, playtime.
- Original subjects only. Never name or hint at a real brand, show, character or
  existing song. Nothing scary, sad, violent, commercial or educational-beyond-preschool.

Reply as JSON: {{"ideas": [{{"topic": "...", "video_type": "{video_type}", "hook": "..."}}]}}
"""

# Mirrors pipeline.kidsong.lyrics.FALLBACK_SONG: when the LLM is unreachable the
# cascade must still produce safe, original, child-appropriate work.
KIDSONG_FALLBACK_TOPICS = [
    "brushing teeth in the morning",
    "counting to ten with fingers",
    "colors of the rainbow",
    "sharing toys with friends",
    "washing hands before we eat",
    "getting dressed all by myself",
    "clean up time in the playroom",
    "the wheels on the little red wagon",
    "saying please and thank you",
    "hopping like a bunny",
    "bedtime and goodnight kisses",
    "eating fruit for a healthy snack",
    "shapes we see all around us",
    "rainy day puddle jumping",
    "my happy feelings song",
    "farm animal sounds",
    "putting on shoes and socks",
    "the days of the week",
    "big hugs for my family",
    "tidy toys and quiet time",
]

# ------------------------------------------------------------ hyperframes ----
# One explainer concept per video (hook -> core idea -> 2-3 steps -> memorable
# takeaway). Ideas must be self-contained concepts, not branded IP and not
# tied to breaking news, which goes stale within days of a channel's queue.
HYPERFRAMES_IDEA_SYSTEM = (
    "You generate concepts for short vertical explainer videos (YouTube Shorts). "
    "Each concept is a single, self-contained idea a viewer can learn in under a "
    "minute: hook, core idea, 2-3 concrete steps, one memorable takeaway line. "
    "Never propose a branded product, franchise, company, or celebrity as the "
    "concept, and never propose breaking or dated news (it goes stale in days). "
    "You ALWAYS reply with a single valid JSON object and nothing else."
)

HYPERFRAMES_IDEA_PROMPT = """Generate {k} fresh explainer-video concepts for this YouTube Shorts channel.

Channel niche: {niche}
Channel description: {description}

Avoid anything close to these already-used concepts:
{avoid}

Rules:
- Each "topic" is a single explainable concept, phrased concretely and
  self-contained (e.g. "why the sky looks blue", "how compound interest
  snowballs", "the placebo effect explained", "why we yawn when others yawn").
- The concept must stand on its own: no branded products, companies,
  franchises, celebrities or public figures, and no news events or anything
  that depends on "right now" to make sense.
- Aim for concepts explainable in a Hook -> Core idea -> 2-3 steps -> memorable
  takeaway structure inside 60 seconds.

Reply as JSON: {{"ideas": [{{"topic": "...", "video_type": "{video_type}", "hook": "..."}}]}}
"""

# ----------------------------------------------------------------- brainrot --
# This is the "brainrot" VIDEO TYPE (a content format: hooky, meme-driven
# shorts across varied topics) — NOT the "brainrot" STYLE from
# `studio/views/__init__.py` STYLES / `studio/svm.py`, which is a
# look-and-sound preset any video_type can wear. Same word, two different
# columns (jobs.video_type vs. jobs.style/channels.style); `is_kidsong_type`
# and `is_kids_style` each read only their own column, so a video_type of
# "brainrot" can never be mistaken for the "brainrot" style or vice versa.
# Ideas here must stay hooky AND monetizable: YouTube demonetizes/limits reach
# on harassment, hate, sexualized content, real identifiable private people,
# and unlicensed use of protected characters/brands.
BRAINROT_IDEA_SYSTEM = (
    "You generate hooky, meme-driven video ideas for short vertical YouTube "
    "Shorts covering wide-open, varied topics. Ideas must stay YouTube-"
    "monetizable: no insults, harassment or hate targeting any person or "
    "group, no sexualized content, no real identifiable private individuals, "
    "and no copyrighted or trademarked characters, shows, games or brands. "
    "You ALWAYS reply with a single valid JSON object and nothing else."
)

BRAINROT_IDEA_PROMPT = """Generate {k} fresh hooky meme-format video ideas for this YouTube Shorts channel.

Channel niche: {niche}
Channel description: {description}

Avoid anything close to these already-used topics:
{avoid}

Rules:
- Each "topic" is a punchy, scroll-stopping premise for a short, addictive video.
- Wide-open subject matter is welcome, but stay YouTube-monetizable: no
  insults, harassment or hate targeting any person or group; no sexualized
  content; no real identifiable private individuals; no copyrighted or
  trademarked characters, shows, games or brands.
- Original, ownable premises only.

Reply as JSON: {{"ideas": [{{"topic": "...", "video_type": "{video_type}", "hook": "..."}}]}}
"""

# Video types that, like kidsong, are a single self-contained format rather
# than a member of the generic multi-type pool: the idea LLM gets a dedicated
# system/prompt pair and the channel is only ever offered its own type.
SOLO_IDEA_PROMPTS = {
    "hyperframes": (HYPERFRAMES_IDEA_SYSTEM, HYPERFRAMES_IDEA_PROMPT),
    "brainrot": (BRAINROT_IDEA_SYSTEM, BRAINROT_IDEA_PROMPT),
}

DEFAULT_ALLOWED_TYPES = ["reddit", "conversation", "facts", "scary"]


def normalize_topic(topic):
    t = re.sub(r"[^\w\s]", "", (topic or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def dedupe_hash(topic):
    return hashlib.sha1(normalize_topic(topic).encode("utf-8")).hexdigest()


def is_duplicate(topic, existing_topics):
    """Exact-normalized or fuzzy (>0.85) match against a list of topics."""
    norm = normalize_topic(topic)
    if not norm:
        return True
    for other in existing_topics:
        other_norm = normalize_topic(other)
        if norm == other_norm:
            return True
        if difflib.SequenceMatcher(None, norm, other_norm).ratio() > FUZZY_THRESHOLD:
            return True
    return False


def is_kidsong_type(video_type):
    return (video_type or "") in KIDSONG_TYPES


# A style preset, unlike a video type, says how the video LOOKS AND SOUNDS.
# `studio/svm.py` reads "kids" and picks a kids voice, happy music at high
# volume and a pink caption background; `studio/views/api.py` forces it on for
# the kids video type. So a "kids"-styled video is child-directed output
# whatever its video_type column says — which is why the COPPA declaration
# consults this too (see `scheduler._made_for_kids`).
KIDS_STYLES = ("kids", "kidsong")


def is_kids_style(style):
    return (style or "").strip().lower() in KIDS_STYLES


# ------------------------------------------------------- kidsong safety gate --
# Why this exists: the real channel KinderKrippenLernLieder used to be a 'facts'
# channel and still had brainrot ideas sitting pending ("Germany's scariest
# folkloric creatures come to life"). Repointing default_video_type to kidsong
# made those ideas eligible and the cascade queued them as toddler sing-alongs.
#
# Design: ALLOW-LEANING. A topic must (a) hit no deny term and (b) hit at least
# one term from the toddler subject space. A pure blocklist only ever catches
# the weird topic you already thought of; the next one walks straight through.
# Toddler subject matter really is a small, enumerable space, so requiring a
# positive match is affordable here in a way it would not be for a facts channel.
#
# Matching is on normalized word tokens with prefix ("stem") semantics, so
# "counting"/"counts" both hit "count" while "scratching" does NOT hit "scare".

KIDSONG_ALLOW_TERMS = (
    # routines & self-care
    "wash", "brush", "teeth", "tooth", "bath", "bathtime", "soap", "bubble",
    "dress", "shoe", "sock", "shirt", "hat", "coat", "mitten", "scarf", "boot",
    "potty", "diaper", "nap", "sleep", "bed", "bedtime", "goodnight", "wake",
    "morning", "evening", "night", "day", "week", "clean", "tidy", "help",
    "eat", "food", "snack", "breakfast", "lunch", "dinner", "fruit", "veg",
    "apple", "banana", "carrot", "berry", "milk", "water", "juice", "bread",
    "spoon", "cup", "plate", "seed", "healthy", "hungry", "thirsty", "drink",
    # people & feelings
    "mama", "mommy", "mom", "papa", "daddy", "dad", "family", "baby", "toddler",
    "brother", "sister", "grandma", "grandpa", "friend", "neighbor", "teacher",
    "hug", "kiss", "cuddle", "love", "kind", "kindness", "shar", "care", "gentle",
    "happy", "smile", "laugh", "giggle", "sad", "calm", "quiet", "brave", "proud",
    "feeling", "please", "thank", "sorry", "hello", "goodbye", "welcome",
    # animals
    "animal", "pet", "cat", "kitten", "dog", "puppy", "duck", "duckling", "cow",
    "sheep", "lamb", "pig", "horse", "pony", "goat", "hen", "chick", "rooster",
    "bird", "birdie", "owl", "fish", "frog", "bunny", "rabbit", "mouse", "bee",
    "butterfly", "ladybug", "snail", "turtle", "elephant", "lion", "monkey",
    "bear", "farm", "barn", "nest", "paw", "tail", "wing",
    # numbers, letters, colors, shapes
    "count", "number", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "first", "letter", "alphabet", "abc", "spell",
    "color", "colour", "rainbow", "red", "blue", "green", "yellow", "orange",
    "purple", "pink", "brown", "black", "white", "shape", "circle", "square",
    "triangle", "star", "heart",
    # weather, nature, world
    "weather", "rain", "rainy", "sun", "sunny", "sunset", "sunrise", "sunflower",
    "cloud", "sky", "wind", "windy", "snow", "snowy", "puddle", "moon", "night",
    "tree", "leaf", "leaves", "flower", "garden", "grass", "beach", "sand",
    "sea", "river", "season", "spring", "summer", "autumn", "fall", "winter",
    # play & movement
    "play", "playroom", "playground", "toy", "block", "ball", "doll", "teddy",
    "puzzle", "game", "draw", "paint", "color", "sing", "song", "sings", "dance",
    "music", "clap", "stomp", "wiggle", "jump", "hop", "skip", "run", "walk",
    "swing", "slide", "ride", "wagon", "wheel", "bus", "train", "car", "boat",
    "bike", "truck", "tractor", "park", "story", "book", "read", "read",
    "hand", "finger", "toe", "foot", "feet", "head", "nose", "ear", "eye",
    "little", "big", "small", "tiny", "home", "house", "room", "door", "window",
)

# Hard vetoes. Everything the incident produced is in here, plus the obvious
# neighbours. A deny hit rejects even if an allow term also matched, so
# "People SHARE the weirdest things..." cannot ride in on "share".
KIDSONG_DENY_TERMS = (
    # frightening / dark folklore — the exact failure mode
    "scary", "scarie", "scare", "scared", "creepy", "spooky", "spook",
    "creature", "monster", "beast", "ghost", "ghoul", "goblin", "troll",
    "witch", "demon", "devil", "satan", "zombie", "vampire", "werewolf",
    "haunt", "horror", "terrif", "terror", "frighten", "fright", "nightmare",
    "folklor", "myth", "legend", "curse", "occult", "paranormal", "supernatural",
    "grave", "cemetery", "skeleton", "skull", "corpse", "hell", "evil",
    "dark", "shadow", "creep", "lurk", "eerie", "sinister", "ominous",
    # harm / violence / death
    "death", "dead", "dying", "kill", "murder", "blood", "gore", "weapon",
    "gun", "knife", "warzone", "warrior", "battle", "violen", "fight", "injur",
    "danger", "disaster", "crash", "poison", "disease", "sick", "hospital",
    # adult / meta / commentary — the other half of the incident
    "interview", "confession", "reddit", "aita", "gossip", "rant", "drama",
    "controvers", "conspiracy", "politic", "religio", "sex", "sexy", "adult",
    "alcohol", "beer", "wine", "drunk", "drug", "smok", "cigarett", "gambl",
    "crime", "criminal", "prison", "police", "steal", "stole", "thief",
    "weird", "weirdest", "bizarre", "obscure", "strange", "shocking", "shock",
    "disturb", "cring", "trivia", "fact", "facts", "brainrot", "viral",
    "culture", "cultural", "stereotype", "tradition", "festival", "ritual",
    "translat", "pronounc", "vocabulary", "grammar", "dialect", "slang",
    "adults", "teenager", "grownup", "grown",
)

# Phrases that are only unsafe as phrases (the individual words are innocent).
KIDSONG_DENY_PHRASES = (
    "people share",
    "people reveal",
    "you thought",
    "they thought",
    "did you know",
    "growing up",
    "will leave you",
    "scratching your head",
    "you wont believe",
    "things that",
)


def _idea_field(idea, name):
    """Read a column from a sqlite3.Row or a plain dict."""
    try:
        return idea[name]
    except (IndexError, KeyError, TypeError):
        return None


def _first_stem_hit(tokens, terms):
    """First `terms` entry that prefixes one of `tokens`, else None."""
    for token in tokens:
        for term in terms:
            if token.startswith(term):
                return term
    return None


def kidsong_topic_rejection(topic):
    """None if `topic` is a safe toddler sing-along subject, else a reason str.

    An empty/missing topic is allowed on purpose: the kidsong pipeline then
    invents its own subject from the (already child-safe) lyrics prompt, so
    there is nothing unsafe to gate.
    """
    normalized = normalize_topic(topic)
    if not normalized:
        return None

    for phrase in KIDSONG_DENY_PHRASES:
        if phrase in normalized:
            return "denied phrase %r" % phrase

    tokens = normalized.split()
    denied = _first_stem_hit(tokens, KIDSONG_DENY_TERMS)
    if denied:
        return "denied term %r" % denied

    if not _first_stem_hit(tokens, KIDSONG_ALLOW_TERMS):
        return "no toddler subject matched"
    return None


def is_kidsong_topic_safe(topic):
    return kidsong_topic_rejection(topic) is None


def kidsong_idea_rejection(idea):
    """None if this idea row may become a kidsong job, else a reason str.

    Two independent signals, both of which must pass:
      * the idea's own video_type — an idea written by the facts prompt is not
        a song subject no matter how harmless its wording looks;
      * the topic text itself — covers ideas with no video_type, hand-typed
        ideas, and anything the LLM mislabels.
    """
    if idea is None:
        return "no idea"
    video_type = _idea_field(idea, "video_type")
    if video_type and not is_kidsong_type(video_type):
        return "idea video_type %r is not a kidsong type" % video_type
    return kidsong_topic_rejection(_idea_field(idea, "topic"))


def _insert_auto_ideas(channel, candidates, allowed, seen):
    """Dedupe candidates against `seen` + insert survivors as source='auto'.

    `seen` is extended in place so callers can chain several batches.
    Returns the number of ideas inserted.
    """
    inserted = 0
    for cand in candidates:
        if isinstance(cand, str):
            cand = {"topic": cand}
        topic = (cand.get("topic") or "").strip()
        if not topic or is_duplicate(topic, seen):
            continue
        vt = cand.get("video_type")
        if vt not in allowed:
            vt = channel.get("default_video_type") or allowed[0]
        idea_id = models.add_idea(
            channel["id"], topic, source="auto", video_type=vt,
            priority=0, dedupe_hash=dedupe_hash(topic),
        )
        if idea_id:
            inserted += 1
            seen.append(topic)
    return inserted


def generate_ideas(channel, cfg, k=5, allowed_types=None, video_type=None):
    """Ask the LLM for k ideas, dedupe them, insert survivors as source='auto'.

    The prompt is video-type aware: a kidsong channel gets children's-song
    subjects, hyperframes/brainrot each get their own dedicated prompt (see
    SOLO_IDEA_PROMPTS), and everything else gets the generic multi-type prompt.

    Returns the number of ideas inserted.
    """
    from pipeline.script_gen import complete, _extract_json

    vt_default = video_type or channel.get("default_video_type") or "facts"
    kidsong = is_kidsong_type(vt_default)
    solo_spec = SOLO_IDEA_PROMPTS.get(vt_default)
    allowed = allowed_types or (
        [vt_default] if (kidsong or solo_spec) else list(DEFAULT_ALLOWED_TYPES)
    )

    recent = models.recent_idea_topics(channel["id"], limit=50)
    avoid = "\n".join(f"- {t}" for t in recent) or "- (none yet)"

    if kidsong:
        prompt = KIDSONG_IDEA_PROMPT.format(
            k=k,
            niche=channel.get("niche") or channel.get("name") or "preschool sing-along songs",
            description=channel.get("description") or "-",
            video_type=vt_default,
            avoid=avoid,
        )
        system = KIDSONG_IDEA_SYSTEM
    elif solo_spec:
        system, template = solo_spec
        prompt = template.format(
            k=k,
            niche=channel.get("niche") or channel.get("name") or "general content",
            description=channel.get("description") or "-",
            video_type=vt_default,
            avoid=avoid,
        )
    else:
        prompt = IDEA_PROMPT.format(
            k=k,
            niche=channel.get("niche") or channel.get("name") or "general viral content",
            description=channel.get("description") or "-",
            video_type=vt_default,
            allowed_types=", ".join(allowed),
            avoid=avoid,
        )
        system = IDEA_SYSTEM

    raw = complete(prompt, cfg, system=system)
    data = _extract_json(raw)
    candidates = data.get("ideas") or []

    return _insert_auto_ideas(channel, candidates, allowed, list(recent))


def add_fallback_kidsong_ideas(channel, k, video_type="kidsong"):
    """Insert up to k built-in kidsong topics (LLM unreachable / came up short).

    Same dedupe path as the LLM ideas, so re-running never piles up duplicates.
    Returns the number of ideas inserted.
    """
    if k <= 0:
        return 0
    seen = list(models.recent_idea_topics(channel["id"], limit=200))
    inserted = 0
    for topic in KIDSONG_FALLBACK_TOPICS:
        if inserted >= k:
            break
        inserted += _insert_auto_ideas(
            channel, [{"topic": topic, "video_type": video_type}], [video_type], seen
        )
    return inserted


def add_user_idea(channel_id, topic, video_type=None, priority=10):
    """Insert a user idea (always wins over auto ideas). Returns id or None if dupe."""
    topic = (topic or "").strip()
    if not topic:
        return None
    return models.add_idea(
        channel_id, topic, source="user", video_type=video_type,
        priority=priority, dedupe_hash=dedupe_hash(topic),
    )
