"""Say, on every published video, that a machine made it.

Nothing in this repo did. Measured before this module existed: the body sent to
`videos.insert` carried `privacyStatus` and `selfDeclaredMadeForKids` and
nothing else; the description was passed through untouched except for the
`#Shorts` tag; the rendered MP4 carried no tags at all. A grep of the whole
tree for any disclosure vocabulary returned sixteen incidental hits and zero
disclosures.

Two separate obligations, which are easy to conflate:

  * **AI Act Art. 50(2)** — machine-readable marking of synthetic output — binds
    the *provider* of the generative system (the people who ship LTX-2,
    ACE-Step, Gemma), not an operator rendering with it. Systems already on the
    market before 2026-08-02 additionally have until 2026-12-02.
  * **AI Act Art. 50(4)** — disclosure — binds the *deployer*, i.e. whoever runs
    this repo. It is scoped to "deep fakes": content resembling real persons,
    places or events that would falsely appear authentic (Art. 3(60)). Per the
    Commission's final Art. 50 guidelines a realistic depiction of a *fictitious
    but natural-looking* person counts, while plainly unrealistic imagery does
    not. A stylised toon of invented toddlers most likely sits outside that, and
    the exemption for "evidently artistic, creative or fictional" work would in
    any case reduce the duty to a disclosure that does not spoil the work — so
    no burnt-in overlay, which is why this module does not add one.

The obligation is therefore probably not binding for the shipped kids channel,
and this module still exists, because:

  * the boundary is a judgement call about 3D-rendered children, not a fact;
  * `pipeline/generate.py` produces stock-footage videos on the same uploader,
    where the reading is entirely different;
  * YouTube's own altered-or-synthetic rule applies regardless of the AI Act,
    and its penalty is removal or loss of monetisation;
  * and the whole thing is one config flag plus one API field.

`youtube_synthetic_media: "auto"` therefore resolves to **False for the video
types listed as animated** and True for everything else — because YouTube
explicitly does not want the altered-content label on wholly animated work, and
a disclosure that is *wrong* is its own defect. Setting the key to a literal
`true`/`false` overrides the inference.

Nothing here can ever REMOVE a disclosure: `declare_synthetic_media` takes an
opt-in override that can only turn the flag on, the same asymmetry
`studio.scheduler._made_for_kids` uses for the COPPA declaration, and for the
same reason — the failure that matters is the one where a declaration silently
goes missing.
"""
import os
import subprocess

# Sent as-is if the operator has not configured anything. German first: the
# deployer is in the EU, which is whose rules create the duty.
DEFAULT_NOTE = (
    "Dieses Video wurde mit Hilfe von künstlicher Intelligenz erstellt. "
    "/ This video was created using artificial intelligence."
)

_DEFAULTS = {
    "enabled": True,
    "description_note": DEFAULT_NOTE,
    "youtube_synthetic_media": "auto",
    "embed_metadata": True,
    # Wholly animated output. YouTube's altered/synthetic disclosure explicitly
    # does not cover it, so "auto" leaves the flag off for these.
    "animated_video_types": ["kidsong"],
}


def settings(cfg):
    """The `ai_disclosure` block with defaults filled in.

    A missing block means "defaults", not "off" — an operator upgrading from an
    older config.json must not silently end up publishing undisclosed.
    """
    raw = (cfg or {}).get("ai_disclosure")
    merged = dict(_DEFAULTS)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in merged:
                merged[key] = value
    return merged


def is_enabled(cfg):
    return bool(settings(cfg)["enabled"])


def note(cfg):
    """The disclosure sentence, or "" when disclosure is switched off."""
    conf = settings(cfg)
    if not conf["enabled"]:
        return ""
    text = conf["description_note"]
    if not isinstance(text, str):
        return DEFAULT_NOTE
    return " ".join(text.split())


def has_note(description, cfg):
    """Whether `description` already carries this config's disclosure.

    Compared on collapsed whitespace so a note that was wrapped across lines by
    an editor still counts as present.
    """
    wanted = note(cfg)
    if not wanted:
        return False
    return wanted.lower() in " ".join((description or "").split()).lower()


def append_note(description, cfg, limit=None):
    """Return `description` with the disclosure appended, at most once.

    `limit` is the platform's description cap. It is applied to the OPERATOR's
    text, never to the disclosure: appending first and truncating afterwards
    would let a long description push the disclosure off the end — silently
    producing exactly the undisclosed upload this module exists to prevent.
    """
    text = description or ""
    if not is_enabled(cfg):
        return text[:limit] if limit else text
    suffix = note(cfg)
    if not suffix:
        return text[:limit] if limit else text
    if has_note(text, cfg):
        return text[:limit] if limit else text

    joiner = "\n\n" if text.strip() else ""
    if limit is not None:
        room = limit - len(suffix) - len(joiner)
        if room < 0:
            # Pathological cap: the disclosure alone wins over operator prose.
            return suffix[:limit]
        text = text[:room].rstrip()
        joiner = "\n\n" if text.strip() else ""
    return f"{text.rstrip()}{joiner}{suffix}"


def declare_synthetic_media(cfg, video_type=None, override=None):
    """Whether to set YouTube's `status.containsSyntheticMedia` for this upload.

    `override` is opt-IN only: True forces the declaration on, False and None
    both defer to config. A caller can add a disclosure, never drop one.
    """
    if override is True:
        return True
    conf = settings(cfg)
    if not conf["enabled"]:
        return False

    mode = conf["youtube_synthetic_media"]
    if isinstance(mode, bool):
        return mode
    if isinstance(mode, str) and mode.strip().lower() in ("true", "yes", "on"):
        return True
    if isinstance(mode, str) and mode.strip().lower() in ("false", "no", "off"):
        return False

    # "auto", or anything unrecognised: declare unless this is animated output.
    animated = conf["animated_video_types"] or []
    if not isinstance(animated, (list, tuple)):
        animated = [animated]
    vt = str(video_type or "").strip().lower()
    return not any(vt == str(a).strip().lower() for a in animated if a)


def metadata_tags(cfg):
    """The tags written into the container, or {} when disclosure is off."""
    conf = settings(cfg)
    if not conf["enabled"] or not conf["embed_metadata"]:
        return {}
    text = note(cfg)
    if not text:
        return {}
    return {"comment": text, "description": text}


def tag_file(path, cfg, log=None):
    """Write the disclosure into an existing MP4's container metadata.

    Stream copy, so this costs a remux rather than a re-encode, and it goes
    through a `.part` sibling + `os.replace` for the same reason every other
    writer in this repo does (see `pipeline/atomicio`): a killed ffmpeg must not
    leave a truncated file where a finished episode belongs.

    Best effort by design — a missing ffmpeg must not fail a render that has
    already cost an hour of GPU. Returns True only if the file really was
    rewritten.
    """
    tags = metadata_tags(cfg)
    if not tags or not path or not os.path.exists(path):
        return False

    tmp = f"{path}.disclosure.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", path, "-map", "0", "-c", "copy"]
    for key, value in tags.items():
        cmd += ["-metadata", f"{key}={value}"]
    cmd += ["-movflags", "+faststart", tmp]

    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp, path)
        return True
    except Exception as exc:  # pragma: no cover - graceful degradation
        if log:
            log(f"AI-disclosure metadata skipped ({exc})")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
