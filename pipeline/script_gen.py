"""
script_gen.py — Generate a brainrot video script with a free LLM.

Backends (set in config.json -> llm.backend):
  - groq   : free API key from https://console.groq.com (fast, recommended)
  - ollama : fully local, no account, runs on your GPU (https://ollama.com)
  - gemini : free API key from https://aistudio.google.com

The model is asked to return strict JSON shaped like:
  {
    "title": "...",            # YouTube title
    "description": "...",      # YouTube description
    "tags": ["...", ...],
    "lines": [
      {"speaker": "narrator", "text": "..."},
      {"speaker": "speaker_a", "text": "..."},
      ...
    ]
  }
Valid speaker values: "narrator", "speaker_a", "speaker_b".

Script LANGUAGE: generate_script() writes English scripts unless told
otherwise, because every prompts/*.txt template is written in English. The
target language is resolved from config as `(cfg.get("kidsong") or
{}).get("language", "en")` — yes, the "kidsong" key, even for non-kidsong
video types like this module's facts/hyperframes/brainrot. That key is the
existing, already-general "content language" pin: studio/profiles.py's
apply_profile() writes the channel's resolved language onto exactly that key
for every video type a channel can run (not just literal kidsong jobs), and
pipeline/captions.py threads it onward into whisper.language purely for
Whisper's STT decode. whisper.language is therefore a *derived*,
transcription-only setting one layer downstream — never the source of truth
for what language a script should be written in. Reusing kidsong.language
here (instead of inventing a third key) means a German channel already
configured today needs no new config to also get a German brainrot/facts/
hyperframes script. Callers may still override with the explicit `language`
kwarg on generate_script(). When the resolved language is English (the
default) the built prompt is byte-identical to before this feature existed —
no existing channel's output changes.
"""
import json
import os
import re

VALID_SPEAKERS = {"narrator", "speaker_a", "speaker_b"}

# Human-readable names for the language instruction appended to non-English
# prompts. Not exhaustive — an unlisted-but-valid code still gets an
# instruction, just using the raw code instead of a friendly name (see
# _language_instruction). "en" never needs an entry: it's the default and
# gets no instruction at all (byte-identical prompt guarantee).
_LANGUAGE_NAMES = {
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
}


def _normalize_target_language(lang):
    """Normalize a script-language config/param value to a lowercase subtag.

    Deliberately stricter about *type* than studio.models._normalize_language:
    only a non-empty string is accepted as a real language tag. Anything else
    -- a number, list, dict, or a string that's blank/whitespace after
    stripping -- is a malformed value, not a language, and must never raise
    (a broken config line or a bad override must not be able to kill a
    scripting job); it silently falls back to "en", same as unset.
    """
    if not isinstance(lang, str):
        return "en"
    lang = lang.strip().lower()
    if not lang:
        return "en"
    return lang.split("-")[0]


def _resolve_language(cfg, language=None):
    """Target script language, in precedence order: the explicit `language`
    kwarg, then top-level ``cfg["language"]``, then ``cfg["kidsong"]["language"]``,
    then "en". Always returns a normalized, safe subtag -- never raises.

    The top-level key is the one that MEANS this: a brainrot short or an
    explainer has no business reading its language out of the kidsong block.
    `kidsong.language` stays in the chain as a fallback, and deliberately so —
    it was the only language key this repo had, every channel profile still
    pins it, and dropping it would silently switch existing German channels
    back to English. New config should set the top-level key; the old one keeps
    working.
    """
    if language is None:
        cfg = cfg or {}
        language = cfg.get("language") or (cfg.get("kidsong") or {}).get("language")
    return _normalize_target_language(language)


def _language_instruction(language):
    """A short, unambiguous instruction to append to the prompt so the LLM
    writes the whole script in `language` instead of English. Returns "" for
    English (or anything that normalized to it) so the byte-identical
    guarantee holds for every channel that doesn't opt in."""
    if language == "en":
        return ""
    name = _LANGUAGE_NAMES.get(language, language)
    return (
        f"\n\nWrite the ENTIRE response in {name} ({language}): the title, "
        f"description, tags, and every script line. Keep the JSON keys "
        f'("title", "description", "tags", "lines", "speaker", "text", '
        f'"visual", "graphic", "camera") in English -- only the VALUES are '
        f"in {name}."
    )


def _normalize_graphic(g):
    """Validate an optional per-line motion-graphic spec from the LLM.

    Returns a clean dict ({"type": ...}) the motion engine understands, or None
    if the spec is missing/malformed. Keeps the pipeline robust to whatever the
    model emits.
    """
    if not isinstance(g, dict):
        return None
    gtype = str(g.get("type", "")).strip().lower()
    if gtype == "stat":
        value = g.get("value")
        if value in (None, ""):
            return None
        out = {"type": "stat", "value": str(value)}
        if g.get("label"):
            out["label"] = str(g["label"]).strip()
        return out
    if gtype == "card":
        text = str(g.get("text", "")).strip()
        return {"type": "card", "text": text} if text else None
    if gtype == "compare":
        try:
            return {
                "type": "compare",
                "a_label": str(g["a_label"]).strip(),
                "a_value": g["a_value"],
                "b_label": str(g["b_label"]).strip(),
                "b_value": g["b_value"],
            }
        except (KeyError, TypeError):
            return None
    return None


_CAMERA_MOVES = {"push_in", "pull_out", "drift_pan", "ken_burns", "punch_in", "handheld", "static"}
_CAMERA_SYNONYMS = {"zoom_in": "push_in", "zoom_out": "pull_out", "shake": "handheld", "pan": "drift_pan"}


def _normalize_camera(c):
    """Validate an optional per-line camera hint from the LLM.

    Returns a canonical move name the camera engine understands, or None.
    """
    if not isinstance(c, str):
        return None
    c = c.strip().lower().replace("-", "_").replace(" ", "_")
    c = _CAMERA_SYNONYMS.get(c, c)
    return c if c in _CAMERA_MOVES else None


def _load_prompt(video_type, root):
    path = os.path.join(root, "prompts", f"{video_type}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No prompt template for video type '{video_type}'. "
            f"Expected file: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json(text):
    """LLMs sometimes wrap JSON in prose or ```json fences. Pull out the object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # Grab the outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM did not return JSON. Got:\n{text[:500]}")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------- backends ----
def _call_groq(system, user, model):
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it (see README), or switch llm.backend to 'ollama' in config.json."
        )
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.9,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _call_ollama(system, user, model, host):
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


def _call_gemini(system, user, model):
    import google.generativeai as genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Get one at https://aistudio.google.com")
    genai.configure(api_key=key)
    gm = genai.GenerativeModel(model, system_instruction=system)
    resp = gm.generate_content(
        user,
        generation_config={"temperature": 0.9, "response_mime_type": "application/json"},
    )
    return resp.text


# ----------------------------------------------------------------- public -----
SYSTEM_PROMPT = (
    "You are a scriptwriter for short vertical 'brainrot' videos (YouTube Shorts / "
    "TikTok style). You write punchy, fast, addictive scripts that hook in the first "
    "two seconds. You ALWAYS reply with a single valid JSON object and nothing else."
)


def complete(user, cfg, system=None):
    """Run one completion against the configured backend and return raw text.

    Shared by script generation and the studio idea generator.
    """
    system = system or SYSTEM_PROMPT
    llm = cfg["llm"]
    backend = llm["backend"]
    if backend == "groq":
        return _call_groq(system, user, llm["groq_model"])
    if backend == "ollama":
        return _call_ollama(system, user, llm["ollama_model"], llm["ollama_host"])
    if backend == "gemini":
        return _call_gemini(system, user, llm["gemini_model"])
    raise ValueError(f"Unknown llm.backend: {backend}")


def generate_script(video_type, topic=None, cfg=None, extra_context=None, language=None):
    """Generate one script. `language` overrides the config-resolved target
    language (see module docstring); leave unset to use
    cfg["kidsong"]["language"] / default to English."""
    if cfg is None:
        from pipeline.config import load_config

        cfg = load_config()
    root = cfg["_root"]
    template = _load_prompt(video_type, root)

    user = template.replace("{{TOPIC}}", topic or "pick a wildly engaging topic yourself")
    if extra_context:
        user += f"\n\nChannel context (match this niche and audience): {extra_context}"
    user += _language_instruction(_resolve_language(cfg, language))

    raw = complete(user, cfg)

    data = _extract_json(raw)
    return _normalize_script(data)


def _normalize_script(data):
    """Validate and coerce a raw LLM script dict into the shape the rest of the
    pipeline relies on. LLMs are inconsistent, so never trust the types: `tags`
    may come back as a comma-separated string or null, and `title`/`description`
    as numbers — all of which break downstream slicing in youtube_upload."""
    if not isinstance(data, dict):
        raise ValueError("Script must be a JSON object.")

    # ---- lines ----
    lines = data.get("lines") or []
    clean = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        spk = str(ln.get("speaker") or "narrator").strip().lower()
        if spk not in VALID_SPEAKERS:
            spk = "narrator"
        txt = str(ln.get("text") or "").strip()
        if txt:
            entry = {"speaker": spk, "text": txt}
            visual = (ln.get("visual") or "").strip()
            if visual:
                entry["visual"] = visual
            graphic = _normalize_graphic(ln.get("graphic"))
            if graphic:
                entry["graphic"] = graphic
            camera = _normalize_camera(ln.get("camera"))
            if camera:
                entry["camera"] = camera
            clean.append(entry)
    if not clean:
        raise ValueError("Script had no usable lines.")
    data["lines"] = clean

    # ---- title / description (coerce to non-empty strings) ----
    title = str(data.get("title") or "").strip()
    data["title"] = title or "You won't believe this 🤯"
    data["description"] = str(data.get("description") or "").strip()

    # ---- tags (accept list or comma/space-separated string) ----
    data["tags"] = _normalize_tags(data.get("tags"))
    return data


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
    return tags or ["shorts", "brainrot", "fyp"]


if __name__ == "__main__":
    import sys

    vt = sys.argv[1] if len(sys.argv) > 1 else "facts"
    out = generate_script(vt)
    print(json.dumps(out, indent=2, ensure_ascii=False))
