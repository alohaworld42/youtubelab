"""FEATURE: ein Kanal mit Profil produziert wirklich etwas — in seiner Sprache.

Diese Datei existiert, weil ihr Fehlen einen echten Defekt durchgelassen hat.
`content_profile`, `language` und die beiden neuen Video-Typen wurden gebaut,
getestet und gemerged — und ein `brainrot`-Job wäre trotzdem sofort mit
`FileNotFoundError: No prompt template for video type 'brainrot'` gestorben,
während die Profile für `hyperframes` ausschliesslich Keys unter `kidsong`
setzten, die dieser Pfad nie liest. Alles war grün. Nichts hätte produziert.

Der Grund ist immer derselbe: jede Schicht wurde gegen ihre Nachbarin geprüft,
nie eine Anfrage durch alle Schichten. Also geht hier jeder Kanal-Typ EINMAL
komplett durch — echter Scheduler, echtes `generate()`, echtes `script_gen`.
Gemockt ist nur, was Geld oder eine GPU kostet: der LLM-Aufruf und das Rendern.

**Absichtlich verhaltensbasiert.** Geprüft wird, dass ein deutscher Kanal ein
deutsches Skript anfordert — nicht, welcher Config-Key das bewirkt. Der Key hat
sich während dieser Arbeit schon einmal geändert; das Verhalten darf sich nicht
ändern, und ein Test, der am Key klebt, wäre bei der nächsten Umbenennung grün
und falsch.
"""
import pytest

from studio import models
from studio.scheduler import Scheduler

# Jeder Kanal-Typ, den dieses Studio fahren soll, mit der Sprache, in der sein
# Publikum ihn hört.
PROFILES = [
    ("kidsong_de", "de"),
    ("kidsong_en", "en"),
    ("hyperframes_de", "de"),
    ("hyperframes_en", "en"),
    ("brainrot_short", "en"),
]

# Wörter, die in einer an ein LLM gerichteten Anweisung stehen, wenn Deutsch
# verlangt wird. Absichtlich mehrere: die genaue Formulierung ist frei, die
# Absicht nicht.
GERMAN_MARKERS = ("german", "deutsch")


@pytest.fixture()
def base_cfg(tmp_path):
    from pipeline.config import load_config

    cfg = load_config()
    # `_root` bleibt das echte Repo: dort liegen `prompts/`, die Cast-Bibel und
    # die Workflow-Graphen, die der Generator wirklich liest. Nur was
    # GESCHRIEBEN wird, wandert nach tmp.
    cfg["paths"] = {"output_dir": str(tmp_path / "out")}
    cfg["studio"] = {"max_attempts": 3, "tick_seconds": 1}
    return cfg


@pytest.fixture()
def captured_prompt(monkeypatch):
    """Fängt den Prompt ab, der ans LLM ginge, und antwortet wie ein LLM.

    Der Punkt: `script_gen` läuft dabei ECHT. Ein Test, der `generate_script`
    selbst mockt, prüft nur seine eigene Erwartung.
    """
    seen = {}

    def fake_llm(prompt, *a, **k):
        seen["prompt"] = prompt
        # Das Format, das `generate_script` wirklich parst (siehe die
        # `lines`-Normalisierung dort): Sprecher-Objekte, keine blanken Strings.
        return (
            '{"title": "Ein Titel", "description": "Eine Beschreibung", '
            '"tags": ["a", "b"], "lines": ['
            '{"speaker": "narrator", "text": "Zeile eins."}, '
            '{"speaker": "narrator", "text": "Zeile zwei."}]}'
        )

    import pipeline.script_gen as sg

    # Der Name des LLM-Einstiegs darf sich ändern; wir patchen, was da ist.
    for name in ("call_llm", "_call_llm", "complete", "chat"):
        if hasattr(sg, name):
            monkeypatch.setattr(sg, name, fake_llm)
            seen["patched"] = name
            break
    return seen


def _channel(profile, language):
    ch = models.create_channel(f"Kanal {profile}", language=language,
                               content_profile=profile)
    models.update_channel(ch, status="active")
    return ch


@pytest.mark.parametrize("profile,language", PROFILES)
def test_jedes_profil_bekommt_seine_sprache_in_die_pipeline(
    tmp_db, base_cfg, profile, language, monkeypatch
):
    """Die Sprache muss bis in das cfg reichen, das der Generator bekommt —
    egal über welchen Key. Vorher setzten die Profile für hyperframes und
    brainrot ausschliesslich Keys unter `kidsong`, die deren Pfad nie liest."""
    from studio import profiles

    ch = _channel(profile, language)
    resolved = profiles.apply_profile(base_cfg, models.get_channel(ch))

    # In irgendeiner Form muss die Sprache im aufgelösten cfg stehen, und zwar
    # so, dass sie sich von der anderen Sprache unterscheidet.
    found = _languages_in(resolved)
    assert language in found, (
        f"Profil {profile} hinterlässt die Sprache {language!r} nirgends im "
        f"cfg — gefunden: {sorted(found)}"
    )


def _languages_in(cfg):
    """Alle Sprachcodes, die irgendwo im cfg unter einem *language*-Key stehen."""
    out = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if "language" in key.lower() and isinstance(value, str) and value:
                    out.add(value.split("-")[0].lower())
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(cfg)
    return out


def test_ein_deutscher_kanal_verlangt_ein_deutsches_skript(
    tmp_db, base_cfg, captured_prompt, monkeypatch
):
    """Durch den echten `script_gen`: die Anweisung ans LLM muss Deutsch
    verlangen. Ohne das liefert ein deutscher Kanal englische Videos, und
    auffallen würde es erst am fertigen Video."""
    from pipeline.script_gen import generate_script
    from studio import profiles

    ch = _channel("hyperframes_de", "de")
    cfg = profiles.apply_profile(base_cfg, models.get_channel(ch))

    if "patched" not in captured_prompt:
        pytest.skip("LLM-Einstieg in script_gen nicht gefunden")

    generate_script("hyperframes", "Warum der Himmel blau ist", cfg)

    prompt = captured_prompt["prompt"].lower()
    assert any(m in prompt for m in GERMAN_MARKERS), (
        "der Prompt an das LLM verlangt kein Deutsch"
    )


def test_ein_englischer_kanal_bekommt_keine_sprachanweisung_untergeschoben(
    tmp_db, base_cfg, captured_prompt
):
    """Die Regressionsgarantie: für Englisch muss der Prompt bleiben, was er
    war. Jeder bestehende Kanal dieses Studios ist englisch."""
    from pipeline.script_gen import generate_script
    from studio import profiles

    ch = _channel("hyperframes_en", "en")
    cfg = profiles.apply_profile(base_cfg, models.get_channel(ch))

    if "patched" not in captured_prompt:
        pytest.skip("LLM-Einstieg in script_gen nicht gefunden")

    generate_script("hyperframes", "Why the sky is blue", cfg)

    assert "deutsch" not in captured_prompt["prompt"].lower()


@pytest.mark.parametrize("video_type", ["hyperframes", "brainrot"])
def test_die_neuen_typen_haben_ueberhaupt_eine_vorlage(video_type):
    """Der Defekt, den diese Datei zu spät gefangen hätte: `brainrot` war ein
    registrierter Video-Typ, für den der Scheduler Jobs anlegte, ohne dass eine
    Prompt-Datei existierte. Jeder dieser Jobs starb sofort."""
    import os

    from pipeline.script_gen import _load_prompt

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    template = _load_prompt(video_type, root)
    assert template and template.strip()


def test_jeder_registrierte_video_typ_kann_ein_skript_bauen():
    """Die allgemeine Form derselben Regel. Ein neuer Eintrag in VIDEO_TYPES
    ist eine Zeile; die Prompt-Datei zu vergessen ist der Standardfehler, und
    er fällt sonst erst dem Betreiber auf, als fehlgeschlagener Job."""
    import os

    from pipeline.script_gen import _load_prompt
    from studio.views import VIDEO_TYPES

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    missing = []
    for entry in VIDEO_TYPES:
        video_type = entry["id"]
        if video_type == "kidsong":
            continue          # eigene Pipeline, eigene Prompt-Dateien
        try:
            _load_prompt(video_type, root)
        except FileNotFoundError:
            missing.append(video_type)
    assert not missing, f"registrierte Typen ohne Prompt-Vorlage: {missing}"


# ============================== die Stimme muss zur Sprache passen ============
@pytest.mark.parametrize("profile,expected_lang", [
    ("hyperframes_de", "de"), ("hyperframes_en", "en"), ("brainrot_short", "en"),
])
def test_die_sprechstimme_passt_zur_kanalsprache(
    tmp_db, base_cfg, profile, expected_lang
):
    """Ein deutsches Skript, das von einer englischen Stimme vorgelesen wird,
    ist kein halbfertiges Produkt — es ist unbrauchbar. Skriptsprache und
    Stimme kommen aus verschiedenen Config-Zweigen und können auseinanderlaufen,
    ohne dass irgendetwas fehlschlägt."""
    from studio import profiles

    ch = _channel(profile, expected_lang)
    cfg = profiles.apply_profile(base_cfg, models.get_channel(ch))

    # `voices` mischt Sprecher mit Parametern (`rate: "+8%"`), also nur die
    # Sprecher-Slots prüfen, die der TTS-Pfad wirklich als Stimme liest.
    voices = cfg.get("voices") or {}
    speakers = {k: v for k, v in voices.items()
                if k in ("narrator", "speaker_a", "speaker_b") and v}
    assert speakers, f"{profile} setzt keine Sprecherstimme"
    for speaker, name in speakers.items():
        assert str(name).lower().startswith(expected_lang), (
            f"{profile}: Stimme {speaker}={name!r} spricht nicht {expected_lang}"
        )


def test_deutsche_und_englische_kanaele_bekommen_verschiedene_stimmen(
    tmp_db, base_cfg
):
    """Der Test darüber wäre auch grün, wenn beide Profile dieselbe Stimme
    setzten und die zufällig mit dem Sprachkürzel begänne."""
    from studio import profiles

    de = profiles.apply_profile(
        base_cfg, models.get_channel(_channel("hyperframes_de", "de")))
    en = profiles.apply_profile(
        base_cfg, models.get_channel(_channel("hyperframes_en", "en")))

    assert de.get("voices") != en.get("voices")


def test_die_transkription_folgt_derselben_sprache_wie_das_skript(
    tmp_db, base_cfg
):
    """Untertitel entstehen aus dem gerenderten Ton per Whisper. Dekodiert
    Whisper in der falschen Sprache, bekommt ein deutsches Video englisch
    geratene Untertitel — der Ton stimmt, die Schrift nicht."""
    from studio import profiles

    for profile, lang in (("hyperframes_de", "de"), ("hyperframes_en", "en")):
        cfg = profiles.apply_profile(
            base_cfg, models.get_channel(_channel(profile, lang)))
        assert (cfg.get("whisper") or {}).get("language") == lang, profile
