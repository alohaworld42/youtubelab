"""FEATURE: der Betreiber gibt nur frei — Titel, Beschreibung und Tags entstehen selbst.

Das Ziel dieses Studios ist, mehrere Kanäle zu fahren, ohne pro Video Metadaten
zu tippen. Dafür reicht es nicht, dass `studio/branding.py` die richtigen
Strings baut (das prüft `tests/test_branding.py`); es muss auch tatsächlich am
Weg des Videos in die Datenbank hängen. Genau diese Naht ist die, an der so
etwas verloren geht — die Engine bleibt korrekt, und niemand ruft sie auf.

Zwei Dinge werden hier festgenagelt, die beide leicht falsch herum wären:

  * **Wann** gebrandet wird. Die Metadaten stehen in der `videos`-Zeile, nicht
    erst im Upload-Tick. Der Operator kann nur freigeben, was er sieht — würde
    der Titel erst beim Hochladen entstehen, zeigte die Review-Seite einen
    anderen als YouTube bekäme.
  * **Dass die Sprache dem Kanal folgt.** Ein deutscher Kanal darf keinen
    englischen KI-Hinweis tragen; der Sinn des Hinweises ist, dass der
    Zuschauer ihn liest.

Kein GPU, kein Netz: `pipeline.generate.generate` ist gepatcht, alles darunter
ist echt.
"""
import json

import pytest

from studio import models
from studio.scheduler import Scheduler

DE_NOTE = "Dieses Video wurde mit Hilfe von künstlicher Intelligenz erstellt."
EN_NOTE = "This video was created using artificial intelligence."


def _cfg(tmp_path, **over):
    cfg = {
        "_root": str(tmp_path),
        "paths": {"output_dir": "out"},
        "studio": {"max_attempts": 3, "tick_seconds": 1},
        "ai_disclosure": {
            "enabled": True,
            "description_note": EN_NOTE,
            "description_note_by_language": {"de": DE_NOTE, "en": EN_NOTE},
        },
    }
    cfg.update(over)
    return cfg


def _plain_result(**over):
    base = {
        "video_path": "out/video.mp4",
        "title": "Rohtitel aus der Pipeline",
        "description": "Rohbeschreibung.",
        "tags": ["roh"],
        "duration": 42.0,
    }
    base.update(over)
    return base


def _run(tmp_path, monkeypatch, channel_id, topic="Zähne putzen", **result_over):
    # Eine Datei, die es wirklich gibt: der Upload-Tick verwirft einen Job,
    # dessen Videodatei fehlt, und der Test darunter würde still nichts prüfen.
    real = tmp_path / "video.mp4"
    real.write_bytes(b"\0" * 16)
    result_over.setdefault("video_path", str(real))

    monkeypatch.setattr("pipeline.generate.generate",
                        lambda *a, **k: _plain_result(**result_over))
    job_id = models.create_job(channel_id=channel_id, video_type="kidsong", topic=topic)
    Scheduler(_cfg(tmp_path))._run_pipeline(models.get_job(job_id))
    return models.get_video_by_job(job_id)


# ============================================ Branding erreicht die Datenbank ==
def test_der_kanaltitel_entsteht_aus_der_vorlage(tmp_db, tmp_path, monkeypatch):
    ch = models.create_channel(
        "Lernlieder",
        language="de",
        branding={"title_template": "{topic} | Lernlieder für Kinder"},
    )
    video = _run(tmp_path, monkeypatch, ch)

    assert video["title"].startswith("Zähne putzen")
    assert "Lernlieder für Kinder" in video["title"]


def test_ohne_vorlage_bleibt_der_pipeline_titel_stehen(tmp_db, tmp_path, monkeypatch):
    """Branding ist ein Default, kein Zwang: ein Kanal ohne Vorlage darf nicht
    plötzlich einen leeren oder generierten Titel bekommen."""
    ch = models.create_channel("Ohne Branding")
    video = _run(tmp_path, monkeypatch, ch)

    assert video["title"] == "Rohtitel aus der Pipeline"


def test_die_kanal_tags_haengen_an_den_video_tags(tmp_db, tmp_path, monkeypatch):
    ch = models.create_channel(
        "Lernlieder", branding={"tags": ["kinderlieder", "lernen"]},
    )
    video = _run(tmp_path, monkeypatch, ch)

    tags = video["tags"]
    assert "roh" in tags
    assert "kinderlieder" in tags and "lernen" in tags


def test_der_titel_reisst_das_youtube_limit_nicht(tmp_db, tmp_path, monkeypatch):
    """100 Zeichen sind hart — YouTube weist einen längeren Titel ab, und das
    passierte dann erst im Upload-Tick, lange nach der Freigabe."""
    ch = models.create_channel(
        "Lang", branding={"title_template": "{topic} " + "x" * 200},
    )
    video = _run(tmp_path, monkeypatch, ch)

    assert len(video["title"]) <= 100


# ================================================== die Sprache folgt dem Kanal ==
def test_ein_deutscher_kanal_bekommt_den_deutschen_ki_hinweis(
    tmp_db, tmp_path, monkeypatch
):
    ch = models.create_channel("Lernlieder", language="de")
    video = _run(tmp_path, monkeypatch, ch)

    assert DE_NOTE in video["description"]
    assert EN_NOTE not in video["description"]


def test_ein_englischer_kanal_bekommt_den_englischen(tmp_db, tmp_path, monkeypatch):
    ch = models.create_channel("Nursery Rhymes", language="en")
    video = _run(tmp_path, monkeypatch, ch)

    assert EN_NOTE in video["description"]
    assert DE_NOTE not in video["description"]


def test_zwei_kanaele_verschiedener_sprache_stoeren_sich_nicht(
    tmp_db, tmp_path, monkeypatch
):
    """Der eigentliche Zweck: ein Dashboard, mehrere Kanäle, jeder in seiner
    Sprache. Ein globaler Zustand hier wäre genau der Fehler, den man erst
    beim zweiten Kanal bemerkt."""
    de = models.create_channel("DE", language="de")
    en = models.create_channel("EN", language="en")

    video_de = _run(tmp_path, monkeypatch, de)
    video_en = _run(tmp_path, monkeypatch, en)

    assert DE_NOTE in video_de["description"]
    assert EN_NOTE in video_en["description"]
    assert EN_NOTE not in video_de["description"]


def test_der_hinweis_steht_genau_einmal_da(tmp_db, tmp_path, monkeypatch):
    """Die Pipeline liefert bereits eine Beschreibung; trägt die den Hinweis
    schon, darf Branding ihn nicht ein zweites Mal anhängen."""
    ch = models.create_channel("Lernlieder", language="de")
    video = _run(tmp_path, monkeypatch, ch,
                 description=f"Rohbeschreibung.\n\n{DE_NOTE}")

    assert video["description"].count(DE_NOTE) == 1


# ======================================= was der Operator sieht, geht auch raus ==
def test_die_freigabe_zeigt_dieselben_metadaten_die_hochgeladen_werden(
    tmp_db, tmp_path, monkeypatch
):
    """Der Kern der Entscheidung, WANN gebrandet wird. Der Upload liest die
    `videos`-Zeile; die Review-Seite liest dieselbe Zeile. Entstünde der Titel
    erst im Upload-Tick, liefen die beiden auseinander und der Operator gäbe
    etwas frei, das er nie gesehen hat."""
    ch = models.create_channel(
        "Lernlieder", language="de",
        branding={"title_template": "{topic} | Lernlieder", "tags": ["kinderlieder"]},
    )
    video = _run(tmp_path, monkeypatch, ch)
    models.set_job_status(video["job_id"], "approved")

    seen = {}

    def fake_upload(path, title, description, tags, cfg, **kwargs):
        seen.update(path=path, title=title, description=description, tags=tags)
        return {"video_id": "x", "url": "https://youtu.be/x", "privacy": "private"}

    import pipeline.youtube_upload as yt

    monkeypatch.setattr(yt, "upload", fake_upload)
    monkeypatch.setattr("studio.oauth.channel_upload_paths",
                        lambda *a, **k: ("secret.json", "token.json"))

    Scheduler(_cfg(tmp_path))._process_uploads()

    assert seen, "der Upload-Tick hat gar nicht hochgeladen — der Test prüfte nichts"
    assert seen["title"] == video["title"]
    assert seen["description"] == video["description"]
    assert list(seen["tags"]) == list(video["tags"])
    # und das Gebrandete ist wirklich drin, nicht nur beidseitig gleich leer
    assert "Lernlieder" in seen["title"]
    assert DE_NOTE in seen["description"]
    assert "kinderlieder" in seen["tags"]


# ================================ das Profil steuert die Pipeline, ohne zu kleben ==
def test_das_profil_erreicht_die_pipeline(tmp_db, tmp_path, monkeypatch):
    """`content_profile` ist der eine Schalter, der einen Kanal zu einem Produkt
    macht. Kommt er nicht bis zur Pipeline durch, ist das ganze Konzept nur eine
    Spalte in der Datenbank."""
    from pipeline.config import load_config

    base = load_config()
    base["_root"] = str(tmp_path)
    base["paths"] = {"output_dir": "out"}
    base["studio"] = {"max_attempts": 3, "tick_seconds": 1}

    ch = models.create_channel("DE-Lieder", content_profile="kidsong_de")
    seen = {}
    real = tmp_path / "video.mp4"
    real.write_bytes(b"\0" * 16)

    def fake_generate(*a, **k):
        seen["cfg"] = k.get("cfg")
        return _plain_result(video_path=str(real))

    monkeypatch.setattr("pipeline.generate.generate", fake_generate)
    job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="Zaehne")
    Scheduler(base)._run_pipeline(models.get_job(job_id))

    assert seen["cfg"]["kidsong"]["language"] == "de"
    assert seen["cfg"]["kidsong"]["render_style"] == "pixar_toon"


def test_der_scheduler_bleibt_nicht_in_der_letzten_sprache_haengen(
    tmp_db, tmp_path, monkeypatch
):
    """EIN Scheduler-Objekt bedient alle Kanäle, prozesslang. Würde das
    aufgelöste Profil auf `self.cfg` zurückgeschrieben, liefe nach einem
    deutschen Kanal auch der englische auf Deutsch — und auffallen würde es
    erst beim zweiten Kanal, also nie im Test eines einzelnen."""
    from pipeline.config import load_config

    base = load_config()
    base["_root"] = str(tmp_path)
    base["paths"] = {"output_dir": "out"}
    base["studio"] = {"max_attempts": 3, "tick_seconds": 1}
    before = base["kidsong"]["language"]

    de = models.create_channel("DE", content_profile="kidsong_de")
    en = models.create_channel("EN", content_profile="kidsong_en")
    real = tmp_path / "video.mp4"
    real.write_bytes(b"\0" * 16)

    seen = []
    monkeypatch.setattr(
        "pipeline.generate.generate",
        lambda *a, **k: (seen.append(k["cfg"]["kidsong"]["language"]),
                         _plain_result(video_path=str(real)))[1],
    )

    sched = Scheduler(base)
    for ch in (de, en, de):
        job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="t")
        sched._run_pipeline(models.get_job(job_id))

    assert seen == ["de", "en", "de"], f"Sprache klebt zwischen Kanaelen: {seen}"
    assert sched.cfg["kidsong"]["language"] == before, "self.cfg wurde mutiert"


# ==================================== das Profil bestimmt, was der Kanal macht ==
def _base_cfg(tmp_path):
    from pipeline.config import load_config
    cfg = load_config()
    cfg["_root"] = str(tmp_path)
    cfg["paths"] = {"output_dir": "out"}
    cfg["studio"] = {"max_attempts": 3, "tick_seconds": 1}
    return cfg


def test_ein_profil_kanal_bekommt_seinen_typ_nicht_den_der_idee(
    tmp_db, tmp_path, monkeypatch
):
    """Ideen-Zeilen überleben Konfigurationsänderungen. Stellt man einen Kanal
    von kidsong auf hyperframes um, produzierte der alte Backlog weiter den
    alten Typ, solange er reichte — was sich anfühlt, als hätte die Umstellung
    nichts getan."""
    ch = models.create_channel("Hyperframes DE", content_profile="hyperframes_de")
    models.update_channel(ch, default_video_type="facts", schedule=["00:01"])
    models.add_idea(ch, "Warum der Himmel blau ist", video_type="kidsong")

    Scheduler(_base_cfg(tmp_path))._enqueue_due_slots()

    jobs = models.jobs_in_states(["queued"])
    assert jobs, "kein Job angelegt"
    assert jobs[0]["video_type"] == "hyperframes"


def test_ohne_profil_gewinnt_weiterhin_die_idee(tmp_db, tmp_path, monkeypatch):
    """Die Vorrangregel für profillose Kanäle bleibt exakt wie vorher — sonst
    wäre das hier eine stille Verhaltensänderung für jeden bestehenden Kanal."""
    ch = models.create_channel("Alt")
    models.update_channel(ch, default_video_type="facts", schedule=["00:01"])
    models.add_idea(ch, "etwas", video_type="scary")

    Scheduler(_base_cfg(tmp_path))._enqueue_due_slots()

    jobs = models.jobs_in_states(["queued"])
    assert jobs and jobs[0]["video_type"] == "scary"
