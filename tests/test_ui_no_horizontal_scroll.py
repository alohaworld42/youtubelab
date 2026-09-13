"""Keine Seite darf seitwärts scrollen — gemessen, nicht gelesen.

`tests/test_ui_reachability.py` prüft die *Form* der Templates und sagt in
seinem eigenen Docstring, warum es das tut statt Pixel zu messen: die Suite muss
CPU-only bleiben. Das hat eine Grenze, und die wurde hier gefunden.

Eine Tabelle ist das eine Element, das sich nicht schmaler machen lässt als sein
Inhalt. `width:100%` ist eine Bitte; ein langes Thema oder eine
Fehlermeldung — im Fund war es ein `FileNotFoundError` mit vollem Pfad in der
Spalte "Fehler" — überstimmt sie. Gemessen auf einem 390px-Telefon war die
Job-Tabelle des Dashboards **400px** breit und zog die ganze Seite mit: der Body
scrollte seitwärts, auf der Startseite, im Normalbetrieb.

Kein Blick ins Template hätte das gezeigt. `<table>` sah dort völlig in Ordnung
aus, und die Zellbreite entsteht erst beim Rendern aus Daten, die es zur
Testzeit noch nicht gibt. Deshalb misst diese Datei in einem echten Browser —
und überspringt sauber, wo keiner ist, damit die Hauptsuite CPU-only bleibt.

Der Fix ist `.table-wrap` in `templates/base.html`: jede Tabelle scrollt in
ihrem eigenen Container, damit die Seite es nie tut.
"""
import os
import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api", reason="playwright nicht installiert")

CHROMIUM = "/opt/pw-browsers/chromium"

pytestmark = pytest.mark.skipif(
    not os.path.exists(CHROMIUM), reason="kein vorinstalliertes Chromium"
)

# Jede Seite, die der Betreiber im Alltag öffnet. `/` ist die wichtigste: dort
# stand der Fund.
PAGES = ["/", "/channels/", "/ideas", "/history", "/review"]

# 390x844 = iPhone 12/13/14. Die engste Breite, die real vorkommt; alles was
# hier passt, passt auf jedem größeren Gerät.
PHONE = {"width": 390, "height": 844}
DESKTOP = {"width": 1280, "height": 900}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_studio(tmp_path_factory):
    """Die ECHTE App auf einem echten Port, mit ein paar Kanälen darin.

    Ein Flask-Testclient reicht hier nicht: gemessen wird Layout, und Layout
    entsteht nur in einer Engine.
    """
    tmp = tmp_path_factory.mktemp("live")
    os.environ["STUDIO_DB_PATH"] = str(tmp / "studio.db")
    os.environ["STUDIO_SECRET_KEY"] = "no-h-scroll"
    os.environ.pop("STUDIO_PASSWORD", None)

    from studio import db, models

    db.reset_for_tests()
    ch = models.create_channel("KinderKrippenLernLieder", language="de",
                               content_profile="kidsong_de")
    models.update_channel(ch, schedule=["07:00"], status="active")
    job = models.create_job(channel_id=ch, video_type="kidsong",
                            topic="Zähne putzen am Morgen")
    # Genau die Zutat, die den Fund ausgelöst hat: eine lange, ununterbrochene
    # Fehlermeldung mit vollem Pfad in einer Tabellenzelle.
    models.update_job(
        job, status="failed",
        error="FileNotFoundError: Cannot resume 20260810-091940-kidsong-alle-"
              "meine-entchen-ein-froehliches-mitmachlied-fuer-: /home/user/"
              "youtubegenerator/output/20260810-091940-kidsong-song.json is "
              "missing or unreadable.",
    )

    import app as app_module

    flask_app = app_module.create_app()
    flask_app.config["TESTING"] = True
    port = _free_port()
    server = threading.Thread(
        target=lambda: flask_app.run(host="127.0.0.1", port=port,
                                     use_reloader=False, threaded=True),
        daemon=True,
    )
    server.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(100):                       # auf den Server warten
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.skip("App kam nicht hoch")

    yield base
    db.reset_for_tests()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROMIUM)
        yield b
        b.close()


def _measure(browser, base, path, viewport):
    ctx = browser.new_context(viewport=viewport)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    response = page.goto(base + path, wait_until="networkidle")
    scroll_width = page.evaluate("document.documentElement.scrollWidth")
    ctx.close()
    return response.status, scroll_width, errors


@pytest.mark.parametrize("path", PAGES)
def test_keine_seite_scrollt_auf_dem_telefon_seitwaerts(browser, live_studio, path):
    """Der eigentliche Fund: 400px Inhalt in einem 390px-Fenster."""
    status, scroll_width, _ = _measure(browser, live_studio, path, PHONE)

    assert status == 200, f"{path} antwortete {status}"
    assert scroll_width <= PHONE["width"], (
        f"{path} ist {scroll_width}px breit bei {PHONE['width']}px Fenster — "
        "die Seite scrollt seitwärts"
    )


@pytest.mark.parametrize("path", PAGES)
def test_keine_seite_scrollt_auf_dem_desktop_seitwaerts(browser, live_studio, path):
    status, scroll_width, _ = _measure(browser, live_studio, path, DESKTOP)

    assert status == 200
    assert scroll_width <= DESKTOP["width"]


@pytest.mark.parametrize("path", PAGES)
def test_keine_seite_wirft_javascript_fehler(browser, live_studio, path):
    """Kostet nichts, weil der Browser ohnehin läuft — und ein toter Script-Tag
    macht eine Seite unbedienbar, ohne den Statuscode zu ändern."""
    _, _, errors = _measure(browser, live_studio, path, PHONE)

    assert errors == [], f"{path}: {errors}"


def test_jede_tabelle_steckt_in_einem_scroll_container():
    """Die statische Hälfte derselben Regel, damit sie auch dann greift, wenn
    diese Datei mangels Browser übersprungen wird. Eine neue Tabelle ohne
    `.table-wrap` ist die nächste Seite, die seitwärts scrollt."""
    import glob
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for path in glob.glob(os.path.join(root, "templates", "*.html")):
        text = open(path, encoding="utf-8").read()
        for match in re.finditer(r"<table[\s>]", text):
            before = text[:match.start()]
            # der letzte geöffnete Container vor der Tabelle muss der Wrapper sein
            if 'class="table-wrap"' not in before[-400:]:
                offenders.append(f"{os.path.basename(path)}:{before.count(chr(10)) + 1}")
    assert not offenders, f"Tabelle ohne .table-wrap: {offenders}"
