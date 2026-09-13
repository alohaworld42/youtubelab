"""FEATURE: a video this studio publishes says that a machine made it.

The unit tests in `tests/test_ai_disclosure.py` prove the helper composes the
right strings. That is not the same claim. What matters is the *body that
reaches YouTube*, produced by the real `upload()` after the real scheduler has
chosen the arguments — the seam where a view forgets to pass `cfg`, or a second
call site never learns about the new parameter, and the helper stays perfectly
correct while nothing is disclosed.

There are three upload call sites (`studio.scheduler` twice, once with a channel
and once without, plus `pipeline.generate`) and all of them funnel through
`pipeline.youtube_upload.upload`. These tests run that function for real and
read the dict it hands to `videos().insert`.

Only the transport is replaced: `googleapiclient` is not a test dependency and
uploading for real is not a test. Everything from the scheduler's decision down
to `body["snippet"]["description"]` is production code.
"""
import sys
import types

import pytest

from pipeline import ai_disclosure
from studio import models
from studio.scheduler import Scheduler

NOTE = ai_disclosure.DEFAULT_NOTE


@pytest.fixture()
def youtube(monkeypatch):
    """Capture the request body the real `upload()` builds."""
    captured = {}

    # googleapiclient is not installed in CI; upload() imports MediaFileUpload
    # at call time, so a stub module is enough to let the real code run.
    fake_http = types.ModuleType("googleapiclient.http")
    fake_http.MediaFileUpload = lambda *a, **k: object()
    fake_pkg = types.ModuleType("googleapiclient")
    fake_pkg.http = fake_http
    monkeypatch.setitem(sys.modules, "googleapiclient", fake_pkg)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", fake_http)

    class _Request:
        def next_chunk(self):
            return None, {"id": "vid123"}

    class _Videos:
        def insert(self, part, body, media_body):
            captured["part"] = part
            captured["body"] = body
            return _Request()

    class _Service:
        def videos(self):
            return _Videos()

    import pipeline.youtube_upload as yt

    monkeypatch.setattr(yt, "get_service", lambda *a, **k: _Service())
    monkeypatch.setattr(yt, "_get_service", lambda *a, **k: _Service())
    return captured


def _cfg(tmp_path, **youtube_over):
    yt = {"privacy": "private", "category_id": "24", "append_shorts_tag": False}
    yt.update(youtube_over)
    return {
        "_root": str(tmp_path),
        "paths": {"output_dir": "out"},
        "youtube": yt,
        "studio": {"max_attempts": 3, "tick_seconds": 1, "daily_upload_cap": 2},
    }


def _upload(tmp_path, description="A song about brushing teeth.", **kwargs):
    from pipeline.youtube_upload import upload

    video = tmp_path / "v.mp4"
    video.write_bytes(b"\0" * 16)
    return upload(str(video), "Title", description, ["t"], _cfg(tmp_path), **kwargs)


# ================================================= what reaches YouTube ======
def test_every_upload_carries_the_disclosure_in_its_description(youtube, tmp_path):
    _upload(tmp_path)

    description = youtube["body"]["snippet"]["description"]
    assert NOTE in description, "an AI-generated video was published undisclosed"


def test_the_disclosure_survives_a_description_that_fills_the_whole_field(
    youtube, tmp_path
):
    """YouTube's description cap is enforced in `upload()`. If the note were
    appended before that slice, a long description would push it off the end and
    publish undisclosed — with no error anywhere."""
    _upload(tmp_path, description="x" * 6000)

    description = youtube["body"]["snippet"]["description"]
    assert len(description) <= 4900
    assert description.endswith(NOTE)


def test_a_description_that_already_says_it_is_not_told_twice(youtube, tmp_path):
    _upload(tmp_path, description=f"A song.\n\n{NOTE}")

    assert youtube["body"]["snippet"]["description"].count(NOTE) == 1


def test_the_shorts_tag_and_the_disclosure_both_survive(youtube, tmp_path):
    """Two things append to the same field; the second must not eat the first."""
    from pipeline.youtube_upload import upload

    video = tmp_path / "v.mp4"
    video.write_bytes(b"\0" * 16)
    upload(str(video), "T", "A song.", ["t"],
           _cfg(tmp_path, append_shorts_tag=True))

    description = youtube["body"]["snippet"]["description"]
    assert "#Shorts" in description
    assert NOTE in description


# ======================================== the altered/synthetic-media flag ===
def test_an_animated_kidsong_is_not_declared_altered_or_synthetic(youtube, tmp_path):
    """YouTube exempts wholly animated work, so declaring it would be a wrong
    label. The description still discloses — that is the AI Act's concern, and
    it is a different question from YouTube's."""
    _upload(tmp_path, video_type="kidsong")

    assert youtube["body"]["status"]["containsSyntheticMedia"] is False
    assert NOTE in youtube["body"]["snippet"]["description"]


def test_a_stock_footage_video_is_declared(youtube, tmp_path):
    _upload(tmp_path, video_type="facts")

    assert youtube["body"]["status"]["containsSyntheticMedia"] is True


def test_the_flag_is_part_of_the_status_the_request_actually_sends(youtube, tmp_path):
    """`part` must name `status`, or the field is dropped server-side and the
    declaration silently never happens."""
    _upload(tmp_path, video_type="facts")

    assert "status" in youtube["part"]


def test_the_kids_declaration_still_works_alongside_it(youtube, tmp_path):
    _upload(tmp_path, video_type="kidsong", made_for_kids=True)

    status = youtube["body"]["status"]
    assert status["selfDeclaredMadeForKids"] is True
    assert status["containsSyntheticMedia"] is False


# ============================== the scheduler, end to end, no stubbed upload ==
def _approved_job(tmp_path, channel_id, video_type, description="D"):
    job_id = models.create_job(channel_id=channel_id, video_type=video_type,
                               topic="counting")
    (tmp_path / "v.mp4").write_bytes(b"x")
    models.create_video(job_id=job_id, channel_id=channel_id,
                        path=str(tmp_path / "v.mp4"), title="T",
                        description=description, tags=["t"], duration=1.0)
    models.set_job_status(job_id, "approved")
    return job_id


def test_a_scheduled_upload_is_disclosed_without_anyone_asking(
    youtube, tmp_db, tmp_path, monkeypatch
):
    """The whole chain: an approved job, the real uploader tick, the real
    `upload()`, the real body. Nothing in the scheduler mentions disclosure —
    it must happen anyway."""
    monkeypatch.setattr("studio.oauth.channel_upload_paths",
                        lambda *a, **k: ("secret.json", "token.json"))
    ch = models.create_channel("Test")
    _approved_job(tmp_path, ch, "kidsong")

    Scheduler(_cfg(tmp_path))._process_uploads()

    body = youtube["body"]
    assert NOTE in body["snippet"]["description"]
    assert body["status"]["containsSyntheticMedia"] is False, (
        "the scheduler did not pass the job's video_type, so an animated "
        "episode was mislabelled as realistic synthetic media"
    )


def test_a_scheduled_non_animated_upload_is_declared(
    youtube, tmp_db, tmp_path, monkeypatch
):
    monkeypatch.setattr("studio.oauth.channel_upload_paths",
                        lambda *a, **k: ("secret.json", "token.json"))
    ch = models.create_channel("Test")
    _approved_job(tmp_path, ch, "facts")

    Scheduler(_cfg(tmp_path))._process_uploads()

    assert youtube["body"]["status"]["containsSyntheticMedia"] is True


def test_the_channel_less_upload_path_discloses_too(
    youtube, tmp_db, tmp_path, monkeypatch
):
    """`_process_uploads` has TWO upload calls — one with a channel, one
    without. The second is exactly the kind of call site that gets forgotten."""
    _approved_job(tmp_path, None, "facts")

    Scheduler(_cfg(tmp_path))._process_uploads()

    body = youtube["body"]
    assert NOTE in body["snippet"]["description"]
    assert body["status"]["containsSyntheticMedia"] is True


# ================================================= turning it off is possible ==
def test_an_operator_who_switches_it_off_publishes_without_it(youtube, tmp_path):
    from pipeline.youtube_upload import upload

    cfg = _cfg(tmp_path)
    cfg["ai_disclosure"] = {"enabled": False}
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\0" * 16)
    upload(str(video), "T", "A song.", ["t"], cfg)

    body = youtube["body"]
    assert NOTE not in body["snippet"]["description"]
    assert body["status"]["containsSyntheticMedia"] is False
