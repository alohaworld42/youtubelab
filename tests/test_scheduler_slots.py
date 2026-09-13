import time

from studio import models
from studio.scheduler import due_slots, slot_key_for


def _at(hour, minute):
    return time.struct_time((2026, 7, 2, hour, minute, 0, 3, 183, -1))


def test_due_slots_only_past_times():
    schedule = ["09:00", "18:00"]
    assert due_slots(schedule, _at(8, 59)) == []
    assert due_slots(schedule, _at(9, 0)) == ["09:00"]
    assert due_slots(schedule, _at(23, 30)) == ["09:00", "18:00"]


def test_due_slots_normalizes_format():
    assert due_slots(["9:5"], _at(10, 0)) == ["09:05"]


def test_due_slots_ignores_garbage():
    assert due_slots(["blah", "", "25:99", None], _at(23, 59)) == []
    # 25:99 parses but 25*60+99 > now_minutes for any real time of day
    assert due_slots([], _at(12, 0)) == []


def test_slot_key_unique_prevents_double_enqueue(tmp_db):
    ch = models.create_channel("Test")
    key = slot_key_for(ch, "2026-07-02", "09:00")
    first = models.create_job(channel_id=ch, video_type="facts", slot_key=key)
    second = models.create_job(channel_id=ch, video_type="facts", slot_key=key)
    assert first is not None
    assert second is None
    assert models.slot_key_exists(key)


def test_manual_jobs_have_no_slot_key(tmp_db):
    ch = models.create_channel("Test")
    a = models.create_job(channel_id=ch, video_type="facts")
    b = models.create_job(channel_id=ch, video_type="facts")
    assert a is not None and b is not None and a != b
