from studio.scenes import build_scenes, frames_for, scene_prompt


def _script(n):
    return {
        "title": "T",
        "characters": "a fluffy yellow chick",
        "lines": [
            {"speaker": "narrator", "text": f"line {i}", "visual": f"chick does thing {i}"}
            for i in range(n)
        ],
    }


def _segments(durs):
    segs, t = [], 0.0
    for d in durs:
        segs.append({"speaker": "narrator", "text": "x", "start": t, "end": t + d})
        t += d
    return segs


def test_build_scenes_groups_to_target():
    script = _script(8)
    segs = _segments([2.0] * 8)  # 16s total
    scenes = build_scenes(script, segs, max_scenes=6, target_seconds=5.0)
    assert 3 <= len(scenes) <= 4
    assert scenes[0]["start"] == 0.0
    assert abs(scenes[-1]["end"] - 16.0) < 0.01
    # contiguous, no gaps
    for a, b in zip(scenes, scenes[1:]):
        assert abs(a["end"] - b["start"]) < 0.01


def test_build_scenes_caps_scene_count():
    script = _script(20)
    segs = _segments([3.0] * 20)  # 60s
    scenes = build_scenes(script, segs, max_scenes=6, target_seconds=3.0)
    assert len(scenes) <= 7  # target stretches to total/max_scenes


def test_tiny_tail_merged():
    script = _script(3)
    segs = _segments([5.0, 5.0, 1.0])
    scenes = build_scenes(script, segs, max_scenes=6, target_seconds=5.0)
    assert abs(scenes[-1]["end"] - 11.0) < 0.01
    assert len(scenes) == 2


def test_frames_for_ltx_constraint():
    for dur in (1.0, 3.0, 4.0, 6.0, 30.0):
        f = frames_for(dur, fps=24)
        assert f % 8 == 1
        assert 49 <= f <= 161


def test_scene_prompt_uses_characters_and_visual():
    script = _script(2)
    scenes = build_scenes(script, _segments([3.0, 3.0]), target_seconds=5.0)
    cfg = {"comfyui": {}}
    p = scene_prompt(scenes[0], script, cfg)
    assert "fluffy yellow chick" in p
    assert "chick does thing 0" in p


def test_scene_prompt_falls_back_to_text():
    script = {"title": "T", "lines": [{"speaker": "narrator", "text": "hello world"}]}
    segs = [{"speaker": "narrator", "text": "hello world", "start": 0.0, "end": 3.0}]
    scenes = build_scenes(script, segs, target_seconds=5.0)
    p = scene_prompt(scenes[0], script, {"comfyui": {}})
    assert "hello world" in p
