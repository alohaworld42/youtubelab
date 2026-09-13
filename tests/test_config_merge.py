from pipeline.config import merge_overrides


def test_none_or_empty_overrides_return_original():
    cfg = {"a": 1}
    assert merge_overrides(cfg, None) is cfg
    assert merge_overrides(cfg, {}) is cfg


def test_deep_merge_keeps_untouched_keys():
    cfg = {"voices": {"narrator": "en-A", "rate": "+8%"}, "video": {"fps": 30}}
    merged = merge_overrides(cfg, {"voices": {"narrator": "de-B"}})
    assert merged["voices"]["narrator"] == "de-B"
    assert merged["voices"]["rate"] == "+8%"
    assert merged["video"]["fps"] == 30


def test_original_not_mutated():
    cfg = {"voices": {"narrator": "en-A"}}
    merge_overrides(cfg, {"voices": {"narrator": "de-B"}})
    assert cfg["voices"]["narrator"] == "en-A"


def test_scalar_replaces_dict_and_vice_versa():
    cfg = {"x": {"y": 1}}
    assert merge_overrides(cfg, {"x": 5})["x"] == 5
    cfg2 = {"x": 5}
    assert merge_overrides(cfg2, {"x": {"y": 1}})["x"] == {"y": 1}
