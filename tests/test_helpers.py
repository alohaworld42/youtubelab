"""Tests for the small pure helpers across the pipeline: slugs, config comment
stripping / path resolution, word cleaning, and caption grouping."""
import os

import pytest

from pipeline.config import _strip_comments, abspath
from pipeline.generate import _slug
from pipeline.captions import _clean
from pipeline.caption_render import group_words


# --------------------------------------------------------------------- _slug --
def test_slug_basic():
    assert _slug("Hello World") == "hello-world"


def test_slug_strips_punctuation_and_emoji():
    assert _slug("You won't BELIEVE this?! 🤯") == "you-wont-believe-this"


def test_slug_collapses_separators():
    assert _slug("a___b   c--d") == "a-b-c-d"


def test_slug_truncates_to_50_chars():
    assert len(_slug("word " * 40)) <= 50


def test_slug_empty_falls_back():
    assert _slug("!!!") == "video"
    assert _slug("") == "video"


# ----------------------------------------------------------- _strip_comments --
def test_strip_comments_removes_underscore_keys():
    src = {"keep": 1, "_comment": "x", "nested": {"_c": 2, "ok": 3}}
    assert _strip_comments(src) == {"keep": 1, "nested": {"ok": 3}}


def test_strip_comments_recurses_into_lists():
    src = {"items": [{"ok": 1, "_c": 2}, {"ok": 3}]}
    assert _strip_comments(src) == {"items": [{"ok": 1}, {"ok": 3}]}


def test_strip_comments_leaves_scalars():
    assert _strip_comments("hi") == "hi"
    assert _strip_comments(5) == 5


# --------------------------------------------------------------------- abspath --
def test_abspath_keeps_absolute_paths():
    p = os.path.abspath(os.sep + "already" + os.sep + "abs")
    assert abspath({"_root": "/root"}, p) == p


def test_abspath_joins_relative_to_root():
    assert abspath({"_root": "/root"}, "output") == os.path.join("/root", "output")


# --------------------------------------------------------------------- _clean --
def test_clean_strips_surrounding_punctuation():
    assert _clean("hello,") == "hello"
    assert _clean('"Wow!"') == "Wow"


def test_clean_keeps_inner_apostrophe():
    assert _clean("don't.") == "don't"


def test_clean_all_punctuation_becomes_empty():
    assert _clean("!!!") == ""


# ---------------------------------------------------------------- group_words --
def _words(*pairs):
    return [{"word": w, "start": s, "end": s + 0.4} for w, s in pairs]


def test_group_words_single_per_group():
    words = _words(("A", 0.0), ("B", 1.0))
    groups = group_words(words, 1)
    assert [g["text"] for g in groups] == ["A", "B"]
    assert groups[0]["start"] == 0.0 and groups[0]["end"] == 0.4


def test_group_words_multiple_per_group():
    words = _words(("A", 0.0), ("B", 1.0), ("C", 2.0))
    groups = group_words(words, 2)
    assert [g["text"] for g in groups] == ["A B", "C"]
    # Group spans from first word's start to last word's end.
    assert groups[0]["start"] == 0.0
    assert groups[0]["end"] == 1.4


@pytest.mark.parametrize("bad_n", [0, -3])
def test_group_words_bad_size_defaults_to_one(bad_n):
    words = _words(("A", 0.0), ("B", 1.0))
    assert len(group_words(words, bad_n)) == 2


def test_group_words_empty_input():
    assert group_words([], 2) == []
