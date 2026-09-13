from studio import models
from studio.ideas import add_user_idea, dedupe_hash, is_duplicate, normalize_topic


def test_normalize_strips_punctuation_and_case():
    assert normalize_topic("  Der GRUSELIGSTE Leuchtturm!!! ") == "der gruseligste leuchtturm"


def test_hash_equal_for_normalized_variants():
    assert dedupe_hash("Cats vs Dogs?") == dedupe_hash("cats vs dogs")


def test_fuzzy_duplicate_detection():
    existing = ["the scariest lighthouse in germany"]
    assert is_duplicate("The scariest lighthouse in Germany!", existing)
    assert is_duplicate("the scariest lighthouse in germanys", existing)  # near match
    assert not is_duplicate("why octopuses have three hearts", existing)


def test_empty_topic_is_duplicate():
    assert is_duplicate("", [])
    assert is_duplicate("!!!", [])


def test_add_user_idea_rejects_exact_dupe(tmp_db):
    ch = models.create_channel("Test")
    first = add_user_idea(ch, "Cats vs Dogs")
    second = add_user_idea(ch, "cats vs DOGS!")  # same after normalization
    assert first is not None
    assert second is None


def test_same_topic_allowed_on_other_channel(tmp_db):
    a = models.create_channel("A")
    b = models.create_channel("B")
    assert add_user_idea(a, "Cats vs Dogs") is not None
    assert add_user_idea(b, "Cats vs Dogs") is not None


def test_user_ideas_beat_auto_ideas(tmp_db):
    ch = models.create_channel("Test")
    models.add_idea(ch, "auto topic", source="auto", priority=99, dedupe_hash="h1")
    models.add_idea(ch, "user topic", source="user", priority=0, dedupe_hash="h2")
    nxt = models.next_pending_idea(ch)
    assert nxt["topic"] == "user topic"
