from app.models import RED_FLAG_CATEGORIES
from app.synthetic_fewshots import EXPANDED_FEW_SHOT_EXAMPLES


def test_expanded_few_shot_labels_valid() -> None:
    allowed = set(RED_FLAG_CATEGORIES) | {"clean"}
    assert len(EXPANDED_FEW_SHOT_EXAMPLES) >= 40
    for label, dialogue in EXPANDED_FEW_SHOT_EXAMPLES:
        assert label in allowed, label
        assert dialogue.strip()
        assert "user:" in dialogue.lower()


def test_expanded_covers_all_violation_classes() -> None:
    by_label = {label for label, _ in EXPANDED_FEW_SHOT_EXAMPLES}
    assert by_label >= set(RED_FLAG_CATEGORIES)
