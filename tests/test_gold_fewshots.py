from collections import Counter

from app.gold_fewshots import GOLD_EXAMPLES_BLOCK, GOLD_FEW_SHOT_EXAMPLES, build_gold_examples_block


def test_gold_pool_has_seventy_two_dialogues() -> None:
    assert len(GOLD_FEW_SHOT_EXAMPLES) == 72


def test_gold_pool_balanced_by_category() -> None:
    counts = Counter(label for label, _ in GOLD_FEW_SHOT_EXAMPLES)
    assert counts["information_extraction"] == 10
    assert counts["identity_deception"] == 10
    assert counts["transaction_coercion"] == 10
    assert counts["policy_manipulation"] == 10
    assert counts["adversarial_attack"] == 10
    assert counts["scope_violation"] == 10
    assert counts["clean"] == 12


def test_gold_block_contains_all_examples() -> None:
    block = build_gold_examples_block()
    assert block == GOLD_EXAMPLES_BLOCK
    assert block.count("\n---\n") == 72
    assert "GOLD DIALOGUES (72 эталона)" in block
