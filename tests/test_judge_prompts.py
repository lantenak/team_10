from app.prompts import (
    build_clean_validator_prompt,
    build_final_judge_prompt,
    build_multi_scan_judge_prompt,
)


def test_judge_prompts_contain_dialogue() -> None:
    dialogue = "user: test\nsupport: ok"
    assert "user: test" in build_multi_scan_judge_prompt(dialogue)
    assert "user: test" in build_final_judge_prompt(dialogue, ["information_extraction"])
    assert "user: test" in build_clean_validator_prompt(dialogue)
