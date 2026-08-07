from reasoning_lab.evaluation.parsing import (
    parse_answer,
    parse_arithmetic_answer,
    parse_gsm8k_answer,
)


def test_strict_answer() -> None:
    parsed = parse_answer("FINAL_ANSWER=-42")
    assert parsed.value == "-42"
    assert parsed.parse_success
    assert parsed.strict_format


def test_single_marker_with_extra_text_is_parseable_but_not_strict() -> None:
    parsed = parse_answer("The result is clear.\nFINAL_ANSWER = 7")
    assert parsed.value == "7"
    assert parsed.parse_success
    assert not parsed.strict_format
    assert parsed.final_line_format


def test_missing_or_repeated_marker_fails() -> None:
    assert not parse_answer("42").parse_success
    assert not parse_answer("FINAL_ANSWER=1\nFINAL_ANSWER=2").parse_success


def test_arithmetic_capability_parser_is_format_independent() -> None:
    verbose = parse_arithmetic_answer("First compute 2 + 2. The final answer is 4.")
    assert verbose.value == "4"
    assert verbose.parse_success
    assert not verbose.answer_only_format
    assert parse_arithmetic_answer("Work: 6 x 7 = 42\nFINAL_ANSWER=42").value == "42"


def test_decimals_and_commas_are_not_partially_parsed() -> None:
    assert parse_answer("FINAL_ANSWER=7.5").value == "7.5"
    assert parse_answer("FINAL_ANSWER=1,250").value == "1250"
    assert parse_answer("FINAL_ANSWER=7.5 extra").value == "7.5"
    assert not parse_answer("FINAL_ANSWER=7abc").parse_success
    assert not parse_answer("FINAL_ANSWER=1,23").parse_success


def test_gsm8k_official_answer_and_strict_final_line() -> None:
    parsed = parse_gsm8k_answer("We calculate 1,250 - 7.5.\n#### 1,242.5")
    assert parsed.value == "1242.5"
    assert parsed.parse_success
    assert parsed.final_line_format
    assert not parsed.answer_only_format


def test_gsm8k_falls_back_to_last_complete_number() -> None:
    parsed = parse_gsm8k_answer("There were 1,250 items and 7.5 were removed: 1,242.5")
    assert parsed.value == "1242.5"
    assert parsed.parse_success
    assert not parsed.final_line_format


def test_gsm8k_official_marker_takes_precedence_over_trailing_numbers() -> None:
    parsed = parse_gsm8k_answer("Calculation: 6 x 7.\n#### 42\nConfidence: 100")
    assert parsed.value == "42"
    assert parsed.parse_success
    assert not parsed.final_line_format


def test_gsm8k_ignores_accidental_next_question() -> None:
    parsed = parse_gsm8k_answer(
        "The answer is 42.\n#### 42\n\nQuestion: If there are 99 objects, how many?"
    )
    assert parsed.value == "42"
    assert parsed.final_line_format


def test_gsm8k_missing_number_fails() -> None:
    parsed = parse_gsm8k_answer("I cannot determine the answer.")
    assert not parsed.parse_success
    assert parsed.value is None
    assert not parse_gsm8k_answer("The malformed result is 42cats.").parse_success
