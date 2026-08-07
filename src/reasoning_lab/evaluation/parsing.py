from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


NUMBER_PATTERN = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
NUMBER_END_BOUNDARY = r"(?![\w,]|\.\d)"
MARKER_PATTERN = re.compile(
    rf"FINAL_ANSWER\s*=\s*({NUMBER_PATTERN}){NUMBER_END_BOUNDARY}"
)
STRICT_PATTERN = re.compile(rf"^\s*FINAL_ANSWER\s*=\s*({NUMBER_PATTERN})\s*$")
GSM8K_MARKER_PATTERN = re.compile(
    rf"(?<!#)####(?!#)\s*({NUMBER_PATTERN}){NUMBER_END_BOUNDARY}"
)
GSM8K_STRICT_PATTERN = re.compile(rf"^\s*####\s*({NUMBER_PATTERN})\s*$")
NUMBER_TOKEN_PATTERN = re.compile(
    rf"(?<![\w.,])({NUMBER_PATTERN}){NUMBER_END_BOUNDARY}"
)
NEXT_QUESTION_PATTERN = re.compile(r"\r?\nQuestion:")


@dataclass(frozen=True)
class ParsedAnswer:
    value: str | None
    parse_success: bool
    answer_only_format: bool
    final_line_format: bool

    @property
    def strict_format(self) -> bool:
        """Backward-compatible alias for the answer-only contract."""
        return self.answer_only_format


def normalize_number(value: str | int | float) -> str:
    raw = str(value).strip().replace(",", "")
    try:
        number = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"Invalid numeric answer: {value!r}") from error
    if not number.is_finite():
        raise ValueError(f"Non-finite numeric answer: {value!r}")
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def parse_answer(text: str) -> ParsedAnswer:
    matches = MARKER_PATTERN.findall(text)
    answer_only = STRICT_PATTERN.fullmatch(text) is not None
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    final_line = bool(non_empty_lines and STRICT_PATTERN.fullmatch(non_empty_lines[-1]))
    if len(matches) != 1:
        return ParsedAnswer(
            value=None,
            parse_success=False,
            answer_only_format=False,
            final_line_format=False,
        )
    return ParsedAnswer(
        value=normalize_number(matches[0]),
        parse_success=True,
        answer_only_format=answer_only,
        final_line_format=final_line,
    )


def parse_arithmetic_answer(text: str) -> ParsedAnswer:
    """Score arithmetic correctness independently from contract compliance.

    A ``FINAL_ANSWER=`` marker takes precedence when present. Otherwise the
    last complete numeric token is used, so a mathematically correct verbose
    answer is not counted as wrong merely for missing the requested format.
    The two format flags still describe strict contract compliance.
    """

    marker_matches = MARKER_PATTERN.findall(text)
    answer_only = STRICT_PATTERN.fullmatch(text) is not None
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    final_line = bool(non_empty_lines and STRICT_PATTERN.fullmatch(non_empty_lines[-1]))
    if marker_matches:
        value = normalize_number(marker_matches[-1])
    else:
        number_matches = NUMBER_TOKEN_PATTERN.findall(text)
        if not number_matches:
            return ParsedAnswer(
                value=None,
                parse_success=False,
                answer_only_format=False,
                final_line_format=False,
            )
        value = normalize_number(number_matches[-1])
    return ParsedAnswer(
        value=value,
        parse_success=True,
        answer_only_format=answer_only,
        final_line_format=final_line,
    )


def parse_gsm8k_answer(text: str) -> ParsedAnswer:
    """Parse a GSM8K completion while keeping correctness and format separate.

    The official ``#### number`` marker takes precedence when present. Models do
    not always reproduce that marker, so an otherwise well-formed completion is
    still scoreable via its last complete number. If generation starts another
    few-shot ``Question:`` block, that accidental continuation is excluded from
    both answer extraction and format validation.
    """
    continuation = NEXT_QUESTION_PATTERN.search(text)
    candidate = text[: continuation.start()] if continuation else text
    official_matches = GSM8K_MARKER_PATTERN.findall(candidate)
    non_empty_lines = [line for line in candidate.splitlines() if line.strip()]
    final_line = bool(
        non_empty_lines and GSM8K_STRICT_PATTERN.fullmatch(non_empty_lines[-1])
    )
    answer_only = GSM8K_STRICT_PATTERN.fullmatch(candidate) is not None

    if official_matches:
        value = normalize_number(official_matches[-1])
    else:
        number_matches = NUMBER_TOKEN_PATTERN.findall(candidate)
        if not number_matches:
            return ParsedAnswer(
                value=None,
                parse_success=False,
                answer_only_format=False,
                final_line_format=False,
            )
        value = normalize_number(number_matches[-1])

    return ParsedAnswer(
        value=value,
        parse_success=True,
        answer_only_format=answer_only,
        final_line_format=final_line,
    )
