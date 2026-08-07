from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from reasoning_lab.io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from reasoning_lab.preference_tokenization import trl_preference_token_ids
from reasoning_lab.data.stage4_retired_v1 import (
    RETIRED_V1_BLOOM_BITS,
    RETIRED_V1_BLOOM_HASHES,
    RETIRED_V1_BLOOM_SHA256,
    RETIRED_V1_SEMANTIC_COUNT,
    RETIRED_V1_SEMANTIC_SET_SHA256,
    retired_v1_semantic_maybe_contains,
)


GENERATOR_VERSION = "stage4_reasoning_v2"
ROLES = (
    "preference_train",
    "preference_validation",
    "preference_confirmation",
    "benchmark_validation",
    "benchmark_confirmation",
)
FAMILIES = (
    "arithmetic_chain",
    "exact_division",
    "linear_equation",
    "percentage_change",
    "ratio_allocation",
    "logic_code",
    "table_aggregation",
)
FAMILY_DOMAINS = {
    "arithmetic_chain": "mathematical",
    "exact_division": "mathematical",
    "linear_equation": "mathematical",
    "percentage_change": "mathematical",
    "ratio_allocation": "structured",
    "logic_code": "logical",
    "table_aggregation": "structured",
}
DEFAULT_ROLE_SIZES = {
    "preference_train": 128,
    "preference_validation": 48,
    "preference_confirmation": 48,
    "benchmark_validation": 96,
    "benchmark_confirmation": 128,
}

TokenCounter = Callable[[str], Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _set_hash(values: Sequence[str]) -> str:
    return _sha256_json(sorted(values))


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def _lit(value: Any) -> dict[str, Any]:
    return {"literal": value}


def _ref(name: str) -> dict[str, str]:
    return {"ref": name}


def _step(name: str, op: str, args: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return {"name": name, "op": op, "args": args, "label": label}


def _difficulty_range(difficulty: int) -> tuple[int, int]:
    if difficulty == 1:
        return 4, 18
    if difficulty == 2:
        return 15, 60
    if difficulty == 3:
        return 45, 180
    raise ValueError(f"Difficulty must be 1, 2, or 3; got {difficulty}")


def _build_arithmetic_chain(
    rng: random.Random, difficulty: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    low, high = _difficulty_range(difficulty)
    params = {
        "a": rng.randint(low, high),
        "b": rng.randint(2, 5 + 3 * difficulty),
        "c": rng.randint(low, high),
        "d": rng.randint(low // 2, high),
    }
    graph = [
        _step("s1", "mul", [_lit(params["a"]), _lit(params["b"])], "multiply"),
        _step("s2", "add", [_ref("s1"), _lit(params["c"])], "add the offset"),
        _step("s3", "sub", [_ref("s2"), _lit(params["d"])], "subtract the final value"),
    ]
    return params, graph


def _build_exact_division(
    rng: random.Random, difficulty: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    low, high = _difficulty_range(difficulty)
    divisor = rng.randint(2, 5 + 2 * difficulty)
    quotient = rng.randint(low, high)
    removed = rng.randint(1, high // 2 + 1)
    offset = rng.randint(1, 3 + 2 * difficulty)
    params = {
        "total": divisor * quotient + removed,
        "removed": removed,
        "divisor": divisor,
        "offset": offset,
    }
    graph = [
        _step("s1", "sub", [_lit(params["total"]), _lit(removed)], "remove the reserved amount"),
        _step("s2", "exact_div", [_ref("s1"), _lit(divisor)], "divide evenly"),
        _step("s3", "add", [_ref("s2"), _lit(offset)], "apply the final offset"),
    ]
    return params, graph


def _build_linear_equation(
    rng: random.Random, difficulty: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    low, high = _difficulty_range(difficulty)
    coefficient = rng.randint(2, 5 + 3 * difficulty)
    solution = rng.randint(low, high)
    intercept = rng.randint(1, high)
    rhs = coefficient * solution + intercept
    params = {
        "coefficient": coefficient,
        "intercept": intercept,
        "rhs": rhs,
    }
    graph = [
        _step("s1", "sub", [_lit(rhs), _lit(intercept)], "isolate the multiplied term"),
        _step("s2", "exact_div", [_ref("s1"), _lit(coefficient)], "solve for x"),
    ]
    return params, graph


def _build_percentage_change(
    rng: random.Random, difficulty: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    percent = rng.choice((5, 10, 12, 15, 20, 25, 30, 40, 50))
    unit_low, unit_high = ((2, 35), (30, 120), (110, 320))[difficulty - 1]
    base = 100 * rng.randint(unit_low, unit_high)
    direction = rng.choice(("increase", "decrease"))
    params = {"base": base, "percent": percent, "direction": direction}
    graph = [
        _step("s1", "mul", [_lit(base), _lit(percent)], "multiply by the percentage"),
        _step("s2", "exact_div", [_ref("s1"), _lit(100)], "find the percentage amount"),
        _step(
            "s3",
            "add" if direction == "increase" else "sub",
            [_lit(base), _ref("s2")],
            "apply the percentage change",
        ),
    ]
    return params, graph


def _build_ratio_allocation(
    rng: random.Random, difficulty: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parts = [rng.randint(1, 3 + difficulty) for _ in range(3)]
    unit = rng.randint(3 * difficulty, 12 * difficulty + 4)
    total = unit * sum(parts)
    target_index = rng.randrange(3)
    params = {"parts": parts, "total": total, "target_index": target_index}
    graph = [
        _step("s1", "add", [_lit(parts[0]), _lit(parts[1])], "add the first two ratio parts"),
        _step("s2", "add", [_ref("s1"), _lit(parts[2])], "find all ratio parts"),
        _step("s3", "exact_div", [_lit(total), _ref("s2")], "find one ratio unit"),
        _step("s4", "mul", [_ref("s3"), _lit(parts[target_index])], "find the requested share"),
    ]
    return params, graph


def _build_logic_code(
    rng: random.Random, difficulty: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, high = _difficulty_range(difficulty)
    first = rng.randint(1, high)
    second = rng.randint(1, high)
    parity_value = rng.randint(1, high)
    divisibility_value = rng.randint(1, high * 2)
    divisor = rng.choice((3, 4, 5, 6))
    params = {
        "first": first,
        "second": second,
        "parity_value": parity_value,
        "divisibility_value": divisibility_value,
        "divisor": divisor,
    }
    graph = [
        _step("s1", "gt", [_lit(first), _lit(second)], "test whether the first value is larger"),
        _step("s2", "is_even", [_lit(parity_value)], "test whether the parity value is even"),
        _step(
            "s3",
            "divisible",
            [_lit(divisibility_value), _lit(divisor)],
            "test divisibility",
        ),
        _step("s4", "mul", [_ref("s1"), _lit(100)], "encode the first truth value"),
        _step("s5", "mul", [_ref("s2"), _lit(10)], "encode the second truth value"),
        _step("s6", "add", [_ref("s4"), _ref("s5")], "combine the first two digits"),
        _step("s7", "add", [_ref("s6"), _ref("s3")], "append the final truth value"),
    ]
    return params, graph


def _build_table_aggregation(
    rng: random.Random, difficulty: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = ("A", "B", "C")
    row_count = 4 + 2 * difficulty
    rows = [
        {
            "group": rng.choice(groups),
            "points": rng.randint(3, 20 * difficulty + 5),
            "active": bool(rng.getrandbits(1)),
        }
        for _ in range(row_count)
    ]
    target = rng.choice(groups)
    # Guarantee that the requested filter has at least one visible match.
    rows[0] = {**rows[0], "group": target, "active": True}
    # Keep the last two operations semantically meaningful.  If adjustment and
    # penalty were equal, they would cancel and make the table-only subtotal a
    # shortcut to the answer.
    adjustment = rng.randint(2, 5 * difficulty + 2)
    penalty = rng.randint(1, adjustment - 1)
    params = {
        "rows": rows,
        "target_group": target,
        "adjustment": adjustment,
        "penalty": penalty,
    }
    graph = [
        _step(
            "s1",
            "filtered_sum",
            [_lit(rows), _lit(target)],
            "sum active rows in the target group",
        ),
        _step("s2", "add", [_ref("s1"), _lit(adjustment)], "add the adjustment"),
        _step("s3", "sub", [_ref("s2"), _lit(penalty)], "subtract the penalty"),
    ]
    return params, graph


_BUILDERS = {
    "arithmetic_chain": _build_arithmetic_chain,
    "exact_division": _build_exact_division,
    "linear_equation": _build_linear_equation,
    "percentage_change": _build_percentage_change,
    "ratio_allocation": _build_ratio_allocation,
    "logic_code": _build_logic_code,
    "table_aggregation": _build_table_aggregation,
}


def _graph_for(family: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rebuild a computation graph from task parameters, independently of stored traces."""

    if family == "arithmetic_chain":
        return [
            _step("s1", "mul", [_lit(params["a"]), _lit(params["b"])], "multiply"),
            _step("s2", "add", [_ref("s1"), _lit(params["c"])], "add the offset"),
            _step("s3", "sub", [_ref("s2"), _lit(params["d"])], "subtract the final value"),
        ]
    if family == "exact_division":
        return [
            _step("s1", "sub", [_lit(params["total"]), _lit(params["removed"])], "remove the reserved amount"),
            _step("s2", "exact_div", [_ref("s1"), _lit(params["divisor"])], "divide evenly"),
            _step("s3", "add", [_ref("s2"), _lit(params["offset"])], "apply the final offset"),
        ]
    if family == "linear_equation":
        return [
            _step("s1", "sub", [_lit(params["rhs"]), _lit(params["intercept"])], "isolate the multiplied term"),
            _step("s2", "exact_div", [_ref("s1"), _lit(params["coefficient"])], "solve for x"),
        ]
    if family == "percentage_change":
        return [
            _step("s1", "mul", [_lit(params["base"]), _lit(params["percent"])], "multiply by the percentage"),
            _step("s2", "exact_div", [_ref("s1"), _lit(100)], "find the percentage amount"),
            _step(
                "s3",
                "add" if params["direction"] == "increase" else "sub",
                [_lit(params["base"]), _ref("s2")],
                "apply the percentage change",
            ),
        ]
    if family == "ratio_allocation":
        parts = params["parts"]
        target_index = int(params["target_index"])
        return [
            _step("s1", "add", [_lit(parts[0]), _lit(parts[1])], "add the first two ratio parts"),
            _step("s2", "add", [_ref("s1"), _lit(parts[2])], "find all ratio parts"),
            _step("s3", "exact_div", [_lit(params["total"]), _ref("s2")], "find one ratio unit"),
            _step("s4", "mul", [_ref("s3"), _lit(parts[target_index])], "find the requested share"),
        ]
    if family == "logic_code":
        return [
            _step("s1", "gt", [_lit(params["first"]), _lit(params["second"])], "test whether the first value is larger"),
            _step("s2", "is_even", [_lit(params["parity_value"])], "test whether the parity value is even"),
            _step("s3", "divisible", [_lit(params["divisibility_value"]), _lit(params["divisor"])], "test divisibility"),
            _step("s4", "mul", [_ref("s1"), _lit(100)], "encode the first truth value"),
            _step("s5", "mul", [_ref("s2"), _lit(10)], "encode the second truth value"),
            _step("s6", "add", [_ref("s4"), _ref("s5")], "combine the first two digits"),
            _step("s7", "add", [_ref("s6"), _ref("s3")], "append the final truth value"),
        ]
    if family == "table_aggregation":
        return [
            _step("s1", "filtered_sum", [_lit(params["rows"]), _lit(params["target_group"])], "sum active rows in the target group"),
            _step("s2", "add", [_ref("s1"), _lit(params["adjustment"])], "add the adjustment"),
            _step("s3", "sub", [_ref("s2"), _lit(params["penalty"])], "subtract the penalty"),
        ]
    raise ValueError(f"Unknown Stage 4 family: {family}")


def _render_prompt(family: str, params: Mapping[str, Any], template_variant: int) -> str:
    suffixes = (
        "Show concise calculations, then end with exactly one line: FINAL_ANSWER=<integer>\n",
        "Reason step by step. The last line must be exactly FINAL_ANSWER=<integer>\n",
    )
    if template_variant not in (0, 1):
        raise ValueError(f"Unknown template variant: {template_variant}")

    if family == "arithmetic_chain":
        expression = f"(({params['a']} * {params['b']}) + {params['c']}) - {params['d']}"
        lead = (
            f"Evaluate the integer expression {expression}.\n"
            if template_variant == 0
            else f"Find the value of this chained calculation: {expression}.\n"
        )
    elif family == "exact_division":
        expression = (
            f"(({params['total']} - {params['removed']}) / {params['divisor']}) "
            f"+ {params['offset']}"
        )
        lead = (
            f"Evaluate {expression}; the division is exact.\n"
            if template_variant == 0
            else f"Compute this exact integer calculation: {expression}.\n"
        )
    elif family == "linear_equation":
        lead = (
            f"Solve for integer x: {params['coefficient']}x + {params['intercept']} = {params['rhs']}.\n"
            if template_variant == 0
            else f"What integer x satisfies {params['coefficient']} * x + {params['intercept']} = {params['rhs']}?\n"
        )
    elif family == "percentage_change":
        lead = (
            f"Apply a {params['percent']}% {params['direction']} to {params['base']}.\n"
            if template_variant == 0
            else f"A value starts at {params['base']} and then has a {params['percent']} percent {params['direction']}. Find the result.\n"
        )
    elif family == "ratio_allocation":
        target = int(params["target_index"]) + 1
        parts = ":".join(str(value) for value in params["parts"])
        lead = (
            f"Split {params['total']} in the ratio {parts}. Find share {target}.\n"
            if template_variant == 0
            else f"A total of {params['total']} is allocated in ratio {parts}. What is the value of part {target}?\n"
        )
    elif family == "logic_code":
        lead = (
            "Encode three truth values as 100*b1 + 10*b2 + b3, using 1 for true and 0 for false:\n"
            if template_variant == 0
            else "Evaluate the conditions below, let their 1/0 results be b1, b2, b3, and return 100*b1 + 10*b2 + b3:\n"
        )
        lead += (
            f"1) {params['first']} > {params['second']}\n"
            f"2) {params['parity_value']} is even\n"
            f"3) {params['divisibility_value']} is divisible by {params['divisor']}\n"
        )
    elif family == "table_aggregation":
        lines = [
            f"{index + 1}. group={row['group']}; points={row['points']}; active={str(row['active']).lower()}"
            for index, row in enumerate(params["rows"])
        ]
        lead = (
            "Use the records below. " if template_variant == 0 else "Inspect this structured table. "
        )
        lead += (
            f"Sum points for active rows in group {params['target_group']}, add {params['adjustment']}, "
            f"then subtract {params['penalty']}.\n" + "\n".join(lines) + "\n"
        )
    else:
        raise ValueError(f"Unknown Stage 4 family: {family}")
    return lead + suffixes[template_variant]


def _resolve_arg(argument: Mapping[str, Any], values: Mapping[str, int]) -> Any:
    if set(argument) == {"literal"}:
        return argument["literal"]
    if set(argument) == {"ref"}:
        name = str(argument["ref"])
        if name not in values:
            raise ValueError(f"Step references an unavailable value: {name}")
        return values[name]
    raise ValueError(f"Invalid graph argument: {argument}")


def _apply_op(op: str, inputs: list[Any]) -> int:
    if op in {"add", "sub", "mul", "exact_div", "gt"} and len(inputs) != 2:
        raise ValueError(f"Operation {op} requires two inputs")
    if op == "add":
        return int(inputs[0]) + int(inputs[1])
    if op == "sub":
        return int(inputs[0]) - int(inputs[1])
    if op == "mul":
        return int(inputs[0]) * int(inputs[1])
    if op == "exact_div":
        numerator, denominator = int(inputs[0]), int(inputs[1])
        if denominator == 0 or numerator % denominator:
            raise ValueError(f"Division is not exact: {numerator} / {denominator}")
        return numerator // denominator
    if op == "gt":
        return int(int(inputs[0]) > int(inputs[1]))
    if op == "is_even":
        if len(inputs) != 1:
            raise ValueError("Operation is_even requires one input")
        return int(int(inputs[0]) % 2 == 0)
    if op == "divisible":
        if len(inputs) != 2 or int(inputs[1]) == 0:
            raise ValueError("Operation divisible requires two inputs and a nonzero divisor")
        return int(int(inputs[0]) % int(inputs[1]) == 0)
    if op == "filtered_sum":
        if len(inputs) != 2 or not isinstance(inputs[0], list):
            raise ValueError("Operation filtered_sum requires records and a target group")
        target = str(inputs[1])
        return sum(
            int(row["points"])
            for row in inputs[0]
            if bool(row["active"]) and str(row["group"]) == target
        )
    raise ValueError(f"Unsupported Stage 4 operation: {op}")


def _materialize_steps(
    graph: Sequence[Mapping[str, Any]], *, override: tuple[str, int] | None = None
) -> list[dict[str, Any]]:
    values: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for spec in graph:
        name = str(spec["name"])
        if name in names:
            raise ValueError(f"Duplicate step name: {name}")
        names.add(name)
        inputs = [_resolve_arg(argument, values) for argument in spec["args"]]
        expected = _apply_op(str(spec["op"]), inputs)
        value = override[1] if override is not None and name == override[0] else expected
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Step {name} did not produce an integer")
        values[name] = value
        rows.append(
            {
                "name": name,
                "op": str(spec["op"]),
                "args": spec["args"],
                "label": str(spec["label"]),
                "inputs": inputs,
                "value": value,
                "corrupted": override is not None and name == override[0],
            }
        )
    if override is not None and override[0] not in names:
        raise ValueError(f"Unknown override step: {override[0]}")
    return rows


def _input_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _format_calculation(step: Mapping[str, Any]) -> str:
    op = str(step["op"])
    inputs = step["inputs"]
    value = int(step["value"])
    if op in {"add", "sub", "mul", "exact_div", "gt"}:
        symbol = {"add": "+", "sub": "-", "mul": "*", "exact_div": "/", "gt": ">"}[op]
        return f"{_input_text(inputs[0])} {symbol} {_input_text(inputs[1])} = {value}"
    if op == "is_even":
        return f"is_even({_input_text(inputs[0])}) = {value}"
    if op == "divisible":
        return f"divisible({_input_text(inputs[0])}, {_input_text(inputs[1])}) = {value}"
    if op == "filtered_sum":
        return f"filtered_sum(group={_input_text(inputs[1])}) = {value}"
    raise ValueError(f"Cannot render operation: {op}")


def _render_trace(steps: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"Step {index + 1} ({step['label']}): {_format_calculation(step)}"
        for index, step in enumerate(steps)
    ]
    lines.append(f"FINAL_ANSWER={int(steps[-1]['value'])}")
    return "\n".join(lines)


def _plausible_error_bound(value: int, op: str) -> int:
    if op in {"gt", "is_even", "divisible"} and value in (0, 1):
        return 1
    # A local slip may be a small absolute error for small intermediates or a
    # small relative error for large products.  The hard cap prevents the old
    # pathological examples (for example turning a percentage amount negative
    # merely to preserve string width).
    return max(4, min(256, int(abs(value) * 0.10) + 1))


def _corruption_values(value: int, op: str) -> list[int]:
    if op in {"gt", "is_even", "divisible"} and value in (0, 1):
        return [1 - value]
    bound = _plausible_error_bound(value, op)
    deltas: list[int] = []
    for amount in range(1, bound + 1):
        deltas.extend((amount, -amount))
    # Prefer values with the same printed width to limit superficial length cues.
    candidates = [value + delta for delta in deltas]
    return sorted(
        candidates,
        key=lambda candidate: (
            len(str(candidate)) != len(str(value)),
            abs(candidate - value),
            candidate,
        ),
    )


def _valid_corrupted_value(
    family: str,
    spec: Mapping[str, Any],
    original: int,
    wrong_value: int,
) -> bool:
    op = str(spec["op"])
    name = str(spec["name"])
    if wrong_value == original:
        return False
    if abs(wrong_value - original) > _plausible_error_bound(original, op):
        return False
    # Logic corruptions are genuine predicate mistakes, not malformed decimal
    # place-value arithmetic such as 100 -> 99.
    if family == "logic_code":
        # The first bit propagates through three 0/100-valued renderings and
        # creates a >2-token cue with the project tokenizer.  The parity and
        # divisibility predicates provide two independent, length-safe error
        # locations.
        return (
            name in {"s2", "s3"}
            and op in {"is_even", "divisible"}
            and wrong_value in (0, 1)
        )
    # Every other generator has non-negative canonical intermediates.  Retain
    # that domain constraint in rejected traces as well.
    if wrong_value < 0:
        return False
    if family == "ratio_allocation" and name == "s3" and wrong_value <= 0:
        return False
    # Percentage products must remain an integer number of percentage points so
    # that the following exact division is still interpretable.
    if family == "percentage_change" and name == "s1" and wrong_value % 100:
        return False
    return True


def _build_rejected(
    family: str,
    graph: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
    stable_key: str,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    canonical_answer = int(canonical[-1]["value"])
    digest = hashlib.sha256(f"stage4-error|{stable_key}".encode("utf-8")).digest()
    desired_direction = "higher" if digest[0] & 1 else "lower"
    candidate_indexes = list(range(len(graph) - 1))
    canonical_text = _render_trace(canonical)
    best_by_step: dict[
        int, tuple[tuple[int, int, int, int], list[dict[str, Any]]]
    ] = {}
    for index in candidate_indexes:
        spec = graph[index]
        original = int(canonical[index]["value"])
        for wrong_value in _corruption_values(original, str(spec["op"])):
            if not _valid_corrupted_value(family, spec, original, wrong_value):
                continue
            try:
                rejected = _materialize_steps(
                    graph, override=(str(spec["name"]), wrong_value)
                )
            except ValueError:
                continue
            final_delta = int(rejected[-1]["value"]) - canonical_answer
            if final_delta == 0:
                continue
            direction = "higher" if final_delta > 0 else "lower"
            if direction != desired_direction:
                continue
            rejected_text = _render_trace(rejected)
            character_delta = abs(len(rejected_text) - len(canonical_text))
            score = (
                character_delta,
                len(str(wrong_value)) != len(str(original)),
                abs(final_delta),
                abs(wrong_value - original),
            )
            current = best_by_step.get(index)
            if current is None or score < current[0]:
                best_by_step[index] = (score, rejected)
    if not best_by_step:
        raise ValueError("Could not create a propagated single-error rejected trace")
    eligible_indexes = sorted(best_by_step)
    selected_offset = int.from_bytes(digest[1:9], "big") % len(eligible_indexes)
    selected_index = eligible_indexes[selected_offset]
    rejected = best_by_step[selected_index][1]
    corrupted_step = str(graph[selected_index]["name"])
    original = int(canonical[selected_index]["value"])
    wrong_value = int(rejected[selected_index]["value"])
    final_delta = int(rejected[-1]["value"]) - canonical_answer
    diagnostics = {
        "error_type": "propagated_intermediate_value",
        "error_direction": "higher" if final_delta > 0 else "lower",
        "error_step_index": selected_index,
        "eligible_error_step_indexes": eligible_indexes,
        "eligible_corrupted_steps": [str(graph[index]["name"]) for index in eligible_indexes],
        "local_error_delta": wrong_value - original,
        "final_error_delta": final_delta,
        "local_error_bound": _plausible_error_bound(original, str(graph[selected_index]["op"])),
    }
    return rejected, corrupted_step, diagnostics


def _task_identity(
    *, family: str, difficulty: int, template_variant: int, params: Mapping[str, Any], generator_version: str
) -> dict[str, Any]:
    return {
        "generator_version": generator_version,
        "family": family,
        "difficulty": difficulty,
        "template_variant": template_variant,
        "params": params,
    }


def _semantic_identity(
    *, family: str, template_variant: int, params: Mapping[str, Any]
) -> dict[str, Any]:
    """Identity of the user-visible task, excluding provenance-only difficulty."""

    return {
        "family": family,
        "template_variant": template_variant,
        "params": params,
    }


def _candidate_rows(
    seed: int,
    generator_version: str,
    *,
    reserved_prompts: set[str] | None = None,
):
    rng = random.Random(seed)
    candidate_index = 0
    seen_semantic_hashes: set[str] = set()
    seen_prompt_hashes: set[str] = set()
    seen_normalized_prompt_hashes: set[str] = set()
    reserved_exact = set(reserved_prompts or ())
    reserved_normalized = {_normalize_prompt(value) for value in reserved_exact}
    while True:
        family = FAMILIES[candidate_index % len(FAMILIES)]
        difficulty = 1 + ((candidate_index // len(FAMILIES)) % 3)
        candidate_index += 1
        params, graph = _BUILDERS[family](rng, difficulty)
        template_variant = rng.randrange(2)
        identity = _task_identity(
            family=family,
            difficulty=difficulty,
            template_variant=template_variant,
            params=params,
            generator_version=generator_version,
        )
        task_sha256 = _sha256_json(identity)
        semantic_sha256 = _sha256_json(
            _semantic_identity(
                family=family,
                template_variant=template_variant,
                params=params,
            )
        )
        if generator_version.startswith(
            "stage4_reasoning_v2"
        ) and retired_v1_semantic_maybe_contains(semantic_sha256):
            continue
        prompt = _render_prompt(family, params, template_variant)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        normalized_prompt = _normalize_prompt(prompt)
        normalized_prompt_sha256 = hashlib.sha256(
            normalized_prompt.encode("utf-8")
        ).hexdigest()
        if (
            semantic_sha256 in seen_semantic_hashes
            or prompt_sha256 in seen_prompt_hashes
            or normalized_prompt_sha256 in seen_normalized_prompt_hashes
            or prompt in reserved_exact
            or normalized_prompt in reserved_normalized
        ):
            continue
        seen_semantic_hashes.add(semantic_sha256)
        seen_prompt_hashes.add(prompt_sha256)
        seen_normalized_prompt_hashes.add(normalized_prompt_sha256)
        canonical = _materialize_steps(graph)
        try:
            rejected, corrupted_step, error_diagnostics = _build_rejected(
                family, graph, canonical, task_sha256
            )
        except ValueError:
            # Some arithmetic instances cannot satisfy exact propagation,
            # plausibility, direction, and length constraints simultaneously.
            # Deterministically skip them without weakening any constraint.
            continue
        chosen_text = _render_trace(canonical)
        rejected_text = _render_trace(rejected)
        yield {
            "id": f"s4r_{task_sha256[:20]}",
            "task_sha256": task_sha256,
            "semantic_sha256": semantic_sha256,
            "prompt_sha256": prompt_sha256,
            "normalized_prompt_sha256": normalized_prompt_sha256,
            "family": family,
            "domain": FAMILY_DOMAINS[family],
            "difficulty": difficulty,
            "template_id": f"{family}_t{template_variant + 1}",
            "prompt": prompt,
            "answer": int(canonical[-1]["value"]),
            "output_contract": "reasoning_then_answer",
            "chosen": chosen_text,
            "rejected": rejected_text,
            "completion": chosen_text,
            "params": params,
            "canonical_steps": canonical,
            "rejected_steps": rejected,
            "corrupted_step": corrupted_step,
            **error_diagnostics,
            "source_dataset": "stage4_reasoning_synthetic",
            "generator_version": generator_version,
            "pair_length": {
                "chosen_characters": len(chosen_text),
                "rejected_characters": len(rejected_text),
                "character_delta": abs(len(chosen_text) - len(rejected_text)),
            },
        }


def _assigned_role(row_id: str, seed: int, generator_version: str) -> str:
    digest = hashlib.sha256(
        f"{generator_version}|{seed}|role|{row_id}".encode("utf-8")
    ).digest()
    return ROLES[int.from_bytes(digest[:8], "big") % len(ROLES)]


def _expected_error_direction(
    role: str,
    family: str,
    family_slot: int,
    seed: int,
    generator_version: str,
) -> str:
    digest = hashlib.sha256(
        f"{generator_version}|{seed}|direction-balance|{role}|{family}".encode(
            "utf-8"
        )
    ).digest()
    start_higher = bool(digest[0] & 1)
    higher = start_higher if family_slot % 2 == 0 else not start_higher
    return "higher" if higher else "lower"


def _validate_pair_length(
    row: Mapping[str, Any],
    *,
    tokenizer: TokenCounter | None,
    max_pair_token_delta: int | None,
    max_completion_tokens: int | None,
) -> dict[str, int] | None:
    chosen = str(row["chosen"])
    rejected = str(row["rejected"])
    if chosen.count("\nStep ") != rejected.count("\nStep "):
        raise ValueError(f"Preference pair has unequal reasoning-step counts: {row['id']}")
    lengths = row.get("pair_length")
    expected_lengths = {
        "chosen_characters": len(chosen),
        "rejected_characters": len(rejected),
        "character_delta": abs(len(chosen) - len(rejected)),
    }
    if lengths != expected_lengths:
        raise ValueError(f"Character-length metadata mismatch: {row['id']}")
    if tokenizer is None:
        return None
    _, chosen_ids, _ = trl_preference_token_ids(
        tokenizer,
        prompt=str(row["prompt"]),
        completion=chosen,
    )
    _, rejected_ids, _ = trl_preference_token_ids(
        tokenizer,
        prompt=str(row["prompt"]),
        completion=rejected,
    )
    chosen_tokens = len(chosen_ids)
    rejected_tokens = len(rejected_ids)
    token_delta = abs(chosen_tokens - rejected_tokens)
    if max_pair_token_delta is not None and token_delta > max_pair_token_delta:
        raise ValueError(
            f"Preference token-length delta {token_delta} exceeds {max_pair_token_delta}: {row['id']}"
        )
    if max_completion_tokens is not None and max(chosen_tokens, rejected_tokens) > max_completion_tokens:
        raise ValueError(
            f"Preference completion exceeds {max_completion_tokens} tokens: {row['id']}"
        )
    return {
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
        "token_delta": token_delta,
    }


def validate_stage4_row(
    row: Mapping[str, Any],
    *,
    tokenizer: TokenCounter | None = None,
    max_pair_token_delta: int | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, int] | None:
    """Verify identity, prompt, labels, and the single-error preference construction."""

    required = {
        "id",
        "task_sha256",
        "semantic_sha256",
        "prompt_sha256",
        "normalized_prompt_sha256",
        "role",
        "family",
        "domain",
        "difficulty",
        "template_id",
        "prompt",
        "answer",
        "output_contract",
        "chosen",
        "rejected",
        "completion",
        "params",
        "canonical_steps",
        "rejected_steps",
        "corrupted_step",
        "error_type",
        "error_direction",
        "error_step_index",
        "eligible_error_step_indexes",
        "eligible_corrupted_steps",
        "local_error_delta",
        "final_error_delta",
        "local_error_bound",
        "source_dataset",
        "generator_version",
        "allocation_seed",
        "pair_length",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Stage 4 row is missing fields: {sorted(missing)}")
    family = str(row["family"])
    if family not in FAMILIES:
        raise ValueError(f"Unknown Stage 4 family: {family}")
    if not isinstance(row["difficulty"], int) or isinstance(row["difficulty"], bool):
        raise ValueError(f"Difficulty must be an integer: {row['id']}")
    difficulty = row["difficulty"]
    if difficulty not in (1, 2, 3):
        raise ValueError(f"Invalid difficulty: {difficulty}")
    if row["domain"] != FAMILY_DOMAINS[family]:
        raise ValueError(f"Domain mismatch: {row['id']}")
    if row["role"] not in ROLES:
        raise ValueError(f"Unknown Stage 4 role: {row['role']}")
    if not isinstance(row["allocation_seed"], int) or isinstance(row["allocation_seed"], bool):
        raise ValueError(f"Allocation seed must be an integer: {row['id']}")
    if row["role"] != _assigned_role(
        str(row["id"]), row["allocation_seed"], str(row["generator_version"])
    ):
        raise ValueError(f"Stable role assignment mismatch: {row['id']}")
    if row["output_contract"] != "reasoning_then_answer":
        raise ValueError(f"Unexpected output contract: {row['id']}")
    if row["source_dataset"] != "stage4_reasoning_synthetic":
        raise ValueError(f"Unexpected source dataset: {row['id']}")
    if not isinstance(row["params"], Mapping):
        raise ValueError(f"Task parameters must be a mapping: {row['id']}")

    template_prefix = f"{family}_t"
    if not str(row["template_id"]).startswith(template_prefix):
        raise ValueError(f"Template/family mismatch: {row['id']}")
    try:
        template_variant = int(str(row["template_id"])[len(template_prefix) :]) - 1
    except ValueError as exc:
        raise ValueError(f"Invalid template id: {row['template_id']}") from exc

    identity = _task_identity(
        family=family,
        difficulty=difficulty,
        template_variant=template_variant,
        params=row["params"],
        generator_version=str(row["generator_version"]),
    )
    expected_task_hash = _sha256_json(identity)
    if row["task_sha256"] != expected_task_hash or row["id"] != f"s4r_{expected_task_hash[:20]}":
        raise ValueError(f"Stable task identity mismatch: {row['id']}")
    expected_prompt = _render_prompt(family, row["params"], template_variant)
    if row["prompt"] != expected_prompt:
        raise ValueError(f"Prompt does not match task parameters: {row['id']}")
    expected_semantic_hash = _sha256_json(
        _semantic_identity(
            family=family,
            template_variant=template_variant,
            params=row["params"],
        )
    )
    if row["semantic_sha256"] != expected_semantic_hash:
        raise ValueError(f"Stable semantic identity mismatch: {row['id']}")
    expected_prompt_hash = hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
    if row["prompt_sha256"] != expected_prompt_hash:
        raise ValueError(f"Stable prompt identity mismatch: {row['id']}")
    expected_normalized_prompt_hash = hashlib.sha256(
        _normalize_prompt(expected_prompt).encode("utf-8")
    ).hexdigest()
    if row["normalized_prompt_sha256"] != expected_normalized_prompt_hash:
        raise ValueError(f"Stable normalized prompt identity mismatch: {row['id']}")
    if family == "table_aggregation" and int(row["params"]["adjustment"]) == int(
        row["params"]["penalty"]
    ):
        raise ValueError(f"Table adjustment and penalty must not cancel: {row['id']}")

    graph = _graph_for(family, row["params"])
    canonical = _materialize_steps(graph)
    if row["canonical_steps"] != canonical:
        raise ValueError(f"Canonical computation trace mismatch: {row['id']}")
    answer = int(canonical[-1]["value"])
    if (
        not isinstance(row["answer"], int)
        or isinstance(row["answer"], bool)
        or row["answer"] != answer
    ):
        raise ValueError(f"Incorrect integer answer: {row['id']}")
    chosen = _render_trace(canonical)
    if row["chosen"] != chosen or row.get("completion") != chosen:
        raise ValueError(f"Chosen text is not the canonical trace: {row['id']}")

    rejected = row["rejected_steps"]
    if not isinstance(rejected, list) or len(rejected) != len(graph):
        raise ValueError(f"Rejected trace has the wrong number of steps: {row['id']}")
    values: dict[str, int] = {}
    local_errors: list[str] = []
    corruption_markers: list[str] = []
    for spec, stored in zip(graph, rejected, strict=True):
        name = str(spec["name"])
        inputs = [_resolve_arg(argument, values) for argument in spec["args"]]
        expected = _apply_op(str(spec["op"]), inputs)
        if stored.get("name") != name or stored.get("op") != spec["op"]:
            raise ValueError(f"Rejected graph structure mismatch: {row['id']}")
        if stored.get("args") != spec["args"] or stored.get("label") != spec["label"]:
            raise ValueError(f"Rejected graph metadata mismatch: {row['id']}")
        if stored.get("inputs") != inputs:
            raise ValueError(f"Rejected propagation inputs mismatch: {row['id']}")
        value = stored.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Rejected step is not integer-valued: {row['id']}")
        if value != expected:
            local_errors.append(name)
        if stored.get("corrupted") is True:
            corruption_markers.append(name)
        values[name] = value
    expected_corruption = str(row["corrupted_step"])
    if local_errors != [expected_corruption] or corruption_markers != [expected_corruption]:
        raise ValueError(f"Rejected trace must contain exactly one declared local error: {row['id']}")
    if expected_corruption == str(graph[-1]["name"]):
        raise ValueError(f"The rejected error must be an intermediate step: {row['id']}")
    if int(rejected[-1]["value"]) == answer:
        raise ValueError(f"Rejected error did not propagate to the final answer: {row['id']}")
    if row["rejected"] != _render_trace(rejected):
        raise ValueError(f"Rejected text does not match its computation trace: {row['id']}")
    if row["chosen"] == row["rejected"]:
        raise ValueError(f"Chosen and rejected completions are identical: {row['id']}")

    expected_rejected, expected_step, expected_diagnostics = _build_rejected(
        family, graph, canonical, expected_task_hash
    )
    if rejected != expected_rejected or expected_corruption != expected_step:
        raise ValueError(f"Rejected trace is not the deterministic bounded corruption: {row['id']}")
    for key, expected in expected_diagnostics.items():
        if row.get(key) != expected:
            raise ValueError(f"Rejected-error metadata mismatch for {key}: {row['id']}")

    return _validate_pair_length(
        row,
        tokenizer=tokenizer,
        max_pair_token_delta=max_pair_token_delta,
        max_completion_tokens=max_completion_tokens,
    )


def validate_stage4_roles(
    rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    tokenizer: TokenCounter | None = None,
    max_pair_token_delta: int | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    if set(rows_by_role) != set(ROLES):
        raise ValueError(f"Stage 4 roles must be exactly {ROLES}")
    seen_ids: set[str] = set()
    seen_tasks: set[str] = set()
    seen_semantics: set[str] = set()
    seen_prompts: set[str] = set()
    seen_normalized_prompts: set[str] = set()
    token_stats: list[dict[str, int]] = []
    direction_counts: Counter[str] = Counter()
    step_counts: dict[str, Counter[str]] = {
        family: Counter() for family in FAMILIES
    }
    final_delta_counts: dict[str, Counter[int]] = {
        family: Counter() for family in FAMILIES
    }
    eligible_steps_by_family: dict[str, set[str]] = {
        family: set() for family in FAMILIES
    }
    for role in ROLES:
        rows = rows_by_role[role]
        role_ids: set[str] = set()
        role_tasks: set[str] = set()
        role_semantics: set[str] = set()
        role_prompts: set[str] = set()
        role_normalized_prompts: set[str] = set()
        role_direction_counts: Counter[str] = Counter()
        role_family_slots: Counter[str] = Counter()
        role_family_directions: dict[str, Counter[str]] = {
            family: Counter() for family in FAMILIES
        }
        for row in rows:
            if row.get("role") != role:
                raise ValueError(f"Row role mismatch in {role}: {row.get('id')}")
            stats = validate_stage4_row(
                row,
                tokenizer=tokenizer,
                max_pair_token_delta=max_pair_token_delta,
                max_completion_tokens=max_completion_tokens,
            )
            if stats is not None:
                token_stats.append(stats)
            row_id = str(row["id"])
            task_hash = str(row["task_sha256"])
            semantic_hash = str(row["semantic_sha256"])
            prompt_hash = str(row["prompt_sha256"])
            normalized_prompt_hash = str(row["normalized_prompt_sha256"])
            family = str(row["family"])
            if (
                row_id in role_ids
                or task_hash in role_tasks
                or semantic_hash in role_semantics
                or prompt_hash in role_prompts
                or normalized_prompt_hash in role_normalized_prompts
            ):
                raise ValueError(f"Duplicate task inside {role}: {row_id}")
            expected_direction = _expected_error_direction(
                role,
                family,
                role_family_slots[family],
                int(row["allocation_seed"]),
                str(row["generator_version"]),
            )
            if row["error_direction"] != expected_direction:
                raise ValueError(f"Rejected-answer direction schedule mismatch: {row_id}")
            role_direction_counts[str(row["error_direction"])] += 1
            role_family_directions[family][str(row["error_direction"])] += 1
            role_family_slots[family] += 1
            direction_counts[str(row["error_direction"])] += 1
            step_counts[family][str(row["corrupted_step"])] += 1
            final_delta_counts[family][int(row["final_error_delta"])] += 1
            eligible_steps_by_family[family].update(
                str(value) for value in row["eligible_corrupted_steps"]
            )
            role_ids.add(row_id)
            role_tasks.add(task_hash)
            role_semantics.add(semantic_hash)
            role_prompts.add(prompt_hash)
            role_normalized_prompts.add(normalized_prompt_hash)
        if abs(role_direction_counts["higher"] - role_direction_counts["lower"]) > len(
            FAMILIES
        ):
            raise ValueError(f"Rejected-answer directions are imbalanced in {role}")
        for family, counts in role_family_directions.items():
            if abs(counts["higher"] - counts["lower"]) > 1:
                raise ValueError(
                    f"Rejected-answer directions are imbalanced in {role}/{family}"
                )
        if (
            role_ids & seen_ids
            or role_tasks & seen_tasks
            or role_semantics & seen_semantics
            or role_prompts & seen_prompts
            or role_normalized_prompts & seen_normalized_prompts
        ):
            raise ValueError(f"Stage 4 role overlap detected in {role}")
        seen_ids.update(role_ids)
        seen_tasks.update(role_tasks)
        seen_semantics.update(role_semantics)
        seen_prompts.update(role_prompts)
        seen_normalized_prompts.update(role_normalized_prompts)
    for family in FAMILIES:
        family_total = sum(step_counts[family].values())
        if family_total == 0:
            continue
        if (
            family_total >= 12
            and len(eligible_steps_by_family[family]) >= 2
            and len(step_counts[family]) < 2
        ):
            raise ValueError(f"Rejected-error positions did not vary for {family}")
        if family_total >= 12 and len(eligible_steps_by_family[family]) >= 2:
            shares = [count / family_total for count in step_counts[family].values()]
            if max(shares) > 0.80 or min(shares) < 0.15:
                raise ValueError(
                    f"Rejected-error position distribution is imbalanced for {family}"
                )
        dominant_delta_share = max(final_delta_counts[family].values()) / family_total
        if family_total >= 12 and dominant_delta_share > 0.60:
            raise ValueError(f"One signed final-error delta dominates {family}")
    direction_shortcut_accuracy = (
        max(direction_counts.values()) / len(seen_ids) if seen_ids else 0.0
    )
    if len(seen_ids) >= 64 and direction_shortcut_accuracy > 0.60:
        raise ValueError("Final-answer direction is a preference shortcut")
    result: dict[str, Any] = {
        "total_rows": len(seen_ids),
        "unique_ids": len(seen_ids),
        "unique_task_hashes": len(seen_tasks),
        "unique_semantic_hashes": len(seen_semantics),
        "unique_prompt_hashes": len(seen_prompts),
        "unique_normalized_prompt_hashes": len(seen_normalized_prompts),
        "overlap_count": 0,
        "rejected_error_diagnostics": {
            "directions": dict(sorted(direction_counts.items())),
            "direction_imbalance": abs(
                direction_counts["higher"] - direction_counts["lower"]
            ),
            "steps_by_family": {
                family: dict(sorted(step_counts[family].items()))
                for family in FAMILIES
            },
            "signed_final_deltas_by_family": {
                family: {
                    str(delta): count
                    for delta, count in sorted(final_delta_counts[family].items())
                }
                for family in FAMILIES
            },
        },
        "completion_shortcut_audit": {
            "maximum_allowed_accuracy": 0.60,
            "prefer_higher_or_lower_final_answer_accuracy": direction_shortcut_accuracy,
            "maximum_signed_delta_share_by_family": max(
                (
                    max(counts.values()) / sum(counts.values())
                    for counts in final_delta_counts.values()
                    if counts
                ),
                default=0.0,
            ),
        },
    }
    if token_stats:
        ties = sum(
            item["chosen_tokens"] == item["rejected_tokens"] for item in token_stats
        )
        length_shortcut_accuracy = max(
            sum(item["chosen_tokens"] > item["rejected_tokens"] for item in token_stats)
            + 0.5 * ties,
            sum(item["rejected_tokens"] > item["chosen_tokens"] for item in token_stats)
            + 0.5 * ties,
        ) / len(token_stats)
        if len(token_stats) >= 64 and length_shortcut_accuracy > 0.60:
            raise ValueError("Completion length is a preference shortcut")
        result["completion_shortcut_audit"][
            "prefer_longer_or_shorter_completion_accuracy"
        ] = length_shortcut_accuracy
        result["token_lengths"] = {
            "maximum_completion_tokens": max(
                max(item["chosen_tokens"], item["rejected_tokens"]) for item in token_stats
            ),
            "maximum_pair_delta": max(item["token_delta"] for item in token_stats),
            "mean_chosen_minus_rejected": sum(
                item["chosen_tokens"] - item["rejected_tokens"] for item in token_stats
            )
            / len(token_stats),
            "chosen_longer_count": sum(
                item["chosen_tokens"] > item["rejected_tokens"] for item in token_stats
            ),
            "rejected_longer_count": sum(
                item["rejected_tokens"] > item["chosen_tokens"] for item in token_stats
            ),
        }
    return result


def build_stage4_roles(
    *,
    sizes: Mapping[str, int] | None = None,
    seed: int = 4704,
    generator_version: str = GENERATOR_VERSION,
    tokenizer: TokenCounter | None = None,
    max_pair_token_delta: int | None = None,
    max_completion_tokens: int | None = None,
    reserved_prompts: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build fixed hash-partitioned role prefixes from one deterministic stream."""

    resolved_sizes = dict(DEFAULT_ROLE_SIZES if sizes is None else sizes)
    if set(resolved_sizes) != set(ROLES):
        raise ValueError(f"Sizes must name exactly these roles: {ROLES}")
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0
        for size in resolved_sizes.values()
    ):
        raise ValueError("Stage 4 role sizes must be non-negative integers")
    rows_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    family_slots_by_role: dict[str, Counter[str]] = {
        role: Counter() for role in ROLES
    }
    if not any(resolved_sizes.values()):
        return rows_by_role

    stream = _candidate_rows(
        seed=seed,
        generator_version=generator_version,
        reserved_prompts=reserved_prompts,
    )
    while any(len(rows_by_role[role]) < resolved_sizes[role] for role in ROLES):
        row = next(stream)
        role = _assigned_role(str(row["id"]), seed, generator_version)
        if len(rows_by_role[role]) < resolved_sizes[role]:
            family = str(row["family"])
            family_slot = family_slots_by_role[role][family]
            if row["error_direction"] != _expected_error_direction(
                role, family, family_slot, seed, generator_version
            ):
                continue
            assigned = {**row, "role": role, "allocation_seed": seed}
            try:
                validate_stage4_row(
                    assigned,
                    tokenizer=tokenizer,
                    max_pair_token_delta=max_pair_token_delta,
                    max_completion_tokens=max_completion_tokens,
                )
            except ValueError as error:
                # Length is a corpus construction gate, not a reason to weaken
                # the policy. Deterministically skip the candidate and keep
                # drawing from the frozen stream until the role quota is full.
                message = str(error)
                if tokenizer is not None and (
                    message.startswith("Preference token-length delta")
                    or message.startswith("Preference completion exceeds")
                ):
                    continue
                raise
            rows_by_role[role].append(assigned)
            family_slots_by_role[role][family] += 1
    validate_stage4_roles(
        rows_by_role,
        tokenizer=tokenizer,
        max_pair_token_delta=max_pair_token_delta,
        max_completion_tokens=max_completion_tokens,
    )
    return rows_by_role


def _config_values(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "stage4_reasoning" in config:
        nested = config["stage4_reasoning"]
        if not isinstance(nested, Mapping):
            raise ValueError("stage4_reasoning config must be a mapping")
        return nested
    return config


def generate_stage4_dataset(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    tokenizer: TokenCounter | None = None,
    reserved_prompts: set[str] | None = None,
) -> dict[str, Any]:
    values = _config_values(config)
    sizes = {
        role: int(values.get(f"{role}_size", DEFAULT_ROLE_SIZES[role])) for role in ROLES
    }
    seed = int(values.get("seed", 4704))
    generator_version = str(values.get("generator_version", GENERATOR_VERSION))
    max_pair_token_delta_value = values.get("max_pair_token_delta")
    max_completion_tokens_value = values.get("max_completion_tokens")
    max_pair_token_delta = (
        int(max_pair_token_delta_value) if max_pair_token_delta_value is not None else None
    )
    max_completion_tokens = (
        int(max_completion_tokens_value) if max_completion_tokens_value is not None else None
    )
    if max_pair_token_delta is not None and max_pair_token_delta < 0:
        raise ValueError("max_pair_token_delta must be non-negative")
    if max_completion_tokens is not None and max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive")

    rows_by_role = build_stage4_roles(
        sizes=sizes,
        seed=seed,
        generator_version=generator_version,
        tokenizer=tokenizer,
        max_pair_token_delta=max_pair_token_delta,
        max_completion_tokens=max_completion_tokens,
        reserved_prompts=reserved_prompts,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    role_manifest: dict[str, Any] = {}
    for role in ROLES:
        rows = rows_by_role[role]
        path = destination / f"{role}.jsonl"
        write_jsonl(path, rows)
        role_manifest[role] = {
            # Keep the manifest portable and byte-stable across checkout roots.
            "path": path.name,
            "count": len(rows),
            "sha256": sha256_file(path),
            "row_id_set_sha256": _set_hash([str(row["id"]) for row in rows]),
            "task_set_sha256": _set_hash([str(row["task_sha256"]) for row in rows]),
            "semantic_set_sha256": _set_hash(
                [str(row["semantic_sha256"]) for row in rows]
            ),
            "prompt_set_sha256": _set_hash(
                [str(row["prompt_sha256"]) for row in rows]
            ),
            "normalized_prompt_set_sha256": _set_hash(
                [str(row["normalized_prompt_sha256"]) for row in rows]
            ),
            "families": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
            "domains": dict(sorted(Counter(str(row["domain"]) for row in rows).items())),
            "difficulties": dict(
                sorted(Counter(str(row["difficulty"]) for row in rows).items())
            ),
            "error_directions": dict(
                sorted(Counter(str(row["error_direction"]) for row in rows).items())
            ),
            "error_steps": dict(
                sorted(Counter(str(row["corrupted_step"]) for row in rows).items())
            ),
        }
    integrity = validate_stage4_roles(
        rows_by_role,
        tokenizer=tokenizer,
        max_pair_token_delta=max_pair_token_delta,
        max_completion_tokens=max_completion_tokens,
    )
    manifest = {
        "protocol": "stage4_reasoning_preferences_v2",
        "generator_version": generator_version,
        "seed": seed,
        "roles": role_manifest,
        "integrity": integrity,
        "length_policy": {
            "tokenizer_checked": tokenizer is not None,
            "max_pair_token_delta": max_pair_token_delta,
            "max_completion_tokens": max_completion_tokens,
        },
        "reserved_prompt_policy": {
            "count": len(set(reserved_prompts or ())),
            "exact_set_sha256": _set_hash(list(set(reserved_prompts or ()))),
            "normalized_set_sha256": _set_hash(
                [_normalize_prompt(value) for value in set(reserved_prompts or ())]
            ),
        },
        "retired_v1_semantic_policy": {
            "enabled": generator_version.startswith("stage4_reasoning_v2"),
            "no_false_negatives": True,
            "possible_false_positives_only_skip_additional_candidates": True,
            "source_semantic_count": RETIRED_V1_SEMANTIC_COUNT,
            "source_semantic_set_sha256": RETIRED_V1_SEMANTIC_SET_SHA256,
            "bloom_bits": RETIRED_V1_BLOOM_BITS,
            "bloom_hashes": RETIRED_V1_BLOOM_HASHES,
            "bloom_sha256": RETIRED_V1_BLOOM_SHA256,
        },
    }
    write_json(destination / "dataset_manifest.json", manifest)
    return manifest


def validate_stage4_dataset(
    data_dir: str | Path,
    *,
    tokenizer: TokenCounter | None = None,
    max_pair_token_delta: int | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    root = Path(data_dir)
    manifest = read_json(root / "dataset_manifest.json")
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in ROLES:
        path = root / f"{role}.jsonl"
        expected = manifest["roles"][role]
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Frozen Stage 4 file hash mismatch: {role}")
        rows = read_jsonl(path)
        if len(rows) != int(expected["count"]):
            raise ValueError(f"Frozen Stage 4 row count mismatch: {role}")
        if _set_hash([str(row["id"]) for row in rows]) != expected["row_id_set_sha256"]:
            raise ValueError(f"Frozen Stage 4 row identity mismatch: {role}")
        if _set_hash([str(row["task_sha256"]) for row in rows]) != expected["task_set_sha256"]:
            raise ValueError(f"Frozen Stage 4 task identity mismatch: {role}")
        if _set_hash([str(row["semantic_sha256"]) for row in rows]) != expected[
            "semantic_set_sha256"
        ]:
            raise ValueError(f"Frozen Stage 4 semantic identity mismatch: {role}")
        if _set_hash([str(row["prompt_sha256"]) for row in rows]) != expected[
            "prompt_set_sha256"
        ]:
            raise ValueError(f"Frozen Stage 4 prompt identity mismatch: {role}")
        if _set_hash([str(row["normalized_prompt_sha256"]) for row in rows]) != expected[
            "normalized_prompt_set_sha256"
        ]:
            raise ValueError(f"Frozen Stage 4 normalized prompt identity mismatch: {role}")
        rows_by_role[role] = rows
    integrity = validate_stage4_roles(
        rows_by_role,
        tokenizer=tokenizer,
        max_pair_token_delta=max_pair_token_delta,
        max_completion_tokens=max_completion_tokens,
    )
    if {
        key: manifest["integrity"][key]
        for key in ("total_rows", "unique_ids", "unique_task_hashes", "overlap_count")
    } != {
        key: integrity[key]
        for key in ("total_rows", "unique_ids", "unique_task_hashes", "overlap_count")
    }:
        raise ValueError("Frozen Stage 4 integrity summary mismatch")
    return integrity
