from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from reasoning_lab.data.arithmetic import evaluate_expression
from reasoning_lab.io_utils import read_jsonl, sha256_file, write_json, write_jsonl


DERIVATION_VERSION = "stage3_sft_mixture_v1"
FLEXIBLE_PROMPT = (
    "Evaluate the integer expression carefully. You may show concise arithmetic steps.\n"
    "Expression: {expression}\n"
    "End with exactly one final nonempty line: FINAL_ANSWER=<integer>\n"
)
REASONING_PROMPT = (
    "Evaluate the integer expression carefully. Show concise arithmetic steps.\n"
    "Expression: {expression}\n"
    "End with exactly one final nonempty line: FINAL_ANSWER=<integer>\n"
)
PROMPT_CONDITIONING_MODES = {"shared_flexible", "contract_conditioned"}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evaluate_with_steps(expression: str) -> tuple[int, list[str]]:
    """Evaluate the restricted grammar and return verified post-order equations."""

    def visit(node: ast.AST) -> tuple[int, list[str]]:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value), []
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value, steps = visit(node.operand)
            return -value, steps
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult)
        ):
            left, left_steps = visit(node.left)
            right, right_steps = visit(node.right)
            if isinstance(node.op, ast.Add):
                symbol, result = "+", left + right
            elif isinstance(node.op, ast.Sub):
                symbol, result = "-", left - right
            else:
                symbol, result = "*", left * right
            return result, [*left_steps, *right_steps, f"{left} {symbol} {right} = {result}"]
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    result, steps = visit(ast.parse(expression, mode="eval"))
    if result != evaluate_expression(expression):
        raise AssertionError("Independent arithmetic evaluators disagree")
    if not steps:
        raise ValueError("A reasoning target requires at least one arithmetic operation")
    return result, steps


def reasoning_completion(expression: str) -> str:
    answer, steps = _evaluate_with_steps(expression)
    rendered = [f"Step {index}: {equation}." for index, equation in enumerate(steps, 1)]
    rendered.append(f"FINAL_ANSWER={answer}")
    return "\n".join(rendered)


def _reasoning_ids(
    rows: list[dict[str, Any]], *, reasoning_fraction: float, mixture_seed: int
) -> set[str]:
    target = round(len(rows) * reasoning_fraction)
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"{DERIVATION_VERSION}|{mixture_seed}|{row['id']}".encode("utf-8")
            ).hexdigest(),
            str(row["id"]),
        ),
    )
    return {str(row["id"]) for row in ranked[:target]}


def materialize_training_view(
    source_path: str | Path,
    output_path: str | Path,
    *,
    candidate_id: str,
    reasoning_fraction: float,
    mixture_seed: int,
    prompt_conditioning: str = "shared_flexible",
) -> dict[str, Any]:
    """Derive a deterministic training-only view without changing source rows."""

    if not candidate_id or any(character.isspace() for character in candidate_id):
        raise ValueError("candidate_id must be a nonempty whitespace-free identifier")
    if not 0.0 <= reasoning_fraction <= 1.0:
        raise ValueError("reasoning_fraction must be between 0 and 1")
    if prompt_conditioning not in PROMPT_CONDITIONING_MODES:
        raise ValueError(
            f"prompt_conditioning must be one of {sorted(PROMPT_CONDITIONING_MODES)}"
        )

    source = Path(source_path).resolve()
    rows = read_jsonl(source)
    if not rows:
        raise ValueError("Training source is empty")
    source_splits = {str(row.get("split", "")) for row in rows}
    if not source_splits or not source_splits <= {"train", "continuation"}:
        raise ValueError(
            "Stage 3 views may only be derived from train or continuation rows"
        )
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Training source contains duplicate IDs")

    reasoning_ids = _reasoning_ids(
        rows,
        reasoning_fraction=reasoning_fraction,
        mixture_seed=mixture_seed,
    )
    derived: list[dict[str, Any]] = []
    for row in rows:
        answer = evaluate_expression(str(row["expression"]))
        if answer != int(row["answer"]):
            raise ValueError(f"Incorrect source label for {row['id']}")
        reasoning = str(row["id"]) in reasoning_ids
        completion = (
            reasoning_completion(str(row["expression"]))
            if reasoning
            else f"FINAL_ANSWER={answer}"
        )
        if prompt_conditioning == "contract_conditioned":
            prompt = (
                REASONING_PROMPT.format(expression=row["expression"])
                if reasoning
                else str(row["prompt"])
            )
        else:
            prompt = FLEXIBLE_PROMPT.format(expression=row["expression"])
        derived.append(
            {
                **row,
                "prompt": prompt,
                "completion": completion,
                "output_contract": (
                    "reasoning_then_answer" if reasoning else "answer_only"
                ),
                "source_id": row["id"],
                "supervision_style": "concise_reasoning" if reasoning else "direct",
                "derivation_version": DERIVATION_VERSION,
                "mixture_candidate": candidate_id,
            }
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination, derived)
    style_counts = {
        style: sum(row["supervision_style"] == style for row in derived)
        for style in ("concise_reasoning", "direct")
    }
    spec = {
        "candidate_id": candidate_id,
        "derivation_version": DERIVATION_VERSION,
        "mixture_seed": int(mixture_seed),
        "reasoning_fraction": float(reasoning_fraction),
        "prompt_conditioning": prompt_conditioning,
        "prompt_templates": {
            "shared_flexible": FLEXIBLE_PROMPT,
            "reasoning": REASONING_PROMPT,
            "direct": "frozen source prompt",
        },
    }
    manifest = {
        "spec": spec,
        "spec_sha256": _canonical_hash(spec),
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "count": len(rows),
            "splits": sorted(source_splits),
        },
        "output": {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "count": len(derived),
            "style_counts": style_counts,
            "reasoning_id_set_sha256": _canonical_hash(sorted(reasoning_ids)),
        },
    }
    write_json(destination.with_suffix(".manifest.json"), manifest)
    return manifest
