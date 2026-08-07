from __future__ import annotations

import ast
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any

from reasoning_lab.io_utils import read_jsonl, sha256_file, write_json, write_jsonl


FAMILIES = ("add_sub", "multiplication", "mul_add", "parenthesized")
STABLE_SPLITS = ("train", "validation", "test", "continuation")
PROMPT_TEMPLATES = (
    "Evaluate the integer expression.\nExpression: {expression}\n"
    "Return exactly one line: FINAL_ANSWER=<integer>\n",
    "Compute this expression using integer arithmetic.\nExpression: {expression}\n"
    "Output exactly: FINAL_ANSWER=<integer>\n",
    "Find the value of the expression below.\nExpression: {expression}\n"
    "Your entire response must be FINAL_ANSWER=<integer>\n",
    "Solve the integer calculation.\nExpression: {expression}\n"
    "Use one line in the form FINAL_ANSWER=<integer>\n",
)


def evaluate_expression(expression: str) -> int:
    """Evaluate the restricted integer grammar independently of the generator."""

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return visit(node.left) + visit(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            return visit(node.left) - visit(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return visit(node.left) * visit(node.right)
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    return visit(ast.parse(expression, mode="eval"))


def _operand(rng: random.Random, difficulty: int, *, multiplier: bool = False) -> int:
    bounds = {1: 12, 2: 30, 3: 75}
    bound = min(bounds[difficulty], 20) if multiplier else bounds[difficulty]
    value = rng.randint(-bound, bound)
    if multiplier and value == 0:
        value = 1
    return value


def _expression(rng: random.Random, family: str, difficulty: int) -> str:
    if family == "add_sub":
        a = _operand(rng, difficulty)
        b = _operand(rng, difficulty)
        c = _operand(rng, difficulty)
        return f"{a} + {b} - {c}"
    if family == "multiplication":
        a = _operand(rng, difficulty, multiplier=True)
        b = _operand(rng, difficulty, multiplier=True)
        return f"{a} * {b}"
    if family == "mul_add":
        a = _operand(rng, difficulty, multiplier=True)
        b = _operand(rng, difficulty, multiplier=True)
        c = _operand(rng, difficulty)
        return f"({a} * {b}) + {c}"
    if family == "parenthesized":
        a = _operand(rng, difficulty)
        b = _operand(rng, difficulty)
        c = _operand(rng, difficulty, multiplier=True)
        return f"({a} + {b}) * {c}"
    raise ValueError(f"Unknown family: {family}")


def _wide_multiplier_operand(rng: random.Random, difficulty: int) -> int:
    value = _operand(rng, difficulty)
    return 1 if value == 0 else value


def _expression_v2(rng: random.Random, family: str, difficulty: int) -> str:
    """Generate the v2 grammar without the legacy +/-20 multiplier cap."""

    if family == "add_sub":
        a = _operand(rng, difficulty)
        b = _operand(rng, difficulty)
        c = _operand(rng, difficulty)
        return f"{a} + {b} - {c}"
    if family == "multiplication":
        a = _wide_multiplier_operand(rng, difficulty)
        b = _wide_multiplier_operand(rng, difficulty)
        return f"{a} * {b}"
    if family == "mul_add":
        a = _wide_multiplier_operand(rng, difficulty)
        b = _wide_multiplier_operand(rng, difficulty)
        c = _operand(rng, difficulty)
        return f"({a} * {b}) + {c}"
    if family == "parenthesized":
        a = _operand(rng, difficulty)
        b = _operand(rng, difficulty)
        c = _wide_multiplier_operand(rng, difficulty)
        return f"({a} + {b}) * {c}"
    raise ValueError(f"Unknown family: {family}")


def _uses_stable_splits(generator_version: str) -> bool:
    version = generator_version.strip().lstrip("vV")
    try:
        return int(version.split(".", maxsplit=1)[0]) >= 2
    except ValueError:
        return False


def _candidate_rows(seed: int, generator_version: str):
    """Yield a deterministic stream of unique examples.

    The legacy expression generator and RNG call order are retained for 1.x.
    Version 2.x gets a larger multiplier space, while row identities stay tied
    only to immutable example content and the declared generator version.
    """

    rng = random.Random(seed)
    expressions: set[str] = set()
    candidate_index = 0
    expression_builder = (
        _expression_v2 if _uses_stable_splits(generator_version) else _expression
    )
    source_dataset = (
        "arithmetic_v2" if _uses_stable_splits(generator_version) else "arithmetic_v1"
    )

    while True:
        family = FAMILIES[candidate_index % len(FAMILIES)]
        difficulty = 1 + ((candidate_index // len(FAMILIES)) % 3)
        candidate_index += 1
        expression = expression_builder(rng, family, difficulty)
        if expression in expressions:
            continue
        expressions.add(expression)

        # Keep prompt wording independent from the task family to avoid a
        # family/template confound in per-family metrics.
        template_index = rng.randrange(len(PROMPT_TEMPLATES))
        template_id = f"t{template_index + 1}"
        answer = evaluate_expression(expression)
        stable_key = f"{generator_version}|{expression}|{template_id}"
        example_id = "arith_" + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:20]
        yield {
            "id": example_id,
            "family": family,
            "difficulty": difficulty,
            "template_id": template_id,
            "expression": expression,
            "prompt": PROMPT_TEMPLATES[template_index].format(expression=expression),
            "completion": f"FINAL_ANSWER={answer}",
            "answer": answer,
            "output_contract": "answer_only",
            "source_dataset": source_dataset,
            "generator_version": generator_version,
        }


def build_examples(total: int, seed: int, generator_version: str) -> list[dict[str, Any]]:
    stream = _candidate_rows(seed=seed, generator_version=generator_version)
    return [next(stream) for _ in range(total)]


def _stable_split(expression: str, seed: int, generator_version: str) -> str:
    stable_key = f"{generator_version}|{seed}|split|{expression}"
    digest = hashlib.sha256(stable_key.encode("utf-8")).digest()
    return STABLE_SPLITS[int.from_bytes(digest[:8], "big") % len(STABLE_SPLITS)]


def _build_stable_splits(
    sizes: dict[str, int], seed: int, generator_version: str
) -> dict[str, list[dict[str, Any]]]:
    """Fill fixed hash partitions from one immutable candidate stream.

    Every split is the prefix of its own hash-assigned stream. Therefore a
    quota change in one split cannot perturb the examples in another split.
    """

    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in STABLE_SPLITS
    }
    stream = _candidate_rows(seed=seed, generator_version=generator_version)
    while any(len(rows_by_split[split]) < sizes[split] for split in STABLE_SPLITS):
        row = next(stream)
        split = _stable_split(row["expression"], seed, generator_version)
        if len(rows_by_split[split]) < sizes[split]:
            rows_by_split[split].append({**row, "split": split})
    return rows_by_split


def generate_dataset(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    data_config = config["data"]
    generator_version = str(data_config["generator_version"])
    stable_splits = _uses_stable_splits(generator_version)
    sizes = {
        "train": int(data_config["train_size"]),
        "validation": int(data_config["validation_size"]),
        "test": int(data_config["test_size"]),
    }
    if stable_splits:
        sizes["continuation"] = int(data_config.get("continuation_size", 0))

    if any(size < 0 for size in sizes.values()):
        raise ValueError("Split sizes must be non-negative")

    if stable_splits:
        rows_by_split = _build_stable_splits(
            sizes=sizes,
            seed=int(data_config["seed"]),
            generator_version=generator_version,
        )
    else:
        rows = build_examples(
            total=sum(sizes.values()),
            seed=int(data_config["seed"]),
            generator_version=generator_version,
        )
        rows_by_split = {}
        offset = 0
        for split, size in sizes.items():
            rows_by_split[split] = [
                {**row, "split": split} for row in rows[offset : offset + size]
            ]
            offset += size

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "generator_version": generator_version,
        "seed": int(data_config["seed"]),
        "splits": {},
    }

    for split, size in sizes.items():
        split_rows = rows_by_split[split]
        path = destination / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        manifest["splits"][split] = {
            "count": len(split_rows),
            "sha256": sha256_file(path),
            "families": dict(sorted(Counter(row["family"] for row in split_rows).items())),
            "difficulties": dict(
                sorted(Counter(str(row["difficulty"]) for row in split_rows).items())
            ),
        }

    validate_split_integrity(destination)
    manifest_path = destination / "dataset_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def validate_split_integrity(data_dir: str | Path) -> None:
    root = Path(data_dir)
    seen_ids: set[str] = set()
    seen_expressions: set[str] = set()
    splits = ["train", "validation", "test"]
    if (root / "continuation.jsonl").exists():
        splits.append("continuation")
    for split in splits:
        rows = read_jsonl(root / f"{split}.jsonl")
        ids = {row["id"] for row in rows}
        expressions = {row["expression"] for row in rows}
        if len(ids) != len(rows) or len(expressions) != len(rows):
            raise ValueError(f"Duplicate examples inside {split}")
        if ids & seen_ids or expressions & seen_expressions:
            raise ValueError(f"Data leakage detected in {split}")
        for row in rows:
            if evaluate_expression(row["expression"]) != int(row["answer"]):
                raise ValueError(f"Incorrect label for {row['id']}")
        seen_ids.update(ids)
        seen_expressions.update(expressions)
