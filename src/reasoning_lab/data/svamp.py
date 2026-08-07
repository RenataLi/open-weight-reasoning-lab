from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import urllib.request
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from reasoning_lab.evaluation.parsing import normalize_number
from reasoning_lab.io_utils import sha256_file, write_json, write_jsonl


SVAMP_COMMIT = "689d7ccac74b9983a2ac7cc3b264f441b99e7c53"
SVAMP_SOURCE_URL = (
    "https://raw.githubusercontent.com/arkilpatel/SVAMP/"
    f"{SVAMP_COMMIT}/SVAMP.json"
)
SVAMP_BYTES = 339970
SVAMP_SHA256 = "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
KNOWN_INVALID_ROWS = {
    "chal-680": "Equation evaluates to 5 while the published Answer field is 1",
}
def _download_svamp_bytes(timeout_seconds: int = 60) -> bytes:
    request = urllib.request.Request(
        SVAMP_SOURCE_URL,
        headers={"User-Agent": "reasoning-lab-stage4-snapshot/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def _verify_svamp_bytes(payload: bytes) -> None:
    if len(payload) != SVAMP_BYTES:
        raise ValueError(
            f"SVAMP source byte count differs from pinned snapshot: {len(payload)} != {SVAMP_BYTES}"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SVAMP_SHA256:
        raise ValueError(
            f"SVAMP source checksum differs from pinned snapshot: {digest} != {SVAMP_SHA256}"
        )


def ensure_svamp_source(
    source_path: str | Path, *, timeout_seconds: int = 60
) -> Path:
    """Return a verified pinned source, downloading atomically only when absent."""

    source = Path(source_path).resolve()
    if source.exists():
        if not source.is_file():
            raise ValueError(f"SVAMP source path is not a file: {source}")
        _verify_svamp_bytes(source.read_bytes())
        return source

    payload = _download_svamp_bytes(timeout_seconds)
    _verify_svamp_bytes(payload)
    source.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=source.parent,
            prefix=f".{source.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        temporary_path.replace(source)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return source


def _split_development_quarantine(
    rows: list[dict[str, Any]],
    *,
    forced_validation_ids: set[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Keep legacy validation fixed and quarantine every other exposed row."""

    normalized_forced = {
        value if value.startswith("svamp_") else f"svamp_{value}"
        for value in forced_validation_ids
    }
    available = {str(row["id"]): row for row in rows}
    missing = sorted(normalized_forced - set(available))
    if missing:
        raise ValueError(f"Forced SVAMP validation IDs are absent: {missing[:10]}")

    validation = sorted(
        (row for row in rows if str(row["id"]) in normalized_forced),
        key=lambda row: str(row["id"]),
    )
    development_quarantine = sorted(
        (row for row in rows if str(row["id"]) not in normalized_forced),
        key=lambda row: str(row["id"]),
    )
    overall_family_counts = Counter(str(row["family"]) for row in rows)
    validation_family_counts = Counter(str(row["family"]) for row in validation)
    family_proportion_deviations = {
        family: validation_family_counts[family] / len(validation)
        - overall_family_counts[family] / len(rows)
        for family in sorted(overall_family_counts)
    }
    diagnostics = {
        "forced_validation_requested_count": len(normalized_forced),
        "forced_validation_matched_count": len(validation),
        "development_quarantine_count": len(development_quarantine),
        "all_verified_rows_development_exposed": True,
        "confirmation_count": 0,
        "validation_family_counts": dict(sorted(validation_family_counts.items())),
        "overall_family_counts": dict(sorted(overall_family_counts.items())),
        "family_proportion_deviations": family_proportion_deviations,
        "maximum_family_proportion_deviation": max(
            (abs(value) for value in family_proportion_deviations.values()),
            default=0.0,
        ),
    }
    return validation, development_quarantine, diagnostics


def _fraction(value: str | int | float) -> Fraction:
    normalized = normalize_number(value)
    return Fraction(normalized)


def _evaluate_equation(expression: str) -> Fraction:
    tree = ast.parse(expression.strip(), mode="eval")

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("SVAMP equation divides by zero")
                return left / right
        raise ValueError(f"Unsupported SVAMP equation node: {ast.dump(node)}")

    return evaluate(tree)


def _id_set_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        sorted(str(row["id"]) for row in rows),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    source_id = str(row["ID"])
    body = str(row["Body"]).strip()
    question = str(row["Question"]).strip()
    answer = normalize_number(row["Answer"])
    equation = str(row["Equation"]).strip()
    evaluated = _evaluate_equation(equation)
    if evaluated != _fraction(answer):
        raise ValueError(
            f"SVAMP {source_id} equation/answer mismatch: {evaluated} != {answer}"
        )
    family = str(row["Type"]).replace("Common-Divison", "Common-Division")
    full_question = f"{body} {question}".strip()
    prompt = (
        f"Question: {full_question}\n"
        "Show concise reasoning and end with a separate final line exactly like "
        "FINAL_ANSWER=<number>.\nAnswer:"
    )
    return {
        "id": f"svamp_{source_id}",
        "family": f"svamp_{family.lower()}",
        "difficulty": "challenge",
        "template_id": "svamp_reasoning_v1",
        "question": full_question,
        "expression": equation,
        "prompt": prompt,
        "answer": answer,
        "output_contract": "reasoning_then_answer",
        "source_dataset": "arkilpatel/SVAMP",
        "source_commit": SVAMP_COMMIT,
        "source_split": "challenge",
        "source_index": source_index,
        "source_id": source_id,
    }


def prepare_svamp_snapshot(
    config: dict[str, Any],
    *,
    source_path: str | Path,
    output_dir: str | Path,
    forced_validation_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Freeze legacy validation and quarantine all other development-exposed rows."""

    source = ensure_svamp_source(source_path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 1000:
        raise ValueError("Pinned SVAMP source must contain exactly 1000 rows")

    verified: list[dict[str, Any]] = []
    observed_invalid: dict[str, str] = {}
    for index, row in enumerate(raw):
        source_id = str(row.get("ID", ""))
        try:
            normalized = _normalize(dict(row), index)
        except ValueError as error:
            if source_id not in KNOWN_INVALID_ROWS:
                raise
            observed_invalid[source_id] = str(error)
            continue
        if source_id in KNOWN_INVALID_ROWS:
            raise ValueError(f"Expected pinned-invalid SVAMP row to fail: {source_id}")
        verified.append(normalized)
    if set(observed_invalid) != set(KNOWN_INVALID_ROWS):
        raise ValueError("SVAMP invalid-row audit differs from the preregistered exclusion")
    if len(verified) != 999:
        raise ValueError("SVAMP verified row count must be 999")

    stage4 = config["stage4"]
    if "svamp_confirmation_size" in stage4 or "svamp_retired_size" in stage4:
        raise ValueError(
            "SVAMP confirmation/retired sizes are forbidden; all non-validation rows "
            "must be development quarantine"
        )
    validation_size = int(stage4["svamp_validation_size"])
    quarantine_size = int(stage4["svamp_development_quarantine_size"])
    if validation_size + quarantine_size != len(verified):
        raise ValueError(
            "SVAMP validation plus development quarantine must cover all 999 verified rows"
        )
    split_seed = int(stage4["data_seed"])
    legacy_validation_seed = int(
        stage4.get("svamp_forced_validation_seed", 20260807)
    )
    legacy_ordered = sorted(
        verified,
        key=lambda row: hashlib.sha256(
            f"stage4-svamp|{legacy_validation_seed}|{row['id']}".encode("utf-8")
        ).hexdigest(),
    )
    legacy_validation_ids = {
        str(row["id"]) for row in legacy_ordered[:validation_size]
    }
    if forced_validation_ids is not None:
        supplied_validation_ids = {
            value if value.startswith("svamp_") else f"svamp_{value}"
            for value in forced_validation_ids
        }
        combined_validation_ids = legacy_validation_ids | supplied_validation_ids
        if combined_validation_ids != legacy_validation_ids:
            unexpected = sorted(combined_validation_ids - legacy_validation_ids)
            raise ValueError(
                "Supplied forced SVAMP IDs differ from the reproducible legacy split: "
                f"{unexpected[:10]}"
            )
        if supplied_validation_ids != legacy_validation_ids:
            missing = sorted(legacy_validation_ids - supplied_validation_ids)
            raise ValueError(
                "Supplied forced SVAMP IDs omit rows from the reproducible legacy split: "
                f"{missing[:10]}"
            )
    validation, development_quarantine, split_diagnostics = (
        _split_development_quarantine(
            verified,
            forced_validation_ids=legacy_validation_ids,
        )
    )
    strategy = (
        "reproduced_legacy_validation_plus_all_remaining_development_quarantine"
    )
    if len(validation) != validation_size:
        raise RuntimeError("Legacy SVAMP validation split has the wrong size")
    if len(development_quarantine) != quarantine_size:
        raise RuntimeError(
            "SVAMP development quarantine size differs from the frozen protocol: "
            f"{len(development_quarantine)} != {quarantine_size}"
        )
    role_id_sets = {
        "validation": {str(row["id"]) for row in validation},
        "development_quarantine": {
            str(row["id"]) for row in development_quarantine
        },
    }
    if role_id_sets["validation"] & role_id_sets["development_quarantine"]:
        raise RuntimeError("SVAMP validation and development quarantine rows overlap")
    if sum(len(values) for values in role_id_sets.values()) != len(verified):
        raise RuntimeError("SVAMP split roles do not cover all verified rows")

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"SVAMP output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for filename, role, rows in (
        (
            "validation.jsonl",
            "stage4_svamp_validation",
            validation,
        ),
        (
            "development_quarantine.jsonl",
            "stage4_svamp_development_quarantine",
            development_quarantine,
        ),
    ):
        path = destination / filename
        write_jsonl(path, rows)
        files[role] = {
            "role": role,
            "path": str(path.resolve()),
            "count": len(rows),
            "sha256": sha256_file(path),
            "row_id_set_sha256": _id_set_hash(rows),
            "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        }
    manifest = {
        "dataset_id": "arkilpatel/SVAMP",
        "license": "MIT",
        "source_url": SVAMP_SOURCE_URL,
        "source_commit": SVAMP_COMMIT,
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "published_count": len(raw),
            "verified_count": len(verified),
        },
        "excluded_rows": {
            source_id: {
                "preregistered_reason": KNOWN_INVALID_ROWS[source_id],
                "observed_error": observed_invalid[source_id],
            }
            for source_id in sorted(observed_invalid)
        },
        "split_seed": split_seed,
        "split_protocol": {
            "strategy": strategy,
            "legacy_validation_protocol": "sort_sha256_stage4-svamp_seed_row-id_take_first_n_v1",
            "legacy_validation_seed": legacy_validation_seed,
            "legacy_validation_size": validation_size,
            "legacy_validation_id_set_sha256": _id_set_hash(
                [{"id": value} for value in sorted(legacy_validation_ids)]
            ),
            "validation_family_tolerance_applied": False,
            **split_diagnostics,
        },
        "development_exposure": {
            "all_verified_rows_development_exposed": True,
            "verified_development_exposed_count": len(verified),
            "exposure_basis": "prior_aborted_stage4_runs_and_independent_data_audit",
            "validation_count": len(validation),
            "development_quarantine_count": len(development_quarantine),
            "confirmation_count": 0,
            "confirmation_role_present": False,
            "eligible_for_final_claim": False,
            "claim_scope": "development_diagnostics_only",
        },
        "files": files,
    }
    write_json(destination / "dataset_manifest.json", manifest)
    return manifest
