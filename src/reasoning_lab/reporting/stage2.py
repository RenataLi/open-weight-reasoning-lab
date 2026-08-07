from __future__ import annotations

from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _signed_points(value: float) -> str:
    return f"{100.0 * value:+.1f} pp"


def _metric_row(label: str, values: dict[str, dict[str, Any]], key: str) -> str:
    return (
        f"| {label} | {_percent(float(values['base']['overall'][key]))} | "
        f"{_percent(float(values['sft']['overall'][key]))} | "
        f"{_percent(float(values['continued_sft']['overall'][key]))} |"
    )


def write_stage2_report(
    path: str | Path,
    *,
    config: dict[str, Any],
    model_revision: str,
    dataset_revision: str,
    internal_metrics: dict[str, dict[str, Any]],
    external_metrics: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    sft_summary: dict[str, Any],
    continued_sft_summary: dict[str, Any],
    readiness: dict[str, Any],
) -> None:
    lines = [
        "# Stage 2 controlled-baseline pilot",
        "",
        "> This is a local pilot, not evidence that broad reasoning capability improved. "
        "GSM8K is used only for frozen external evaluation and never for fine-tuning.",
        "",
        "## Provenance",
        "",
        f"- Base model: `{config['model']['id']}`",
        f"- Exact model revision: `{model_revision}`",
        f"- External dataset: `{config['external_evaluation']['dataset_id']}` / "
        f"`{config['external_evaluation']['subset']}`",
        f"- Exact dataset revision: `{dataset_revision}`",
        f"- Synthetic generator: `{config['data']['generator_version']}`",
        f"- Synthetic train / continuation examples: "
        f"`{config['data']['train_size']}` / `{config['data']['continuation_size']}`",
        f"- Frozen GSM8K test subset: `{config['external_evaluation']['test_size']}` examples, "
        f"`{config['external_evaluation']['fewshot_count']}`-shot",
        "",
        "## Synthetic held-out test",
        "",
        "| Metric | Base | SFT | Continued-SFT |",
        "|---|---:|---:|---:|",
        _metric_row("Answer exact match", internal_metrics, "answer_exact_match"),
        _metric_row("Valid output format", internal_metrics, "format_valid_rate"),
        _metric_row("Parse success", internal_metrics, "parse_success_rate"),
        _metric_row("Hit generation limit", internal_metrics, "hit_max_new_tokens_rate"),
        "",
        "## External GSM8K pilot",
        "",
        "| Metric | Base | SFT | Continued-SFT |",
        "|---|---:|---:|---:|",
        _metric_row("Answer exact match", external_metrics, "answer_exact_match"),
        _metric_row("Strict `####` format", external_metrics, "format_valid_rate"),
        _metric_row("Parse success", external_metrics, "parse_success_rate"),
        _metric_row("Hit generation limit", external_metrics, "hit_max_new_tokens_rate"),
        "",
        "The GSM8K scorer accepts a strict final `#### number` when present and otherwise "
        "uses a benchmark-specific final-number fallback. Format compliance is reported "
        "separately from answer correctness.",
        "",
        "## Paired changes",
        "",
    ]
    labels = {
        "internal_base_to_sft": "Synthetic: Base -> SFT",
        "internal_sft_to_continued_sft": "Synthetic: SFT -> Continued-SFT",
        "external_base_to_sft": "GSM8K: Base -> SFT",
        "external_sft_to_continued_sft": "GSM8K: SFT -> Continued-SFT",
    }
    for key, label in labels.items():
        comparison = comparisons[key]
        low, high = comparison["paired_bootstrap_95_ci"]
        lines.append(
            f"- {label}: `{_signed_points(comparison['answer_exact_match_delta'])}` "
            f"(paired bootstrap 95% CI "
            f"`[{_signed_points(low)}, {_signed_points(high)}]`; "
            f"wrong -> right `{comparison['wrong_to_right']}`, "
            f"right -> wrong `{comparison['right_to_wrong']}`)."
        )
    lines.extend(
        [
            "",
            "## Training controls",
            "",
            f"- Initial SFT optimizer steps: `{sft_summary['global_step']}`; trainable "
            f"parameters: `{int(sft_summary['trainable_parameters']):,}`.",
            f"- Continued-SFT optimizer steps: `{continued_sft_summary['global_step']}`.",
            "- Continued-SFT starts from the saved SFT adapter but creates a fresh optimizer, "
            "scheduler, and global step; it is a branch, not checkpoint resume.",
            "- Its prompts come from a deterministic continuation split disjoint from train, "
            "validation, and test.",
            "",
            "## Readiness gate",
            "",
        ]
    )
    for name, check in readiness["checks"].items():
        marker = "PASS" if bool(check["passed"]) else "FAIL"
        lines.append(f"- **{marker}** - {name}: {check['detail']}")
    lines.extend(
        [
            "",
            f"Overall status: **{readiness['status']}**.",
            "",
            "## Interpretation boundary",
            "",
            (
                "Format-independent synthetic exact match changed by "
                f"`{_signed_points(comparisons['internal_base_to_sft']['answer_exact_match_delta'])}` "
                "from Base to SFT, while strict output-format compliance is reported separately. "
                "This distinction prevents format learning from being mislabeled as a reasoning gain."
            ),
            "",
            "The synthetic score measures adaptation to a narrow generated arithmetic format. "
            "The external score is a small, frozen GSM8K subset and is not a substitute for the "
            "full benchmark, multiple seeds, or a hidden contamination-resistant evaluation. "
            "A DPO comparison should only be added after this baseline and its evaluation gate "
            "remain stable in a larger confirmatory run.",
            "",
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
