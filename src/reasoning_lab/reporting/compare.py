from __future__ import annotations

from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(
    path: str | Path,
    *,
    config: dict[str, Any],
    revision: str,
    baseline_metrics: dict[str, Any],
    post_metrics: dict[str, Any],
    comparison: dict[str, Any],
    training_summary: dict[str, Any],
) -> None:
    baseline = baseline_metrics["overall"]
    post = post_metrics["overall"]
    ci_low, ci_high = comparison["paired_bootstrap_95_ci"]
    lines = [
        "# Stage 1 end-to-end report",
        "",
        "> This run validates the local QLoRA pipeline on synthetic integer arithmetic. "
        "It is not evidence of broad reasoning improvement.",
        "",
        "## Run",
        "",
        f"- Base model: `{config['model']['id']}`",
        f"- Exact revision: `{revision}`",
        f"- Training steps: `{training_summary['global_step']}`",
        f"- Trainable parameters: `{training_summary['trainable_parameters']:,}`",
        f"- Peak allocated GPU memory: `{training_summary['peak_gpu_memory_bytes'] / 2**30:.2f} GiB`",
        "",
        "## Held-out test metrics",
        "",
        "| Metric | Base | Post-SFT | Delta |",
        "|---|---:|---:|---:|",
        f"| Answer exact match | {_percent(baseline['answer_exact_match'])} | "
        f"{_percent(post['answer_exact_match'])} | "
        f"{100.0 * comparison['answer_exact_match_delta']:+.1f} pp |",
        f"| Valid output format | {_percent(baseline['format_valid_rate'])} | "
        f"{_percent(post['format_valid_rate'])} | "
        f"{100.0 * (post['format_valid_rate'] - baseline['format_valid_rate']):+.1f} pp |",
        f"| Valid format + correct | {_percent(baseline['strict_correct_rate'])} | "
        f"{_percent(post['strict_correct_rate'])} | "
        f"{100.0 * (post['strict_correct_rate'] - baseline['strict_correct_rate']):+.1f} pp |",
        f"| Parse success | {_percent(baseline['parse_success_rate'])} | "
        f"{_percent(post['parse_success_rate'])} | "
        f"{100.0 * (post['parse_success_rate'] - baseline['parse_success_rate']):+.1f} pp |",
        f"| Hit generation limit | {_percent(baseline['hit_max_new_tokens_rate'])} | "
        f"{_percent(post['hit_max_new_tokens_rate'])} | "
        f"{100.0 * (post['hit_max_new_tokens_rate'] - baseline['hit_max_new_tokens_rate']):+.1f} pp |",
        "",
        f"Paired bootstrap 95% CI for exact-match delta: "
        f"`[{100.0 * ci_low:+.1f}, {100.0 * ci_high:+.1f}]` percentage points.",
        "",
        f"Wrong -> right: `{comparison['wrong_to_right']}`; "
        f"right -> wrong: `{comparison['right_to_wrong']}`.",
        "",
        "## Per-family exact-match delta",
        "",
    ]
    for family, delta in comparison["per_family_delta"].items():
        lines.append(f"- `{family}`: {100.0 * delta:+.1f} pp")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The benchmark separates answer parsing from strict output-format compliance. "
            "Any observed gain may come from arithmetic, format learning, or both. "
            "Stage 2 must add external held-out tasks and stronger baselines before a capability claim.",
            "",
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
