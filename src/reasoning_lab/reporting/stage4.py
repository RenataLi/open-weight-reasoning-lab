from __future__ import annotations

from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _preference_cells(metrics: dict[str, Any]) -> tuple[str, str]:
    return (
        _percent(metrics["summed"]["overall"]["reward_accuracy"]),
        _percent(metrics["normalized"]["overall"]["reward_accuracy"]),
    )


def write_stage4_search_report(
    path: str | Path,
    *,
    selection: dict[str, Any],
    control_selection: dict[str, Any],
    base: dict[str, Any],
    candidates: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    data_manifest: dict[str, Any],
) -> None:
    lines = [
        "# Stage 4 DPO feasibility search",
        "",
        "> Development-only search. No confirmation output was read to select the "
        "DPO objective, adapter, or extra-SFT control.",
        "",
        "## Frozen protocol",
        "",
        f"- Preference train/dev/final: "
        f"`{data_manifest['roles']['stage4_preference_train']['count']}` / "
        f"`{data_manifest['roles']['stage4_preference_validation']['count']}` / "
        f"`{data_manifest['roles']['stage4_preference_confirmation']['count']}`.",
        f"- Synthetic dev/final: "
        f"`{data_manifest['roles']['stage4_synthetic_validation']['count']}` / "
        f"`{data_manifest['roles']['stage4_synthetic_confirmation']['count']}`.",
        f"- SVAMP development: "
        f"`{data_manifest['roles']['stage4_svamp_validation']['count']}`.",
        f"- SVAMP development quarantine (never a final endpoint): "
        f"`{data_manifest['roles']['stage4_svamp_development_quarantine']['count']}`.",
        "- All 999 valid SVAMP rows are development-exposed. No SVAMP row can be "
        "used for a fresh confirmation claim in this protocol.",
        "- DPO and both controls start from the same immutable Stage 3 seed-101 adapter.",
        "- Adapter dropout is disabled for DPO, its frozen reference, and both "
        "extra-SFT controls so reference log-probabilities are deterministic.",
        "- The update-matched control uses exactly the chosen responses from DPO train. "
        "The token-proxy control doubles SFT steps; neither is claimed to be FLOP-matched.",
        "",
        "## Development metrics",
        "",
        "| Branch | Synthetic exact | Synthetic strict | SVAMP exact | SVAMP strict | Pref acc (sum) | Pref acc (norm) | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Frozen SFT | {_percent(base['synthetic']['overall']['answer_exact_match'])} | "
        f"{_percent(base['synthetic']['overall']['strict_correct_rate'])} | "
        f"{_percent(base['external']['overall']['answer_exact_match'])} | "
        f"{_percent(base['external']['overall']['strict_correct_rate'])} | - | - | - |",
    ]
    selection_by_id = {record["id"]: record for record in selection["records"]}
    for candidate in candidates:
        record = selection_by_id[candidate["id"]]
        summed, normalized = _preference_cells(candidate["preference"])
        lines.append(
            f"| {candidate['id']} | "
            f"{_percent(candidate['synthetic']['overall']['answer_exact_match'])} | "
            f"{_percent(candidate['synthetic']['overall']['strict_correct_rate'])} | "
            f"{_percent(candidate['external']['overall']['answer_exact_match'])} | "
            f"{_percent(candidate['external']['overall']['strict_correct_rate'])} | "
            f"{summed} | {normalized} | {'yes' if record['eligible'] else 'no'} |"
        )
    for control in controls:
        summed, normalized = _preference_cells(control["preference"])
        lines.append(
            f"| {control['id']} | "
            f"{_percent(control['synthetic']['overall']['answer_exact_match'])} | "
            f"{_percent(control['synthetic']['overall']['strict_correct_rate'])} | "
            f"{_percent(control['external']['overall']['answer_exact_match'])} | "
            f"{_percent(control['external']['overall']['strict_correct_rate'])} | "
            f"{summed} | {normalized} | control |"
        )
    lines.extend(["", "## Search outcome", ""])
    if selection["selected_candidate"] is None:
        lines.extend(
            [
                "- No DPO candidate passed every preregistered development gate.",
                "- Confirmation is therefore prohibited for this protocol.",
            ]
        )
    else:
        lines.extend(
            [
                f"- DPO: `{selection['selected_candidate']}`.",
                f"- Extra-SFT comparator: `{control_selection['selected_control']}`.",
                "- Selection roles: `stage4_synthetic_validation`, "
                "`stage4_svamp_validation`, and `stage4_preference_validation` only.",
                "- Fresh synthetic and preference confirmation files were materialized "
                "and hashed before search, but remained unexposed. A separate command "
                "must claim them before evaluation.",
            ]
        )
    lines.append("")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def write_stage4_confirmation_report(
    path: str | Path,
    *,
    verdict: dict[str, Any],
    metrics: dict[str, Any],
    comparisons: dict[str, Any],
    lock: dict[str, Any],
) -> None:
    lines = [
        "# Stage 4 DPO feasibility confirmation",
        "",
        f"> **{verdict['status']}**",
        "",
        "This is a one-time confirmation on fresh synthetic and preference rows. "
        "SVAMP is development-only because all of its valid rows were exposed during "
        "protocol auditing; it is not used for a final claim.",
        "",
        "## Locked branches",
        "",
        f"- Frozen reference: `{lock['parent']['adapter_sha256']}`.",
        f"- DPO: `{lock['selected_candidate']['id']}`.",
        f"- Matched extra-SFT: `{lock['selected_control']['id']}`.",
        "",
        "## Confirmation metrics",
        "",
        "| Branch | Synthetic exact | Synthetic strict | Synthetic format | Mean tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for branch in ("parent", "control", "dpo"):
        overall = metrics[branch]["synthetic"]["overall"]
        lines.append(
            f"| {branch} | {_percent(overall['answer_exact_match'])} | "
            f"{_percent(overall['strict_correct_rate'])} | "
            f"{_percent(overall['format_valid_rate'])} | "
            f"{float(overall['generated_tokens_mean']):.1f} |"
        )
    dpo_preference = metrics["dpo_preference"]["objective"]["overall"]
    control_preference = metrics["control_preference"]["objective"]["overall"]
    lines.extend(
        [
            "",
            "| Preference branch | Accuracy | Mean objective margin |",
            "|---|---:|---:|",
            f"| DPO | {_percent(dpo_preference['reward_accuracy'])} | "
            f"{dpo_preference['reward_margin_mean']:.4f} |",
            f"| Extra-SFT | {_percent(control_preference['reward_accuracy'])} | "
            f"{control_preference['reward_margin_mean']:.4f} |",
            "",
            "## Paired comparisons",
            "",
            "| Comparison | Benchmark | Exact delta | Paired 95% CI | Wrong-to-right | Right-to-wrong |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for comparison_name, datasets in comparisons.items():
        for dataset_name, comparison in datasets.items():
            low, high = comparison["paired_bootstrap_95_ci"]
            lines.append(
                f"| {comparison_name} | {dataset_name} | "
                f"{100.0 * comparison['answer_exact_match_delta']:+.1f} pp | "
                f"[{100.0 * low:+.1f}, {100.0 * high:+.1f}] pp | "
                f"{comparison['wrong_to_right']} | {comparison['right_to_wrong']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(verdict["interpretation"]),
            "",
            "The synthetic benchmark is the sole preregistered benefit endpoint. "
            "The pilot uses one preregistered DPO training seed, so a positive outcome "
            "is feasibility evidence, not a broad reasoning claim.",
            "",
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
