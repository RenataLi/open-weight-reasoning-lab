from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from reasoning_lab.config import load_config, resolve_path
from reasoning_lab.data.arithmetic import generate_dataset
from reasoning_lab.io_utils import write_json
from reasoning_lab.pipeline import run_pipeline
from reasoning_lab.reproducibility import environment_snapshot, run_gpu_preflight
from reasoning_lab.stage2_pipeline import (
    mark_incomplete_run_aborted,
    refresh_stage2_audit,
    run_stage2_pipeline,
)
from reasoning_lab.stage3_pipeline import run_stage3_confirm, run_stage3_search
from reasoning_lab.stage4_pipeline import run_stage4_confirm, run_stage4_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reasoning-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the complete Stage 1 pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--run-dir")

    stage2 = subparsers.add_parser(
        "run-stage2", help="Run the controlled Qwen3-1.7B Stage 2 pilot"
    )
    stage2.add_argument("--config", required=True)
    stage2.add_argument("--run-dir")

    audit_stage2 = subparsers.add_parser(
        "audit-stage2", help="Refresh derived readiness and report files for a completed run"
    )
    audit_stage2.add_argument("--config", required=True)
    audit_stage2.add_argument("--run-dir", required=True)

    abort_stage2 = subparsers.add_parser(
        "mark-stage2-aborted", help="Close an externally interrupted Stage 2 run"
    )
    abort_stage2.add_argument("--run-dir", required=True)
    abort_stage2.add_argument("--reason", required=True)

    stage3_search = subparsers.add_parser(
        "run-stage3-search",
        help="Run validation-only multi-seed SFT mixture/objective search",
    )
    stage3_search.add_argument("--config", required=True)
    stage3_search.add_argument("--run-dir")

    stage3_confirm = subparsers.add_parser(
        "run-stage3-confirm",
        help="Confirm the immutable Stage 3 selection on fresh held-out data",
    )
    stage3_confirm.add_argument("--config", required=True)
    stage3_confirm.add_argument("--search-run", required=True)
    stage3_confirm.add_argument("--run-dir")

    stage4_search = subparsers.add_parser(
        "run-stage4-search",
        help="Run development-only DPO and matched extra-SFT search",
    )
    stage4_search.add_argument("--config", required=True)
    stage4_search.add_argument("--run-dir")

    stage4_confirm = subparsers.add_parser(
        "run-stage4-confirm",
        help="Run the one-time Stage 4 confirmation on locked fresh data",
    )
    stage4_confirm.add_argument("--config", required=True)
    stage4_confirm.add_argument("--search-run", required=True)
    stage4_confirm.add_argument("--run-dir")

    abort_stage4 = subparsers.add_parser(
        "mark-stage4-aborted", help="Close an externally interrupted Stage 4 run"
    )
    abort_stage4.add_argument("--run-dir", required=True)
    abort_stage4.add_argument("--reason", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate CUDA and NF4 locally")
    preflight.add_argument("--output")

    data = subparsers.add_parser("generate-data", help="Generate deterministic arithmetic data")
    data.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = load_config(args.config)
        destination = run_pipeline(config, run_dir=args.run_dir)
        print(f"SUCCESS: {destination}")
        return
    if args.command == "preflight":
        result = {
            "environment": environment_snapshot(),
            "preflight": run_gpu_preflight(),
        }
        if args.output:
            write_json(Path(args.output), result)
        print(result)
        return
    if args.command == "run-stage2":
        config = load_config(args.config)
        destination = run_stage2_pipeline(config, run_dir=args.run_dir)
        print(f"SUCCESS: {destination}")
        return
    if args.command == "audit-stage2":
        config = load_config(args.config)
        readiness = refresh_stage2_audit(config, run_dir=args.run_dir)
        print(readiness)
        return
    if args.command == "mark-stage2-aborted":
        manifest = mark_incomplete_run_aborted(
            run_dir=args.run_dir,
            reason=args.reason,
        )
        print({"status": manifest["status"], "run_dir": args.run_dir})
        return
    if args.command == "run-stage3-search":
        config = load_config(args.config)
        destination = run_stage3_search(config, run_dir=args.run_dir)
        print(f"SUCCESS: {destination}")
        return
    if args.command == "run-stage3-confirm":
        config = load_config(args.config)
        destination = run_stage3_confirm(
            config,
            search_run=args.search_run,
            run_dir=args.run_dir,
        )
        print(f"SUCCESS: {destination}")
        return
    if args.command == "run-stage4-search":
        config = load_config(args.config)
        destination = run_stage4_search(config, run_dir=args.run_dir)
        print(f"STAGE 4 SEARCH COMPLETE: {destination}")
        return
    if args.command == "run-stage4-confirm":
        config = load_config(args.config)
        destination = run_stage4_confirm(
            config,
            search_run=args.search_run,
            run_dir=args.run_dir,
        )
        print(f"STAGE 4 CONFIRMATION COMPLETE: {destination}")
        return
    if args.command == "mark-stage4-aborted":
        manifest = mark_incomplete_run_aborted(
            run_dir=args.run_dir,
            reason=args.reason,
        )
        print({"status": manifest["status"], "run_dir": args.run_dir})
        return
    if args.command == "generate-data":
        config = load_config(args.config)
        manifest = generate_dataset(config, resolve_path(config, "data_dir"))
        print(manifest)
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
