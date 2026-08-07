# Reproducibility guide

## Validated environment

The recorded full experiments used:

- Windows 11;
- Python 3.12;
- NVIDIA RTX 5090 (32 GiB);
- PyTorch 2.13 with CUDA 13.0;
- Transformers, TRL, PEFT, Accelerate, datasets, and bitsandbytes at the versions pinned in `requirements.lock.txt`.

The selected Stage 4 DPO branch peaked at 6.91 GiB of allocated GPU memory. Evaluation loads can be larger than training loads because the recorded protocol evaluates without 4-bit quantization.

## Fresh checkout

```powershell
git clone https://github.com/RenataLi/open-weight-reasoning-lab.git
cd open-weight-reasoning-lab

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Run the hardware preflight before a training job:

```powershell
reasoning-lab preflight
```

Run the full test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Compact reproduction

The smoke config exercises deterministic data creation, training, evaluation, and reporting with a smaller model:

```powershell
reasoning-lab generate-data --config configs/e2e_smoke.yaml
reasoning-lab run --config configs/e2e_smoke.yaml
```

## Research lineage

Run stages in order because later configs bind to immutable upstream artifacts.

```powershell
reasoning-lab run-stage2 --config configs/stage2_qwen3_1_7b.yaml

reasoning-lab run-stage3-search `
  --config configs/stage3_reasoning_contract_qwen3_1_7b.yaml

reasoning-lab run-stage3-confirm `
  --config configs/stage3_reasoning_contract_qwen3_1_7b.yaml `
  --search-run runs\<completed-stage3-search>

reasoning-lab run-stage4-search `
  --config configs/stage4_dpo_qwen3_1_7b.yaml
```

A newly reproduced Stage 3 run has a new run ID and adapter hash. Create a new Stage 4 protocol revision and update its upstream run paths and hash anchors accordingly. This explicit rebinding is part of the experiment contract: late stages should never discover an arbitrary "latest" checkpoint.

## Artifact checklist

A completed research run should include:

- `run_manifest.json` with a terminal status;
- `resolved_config.yaml`;
- `environment.json` and successful GPU preflight;
- deterministic dataset manifests;
- training summaries and adapter hashes;
- evaluation metrics and predictions;
- selection result and lock where applicable;
- source/implementation attestation;
- report and success marker.

For a selection-plus-confirmation protocol, verify that the selection lock predates the test-start marker and that the run manifest anchors the same lock hash.

## Public artifact policy

This repository intentionally tracks only lightweight, reviewable artifacts:

- source code and tests;
- pinned configs and dependencies;
- aggregate reports;
- diagrams and figures.

Local model caches, generated/external data, checkpoints, adapters, raw generations, predictions, logs, and confirmation exposure registries remain ignored. Recreate them through the CLI so their provenance is captured by the current code and configuration.

## Determinism notes

The pipeline fixes Python, NumPy, and PyTorch seeds, records data-loader seeds, uses greedy generation, and hashes every important input. GPU kernels and third-party dependencies can still vary across environments, so comparisons should preserve the recorded package/CUDA stack and should be interpreted through the stored seed range and paired uncertainty estimates.
