# Stage 4 — DPO development search

## Executive summary

The `stage4_dpo_feasibility_v2_1` search completed successfully and produced an immutable selection lock. Three preregistered DPO variants and two fresh-state extra-SFT controls were trained from the same Stage 3 parent. Every DPO candidate passed all 15 development checks.

The safety-first selector chose `dpo_sigmoid_lr2e6`, the branch with the strongest worst-task retention. On the frozen development roles it:

- increased synthetic exact match from **69.79% to 70.31%**;
- increased SVAMP exact match from **75.38% to 76.38%**;
- increased SVAMP strict correctness from **18.09% to 20.60%**;
- reached **64.06%** chosen-over-rejected accuracy, versus **38.28%** for the selected extra-SFT comparator;
- preserved 100% parse success and near-parent generation length.

All numbers in this report are aggregate development metrics. Raw prompts, predictions, adapters, local paths, and sealed evaluation rows are intentionally excluded from the public repository.

## Protocol

| Component | Frozen setting |
|---|---|
| Base model | `Qwen/Qwen3-1.7B-Base` |
| Exact model revision | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| Parent | Stage 3 `conditioned90_nll_lr5e5`, seed 101 |
| Preference train / validation | 512 / 128 pairs |
| Synthetic validation | 384 tasks, 7 families |
| SVAMP development validation | 199 tasks |
| DPO optimizer steps | 16 |
| Effective batch size | 32 |
| Beta | 0.1 |
| Runtime dropout | disabled for every DPO/control branch |
| Selection priority | worst-task retention, then aggregate capability and preference metrics |

The search compared:

1. sigmoid DPO, learning rate `2e-6`;
2. sigmoid DPO, learning rate `4e-6`;
3. sigmoid DPO with a `0.25` chosen-NLL anchor, learning rate `4e-6`;
4. a 16-step update-matched continued-SFT control;
5. a 32-step token-proxy continued-SFT control.

## Selected branch scorecard

| Metric | Frozen parent | Selected DPO | Delta |
|---|---:|---:|---:|
| Synthetic exact match | 69.79% | **70.31%** | **+0.52 pp** |
| Synthetic strict correct | 64.58% | **64.58%** | **0.00 pp** |
| Synthetic valid format | 84.64% | **83.59%** | **−1.04 pp** |
| SVAMP exact match | 75.38% | **76.38%** | **+1.01 pp** |
| SVAMP strict correct | 18.09% | **20.60%** | **+2.51 pp** |
| SVAMP valid format | 21.11% | **23.62%** | **+2.51 pp** |
| Parse success | 100.0% | **100.0%** | 0.00 pp |

The format change remained inside the preregistered retention envelope. Synthetic length was `0.996×` the parent and SVAMP length was `1.025×`, so the measured preference signal was not explained by a large length shift.

## Preference optimization signal

Both the DPO branch and selected extra-SFT control were scored on the same 128 frozen validation pairs against the same reference adapter.

| Branch | Chosen-over-rejected accuracy | Mean reward margin |
|---|---:|---:|
| Selected DPO | **64.06%** | **+0.00920** |
| Selected extra-SFT control | 38.28% | −0.00987 |
| DPO advantage | **+25.78 pp** | **+0.01907** |

## Safety-first candidate selection

All candidates were eligible. The ranking deliberately prioritized the worst retained task family over raw preference accuracy.

| Candidate | Preference accuracy | Worst-family delta | Eligible | Rank outcome |
|---|---:|---:|---:|---|
| `dpo_sigmoid_lr2e6` | 64.06% | **−2.44 pp** | Yes | **Selected** |
| `dpo_sigmoid_lr4e6` | **80.47%** | −3.28 pp | Yes | Eligible |
| `dpo_sigmoid_nll_a025_lr4e6` | 79.69% | −3.28 pp | Yes | Eligible |

This demonstrates the protocol's central design choice: preference strength is useful only when capability, format, and worst-family retention remain inside locked bounds.

## Systems profile

The selected branch trained **17,432,576 parameters** with NF4 QLoRA, bf16 compute, gradient checkpointing, and precomputed reference log-probabilities. It completed 16 optimizer steps in **36.98 seconds** with **6.91 GiB** peak allocated GPU memory in the recorded RTX 5090 environment.

## Integrity audit

The locked search passed an independent read-only audit:

- search manifest and success marker present;
- selection-lock SHA-256 matched the manifest and immutable marker;
- **59 / 59** search-evidence hashes verified;
- **40 / 40** attested implementation files unchanged;
- candidate and control selection reproduced from the preregistered ranking;
- selection used only the three declared validation roles;
- no raw final-evaluation content was included in the public report.

The sanitized aggregate values are available in [`stage4_development_summary.json`](stage4_development_summary.json).
