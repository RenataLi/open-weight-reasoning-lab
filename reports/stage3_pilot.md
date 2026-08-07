# Stage 3: locked multi-seed SFT confirmation

## Abstract

Stage 3 asked whether a reasoning-aware SFT mixture could restore the capability lost by the answer-only Stage 2 recipe while retaining a machine-checkable output contract. Four-candidate searches were run across three seeds, candidate selection used validation data only, and a SHA-256 selection lock was written before a single confirmation pass on fresh data.

The selected QLoRA recipe retained Base accuracy within preregistered non-inferiority limits across all three seeds. On 512 fresh synthetic tasks, Base scored 99.4% and SFT averaged 98.4% ± 0.3%; valid final-answer formatting changed from 22.7% to 100%. On 1,191 previously unused GSM8K test tasks, Base scored 65.5% and every SFT seed scored 65.2%. All paired confidence-bound, format, parsing, token-limit, continued-SFT, and seed-stability checks passed.

Status: **READY TO BUILD A SCOPED DPO PILOT**. This is evidence of a stable SFT foundation, not evidence that DPO improves reasoning.

## Research question and locked gate

The primary question was non-inferiority: can SFT learn a concise reasoning-plus-final-answer contract without the large format-independent capability loss observed in Stage 2?

The confirmation policy was fixed before viewing confirmation metrics:

- synthetic exact-match regression no worse than `-5 pp` for every seed;
- GSM8K exact-match regression no worse than `-3 pp` for every seed;
- paired-bootstrap 95% CI lower bounds above the same `-5 pp` / `-3 pp` guardrails;
- synthetic valid final-answer format at least `95%`;
- parse success at least `99%` on both task sets;
- generation-limit rate at most `5%` synthetic and `10%` external;
- continued-SFT regression relative to its SFT parent no worse than `-5 pp`;
- between-seed exact-match range at most `3 pp` synthetic and `4 pp` external.

The external `#### number` format rate is reported descriptively rather than gated: correctness is scored independently with a benchmark-specific final-number fallback, and parse success remains a required check.

## Stage 2 diagnosis

Stage 2 used answer-only targets. It changed synthetic exact match from `95.3%` to `71.9%`, while strict format changed from `0%` to `100%` and the mean generated response shortened from about `108` to `6` tokens. The largest losses occurred in multi-step families. This supports, but does not prove, the hypothesis that answer-only supervision suppressed visible computation needed by this 1.7B Base model.

Stage 3 therefore varied reasoning-target share and prompt/response contract while keeping the model, LoRA shape, example count, step budget, and seeds controlled within each search. Reasoning completions contain more supervised tokens, so the search is example- and step-matched, not token- or compute-matched.

## Data integrity

| Role | Count | Source | SHA-256 |
|---|---:|---|---|
| Synthetic validation | 512 | deterministic `arithmetic_v2` extension | recorded in the search run |
| GSM8K development | 256 | fixed, train-derived split | recorded in the search run |
| Synthetic confirmation | 512 | new deterministic IDs | `1913068be66d4d27991e79ec2ab89e39e0e613e653bf84c3336659d606c11f0f` |
| GSM8K confirmation | 1,191 | official test complement after excluding Stage 2's 128 examples | `a575174cb5d5f9a8adec03319b869ee902a6d44ba77ebc070a212d72ff7a29c9` |

Confirmation row-ID set hashes are `123b91822bcbb0be3fcaab10bb7d0eebed1c0954b0f758fd6627053c471f0b53` synthetic and `12a858787fa44a98c7cb6186cd24801c6333a386a576dafe81e843441503cded` GSM8K. Independent checks found zero ID overlap between selection and confirmation, and zero overlap between Stage 2 held-out IDs and the Stage 3 confirmation sets.

The external dataset is `openai/gsm8k/main` at exact revision `740312add88f781978c0658806c59bc2815b9866`. It was never used for SFT; only the fixed train-derived development split was visible during selection.

## Search protocol and negative results

Three sequential protocols were retained. v1 and v2 failed, which prevented confirmation from being opened. v3 changed the primary synthetic contract to concise reasoning followed by a final `FINAL_ANSWER=<integer>` line.

| Search | Output contract | Outcome | Best diagnostic result |
|---|---|---|---|
| v1 | Shared prompt, answer-only evaluation | No eligible candidate | 75% reasoning: only `-0.4 pp` worst retention, but `14.5%` mean strict correctness |
| v2 | Contract-conditioned prompt, answer-only evaluation | No eligible candidate | 90% reasoning: `-15.0 pp` worst retention and `38.6%` mean strict correctness |
| v3 | Reasoning then final answer | Two eligible candidates | Contract-conditioned mixtures resolved the validation retention/format trade-off |

The v1/v2 synthetic Base score was 93.9%, while v3 Base was 98.4%, because the output contract and prompts differ. These values must not be compared as if they were one benchmark.

The locked v3 candidate table was:

| Candidate | All-seed eligible | Worst-seed retention | Mean synthetic strict correct | Mean GSM8K-dev exact |
|---|---:|---:|---:|---:|
| `shared75_nll_lr5e5` | no | -1.8 pp | 91.4% | 76.6% |
| `conditioned75_nll_lr5e5` | yes | -1.8 pp | 96.7% | 77.7% |
| `conditioned90_nll_lr5e5` | yes | -1.6 pp | 96.9% | 77.0% |
| `shared75_nll_lr2e5` | no | -8.2 pp | 79.8% | 76.6% |

The preregistered ranking prioritized worst-seed retention, so it selected `conditioned90_nll_lr5e5` rather than the candidate with the highest mean GSM8K-dev score. The final v3 matrix is NLL-only; no final objective-ablation claim is made. An earlier exploratory protocol included DFT but produced no eligible candidate under its different contract.

## Immutable selection

- Candidate: `conditioned90_nll_lr5e5`.
- Mixture: 90% verified concise reasoning, 10% direct response.
- Prompt rule: `contract_conditioned`.
- Loss: completion-only NLL.
- Learning rate / steps: `5e-5` / `32`.
- QLoRA: NF4 double quantization, BF16 compute, rank 16, alpha 32, dropout 0.05, all-linear targets, gradient checkpointing.
- Seeds: `101`, `202`, `303`.
- Base revision: `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Resolved config hash: `ebfdb4683c864517fe7f9ad0e214d7fcdafb195b019213d850c48b786ec86689`.
- Selection lock hash: `20b23bc3b08bd19ec948da6384f4309179f2374423a3e7cdb96342d4a9df97bf`.

Locked adapter hashes were:

| Seed | Adapter SHA-256 |
|---:|---|
| 101 | `2d21f205d13de9eabd7e3153109ad7f57680dda8dac23d774d4ba8c81cb66a7b` |
| 202 | `dd9dea70fe7a45b1c102c113d2661f45f41078c64c5f05d1fcfc5a692fed506a` |
| 303 | `ffa4f4095dcfad9e1ac64eda15269c8639cdcae583dad5ae8cf2f9ff98628ec3` |

The lock timestamp was `2026-08-06T18:36:43.874946Z`; the test-start marker was `2026-08-06T18:37:43.209633Z`. `test_metrics_used_for_selection` is false.

## Held-out confirmation

### Per-seed results

| Model | Seed | Synthetic exact | Synthetic format | GSM8K exact | GSM8K final `####` | GSM8K strict correct | GSM8K token-limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | — | 99.4% | 22.7% | 65.5% | 88.9% | 60.4% | 3.7% |
| SFT | 101 | 98.6% | 100.0% | 65.2% | 89.6% | 59.9% | 2.8% |
| SFT | 202 | 98.0% | 100.0% | 65.2% | 88.6% | 59.4% | 2.9% |
| SFT | 303 | 98.4% | 100.0% | 65.2% | 87.6% | 58.6% | 2.4% |
| Continued-SFT | 101 | 98.4% | 100.0% | 65.2% | 92.2% | 61.8% | 2.9% |
| Continued-SFT | 202 | 98.4% | 100.0% | 66.3% | 90.5% | 61.9% | 2.2% |
| Continued-SFT | 303 | 98.6% | 100.0% | 65.8% | 91.6% | 62.0% | 2.5% |

Parse success was 100% in every cell. Synthetic generation never hit its token limit.

### Aggregate results

| Metric | Base | SFT mean ± sample SD | Continued-SFT mean ± sample SD |
|---|---:|---:|---:|
| Synthetic answer exact | 99.4% | 98.4% ± 0.3% | 98.5% ± 0.1% |
| Synthetic strict correct | 22.5% | 98.4% ± 0.3% | 98.5% ± 0.1% |
| Synthetic format valid | 22.7% | 100.0% ± 0.0% | 100.0% ± 0.0% |
| Synthetic mean generated tokens | 97.4 | 34.1 ± 0.0 | 34.1 ± 0.0 |
| GSM8K answer exact | 65.5% | 65.2% ± 0.0% | 65.8% ± 0.6% |
| GSM8K strict correct | 60.4% | 59.3% ± 0.6% | 61.9% ± 0.1% |
| GSM8K final-marker format | 88.9% | 88.6% ± 1.0% | 91.4% ± 0.9% |
| GSM8K mean generated tokens | 115.4 | 103.9 ± 0.9 | 100.7 ± 1.4 |

### Paired Base-to-SFT changes

| Seed | Synthetic delta | Synthetic paired 95% CI | GSM8K delta | GSM8K paired 95% CI | GSM wrong→right / right→wrong |
|---:|---:|---:|---:|---:|---:|
| 101 | -0.78 pp | [-1.76, 0.00] | -0.34 pp | [-2.43, +1.68] | 75 / 79 |
| 202 | -1.37 pp | [-2.54, -0.20] | -0.34 pp | [-2.43, +1.76] | 77 / 81 |
| 303 | -0.98 pp | [-2.15, 0.00] | -0.34 pp | [-2.43, +1.76] | 78 / 82 |

The synthetic result is non-inferior, not improved. Seed 202's interval excludes zero and indicates a small real loss on this sample, but its lower bound remains far above the locked `-5 pp` guardrail. On GSM8K, every SFT seed changes only four more Base-correct examples to wrong than vice versa, while the interval excludes any regression larger than the locked `-3 pp` bound.

Identical aggregate GSM8K scores do not imply identical behavior: each adapter flips roughly 154–160 task outcomes. Transition counts therefore remain required diagnostics.

### Worst strata

All seeds retained 100% on difficulty 1 and 99.4% on difficulty 2. Difficulty 3 was consistently lower: `96.6%`, `95.0%`, and `96.1%` versus Base `98.3%`. Multiplication was the worst family, with Base-to-SFT deltas from `-1.6` to `-3.3 pp`. A DPO pilot should preregister both strata as abort checks.

## Continued-SFT control

Each locked SFT adapter branched into a 16-step continued-SFT run on the same locked 256-example continuation mixture. Every branch records:

- exact parent adapter hash and Base revision;
- fresh optimizer and scheduler;
- initial global step `0`, final global step `16`;
- no checkpoint resume;
- continuation data hash `dcebb4880d3399ec4688f7af9dc26d32b9383cdebc007674dda7be109a29cb4f`.

Synthetic SFT-to-continued deltas were `-0.20`, `+0.39`, and `+0.20 pp`. GSM8K deltas were `0.00`, `+1.18`, and `+0.67 pp`; all intervals were compatible with small changes. This is a stability stress test, not evidence that more SFT or future DPO helps.

## Gate audit

| Check | Observed | Threshold | Result |
|---|---|---|---:|
| Lock predates test | ~59 seconds | strictly earlier, no test metrics in selection | PASS |
| Locked seeds completed | 3 / 3 | 3 / 3 | PASS |
| Synthetic exact regression | -0.78 to -1.37 pp | no worse than -5 pp | PASS |
| GSM8K exact regression | -0.34 pp for each seed | no worse than -3 pp | PASS |
| Synthetic CI lower bound | -2.54 pp worst | above -5 pp | PASS |
| GSM8K CI lower bound | -2.43 pp worst | above -3 pp | PASS |
| Synthetic format | 100% for every SFT seed | at least 95% | PASS |
| Parse success | 100% in all SFT cells | at least 99% | PASS |
| Token-limit rate | 0% synthetic; 2.4–2.9% GSM8K | at most 5% / 10% | PASS |
| SFT seed range | 0.59 pp synthetic; 0 pp GSM8K | at most 3 pp / 4 pp | PASS |
| Continued-SFT retention | worst -0.20 pp | no worse than -5 pp | PASS |

Overall: **READY TO BUILD A SCOPED DPO PILOT**.

## Integrity audit and implementation hardening

An independent post-run audit recomputed counts from raw predictions, verified selection/confirmation role tags, checked all locked data and adapter hashes, found zero split overlap, and verified every continued-SFT parent and fresh-state field. The recorded confirmation numbers match the raw rows exactly.

The audit also identified implementation weaknesses in the original Stage 3 runner: repeated confirmation was not blocked, the selection lock and final artifact maps did not cover every evidence file, and the gate trusted some provenance fields instead of rechecking them. The code was hardened after the experiment for future runs without changing thresholds or recomputing model outputs. The completed run remains historical evidence; this report states separately what its original lock covered and what was verified post hoc.

The confirmation sets are now classified as exposed. They must not be reused for DPO hyperparameter selection or claimed as fresh in a later stage.

## Interpretation and next experiment

Stage 3 succeeded at capability preservation and contract learning. It did not demonstrate a reasoning gain: all SFT seeds are slightly below Base on synthetic exact match, and GSM8K is effectively flat within uncertainty. The synthetic confirmation set also has a severe ceiling effect because Base already solves 509 of 512 tasks.

A valid DPO feasibility pilot therefore needs:

1. one initialization chosen a priori, not the best seed after confirmation;
2. preference pairs generated only from training-only or new prompts;
3. a matched additional-SFT control using the chosen responses;
4. explicit chosen/rejected length-bias, KL, response-length, and token-cap diagnostics;
5. a harder fresh synthetic development set and a separate untouched final holdout;
6. beta, learning rate, seed, decoding, and stop rules locked before final evaluation;
7. abort strata for multiplication, difficulty 3, GSM8K format/strict correctness, and example-level churn.

## Reproduction

Validated local environment: Windows 11, Python 3.12.13, NVIDIA RTX 5090, PyTorch 2.13.0+cu130, Transformers 5.14.1, TRL 1.9.2, PEFT 0.20.0, and bitsandbytes 0.50.0.

Search run ID:

```text
20260806_211000_qwen3-1.7b-stage3-reasoning-contract-search
```

Confirmation run ID:

```text
20260806_213800_qwen3-1.7b-stage3-reasoning-contract-confirm
```

Commands:

```powershell
.\.venv\Scripts\python.exe -m reasoning_lab.cli run-stage3-search `
  --config configs\stage3_reasoning_contract_qwen3_1_7b.yaml

.\.venv\Scripts\python.exe -m reasoning_lab.cli run-stage3-confirm `
  --config configs\stage3_reasoning_contract_qwen3_1_7b.yaml `
  --search-run runs\<completed-search-run>
```

The ignored local run directory contains raw predictions, metrics, training summaries, adapters, `selection_lock.json`, `comparisons.json`, `readiness.json`, manifests, and success/test-start markers. The compact publication-safe values are in [`stage3_summary.json`](stage3_summary.json).

## Primary sources

- [Qwen3 training guide](https://qwen.readthedocs.io/en/latest/training/unsloth.html)
- [Qwen3 official blog](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/en/sft_trainer)
- [Dynamic Fine-Tuning](https://arxiv.org/abs/2508.05629)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [GSM8K](https://arxiv.org/abs/2110.14168)

