# Stage 2 controlled-baseline pilot

> This is a local pilot, not evidence that broad reasoning capability improved. GSM8K is used only for frozen external evaluation and never for fine-tuning.

## Provenance

- Base model: `Qwen/Qwen3-1.7B-Base`
- Exact model revision: `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`
- External dataset: `openai/gsm8k` / `main`
- Exact dataset revision: `740312add88f781978c0658806c59bc2815b9866`
- Synthetic generator: `2.0.0`
- Synthetic train / continuation examples: `1024` / `256`
- Frozen GSM8K test subset: `128` examples, `5`-shot

## Synthetic held-out test

| Metric | Base | SFT | Continued-SFT |
|---|---:|---:|---:|
| Answer exact match | 95.3% | 71.9% | 70.3% |
| Valid output format | 0.0% | 100.0% | 100.0% |
| Parse success | 100.0% | 100.0% | 100.0% |
| Hit generation limit | 0.0% | 0.0% | 0.0% |

## External GSM8K pilot

| Metric | Base | SFT | Continued-SFT |
|---|---:|---:|---:|
| Answer exact match | 67.2% | 75.0% | 74.2% |
| Strict `####` format | 94.5% | 80.5% | 79.7% |
| Parse success | 100.0% | 100.0% | 100.0% |
| Hit generation limit | 1.6% | 1.6% | 1.6% |

The GSM8K scorer accepts a strict final `#### number` when present and otherwise uses a benchmark-specific final-number fallback. Format compliance is reported separately from answer correctness.

## Paired changes

- Synthetic: Base -> SFT: `-23.4 pp` (paired bootstrap 95% CI `[-32.0 pp, -14.8 pp]`; wrong -> right `3`, right -> wrong `33`).
- Synthetic: SFT -> Continued-SFT: `-1.6 pp` (paired bootstrap 95% CI `[-6.2 pp, +3.1 pp]`; wrong -> right `3`, right -> wrong `5`).
- GSM8K: Base -> SFT: `+7.8 pp` (paired bootstrap 95% CI `[+0.8 pp, +14.8 pp]`; wrong -> right `16`, right -> wrong `6`).
- GSM8K: SFT -> Continued-SFT: `-0.8 pp` (paired bootstrap 95% CI `[-5.5 pp, +3.9 pp]`; wrong -> right `4`, right -> wrong `5`).

## Training controls

- Initial SFT optimizer steps: `64`; trainable parameters: `17,432,576`.
- Continued-SFT optimizer steps: `16`.
- Continued-SFT starts from the saved SFT adapter but creates a fresh optimizer, scheduler, and global step; it is a branch, not checkpoint resume.
- Its prompts come from a deterministic continuation split disjoint from train, validation, and test.

## Readiness gate

- **PASS** - Synthetic generation is not length-confounded: worst token-limit rate 0.0%, threshold 5.0%
- **PASS** - External generation is not length-confounded: worst token-limit rate 1.6%, threshold 10.0%
- **PASS** - External parser is reliable: lowest parse-success rate 100.0%, threshold 90.0%
- **PASS** - Frozen external subset is complete: observed counts [128], expected 128
- **PASS** - Initial SFT completed as configured: 64 / 64 optimizer steps
- **FAIL** - SFT preserves format-independent synthetic capability: Base 95.3%, SFT 71.9%, delta -23.4%; allowed regression 5.0%
- **PASS** - Continued-SFT does not materially regress its SFT parent: SFT 71.9%, continued-SFT 70.3%, delta -1.6%; allowed regression 5.0%
- **PASS** - Continued-SFT is a fresh matched branch: parent adapter loaded with fresh optimizer, scheduler, and global step
- **PASS** - Parent adapter lineage is exact: recorded base revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1

Overall status: **NOT READY FOR DPO**.

## Interpretation boundary

Format-independent synthetic exact match changed by `-23.4 pp` from Base to SFT, while strict output-format compliance is reported separately. This distinction prevents format learning from being mislabeled as a reasoning gain.

The synthetic score measures adaptation to a narrow generated arithmetic format. The external score is a small, frozen GSM8K subset and is not a substitute for the full benchmark, multiple seeds, or a hidden contamination-resistant evaluation. A DPO comparison should only be added after this baseline and its evaluation gate remain stable in a larger confirmatory run.
