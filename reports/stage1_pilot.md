# Stage 1 pilot: QLoRA SFT vertical slice

Date: 2026-08-06  
Status: completed locally; not published

## Question

Can the local pipeline reproducibly generate disjoint data, evaluate an immutable base checkpoint, train a resumable QLoRA adapter, reload it, re-evaluate identical held-out IDs, and produce auditable artifacts?

For this engineering gate, the answer is **yes**.

## Fixed setup

- Base model: `Qwen/Qwen3-0.6B-Base`
- Resolved revision: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
- Quantization: 4-bit NF4 with double quantization and BF16 compute
- Adapter: LoRA rank 16, alpha 32, `all-linear`
- Data: deterministic `arithmetic_v1`, 512 train / 128 validation / 128 test
- Training: 40 optimizer steps, seed 42
- Evaluation: greedy decoding on the same 128 held-out test IDs
- Hardware: NVIDIA GeForce RTX 5090, 32 GB; CUDA 13.0

## Result

> **Post-run audit:** all 128 base generations reached the configured 24-token
> limit, while none of the post-SFT generations did. The historical accuracy
> delta below is therefore length-confounded and must not be treated as a clean
> model-capability comparison. Stage 1 remains valid only as an engineering
> end-to-end smoke test. Stage 2 records token-limit termination explicitly and
> refuses silent prompt truncation.

| Metric | Base | Post-SFT | Delta |
|---|---:|---:|---:|
| Answer exact match | 2.3% | 70.3% | +68.0 pp |
| Strict format and correct | 0.0% | 70.3% | +70.3 pp |
| Parse success | 3.9% | 100.0% | +96.1 pp |

Paired bootstrap 95% CI for exact-match delta: **+60.2 to +76.6 percentage points**. There were 87 wrong-to-right transitions and no right-to-wrong transitions.

Per-family exact-match deltas:

- `add_sub`: +68.8 pp
- `mul_add`: +69.7 pp
- `multiplication`: +93.5 pp
- `parenthesized`: +40.6 pp

Training took 25.1 seconds after preprocessing and used 1.46 GiB peak allocated GPU memory. The final adapter has 10,092,544 trainable parameters (2.61% of the quantized PEFT model parameter count reported by the trainer).

## Evidence saved locally

The raw local run is `runs/20260806_180052_qwen3-0.6b-arithmetic-v1/`. It contains:

- resolved configuration and environment snapshot;
- stage timings and immutable model revision in `run_manifest.json`;
- raw baseline and post-SFT generations;
- aggregate and per-family metrics;
- full checkpoints with optimizer, scheduler, Trainer state, and RNG state;
- final LoRA adapter;
- hashes for the principal artifacts;
- a `SUCCESS` marker.

## Interpretation boundary

This experiment does **not** demonstrate a general reasoning gain. The base model rarely followed the requested output format, while the adapter learned that format reliably. The post-SFT model also made genuine arithmetic errors; for example, it returned `FINAL_ANSWER=-16` for `(-7 + 5) * -8`, whose correct value is `16`.

The result therefore demonstrates a working post-training pipeline and adaptation to a narrow synthetic distribution, but its original metric comparison is not methodologically clean. Capability claims require external held-out benchmarks, prompt and decoding controls, matched baselines, and multiple training seeds in later stages.

## Next gate

Stage 2 should keep this pipeline fixed while adding a 1.7B base model, validation-only tuning, GSM8K-style held-out math evaluation, and a continued-SFT matched baseline. DPO should begin only after that baseline is stable.
