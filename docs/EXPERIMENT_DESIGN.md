# Experiment design

## Objective

Reasoning Post-Training Lab studies how parameter-efficient post-training changes three coupled properties of an open-weight language model:

1. task capability;
2. response-contract compliance;
3. preference alignment.

The system is designed to separate those signals. Every stage uses explicit data roles, matched baselines, frozen selection rules, and auditable lineage.

## Stage structure

### Stage 1 — compact end-to-end baseline

The first stage validates the complete local workflow: deterministic task generation, QLoRA SFT, checkpointing, greedy evaluation, structured parsing, metric aggregation, and report generation.

### Stage 2 — controlled QLoRA pilot

Stage 2 moves to Qwen3-1.7B and introduces a public reasoning benchmark, exact dataset revisions, a continued-SFT comparator, paired example-level analysis, and a readiness gate.

### Stage 3 — reasoning-contract SFT search

Stage 3 searches prompt contracts, data mixtures, and SFT objectives across three training seeds. Selection uses only development roles. The winning recipe is frozen in a SHA-256 selection lock before a one-time held-out evaluation.

### Stage 4 — DPO with matched controls

Stage 4 constructs deterministic chosen/rejected pairs across seven reasoning families. DPO variants are compared with update-matched and token-proxy continued-SFT controls initialized from the same parent. Candidate selection requires both preference separation and capability retention.

## Data roles

The pipeline assigns a semantic role to every split and records it in the dataset manifest.

| Role class | Purpose | Permitted use |
|---|---|---|
| Train | Parameter updates | Training only |
| Development / validation | Hyperparameter and candidate selection | Search and diagnostics |
| Confirmation | One-time locked evaluation | Evaluation after selection lock |
| Quarantine | Audit and provenance | Never candidate selection |

Integrity checks operate on file hashes, row-ID-set hashes, normalized prompt hashes, and semantic task hashes. This catches both byte-level changes and content reuse under renamed IDs.

## Preference-pair construction

Stage 4 preference pairs are generated from verified multi-step solutions. A rejected response contains one deterministic, step-local corruption while preserving the surrounding response structure. The generator records:

- task family, domain, and difficulty;
- corrupted step;
- higher/lower error direction;
- expected final answer;
- chosen and rejected response contracts;
- stable identity and semantic hashes.

The data audit checks role disjointness, semantic uniqueness, prior-experiment reservation, balanced corruption direction, and resistance to simple response-length shortcuts.

## Selection policy

The Stage 4 selector evaluates each candidate against locked thresholds for:

- synthetic and external exact-match retention;
- worst-family retention;
- preference accuracy and positive reward margin;
- format validity and strict correctness;
- parse success;
- generation-limit rate;
- generated-length ratio.

Only eligible candidates enter ranking. Ranking prioritizes worst-task retention before aggregate capability and preference metrics, making the chosen branch robust across task families rather than optimized for a single headline score.

## Matched baselines

Two continued-SFT controls make the DPO comparison interpretable:

- **update-matched:** the same number of optimizer steps as DPO;
- **token-proxy:** twice the SFT steps as a closer proxy for DPO's chosen/rejected token exposure.

All branches start from the same frozen parent, use fresh optimizer/scheduler state, and disable runtime dropout during the comparison. Preference scoring uses the same frozen reference adapter and tokenizer/EOS construction.

## Evaluation

Generation is deterministic. Metrics include:

- answer exact match;
- valid final-answer format;
- strict correctness;
- parse success;
- generated-token statistics and limit-hit rate;
- per-family and per-difficulty slices;
- preference reward accuracy, margin, median, and tie rate;
- paired wrong-to-right / right-to-wrong transitions;
- paired bootstrap confidence intervals.

## Provenance model

Each run records:

```text
configuration hash
  + exact model and dataset revisions
  + deterministic data identities
  + source-code attestation
  + environment snapshot
  + adapter lineage and parameter manifest
  + metrics and selection evidence
  = immutable experiment record
```

Selection locks anchor the precise artifacts needed for the next phase. Confirmation exposure is managed through an atomic claim registry, preventing the same final rows from silently becoming fresh evidence twice.
