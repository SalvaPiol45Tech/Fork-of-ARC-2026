# ARC 2026 — NVARC + TRM

**Offline ARC-AGI-2 inference with two independent neural solver families**

This repository contains an experimental ARC-AGI-2 inference system combining two independent solver families:

* **NVARC** — a neural ARC solver derived from the public *Failed in AIMO* v1 notebook and the Sorokin Qwen model.
* **TRM** — a Tiny Recursive Model based on Samsung SAIL Montreal's TinyRecursiveModels implementation and the public `cpmpml/arc-prize-trm-031` checkpoint.

The system is designed for **offline inference on a single Kaggle GPU**, with managed GPU scheduling between TRM training and NVARC inference.

> **Important:** The measured-input independence projection reported for this candidate is **40.0419%**. This is **not an ARC-AGI-2 competition score** and must not be interpreted as a leaderboard result. Promotion requires a completed Kaggle rerun and a corresponding leaderboard receipt.

---

## Overview

The candidate combines two independent neural solver families:

### NVARC

NVARC provides the primary neural ARC solver family.

The experimental system can run multiple NVARC workers through managed scheduling, but the Kaggle execution environment uses **one GPU**, so the workers are executed through sequential/resource-managed GPU scheduling rather than assuming four physical GPUs.

### TRM

TRM uses the same Kaggle GPU for its training phase.

TRM trains for up to:

```text
4,000 epochs
```

and exports two receipts:

```text
TRM @ 2,000 epochs
TRM @ 4,000 epochs
```

After the TRM phase completes, the GPU is released for the NVARC inference phase.

---

## Kaggle Runtime

The intended competition environment is:

```text
Kaggle
Offline execution
1 × NVIDIA L4 GPU
12-hour rerun limit
```

The project is therefore designed around **one physical GPU**, not four simultaneously available GPUs.

The GPU is managed between the two solver families:

```text
                 Kaggle
                   │
             1 × NVIDIA L4
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 │
     TRM training           │
          │                 │
    2,000 epochs            │
    4,000 epochs            │
          │                 │
          ▼                 │
      TRM receipt           │
          │                 │
          ▼                 │
       GPU released         │
          │                 │
          ▼                 │
     NVARC inference        │
          │                 │
          ▼                 │
    Candidate generation    │
          │                 │
          └────────┬────────┘
                   ▼
          Evidence selector
                   │
                   ▼
             submission.json
```

The execution is designed to stop approximately **10 minutes before the 12-hour rerun limit**.

---

## Solver Pipeline

The high-level pipeline is:

```text
ARC-AGI-2 Tasks
       │
       ▼
Input-only work estimation
       │
       ▼
Lowest-work-first queue
       │
       ├───────────────┐
       │               │
       ▼               ▼
     NVARC            TRM
       │          2,000 / 4,000
       │             epochs
       │               │
       │               ▼
       │          Final TRM
       │          candidates
       │               │
       └───────┬───────┘
               ▼
       Label-free evidence
           selection
               │
       ┌───────┴────────┐
       ▼                ▼
  NVARC rank 1      TRM rank 1
       │                │
       └───────┬────────┘
               ▼
      Duplicate-aware
         fallbacks
               │
               ▼
        submission.json
```

---

## Candidate Selection

The final candidate uses a **label-free evidence selector**.

The selection policy combines evidence from the independent solver families.

Priority is given to:

1. **Cross-solver agreement**
2. **Rules that are exact on all available training pairs**
3. **NVARC rank-one candidate**
4. **Final TRM rank-one candidate**
5. **Distinct rank-two NVARC/TRM fallbacks when required for duplicates**

The scored attempts are therefore based on:

```text
NVARC rank 1
TRM rank 1
```

Distinct NVARC/TRM rank-two candidates are retained only as fallbacks for duplicate cases.

---

## Label-Free Evaluation

A key constraint of this candidate is:

> **Evaluation labels are never read during inference.**

The inference process uses the task inputs and training pairs to generate candidates.

The evidence selector uses:

* cross-solver agreement,
* rules discovered from training pairs,
* solver candidate rankings,

without reading evaluation answers to choose predictions.

Evaluation labels, where available, are used for post-inference measurement rather than candidate selection.

---

## Input-Only Work Ordering

This experimental sibling orders NVARC tasks using an **input-only estimate of augmented token and output-decoding work**.

The purpose is to prioritize tasks according to estimated computational cost without using evaluation labels.

The exact sorted-order control and the aggressive reference are preserved as separate experimental notebooks.

---

## Alternative Variants

### Exact Sorted-Order Control

A control variant preserving the exact sorted task ordering.

### Aggressive Reference

An aggressive reference configuration preserved separately for comparison.

### Evidence-Selection Candidate

This candidate combines:

* input-only lowest-work-first scheduling,
* label-free evidence selection,
* cross-solver agreement,
* training-pair exact-rule evidence,
* NVARC rank-one output,
* final TRM rank-one output,
* duplicate-aware rank-two fallbacks.

---

## TRM

TRM is trained on the Kaggle GPU before the GPU is handed to NVARC inference.

The training produces:

```text
TRM @ 2,000 epochs
TRM @ 4,000 epochs
```

The two checkpoints/receipts provide an intermediate and final TRM candidate.

The final TRM candidate is used as the **TRM rank-one** solver output.

---

## NVARC

NVARC is the second independent neural solver family.

Its provenance is based on:

* Koushik Rudra's Apache-2.0 public *Failed in AIMO version 1* notebook
* the Apache-2.0 Sorokin Qwen model

NVARC generates ranked candidate solutions for ARC tasks.

The final selector uses the NVARC rank-one candidate as one of the two primary scored candidates.

---

## Runtime Constraints

```text
Platform:       Kaggle
GPU:            1 × NVIDIA L4
Internet:       Disabled
Execution:      Offline
Time limit:     12 hours
Safety margin:  ~10 minutes
```

The system does **not** assume access to four physical GPUs.

The single GPU is reused between computational phases:

```text
TRM → GPU release → NVARC
```

This makes the candidate compatible with a one-GPU Kaggle execution environment.

---

## Output Artifacts

Important runtime artifacts include:

```text
submission.json

nvarc_submission.json

trm_submission_early.json
trm_submission_final.json

selector-receipt.json

trm-gpu3.log
trm-gpu3-released.json
```

These artifacts provide evidence for:

* NVARC candidate generation
* TRM intermediate output
* TRM final output
* final candidate selection
* solver agreement
* GPU phase transitions
* duplicate handling

---

## Independence Projection

This candidate reports a measured-input independence projection of:

```text
40.0419%
```

This value is **not**:

* an ARC-AGI-2 leaderboard score,
* a Kaggle competition score,
* a validation accuracy,
* or a prediction of final competition performance.

It is an experimental independence measurement.

Promotion of the candidate requires:

```text
Completed Kaggle rerun
        +
Leaderboard receipt
```

Only the resulting competition evidence should be used to report an actual competition score.

---

## Provenance and Licensing

### NVARC

NVARC is derived from:

* Koushik Rudra's Apache-2.0 public *Failed in AIMO version 1* notebook
* the Apache-2.0 Sorokin Qwen model

Inherited components retain their original licenses.

### TRM

TRM uses:

* Samsung SAIL Montreal's MIT **TinyRecursiveModels** source
* the public CC0 `cpmpml/arc-prize-trm-031` checkpoint

Inherited components retain their original licenses.

### Original Orchestration

The original orchestration for this experimental candidate is:

```text
Copyright 2026 Christopher D. Aleman
MIT-0
```

Inherited components remain subject to their respective licenses.

---

## License

The original orchestration is released under:

```text
MIT-0
```

This repository also contains or derives from components distributed under other licenses.

Therefore, the repository should **not** be interpreted as relicensing inherited components.

Each inherited component retains its applicable upstream license.

---

## Reproducibility

The intended execution environment is:

```text
Kaggle
1 × NVIDIA L4
Offline
≤ 12 hours
```

The candidate is designed specifically around the single-GPU constraint.

The core scheduling strategy is:

```text
1. Start TRM on the Kaggle GPU.
2. Train through the required TRM checkpoints.
3. Export the 2,000-epoch and 4,000-epoch receipts.
4. Release the GPU.
5. Run NVARC inference using the same GPU.
6. Generate ranked NVARC candidates.
7. Combine NVARC and TRM candidates.
8. Apply the label-free evidence selector.
9. Resolve duplicate candidates using distinct rank-two fallbacks.
10. Write the final submission.
```

---

## Experimental Status

**Status: Experimental ARC-AGI-2 candidate**

This project combines:

* independent neural solver families,
* single-GPU Kaggle execution,
* managed GPU reuse,
* input-only workload estimation,
* label-free solver selection,
* cross-solver agreement,
* exact training-pair rule evidence,
* duplicate-aware candidate generation.

The reported **40.0419% independence projection is not a competition score**.

A competition result should only be reported after a completed Kaggle rerun and the corresponding leaderboard receipt.

---

## Acknowledgements

This project builds upon publicly available work from:

* Koushik Rudra / *Failed in AIMO*
* Sorokin / Qwen
* Samsung SAIL Montreal / *TinyRecursiveModels*
* `cpmpml/arc-prize-trm-031`

All inherited components remain subject to their respective licenses.

---

## Citation

```text
ARC 2026 — NVARC + TRM

Offline ARC-AGI-2 inference with two independent
neural solver families, single-GPU Kaggle scheduling,
and label-free evidence selection.

Copyright 2026 Christopher D. Aleman
MIT-0
```
