# ARC 2026 — NVARC + TRM

**Offline ARC-AGI-2 inference with two independent neural solver families**

This repository contains an experimental ARC-AGI-2 inference system combining two independent solver families:

* **NVARC** — a neural ARC solver derived from the public *Failed in AIMO* v1 notebook and the Sorokin Qwen model.
* **TRM** — a Tiny Recursive Model based on Samsung SAIL Montreal's TinyRecursiveModels implementation and the public `cpmpml/arc-prize-trm-031` checkpoint.

The system is designed for **offline inference under a four-GPU L4 execution budget**, with GPU scheduling, solver handoff, independent candidate generation, and label-free evidence-based selection.

> **Important:** The measured-input independence projection reported for this candidate is **40.0419%**. This is **not an ARC-AGI-2 competition score** and must not be interpreted as a leaderboard result. Promotion requires a completed Kaggle rerun and a corresponding leaderboard receipt.

---

## Overview

The candidate runs two solver families with different computational roles.

### NVARC

Three NVARC workers initially occupy:

```text
GPU 0 → NVARC worker 0
GPU 1 → NVARC worker 1
GPU 2 → NVARC worker 2
```

The tasks are ordered using an **input-only estimate of computational work**, based on:

* training input size,
* training output size,
* estimated test output size,
* augmented token work,
* estimated decoding work.

No evaluation labels are used to construct this ordering.

### TRM

While the first three NVARC workers run, GPU 3 is reserved for TRM:

```text
GPU 3 → TRM training
```

TRM runs for up to:

```text
4,000 epochs
```

and exports two receipts:

```text
TRM @ 2,000 epochs
TRM @ 4,000 epochs
```

After TRM completes, GPU 3 is released and becomes the delayed fourth NVARC worker:

```text
GPU 3 → NVARC worker 3
```

The handoff is coordinated through a filesystem marker so that the fourth NVARC worker does not claim GPU 3 while TRM still owns it.

---

## Solver Architecture

The high-level execution flow is:

```text
                    ARC-AGI-2 Tasks
                           │
                           ▼
              Input-only work estimation
                           │
                           ▼
                 Lowest-work-first queue
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
      NVARC Workers                  TRM Training
        GPU 0-2                         GPU 3
            │                             │
            │                        2,000 epochs
            │                        4,000 epochs
            │                             │
            │                             ▼
            │                       GPU 3 released
            │                             │
            └──────────────┬──────────────┘
                           ▼
                  NVARC Worker 3
                       GPU 3
                           │
                           ▼
                 Candidate generation
                           │
                           ▼
             Label-free evidence selector
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     NVARC rank 1     TRM rank 1       Fallbacks
          │                │                │
          └────────────────┴────────────────┘
                           │
                           ▼
                    submission.json
```

---

## Candidate Selection

The final candidate uses a **label-free evidence selector**.

The selection policy combines evidence from the independent solver families rather than simply selecting a single model's output.

The intended priority is:

1. **Cross-solver agreement**
2. **Rules that are exact on all available training pairs**
3. **NVARC rank-one candidate**
4. **Final TRM rank-one candidate**
5. **Distinct rank-two fallbacks when required to avoid duplicate attempts**

The final scored attempts are therefore based on:

```text
NVARC rank 1
TRM rank 1
```

with distinct NVARC/TRM rank-two candidates retained as fallbacks for duplicate cases.

The final merge stage uses an evidence policy and produces a selection receipt.

---

## Label-Free Evaluation Design

A key constraint of this candidate is:

> **Evaluation labels are never read during inference.**

In evaluation mode, the notebook loads:

```text
arc-agi_evaluation_challenges.json
```

and uses the corresponding solutions only for **post-inference validation/benchmarking**.

The inference process itself does not use evaluation labels to select predictions.

For competition reruns, the system switches to:

```text
arc-agi_test_challenges.json
```

and produces the final competition submission.

---

## Work Ordering

This experimental sibling uses an **input-only lowest-work-first queue**.

For each ARC task, the estimated work includes:

```text
training input cells
+ training output cells
+ estimated test output cells
```

with the training component scaled for augmentation and the decoding component scaled according to estimated output size.

Conceptually:

```text
estimated work =
    16 × (training input + training output)
    + 8 × estimated test output
```

The estimated test output is derived from the median training input/output size ratio.

Tasks are then sorted by:

```text
estimated work
estimated output size
total training size
test input size
task ID
```

This ordering is based only on information available from the task inputs/training examples.

---

## Alternative Variants

This repository represents one experimental sibling of the ARC 2026 system.

Other variants are intentionally preserved separately:

### Exact Sorted-Order Control

A control variant preserving the exact sorted task ordering.

### Aggressive Reference

An aggressive reference configuration preserved for comparison against the evidence-selection candidate.

These variants allow the effect of queue ordering and candidate-selection policy to be evaluated independently.

---

## Runtime

The intended environment is:

```text
Offline execution
4 × NVIDIA L4 GPUs
Kaggle ARC-AGI-2 environment
```

GPU allocation:

| GPU   | Initial Role   | Later Role     |
| ----- | -------------- | -------------- |
| GPU 0 | NVARC worker 0 | NVARC          |
| GPU 1 | NVARC worker 1 | NVARC          |
| GPU 2 | NVARC worker 2 | NVARC          |
| GPU 3 | TRM training   | NVARC worker 3 |

The execution is designed to stop approximately **10 minutes before the 12-hour rerun limit**.

Internet access is disabled during the intended execution.

---

## TRM Configuration

During competition rerun mode, TRM is launched on physical GPU 3 with:

```text
TRM_WORLD_SIZE       = 1
TRM_EPOCHS           = 4000
TRM_EVAL_INTERVAL    = 2000
TRM_GLOBAL_BATCH_SIZE = 112
TRM_LR                = 0.0000875
TRM_WARMUP_STEPS      = 229
TRM_NUM_AUG          = 128
OMP_NUM_THREADS       = 4
```

The runtime generates:

```text
trm_submission_early.json
trm_submission_final.json
```

corresponding to the intermediate and final TRM checkpoints.

---

## NVARC Runtime

NVARC workers are launched through multiprocessing.

Each worker receives a portion of the shared task queue and is assigned a physical GPU through:

```text
CUDA_VISIBLE_DEVICES
```

The first three workers begin independently while GPU 3 remains reserved for TRM.

The final NVARC worker waits for the TRM handoff marker before starting.

This avoids simultaneous ownership of GPU 3 by TRM and NVARC.

---

## Competition vs Evaluation Mode

The notebook distinguishes between local/evaluation execution and competition rerun execution.

### Evaluation mode

Uses:

```text
arc-agi_evaluation_challenges.json
arc-agi_evaluation_solutions.json
```

The evaluation solutions are loaded only for validation and benchmarking after candidate generation.

### Competition rerun

Uses:

```text
arc-agi_test_challenges.json
```

No test labels are available to the inference pipeline.

The final output is:

```text
submission.json
```

and the NVARC intermediate submission is retained as:

```text
nvarc_submission.json
```

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

* NVARC output generation
* TRM intermediate output
* TRM final output
* final candidate selection
* TRM execution status
* GPU handoff
* solver agreement/selection decisions

---

## Independence Projection

This candidate reports a measured-input independence projection of:

```text
40.0419%
```

This value is a **measurement associated with the experimental independence analysis**.

It is **not**:

* an ARC-AGI-2 leaderboard score,
* a Kaggle competition score,
* a validation accuracy,
* or evidence of final competition performance.

A competition claim requires an actual completed Kaggle rerun and the resulting leaderboard receipt.

---

## Provenance

### NVARC

NVARC is derived from:

* Koushik Rudra's public **Failed in AIMO version 1** notebook
* the Apache-2.0 Sorokin Qwen model

The inherited components retain their original licensing.

### TRM

TRM uses:

* Samsung SAIL Montreal's **TinyRecursiveModels** source
* the public `cpmpml/arc-prize-trm-031` checkpoint

The referenced TinyRecursiveModels source is licensed under MIT, while the referenced public checkpoint is distributed under CC0.

### Original orchestration

The orchestration and integration introduced for this project are:

```text
Copyright 2026 Christopher D. Aleman
MIT-0
```

Inherited components retain their respective licenses.

---

## License

The original orchestration is released under:

```text
MIT-0
```

However, this repository incorporates components with their own licenses.

Therefore:

> **Do not treat the entire repository as having a single inherited license for every component.**

The applicable license for each inherited component remains the license under which that component was originally released.

See the provenance section and the corresponding upstream sources for component-specific licensing.

---

## Reproducibility Notes

This project is intentionally designed around an offline execution environment.

The intended constraints are:

```text
Internet: disabled
GPUs: 4 × NVIDIA L4
Execution budget: < 12 hours
Safety margin: ~10 minutes
```

The system also relies on the competition/runtime environment providing the expected ARC-AGI-2 datasets and, for the TRM rerun, the attached TRM runtime source and local wheels.

The TRM competition rerun expects the attached source package containing:

```text
trm_runtime.py
bootstrap_runtime.py
merge_agreement.py
wheels/
```

---

## Repository Structure

A recommended repository structure is:

```text
arc-2026-nvarc-trm/
│
├── notebooks/
│   ├── fork-of-arc-2026.ipynb
│   ├── sorted-order-control.ipynb
│   └── aggressive-reference.ipynb
│
├── src/
│   ├── starter.py
│   ├── arc_solver.py
│   ├── arc_loader.py
│   ├── arc_decoder.py
│   └── ...
│
├── trm/
│   ├── trm_runtime.py
│   ├── merge_agreement.py
│   └── ...
│
├── receipts/
│   ├── trm-2000.json
│   ├── trm-4000.json
│   └── selector-receipt.json
│
├── submissions/
│   ├── nvarc_submission.json
│   ├── trm_submission_early.json
│   ├── trm_submission_final.json
│   └── submission.json
│
├── LICENSE
└── README.md
```

The exact repository layout may differ from the Kaggle notebook layout.

---

## Experimental Status

**Status: Experimental ARC-AGI-2 candidate**

The system is intended as a research/competition experiment combining:

* independent neural solver families,
* offline execution,
* multi-GPU scheduling,
* delayed GPU handoff,
* input-only workload estimation,
* solver agreement,
* training-pair rule evidence,
* duplicate-aware candidate selection.

The reported **40.0419% independence projection should not be presented as a competition score**.

Final competition performance should only be reported after the corresponding Kaggle rerun and leaderboard receipt have been completed.

---

## Acknowledgements

This project builds upon publicly available work from:

* Koushik Rudra / **Failed in AIMO**
* Sorokin / **Qwen**
* Samsung SAIL Montreal / **TinyRecursiveModels**
* `cpmpml/arc-prize-trm-031`

All inherited components remain subject to their respective licenses.

---

## Citation

If you use the orchestration or experimental methodology from this repository, please preserve the provenance and licensing information for the inherited components.

```text
ARC 2026 — NVARC + TRM
Offline ARC-AGI-2 inference with independent neural solver families,
multi-GPU scheduling, and label-free evidence selection.

Copyright 2026 Christopher D. Aleman
MIT-0
```
