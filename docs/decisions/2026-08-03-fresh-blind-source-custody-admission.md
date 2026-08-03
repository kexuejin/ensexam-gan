# Fresh Blind Source And Custody Admission

## Terminal

`NOT_FOUND_WITHIN_BOUND`

No source is admitted. The result parks the fresh-data lane as
`external_data_prerequisite`; it does not reject Candidate 5 and does not
authorize inference, training, synthetic blind-set construction, or routing.

## Bounded Review

The already consumed HW5K official test is context, not a new candidate:

- HW5K / DocUnfold — <https://huggingface.co/datasets/lkljty/HW5K> and
  <https://github.com/CXH-Research/DocUnfold>. The 525 official test pairs were
  registered, scored, and audited for current-primary on 2026-07-26. They cannot
  be reused for a later promotion claim.

Five source families were checked against existing repository custody records:

| Source | Pair/Task fit | Prior-use or custody result | Admission |
| --- | --- | --- | --- |
| SCUT-EnsExam — <https://github.com/SCUT-DLVCLab/SCUT-EnsExam> | Same-task paired exam pages | Train/test pages appear throughout training, metrics, selector, and review history | Ineligible |
| ExamInk-Seg — <https://huggingface.co/datasets/ynyg/ExamInk-Seg> | Paired source/target/mask and relevant erasure supervision | Local smoke training and target-aware evaluation already used samples; repository tooling treats it as a training source | Ineligible |
| Baidu `dehw` family | Train split has labels/masks; testA has 200 inputs without clean labels | Train is documented for train/validation; testA cannot score residual or overerase | Ineligible |
| Handwriting Inpainting Dataset | Handwritten text/mask inpainting resources | No independently captured paired printed-document input/clean-target blind root; constructing one would be synthetic and target-aware | Task/custody mismatch |
| HCCD | Paired degraded/clean handwritten-document enhancement | Enhancement/restoration is not handwriting removal over printed content | Task mismatch |

The first public sourcing audit reached the same local conclusion before HW5K
was acquired: no identified source was immediately admissible, and a new
external root plus provenance was required
(`docs/decisions/2026-07-24-public-reserved-set-sourcing-audit.md:18`). The later
mounted-root audit records that SCUT, ExamInk-Seg, and Baidu train are consumed
or development sources and Baidu testA lacks clean labels
(`docs/blind-training-and-validation-plan.md:174`).

## Metadata Verification Limit

No dataset payload was downloaded and no image or target was opened. Attempts to
refresh official GitHub and Hugging Face metadata failed because the execution
environment could not resolve the public hosts, and no application browser was
available. Consequently, uncertain or newly published sources were not admitted
from memory or secondary descriptions.

This terminal is bounded, not globally exhaustive: it means no admissible source
exists among the five project-known families under verifiable custody evidence.

## Sole Next Action

Proceed with the data-independent M2 specialist product-contract freeze. The
fresh-data lane can later resume only from one of these inputs:

1. a user-controlled external root with paired inputs/clean targets and truthful
   no-training/no-selection/no-review provenance; or
2. a newly preregistered public-source shortlist after official metadata access
   is restored.

Either input must pass preflight, content-identity contamination audit, formal
registration, and isolation validation before any Candidate 5 inference. If M2
finishes while neither input exists, the product program becomes
`blocked_external_prerequisite` rather than opening a router branch.

## Verification

- Reviewed source families: five, excluding consumed HW5K context.
- Dataset payload downloads: zero.
- Training/inference/image review: zero.
- Checkpoints, predictions, thresholds, and current-primary changes: zero.
