# Reproducibility status

## Verified by static audit

- Python source passes `python3 -m compileall -q .`.
- The loader implements video-level grouping, stride-3 sampling, seed 42, and
  80/10/10 split defaults.
- The local dataset has 23 classes and 27,370 image files; stride-3 sampling
  yields 13,230 items.
- Dense and BlockTT training entry points exist.
- BlockTT replacement has maximum-rank configuration and leaves `head` dense.
- Test reports and confusion-matrix image artifacts exist under
  `results/`.

## Not verified

- Model instantiation, because PyTorch is not installed in the audit
  environment.
- Any training, checkpoint loading, or full evaluation.
- Exact train/validation/test sample counts from the installed
  `scikit-learn` splitter.
- Headline parameter counts, compression ratio, MCC, or weighted F1.
- Dataset redistribution rights, checkpoint rights, and publication metadata.

## Release blockers

1. Install and pin a tested Python/PyTorch environment.
2. Add a repository-level dataset preparation/validation command.
3. Add explicit dense, BlockTT, evaluation, and inference CLIs.
4. Resolve the metric-definition mismatch (macro helper versus weighted
   headline F1) and add specificity/MCC/parameter reporting.
5. Record checkpoint provenance, seeds, device/software versions, and exact
   selected epochs.
6. Resolve contradictory parameter and accuracy claims against actual
   checkpoints and a single evaluation run.
7. Verify licenses before selecting a project license or redistributing
   checkpoints, dataset content, or `torch-mpo/`.
