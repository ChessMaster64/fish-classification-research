# Phase 3 runtime verification

Date: 2026-11-01

## Environment

| Component | Result |
|---|---|
| Python interpreter | PASS: Python 3.12.3 in the verified project environment |
| PyTorch | PASS: 2.12.0.dev20260405+cu128 |
| torchvision | PASS: 0.27.0.dev20260405+cu128 |
| torch-geometric | PASS: 2.8.0.post1 |
| NumPy, scikit-learn, Pillow, OpenCV | PASS: installed and imported successfully |
| matplotlib | PASS: 3.11.0 |
| pytest | BLOCKED: not installed |

No package was installed, upgraded, downgraded, or removed.

## Commands executed

### Static validation

```bash
python -m compileall -q .
```

Result: **PASS**. All release Python files compiled successfully.

### Lightweight tests

```bash
python -m pytest tests/test_metrics_release.py tests/test_checkpoint_compat.py
```

Result: **BLOCKED**:

```text
python: No module named pytest
```

The tests were not executed through pytest. Equivalent test functions were
executed directly and passed.

### Dependency probe

Direct imports of `torch`, `torchvision`, `torch_geometric`, `numpy`,
`sklearn`, `PIL`, `cv2`, and `matplotlib` passed. `pytest` remains absent.

### Checkpoint inspection

Dense and BlockTT checkpoints were loaded read-only with strict state-dict
matching. Both had zero missing and unexpected keys. Their metadata contains
serialized `epoch`, optimizer, scheduler, and `best_acc` fields. No checkpoint
was copied or modified.

## Dataset verification

The original dataset was inspected read-only:

- 23 class directories: **PASS**
- 27,370 source image files: **PASS**
- 8,725 video groups under the implemented filename parser: **PASS**
- stride-3 sampled total: 13,230 images: **PASS**
- filename-derived video grouping: **PASS**
- leakage validation through the release script: **PASS**
- exact train/validation/test image counts: **PASS** (10,557 / 1,360 / 1,313)

The implemented 80/10/10 split has 6,980/872/873 video IDs by partition
cardinality and produces the paper sample counts. No original split manifest
was found.

## Model and import verification

| Component | Status | Reason |
|---|---|---|
| Final dense model construction | PASS | Synthetic forward returned `(1, 23)` |
| Final BlockTT construction | PASS | Synthetic forward returned `(1, 23)` |
| Recovered baseline construction | NEEDS VERIFICATION | Not all historical variants were constructed |
| Release import compatibility | PASS | Release modules imported successfully |
| Parameter counting | PASS | Dense and BlockTT counts computed |
| Compression ratio/reduction | PASS | Computed from constructed models |

## Evaluation and result generation

The following command interfaces were checked statically against existing
files and arguments:

```bash
PYTHONPATH=. python training/train_dense.py
PYTHONPATH=. python train_blocktt.py
python scripts/evaluate.py --model dense --checkpoint PATH --dataset PATH --output PATH
python scripts/evaluate.py --model blocktt --checkpoint PATH --dataset PATH --output PATH
python scripts/model_statistics.py --output PATH
python scripts/generate_results.py --reports PATH... --output-dir PATH
python scripts/validate_split.py --dataset PATH --output PATH
python -m pytest tests
```

Static command-target check: **PASS**.

Runtime evaluation and result generation were subsequently verified with the
existing environment. The controlled Dense and BlockTT reproduction outputs
are preserved locally under `results/reproduction_run/` and are excluded from
the initial public release. No paper values are hardcoded into the evaluator.

## README/reproducibility status

- Installation commands: **PASS statically**, package installation was not run.
- Dataset setup instructions: **PASS**, with explicit external dataset path
  requirement.
- Dense, BlockTT, and historical baseline commands: **PASS statically**;
  historical commands remain unverified.
- Evaluation, model-statistics, split-validation, and result-generation
  commands: **PASS for controlled runtime checks**.
- Standalone single-image inference: **NOT IMPLEMENTED**, documented as such.

## Remaining issues before publication

1. Install pytest if the complete test suite is required.
2. Verify all historical baseline variants independently.
3. Resolve the non-GNN Deep-Interleaved checkpoint provenance.
4. Establish a reproducible split manifest for future releases.

## Overall result

**STATUS: PARTIALLY VERIFIED.**

The release passes static validation and the final dense/BlockTT runtime,
checkpoint, metric, split, and controlled evaluation checks. Full pytest
execution and complete historical-baseline verification remain outstanding.
