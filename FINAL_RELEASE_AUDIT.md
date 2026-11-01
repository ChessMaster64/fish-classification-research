# Final release completeness audit

Date: 2026-11-01

## Scope and evidence

This audit compares the new release repository with:

- the supplied paper research context;
- the previous `FINAL_RESEARCH_CODE_INVENTORY.md`;
- the original worktree's `research/`, `docs/`, `results/`, `older start/`,
  `archive/`, and legacy `codes/` trees.

No training, full-dataset evaluation, checkpoint loading, or expensive
benchmark was run. No model logic, training logic, preprocessing, metrics, or
reported results were modified.

## Executive conclusion

**The release is not yet complete or publication-ready.**

The final dense hybrid and BlockTT model code is present, and the source tree
is syntactically complete. The release now includes provenance copies of the
historical baselines that could be established, separate checkpoint
compatibility and evaluation tools, split validation, and result generation.
It still cannot claim full paper reproduction: the non-GNN Deep-Interleaved
result remains unverified, runtime dependencies were not installed here, and
some historical figures/records are unavailable.

## 1. Architecture and baseline coverage

### Complete

| Research component | Release status | Evidence |
|---|---|---|
| Final dense Deep Interleaved ResNet50 + ViT + 2 GNN | Present | `models/hybrid_model.py`, `models/dynamic_gnn.py`, Transformer modules |
| ResNet50 stages and spatial reconstruction | Present | `models/hybrid_model.py`, `models/resnet50_encoder.py`, `backbones/resnet50_backbone.py` |
| Transformer depths 3, 3, 2 | Present/configured | `config.py`, `models/hybrid_model.py` |
| Dynamic GNN with k=6 in final model | Present | `config.py`, `models/hybrid_model.py`, `models/dynamic_gnn.py` |
| Dense training entry point | Present | `training/train_dense.py` |
| BlockTT-compressed final model | Present | `train_blocktt.py`, `compression/replace.py`, `layers/block_tt_linear.py` |
| BlockTT rank 16 and 2 x 2 block configuration | Present | `train_blocktt.py`, `compression/config.py` |
| Dense convolutional backbone and dense final head | Present | `compression/replace.py` |

### Missing or unavailable

The previous inventory and `docs/Implementation.md` record the following
baseline table:

| Reported baseline/ablation | Historical evidence | Release status |
|---|---|---|
| ResNet50 + ViT, reported 98.78% | Historical source/evaluator and `resnet50_vit_best.pth` | **Recovered under `baselines/older_start/`** |
| Deep Interleaved ResNet50 + ViT, reported 98.71% | Historical training source; available evaluator/checkpoint are the GNN variant | **Source recovered; result NEEDS VERIFICATION** |
| Deep Interleaved ResNet50 + ViT + 2 GNN, reported 98.93% | Final candidate and historical implementation | **Present as final model** |
| Middle Interleaved ResNet50 + ViT, reported 97.56% | Historical source/evaluator and `resnet50_vit_middleinterleaved_best.pth` | **Recovered under `baselines/older_start/`** |
| Middle Interleaved + 2 GNN | `older start/resnet50_vit_middleinterleaved_gnn.py` and checkpoint | **Documented as incomplete in notes; not included** |

Additional historical implementations exist for ViT-only, ViT-GNN one/two/three
layer variants, MPO variants, Tensor Ring variants, and alternative ResNet/ViT
interleavings. Their presence and checkpoints prove experiments occurred, but
the available notes do not prove that each belongs to the submitted paper's
comparison table. They are therefore not classified as required paper
baselines without author confirmation. The release must either:

1. include the three verified successful baselines above; or
2. explicitly state in its README and inventory that the paper's baseline
   implementations are unavailable and provide their historical paths and
   provenance.

## 2. Dataset, preprocessing, and splitting

### Complete in source

- `dataset/dataloader.py` contains ImageFolder loading.
- Class count is configured as 23.
- Video ID extraction and video-level splitting are implemented.
- Frame stride 3 is implemented.
- Seed 42 and 80/10/10 loader defaults are implemented.
- CLAHE preprocessing is implemented.
- Underwater colour and haze augmentation is implemented.
- Training crop, flip, rotation, colour jitter, blur, normalization, and test
  transforms are implemented.
- WeightedRandomSampler logic is included.

### Incomplete for reproducibility

- `scripts/validate_split.py` reports split counts and video leakage; no
  manifest is checked in.
- No corrupted-image validation/quarantine command exists.
- The paper's stated 70/15/15 values conflict with the loader's 80/10/10
  defaults and the duplicated `config.py` split constants.
- The release does not provide a dataset-preparation command; it provides only
  manual placement instructions.
- The exact split counts are not emitted or persisted as a reproducibility
  artifact.
- `config.DATASET_PATH` points to a path under the original `research`
  layout, while the loader default resolves from the new repository root.

## 3. Training, validation, and testing

### Complete

- Dense training loop is present.
- BlockTT training loop is present.
- AdamW, cosine scheduler, gradient clipping, AMP handling, and checkpoint
  resume logic are represented.
- Per-epoch validation is present.
- Unit/component tests are included.

### Missing or inconsistent

- The release's `training/train_dense.py` is present, but the previous
  inventory identified the effective scripts as different working-directory
  entry points; no release-level launcher or verified invocation has been run.
- The test loader is created by training scripts but final test evaluation is
  not performed by either training entry point; `scripts/evaluate.py` now
  provides the separate final test workflow.
- `training/checkpoint.py` expects keys such as `model_state_dict`, while the
  two training scripts write/read a different checkpoint schema using
  `model`, `optimizer`, and `scheduler`. `training/checkpoint_compat.py`
  provides the separate adapter used by the release evaluator; the older
  helper remains schema-specific.
- `SEED` is declared in `config.py`, but the training entry points do not call
  the seed utility before training.
- `LABEL_SMOOTHING` is declared but plain `CrossEntropyLoss()` is used.
- The notes mention weighted cross-entropy and mixup, but the final training
  scripts do not implement those claims.
- Historical baseline training scripts are included under
  `baselines/older_start/`; their checkpoint selection remains historical.
- No automatic paper checkpoint-selection policy exists; the evaluator
  requires an explicit checkpoint path.

## 4. Dense and BlockTT completeness

### Dense model

The final dense model class and its imported components are included. The
architecture is structurally complete for construction, subject to installing
dependencies and obtaining ImageNet ResNet50 weights.

### BlockTT model

The BlockTT conversion path is included:

```text
train_blocktt.py
  -> replace_linear_layers()
  -> compress_linear_to_blocktt()
  -> build_tt_block()
  -> BlockTTLinear
```

The classifier named `head` is skipped and convolutional layers remain dense.
The release does not include the large trained BlockTT checkpoint, which is
appropriate for a source release but prevents checkpoint-level reproduction.

### Reproducibility gap

There is no release script that constructs both variants, counts parameters,
checks the dense/BlockTT layer inventory, and writes a machine-readable
comparison report.

## 5. Parameter counts and compression calculations

### Present

- `training/train_dense.py` has a local trainable-parameter counter.
- `utils/model_stats.py` reports total, trainable, frozen parameters, and
  estimated model size.
- BlockTT layers expose parameters through PyTorch modules.

### Missing

- `scripts/model_statistics.py` computes dense/BlockTT counts, ratio, and
  reduction from constructed models; the reported values remain unverified.
- No script verifies that the head and convolutional layers were excluded.
- Existing notes also report 34,777,047 parameters, conflicting with the
  supplied headline values.

Therefore the reported 32.88% reduction and 1.49x ratio cannot currently be
reproduced from the release by a documented command.

## 6. Reported metric coverage

| Metric | Code status | Finding |
|---|---|---|
| Accuracy | Partial | `evaluation/metrics.py` computes it, but no final test runner calls it |
| Precision | Partial | Macro precision only; paper request is not explicitly weighted |
| Recall | Partial | Macro recall only; no end-to-end test report |
| Weighted F1 | Present | `evaluation.metrics` now uses weighted averaging |
| Specificity | Present | `evaluation.metrics.specificity_score` computes macro one-vs-rest specificity |
| MCC | Present | Helper and `scripts/evaluate.py` emit MCC |
| Confusion matrix | Partial | Utility exists, but no release evaluator calls it |
| Classification report | Partial | Utility exists, but no checkpoint/test orchestration calls it |

The paper's weighted F1 values therefore cannot be generated by the current
release evaluation API without adding metric logic.

## 7. Tables and figures

### Present utilities/artifacts

- Confusion-matrix utility in `evaluation/confusion.py`.
- Classification-report utility in `evaluation/report.py`.
- FLOP, latency, and representation utility modules exist.
- Historical result text and image artifacts existed in the original worktree.

### Missing or partial

- `scripts/generate_results.py` generates comparison CSV, MCC, and weighted-F1
  figures from actual evaluation JSON records.
- Parameter/compression summary is produced by `scripts/model_statistics.py`.
- FLOP/latency paper-table generation remains unavailable.
- Generated result images and reports were intentionally not copied into the
  clean release.

The current release cannot generate the submitted paper's tables or figures
from a fresh checkout.

## 8. Successful research code accidentally excluded

The release copied all non-artifact files from the original `research/` tree.
The only original `research/` file omitted by path comparison was
`research/evaluation.zip`, which is an archive rather than source and should
remain excluded.

Historical baseline source was initially excluded, but is now recovered under
`baselines/older_start/`. The large checkpoints remain excluded:

- `older start/resnet50_vit.py`
- `older start/resnet50_vit_deepinterleaved.py`
- `older start/resnet50_vit_middleinterleaved.py`
- their corresponding evaluation scripts
- associated baseline checkpoints

The non-GNN Deep-Interleaved result remains unresolved because only a GNN
evaluator/checkpoint was found.

## 9. Imports, configuration, and execution paths

### Static status

The copied Python tree passes syntax compilation. Package initializers were
added for importable directories.

### Blocking issues

1. `gnn/dynamic_gnn.py` imports `torch_geometric.nn.SAGEConv`; the package is
   now listed in `requirements.txt`, but installation compatibility is
   unverified.
2. The release uses absolute imports such as `from dataset...` and
   `from models...`. These are fragile when scripts are invoked as modules or
   from outside the repository root.
3. `python training/train_dense.py` and `python train_blocktt.py` have not been
   runtime-tested in the release environment.
4. `config.py` creates `checkpoints`, `results`, and `logs` directories at
   import time, which is an undocumented side effect.
5. The current config has duplicated model and BlockTT sections.
6. `DATASET_PATH` in `config.py` is not used consistently by the loader.
7. Checkpoint paths are relative to the current working directory.
8. The model construction requires torchvision ImageNet weights, but the
   release does not document a tested offline/cache workflow.

## 10. README and command audit

### Correct or substantially correct

- Installation command corresponds to `requirements.txt`.
- `python training/train_dense.py` names an existing script.
- `python train_blocktt.py` names an existing script.
- `python -m pytest tests` names the included test directory.
- README documents the new evaluation CLI and correctly states that standalone
  single-image inference is not implemented.

### Incorrect or incomplete

- It calls the repository a release while the documented workflow cannot
  reproduce the paper metrics.
- It does not yet explain the distinction between the legacy checkpoint
  helper and the new compatibility adapter.
- It does not distinguish the loader's 80/10/10 split from the paper/inventory
  claims clearly enough.

## 11. Dependency audit

### Listed and used

`torch`, `torchvision`, `numpy`, `opencv-python`, `Pillow`, `scikit-learn`,
`tqdm`, `matplotlib`, and `seaborn` are used or required by the copied source.

### Added but unverified

- `torch-geometric` is now listed for `gnn/dynamic_gnn.py`.
- `pytest` is now listed for the documented test command.

### Not pinned or compatibility-tested

- PyTorch and torchvision compatibility is not pinned.
- PyTorch Geometric compatibility with the chosen PyTorch/CUDA build is not
  specified.
- No tested Python version, CUDA version, or CPU-only installation matrix is
  provided.
- No lockfile or environment file exists.

## 12. Exact fixes required before GitHub publication

These are release-preparation fixes, not model changes:

1. Resolve the non-GNN Deep-Interleaved result provenance. The source is
    included, but no matching checkpoint/evaluator was found.
2. Verify the copied historical baselines against the submitted paper and
    decide whether their large checkpoints may be hosted separately.
3. Verify the `torch-geometric` and `pytest` dependency entries with a tested
    compatibility matrix or installation instructions.
4. Run `scripts/model_statistics.py` in a dependency-complete environment and
    compare its counts with the paper.
5. Run the test-evaluation script that loads the actual training checkpoint
    schema and emits accuracy, macro/weighted precision and recall, weighted
    F1, specificity, MCC, confusion matrix, and classification report.
6. Add result-generation scripts for every remaining paper table and figure, or clearly
    document that those artifacts are unavailable.
7. Resolve and document the 70/15/15 versus 80/10/10 split discrepancy.
8. Resolve the 34,777,047 versus 39.50M/26.51M parameter discrepancy using
    one verified counting definition.
9. Either document or unify the legacy checkpoint helper versus
   training-script key mismatch; the release evaluator already uses the
   compatibility adapter.
10. Wire the declared seed and intended loss configuration, or remove claims
    that are not implemented. This must be done deliberately because it can
    change research behavior.
11. Verify README commands from the release root in a clean environment.
12. Add checkpoint provenance, exact selected epochs, software versions,
    hardware, and dataset acquisition/licensing information.
13. Add a project license only after verifying ownership and third-party/data
    licensing.

## Final status

| Area | Status |
|---|---|
| Final dense model | Complete in source |
| Final BlockTT model | Complete in source, not end-to-end verified |
| Dataset/preprocessing | Present, reproducibility metadata incomplete |
| Historical baselines | Partially recovered; one result needs verification |
| Training/validation | Present but not fully reproducible |
| Final testing | End-to-end runner added; runtime unverified |
| Requested metrics | Implemented; runtime unverified |
| Parameter/compression reproduction | Implemented; runtime and paper values unverified |
| Paper tables/figures | Partial; FLOP/latency outputs missing |
| Dependencies | Expanded; compatibility unverified |
| GitHub publication readiness | **Not ready; runtime verification blocked** |

## Phase 2 implementation record

### Recovered implementations

- `baselines/older_start/resnet50_vit.py` and
  `evalresnet50_vit.py`, with matching historical checkpoint evidence.
- `baselines/older_start/resnet50_vit_middleinterleaved.py` and
  `evalresnet50_vit_middleinterleaved.py`, with matching historical
  checkpoint evidence.
- `baselines/older_start/resnet50_vit_deepinterleaved.py`, with the caveat
  that the available evaluation/checkpoint pair is the 2-GNN variant.

The copied baseline files are unchanged provenance copies. Their original
GPU/path assumptions are intentionally not silently altered.

### Added reproducibility components

- `training/checkpoint_compat.py` supports raw state dictionaries and
  `model`/`model_state_dict`/`state_dict` checkpoint containers.
- `scripts/evaluate.py` performs checkpoint loading, test inference, weighted
  precision/recall/F1, macro one-vs-rest specificity, MCC, classification
  report, confusion matrix, predictions, and parameter reporting.
- `scripts/model_statistics.py` computes dense/BlockTT parameter counts,
  parameter reduction, and compression ratio.
- `scripts/generate_results.py` generates comparison CSV, MCC, and weighted-F1
  figures from evaluation JSON records.
- `scripts/validate_split.py` reports sampled split sizes and video overlap.
- Lightweight synthetic tests cover metric calculations and checkpoint-schema
  extraction.

## Phase 3 runtime verification

See [PHASE3_RUNTIME_VERIFICATION.md](./PHASE3_RUNTIME_VERIFICATION.md).

- Python syntax compilation: **PASS**.
- Dataset inventory: **PASS** for 23 classes, 27,370 source images, 8,725
  video groups, and 13,230 stride-3 samples.
- Existing checkpoint archive inspection: **PASS** without loading or modifying
  files.
- Runtime dependency discovery: **BLOCKED**; no virtual environment was
  available and all required scientific packages were absent.
- Synthetic metric/checkpoint tests: **BLOCKED** because pytest and runtime
  dependencies were absent.
- Model construction, strict checkpoint loading, dataset split execution,
  parameter counting, evaluation, and figure generation: **BLOCKED**.

No large packages were installed, no training or full evaluation was run, and
the original repository remains untouched.

### Tests run

- Python syntax compilation of the release tree: **passed**.
- Runtime unit tests: **not run** because the audit environment lacks PyTorch
  and scikit-learn.
- Model construction, checkpoint loading, split validation, and result
  generation: **not runtime-verified** for the same dependency limitation.
