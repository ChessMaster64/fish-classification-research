# Final research code inventory

## Audit basis

The audit covered the repository worktree, including the current release
tree, legacy `codes/` copies, `older start/`, `archive/`, documentation,
tests, checkpoints, the bundled dataset, and the separate `torch-mpo/` tree.
Generated Python caches were excluded from source classifications.

The current candidate contains 151 non-cache files, including 139 Python
files. The dataset contains 23 class directories and 27,370 image files.

## Classification

| Category | Current evidence |
|---|---|
| A. Essential final implementation | `config.py`, `dataset/dataloader.py`, `models/`, `gnn/`, `layers/`, `compression/`, `training/`, `evaluation/`, `train_blocktt.py`, and `training/train_dense.py` |
| B. Supporting modules | `backbones/`, `tensorization/`, `decompositions/`, `utils/`, and package initializers |
| C. Reproducibility assets | `results/`, `experiments/log.md`, and checkpoints under `checkpoints/` (large and not suitable for automatic publication) |
| D. Baselines/comparisons | `older start/` model variants and the historical root scripts in `codes/`; exact inclusion in the reported table is not fully proven |
| E. Experimental/obsolete candidates | `archive/v1_baseline` through `archive/v8_tr_transformer`, most `older start/` variants, and the deleted root implementations |
| F. Manual verification required | dataset license, author/title metadata, exact checkpoint provenance, exact reported parameter counts/MCC, and whether every old baseline was used in the paper |

The classification is based on imports and execution paths, not filenames alone.
No source or checkpoint was deleted. Exact historical baseline provenance
copies are now under `baselines/older_start/`.

The recovered baseline files are:
`resnet50_vit.py`, `evalresnet50_vit.py`,
`resnet50_vit_middleinterleaved.py`,
`evalresnet50_vit_middleinterleaved.py`, and the
`resnet50_vit_deepinterleaved.py` source/evaluator pair. The last pair needs
verification because the available evaluator/checkpoint is the 2-GNN variant.

## Final candidate execution path

- Dense training: `training/train_dense.py`
  -> `dataset.dataloader.create_dataloaders`
  -> `models.hybrid_model.DeepInterleavedResNet50ViTGNN`
  -> `training.trainer`.
- BlockTT training: `train_blocktt.py`
  -> the same loader/model
  -> `compression.replace.replace_linear_layers`
  -> `training.trainer`.
- Evaluation building blocks: `evaluation/inference.py`,
  `metrics.py`, `report.py`, `confusion.py`, and `parameters.py`.
- `scripts/evaluate.py` connects checkpoint loading, test prediction, all
  requested classification metrics, and JSON/CSV prediction output.
- `scripts/model_statistics.py` computes dense/BlockTT parameter summaries.
- `scripts/generate_results.py` generates comparison CSV and MCC/weighted-F1
  figures from evaluation JSON records.

## Important discrepancies

1. `dataset/dataloader.py` defaults to 80/10/10 video splits and
   stride 3. The 27,370 local images therefore produce 13,230 sampled images,
   consistent with the supplied 10,557/1,360/1,313 totals, but this exact split
   was not executed in the dependency-free environment.
2. `config.py` has duplicated model and BlockTT sections.
3. The config declares label smoothing, but both training entry points use
   plain `CrossEntropyLoss()` without smoothing or class weights.
4. Existing notes report 34,777,047 parameters, while the supplied final
   context reports 39.50M and 26.51M. No single verified counting workflow
   resolves this conflict.
5. `evaluation/metrics.py` now computes weighted precision, recall, and F1,
   plus macro one-vs-rest specificity; runtime verification remains pending.
6. `config.py` points `DATASET_PATH` at `fish4knowledge`,
   while the loader default resolves the repository-root dataset.
7. `train_blocktt.py` imports as a script from the root
   directory rather than as an installed package, and its checkpoint paths
   depend on the current working directory.

## Baselines

The historical notes identify `DeepInterleavedResNet50ViTGNN` as the best
baseline and list ResNet50 + ViT and several interleaving variants. Their
checkpoint provenance and exact comparison protocol remain manual-verification
items; they should not be presented as reproducible baselines without that
evidence.
