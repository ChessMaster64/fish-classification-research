# Fish Species Classification with ResNet50, ViT, Dynamic GNN, and BlockTT

Research release repository for underwater fish-species classification using a
hybrid ResNet50, Vision Transformer, and dynamic graph neural network, with an
experimental Block Tensor-Train (BlockTT) compression path.

## Abstract

The candidate model extracts spatial features with ResNet50, processes 196
tokens through three Transformer stages and two dynamic k-nearest-neighbour
graph stages, then reconstructs spatial features for the final ResNet50 stage.
BlockTT replaces eligible linear transformations while keeping convolutional
layers and the 23-class head dense.

This release preserves the audited implementation without renaming importable
modules. Reported paper results are reproduced below as claims; no independent
training run is claimed for this release.

## Research objectives and contributions

- Classify 23 Fish4Knowledge species under video-level data partitioning.
- Combine spatial, token, and feature-relationship modelling.
- Evaluate BlockTT compression of eligible dense transformations.
- Preserve the original preprocessing and checkpoint-compatible model names.

## Architecture

```text
Image
  -> ResNet50 stem + layers 1-3
  -> 14 x 14 x 1024 feature map -> 196 tokens
  -> projection 1024 -> 384 + positional embedding
  -> Transformer (3) -> Dynamic GNN (k=6)
  -> Transformer (3) -> Dynamic GNN (k=6)
  -> Transformer (2)
  -> projection 384 -> 1024 -> reshape
  -> ResNet50 layer 4 -> global average pool -> 23 classes
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the implementation mapping and
BlockTT details.

## Dataset and preparation

The loader expects a torchvision `ImageFolder` layout:

```text
fish4knowledge/
  01_Dascyllus_reticulatus/
    image files...
  ...
  23_siganus_fuscescen/
```

The bundled local copy is excluded from Git by `.gitignore`. Dataset terms and
redistribution rights were not established by this audit; obtain Fish4Knowledge
from its authoritative source and review its terms before publishing data.
See [DATASET_SETUP.md](./DATASET_SETUP.md). The dataset is intentionally not
included. Review Fish4Knowledge redistribution terms before sharing data or
derived checkpoints.

## Installation and requirements

Use Python 3.10+ and install the dependencies listed in
[requirements.txt](./requirements.txt). Exact versions were not recoverable
from the candidate project, so the file uses minimum versions rather than
claiming a tested lockfile.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Training commands

Run from the repository root so the preserved absolute imports resolve:

```bash
PYTHONPATH=. python training/train_dense.py
PYTHONPATH=. python train_blocktt.py
```

The dense and BlockTT scripts create or resume checkpoints in a
working-directory `checkpoints/` folder. These commands have not been
executed in this release environment.

Historical baseline training sources are preserved unchanged:

```bash
PYTHONPATH=baselines/older_start python baselines/older_start/resnet50_vit.py
PYTHONPATH=baselines/older_start python baselines/older_start/resnet50_vit_middleinterleaved.py
PYTHONPATH=baselines/older_start python baselines/older_start/resnet50_vit_deepinterleaved.py
```

These legacy scripts require CUDA and retain their original dataset and
training assumptions. They are provenance commands, not runtime-verified
reproduction commands.

## Validation, testing, evaluation, and inference

Validation runs inside each training epoch through
`training.trainer.validate`. Final test evaluation is available through
`scripts/evaluate.py` after a checkpoint exists:

```bash
python scripts/evaluate.py --model dense \
  --checkpoint checkpoints/dense_best.pth \
  --dataset /path/to/fish4knowledge \
  --output results/dense_evaluation.json

python scripts/evaluate.py --model blocktt \
  --checkpoint checkpoints/blocktt_best.pth \
  --dataset /path/to/fish4knowledge \
  --output results/blocktt_evaluation.json
```

The evaluator writes JSON metrics, a classification report, confusion matrix,
and prediction CSV. Metrics include weighted precision, weighted recall,
weighted F1, macro one-vs-rest specificity, and MCC. A standalone
single-image inference command is not implemented.

Model parameter and compression statistics:

```bash
python scripts/model_statistics.py --output results/model_statistics.json
```

Generate comparison tables and figures only from actual evaluation JSON files:

```bash
python scripts/generate_results.py \
  --reports results/dense_evaluation.json results/blocktt_evaluation.json \
  --output-dir results/comparison
```

Validate the video-level split without training:

```bash
python scripts/validate_split.py \
  --dataset /path/to/fish4knowledge \
  --output results/split.json
```

The test suite can be run after installing dependencies:

```bash
python -m pytest tests
```

Do not start training or full evaluation without confirming the dataset,
hardware, and intended runtime.

## Reported results

The supplied research context reports:

| Model | Parameters | Accuracy | Weighted F1 | MCC |
|---|---:|---:|---:|---:|
| Uncompressed | 39.50M | 98.93% | 98.99% | 0.9861 |
| BlockTT | 26.51M | 98.25% | 98.23% | 0.9771 |

These values conflict with other repository notes and are therefore labelled
reported, not independently verified. The checked-in classification reports
contain rounded values and do not contain MCC or parameter counts.

## Repository structure

```text
backbones/       ResNet feature extraction
compression/     BlockTT conversion and compression helpers
dataset/         ImageFolder loading, video grouping, transforms
decompositions/  Tensor-train decomposition primitives
evaluation/      Metrics, reports, inference helpers, plots
gnn/             Dynamic graph construction and layers
layers/          Dense and tensorized neural-network layers
models/          Hybrid model and Transformer components
tensorization/   Dimension factorization and tensorization
training/        Training, optimization, checkpoint, and seed helpers
tests/           Unit and component tests
baselines/       Provenance copies of successful historical baselines
scripts/         Evaluation, split, statistics, and result-generation tools
docs/            Audited implementation and testing notes
```

The original module names and import layout are deliberately preserved for
checkpoint and execution compatibility. See
[FINAL_RESEARCH_CODE_INVENTORY.md](./FINAL_RESEARCH_CODE_INVENTORY.md).

The historical ResNet50+ViT and Middle-Interleaved sources are included under
`baselines/older_start/`. The reported non-GNN Deep-Interleaved result remains
`NEEDS VERIFICATION` because the available checkpoint/evaluator evidence is
for the 2-GNN variant.

## Hardware, limitations, and status

The implementation requires a PyTorch-compatible CPU or CUDA environment and
ImageNet ResNet50 weights. Exact hardware and package versions used for the
paper were not recoverable. Current limitations include hard-coded training
configuration, no standalone single-image inference CLI, unresolved split and
parameter-count differences, and unverified checkpoint/dataset provenance.

## License and citation

No project license is selected because dataset and third-party-code rights
were not verified. Do not publish the dataset, checkpoints, or vendored
third-party code until their terms are reviewed. Author and publication
metadata are pending in [CITATION.cff](./CITATION.cff).
