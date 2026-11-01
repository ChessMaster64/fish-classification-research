# Final research pipeline

## 1. Dataset preparation

Place Fish4Knowledge outside version control using the ImageFolder layout
described in [DATASET_SETUP.md](./DATASET_SETUP.md). The loader derives a video
ID from the second underscore-delimited filename component, samples every
third frame, and splits video IDs with a fixed seed of 42.

## 2. Preprocessing

Training applies CLAHE, underwater colour/haze augmentation, random resized
crop to 224, horizontal flip, rotation, colour jitter, Gaussian blur,
conversion to tensor, and normalization with mean
`[0.4729, 0.6096, 0.5808]` and standard deviation
`[0.2020, 0.2574, 0.2559]`.

Validation and test apply resize to 224 x 224, CLAHE, tensor conversion, and
the same normalization. The implementation is in
[`dataset/dataloader.py`](./dataset/dataloader.py).

## 3. Model construction

```python
from models.hybrid_model import DeepInterleavedResNet50ViTGNN
model = DeepInterleavedResNet50ViTGNN()
```

The dense model uses ImageNet ResNet50 weights, so the first construction may
need network access or a locally cached torchvision weight file.

For BlockTT:

```python
from compression.replace import replace_linear_layers
from config import BlockTTConfig
model = replace_linear_layers(model, BlockTTConfig(max_rank=16))
```

This compresses eligible linear layers with 2 x 2 block partitioning and keeps
the classifier named `head` dense. The convolutional ResNet layers remain
dense.

## 4. Training

Current commands:

```bash
PYTHONPATH=. python training/train_dense.py
PYTHONPATH=. python train_blocktt.py
```

The first command is the dense training entry point. The second is the
BlockTT entry point. Both use AdamW, learning rate `3e-4`, weight decay
`1e-4`, cosine annealing over 10 epochs, batch size 8, gradient clipping 1.0,
and seed/configuration values only partially wired from `config.py`.

These commands have not been executed in this audit. They are not yet
portable CLI tools: paths, hyperparameters, and checkpoint names are hard-coded
or current-working-directory dependent.

Historical baseline commands are preserved source copies and are not runtime
verified in this environment:

```bash
PYTHONPATH=baselines/older_start python baselines/older_start/resnet50_vit.py
PYTHONPATH=baselines/older_start python baselines/older_start/resnet50_vit_middleinterleaved.py
PYTHONPATH=baselines/older_start python baselines/older_start/resnet50_vit_deepinterleaved.py
```

These scripts require a CUDA device and the external dataset at
`fish4knowledge/`; they retain their original training behavior.

## 5. Validation and testing

Validation occurs once per training epoch through
`training.trainer.validate()`. Checkpoint selection uses validation top-1
accuracy. The test loader is created but neither training entry point performs
final test evaluation.

`scripts/evaluate.py` loads either checkpoint schema, runs the test loader once,
and emits accuracy, weighted precision/recall/F1, specificity, MCC, a
classification report, confusion matrix, parameter count, and (for BlockTT)
compression ratio and reduction percentage.

## 6. Inference

`evaluation/inference.py` provides a dataloader-level `predict` function.
Checkpoint schema compatibility is provided by
`training/checkpoint_compat.py`. A standalone single-image command is not
implemented.

## Exact reproducibility commands

The evaluation and result-generation commands in the README are reproducible
workflows once dependencies, dataset, and checkpoints are available. They do
not claim successful reproduction until actually executed on the final split.
