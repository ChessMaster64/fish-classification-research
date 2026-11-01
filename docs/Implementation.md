Research Project Report
Project Title (Working)

Tensor Decomposition Based Compression of a Deep Interleaved ResNet50–Vision Transformer–Graph Neural Network for Fish Species Classification

1. Research Goal

The objective of this research is not to design a new hybrid architecture.

The hybrid architecture has already been developed and experimentally validated.

The objective is to:

Compress the best-performing hybrid model while preserving classification performance.

The optimization objectives are:

Reduce trainable parameters.
Reduce FLOPs.
Reduce inference latency.
Preserve classification accuracy.
2. Dataset

Verified

Dataset:

Fish4Knowledge

Approximately

27,000 images
23 species

Input size

224 × 224

Previous preprocessing already established.

The framework keeps identical preprocessing so experiments remain comparable.

3. Best Baseline

Verified from your results table.

Model	Accuracy
ResNet50 + ViT	98.78%
Deep Interleaved ResNet50 + ViT	98.71%
Deep Interleaved ResNet50 + ViT + 2 GNN	98.93%
Middle Interleaved ResNet50 + ViT	97.56%
Middle Interleaved + 2 GNN	Incomplete

Therefore

Baseline

DeepInterleavedResNet50ViTGNN
4. Verified Architecture

From your code.

Input Image
        │
Stem
(conv1+bn+relu+maxpool)
        │
Layer1
        │
Layer2
        │
Layer3
        │
Flatten Feature Map
        │
Projection
1024 → 384
        │
Position Embedding
        │
Transformer Encoder (3 layers)
        │
Dynamic GNN
        │
Residual
        │
Transformer Encoder (3 layers)
        │
Dynamic GNN
        │
Residual
        │
Transformer Encoder (2 layers)
        │
Projection
384 → 1024
        │
Reshape
        │
Layer4
        │
Adaptive Average Pool
        │
Classifier
2048 → 23

This is now the architecture we will modularize.

5. Transformer Configuration

Verified from code.

Embedding dimension

384

Attention heads

8

Feedforward dimension

768

Activation

GELU

Dropout

0.1

Transformer depth

3
+
3
+
2

=

8 blocks
6. GNN

Verified.

Current GNN

DynamicGNNLayer

Graph construction

k-NN

where

k = 6

Message passing

neighbor - center

Edge MLP

Linear(768 → 384)

↓

GELU

↓

Linear(384 → 384)

Aggregation

Max pooling

Residual

Yes

Normalization

LayerNorm
7. Training Configuration

Verified.

Optimizer

AdamW

Learning rate

2e-4

Weight decay

1e-4

Scheduler

CosineAnnealingLR

Epochs

50

Gradient clipping

1.0

Loss

Weighted Cross Entropy

+

Label Smoothing

0.1

Mixup

Probability

50%

Alpha

0.3
8. Parameter Count

Verified

34,777,047
9. Current Framework

Completely rewritten.

Verified.

config.py

Central configuration.

Dataset

dataset/

Completed

dataloader.py

Training

training/

seed.py

losses.py

optimizer.py

checkpoint.py

trainer.py

Completed.

Evaluation

evaluation/

accuracy.py

metrics.py

inference.py

confusion.py

report.py

Completed.

Tests

Every module tested independently.

10. Model Refactoring Plan

Instead of

200-line monolithic file

the architecture becomes

backbones/

    resnet50.py

layers/

    projection.py

    transformer_block.py

    transformer_encoder.py

gnn/

    dynamic_gnn.py

models/

    deep_interleaved_resnet50_vit_gnn.py

Each module has one responsibility.

11. Compression Strategy

This is the most important point.

Earlier we discussed replacing only

proj_in

proj_out

That is not the plan anymore.

After reviewing the architecture, the plan is:

Replace every dense Linear layer in the architecture.

This includes

Projection
proj_in

proj_out
Classifier
head
Every Transformer block

Q

K

V

Output Projection

Feedforward Layer 1

Feedforward Layer 2

Across

8 transformer blocks
GNN

Edge MLP

Linear 1

Linear 2

for

GNN1

GNN2

Everything above becomes configurable.

Example

Dense

↓

BlockTT

↓

Tensor Ring

↓

Tucker

The architecture itself does not change.

12. What is NOT Compressed

Initially

Conv2d

BatchNorm

LayerNorm

Pooling

Residual

Softmax

remain unchanged.

Reason

This isolates the effect of tensor decomposition on the dense layers, where the majority of transformer and GNN parameters reside.

13. Experimental Pipeline

Dense baseline

↓

Train

↓

Evaluate

↓

Record

Parameters

Accuracy

Latency

FLOPs

↓

Replace Dense with BlockTT

↓

Train

↓

Evaluate

↓

Repeat

↓

Tensor Ring

↓

Repeat

↓

Tucker

14. Final Comparison Table

Target table

Model	Params	FLOPs	Latency	Accuracy	Precision	Recall	F1	MCC
Dense								
BlockTT								
Tensor Ring								
Tucker								
15. Future Extension (Not Part of Current Work)

A future extension could compress the convolutional layers (e.g., TTConv, TuckerConv). That would be a separate experimental question because it changes another major component of the model. For the current study, keeping the convolutional backbone unchanged isolates the impact of compressing the dense linear transformations.



Research Roadmap
Working Title

Tensorized Compression of a Deep Interleaved ResNet50–Vision Transformer–Graph Neural Network for Efficient Fish Species Classification

Phase 1 — Baseline Reproduction (Current Priority)
Goal

Reproduce your validated dense model exactly.

No modifications.

Baseline
Input
   │
ResNet50 Stem
   │
Layer1
   │
Layer2
   │
Layer3
   │
Projection
   │
ViT (3)
   │
GNN
   │
ViT (3)
   │
GNN
   │
ViT (2)
   │
Projection
   │
Layer4
   │
Pooling
   │
Classifier
Success Criteria
Parameter count = 34,777,047
Training reproduces previous behavior
Validation accuracy close to 98.93%

Until this is achieved, nothing else changes.

Phase 2 — Complete Modularization

Convert the monolithic script into reusable modules.

fish-classification-research/
│
├── config.py
│
├── dataset/
│
├── training/
│
├── evaluation/
│
├── backbones/
│     └── resnet50.py
│
├── layers/
│     ├── projection.py
│     ├── transformer_block.py
│     ├── transformer_encoder.py
│     └── classifier.py
│
├── gnn/
│     └── dynamic_gnn.py
│
├── models/
│     └── deep_interleaved.py
│
├── compression/
│
├── tensorization/
│
└── experiments/

Each file should have one responsibility.

Phase 3 — Tensorization Framework

Instead of writing:

nn.Linear(...)

everything should use a common interface.

Example:

TensorLinear(...)

Initially

TensorLinear = DenseLinear

Later

TensorLinear = BlockTTLinear

or

TensorLinear = TensorRingLinear

or

TensorLinear = TuckerLinear

The model code never changes—only the implementation behind the interface.

The same idea applies to convolutions:

TensorConv2d(...)

Initially it wraps a normal Conv2d, then later it can wrap BlockTT/Tensor Ring/Tucker implementations.

Phase 4 — Layer Tensorization
Stage A — Linear Layers

Replace every nn.Linear.

This includes:

Projection
proj_in
proj_out
classifier
Transformer

Every block:

Q projection
K projection
V projection
Output projection
Feedforward 1
Feedforward 2

Across 8 transformer blocks.

GNN

Both GNNs:

Edge Linear 1
Edge Linear 2

This stage alone compresses a substantial portion of the model's learnable parameters.

Stage B — Convolution Layers

Tensorize the ResNet backbone.

This includes:

Stem convolution
Every bottleneck convolution in Layer1–Layer4

The architecture remains identical; only the implementation of the convolution changes.

Phase 5 — Tensor Decomposition Methods

Implement a common interface:

BaseTensorLayer

Derived classes:

Dense
BlockTT
Tensor Ring
Tucker

Both linear and convolution layers should use this abstraction where possible.

Phase 6 — Hyperparameter Search

For each decomposition, vary tensor ranks to identify the best trade-off.

Metrics to collect:

Trainable parameters
Model size
FLOPs
Latency
Peak GPU memory
Accuracy
Precision
Recall
F1-score
MCC

Rather than one compressed model, you'll obtain a compression–accuracy frontier.

Phase 7 — Comparative Evaluation

The final comparison table should look like:

Model	Params	Size (MB)	FLOPs	Latency	Accuracy	Precision	Recall	F1	MCC
Dense									
BlockTT (Linear)									
BlockTT (Full)									
Tensor Ring (Linear)									
Tensor Ring (Full)									
Tucker (Linear)									
Tucker (Full)									

This clearly separates the effect of tensorizing only dense layers from tensorizing the entire architecture.

Phase 8 — Ablation Studies

Potential ablations include:

Effect of tensorizing only transformer layers.
Effect of tensorizing only GNN layers.
Effect of tensorizing only convolution layers.
Effect of tensorizing the complete architecture.
Sensitivity to tensor rank.
Compression ratio vs. accuracy.
FLOPs vs. latency (these are related but not identical).
Phase 9 — Paper Contributions

If the experiments are successful, the contributions could be framed as:

A modular Deep Interleaved ResNet50–ViT–2GNN framework for fish species classification.
A unified tensorization framework supporting multiple tensor decomposition methods across both linear and convolutional layers.
A comprehensive comparison of BlockTT, Tensor Ring, and Tucker decompositions under identical training conditions.
An analysis of the trade-off between compression, computational cost, and predictive performance.
Development Order

To minimize engineering risk while staying aligned with the final objective:

Rebuild the dense baseline exactly (parameter count and accuracy).
Modularize the architecture without changing behavior.
Implement tensorized linear layers and verify correctness.
Integrate them into the model and run the first compression experiments.
Implement tensorized convolution layers.
Run the full-architecture compression experiments.
Repeat the evaluation for BlockTT, Tensor Ring, and Tucker.

This order keeps the implementation manageable while still reaching your end goal of compressing the entire architecture. It also ensures that if a regression appears, you can isolate whether it came from the modularization, the linear tensorization, or the convolution tensorization rather than having to debug all three at once.