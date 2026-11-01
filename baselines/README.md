# Historical baselines

The `older_start/` directory contains source copied unchanged from the
original repository for the baselines documented in the research notes.
Checkpoint files are intentionally not included.

| Architecture | Training source | Evaluation source | Checkpoint evidence | Status |
|---|---|---|---|---|
| ResNet50 + ViT | `resnet50_vit.py` | `evalresnet50_vit.py` | `resnet50_vit_best.pth` | Recovered |
| Deep Interleaved ResNet50 + ViT | `resnet50_vit_deepinterleaved.py` | available evaluator is the GNN variant | no exact non-GNN checkpoint found | Needs verification |
| Deep Interleaved ResNet50 + ViT + 2 GNN | historical evaluator/source | `evalresnet50_vit_deepinterleaved.py` | `resnet50_vit_deepinterleaved_gnn_best.pth` | Recovered historical evidence |
| Middle Interleaved ResNet50 + ViT | `resnet50_vit_middleinterleaved.py` | `evalresnet50_vit_middleinterleaved.py` | `resnet50_vit_middleinterleaved_best.pth` | Recovered |

These scripts use the historical `data_loader.py` in the same directory.
They retain their original GPU assumptions, paths, class names, and training
behavior. They are provenance copies, not replacements for the modular final
model.
