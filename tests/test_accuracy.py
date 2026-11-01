import torch

from evaluation.accuracy import accuracy
from evaluation.accuracy import topk_accuracy


torch.manual_seed(42)

logits = torch.randn(16, 23)

targets = torch.randint(
    0,
    23,
    (16,),
)

acc1 = accuracy(
    logits,
    targets,
)

acc1_top5 = topk_accuracy(
    logits,
    targets,
    topk=(1, 5),
)

print("Top-1:", acc1)

print("Top-1 / Top-5:", acc1_top5)