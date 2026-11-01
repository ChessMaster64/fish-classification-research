import torch


@torch.no_grad()
def topk_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk=(1,),
):
    """
    Computes Top-k accuracy.

    Args:
        outputs : (B, C)
        targets : (B,)
        topk    : tuple of k values

    Returns:
        list of accuracies in percentage.
    """

    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(
        maxk,
        dim=1,
        largest=True,
        sorted=True,
    )

    pred = pred.t()

    correct = pred.eq(
        targets.view(1, -1).expand_as(pred)
    )

    results = []

    for k in topk:

        correct_k = (
            correct[:k]
            .reshape(-1)
            .float()
            .sum()
        )

        results.append(
            correct_k.mul_(100.0 / batch_size).item()
        )

    return results


@torch.no_grad()
def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Standard Top-1 accuracy.
    """

    return topk_accuracy(
        outputs,
        targets,
        topk=(1,),
    )[0]