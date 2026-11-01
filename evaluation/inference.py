import torch


def predict(
    model,
    loader,
    device,
):
    """
    Run inference on a dataloader.

    Returns
    -------
    tuple[list, list]
        (y_true, y_pred)
    """

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            y_true.extend(
                labels.cpu().tolist()
            )

            y_pred.extend(
                predictions.cpu().tolist()
            )

    return y_true, y_pred