import torch

from tqdm import tqdm

from evaluation.accuracy import accuracy


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    grad_clip=None,
    desc="Epoch",
):
    """
    Internal epoch runner.

    If optimizer is None:
        Evaluation mode.

    Otherwise:
        Training mode.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    amp_enabled = (
        scaler is not None
        and device.type == "cuda"
    )

    progress = tqdm(
        loader,
        desc=desc,
        leave=False,
    )

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for images, labels in progress:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            if amp_enabled:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels,
                    )

            else:

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            if training:

                if amp_enabled:

                    scaler.scale(loss).backward()

                    if grad_clip is not None:

                        scaler.unscale_(optimizer)

                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            grad_clip,
                        )

                    scaler.step(optimizer)

                    scaler.update()

                else:

                    loss.backward()

                    if grad_clip is not None:

                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            grad_clip,
                        )

                    optimizer.step()

            batch_size = labels.size(0)

            batch_acc = accuracy(
                outputs.detach(),
                labels,
            )

            running_loss += loss.item() * batch_size
            running_acc += batch_acc * batch_size

            total_samples += batch_size

            progress.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                acc=f"{running_acc / total_samples:.2f}",
            )

    return {
        "loss": running_loss / total_samples,
        "accuracy": running_acc / total_samples,
        "samples": total_samples,
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    grad_clip=None,
):
    return _run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        grad_clip=grad_clip,
        desc="Train",
    )


def validate(
    model,
    loader,
    criterion,
    device,
):
    return _run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        grad_clip=None,
        desc="Validation",
    )