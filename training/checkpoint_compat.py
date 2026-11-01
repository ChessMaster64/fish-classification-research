"""Compatibility loading for historical and current training checkpoints."""

from pathlib import Path

import torch


def extract_model_state(checkpoint):
    """Extract a model state dictionary from supported checkpoint schemas."""
    if not isinstance(checkpoint, dict):
        return checkpoint, "raw_state_dict"

    for key in ("model", "model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state, key

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint, "raw_state_dict"

    raise ValueError(
        "Unsupported checkpoint schema: expected model, model_state_dict, "
        "state_dict, or a raw state dictionary."
    )


def load_model_checkpoint(model, path, device="cpu", strict=True):
    """Load a model checkpoint and return metadata about its schema."""
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, schema = extract_model_state(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    metadata = {
        "path": str(checkpoint_path),
        "schema": schema,
        "strict": strict,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    if isinstance(checkpoint, dict):
        for key in ("epoch", "best_acc", "best_accuracy"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    return metadata
