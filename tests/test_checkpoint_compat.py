import torch

from training.checkpoint_compat import extract_model_state


def test_supported_checkpoint_schemas():
    state = {"weight": torch.ones(1)}
    for checkpoint, schema in (
        ({"model": state}, "model"),
        ({"model_state_dict": state}, "model_state_dict"),
        (state, "raw_state_dict"),
    ):
        extracted, detected = extract_model_state(checkpoint)
        assert detected == schema
        assert extracted is state
