# Contributing

This repository is prepared for research reproducibility. Before changing
model code, document the intended behavior change and preserve the original
configuration and checkpoint compatibility unless the change is explicitly
experimental.

## Checks

Run the available unit tests after installing the dependencies:

```bash
python -m pytest tests
```

Do not add datasets, credentials, or large checkpoints to Git. Record dataset
source and license information in `DATASET_SETUP.md`, and label newly generated
results as reproduced only when the complete experiment was actually run.
