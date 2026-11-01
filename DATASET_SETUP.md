# Dataset setup

## Source and rights

The working tree includes a local Fish4Knowledge copy, but it is excluded from
Git. This audit did not establish the dataset's redistribution license. Obtain
the dataset from its authoritative distribution channel and review the terms
before sharing it or derived checkpoints.

## Directory layout

Configure the repository as:

```text
fish_project/
  fish4knowledge/
    01_Dascyllus_reticulatus/
    02_plectroglyphidodon_dickii/
    ...
    23_siganus_fuscescen/
  fish-classification-research/
```

When using this release, place `fish4knowledge/` beside the release directory
or pass its absolute path through the evaluation and split-validation scripts.
The loader's default is relative to the repository root used by the package,
so explicit `--dataset` arguments are recommended.

## Split behavior

`dataset/dataloader.py` uses torchvision `ImageFolder`, groups samples
by the second underscore-separated filename component, keeps every third frame,
and splits video IDs with `random_state=42`. Defaults are 80% train, 10%
validation, and 10% test. The local copy contains 27,370 images and the
stride-3 sample count is 13,230.

The filename convention is an assumption encoded by `get_video_id`; validate
it against the downloaded dataset before relying on the split.

## Corrupt or missing files

No explicit corrupt-image quarantine or manifest generation is implemented.
`ImageFolder` will discover files at runtime, and image decode failures will
surface during loading. A public release should add a deterministic manifest
and an image-integrity check before training.
