# Publication readiness

Audit scope: release-tree contents, publication exclusions, documentation,
dependencies, citation metadata, and generated reproduction artifacts.

## Status

**BLOCKED pending author decisions and final release preparation.**

The source tree passes syntax validation and documentation-link validation.
Datasets, checkpoints, archives, caches, virtual environments, and local
reproduction outputs are excluded by `.gitignore`. No Git repository was
initialized and no files were staged.

## PASS

- Successful final Dense and BlockTT implementations are present.
- Historical ResNet50+ViT and Middle-Interleaved baseline sources are present.
- The unresolved Deep-Interleaved provenance is explicitly documented.
- No dataset, checkpoint, archive, credential, API key, or private key is in
  the release tree.
- `results/reproduction_run/` is preserved locally and excluded from the
  initial public release.
- Python syntax compilation passed.
- All Markdown relative links resolve.
- Direct runtime dependencies identified by source inspection are declared in
  `requirements.txt`, including `networkx` and `ptflops`.
- No software license has been selected or implied.

## Remaining blockers

- No `LICENSE` file exists and no software license has been selected.
- `CITATION.cff` contains the verified working title but omits author and
  publication metadata because those details were not established in the
  audited project.
- The intended public release repository URL has not been confirmed.
- Historical baseline execution remains environment- and
  provenance-dependent.
- Some legacy tests use the historical `research` package import layout and
  are not guaranteed to run from this release root.
- The Fish4Knowledge redistribution terms remain unverified.

## Local reproduction artifacts

The controlled Dense and BlockTT evaluation reports remain under
`results/reproduction_run/` for local reference only. They are excluded from
the initial public release and contain no modified model, dataset, or
checkpoint files.
