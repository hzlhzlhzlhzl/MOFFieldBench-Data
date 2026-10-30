# External checkpoints

`manifest.json` indexes the 39 selected checkpoints and their expected SHA-256 hashes. Private source paths have been replaced by `<task>/<modalities>/best_model.pt`. Weights are distributed separately in `moffieldbench-checkpoints.tar.gz`, under `multimodal/checkpoints/artifacts/<task>/<modalities>/best_model.pt`. See the [exact archive member inventory](../../docs/checkpoint_archive.json). Extracting the archive into the repository root installs all 39 weights directly; the materialization script is unnecessary for this route.

Alternatively, when importing a separate collection of original weights, arrange them under a local directory using the `source_checkpoint` layout, then run from the module directory:

```bash
python scripts/materialize_checkpoints.py --source-root /path/to/checkpoints
```

The command validates hashes and creates `checkpoints/artifacts/`, which is ignored by Git. No public download location is provided in this release.
