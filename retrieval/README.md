# MOF three-task retrieval package

Checkpoint archive: `moffieldbench-checkpoints.tar.gz`. Extract it into the repository root to restore `retrieval/checkpoints/`. Exact archive member paths and SHA-256 values are listed in [the checkpoint inventory](../docs/checkpoint_archive.json); see [installation instructions](../docs/EXTERNAL_ASSETS.md#checkpoint-archive).

> Public source distribution: raw CIF/CP3D/PLY data, model weights and compiled extensions are not bundled. See [external assets](../docs/EXTERNAL_ASSETS.md). Paths beginning with `external/` are anonymized archival references, not working defaults.

This is a self-contained, reproducible package for the three bidirectional
retrieval tasks used in `figures/best_retrieval_tasks_reference.png`:

| Task | Frozen best configuration | Checkpoint |
| --- | --- | --- |
| CIF ↔ ED | ED threshold 0.02, 6,144 points, Fourier k=2 | `checkpoints/cif_ed_best.pt` |
| CIF ↔ ESP | ESP extreme 3%, 4,096 points, Fourier k=2 | `checkpoints/cif_esp_best.pt` |
| ED ↔ ESP | ED 0.03 + ESP extreme 3%, 4,096 points, xyz + scalar + Fourier k=2 | `checkpoints/ed_esp_best.pt` |

The package contains code, frozen configs, checkpoint metadata, exact split
manifests, reported metrics, and the reference plot. It intentionally excludes
the large CIF/PLY dataset.

## Layout

```text
retrieval/
├── run.py                 # one entry point: train / evaluate / figure
├── configs/               # frozen configs for the three reported models
├── checkpoints/           # best validation checkpoint for every task
├── data/                  # empty, documented data mount point + split manifests
├── src/                   # CGCNN, X3D/OpenPoints subset, training programs
├── results/               # metrics saved with the released checkpoints
└── figures/               # supplied reference figure
```

## Environment

Use Linux, Python 3.9, CUDA and a GPU. PointNet2 source is included; compile it for your Python/PyTorch/CUDA environment. Install the Python packages:

```bash
cd retrieval
python -m pip install -r src/requirements.txt
```

Build the required extension before training or evaluation:

```bash
cd src/cif_ed_retrieval/vendor/openpoints/cpp/pointnet2_batch
python setup.py build_ext --inplace
```

## Place data

Do not place raw data elsewhere in the package. Populate or symlink `data/cifs`
and `data/point_clouds/{ed,esp}` as specified in [data/README.md](data/README.md).
The empty directories are placeholders only. This arrangement covers all ED
thresholds, ESP selection percentages, point counts, and periodicity modes.

## One entry point

All commands are run from the package root. `--task` accepts `cif_ed`,
`cif_esp`, `ed_esp`, or `all`.

Recreate the supplied figure from the packaged metrics (does not need raw data):

```bash
python run.py figure --output-dir runs/figure
```

This writes `runs/figure/best_retrieval_tasks.png`, matching the reported values
in the supplied reference figure.

Evaluate released checkpoints on their exact frozen test splits:

```bash
python run.py evaluate --task all --data-root ./data --output-dir runs/evaluation
```

The command first creates portable manifests at `runs/evaluation/manifests/` and
then evaluates each `checkpoints/*_best.pt`. Its resulting `metrics.json` files
can be compared with `results/*_metrics.json`. GPU selection is controlled by
`CUDA_VISIBLE_DEVICES`, for example `CUDA_VISIBLE_DEVICES=0 python run.py ...`.

For end-to-end reproduction, evaluate the released weights and plot those fresh
test metrics in one command:

```bash
python run.py reproduce --task all --data-root ./data --output-dir runs/reproduce
```

The new plot is written to `runs/reproduce/figure/best_retrieval_tasks.png`.

Retrain one of the frozen configurations (the released checkpoints were trained
for 200 epochs):

```bash
python run.py train --task ed_esp --data-root ./data --output-dir runs/retrain
```

Any supported setting can be changed without creating another script:

```bash
python run.py train --task cif_ed --set data.num_points=8192 training.batch_size=32
```

For a different ED threshold / ESP percentage, use a matching manifest and
override the appropriate data fields; see `data/README.md`. Do not reuse a
checkpoint when the input feature layout changes.

## Reproducibility notes

- `seed: 42`, architecture, optimizer, schedule, 200 epochs, batch size, and
  Fourier positional encoding are frozen in `configs/`.
- The included manifests retain the original deterministic split assignment, so
  evaluation is not affected by filesystem traversal order.
- Training has normal GPU nondeterminism. Exact bitwise equality is not expected;
  the released checkpoint and the frozen test manifests are the authoritative
  route to reproduce the reported figure values.
