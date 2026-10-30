# MOF point-cloud regression package

Checkpoint archive: `moffieldbench-checkpoints.tar.gz`. Extract it into the repository root to restore `point_regression/checkpoints/`. Exact archive member paths and SHA-256 values are listed in [the checkpoint inventory](../docs/checkpoint_archive.json); see [installation instructions](../docs/EXTERNAL_ASSETS.md#checkpoint-archive).

> Public source distribution: raw CIF/CP3D/PLY data, model weights and compiled extensions are not bundled. See [external assets](../docs/EXTERNAL_ASSETS.md). Paths beginning with `external/` are anonymized archival references, not working defaults.

One package and one training entry point (`train.py`) for the 13 MOF-property tasks and QMOF band-gap regression.  The task is determined only by the data configuration (`target_column`, point path and split), so band gap is not a separate code path.

## Layout

```text
point_regression/
├── train.py                    # single entry point for all regression tasks
├── src/mofreg/                 # loaders, models, metrics, config handling
├── configs/
│   ├── property_ed.yaml        # 17,160-MOF ED template
│   ├── property_esp.yaml       # 17,160-MOF ESP template
│   ├── bandgap_ed.yaml         # QMOF band-gap manifest template
│   ├── bandgap_esp.yaml
│   ├── datasets.yaml           # all supported ED/ESP variants and roots
│   └── tasks.yaml              # fixed 13-property suite
├── data/                       # contracts/placeholders only; never raw point clouds
├── checkpoints/fig1/           # copied Fig. 1 band-gap ED/ESP checkpoints
├── checkpoints/FIG1_CHECKPOINTS.yaml
├── scripts/run_property_suite.py
└── vendor/openpoints/          # X3D source dependency retained with the package
```

`runs/` is intentionally the sole location for new training results.  Raw datasets are external and entered through environment variables or paths in a config; they are not mixed with source code or checkpoints.

## Installation

Use the project environment that contains PyTorch and the compiled OpenPoints operators, then install the small Python dependencies:

```bash
cd point_regression
pip install -r requirements.txt
```

For X3D native FPS, the included `vendor/openpoints` must be compatible with the Python/PyTorch/CUDA environment.  Set `model.fps_backend=linspace` for a CPU-safe smoke test; it is not a reproduction of native-FPS results.

## Data setup

The task labels and fixed splits are included under `data/tasks/<task>/` in
CGCNN-compatible `id_prop.csv` + `splits.csv` form.  Df/Di/Density use the
40k-scale merged split; the adsorption tasks use their fixed strict splits;
bandgap has its own 15,435-sample QMOF split. Raw PLY/CIF data is intentionally
not copied, but placeholder directories are included under
`data/pointclouds/{ed,esp}` and `data/cifs/`. You may populate them or leave
data in its existing storage. The exact layout and all 8
point-cloud variants are specified in [configs/datasets.yaml](configs/datasets.yaml):

- ED: `density_0.02`, `density_0.03`, `density_0.05`, `density_0.08`, `density_0.1`
- ESP: `esp_highest_lowest_3pct`, `esp_highest_lowest_5pct`, `esp_highest_lowest_7pct`

Set the roots once per shell (or edit the configs):

```bash
export MOF_PROPERTY_LABELS=$PWD/data/labels/selected_mofs_wide.csv
export MOF_ED_ROOT=external/ED_fps/non_periodic
export MOF_ESP_ROOT=external/EPS_fps/non_periodic
```

For band gap, create `data/manifests/qmof_bandgap_manifest.csv` from the template in the same directory.  It must have `qmof_id,split,ed_path,esp_path,bandgap`; `split` is `train`, `val`/`validation`, or `test`.  This manifest model means a future QMOF target uses the same `train.py`—only `target_column` and its manifest field change.

## Run one task

Property task with ED 0.03:

```bash
python train.py --config configs/property_ed.yaml --set \
  data.target_column=ASA \
  data.point_cloud_variant=density_0.03 \
  training.output_dir=runs/ASA_ed003
```

Property task with a different supported point cloud (ESP 5% shown):

```bash
python train.py --config configs/property_esp.yaml --set \
  data.target_column=post-combustion-vsa__selectivity \
  data.point_cloud_variant=esp_highest_lowest_5pct \
  training.output_dir=runs/postcombustion_esp5
```

Band gap uses exactly the same command and trainer:

```bash
export QMOF_MANIFEST=$PWD/data/manifests/qmof_bandgap_manifest.csv
python train.py --config configs/bandgap_ed.yaml --set training.output_dir=runs/bandgap_ed
python train.py --config configs/bandgap_esp.yaml --set training.output_dir=runs/bandgap_esp
```

Each run produces `best_model.pt`, `metrics.json`, and `history.json`.  The best checkpoint is selected using validation RMSE, and target normalization statistics/configuration are saved inside it.

## Run the 13-task suite

```bash
python scripts/run_property_suite.py \
  --config configs/property_ed.yaml \
  --output-root runs/property13_ed003 \
  --extra-set data.point_cloud_variant=density_0.03
```

Use `--dry-run` to print all thirteen commands first.  The suite is sequential by design, so multi-GPU scheduling can be managed by an external launcher without changing experiment definitions.

## Figure 1 reproducibility

The rendered Figure 1 files are outside this package:

- `external/test_point/result/paper fig1/fig1a_from_fig_data.png`
- `external/test_point/result/paper fig1/fig1b_from_fig_data.png`

The band-gap ED and ESP checkpoints, plus all 24 ED/ESP property checkpoints,
must be installed separately under `checkpoints/fig1/`. Four adsorption-property checkpoint
pairs were recovered from `MOF_X3D_PORTABLE`; their exact source and seed-42
configuration are listed in [checkpoints/FIG1_CHECKPOINTS.yaml](checkpoints/FIG1_CHECKPOINTS.yaml).
