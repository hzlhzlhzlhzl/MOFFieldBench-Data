# MOF 13-task multimodal prediction package

Checkpoint archive: `moffieldbench-checkpoints.tar.gz`. Extract it into the repository root to restore `multimodal/checkpoints/`. Exact archive member paths and SHA-256 values are listed in [the checkpoint inventory](../docs/checkpoint_archive.json); see [installation instructions](../docs/EXTERNAL_ASSETS.md#checkpoint-archive).

> Public source distribution: raw CIF/CP3D/PLY data, model weights and compiled extensions are not bundled. See [external assets](../docs/EXTERNAL_ASSETS.md). Paths beginning with `external/` are anonymized archival references, not working defaults.

This is the consolidated, reproducible package for the 13 tasks shown in the supplied heatmap. Band-gap is a normal registered task, not a separate code path. All training and evaluation use the same entry point:

```bash
cd multimodal
python scripts/run.py --config configs/base.yaml ...
```

## Layout

```text
multimodal/
  mof_multimodal/       # one shared CIF + point-cloud model, loader, trainer
  configs/
    base.yaml           # model/training defaults
    tasks.json          # the 13 registered targets
  data/
    dataset_manifest.example.json
    labels/, cif/, point_clouds/, point_packs/  # intentionally empty placeholders
  third_party/
    point/mof_point/    # bundled point encoder implementation
    cif/vendor/cgcnn/   # bundled CGCNN implementation
  checkpoints/
    manifest.json       # exact 39 selected checkpoints (13 tasks × 3 multimodal settings)
    artifacts/          # materialized local weights; generated and git-ignored
  results/              # numeric values behind the final heatmap
  scripts/
```

The three multimodal settings are `cif_ed`, `cif_esp`, and `cif_ed_esp` (CIF + ED + ESP). `cif` is also available as a CIF-only baseline. The registry includes PLD/Df, LCD/Di, Density, band gap, four selectivity tasks, and five uptake/working-capacity tasks.

## Environment

Use the existing X3D environment used for the archived experiments; it provides PyTorch, NumPy, pandas, scikit-learn, tqdm, and the X3D dependencies. The package bundles the project-specific `mof_point` and CGCNN code. A CUDA GPU is recommended; change `training.device` to `cpu` only for a small smoke test.

## Configure data (raw data are deliberately not bundled)

Copy the template and replace every `<...>` with absolute local paths:

```bash
cd multimodal
cp data/dataset_manifest.example.json data/dataset_manifest.json
```

The manifest keeps data paths out of source code. It supports either unpacked PLY clouds or packed `.npy` clouds with adjacent `.ids.txt`. Each variant has its own name and metadata; it is not restricted to ED threshold `0.03`. Examples include ED thresholds `0.02/0.03/0.05`, ED top-k, ESP percentile/top-k variants, and 4,096/8,192/10,240 points. Add any new variant under `point_cloud_variants` and select it with a config override or make it the default in your local manifest.

For each task, `cif_data_root` must point to the corresponding CGCNN dataset directory and `split_csv` must describe the same train/validation/test IDs as the label and point data. The loader checks cross-modal IDs before training.

## Run

List the registered tasks:

```bash
python scripts/run.py --config configs/base.yaml --list-tasks
```

Train any target through the same command, including band gap:

```bash
# Band gap, CIF + ED
python scripts/run.py \
  --config configs/base.yaml --task bandgap --modalities cif_ed \
  --data-manifest data/dataset_manifest.json

# A MOFFieldBench task, CIF + ESP
python scripts/run.py \
  --config configs/base.yaml --task methane_uptake --modalities cif_esp \
  --data-manifest data/dataset_manifest.json
```

Useful overrides are passed after `--set`, for example:

```bash
--set training.device=cuda:0 training.output_dir=results/my_run training.epochs=50
```

Run all 13 sequentially (choose one of the three multimodal settings):

```bash
python scripts/run_all_13.py \
  --modalities cif_ed_esp --data-manifest data/dataset_manifest.json
```

Add `--dry-run` to either command to validate data alignment, model construction, and one batch without training.

## Reproduce archived selected checkpoints

`checkpoints/manifest.json` is an audited index of the checkpoint selected for every heatmap cell. It records method, test MAE/RMSE/R², source path, seed (42), and SHA-256. The selected band-gap entries are the pure scratch QMOF runs used in the final figure; the other 36 entries come from the corresponding archived 12-task sweep.

To arrange all 39 checkpoints as a clean local collection and verify them:

```bash
python scripts/materialize_checkpoints.py --source-root /path/to/checkpoints
```

Hard links are used when possible, so this normally consumes no additional disk space. See `checkpoints/README.md` for details. Each training run also writes `resolved_config.json`, `alignment_report.json`, `history.csv`, `metrics.json`, `test_predictions.csv`, and `best_model.pt`, so a result can be independently audited and evaluated with `--evaluate-only`.
