# MOF ED / ESP field prediction

Checkpoint archive: `moffieldbench-checkpoints.tar.gz`. Extract it into the repository root to restore `field_prediction/checkpoints/`. Exact archive member paths and SHA-256 values are listed in [the checkpoint inventory](../docs/checkpoint_archive.json); see [installation instructions](../docs/EXTERNAL_ASSETS.md#checkpoint-archive).

> Public source distribution: raw CIF/CP3D/PLY data, model weights and compiled extensions are not bundled. See [external assets](../docs/EXTERNAL_ASSETS.md). Paths beginning with `external/` are anonymized archival references, not working defaults.

This module packages the models, training/prediction/evaluation code, checkpoint configurations, material inventory, and fixed test split behind two regional-error figures. All commands use `run.py`; generated results go to `outputs/`. Raw data locations are placeholders, and weights must be installed separately.

## Layout

```text
field_prediction/
├── run.py                       # Unified entry point
├── requirements.txt             # Inference, evaluation, and training dependencies
├── requirements-plot.txt        # Dependencies for replotting archived results
├── configs/reproduce.json       # Fixed checkpoints, cutoff, and sampling settings
├── src/mof_fields/              # Training, prediction, evaluation, and plotting
├── models/charge3net/           # Model and periodic graph source; upstream LICENSE
├── third_party/cp3d/            # Local CP3D reader/writer source
├── checkpoints/                 # Install external weights here
│   ├── ed/step_157000.pt
│   └── esp/step_294000.pt
├── data/
│   ├── raw/{cif,ed,esp}/        # External data locations
│   ├── manifests/materials.json # Relative paths for 40,345 materials
│   ├── manifests/material_families.json # Public-ID family relationships
│   └── splits/split_40345.json  # Original split membership and ordering
├── reference/
│   ├── metrics/                # Archived per-MOF CSVs and expected summaries
│   ├── figures/                # Reference PNG/PDF figures
│   └── provenance/             # Sources, configuration checks, and hashes
└── outputs/                    # New evaluation, prediction, training, and plots
```

The module can be moved as a whole. Default paths resolve relative to `run.py`; no external `test_gen`, `training_runs`, model-source, or CP3D installation paths are required. Explicit relative command-line paths resolve from the working directory; relative paths inside a manifest resolve from that manifest's directory.

## 1. Environment and checks

Use Python **3.10** and run these commands from this module's root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py verify  # Add --load-models after installing checkpoints.
```

`verify` checks public-file hashes and split integrity for 40,345 materials, as well as checkpoint hashes when weights are installed. `--load-models` also checks step, epoch, and cutoff metadata and strictly loads model parameters. GPU inference requires a PyTorch CUDA build compatible with the driver. The observed working environment is recorded in `reference/provenance/environment.json`; installation in a fresh environment was not validated during packaging.

Replotting archived results requires no PyTorch, GPU, or raw data:

```bash
python -m pip install -r requirements-plot.txt
python run.py verify
python run.py plot
```

The same `verify` and `plot` commands can be used directly in an existing compatible environment.

## 2. Settings behind the two figures

| Setting | ED | ESP |
| --- | --- | --- |
| Checkpoint | `checkpoints/ed/step_157000.pt` | `checkpoints/esp/step_294000.pt` |
| Step / epoch | 157000 / 8 | 294000 / 15 |
| Cutoff | 4 Å | 4 Å |
| Test set | Fixed original set of 2,018 MOFs | Same test set |
| Samples per MOF per region | 2,000 | 2,000 |
| Random seed | 20261003 | 20261003 |
| Inference chunk | 512 probes | 512 probes |
| Displayed regions | Uniform; ED > 0.01 / 0.05 / 0.10 | Uniform; bottom/top 3%; bottom/top 5% |
| Displayed percentages | 0.51, 0.49, 0.46, 0.44 | 8.42, 0.45, 3.72, 0.58, 3.91 |

Model settings: `num_interactions=3, num_neighbors=20, mul=500, lmax=4, basis=gaussian, num_basis=20`. Expected SHA-256 values for both external checkpoints are retained from the original evaluation records; see `checkpoints/README.md`.

For each MOF and region:

```text
NMAE(%) = 100 × sum(abs(prediction - reference)) / sum(abs(reference))
Bar height = arithmetic mean of these percentages across 2,018 MOFs
```

This is neither a pooled metric over all points from all MOFs nor a mean of pointwise percentage errors. “All grid points” means uniform sampling from the complete grid: 2,000 points per MOF, not exhaustive evaluation of every grid point.

ED thresholds use **electron/Å³**; ED values stored in CP3D as electron/Bohr³ are converted during evaluation. ESP regions use reference-potential quantiles for each MOF: upper tails include values `>=` the upper quantile, and lower tails include values `<=` the lower quantile. Sampling uses replacement when a region has fewer than 2,000 candidates.

To preserve original sampling, evaluation retains the complete region order: ED thresholds `0.1, 0.08, 0.05, 0.03, 0.02, 0.01` and ESP tails `3, 5, 7%`. Plotting displays only the regions in the table. Each material first generates up to 100,000 distance candidates, then samples regions. Do not remove undisplayed regions or change random-number call order. The material seed is `20261003 + original_test_index × 20`; the original evaluation used two interleaved shards.

## 3. Replot both figures

```bash
python run.py plot
```

By default, this reads the four per-MOF CSVs in `reference/metrics/`, recomputes macro means, and writes:

```text
outputs/figures/esp_error_by_potential_region_mean_per_mof.png
outputs/figures/esp_error_by_potential_region_mean_per_mof.pdf
outputs/figures/ed_error_by_density_region_mean_per_mof.png
outputs/figures/ed_error_by_density_region_mean_per_mof.pdf
outputs/figures/macro_mean_per_mof_summary.csv
```

This does not run checkpoint inference. Reference CSVs are archived experiment results, not data placeholders. Plotting checks material membership, sample counts, and duplicate rows; a small smoke-test subset cannot stand in for a complete paper figure.

## 4. Prepare raw data

See `data/README.md`. Each material uses `data/raw/cif/<material>.cif`, `data/raw/ed/<material>.cp3d`, and `data/raw/esp/<material>.cp3d`.

Given a source manifest with public material IDs and valid local paths, create links for the test set:

```bash
python run.py prepare-data --source-manifest /path/to/source_manifest.json --split test --mode symlink
python run.py verify --data
```

Use `--mode copy` for an independent data distribution and `--split all` for training data. Existing files are never overwritten. Convert original material IDs using the repository mapping before supplying an old manifest.

Alternatively, keep data in place and pass `--manifest /path/to/source_manifest.json` to `evaluate`, `train`, or `predict`. The anonymized historical manifest location is:

```text
external/mof_cp2k/test_gen/training_runs/paired_cp3d_random/manifest.json
```

This is a provenance reference, not a runtime dependency. Full evaluation reports missing data explicitly. Source-hash verification and replotting remain available without raw data; model loading requires external checkpoints.

## 5. Reevaluate checkpoints

Actual data and weights are required:

```bash
python run.py evaluate --target both --device cuda:0
python run.py compare --metrics-dir outputs/evaluation
python run.py plot --metrics-dir outputs/evaluation --output-dir outputs/recomputed_figures
```

On one GPU, the default runs both shards for ED and ESP sequentially. Individual runs can select `--target ed` or `--target esp`, `--shard-index 0` or `1`, and `--device cuda:N`; each target/shard has a separate output file. Settings are frozen in `configs/reproduce.json`. `--device cpu` is supported but slower.

For a small check, use one MOF per shard; this does not replace full evaluation:

```bash
python run.py evaluate --target both --device cuda:0 --limit 1
python run.py compare --metrics-dir outputs/smoke
```

`compare` checks per-MOF normalized errors for supplied rows with a default tolerance of 0.001 percentage points and writes `comparison.json`. It does not guarantee full test-set coverage; `plot` performs completeness checks. Hardware and library versions may introduce floating-point differences.

Packaging checked the archived weights and settings but **did not rerun the full 2,018-MOF sampling evaluation**. See `reference/provenance/validation.json` for historical checks.

## 6. Prediction and training

Prediction uses a material's CIF and CP3D grid metadata from the manifest. These examples predict only the first 512 points and write preview NPZ files:

```bash
python run.py predict --target ed --device cuda:0 --materials mof_32514 --preview-probes 512
python run.py predict --target esp --device cuda:0 --materials mof_32514 --preview-probes 512
```

Omit `--preview-probes` to predict the full grid and write CP3D. Prediction reads reference CP3D geometry metadata without decoding reference field values; regional evaluation does read those values. Outputs go to `outputs/predictions/ed/` and `outputs/predictions/esp/`.

Training code, default optimization settings, and original random splits are retained. After preparing all data, train from scratch or resume:

```bash
python run.py train --target ed --device cuda:0 --max-steps 157000
python run.py train --target esp --device cuda:0 --max-steps 294000
python run.py train --target ed --device cuda:0 --resume checkpoints/ed/step_157000.pt --max-steps 158000
```

Defaults: Adam, lr=0.005, PowerDecayScheduler(alpha=0.96,beta=3000), batch_size=1, train/validation/test probes=500/1000/1000, seed=20260811. For multiple GPUs, use `torchrun --nproc_per_node=2 run.py train --target ed ...`. Retraining is not guaranteed to produce bitwise-identical weights; the designated external checkpoints define the reference figures. Resuming retains step, epoch, and optimizer state; the original trainer does not save the within-epoch iterator position or all random-number-generator states.

```bash
python run.py --help
python run.py predict --help
python run.py train --help
```

## Provenance and integrity

`reference/provenance/original_runs.json` retains evaluation settings and anonymized historical paths, which are not default runtime locations. `integrity.json` records public-file hashes; expected external checkpoint hashes remain in the reproduction configuration. `environment.json` records observed dependency versions, and `validation.json` records performed and unperformed checks.

Original project files remain in place; this module is a separately organized copy. Upstream model copyright and licensing are in `models/charge3net/LICENSE`.
