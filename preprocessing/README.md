# MOF CP3D → Fourier downsampling → FPS point clouds

> Public source distribution: raw CIF/CP3D/PLY data, model weights and compiled extensions are not bundled. See [external assets](../docs/EXTERNAL_ASSETS.md). Paths beginning with `external/` are anonymized archival references, not working defaults.

This portable `data / code / model` project consolidates production scripts while retaining separate ED and ESP processing chains. `data/` indexes **40,343 materials: 24,908 ARC-MOF and 15,435 QMOF**. It contains material inventories and large-file placeholders, not training splits, CIFs, labels, actual CP3D/PLY files, or model weights.

```text
preprocessing/
├── run.py                         # Unified entry point
├── code/
│   ├── cp3d/                      # CP3D v1/v2 decoding and compression
│   └── mof_pipeline/
│       ├── fourier.py              # 3D-field FFT → spectrum cropping → IFFT
│       ├── pointclouds.py          # ED thresholds, ESP tails, isodensity clouds
│       ├── fps.py                  # Periodic/nonperiodic progressive FPS; PLY I/O
│       ├── pipeline.py             # Single/batch processing, resume, failure reports
│       └── manifest.py             # Source and placeholder-path inventories
├── model/
│   ├── train.py                    # Optional downstream regression
│   ├── src/mofreg/                 # Loaders, Fourier encoding, models, training
│   ├── vendor/openpoints/          # X3D and CUDA operator source
│   └── checkpoints/                # External-weight location
├── data/
│   ├── manifest.csv               # 40,343 materials; no split column
│   ├── inventory.json             # Counts, provenance, original inventory checksum
│   ├── arcmof/{materials.txt,cp2k_cube/}
│   ├── qmof/{materials.txt,cp2k_cube/}
│   ├── point_clouds/              # Intermediate clouds before FPS
│   ├── ED_fps/non_periodic/       # ED output location
│   └── EPS_fps/non_periodic/      # ESP output; original EPS spelling retained
├── configs/                       # Full pipeline, ESP5%/6144, ED/ESP models
├── tests/                         # Numerical, end-to-end, and model interface tests
└── docs/                          # Provenance, implementation changes, validation
```

## Data scope

The original `cif_all_strict_e50/Df/splits.csv` covers 24,908 ARC-MOF materials. The merged `output_merged_2w_qmof/splits/Df.csv` covers 40,343 materials. This module retains material IDs, represented as `mof_<number>`, and a separate `source` column; it does not retain the source training split. These are selected inventories, not the complete ARC-MOF or QMOF databases.

Each `manifest.csv` record provides CP3D discovery patterns and expected paths for ED0.03 / ESP5% clouds with 6,144 points. `data_status=placeholder` means the actual data were not copied. Pipeline configurations control all other variants. Fields such as `*ELECTRON_DENSITY*.cp3d` are glob patterns, allowing filenames to differ from material directory names.

## Environment

Use Python 3.10+:

```bash
cd preprocessing
python -m pip install -r requirements.txt
python run.py --help
```

The CP3D decoder is bundled; an external workspace installation is unnecessary. Base dependencies support field processing, FPS, and PointNet/Transformer. See [model/README.md](model/README.md) for additional X3D dependencies and CUDA compilation.

## Generate a PLY from CP3D

1. Place real field files in `data/arcmof/cp2k_cube/<material>/` or `data/qmof/cp2k_cube/<material>/`, or set `input_roots` in a JSON configuration. Relative paths resolve from this module's root; absolute paths are also accepted. Material directories must use the public IDs.
2. A material normally has `*ELECTRON_DENSITY*.cube.cp3d` and `*v_hartree*.cube.cp3d`. These supply ED and ESP, respectively; ED alone cannot generate ESP.
3. Run the ESP5%, 6,144-point configuration:

```bash
python run.py pipeline --config configs/esp5_6144.json \
  --materials mof_1730 --dry-run
python run.py pipeline --config configs/esp5_6144.json \
  --materials mof_1730
```

Output:

```text
data/EPS_fps/non_periodic/esp_highest_lowest_5pct/
  mof_1730/nonperiodic_fps/
    esp_highest_lowest_5pct_nonperiodic_fps_6144.ply
```

To configure an external data location, create a local configuration and replace the example root with your actual data path:

```bash
python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('configs/esp5_6144.json').read_text())
c['input_roots'] = ['external/mof_cp2k/cp2k_cube']
Path('configs/local.json').write_text(json.dumps(c, indent=2))
PY
python run.py pipeline --config configs/local.json --materials mof_1730
```

A single data root may cover only part of ARC-MOF. Add all batch and QMOF roots when processing the complete inventory. Duplicate material names across input roots raise an error so the source must be resolved explicitly.

## Full pipeline and batch processing

```bash
# Inspect the default 40,343-material inventory without reading large files.
python run.py pipeline --config configs/pipeline.json --dry-run

# Check a small subset after connecting actual data.
python run.py pipeline --config configs/pipeline.json --limit 2

# Process all materials; increase material-level workers if resources allow.
python run.py pipeline --config configs/pipeline.json --workers 1 --threads 2

# Run the two stages separately.
python run.py pipeline --config configs/pipeline.json --stage clouds --materials mof_1730
python run.py pipeline --config configs/pipeline.json --stage fps --materials mof_1730
```

Defaults include ED thresholds `0.02/0.03/0.05/0.08/0.1`, each ESP tail at `3%/5%/7%`, isodensity windows, and nonperiodic FPS at `4096/6144/8192/10240` points. Set `methods` to `["nonperiodic", "periodic"]` to generate both modes. `device` accepts `cpu` or `cuda:<index>`; multiple workers use the same selected GPU.

Completed tasks are skipped when their inputs and configuration are unchanged, and missing outputs are regenerated. Use `--force` after changing inputs or parameters. Successful cloud filtering writes `DONE.json`; FPS produces index `.npy` files, `fps_manifest.json`, `fps_metrics.csv`, and `fps.log`. The intermediate directory receives `RUN_REPORT.json`. Any material failure produces a nonzero exit status; missing data are not treated as success.

## Numerical conventions

| Original grid points | Per-axis downsampling factor |
| --- | ---: |
| ≤ 13,000,000 | 1: retain the original grid |
| ≤ 25,000,000 | 1.25 |
| ≤ 50,000,000 | 1.5 |
| > 50,000,000 | 2 |

Each target axis length is `round(original_length / factor)`. The cell and origin are preserved. The existing pyRho-compatible Fourier implementation treats fields as periodic; it does not clip negative densities, reverse the ESP sign, or force zero-mean ESP. `downsample` can also specify an explicit factor ≥1.

ED selection uses `density >= threshold`, retaining the historical `density_gt_...` filenames. ESP tails use quantiles of the full resampled grid and include points equal to each threshold, so the selected count can slightly exceed the nominal percentage. The upper tail is concatenated before the lower tail, followed by joint FPS; the final tails are not forced to have equal counts.

FPS defaults to seed `20260726`. It computes one ordering for the largest requested count; smaller outputs are prefixes. If too few points are available, `size_policy=clip` retains production behavior by cyclically repeating indices to reach the requested length; `error` raises an error instead. Periodic modes are `validated8/images27/wrapped`. `validated8` checks consistency with 27 neighboring images but does not guarantee the globally shortest image distance for arbitrary extreme skewed cells.

Coordinates retain input units, which are Bohr for the production CP2K data. Reports use `coordinate_unit` and distance fields without misleading unit suffixes. This corrects historical reports labeled `angstrom` without changing coordinate values.

Model-side `frac_pe` is separate from grid downsampling: it evaluates `sin/cos(2π·2^k·fractional_coordinate)` at points already selected by FPS. With xyz, one scalar, and K frequency levels, each point has `4+6K` features.

## Validation

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

Tests generate synthetic fields and cover Fourier means/low-frequency modes, both FPS modes, padding and nested prefixes, resume behavior, ED/ESP geometry mismatches, model loading, and backpropagation. See [historical validation](docs/VALIDATION.md), [implementation changes](docs/CHANGES.md), and [source mapping](docs/SOURCE_MAP.json).

The public module does not bundle machine-wide storage inventories. See [data location guidance](docs/CP3D_LOCATIONS.md).
