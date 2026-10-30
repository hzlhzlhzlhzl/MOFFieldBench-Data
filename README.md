# MOFFieldBench

Research code for MOF electron density (ED), electrostatic potential (ESP), and crystal structures. The collection covers field preprocessing, field prediction, point-cloud property regression, cross-modal retrieval, and multimodal property prediction.



## Material identifiers

All materials use `mof_<number>`. Original names appear only in the [name mapping](docs/mof_name_mapping.csv). There are 40,364 identifiers: the first 40,343 correspond to the CP3D inventory, and another 21 come from the CIF–ED retrieval inventory. The QMOF label table retains 15,435 rows after removing 4,937 additional materials without registered CIF/ED/ESP paths. Existing training, validation, and test splits are unchanged. See [naming rules and validation](docs/MATERIAL_NAMING.md).

## Repository layout

| Directory | Purpose | Entry point |
| --- | --- | --- |
| [`preprocessing/`](preprocessing/README.md) | CP3D → Fourier downsampling → ED/ESP point clouds → FPS | `run.py`; optional downstream models included |
| [`field_prediction/`](field_prediction/README.md) | Field prediction, regional error evaluation, and plotting | `run.py` |
| [`point_regression/`](point_regression/README.md) | Point-cloud regression for MOF properties  | `train.py` |
| [`retrieval/`](retrieval/README.md) | CIF ↔ ED, CIF ↔ ESP, and ED ↔ ESP retrieval | `run.py` |
| [`multimodal/`](multimodal/README.md) | Property prediction from CIF, ED, and ESP | `scripts/run.py` |
| [`docs/`](docs/EXTERNAL_ASSETS.md) | External assets and release documentation | Asset inventories and validation records |

Each module retains its original Python package structure and entry points to preserve internal imports. Run commands from the relevant module directory and use separate environments. The research environments have not been merged into a single dependency environment.

## Quick start: replot reference results

This workflow requires no weights, GPU, or raw field data. Use Python 3.10:

```bash
cd field_prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
python run.py verify
python run.py plot
```

Figures are written to `field_prediction/outputs/figures/`. `verify` checks public-file hashes and material splits, explicitly reporting skipped checks when weights are absent. After installing weights, use `python run.py verify --load-models` to validate model loading.

Generate the retrieval reference figure in an environment with NumPy and Matplotlib:

```bash
cd retrieval
python run.py figure --output-dir runs/figure
```

These commands replot archived metrics; **they do not rerun model evaluation**.

## Data preprocessing

```bash
cd preprocessing
python -m pip install -r requirements.txt
python run.py pipeline --config configs/pipeline.json --dry-run
```

Configure `input_roots` in `configs/pipeline.json` with your actual CP3D locations before running processing. Numerical conventions, coordinate units, thresholds, and FPS behavior are documented in the module README.

## Training and full evaluation

1. Read the selected module's README and `data/README.md`, and prepare data according to its contract.
2. Install its dependencies in a separate environment. Build X3D/OpenPoints CUDA extensions locally; this repository does not provide machine-specific `.so` binaries.
3. Configure local paths through module settings or environment variables. Paths beginning with `external/` are anonymized historical references, not usable default data locations.
4. Install the appropriate checkpoints before reproducing fixed-weight results. See [external assets](docs/EXTERNAL_ASSETS.md).



## Validation

```bash
python scripts/check_release.py
cd preprocessing
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

See the [validation record](docs/VALIDATION.md) for completed and unperformed checks. Historical experiment records inside modules do not imply that training or full reproduction was repeated for this release.



## Licensing and citation

Existing third-party LICENSE files, copyright notices, and upstream references remain in place; see [third-party notices](THIRD_PARTY_NOTICES.md). No license has been chosen on behalf of the maintainers for original project code, and no authors, paper DOI, or citation details have been invented. A public repository does not automatically grant an open-source license. Maintainers should add a root LICENSE and formal citation once these are determined. Use and redistribution of original datasets remain subject to their source terms.
