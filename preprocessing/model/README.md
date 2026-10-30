# Point-cloud model code

This module consolidates X3D, PointNet, Transformer, PLY loaders, fractional-coordinate Fourier encoding, and training/evaluation code from the original point-cloud regression project. `checkpoints/` is reserved for external weights; historical models, logs, and raw data are not bundled.

Inputs are FPS PLY properties `x y z`, an ED/ESP scalar, and `frac_x frac_y frac_z` (also accepting `fx fy fz`). The loader reads properties by name and centers and normalizes coordinates. `frac_pe` uses K sine/cosine frequency levels at `2^k`; K=2 produces 16 input features.

`src/mofreg/models.py` contains PointNet/Transformer and the model factory. `repo_models.py` wraps X3D, whose backbone is `vendor/openpoints/models/backbone/pointmetabase_X3D.py`. Hierarchical FPS inside X3D and preprocessing FPS that generates PLY files are separate stages.

## Optional downstream training

The module's data inventory covers ARC-MOF/QMOF fields and point clouds. For training, supply an external label CSV (for example, columns `base,Df`) and a split CSV (`id,split`, with train/val/test values), both using public material IDs:

```bash
export MOF_LABELS=/absolute/path/to/labels.csv
export MOF_SPLITS=/absolute/path/to/splits.csv
python run.py train --config configs/model_esp.yaml --set model.name=pointnet
# For ED, use configs/model_ed.yaml.
```

Run these commands from `preprocessing/`. Training requires labels, splits, and point clouds to be available. By default, all materials in the external split must have labels and PLY files, preventing silent training on a reduced subset. Model configurations are runnable templates, not claims of reproducing any archived experiment's metrics.

## Native X3D operators

```bash
python -m pip install -r requirements-x3d.txt
cd model/vendor/openpoints/cpp/pointnet2_batch
python setup.py build_ext --inplace
cd ../../../../..
python run.py train --config configs/model_esp.yaml
```

Select a `torch-scatter` build compatible with your PyTorch/CUDA versions and provide a working CUDA compiler toolchain. Only CUDA source is bundled, not `.so` files tied to an older Python ABI. Entry points discover the current build directory instead of assuming Python 3.9. PointNet can run with plain PyTorch before native dependencies are installed.

Training retains label standardization based on the training set, MSE, AdamW, cosine learning rates, checkpoint selection by validation RMSE, and MAE/RMSE/R² reporting. Results are written under the module's `outputs/` directory. No real-data training was launched during packaging.
