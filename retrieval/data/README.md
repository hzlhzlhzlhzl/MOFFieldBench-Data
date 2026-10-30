# Data contract

Raw data are intentionally not included in this package.  Put (or symlink) them
under this directory using the layout below.  The packaged manifests preserve the
exact train/validation/test assignment used for the released checkpoints.

```text
data/
├── cifs/
│   └── <sample_id>.cif
└── point_clouds/
    ├── ed/
    │   └── <periodicity>/density_<threshold>/<sample_id>/<fps_dir>/
    │       └── density_gt_<threshold>_<fps_dir>_<N>.ply
    └── esp/
        └── <periodicity>/esp_highest_lowest_<percent>pct/<sample_id>/<fps_dir>/
            └── esp_highest_lowest_<percent>pct_<fps_dir>_<N>.ply
```

`<periodicity>` is normally `non_periodic` / `nonperiodic` according to the data
source; retain the path below `ED_fps` or `EPS_fps` verbatim. `<fps_dir>` is
normally `nonperiodic_fps`.  A PLY must contain `x`, `y`, `z`, one scalar
(`density` for ED or `esp` for ESP), and `frac_x`, `frac_y`, `frac_z`.

This layout deliberately supports all point-cloud variants rather than treating
ED 0.03 as special: examples include `density_0.02`, `density_0.03`,
`density_0.05`, `density_0.08`, `density_0.1`; ESP 3%, 5%, and 7%; arbitrary FPS
counts such as 4096, 6144, 8192, and 10240; and periodic/non-periodic variants.
For a new variant, copy a config, update its point-cloud fields, and provide a
manifest with the same columns as the packaged one.

The three packaged split manifests are source-path records. `run.py` converts
them to absolute paths under `--data-root` automatically, mapping CIFs by
filename and preserving each PLY's suffix below `ED_fps` / `EPS_fps`.
