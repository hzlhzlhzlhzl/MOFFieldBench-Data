# Data layout

Raw data are excluded. Put or link local data beneath this directory if desired:

```text
data/
  labels/             # CSV: one ID column and every target column
  cif/<dataset>/<task>/
                       # CGCNN CIFData directory with id_prop.csv, atom_init.json and splits.csv
  point_clouds/<variant>/<sample_id>/nonperiodic_fps/*.ply
  point_packs/        # optional .npy packs plus matching .ids.txt files
```

`dataset_manifest.example.json` is the only file whose paths need changing. Use `cif_data_root_template` with `{safe_target}` when each task is in a similarly named CIF directory, or `cif_data_root_by_task` for explicit per-task locations. A point-cloud variant is arbitrary metadata: ESP/ED, threshold, percentile, top-k, sampling method, and point count may all be represented. The code never assumes a fixed threshold such as 0.03.
