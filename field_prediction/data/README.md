# Data layout

`raw/` is reserved for external data; original CIF and CP3D files are not bundled. `manifests/` and `splits/` contain actual material inventories and fixed splits and must remain with the code.

Each material uses the same public material ID across modalities:

```text
data/raw/cif/<material>.cif
data/raw/ed/<material>.cp3d
data/raw/esp/<material>.cp3d
```

`manifests/materials.json` contains 40,345 materials with fields `material, source, cif, ed, esp`. File paths resolve relative to the **manifest directory**, so moving the whole module requires no path changes. Copy or symlink original files according to the repository's name mapping when filenames differ. Use the explicit manifest mapping rather than guessing names from old suffix conventions.

ED values in CP3D use electron/Bohr³; ESP retains the native values of the original CP2K Hartree potential. Geometry origins and axes use Bohr, while CIF cells and coordinates use Å. ED threshold evaluation converts to electron/Å³. Preserve reference fields, cells, origins, grid shapes, and C-order layout; do not regenerate substitute reference data.

The original random split has 36,310 training, 2,017 validation, and 2,018 test materials. Do not generate a new split. Evaluating the two figures requires the test set's 6,054 CIF/ED/ESP files; retraining requires the complete dataset.

Use `--manifest /path/to/manifest.json` for an existing manifest with valid paths and matching public IDs, or use the module README's `prepare-data` command to populate this layout. Symlinks are suitable for local use; use copy mode for independent data distribution.
