# Data placeholders

The inventory contains 40,343 materials: 24,908 ARC-MOF and 15,435 QMOF.

- `manifest.csv`: material ID, source, directory name, input discovery patterns, and expected PLY paths. Every row has `placeholder` status.
- `arcmof/materials.txt` and `qmof/materials.txt`: complete ID lists by source.
- `arcmof/cp2k_cube/` and `qmof/cp2k_cube/`: locations for external ED/Hartree-potential CP3D files.
- `point_clouds/`: intermediate clouds after Fourier processing and threshold/quantile selection.
- `ED_fps/` and `EPS_fps/`: FPS output locations.

Actual fields and point clouds are not bundled; neither are training splits, CIFs, or labels. Material IDs come from the merged 40,343-row inventory, with provenance and checksums in `inventory.json`. Records describe data to be connected, not files already present. Raw fields are discovered using glob patterns; filenames need not exactly match their parent directory names.
