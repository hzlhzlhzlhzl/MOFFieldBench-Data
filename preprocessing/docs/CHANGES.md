# Provenance and packaging changes

`SOURCE_MAP.json` records anonymized source references and original SHA-256 values. Production data and original code were not modified. Packaging added a unified CLI, batch configurations, material inventories, documentation, and tests.

| Packaged component | Original implementation |
| --- | --- |
| `code/cp3d/` | `mof_cp2k/cp3d/src/cp3d/` |
| `code/mof_pipeline/fourier.py` | `fourier_downsample_cp2k_cube.py` |
| `code/mof_pipeline/pointclouds.py` | Selection and writing functions from `build_cp3d_point_clouds_gt13m.py` |
| `code/mof_pipeline/fps.py` | `fps/mof_periodic_nonperiodic_fps.py` |
| `model/` | Training, mofreg, and OpenPoints source from the original point-regression package |

Changes made during packaging:

- Removed hardcoded working directories and the dependency on a grid-statistics CSV. Grid sizes are read from CP3D; input roots and output settings are configured in JSON.
- Retained material IDs for batch inventories; this module's data directory has no splits, CIFs, or labels. Raw data, checkpoints, build products, and caches were not copied.
- Added checks for matching ED/ESP shapes, origins, cells, and units; scalar values and point coordinates must be finite. Failures return nonzero status.
- Stored input fingerprints and configuration for resume behavior. Completion markers are written only after success, and failure logs are retained.
- Fixed an empty negative-frequency slice when a Fourier target axis has length one; ordinary-size behavior is unchanged.
- Explicitly excluded selected rows during FPS to prevent repeated selection of the same row for duplicate/zero-distance inputs. Padding still repeats indices when the requested count exceeds available points. `unique_sample_count` counts distinct selected rows, not distinct spatial coordinates.
- Corrected historical FPS reports that labeled Bohr coordinates as angstrom. Coordinate values are unchanged; final PLY files record `coordinate_unit`.
- Updated model loaders to select PLY properties by name and check truncated/nonfinite data. External training splits can require complete coverage.
- Made X3D discover the current environment's build directory. An unconfigured PointMamba branch was not included in this module.

Periodic-image distances and grid FFT processing retain the original implementations. `images27` searches a finite image neighborhood and is not suitable for every extreme unreduced cell. This module processes existing CP2K scalar fields; it is not a model that generates electron density from CIFs.
