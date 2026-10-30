# Historical validation record

Validation date: 2026-10-29. Later public-release checks are documented in the repository-level validation record.

## Automated checks

The following ran in an existing Python 3.10 / PyTorch 1.13.1 environment:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests -q
10 passed in 8.74s
```

Coverage included Fourier constant/mean preservation for odd, even, and single-cell target axes; low-frequency reconstruction; FPS with duplicate coordinates; CP3D-to-ED/ESP periodic and nonperiodic PLY processing; index prefixes and padding; resume and configuration changes; geometry mismatch rejection; periodic positional encoding; PointNet forward/backward passes; and material-inventory integrity.

## Real production sample

For `mof_1730`, an existing `*v_hartree*.cube.cp3d` file was decoded and processed through Fourier downsampling, upper/lower 5% ESP selection, concatenation, and nonperiodic FPS with 6,144 points.

- Original shape: `192 × 288 × 288`.
- Per-axis factor: `1.25`; output shape: `154 × 230 × 230`.
- Upper tail: 407,331 points; lower tail: 407,330 points; combined: 814,661 points.
- All final `x,y,z,esp,frac_x,frac_y,frac_z` records matched the original production PLY exactly; every column had zero maximum absolute error.
- Updated headers and reports include corrected units, so the comparison used vertex records, not whole-file binary hashes.

See [real_sample_validation.json](real_sample_validation.json). Validation outputs were written to a temporary directory, not `data/`, and original results were not overwritten.

The first CUDA attempt encountered a local multi-GPU visibility initialization error. Restricting the process to physical GPU 3 (`CUDA_VISIBLE_DEVICES=3`, process-local `cuda:0`) succeeded. System settings and existing training processes were unchanged.

## X3D model

A native-CUDA-FPS forward check used an existing Python 3.9 / PyTorch 2.2.2 environment and its compatible external `pointnet2_batch_cuda` extension. The bundled X3D source processed two samples with 128 points and 16 features each, using reduced width/depth. The output shape was `[2]` and all values were finite. This checked the model/operator interface. Only operator source is distributed; no `.so` is bundled, and compilation in a fresh environment was not tested.

## Scope

No full reconstruction of all 40,343 materials or real-data training was performed. Placeholder inventories were checked for row counts, unique IDs, and source counts. This historical validation did not audit the existence of every external CP3D/PLY file.
