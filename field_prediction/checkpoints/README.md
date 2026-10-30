# External reference checkpoints

Weights are distributed in the separate `moffieldbench-checkpoints.tar.gz` archive. The archive members are `field_prediction/checkpoints/ed/step_157000.pt` and `field_prediction/checkpoints/esp/step_294000.pt`. Extract the archive into the repository root to install the files at the module-relative paths below. Original checkpoint metadata may contain private paths; review it before redistribution.

| Target | File | Step | Epoch | Cutoff |
|---|---|---:|---:|---:|
| ED | `ed/step_157000.pt` | 157000 | 8 | 4 Å |
| ESP | `esp/step_294000.pt` | 294000 | 15 | 4 Å |

ED SHA-256: `bc34cb98f635034ad67fe0c06a68f97dc881e0ee0605633dbcdf2d200a4f8346`

ESP SHA-256: `efbb831b17c3841afa0c587d8d5004e8d9d9b441d57e60caca5cbac6ef89114a`

Both hashes match the original evaluation JSON records. Do not substitute another experiment's best/latest weights or checkpoints trained with a 6/8 Å cutoff. Runtime paths come from `configs/reproduce.json` and command-line arguments.
