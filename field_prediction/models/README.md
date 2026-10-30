# Model source

`charge3net/` contains the ChargE3Net model and graph-construction code used in the archived ED/ESP runs. Install weights separately under `checkpoints/`. Upstream `src.*` import names and the original `LICENSE` are retained. Entry points configure local imports automatically, so cloning a model repository at runtime is unnecessary.

Both models use `E3DensityModel(num_interactions=3, num_neighbors=20, mul=500, lmax=4, cutoff=4.0, basis="gaussian", num_basis=20)`. ED and ESP use separate checkpoints.

Only the required source and import dependencies are included; upstream logs, training data, other checkpoints, and example experiments were not copied.
