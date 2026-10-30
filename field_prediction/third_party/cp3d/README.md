# cp3d-mof 1.1.0

`cp3d-mof` provides CP3D reading, compression, and decompression for Python/NumPy/PyTorch. The original **CP3D v1.0.0 NPZ format** is the default stable write format. Experimental C++ CP3D7V2 files currently have read-only support.

## Compatibility

| Format | Read | Write | Notes |
| --- | ---: | ---: | --- |
| CP3D v1 single file `*.cube.cp3d` | Yes | Yes | Bidirectionally compatible with v1.0.0 |
| CP3D v1 directory archive `*.cp3d.zip` | Yes | Yes | Preserves relative paths and `manifest.json` |
| C++ CP3D7V2 | Yes | No | Experimental read-only support |

Previously verified:

- A 225³ file produced by the original v1.0.0 tools can be read by this package.
- v1 files produced by this package can be read by the original v1.0.0 decoder.
- NumPy and Torch outputs agree pointwise.
- v1/v2 fixtures, single-file round trips, and multiworker directory-archive tests pass.

## Installation

Install a wheel if available:

```bash
python3 -m pip install cp3d_mof-1.1.0-py3-none-any.whl
```

Install from a source package:

```bash
python3 -m pip install .
```

NumPy and Numba are required. Torch is optional:

```bash
python3 -m pip install "cp3d-mof[torch]"
```

## Load into memory

NumPy:

```python
import cp3d

array = cp3d.load("sample.cube.cp3d")
print(array.shape, array.dtype)  # (nx, ny, nz), float32
```

Load metadata alongside the array:

```python
array, metadata = cp3d.load(
    "sample.cube.cp3d",
    return_metadata=True,
)
```

Torch:

```python
tensor = cp3d.load(
    "sample.cube.cp3d",
    backend="torch",
)
```

CPU Torch tensors are created with `torch.from_numpy()`. To select a GPU:

```python
tensor = cp3d.load(
    "sample.cube.cp3d",
    backend="torch",
    device="cuda",
)
```

You can also specify `dtype=torch.float64`. Without PyTorch installed, an explanatory optional-dependency error is raised only when the Torch backend is requested.

## Lazy loading of directory archives

```python
dataset = cp3d.open_dataset("MOF_data.cp3d.zip")

print(len(dataset))
print(dataset.members)

first = dataset.load(dataset.members[0])
all_arrays = dataset.load_all(workers=2)
```

Alternatively:

```python
dataset = cp3d.load("MOF_data.cp3d.zip")
array = cp3d.load(
    "MOF_data.cp3d.zip",
    member=dataset.members[0],
)
```

The archive is not decoded all at once; only members requested through `load()` are loaded into memory.

## Save from memory

```python
cp3d.save(
    array,
    "sample.cube.cp3d",
    metadata=metadata,
    tolerance=1e-4,
    quantity="electrostatic_potential",
    unit="Hartree",
)
```

CPU/GPU Torch tensors are also accepted and converted to CPU NumPy arrays before saving.

## Compression and decompression

Python API:

```python
cp3d.compress(
    "sample-v_hartree.cube",
    tolerance=1e-4,
)

cp3d.decompress(
    "sample-v_hartree.cube.cp3d",
    "restored.cube",
)
```

Directories:

```python
cp3d.compress(
    "external/abcd",
    workers=2,
)
# external/abcd.cp3d.zip

cp3d.decompress(
    "external/abcd.cp3d.zip",
    workers=2,
)
```

The CLI retains the convenient `-r` form:

```bash
cp3d compress -r external/abcd -j 2
cp3d decompress -r external/abcd.cp3d.zip -j 2
```

Single-file commands:

```bash
cp3d compress sample.cube -t 0.001
cp3d decompress sample.cube.cp3d
cp3d info sample.cube.cp3d
```

## Multithreading

`workers` / `-j` controls concurrent processing of multiple Cube files or archive members. Automatic selection uses at most two concurrent tasks to limit memory use from large 3D grids. Increase this explicitly if memory permits.

The v1 three-dimensional Lorenzo closed-loop stream has strict forward dependencies, so encoding/decoding within one file remains sequential to preserve full v1.0.0 compatibility. Parallel work mainly covers:

- Compressing multiple Cube files.
- Decoding multiple archive members.
- Writing multiple output files.
- Block INFLATE and DCT reconstruction for experimental v2 files.

For grids of 225³ or larger, two concurrent tasks may require hundreds of MB of temporary memory; more threads are not necessarily faster.

## Reference performance

Mg-MOF-74, 225³ grid, warm runs on a local SSD:

- v1 read into NumPy: median `0.132 s`, approximately `0.345 GB/s` measured against the float32 grid size.
- Raw Cube reading, ASCII parsing, and v1 compression: median `1.757 s`, approximately `0.085 GB/s` measured against the original text size.
- Output at `0.001 Ha`: `3,168,058 bytes`.

The first call triggers Numba compilation and is slower than later calls. Directory-level parallel performance also depends on memory, storage, and ZIP/Deflate implementation details.

## Accuracy

CP3D v1 is lossy. Tolerances are defined relative to the input after parsing into float32; float32 rounding may make the actual maximum error slightly exceed the target. Writing a decompressed text Cube introduces additional scientific-notation rounding.

Validate an acceptable tolerance for the data type before scientific use:

| Data type | Original v1 default |
| --- | ---: |
| Hartree potential | `1e-4` Hartree |
| Electron density | `1e-6` electron/Bohr³ |
| Spin density | `1e-6` electron/Bohr³ |
