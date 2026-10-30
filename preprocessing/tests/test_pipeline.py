from pathlib import Path
import csv
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
import cp3d

from mof_pipeline.fourier import official_compatible_interpolate_fourier
from mof_pipeline.fps import progressive_fps, read_ply
from mof_pipeline.pipeline import ROOT, work
from mofreg.data import MOFPointCloudDataset, SampleRecord, fourier_encode_frac
from mofreg.models import build_model


@pytest.mark.parametrize("shape,target", [((9, 11, 13), (5, 7, 9)), ((8, 10, 12), (4, 6, 8)), ((3, 5, 7), (1, 1, 1))])
def test_fourier_preserves_constant_and_mean(shape, target):
    field = np.full(shape, 0.125)
    np.testing.assert_allclose(official_compatible_interpolate_fourier(field, target), 0.125)
    rng = np.random.default_rng(3)
    field += rng.normal(size=shape)
    result = official_compatible_interpolate_fourier(field, target)
    np.testing.assert_allclose(result.mean(), field.mean(), atol=1e-12)


def test_fourier_retains_resolvable_mode():
    shape, target = (15, 15, 15), (9, 9, 9)
    x = np.arange(shape[0]) / shape[0]
    field = np.broadcast_to(np.sin(2*np.pi*x)[:, None, None], shape)
    result = official_compatible_interpolate_fourier(field, target)
    expected = np.broadcast_to(np.sin(2*np.pi*np.arange(9)/9)[:, None, None], target)
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("method", ["periodic", "nonperiodic"])
def test_fps_duplicates_never_reselect_same_row(method):
    cart = np.zeros((5, 3), np.float32)
    result = progressive_fps(cart, cart, np.eye(3), [3, 8], method, "images27", torch.device("cpu"), 2, 10)
    assert len(set(result.indices)) == 5


def make_cp3d(folder, kind, offset=0.0):
    folder.mkdir(parents=True, exist_ok=True)
    shape = (12, 12, 12)
    i, j, k = np.indices(shape)
    field = 0.3 + 0.12*np.cos(i*2*np.pi/12) + 0.08*np.sin(j*2*np.pi/12) + 0.02*k
    if kind == "v_hartree":
        field = field - 0.45
    cube = folder / f"{folder.name}-{kind}-1_0.cube"
    with cube.open("w") as f:
        f.write(f"synthetic\n{kind}\n1 {offset} 0 0\n12 .2 0 0\n12 .02 .2 0\n12 0 .01 .2\n1 0 0 0 0\n")
        np.savetxt(f, field.reshape(-1, 6), fmt="%.8e")
    cp3d.compress(cube, cube.with_suffix(".cube.cp3d"), tolerance=1e-6)


def config_for(tmp_path):
    config = json.loads((ROOT / "configs/pipeline.json").read_text())
    config.update(input_roots=[str(tmp_path / "raw")], point_root=str(tmp_path / "points"),
        ed_output_root=str(tmp_path / "ED_fps"), esp_output_root=str(tmp_path / "EPS_fps"),
        density_thresholds=[0.03], esp_tail_percentages=[5], surface_esp=True,
        sizes=[16, 32, 128], downsample=2, device="cpu", methods=["nonperiodic", "periodic"])
    return config


def test_end_to_end_and_model(tmp_path):
    name = "mof_synthetic"
    for kind in ["ELECTRON_DENSITY", "v_hartree"]:
        make_cp3d(tmp_path / "raw" / name, kind)
    config = config_for(tmp_path)
    result = work(name, config, "all", False)
    assert result["status"] == "complete", result
    out = tmp_path / "EPS_fps/non_periodic/esp_highest_lowest_5pct" / name
    ply16 = out / "nonperiodic_fps/esp_highest_lowest_5pct_nonperiodic_fps_16.ply"
    ply32 = ply16.with_name(ply16.name.replace("_16.ply", "_32.ply"))
    ply128 = ply16.with_name(ply16.name.replace("_16.ply", "_128.ply"))
    small, large = read_ply(ply16), read_ply(ply32)
    np.testing.assert_array_equal(small.vertices, large.vertices[:16])
    assert read_ply(ply128).vertex_count == 128
    report = json.loads((out / "fps_manifest.json").read_text())
    assert report["coordinate_unit"] == "bohr"
    assert report["input_extra"].endswith("esp_lowest_5pct.ply")
    before = ply32.stat().st_mtime_ns
    assert work(name, config, "all", False)["status"] == "complete"
    assert ply32.stat().st_mtime_ns == before
    changed = dict(config, downsample=3)
    assert work(name, changed, "all", False)["status"] == "failed"
    dataset = MOFPointCloudDataset([SampleRecord(name, ply32, 3.2)],
        feature_blocks=["xyz", "esp", "frac_pe"], augment=False, normalize_xyz=True, frac_pe_frequencies=2)
    sample = dataset[0]
    assert sample["features"].shape == (32, 16)
    model = build_model({"model": {"name": "pointnet", "hidden_dim": 16, "dropout": 0.0}}, input_dim=16)
    prediction = model(sample["features"][None], sample["xyz"][None])
    loss = (prediction - sample["target"])**2
    loss.mean().backward()
    assert torch.isfinite(prediction).all()


def test_reject_misaligned_fields(tmp_path):
    name = "qmof-synthetic"
    make_cp3d(tmp_path / "raw" / name, "ELECTRON_DENSITY")
    make_cp3d(tmp_path / "raw" / name, "v_hartree", offset=1.0)
    report = work(name, config_for(tmp_path), "clouds", False)
    assert report["status"] == "failed"
    assert "differ" in report["error"]
    assert not (tmp_path / "points" / name / "DONE.json").exists()


def test_fractional_encoding_is_periodic():
    frac = np.random.default_rng(1).uniform(size=(20, 3)).astype(np.float32)
    # float32 argument rounding is amplified by the largest 2*pi*4 frequency.
    np.testing.assert_allclose(fourier_encode_frac(frac, 3), fourier_encode_frac(frac+1, 3), atol=8e-6)


def test_inventory_has_only_materials_and_placeholders():
    with (ROOT / "data/manifest.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert "split" not in reader.fieldnames
        assert not any("cif" in k for k in reader.fieldnames)
    assert len(rows) == len({r["sample_id"] for r in rows}) == 40343
    assert sum(r["source"] == "arcmof" for r in rows) == 24908
    assert sum(r["source"] == "qmof" for r in rows) == 15435
    assert {r["data_status"] for r in rows} == {"placeholder"}


def test_public_manifest_preserves_source_and_folder(tmp_path):
    from mof_pipeline.manifest import build_manifest
    with (ROOT / "data/manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [next(r for r in rows if r["source"] == source) for source in ("arcmof", "qmof")]
    source = tmp_path / "ids.csv"
    source.write_text("id\n" + "\n".join(r["sample_id"] for r in selected) + "\n")
    output = tmp_path / "generated"
    report = build_manifest(source, output)
    assert report["source_counts"] == {"arcmof": 1, "qmof": 1}
    with (output / "manifest.csv").open(newline="") as handle:
        actual = list(csv.DictReader(handle))
    for before, after in zip(selected, actual):
        assert after["sample_id"] == after["material"] == before["sample_id"]
        assert after["source"] == before["source"]
        assert f'/{after["sample_id"]}/' in after["density_cp3d_glob"]


def test_public_manifest_unknown_id_requires_source(tmp_path):
    from mof_pipeline.manifest import read_samples
    source = tmp_path / "ids.csv"
    source.write_text("id\nmof_999999999\n")
    with pytest.raises(ValueError, match="Supply source"):
        read_samples(source)
    source.write_text("id,source\nmof_999999999,qmof\n")
    assert read_samples(source) == [{"sample_id": "mof_999999999", "source": "qmof"}]
