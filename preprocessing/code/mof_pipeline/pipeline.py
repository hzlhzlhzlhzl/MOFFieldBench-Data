"""CP3D -> Fourier resampling -> field selection -> nested FPS PLYs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

from .manifest import ROOT, read_samples, material_name

def resolve(path):
    path = Path(os.path.expandvars(str(path))).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)

def signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def file_info(path):
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)

def locate_inputs(name, config):
    # Require one unambiguous material folder. Never pair fields across roots.
    matches = []
    for root in config["input_roots"]:
        for candidate in dict.fromkeys([name, material_name(name)]):
            path = resolve(root) / candidate
            if path.is_dir() and path not in matches:
                matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"{name}: expected one input directory, found {len(matches)}: {matches}")
    folder = matches[0]
    result = {}
    for kind, pattern in [("ed", "*ELECTRON_DENSITY*.cp3d"), ("esp", "*v_hartree*.cp3d")]:
        paths = sorted(folder.glob(pattern))
        needed = (kind == "ed" and (config["density_thresholds"] or config["surface_esp"])) or (kind == "esp" and (config["esp_tail_percentages"] or config["surface_esp"]))
        if needed and len(paths) != 1:
            raise ValueError(f"{folder}: expected one {pattern}, found {len(paths)}")
        if needed:
            result[kind] = paths[0]
    return folder.name, result

def cloud_signature(inputs, config):
    return signature({"inputs": {k: file_info(v) for k, v in inputs.items()},
                      "parameters": {k: config[k] for k in ("density_thresholds", "esp_tail_percentages", "surface_esp", "downsample")}})

def prepare_clouds(name, inputs, config, force):
    import numpy as np
    import cp3d
    from . import pointclouds as pc
    from .fourier import official_compatible_interpolate_fourier
    out = resolve(config["point_root"]) / name
    done = out / "DONE.json"
    key = cloud_signature(inputs, config)
    if done.exists() and not force:
        old = json.loads(done.read_text())
        if old.get("signature") != key:
            raise ValueError(f"{done}: inputs/config changed; use --force to rebuild")
        if all((out / f).is_file() for f in old["output_files"]):
            return old
    # An interrupted rebuild must not leave a valid completion marker.
    done.unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fields, reference = {}, None
    for kind, path in inputs.items():
        values, metadata = cp3d.load(path, return_metadata=True)
        shape = tuple(values.shape)
        origin, lattice = pc.geometry(metadata, shape)
        signed = metadata.get("signed_counts", metadata.get("counts", []))
        unit = "bohr"
        header = metadata.get("cube_header_text")
        if header:
            signed = [int(line.split()[0]) for line in header.splitlines()[3:6]]
        if signed:
            if all(x < 0 for x in signed):
                unit = "angstrom"
            elif not all(x > 0 for x in signed):
                raise ValueError("Mixed CUBE coordinate units are unsupported")
        # CP3D metadata from the production corpus uses signed_shape.
        signed_shape = metadata.get("signed_shape")
        if signed_shape is not None:
            if all(x < 0 for x in signed_shape):
                unit = "angstrom"
            elif not all(x > 0 for x in signed_shape):
                raise ValueError("Mixed CUBE coordinate units are unsupported")
        if reference is None:
            reference = shape, origin, lattice, unit
        elif shape != reference[0] or unit != reference[3] or not np.allclose(origin, reference[1]) or not np.allclose(lattice, reference[2]):
            raise ValueError(f"{name}: ED and ESP grids/origins/lattices/units differ")
        ds = config["downsample"]
        factor = pc.factor_for_points(math.prod(shape)) if ds == "auto" else float(ds)
        target = pc.target_shape(shape, factor)
        if target != shape:
            values = official_compatible_interpolate_fourier(values, target)
        fields[kind] = np.asarray(values, dtype=np.float32)
        if not np.isfinite(fields[kind]).all():
            raise ValueError(f"{name}: non-finite {kind} field")
    shape, origin, lattice, unit = reference
    outputs = {"density": [], "esp_tails": [], "surface_esp": []}
    if config["density_thresholds"]:
        outputs["density"] = pc.write_density_clouds(out, fields["ed"], origin, lattice, tuple(config["density_thresholds"]))
    if config["esp_tail_percentages"]:
        outputs["esp_tails"] = pc.write_tail_clouds(out, fields["esp"], origin, lattice, tuple(v / 100 for v in config["esp_tail_percentages"]))
    if config["surface_esp"]:
        outputs["surface_esp"] = pc.write_surface_clouds(out, name, fields["ed"], fields["esp"], origin, lattice)
    payload = dict(material=name, status="complete", signature=key,
        inputs={k: str(v) for k, v in inputs.items()}, original_shape=shape,
        original_grid_points=math.prod(shape), downsample_factor_per_axis=factor,
        downsampled_shape=target, downsampled_grid_points=math.prod(target),
        fourier_backend="bundled pyRho-compatible" if target != shape else "not applied",
        coordinate_unit=unit, origin=origin.tolist(), lattice_vectors_rows=lattice.tolist(),
        esp_source_missing="esp" not in inputs, **outputs,
        output_files=[Path(item["file"]).name for group in outputs.values() for item in group],
        elapsed_seconds=time.monotonic()-started)
    atomic_json(done, payload)
    return payload

def sample_clouds(name, config, force):
    point = resolve(config["point_root"]) / name
    done = point / "DONE.json"
    if not done.is_file():
        raise FileNotFoundError(f"Generate clouds first: {done}")
    jobs = []
    for threshold in config["density_thresholds"]:
        jobs.append(("ed", f"density_{threshold:g}", f"density_gt_{threshold:g}", [point / f"density_gt_{threshold:g}.ply"]))
    for pct in config["esp_tail_percentages"]:
        stem = f"esp_highest_lowest_{pct}pct"
        # Preserve production concatenation order: high tail first, then low tail.
        jobs.append(("esp", stem, stem, [point / f"esp_highest_{pct}pct.ply", point / f"esp_lowest_{pct}pct.ply"]))
    for kind, variant, stem, paths in jobs:
        for method in config["methods"]:
            method_dir = "non_periodic" if method == "nonperiodic" else "periodic"
            out = resolve(config[f"{kind}_output_root"]) / method_dir / variant / name
            key = signature(dict(inputs=[file_info(p) for p in paths], metadata=file_info(done),
                method=method, sizes=config["sizes"], seed=config["seed"], device=config["device"],
                periodic_mode=config["periodic_mode"], size_policy=config["size_policy"]))
            marker = out / "pipeline_state.json"
            expected = [out / f"{method}_fps/{stem}_{method}_fps_{size}.ply" for size in config["sizes"]]
            expected += [out / "fps_manifest.json", out / "fps_metrics.csv", out / f"{method}_fps/{stem}_{method}_fps_order_{max(config['sizes'])}.npy"]
            if marker.exists() and not force:
                if json.loads(marker.read_text()).get("signature") != key:
                    raise ValueError(f"{out}: parameters changed; use --force")
                if all(p.is_file() for p in expected):
                    continue
            marker.unlink(missing_ok=True)
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / "run.py"), "fps", "--input", str(paths[0]),
                "--output-dir", str(out), "--output-stem", stem, "--methods", method,
                "--sizes", ",".join(map(str, config["sizes"])), "--seed", str(config["seed"]),
                "--device", config["device"], "--periodic-mode", config["periodic_mode"],
                "--size-policy", config["size_policy"], "--progress-every", "2048", "--overwrite"]
            if len(paths) == 2:
                cmd += ["--input-extra", str(paths[1])]
            with (out / "fps.log").open("w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"FPS failed: {out / 'fps.log'}")
            atomic_json(marker, dict(signature=key))

def work(name, config, stage, force):
    try:
        if stage in {"all", "clouds"}:
            name, inputs = locate_inputs(name, config)
            prepare_clouds(name, inputs, config, force)
        else:
            options = [p for p in dict.fromkeys([name, material_name(name)]) if (resolve(config["point_root"]) / p).is_dir()]
            if len(options) != 1:
                raise ValueError(f"{name}: expected one intermediate point directory, found {options}")
            name = options[0]
        if stage in {"all", "fps"}:
            sample_clouds(name, config, force)
        return dict(material=name, status="complete")
    except Exception as exc:
        return dict(material=name, status="failed", error=f"{type(exc).__name__}: {exc}")

def validate_config(c):
    if not c["input_roots"] or not c["sizes"] or any(type(x) is not int or x <= 0 for x in c["sizes"]):
        raise ValueError("input_roots and positive integer sizes are required")
    if not c["methods"] or set(c["methods"]) - {"periodic", "nonperiodic"}:
        raise ValueError("methods must be periodic/nonperiodic")
    if any(type(v) is not int or not 0 < v < 50 for v in c["esp_tail_percentages"]):
        raise ValueError("ESP tail percentages must be integers between 1 and 49")
    if any(not math.isfinite(t) or t < 0 for t in c["density_thresholds"]):
        raise ValueError("Density thresholds must be finite and nonnegative")
    if not (c["density_thresholds"] or c["esp_tail_percentages"] or c["surface_esp"]):
        raise ValueError("Select at least one field output")
    if c["downsample"] != "auto" and (not math.isfinite(float(c["downsample"])) or float(c["downsample"]) < 1):
        raise ValueError("downsample must be auto or a finite factor >= 1")
    if c["periodic_mode"] not in {"validated8", "images27", "wrapped"} or c["size_policy"] not in {"clip", "error"}:
        raise ValueError("Invalid FPS mode/size policy")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    p.add_argument("--materials", nargs="+")
    p.add_argument("--stage", choices=["all", "clouds", "fps"], default="all")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.workers < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        p.error("workers, threads and limit must be positive")
    config = json.loads(args.config.read_text())
    validate_config(config)
    names = args.materials or [row["sample_id"] for row in read_samples(resolve(config["manifest_csv"]))]
    names = list(dict.fromkeys(names))
    for name in names:
        if Path(name).name != name or name in {".", ".."}:
            p.error(f"Invalid material name: {name}")
    if args.limit:
        names = names[:args.limit]
    if args.dry_run:
        print(json.dumps(dict(material_count=len(names), first_materials=names[:5], stage=args.stage,
            input_roots=[str(resolve(r)) for r in config["input_roots"]], config=config), indent=2))
        return
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    reports = []
    if args.workers == 1:
        iterator = (work(name, config, args.stage, args.force) for name in names)
        for report in iterator:
            reports.append(report)
            print(json.dumps(report), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, name, config, args.stage, args.force) for name in names]
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                print(json.dumps(report), flush=True)
    failed = [r for r in reports if r["status"] == "failed"]
    atomic_json(resolve(config["point_root"]) / "RUN_REPORT.json", dict(total=len(names), failed=len(failed), results=reports))
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
