"""Commands for the packaged paper experiment."""
import argparse
import csv
import hashlib
import importlib
import json
import math
import shutil
import sys
from pathlib import Path

from . import ROOT
from .manifest import check_data, load_manifest


def config():
    return json.loads((ROOT / "configs/reproduce.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_checkpoint(target):
    entry = config()["models"][target]
    path = ROOT / entry["checkpoint"]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return path


def invoke(module, arguments):
    previous = sys.argv
    try:
        sys.argv = [module, *map(str, arguments)]
        result = importlib.import_module(f"mof_fields.{module}").main()
        if result:
            raise RuntimeError(f"{module} failed; inspect the failure report")
    finally:
        sys.argv = previous


def verify(args):
    cfg = config()
    inventory = json.loads((ROOT / "reference/provenance/integrity.json").read_text())
    for relative, expected in inventory.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Package file changed: {relative}")
    split = json.loads((ROOT / cfg["split"]).read_text())
    names = [name for key in ("train", "validation", "test") for name in split[key]]
    manifest_names = {r["material"] for r in load_manifest(ROOT / cfg["manifest"])["records"]}
    if len(names) != len(set(names)) or set(names) != manifest_names:
        raise ValueError("Split overlap or mismatch with manifest")
    if len(split["test"]) != cfg["test_mofs"]:
        raise ValueError("Unexpected test-set size")
    for target in cfg["models"]:
        if not (ROOT / cfg["models"][target]["checkpoint"]).is_file() and not args.load_models:
            print(f"{target}: external checkpoint not installed (source-only verification)")
            continue
        path = check_checkpoint(target)
        if args.load_models:
            import torch
            from .predict import load_model
            entry = cfg["models"][target]
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if (checkpoint["step"], checkpoint["epoch"], checkpoint["training_args"]["cutoff"]) != (
                entry["step"], entry["epoch"], entry["cutoff"]
            ):
                raise ValueError(f"Checkpoint metadata mismatch: {target}")
            model = load_model(path, torch.device("cpu"), entry["cutoff"])
            print(f"{target}: loaded strictly; {sum(p.numel() for p in model.parameters()):,} parameters")
            del model, checkpoint
    if args.data:
        check_data(args.manifest or ROOT / cfg["manifest"], split["test"], ("ed", "esp"))
    print(f"Verified {len(inventory)} files; splits: " + ", ".join(
        f"{key}={len(split[key])}" for key in ("train", "validation", "test")))
    return 0


def evaluate(args):
    cfg = config()
    manifest = args.manifest or ROOT / cfg["manifest"]
    split_path = ROOT / cfg["split"]
    names = json.loads(split_path.read_text())["test"]
    targets = ("ed", "esp") if args.target == "both" else (args.target,)
    shards = (0, 1) if args.shard_index is None else (args.shard_index,)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    selected = [n for shard in shards for n in names[shard::2][:args.limit]]
    check_data(manifest, selected, targets)
    output = args.output_dir or ROOT / "outputs" / ("smoke" if args.limit else "evaluation")
    for target in targets:
        checkpoint = check_checkpoint(target)
        for shard in shards:
            arguments = ["--target", target, "--checkpoint", checkpoint,
                         "--manifest", manifest, "--split-file", split_path,
                         "--output", output / f"{target}_shard{shard}.json",
                         "--device", args.device, "--cutoff", cfg["models"][target]["cutoff"],
                         "--probes", cfg["probes"], "--distance-candidates", cfg["distance_candidates"],
                         "--inference-chunk", cfg["inference_chunk"], "--seed", cfg["seed"],
                         "--num-shards", cfg["num_shards"], "--shard-index", shard]
            if args.limit:
                arguments += ["--limit", args.limit]
            invoke("evaluate", arguments)
    print(f"Evaluation saved to {output}")
    return 0


def prepare_data(args):
    cfg = config()
    split = json.loads((ROOT / cfg["split"]).read_text())
    names = split[args.split] if args.split != "all" else sum((split[k] for k in ("train", "validation", "test")), [])
    source = check_data(args.source_manifest, names, ("ed", "esp"))
    destination = {r["material"]: r for r in load_manifest(ROOT / cfg["manifest"])["records"]}
    # Check every destination before changing anything; never overwrite data.
    pairs = []
    for name in names:
        for key in ("cif", "ed", "esp"):
            src, dst = Path(source[name][key]), Path(destination[name][key])
            if dst.exists() or dst.is_symlink():
                if dst.exists() and dst.samefile(src):
                    continue
                raise FileExistsError(f"Destination already exists: {dst}")
            pairs.append((src, dst))
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "symlink":
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)
    print(f"Prepared {len(pairs)} files using {args.mode}; split={args.split}")
    return 0


def compare(args):
    """Compare recomputed per-material regional scores with the archived experiment."""
    reference = {}
    for target in ("ed", "esp"):
        for path in (ROOT / "reference/metrics").glob(f"{target}_shard*_per_material.csv"):
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    reference[(target, row["material"], row["region"])] = row
    checked, maximum, failures = 0, 0.0, []
    files = sorted(args.metrics_dir.glob("*_shard*_per_material.csv"))
    if not files:
        raise ValueError("No recomputed per-material CSV files found")
    seen = set()
    for path in files:
        target = path.name.split("_")[0]
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = (target, row["material"], row["region"])
                if key in seen:
                    raise ValueError(f"Duplicate result: {key}")
                seen.add(key)
                expected = reference[key]
                if int(row["points"]) != int(expected["points"]):
                    failures.append({"key": key, "reason": "point-count mismatch"})
                    continue
                if not int(row["points"]):
                    continue
                error = abs(float(row["normalized_mae_percent"]) - float(expected["normalized_mae_percent"]))
                checked += 1
                maximum = max(maximum, error)
                if not math.isfinite(error) or error > args.atol:
                    failures.append({"key": key, "difference_percentage_points": error})
    report = {"checked_material_regions": checked, "maximum_difference_percentage_points": maximum,
              "absolute_tolerance_percentage_points": args.atol, "failures": failures,
              "scope": "Only rows in the supplied directory are compared; this may be a smoke subset."}
    (args.metrics_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if failures or not checked else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify", help="Check hashes, split integrity, and optionally model loading")
    p.add_argument("--load-models", action="store_true")
    p.add_argument("--data", action="store_true")
    p.add_argument("--manifest", type=Path)
    p = sub.add_parser("plot", help="Draw both figures from archived or recomputed per-MOF metrics")
    p.add_argument("--metrics-dir", type=Path, default=ROOT / "reference/metrics")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    p = sub.add_parser("evaluate", help="Evaluate the fixed paper experiment from checkpoints")
    p.add_argument("--target", choices=("ed", "esp", "both"), default="both")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--shard-index", type=int, choices=(0, 1))
    p.add_argument("--limit", type=int, help="MOFs per shard; saves to outputs/smoke by default")
    p.add_argument("--output-dir", type=Path)
    p = sub.add_parser("prepare-data", help="Populate the canonical data layout from another manifest")
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    p.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    p = sub.add_parser("compare", help="Compare recomputed scores against archived per-MOF scores")
    p.add_argument("--metrics-dir", type=Path, default=ROOT / "outputs/smoke")
    p.add_argument("--atol", type=float, default=0.001, help="Absolute tolerance in percentage points")
    for command in ("train", "predict"):
        p = sub.add_parser(command, help=f"{command.capitalize()}; extra arguments go to the underlying module", add_help=False)
        p.add_argument("--target", choices=("ed", "esp"), default="ed")
        p.add_argument("--manifest", type=Path)
        p.add_argument("--device", default="auto")
    args, extra = parser.parse_known_args()
    if args.command not in ("train", "predict") and extra:
        parser.error(f"Unrecognized arguments: {' '.join(extra)}")
    if args.command == "plot":
        from .plotting import plot_all
        return plot_all(args.metrics_dir, args.output_dir)
    if args.command in ("train", "predict"):
        cfg = config()
        flags = ["--target", args.target, "--paired-manifest", args.manifest or ROOT / cfg["manifest"],
                 "--device", args.device, "--cutoff", cfg["models"][args.target]["cutoff"]]
        if args.command == "train":
            flags += ["--split-file", ROOT / cfg["split"], "--run-dir", ROOT / "outputs/training" / args.target]
        else:
            flags += ["--checkpoint", check_checkpoint(args.target), "--output-root", ROOT / "outputs/predictions" / args.target]
        invoke(args.command, [*flags, *extra])
        return 0
    return {"verify": verify, "evaluate": evaluate, "prepare-data": prepare_data, "compare": compare}[args.command](args)
