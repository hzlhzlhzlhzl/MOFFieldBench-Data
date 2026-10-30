"""Create a payload-free ARC-MOF/QMOF inventory; no training split columns."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]


def read_samples(path):
    with (ROOT / "data/manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        known_sources = {r["sample_id"]: r["source"] for r in csv.DictReader(handle)}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        key = "sample_id" if "sample_id" in (reader.fieldnames or []) else "id"
        if key not in (reader.fieldnames or []):
            raise ValueError("Input must contain sample_id or id")
        rows = [{"sample_id": row[key], "source": row.get("source") or known_sources.get(row[key])} for row in reader]
    seen = set()
    for row in rows:
        name = row["sample_id"]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid material ID: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate material ID: {name}")
        if not re.fullmatch(r"mof_[1-9][0-9]*", name):
            raise ValueError(f"Expected a public material ID: {name}")
        if row["source"] not in {"arcmof", "qmof"}:
            raise ValueError(f"Supply source=arcmof or qmof for material: {name}")
        seen.add(name)
    if not rows:
        raise ValueError("Empty material inventory")
    return rows


def material_name(sample_id):
    return sample_id


def build_manifest(source, output_dir):
    rows = read_samples(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "source", "material", "density_cp3d_glob", "esp_cp3d_glob", "ed_path", "esp_path", "data_status"]
    ids = {"arcmof": [], "qmof": []}
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            name = material_name(row["sample_id"])
            source_name = row["source"]
            ids[source_name].append(row["sample_id"])
            writer.writerow(dict(sample_id=row["sample_id"], source=source_name, material=name,
                density_cp3d_glob=f"data/{source_name}/cp2k_cube/{name}/*ELECTRON_DENSITY*.cp3d",
                esp_cp3d_glob=f"data/{source_name}/cp2k_cube/{name}/*v_hartree*.cp3d",
                ed_path=f"data/ED_fps/non_periodic/density_0.03/{name}/nonperiodic_fps/density_gt_0.03_nonperiodic_fps_6144.ply",
                esp_path=f"data/EPS_fps/non_periodic/esp_highest_lowest_5pct/{name}/nonperiodic_fps/esp_highest_lowest_5pct_nonperiodic_fps_6144.ply",
                data_status="placeholder"))
    for name, values in ids.items():
        (output_dir / name / "cp2k_cube").mkdir(parents=True, exist_ok=True)
        (output_dir / name / "materials.txt").write_text("\n".join(values) + "\n")
        (output_dir / name / "cp2k_cube/.gitkeep").touch()
    report = dict(rows=len(rows), unique_ids=len(rows), source_counts={k: len(v) for k, v in ids.items()},
        source_file=str(Path(source).resolve()), source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        columns_retained_from_source=["id"],
        payload_status="placeholder; no raw fields, PLYs, labels, splits or CIFs copied",
        path_base="package root; glob columns describe discovery patterns")
    (output_dir / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True, help="CSV containing id or sample_id and optional source; unknown IDs require source")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.source_csv.resolve() == (args.output_dir / "manifest.csv").resolve():
        parser.error("Source and output manifest must differ")
    print(json.dumps(build_manifest(args.source_csv, args.output_dir), indent=2))

if __name__ == "__main__":
    main()
