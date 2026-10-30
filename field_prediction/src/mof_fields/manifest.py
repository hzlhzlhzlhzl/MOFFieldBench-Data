"""Manifest paths are relative to the manifest, independent of the current directory."""
import json
import os
from pathlib import Path


def load_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    seen = set()
    for row in payload["records"]:
        name = row["material"]
        if name in seen:
            raise ValueError(f"Duplicate material: {name}")
        seen.add(name)
        for key in ("cif", "ed", "esp"):
            value = Path(row[key])
            # Normalize lexically: resolving 120,000 nonexistent placeholder paths
            # would perform hundreds of thousands of slow network-filesystem stats.
            row[key] = str(value) if value.is_absolute() else os.path.abspath(path.parent / value)
    return payload


def check_data(manifest, names, targets):
    by_name = {r["material"]: r for r in load_manifest(manifest)["records"]}
    missing = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"Manifest does not contain {name}")
        for key in ("cif", *targets):
            path = Path(by_name[name][key])
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} data files are missing. Populate data/raw or pass --manifest. "
            f"First missing files: {missing[:3]}"
        )
    return by_name
