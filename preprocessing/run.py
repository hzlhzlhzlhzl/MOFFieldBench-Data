#!/usr/bin/env python3
"""Portable command entry point; paths in configs are relative to this package."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

def main():
    commands = {"pipeline": "mof_pipeline.pipeline", "fps": "mof_pipeline.fps",
                "fourier": "mof_pipeline.fourier", "manifest": "mof_pipeline.manifest"}
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python run.py {pipeline|fps|fourier|manifest|train} [options]")
        return
    command = sys.argv.pop(1)
    if command == "train":
        os.chdir(ROOT)
        runpy.run_path(str(ROOT / "model/train.py"), run_name="__main__")
    elif command in commands:
        runpy.run_module(commands[command], run_name="__main__")
    else:
        raise SystemExit(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
