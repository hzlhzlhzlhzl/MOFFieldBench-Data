"""Command-line interface for the cp3d-mof package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import compress, decompress, load


def main() -> None:
    parser = argparse.ArgumentParser(prog="cp3d")
    commands = parser.add_subparsers(dest="command", required=True)
    encoder = commands.add_parser("compress", help="compress a Cube or directory")
    encoder.add_argument("input", type=Path, nargs="?")
    encoder.add_argument("-r", "--root", type=Path)
    encoder.add_argument("-o", "--output", type=Path)
    encoder.add_argument("-t", "--tolerance", type=float)
    encoder.add_argument("-j", "--workers", type=int, default=0)
    encoder.add_argument("--quantity")
    encoder.add_argument("--no-verify", action="store_true")
    decoder = commands.add_parser("decompress", help="restore a CP3D file or bundle")
    decoder.add_argument("input", type=Path, nargs="?")
    decoder.add_argument("-r", "--root", type=Path)
    decoder.add_argument("-o", "--output", type=Path)
    decoder.add_argument("-j", "--workers", type=int, default=0)
    info = commands.add_parser("info", help="show CP3D metadata")
    info.add_argument("input", type=Path)
    info.add_argument("--member")
    args = parser.parse_args()
    if args.command == "compress":
        source = args.root or args.input
        if source is None:
            parser.error("compress requires INPUT or -r ROOT")
        result = compress(
            source,
            args.output,
            tolerance=args.tolerance,
            quantity=args.quantity,
            verify=not args.no_verify,
            workers=args.workers,
        )
    elif args.command == "decompress":
        source = args.root or args.input
        if source is None:
            parser.error("decompress requires INPUT or -r ROOT")
        result = decompress(source, args.output, workers=args.workers)
    else:
        loaded = load(
            args.input, member=args.member, return_metadata=True
        )
        if hasattr(loaded, "members"):
            result = {
                "archive": str(args.input.resolve()),
                "members": list(loaded.members),
                "manifest": loaded.manifest,
            }
        else:
            values, metadata = loaded
            result = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "metadata": metadata,
            }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
