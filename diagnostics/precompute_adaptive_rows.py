"""Build immutable physics-only adaptive row stores; never submit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import stim

from adaptive_row_store import precompute_store


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices=("X", "Z", "both"), default="both")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for basis in "XZ" if args.basis == "both" else args.basis:
        circuit = stim.Circuit.from_file(str(ROOT/f"data/competition/{basis}_circuit.stim"))
        manifest = precompute_store(circuit, ROOT/f"data/target03/adaptive_rows_{basis}",
            threads=args.threads, compression_level=args.compression_level, resume=args.resume,
            progress=lambda value: print(basis, json.dumps(value), flush=True))
        print(basis, "STORE_COMPLETE", manifest["metadata"]["compressed_bytes"], flush=True)


if __name__ == "__main__":
    main()
