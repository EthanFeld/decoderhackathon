"""Materialize exact all-order temporal block probability/bit-bias lookup tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.temporal_factor_probe import factor_queries


SCHEMA = "temporal-four-block-signed-marginals-v1"


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_region(model, detectors, minimum_probability=1e-12):
    detectors = np.asarray(detectors, np.int64)
    outside = np.ones(len(model.h), bool); outside[detectors] = False
    columns = np.flatnonzero(~model.h[outside].any(axis=0)).astype(np.int64)
    local_h, local_l = model.h[detectors][:, columns], model.logical[:, columns]
    result = factor_queries(local_h, local_l, model.probabilities[columns], max_bits=21)
    probability = np.ascontiguousarray(result["total_mass"])
    bias = np.ascontiguousarray(result["signed_mass"])
    valid = (probability >= minimum_probability) & np.all(np.abs(bias) <= probability[:, None]+1e-14, axis=1)
    decision = ((bias < 0).astype(np.uint8) @ np.asarray([1, 2, 4, 8], np.uint8)).astype(np.uint8)
    decision[~valid] = 0
    all_syndrome_masks = (model.h[detectors].T @ (1 << np.arange(len(detectors)))).astype(np.uint32)
    local_syndrome_masks = np.zeros(model.h.shape[1], np.uint32); local_syndrome_masks[columns] = all_syndrome_masks[columns]
    global_logical_masks = (model.logical.T @ np.asarray([1, 2, 4, 8])).astype(np.uint8)
    local_logical_masks = np.zeros(model.h.shape[1], np.uint8); local_logical_masks[columns] = global_logical_masks[columns]
    arrays = {"detectors": detectors, "columns": columns, "probability": probability, "bias": bias,
              "valid": valid, "decision": decision, "syndrome_masks": all_syndrome_masks,
              "local_syndrome_masks": local_syndrome_masks, "logical_masks": local_logical_masks}
    summary = {"detectors": detectors.tolist(), "mechanisms": len(columns), "states": len(probability),
               "probability_sum": float(probability.sum()), "probability_minimum": float(probability.min()),
               "negative_probability_entries": int((probability < 0).sum()),
               "negative_probability_absolute_mass": float(-probability[probability < 0].sum()),
               "minimum_probability_gate": minimum_probability,
               "valid_query_entries": int(valid.sum()), "invalid_or_tiny_entries": int((~valid).sum()),
               "maximum_abs_bias_excess_over_probability": float(np.maximum(np.abs(bias)-probability[:, None], 0).max())}
    return arrays, summary


def build_tables(model, times):
    arrays, summaries = {}, {}
    first_last = np.zeros(model.h.shape[1], bool)
    all_local = np.zeros(model.h.shape[1], bool)
    for name, region in (("first", (0, 1)), ("middle2", (2,)), ("middle3", (3,)), ("last", (4, 5))):
        detectors = np.flatnonzero(np.isin(times, region))
        print(f"  building{name}: {len(detectors)} bits", flush=True)
        fields, summary = build_region(model, detectors)
        if np.any(all_local[fields["columns"]]):
            raise ValueError("Temporal local variable groups overlap")
        all_local[fields["columns"]] = True
        if name in ("first", "last"):
            first_last[fields["columns"]] = True
        arrays.update({f"{name}_{key}": value for key, value in fields.items()})
        summaries[name] = summary
    arrays.update({"outside_columns": np.flatnonzero(~first_last).astype(np.int64),
                   "outside_all_columns": np.flatnonzero(~all_local).astype(np.int64),
                   "mechanism_logical_masks": (model.logical.T @ np.asarray([1, 2, 4, 8])).astype(np.uint8),
                   "prior": model.prior.copy(), "probabilities": model.probabilities.copy(),
                   "detector_offset": model.detector_offset.copy(), "logical_offset": model.logical_offset.copy()})
    return arrays, summaries


def load_tables(path, expected_circuit_sha256=None):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].item())
        if metadata.get("schema") != SCHEMA or (expected_circuit_sha256 is not None and metadata["circuit_sha256"] != expected_circuit_sha256):
            raise ValueError("Temporal factor artifact has incompatible schema/circuit")
        arrays = {key: archive[key] for key in archive.files if key != "metadata"}
    return arrays, metadata


def main(argv=None):
    import numba
    import stim
    from prior_model import PriorTiltDecoder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", type=Path, default=ROOT / "data/competition")
    parser.add_argument("--output", type=Path, default=ROOT / "data/target03")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/target03/boundary21_tables.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Report exists; choose a fresh path")
    sources = {name: hash_file(ROOT/name) for name in ("diagnostics/boundary21_tables.py", "diagnostics/temporal_factor_probe.py", "diagnostics/temporal_structure.py")}
    numba.set_num_threads(args.threads)
    report = {"schema": SCHEMA, "source_sha256": sources, "per_basis": {}, "complete": False,
        "probability_meaning": "Exact independent-DEM marginal over all fault orders in each disjoint local variable group.",
        "bias_meaning": "B_i(s)=P(s,Li=0)-P(s,Li=1); P(s) is probability. Four bits, normalized model logical convention.",
        "mask_meaning": "region_syndrome_masks projects EVERY model H column onto the region detector coordinates; region_local_syndrome_masks zeros outside local columns. region_logical_masks stores L nibbles only on local columns. Global affine offsets are stored separately.",
        "numerical_note": "Raw float64 Fourier results retained, including tiny negative masses. Use region_valid, requiring P>=1e-12 and |B|<=P+1e-14; invalid cases need fallback/explicit handling, not blind conditional normalization."}
    args.output.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    for basis in "XZ":
        started = time.perf_counter()
        path = args.output/f"boundary21_{basis}.npz"
        if path.exists():
            raise ValueError(f"Artifact already exists:{path}")
        circuit = stim.Circuit.from_file(str(args.circuits/f"{basis}_circuit.stim"))
        model = PriorTiltDecoder(circuit).model
        times = np.asarray([circuit.get_detector_coordinates()[i][2] for i in range(circuit.num_detectors)])
        arrays, summaries = build_tables(model, times)
        metadata = {"schema": SCHEMA, "basis": basis, "source_sha256": sources,
                    "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
                    "regions": ["first", "middle2", "middle3", "last"], "summaries": summaries,
                    "minimum_probability": 1e-12, "num_variables": model.h.shape[1],
                    "outside_variables_first_last": len(arrays["outside_columns"]),
                    "outside_variables_all_four": len(arrays["outside_all_columns"])}
        temporary = path.with_name(path.name+".building")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        temporary.replace(path)
        report["per_basis"][basis] = {**metadata, "artifact": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hash_file(path), "seconds": time.perf_counter()-started,
            "uncompressed_array_bytes": sum(a.nbytes for a in arrays.values())}
        del arrays
        args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"basis": basis, "artifact": str(path), "bytes": path.stat().st_size,
                          "seconds": report["per_basis"][basis]["seconds"]}), flush=True)
    report["source_sha256_after"] = {name: hash_file(ROOT/name) for name in sources}
    report["complete"] = True; report["inputs_unchanged"] = sources == report["source_sha256_after"]
    args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return 0 if report["inputs_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
