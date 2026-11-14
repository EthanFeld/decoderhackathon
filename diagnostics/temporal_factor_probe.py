"""Probe all-order21-check boundary blocks using existing physical captures.

Independent outside faults stay at the plain decoder's selected assignment.
No BP is run. Fourier tables sum every local fault order before predicting
logical bits; only the label-independent mixed captures are scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.temporal_structure import walsh_inplace, _characteristic_inplace
from diagnostics.boundary_probe import comparison, metrics


def factor_queries(h, logical, probabilities, queries=None, max_bits=21):
    """Return all-order syndrome mass and4 signed bit masses at queried keys."""
    h, logical, p = np.asarray(h, np.uint8), np.asarray(logical, np.uint8), np.asarray(probabilities, float)
    if h.ndim != 2 or logical.shape != (4, h.shape[1]) or p.shape != (h.shape[1],):
        raise ValueError("Factor model dimensions disagree")
    if len(h) > max_bits or np.any((p < 0) | (p >= .5)):
        raise ValueError("Factor too wide or priors not normalized")
    size = 1 << len(h)
    if queries is None:
        queries = np.arange(size, dtype=np.int64)
    queries = np.asarray(queries, np.int64)
    if np.any((queries < 0) | (queries >= size)):
        raise ValueError("Factor query outside syndrome domain")
    masks = (h.T @ (1 << np.arange(len(h)))).astype(np.int64)
    logs = np.log1p(-2*p); sum_logs = math.fsum(logs.tolist())
    scratch = np.empty(size, np.float64)
    selected = np.empty((len(queries), 5), np.float64)
    for projection in range(5):
        scratch.fill(0.)
        signs = np.ones(len(p), np.int16) if projection == 0 else 1-2*logical[projection-1].astype(np.int16)
        for mask, value in zip(masks, logs*signs):
            scratch[mask] += value
        walsh_inplace(scratch); _characteristic_inplace(scratch, sum_logs); walsh_inplace(scratch)
        selected[:, projection] = scratch[queries]/size
    return {"queries": queries, "total_mass": selected[:, 0], "signed_mass": selected[:, 1:],
            "domain_size": size, "scratch_bytes": scratch.nbytes}


def region_refinement(model, errors, detectors, minimum_probability=1e-12):
    """Return local old/new logical bits, preserving unreliable rare queries."""
    detectors = np.asarray(detectors, np.int64)
    outside = np.ones(len(model.h), bool); outside[detectors] = False
    columns = np.flatnonzero(~model.h[outside].any(axis=0))
    local_h, local_l = model.h[detectors][:, columns], model.logical[:, columns]
    selected = errors[:, columns].astype(np.uint8)
    syndrome = (selected @ local_h.T) & 1
    indices = syndrome @ (1 << np.arange(len(detectors)))
    unique, inverse = np.unique(indices, return_inverse=True)
    started = time.perf_counter()
    factor = factor_queries(local_h, local_l, model.probabilities[columns], unique)
    factor_seconds = time.perf_counter()-started
    mass, signed = factor["total_mass"][inverse], factor["signed_mass"][inverse]
    old = (selected @ local_l.T) & 1
    valid = (mass >= minimum_probability) & np.all(np.abs(signed) <= mass[:, None]+1e-14, axis=1)
    new = (signed < 0).astype(np.uint8)
    new[~valid] = old[~valid]
    return old, new, {"detectors": detectors.tolist(), "mechanisms": len(columns),
        "unique_queried_syndromes": len(unique), "factor_domain_size": factor["domain_size"],
        "factor_scratch_bytes": factor["scratch_bytes"], "factor_build_seconds": factor_seconds,
        "query_mass_minimum": float(mass.min()), "minimum_probability_gate": minimum_probability,
        "rare_or_inconsistent_queries_kept_unchanged": int((~valid).sum())}


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    import numba
    import stim
    from prior_model import PriorTiltDecoder
    from lut.frontend import table_for_circuit
    from tuning.run_search import paired_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "reports/diagnostics/explanation_captures.npz")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/target03/temporal_factor_probe.json")
    parser.add_argument("--circuits", type=Path, default=ROOT / "data/competition")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Report exists; choose a fresh path")
    numba.set_num_threads(args.threads)
    sources = {name: hash_file(ROOT/name) for name in ("diagnostics/temporal_factor_probe.py", "diagnostics/temporal_structure.py", "diagnostics/boundary_probe.py")}
    capture_hash = hash_file(args.capture)
    report = {"purpose": "Exploratory all-order21-check boundary refinement, outside physical errors fixed",
        "capture_sha256": capture_hash, "source_sha256": sources, "threads": args.threads,
        "per_basis": {}, "complete": False,
        "interpretation": ["Only label-independent512 mixed shots per basis enter scores and confidence intervals.",
            "The outside physical correction is a guess, not oracle truth; refinements can introduce errors.",
            "All local fault orders are summed. Per-bit signs optimize the requested bit loss conditional on each fixed outside guess.",
            "Current frozen LUT overrides retain precedence on original raw syndromes.",
            "Each factor uses21 syndrome coordinates. Tiny queried probability below1e-12 preserves the old local answer to avoid Fourier cancellation instability.",
            "This development probe is not fresh evaluation or a deployment change."]}
    losses, candidates = {}, {}
    with np.load(args.capture, allow_pickle=False) as archive:
        for basis in "XZ":
            circuit = stim.Circuit.from_file(str(args.circuits/f"{basis}_circuit.stim"))
            decoder = PriorTiltDecoder(circuit); model = decoder.model
            events, labels, plain, current, mixed = [archive[f"{basis}_{key}"] for key in ("events", "labels", "plain_predictions", "predictions", "is_mixed")]
            errors = np.unpackbits(archive[f"{basis}_errors_packed"], axis=1, count=len(model.prior), bitorder="little")
            np.testing.assert_array_equal((errors @ model.h.T) & 1, events.astype(np.uint8)^model.detector_offset)
            np.testing.assert_array_equal(((errors @ model.logical.T) & 1)^model.logical_offset, plain)
            table = table_for_circuit(circuit, decoder.config)
            apply = lambda predictions: table.apply(events, predictions) if table is not None else predictions.copy()
            np.testing.assert_array_equal(apply(plain), current)
            coordinates = circuit.get_detector_coordinates()
            times = np.asarray([coordinates[i][2] for i in range(circuit.num_detectors)])
            variants, regions = {"current_lut": current}, []
            both = plain.copy()
            for name, region in (("first21", (0, 1)), ("last21", (4, 5))):
                print(f"{basis}: exact all-order {name} factor queries", flush=True)
                old, new, details = region_refinement(model, errors, np.flatnonzero(np.isin(times, region)))
                variants[name+"_lut"] = apply(plain ^ old ^ new)
                both ^= old ^ new
                regions.append({"name": name, **details})
            variants["both21_lut"] = apply(both)
            scored = {}
            selected = {name: values[mixed].astype(np.uint8) for name, values in variants.items()}
            candidates[basis] = selected, labels[mixed]
            for name, values in selected.items():
                scored[name], loss = metrics(values, labels[mixed])
                losses.setdefault(name, {})[basis] = loss
            comparisons = {name: {**comparison(values, selected["current_lut"], labels[mixed]),
                "paired": paired_summary({basis: losses[name][basis]}, {basis: losses["current_lut"][basis]})}
                for name, values in selected.items() if name != "current_lut"}
            report["per_basis"][basis] = {"mixed_shots": int(mixed.sum()), "regions": regions,
                "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
                "metrics": scored, "comparisons": comparisons,
                "frozen_capture_and_lut_verified": True}
    overall = {"metrics": {}, "comparisons": {}}
    for name in losses:
        predictions = np.concatenate([candidates[basis][0][name] for basis in "XZ"])
        labels = np.concatenate([candidates[basis][1] for basis in "XZ"])
        overall["metrics"][name] = metrics(predictions, labels)[0]
        if name != "current_lut":
            counts = {key: sum(row["comparisons"][name][key] for row in report["per_basis"].values()) for key in
                ("changed_bits", "changed_shots", "corrected_bits", "introduced_error_bits", "net_fewer_wrong_bits")}
            overall["comparisons"][name] = {**counts, "paired": paired_summary(losses[name], losses["current_lut"])}
    report["overall"] = overall
    report["source_sha256_after"] = {name: hash_file(ROOT/name) for name in sources}
    report["capture_sha256_after"] = hash_file(args.capture)
    report["complete"] = True
    report["inputs_unchanged"] = sources == report["source_sha256_after"] and capture_hash == report["capture_sha256_after"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)
    return 0 if report["inputs_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
