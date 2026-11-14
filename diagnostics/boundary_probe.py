"""Probe exact boundary-factor reranking on existing frozen error explanations.

Outside physical faults remain fixed to the decoder's guess. This is a cheap
conditional heuristic, not inference with a true outside-fault oracle. Only
the label-independent mixed capture subset is used to score each candidate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_factor(h, logical, probabilities, detectors):
    """Return exact P(local syndrome, local logical mask) for local columns."""
    h, logical, probabilities = map(np.asarray, (h, logical, probabilities))
    detectors = np.asarray(detectors, dtype=np.int64)
    if h.ndim != 2 or logical.shape != (4, h.shape[1]) or probabilities.shape != (h.shape[1],):
        raise ValueError("Invalid factor model dimensions")
    if detectors.ndim != 1 or len(detectors) > 10 or len(np.unique(detectors)) != len(detectors):
        raise ValueError("Need at most10 distinct local detector indices")
    if np.any(detectors < 0) or np.any(detectors >= len(h)) or np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid local detectors or fault probabilities")
    if not all(np.all((a == 0) | (a == 1)) for a in (h, logical)):
        raise ValueError("Detector and logical matrices must be binary")
    outside = np.ones(len(h), bool); outside[detectors] = False
    columns = np.flatnonzero(~h[outside].any(axis=0))
    local_states = 1 << len(detectors)
    distribution = np.zeros(local_states * 16); distribution[0] = 1.
    state = np.arange(len(distribution), dtype=np.int64)
    for column in columns:
        detector_effect = int(h[detectors, column] @ (1 << np.arange(len(detectors))))
        logical_effect = int(logical[:, column] @ np.asarray([1, 2, 4, 8]))
        effect = detector_effect | (logical_effect << len(detectors))
        probability = probabilities[column]
        distribution = ((1-probability) * distribution + probability * distribution[state ^ effect])
    joint = distribution.reshape(16, local_states).T.copy()
    return {"detectors": detectors, "columns": columns, "joint": joint,
            "local_h": np.ascontiguousarray(h[detectors][:, columns], dtype=np.uint8),
            "local_logical": np.ascontiguousarray(logical[:, columns], dtype=np.uint8)}


def rerank(errors, plain_predictions, factor):
    """Change only the local logical choice, conditional on the guessed outside."""
    errors, plain = np.asarray(errors), np.asarray(plain_predictions)
    if errors.ndim != 2 or plain.shape != (len(errors), 4) or not all(np.all((a == 0) | (a == 1)) for a in (errors, plain)):
        raise ValueError("Errors and plain predictions must be binary matrices")
    selected = errors[:, factor["columns"]].astype(np.uint8)
    local_syndrome = (selected @ factor["local_h"].T) & 1
    index = local_syndrome @ (1 << np.arange(local_syndrome.shape[1]))
    joint = factor["joint"][index]
    total = joint.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("Chosen local syndrome has zero probability under the factor")
    class_bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.uint8)
    ones = joint @ class_bits
    local_bitwise = (ones > total[:, None]/2).astype(np.uint8)
    local_map = class_bits[joint.argmax(axis=1)]
    old_local = (selected @ factor["local_logical"].T) & 1
    outside = plain.astype(np.uint8) ^ old_local
    return {"class_map": outside ^ local_map, "bitwise": outside ^ local_bitwise}, {
        "local_syndrome_index": index,
        "old_local_logical": old_local, "class_map_local_logical": local_map,
        "bitwise_local_logical": local_bitwise,
        "conditional_bit_one_probability": ones / total[:, None],
        "conditional_logical_class_probabilities": joint / total[:, None],
    }


def metrics(predictions, labels):
    wrong = np.asarray(predictions) != np.asarray(labels)
    losses = wrong.sum(axis=1).astype(np.uint8)
    return {"shots": len(wrong), "wrong_bits": int(wrong.sum()), "bit_ler": float(wrong.mean()),
            "failed_shots": int(losses.astype(bool).sum()),
            "wrong_bits_per_observable": wrong.sum(axis=0).astype(int).tolist()}, losses


def comparison(candidate, baseline, labels):
    wrong_candidate, wrong_baseline = candidate != labels, baseline != labels
    changed = candidate != baseline
    return {"changed_bits": int(changed.sum()), "changed_shots": int(changed.any(axis=1).sum()),
            "corrected_bits": int(np.count_nonzero(changed & wrong_baseline)),
            "introduced_error_bits": int(np.count_nonzero(changed & wrong_candidate)),
            "net_fewer_wrong_bits": int(wrong_baseline.sum()) - int(wrong_candidate.sum())}


def source_hashes():
    names = ("diagnostics/boundary_probe.py", "decoder.py", "fallback.py", "lut/frontend.py",
             "lut/runtime.py", "lut/shared.py", "lut/identity.py", "lut/symmetry.py", "tuning/run_search.py")
    return {name: sha_file(ROOT / name) for name in names if (ROOT / name).is_file()}


def main(argv=None):
    import stim
    from prior_model import PriorTiltDecoder
    from lut.frontend import table_for_circuit
    from lut.identity import algorithm_digest, config_digest
    from tuning.run_search import paired_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "reports/diagnostics/explanation_captures.npz")
    parser.add_argument("--circuits", type=Path, default=ROOT / "data/competition")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/diagnostics/boundary_probe.json")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Report already exists; select a new path")
    sources = source_hashes()
    capture_hash = sha_file(args.capture)
    paths = [ROOT / "lut/tables/shared.npz"]
    if not paths[0].is_file():
        paths = [ROOT / f"lut/tables/{basis}.npz" for basis in "XZ"]
    table_hashes = {str(path.relative_to(ROOT)): sha_file(path) for path in paths if path.is_file()}
    report = {"purpose": "Exploratory fixed-outside boundary-factor reranking; no decoder/LUT changes",
              "capture_sha256": capture_hash, "source_sha256": sources, "table_sha256": table_hashes,
              "algorithm_sha256": algorithm_digest(), "per_basis": {}, "complete": False,
              "interpretation": [
                  "Only the preexisting label-independent random mixed subset contributes metrics or confidence intervals. Failure/success-selected extra captures are excluded.",
                  "The outside correction is the decoder's guess, not an oracle truth. A wrong outside guess can make the local conditional reranking wrong.",
                  "Original independent-DEM priors are convolved exactly inside the final detector slice. BP scores and evaluation labels never enter candidate inference.",
                  "Frozen LUT overrides are applied after reranking, preserving its current precedence.",
                  "Paired intervals are exploratory, shot-clustered and basis-stratified; this small diagnostic sample is not a new performance claim.",
                  "Timings exclude all baseline decoding because explanations were already captured; they are not full-harness speed measurements."]}
    losses, candidates = {}, {}
    with np.load(args.capture, allow_pickle=False) as archive:
        for basis in "XZ":
            circuit = stim.Circuit.from_file(str(args.circuits / f"{basis}_circuit.stim"))
            decoder = PriorTiltDecoder(circuit)
            model = decoder.model
            events, labels, plain, current, mixed = [archive[f"{basis}_{key}"] for key in
                ("events", "labels", "plain_predictions", "predictions", "is_mixed")]
            errors = np.unpackbits(archive[f"{basis}_errors_packed"], axis=1,
                                   count=len(model.prior), bitorder="little")
            if not len(events) or not mixed.any():
                raise ValueError("Capture lacks a nonempty mixed subset")
            if not np.all(archive[f"{basis}_syndrome_valid"]):
                raise ValueError("Captured physical explanation was not syndrome-valid")
            np.testing.assert_array_equal((errors @ model.h.T) & 1, events.astype(np.uint8) ^ model.detector_offset)
            np.testing.assert_array_equal(((errors @ model.logical.T) & 1) ^ model.logical_offset, plain)
            table = table_for_circuit(circuit, decoder.config)
            apply = lambda predictions: table.apply(events, predictions) if table is not None else predictions.copy()
            np.testing.assert_array_equal(apply(plain), current, err_msg="Frozen LUT pipeline changed since capture")
            coordinates = circuit.get_detector_coordinates()
            final_time = max(c[2] for c in coordinates.values())
            detectors = [i for i, c in coordinates.items() if c[2] == final_time]
            started = time.perf_counter()
            factor = build_factor(model.h, model.logical, model.probabilities, detectors)
            factor_seconds = time.perf_counter() - started
            started = time.perf_counter()
            reranked, details = rerank(errors, plain, factor)
            rerank_seconds = time.perf_counter() - started
            predictions = {"plain": plain, "current_lut": current,
                           "class_map_plain": reranked["class_map"], "bitwise_plain": reranked["bitwise"],
                           "class_map_lut": apply(reranked["class_map"]),
                           "bitwise_lut": apply(reranked["bitwise"])}
            scored, basis_losses, selected = {}, {}, {}
            for name, prediction in predictions.items():
                selected[name] = np.ascontiguousarray(prediction[mixed], dtype=np.uint8)
                scored[name], basis_losses[name] = metrics(selected[name], labels[mixed])
                losses.setdefault(name, {})[basis] = basis_losses[name]
            candidates[basis] = selected, labels[mixed]
            comparisons = {}
            for name in ("class_map_plain", "bitwise_plain", "class_map_lut", "bitwise_lut"):
                reference = "current_lut" if name.endswith("_lut") else "plain"
                comparisons[name] = {"reference": reference,
                    **comparison(selected[name], selected[reference], labels[mixed]),
                    "paired": paired_summary({basis: basis_losses[name]}, {basis: basis_losses[reference]})}
            lookup = table.lookup(events) if table is not None else None
            report["per_basis"][basis] = {"captured_shots": len(events), "mixed_shots": int(mixed.sum()),
                "excluded_selected_shots": int((~mixed).sum()), "config": asdict(decoder.config),
                "config_sha256": config_digest(decoder.config),
                "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
                "captured_physical_explanation_verified": True, "frozen_lut_pipeline_verified": True,
                "local_detectors": detectors, "local_mechanisms": len(factor["columns"]),
                "factor_states": int(factor["joint"].size), "factor_probability_sum": float(factor["joint"].sum()),
                "factor_build_seconds": factor_seconds, "rerank_captured_shots_seconds": rerank_seconds,
                "mixed_lut_hit_shots": int(lookup.hit_mask[mixed].sum()) if lookup is not None else 0,
                "mixed_local_syndrome_histogram": {str(i): int(n) for i, n in enumerate(np.bincount(
                    details["local_syndrome_index"][mixed], minlength=128)) if n},
                "metrics": scored, "comparisons": comparisons}
    overall = {"shots": sum(len(v) for v in losses["plain"].values()), "metrics": {}, "comparisons": {}}
    for name in losses:
        predictions = np.concatenate([candidates[basis][0][name] for basis in "XZ"])
        labels = np.concatenate([candidates[basis][1] for basis in "XZ"])
        overall["metrics"][name] = metrics(predictions, labels)[0]
    for name in ("class_map_plain", "bitwise_plain", "class_map_lut", "bitwise_lut"):
        reference = "current_lut" if name.endswith("_lut") else "plain"
        counts = {key: sum(row["comparisons"][name][key] for row in report["per_basis"].values()) for key in
                  ("changed_bits", "changed_shots", "corrected_bits", "introduced_error_bits", "net_fewer_wrong_bits")}
        overall["comparisons"][name] = {"reference": reference, **counts,
                                       "paired": paired_summary(losses[name], losses[reference])}
    overall["bitwise_vs_class_map_lut"] = paired_summary(losses["bitwise_lut"], losses["class_map_lut"])
    report["overall"] = overall
    report["source_sha256_after"] = source_hashes()
    report["capture_sha256_after"] = sha_file(args.capture)
    report["table_sha256_after"] = {name: sha_file(ROOT / name) for name in table_hashes}
    report["complete"] = True
    report["inputs_unchanged"] = (sources == report["source_sha256_after"]
        and capture_hash == report["capture_sha256_after"] and table_hashes == report["table_sha256_after"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)
    return 0 if report["inputs_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
