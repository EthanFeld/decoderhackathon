"""Audit exact temporal inference and compute all-order oracle error bounds.

The oracle reveals every fault crossing chosen detector regions. Inside each
region the full independent DEM is summed, without a fault-order truncation.
Walsh transforms compute four signed logical-bit marginals with2^d storage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit, prange


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@njit(cache=True, parallel=True)
def walsh_inplace(values):
    """Unnormalized Walsh-Hadamard transform; each stage is disjoint pairs."""
    half = 1
    while half < len(values):
        mask = half-1
        for pair in prange(len(values)//2):
            left = (pair & mask) | ((pair & ~mask) << 1)
            right = left + half
            a, b = values[left], values[right]
            values[left], values[right] = a+b, a-b
        half *= 2


@njit(cache=True, parallel=True)
def _characteristic_inplace(values, sum_logs):
    for index in prange(len(values)):
        values[index] = math.exp(.5*(sum_logs-values[index]))


@njit(cache=True, parallel=True)
def _absolute_sum(values):
    total = 0.
    for index in prange(len(values)):
        total += abs(values[index])
    return total


def fourier_bit_risks(h, logical, probabilities, max_bits=21, progress=None):
    """Exact all-order local Bayes bit risk, evaluated in float64.

    B_i(s)=P(s,Li=0)-P(s,Li=1). E_w[(-1)^(w.s)] Fourier inversion gives B.
    Its L1 norm determines Bayes risk: r_i=(1-sum_s|B_i(s)|)/2.
    One reused float64 array limits peak working storage to8*2^d bytes.
    """
    h, logical, p = np.asarray(h, np.uint8), np.asarray(logical, np.uint8), np.asarray(probabilities, float)
    if h.ndim != 2 or logical.shape != (4, h.shape[1]) or p.shape != (h.shape[1],):
        raise ValueError("Model dimensions disagree")
    if len(h) > max_bits or len(h) > 30 or np.any((p < 0) | (p >= .5)):
        raise ValueError("Detector region exceeds storage cap or priors are not normalized below1/2")
    if not all(np.all((a == 0) | (a == 1)) for a in (h, logical)):
        raise ValueError("Detector/logical effects must be binary")
    d = len(h)
    size = 1 << d
    masks = (h.T @ (1 << np.arange(d))).astype(np.int64)
    logs = np.log1p(-2*p)
    sum_logs = math.fsum(logs.tolist())
    available_bytes = None
    if sys.platform == "win32" and size*8 >= 1 << 30:
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("unused", ctypes.c_ulonglong)]
        memory = Memory(); memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise MemoryError("Could not check available RAM before the large transform")
        available_bytes = int(memory.available_physical)
        if available_bytes < size*8 + (768 << 20):
            raise MemoryError("Insufficient free RAM for the single-array transform plus768MiB reserve")
    values = np.empty(size, np.float64)
    risks, norms = [], []
    for bit in range(4):
        values.fill(0.)
        for key, value, label in zip(masks, logs, logical[bit]):
            values[key] += value if not label else -value
        walsh_inplace(values)
        _characteristic_inplace(values, sum_logs)
        walsh_inplace(values)
        norm = float(_absolute_sum(values)/size)
        risk = .5*(1-norm)
        risks.append(risk); norms.append(norm)
        if progress is not None:
            progress(bit, risk)
    # This intentionally loose allowance documents double precision rather
    # than presenting a rounded value as a formal interval-arithmetic proof.
    allowance = max(1e-12, 16*size*np.finfo(float).eps*(d+1)*(abs(sum_logs)+1))
    return {"bit_errors": risks, "mean_bit_error": float(np.mean(risks)),
            "signed_marginal_l1_norms": norms, "working_array_bytes": values.nbytes,
            "available_physical_bytes_before_large_allocation": available_bytes,
            "conservative_roundoff_allowance_per_bit": allowance,
            "lower_values_after_roundoff_allowance": [max(0., risk-allowance) for risk in risks],
            "roundoff_note": "A conservative engineering allowance for float64 transforms/log/exp, not a machine-checked interval-arithmetic certificate."}


def combine_region_risks(regions):
    risks = np.asarray([r["bit_errors"] for r in regions], float)
    combined = .5*(1-np.prod(1-2*risks, axis=0))
    lower = np.asarray([r.get("lower_values_after_roundoff_allowance", r["bit_errors"]) for r in regions], float)
    cautious = .5*(1-np.prod(1-2*lower, axis=0))
    return {"bit_errors": combined.tolist(), "mean_bit_error": float(combined.mean()),
            "lower_after_roundoff_allowances": cautious.tolist(),
            "mean_lower_after_roundoff_allowances": float(cautious.mean()),
            "formula": "r_i=(1-product_region(1-2*r_region_i))/2. Regions have disjoint detector supports and independent local faults; the oracle reveals all remaining faults. This is the exact bitwise Bayes risk with that oracle, and a lower bound without it."}


def detector_graph(h):
    graph = [set() for _ in range(len(h))]
    for column in range(h.shape[1]):
        support = set(np.flatnonzero(h[:, column]).tolist())
        for vertex in support:
            graph[vertex].update(support-{vertex})
    return graph


def elimination_order(h, mode="minfill"):
    """Greedy dual-factor primal graph widths, not optimal treewidth claims."""
    graph = detector_graph(h)
    active = set(range(len(h)))
    order, widths = [], []
    while active:
        if mode == "natural":
            selected = min(active)
        else:
            def score(vertex):
                adjacent = graph[vertex] & active
                neighbors = sorted(adjacent)
                missing = sum(b not in graph[a] for i, a in enumerate(neighbors) for b in neighbors[i+1:])
                return (missing, len(adjacent), vertex) if mode == "minfill" else (len(adjacent), missing, vertex)
            selected = min(active, key=score)
        adjacent = graph[selected] & active
        order.append(selected); widths.append(len(adjacent))
        for neighbor in adjacent:
            graph[neighbor].update(adjacent-{neighbor})
        active.remove(selected)
    width = max(widths, default=0)
    return {"order": order, "induced_width": width, "bag_sizes": [n+1 for n in widths],
            "largest_dense_float64_factor_bytes": (1 << (width+1))*8,
            "dense_entries_across_elimination_bags": sum(1 << (n+1) for n in widths),
            "note": "Heuristic upper bound on detector-dual contraction width. Four logical Fourier characters can be fixed and contracted separately; retaining4 logical indices explicitly costs a further factor16. Original signed Fourier contractions may suffer cancellation on rare syndromes."}


def mechanism_frontier(h, detector_order):
    """Track active syndrome partial parities as faults are processed once."""
    seen = np.zeros(h.shape[1], bool)
    columns = []
    for detector in detector_order:
        new = np.flatnonzero(h[detector].astype(bool) & ~seen)
        columns.extend(new.tolist()); seen[new] = True
    columns.extend(np.flatnonzero(~seen).tolist())
    first, last = np.full(len(h), len(columns), np.int64), np.full(len(h), -1, np.int64)
    for step, column in enumerate(columns):
        support = np.flatnonzero(h[:, column])
        first[support] = np.minimum(first[support], step); last[support] = step
    temporary = [int(np.count_nonzero((first <= step) & (last >= step))) for step in range(len(columns))]
    after = [int(np.count_nonzero((first <= step) & (last > step))) for step in range(len(columns))]
    max_temporary, max_after = max(temporary, default=0), max(after, default=0)
    return {"maximum_temporary_detector_bits": max_temporary, "maximum_after_conditioning_detector_bits": max_after,
            "maximum_bits_with4_logical_accumulators": max_temporary+4,
            "naive_dense_float64_bytes_with4_logical_accumulators": 8*(1 << (max_temporary+4)),
            "meaning": "Process all unprocessed faults incident to each detector in the chosen order, branching each Bernoulli fault and immediately conditioning rows after their final incident fault. Counts are active parity coordinates; algebraic sparsity or a beam can reduce populated states."}


def structure(model, circuit):
    from lut.symmetry import binary_rank

    coordinates = circuit.get_detector_coordinates()
    times = np.asarray([coordinates[i][2] for i in range(circuit.num_detectors)])
    h, logical = model.h, model.logical
    interval_counts, span_counts, intervals = Counter(), Counter(), {}
    for column in range(h.shape[1]):
        support_times = times[h[:, column].astype(bool)]
        key = tuple(sorted(set(support_times.tolist())))
        interval_counts[key] += 1
        span_counts[float(support_times.max()-support_times.min())] += 1
        intervals.setdefault(key, []).append(column)
    ranks = []
    for key, columns in sorted(intervals.items()):
        detectors = np.flatnonzero(np.isin(times, key))
        sub = h[detectors][:, columns]
        rank = binary_rank(sub)
        joint = binary_rank(np.vstack((sub, logical[:, columns])))
        ranks.append({"times": list(key), "mechanisms": len(columns), "detector_bits": len(detectors),
                      "detector_rank": rank, "joint_detector_logical_rank": joint,
                      "dense_joint_distribution_float64_bytes": 8*(1 << joint)})
    eliminations = {mode: elimination_order(h, mode) for mode in ("natural", "minfill", "mindegree")}
    return {"time_slice_counts": {str(t): int(np.count_nonzero(times == t)) for t in sorted(set(times))},
            "mechanism_span_histogram": {str(span): count for span, count in sorted(span_counts.items())},
            "time_supports_and_ranks": ranks, "dual_detector_elimination": eliminations,
            "mechanism_frontiers": {mode: mechanism_frontier(h, row["order"]) for mode, row in eliminations.items()},
            "maximum_fault_neighbors_of_one_syndrome_check": int(h.sum(axis=1).max()),
            "original_fault_primal_clique_width_lower_bound": int(h.sum(axis=1).max()-1),
            "primal_note": "The original fault-variable primal graph makes each high-degree parity check a clique; parity auxiliaries or algebraic elimination can avoid this generic representation, so this is not a lower bound for every tensor construction."}


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    import numba
    import stim
    from prior_model import PriorTiltDecoder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", type=Path, default=ROOT / "data/competition")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/target03/temporal_structure.json")
    parser.add_argument("--interior-pair", action="store_true", help="Use28-bit interior region (2GiB working array)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.report.exists() and not args.overwrite:
        parser.error("Report exists; select a fresh path or explicitly --overwrite")
    if args.threads < 1:
        parser.error("Thread count must be positive")
    numba.set_num_threads(args.threads)
    # Compile tiny arrays before measured regions.
    walsh_inplace(np.ones(4)); _characteristic_inplace(np.ones(4), 0.); _absolute_sum(np.ones(4))
    source = hash_file(__file__)
    partition = [(0, 1), (2, 3), (4, 5)] if args.interior_pair else [(0, 1), (2,), (3,), (4, 5)]
    report = {"purpose": "All-order temporal oracle lower bound and exact-inference feasibility; no decoder changes",
              "source_sha256": source, "partition": partition, "threads": args.threads,
              "environment": {"numpy": np.__version__, "numba": numba.__version__, "stim": stim.__version__},
              "per_basis": {}, "complete": False,
              "scope": "If default detector_error_model succeeds unchanged, its independent DEM represents the circuit's joint detector/logical statistics without approximate_disjoint_errors. Oracle revelation of cross-region faults can only reduce the optimal bit error, so the computed oracle risk lower-bounds every decoder that sees only those detector events.",
              "fourier_formula": "logq_j=log(1-2p_j); f_i[h]=sum_{j:Hj=h}logq_j*(-1)^Lij; chi_i(w)=exp((sum_jlogq_j-FWHT(f_i)(w))/2); B_i=FWHT(chi_i)/2^d; r_i=(1-sum_s|B_i(s)|)/2."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for basis in "XZ":
        circuit = stim.Circuit.from_file(str(args.circuits / f"{basis}_circuit.stim"))
        default = circuit.detector_error_model()
        approximate = circuit.detector_error_model(approximate_disjoint_errors=True)
        decoder = PriorTiltDecoder(circuit)
        times = np.asarray([circuit.get_detector_coordinates()[i][2] for i in range(circuit.num_detectors)])
        row = {"circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
               "default_dem_succeeds": True, "default_and_approximate_dem_text_identical": str(default) == str(approximate),
               "structure": structure(decoder.model, circuit), "regions": []}
        report["per_basis"][basis] = row; save()
        for region in partition:
            detectors = np.flatnonzero(np.isin(times, region))
            outside = np.flatnonzero(~np.isin(times, region))
            columns = np.flatnonzero(~decoder.model.h[outside].any(axis=0))
            print(f"{basis} region{region}: {len(detectors)} detectorbits, {len(columns)} mechanisms", flush=True)
            started = time.perf_counter()
            def progress(bit, risk):
                print(f"{basis} region{region} bit{bit}: oracle error{risk:.9f}", flush=True)
            result = fourier_bit_risks(decoder.model.h[detectors][:, columns], decoder.model.logical[:, columns],
                decoder.model.probabilities[columns], max_bits=28 if args.interior_pair else 21, progress=progress)
            result.update({"times": region, "detectors": len(detectors), "mechanisms": len(columns),
                           "seconds": time.perf_counter()-started})
            row["regions"].append(result); save()
        row["combined_oracle_lower_bound"] = combine_region_risks(row["regions"]); save()
        print(f"{basis}: combined oracle bitLER {row['combined_oracle_lower_bound']['mean_bit_error']:.9f}", flush=True)
    report["mean_oracle_lower_bound"] = float(np.mean([r["combined_oracle_lower_bound"]["mean_bit_error"] for r in report["per_basis"].values()]))
    report["mean_lower_after_roundoff_allowances"] = float(np.mean([r["combined_oracle_lower_bound"]["mean_lower_after_roundoff_allowances"] for r in report["per_basis"].values()]))
    report["source_sha256_after"] = hash_file(__file__)
    report["complete"] = True
    report["source_unchanged"] = source == report["source_sha256_after"]
    save()
    print(json.dumps({"mean_oracle_lower_bound": report["mean_oracle_lower_bound"],
                      "cautious_lower": report["mean_lower_after_roundoff_allowances"]}), flush=True)
    return 0 if report["source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
