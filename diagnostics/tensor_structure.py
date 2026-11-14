"""Read-only edge tensor audit and lossless sparse-kernel coalescing.

This is a diagnostic prototype. It never changes the submission decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit


ROOT = Path(__file__).resolve().parents[1]


def edge_mechanisms(core, left_time=1, right_time=2):
    coordinates = core.circuit.get_detector_coordinates()
    times = np.asarray([coordinates[i][2] for i in range(core.circuit.num_detectors)])
    h, logical = core.model.h, core.model.logical
    left = np.flatnonzero(times == left_time)
    right = np.flatnonzero(times == right_time)
    outside = np.flatnonzero(~np.isin(times, (left_time, right_time)))
    columns = np.flatnonzero(~h[outside].any(axis=0))
    columns = columns[h[left][:, columns].any(axis=0) & h[right][:, columns].any(axis=0)]
    a, b, ell = h[left][:, columns], h[right][:, columns], logical[:, columns]
    return a, b, ell, core.model.probabilities[columns], columns


def coalesce_paths(left, right, labels, weights, bits=14):
    """Sum only identical (left,right,logical) effects, without renormalizing.

    Nonnegative probability masses are added; signed logical channels are
    therefore preserved. Addition order changes float64 rounding slightly.
    """
    if not 1 <= bits <= 28:
        raise ValueError("Packed path keys require 1..28 detector bits per side")
    a, b, ell, w = map(np.asarray, (left, right, labels, weights))
    if any(x.ndim != 1 for x in (a, b, ell, w)) or any(len(x) != len(a) for x in (b, ell, w)):
        raise ValueError("Path arrays must be one-dimensional and equally sized")
    if np.any(a < 0) or np.any(b < 0) or np.any(a >= 1 << bits) or np.any(b >= 1 << bits):
        raise ValueError("Detector path effects exceed the declared bit count")
    if np.any(ell < 0) or np.any(ell >= 16) or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("Need 4-bit labels and finite nonnegative masses")
    key = a.astype(np.uint64) | (b.astype(np.uint64) << bits) | (ell.astype(np.uint64) << (2*bits))
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = np.r_[0, np.flatnonzero(sorted_key[1:] != sorted_key[:-1])+1] if len(key) else np.empty(0, int)
    unique = sorted_key[starts]
    mass = np.add.reduceat(w[order], starts) if len(starts) else np.empty(0, float)
    mask = (1 << bits)-1
    return ((unique & mask).astype(a.dtype), ((unique >> bits) & mask).astype(b.dtype),
            (unique >> (2*bits)).astype(np.uint8), mass)


@njit(cache=True)
def apply_paths(source, left, right, labels, weights):
    """Independent probability/four-signed-channel operator for diagnostics."""
    result = np.zeros_like(source)
    for path in range(len(weights)):
        a, b, ell, mass = left[path], right[path], labels[path], weights[path]
        result[0, b] += mass*source[0, a]
        for bit in range(4):
            result[bit+1, b] += mass*(1.-2.*((ell >> bit) & 1))*source[bit+1, a]
    return result


def independent_diagonal(probabilities):
    """All-order P(d) for independent faults with A=B=one coordinate, L=0."""
    values = np.asarray(probabilities, float)
    if values.ndim != 1 or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Need independent Bernoulli probabilities")
    distribution = np.ones(1)
    for p in values:
        distribution = np.r_[distribution*(1-p), distribution*p]
    return distribution


def sampled_characteristic_cut(matrix, probabilities, order, cut, sample=64, seed=113):
    """Small sampled Fourier cut: suggestive singular values, never a TT bound.

    chi(w)=exp((sum log(1-2p)-sum_j log(1-2p_j)(-1)^(w.Mj))/2).
    Samples cannot bound full-tensor rank, discarded probability mass, or LER.
    """
    matrix = np.asarray(matrix, np.uint8)[order]
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, (sample, cut), dtype=np.uint8)
    y = rng.integers(0, 2, (sample, len(matrix)-cut), dtype=np.uint8)
    ls = 1.-2.*((x @ matrix[:cut]) % 2)
    rs = 1.-2.*((y @ matrix[cut:]) % 2)
    logs = np.log1p(-2*np.asarray(probabilities))
    values = np.exp(.5*(logs.sum()-(ls*logs) @ rs.T))
    singular = np.linalg.svd(values, compute_uv=False)
    energy = np.cumsum(singular[::-1]**2)[::-1]
    return {"sample_size": sample, "cut": cut, "singular_values": singular.tolist(),
            "relative_frobenius_tail_after_rank": {str(k): float(np.sqrt(energy[k]/energy[0]))
                for k in (1, 2, 4, 8, 16, 32) if k < len(singular)},
            "caveat": "Only a sampled Fourier submatrix; no rank, total-variation, or decoder-accuracy guarantee."}


def audit(core, sample=64):
    from diagnostics.temporal_chain import sparse_edge
    from lut.symmetry import binary_rank

    a, b, ell, p, columns = edge_mechanisms(core)
    bits = len(a)
    am = (a.T @ (1 << np.arange(bits))).astype(np.uint16)
    bm = (b.T @ (1 << np.arange(bits))).astype(np.uint16)
    lm = (ell.T @ np.asarray([1, 2, 4, 8])).astype(np.uint8)
    high = p > .01
    paths = sparse_edge(am, bm, lm, p, triple_min_high=2)
    compressed = coalesce_paths(*paths, bits=bits)
    low_paths = sparse_edge(am[~high], bm[~high], lm[~high], p[~high])
    identity_high = bool(np.array_equal(a[:, high], np.eye(bits, dtype=np.uint8))
                         and np.array_equal(a[:, high], b[:, high]) and not ell[:, high].any())
    diagonal = independent_diagonal(p[high]) if identity_high else None
    matrix = np.vstack((a, b))
    orders = {"left_then_right": list(range(2*bits)),
              "paired_coordinates": [v for i in range(bits) for v in (i, i+bits)]}
    cuts = {}
    for name, order in orders.items():
        moved = matrix[order]
        exponents = []
        for cut in range(1, len(matrix)):
            cross = moved[:cut].any(axis=0) & moved[cut:].any(axis=0)
            exponents.append(min(binary_rank(moved[:cut, cross]), binary_rank(moved[cut:, cross])))
        cuts[name] = {"order": order, "crossing_rank_upper_bound_exponents": exponents,
                      "sampled_middle_cut": sampled_characteristic_cut(matrix, p, order, bits, sample)}
    pair_key = paths[0].astype(np.uint64) | (paths[1].astype(np.uint64) << bits)
    return {"faults": len(p), "low_faults": int((~high).sum()), "high_faults": int(high.sum()),
            "model_columns": columns.astype(int).tolist(),
            "gf2_ranks": {name: binary_rank(value) for name, value in {
                "left": a, "right": b, "left_right": matrix,
                "left_right_logical": np.vstack((matrix, ell)), "delta": a ^ b,
                "delta_logical": np.vstack((a ^ b, ell))}.items()},
            "low_single_unique_delta": len(np.unique((am ^ bm)[~high])),
            "low_through2_unique_delta": len(np.unique(low_paths[0] ^ low_paths[1])),
            "low_through2_paths": len(low_paths[0]),
            "dense_delta_operator_float64_bytes_per_channel": int(len(np.unique(low_paths[0] ^ low_paths[1]))*(1 << bits)*8),
            "high_is_exact_independent_coordinate_diagonal": identity_high,
            "high_all_order_mass": None if diagonal is None else float(diagonal.sum()),
            "high_four_or_more_probability": None if diagonal is None else float(diagonal[np.asarray([i.bit_count() for i in range(len(diagonal))]) >= 4].sum()),
            "high2_paths": len(paths[0]), "high2_unique_joint_paths": len(compressed[0]),
            "high2_unique_left_right": len(np.unique(pair_key)),
            "high2_retained_mass": float(paths[3].sum()), "compressed_mass": float(compressed[3].sum()),
            "high2_original_bytes": sum(x.nbytes for x in paths),
            "high2_compressed_bytes": sum(x.nbytes for x in compressed),
            "tensor_cuts": cuts,
            "exact_operator_formula": "K(a,b)=sum_delta C_delta(b)*1[a=b xor delta], C_delta(b)=sum_low_paths_with_AxorB=delta P_low(path)*P_high(b xor Bpath)*logical_character(path). All high orders are included; low support truncation must be stated separately."}


def main(argv=None):
    import numba
    import stim
    from prior_model import PriorTiltDecoder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT/"reports/target03/tensor_structure.json")
    parser.add_argument("--sample", type=int, default=64)
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Report exists; select a new path")
    numba.set_num_threads(2)
    report = {"purpose": "Edge tensor structure only; no inference or submission changes", "per_basis": {},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for basis in "XZ":
        circuit = stim.Circuit.from_file(str(ROOT/f"data/competition/{basis}_circuit.stim"))
        report["per_basis"][basis] = audit(PriorTiltDecoder(circuit), args.sample)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps({basis: {k: report['per_basis'][basis][k] for k in (
        'low_single_unique_delta', 'low_through2_unique_delta', 'gf2_ranks',
        'high2_paths', 'high2_unique_joint_paths', 'high_four_or_more_probability')}
        for basis in "XZ"}, indent=2))


if __name__ == "__main__":
    main()
