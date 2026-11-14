"""Offline full-order edge Fourier transform and small tail-factor prototype.

Only one 2^(2*bits) float64 array is allocated. No dense matrix is persisted.
This diagnostic does not alter any decoder, archive, CSV, or submission.
Set OPENBLAS_NUM_THREADS before starting Python to bound BLAS parallelism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit, prange


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def available_physical_bytes():
    if sys.platform == "win32":
        import ctypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("unused", ctypes.c_ulonglong)]
        memory = Memory(); memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise MemoryError("Could not inspect available physical memory")
        return int(memory.available_physical)
    return int(os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"))


@njit(cache=True, parallel=True)
def flat_walsh(values):
    """Flat butterfly work keeps every transform stage parallel."""
    half = 1
    while half < len(values):
        mask = half-1
        for pair in prange(len(values)//2):
            left = (pair & mask) | ((pair & ~mask) << 1)
            right = left+half
            first, second = values[left], values[right]
            values[left], values[right] = first+second, first-second
        half *= 2


@njit(cache=True, parallel=True)
def characteristic_inplace(values, sum_logs):
    for index in prange(len(values)):
        values[index] = math.exp(.5*(sum_logs-values[index]))


@njit(cache=True, parallel=True)
def scale_inplace(values, factor):
    for index in prange(len(values)):
        values[index] *= factor


@njit(cache=True, parallel=True)
def mass_statistics(values):
    total, negative, square = 0., 0., 0.
    minimum, maximum = np.inf, -np.inf
    for index in prange(len(values)):
        value = values[index]
        total += value
        negative += max(-value, 0.)
        square += value*value
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    return total, negative, square, minimum, maximum


@njit(cache=True, parallel=True)
def threshold_histogram(values, thresholds, chunks=64):
    """One bounded-memory pass; each chunk owns its histogram reductions."""
    counts = np.zeros((chunks, len(thresholds)+1), np.int64)
    mass = np.zeros((chunks, len(thresholds)+1))
    for chunk in prange(chunks):
        start, stop = len(values)*chunk//chunks, len(values)*(chunk+1)//chunks
        for index in range(start, stop):
            value = values[index]
            if value <= 0:
                continue
            lo, hi = 0, len(thresholds)
            while lo < hi:
                middle = (lo+hi)//2
                if value >= thresholds[middle]: lo = middle+1
                else: hi = middle
            counts[chunk, lo] += 1
            mass[chunk, lo] += value
    return counts.sum(axis=0), mass.sum(axis=0)


@njit(cache=True, parallel=True)
def _threshold_chunk_counts(values, threshold, chunks=64):
    counts = np.zeros(chunks, np.int64)
    for chunk in prange(chunks):
        start, stop = len(values)*chunk//chunks, len(values)*(chunk+1)//chunks
        total = 0
        for index in range(start, stop):
            total += values[index] >= threshold
        counts[chunk] = total
    return counts


@njit(cache=True, parallel=True)
def _gather_threshold(values, threshold, offsets):
    indices = np.empty(offsets[-1], np.uint32)
    probability = np.empty(offsets[-1])
    chunks = len(offsets)-1
    for chunk in prange(chunks):
        start, stop = len(values)*chunk//chunks, len(values)*(chunk+1)//chunks
        output = offsets[chunk]
        for index in range(start, stop):
            if values[index] >= threshold:
                indices[output], probability[output] = index, values[index]
                output += 1
    return indices, probability


def sparsity_profile(matrix, thresholds=None):
    thresholds = np.geomspace(1e-18, 1e-2, 65) if thresholds is None else np.asarray(thresholds, float)
    if not np.all(thresholds > 0) or np.any(thresholds[1:] <= thresholds[:-1]):
        raise ValueError("Thresholds must be strictly increasing and positive")
    counts, masses = threshold_histogram(matrix.ravel(), thresholds)
    retained_counts = np.cumsum(counts[::-1])[::-1][1:]
    retained_masses = np.cumsum(masses[::-1])[::-1][1:]
    return [{"threshold": float(t), "entries": int(n), "retained_probability_mass": float(p),
             "one_minus_retained_raw_mass": float(1-p)} for t, n, p in zip(thresholds, retained_counts, retained_masses)]


def gather_support(matrix, threshold):
    values = matrix.ravel()
    counts = _threshold_chunk_counts(values, threshold)
    offsets = np.r_[0, np.cumsum(counts)]
    return _gather_threshold(values, threshold, offsets)


def left_marginal(left, probabilities, bits):
    """All-order marginal for the input syndrome, using only 2^bits storage."""
    left, p = np.asarray(left, np.int64), np.asarray(probabilities, float)
    logs = np.log1p(-2*p)
    values = np.zeros(1 << bits)
    np.add.at(values, left, logs)
    flat_walsh(values)
    characteristic_inplace(values, math.fsum(logs.tolist()))
    flat_walsh(values)
    scale_inplace(values, 1./len(values))
    return values


def tail_row_profile(left, probabilities, bits, paths):
    marginal = left_marginal(left, probabilities, bits)
    retained = np.bincount(paths[0], weights=paths[3], minlength=1 << bits)
    raw_tail = marginal-retained
    # Ranking tiny negative roundoff as zero does not alter the saved kernels.
    order = np.argsort(-np.maximum(raw_tail, 0.), kind="stable")
    cumulative = np.cumsum(raw_tail[order])
    total = float(raw_tail.sum())
    return order.astype(np.uint16), {"full_left_mass": float(marginal.sum()),
        "tail_mass": total, "tail_negative_absolute_mass": float(np.maximum(-raw_tail, 0.).sum()),
        "top_k": {str(k): {"captured_tail_mass": float(cumulative[min(k, len(order))-1]),
            "fraction_of_tail": float(cumulative[min(k, len(order))-1]/total),
            "remaining_edge_omitted_mass": float(total-cumulative[min(k, len(order))-1]),
            "five_channel_float64_bytes": int(min(k, len(order))*(1 << bits)*5*8)}
            for k in (64, 128, 256, 512, 1024)}}


@njit(cache=True)
def _serial_walsh(values):
    half = 1
    while half < len(values):
        for block in range(0, len(values), 2*half):
            for offset in range(half):
                first, second = values[block+offset], values[block+half+offset]
                values[block+offset], values[block+half+offset] = first+second, first-second
        half *= 2


@njit(cache=True, parallel=True)
def _selected_projection_block(left, right, logical, logs, sum_logs, left_bits,
                               selected, first_fourier_right, count, bit):
    result = np.empty((count, len(selected)))
    size = 1 << left_bits
    for row in prange(count):
        fourier_right = first_fourier_right+row
        values = np.zeros(size)
        for j in range(len(logs)):
            word = int(right[j]) & fourier_right
            parity = 0
            while word:
                parity ^= 1
                word &= word-1
            if bit >= 0:
                parity ^= (int(logical[j]) >> bit) & 1
            values[left[j]] += logs[j]*(1.-2.*parity)
        _serial_walsh(values)
        for index in range(size):
            values[index] = math.exp(.5*(sum_logs-values[index]))
        _serial_walsh(values)
        for index in range(len(selected)):
            result[row, index] = values[selected[index]]/size
    return result


@njit(cache=True, parallel=True)
def _inverse_selected_right(values):
    for row in prange(len(values)):
        _serial_walsh(values[row])
        for index in range(values.shape[1]):
            values[row, index] /= values.shape[1]


def selected_channel(left, right, logical, probabilities, left_bits, right_bits,
                     selected, bit=None, block_size=64, progress=None):
    """All-order K[b,a_selected], returned as [selected_index,b].

    Contract the LEFT Fourier index before storing the RIGHT index. This
    needs K*2^right_bits storage rather than 2^(left_bits+right_bits).
    Left/right dimensions may differ; each selected left state is exact
    mathematically, with ordinary float64 transform error documented by
    the caller. No low-order or rank approximation is used here.
    """
    a, b, ell = (np.asarray(value, np.int64) for value in (left, right, logical))
    p, selected = np.asarray(probabilities, float), np.asarray(selected, np.int64)
    if not 1 <= left_bits <= 14 or not 1 <= right_bits <= 14:
        raise ValueError("Each side must contain 1..14 bits")
    if p.ndim != 1 or any(value.shape != p.shape for value in (a, b, ell)):
        raise ValueError("Need equal one-dimensional fault vectors")
    if np.any(a < 0) or np.any(a >= 1 << left_bits) or np.any(b < 0) or np.any(b >= 1 << right_bits):
        raise ValueError("Fault detector effects exceed dimensions")
    if np.any(p < 0) or np.any(p >= .5) or not np.all(np.isfinite(p)) or np.any(ell < 0) or np.any(ell >= 16):
        raise ValueError("Need finite complemented probabilities and 4-bit logical effects")
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= 1 << left_bits):
        raise ValueError("Selected left states must be unique valid indices")
    if bit is not None and bit not in range(4) or block_size < 1:
        raise ValueError("Need logical bit0..3/None and a positive block size")
    logs = np.log1p(-2*p); sum_logs = math.fsum(logs.tolist())
    result = np.empty((len(selected), 1 << right_bits))
    for start in range(0, result.shape[1], block_size):
        count = min(block_size, result.shape[1]-start)
        block = _selected_projection_block(a, b, ell, logs, sum_logs, left_bits,
                                          selected, start, count, -1 if bit is None else bit)
        result[:, start:start+count] = block.T
        if progress and (start+count) % 4096 == 0:
            progress(f"right_fourier_{start+count}")
    _inverse_selected_right(result)
    return result


def subtract_selected_retained(values, selected, paths, left_bits, bit=None):
    positions = np.full(1 << left_bits, -1, np.int64)
    positions[selected] = np.arange(len(selected))
    left, right, labels, weights = paths
    hit = positions[left] >= 0
    signed = weights[hit] if bit is None else weights[hit]*(1.-2.*((labels[hit].astype(int) >> bit) & 1))
    np.add.at(values, (positions[left[hit]], right[hit]), -signed)


@njit(cache=True, parallel=True)
def clip_tail_probability(values):
    defect = 0.
    flat = values.ravel()
    for index in prange(len(flat)):
        if flat[index] < 0:
            defect -= flat[index]
            flat[index] = 0.
    return defect


@njit(cache=True, parallel=True)
def clip_tail_signed(values, probability):
    defect = 0.
    for row in prange(len(values)):
        for col in range(values.shape[1]):
            value, bound = values[row, col], probability[row, col]
            if value > bound:
                defect += value-bound; values[row, col] = bound
            elif value < -bound:
                defect += -bound-value; values[row, col] = -bound
    return defect


def full_channel(left, right, logical, probabilities, bits, bit=None,
                 reserve_bytes=0, progress=None):
    """Return K[b,a]=E[1(A=a,B=b)(-1)^Lbit], all fault orders.

    ``bit=None`` is the nonnegative probability channel. Nothing is clipped.
    Low-order label vectors and complemented-model affine offsets are handled
    by the surrounding chain; this edge distribution has no affine offset.
    """
    a, b, ell = (np.asarray(value, np.uint64) for value in (left, right, logical))
    p = np.asarray(probabilities, float)
    if not 1 <= bits <= 14 or any(value.shape != p.shape for value in (a, b, ell)):
        raise ValueError("Need equally sized vectors and 1..14 bits per side")
    if p.ndim != 1 or np.any(p < 0) or np.any(p >= .5) or not np.all(np.isfinite(p)):
        raise ValueError("Need finite complemented Bernoulli probabilities in [0,1/2)")
    if np.any(a >= 1 << bits) or np.any(b >= 1 << bits) or np.any(ell >= 16):
        raise ValueError("Fault effects exceed declared dimensions")
    if bit is not None and bit not in range(4):
        raise ValueError("Logical bit must be 0..3 or None")
    size = 1 << (2*bits)
    available = available_physical_bytes()
    if available < size*8+reserve_bytes:
        raise MemoryError(f"Need {size*8+reserve_bytes} bytes including reserve; only {available} available")
    logs = np.log1p(-2*p)
    signs = np.ones(len(p)) if bit is None else 1.-2.*((ell >> bit) & 1).astype(float)
    values = np.zeros(size, np.float64)
    keys = a | (b << bits)
    np.add.at(values, keys.astype(np.int64), logs*signs)
    if progress: progress("log_characteristic_filled")
    flat_walsh(values)
    if progress: progress("first_walsh_complete")
    characteristic_inplace(values, math.fsum(logs.tolist()))
    if progress: progress("characteristic_exponentiated")
    flat_walsh(values)
    if progress: progress("inverse_walsh_complete")
    scale_inplace(values, 1./size)
    return values.reshape(1 << bits, 1 << bits)


def subtract_retained(matrix, paths, bit=None):
    """Subtract the exact retained path channel in-place, including duplicates."""
    left, right, labels, weights = paths
    signed = weights if bit is None else weights*(1.-2.*((labels.astype(int) >> bit) & 1))
    np.add.at(matrix, (right, left), -signed)


def randomized_factors(matrix, rank=16, oversample=8, power=0, seed=2731):
    """Model-only randomized range finder, followed by a small exact SVD.

    Returned (U,s,Vh) has orientation A[b,a]~(U*s)@Vh. Frobenius
    residual follows from the orthogonal projected singular values, up to
    float64 arithmetic. This says nothing about posterior or bit error.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or rank < 1 or oversample < 0 or power < 0:
        raise ValueError("Need a matrix and positive rank/nonnegative oversampling and power")
    width = min(rank+oversample, *matrix.shape)
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(matrix.shape[1], width))
    q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(power):
        q_right, _ = np.linalg.qr(matrix.T @ q, mode="reduced")
        q, _ = np.linalg.qr(matrix @ q_right, mode="reduced")
    small = q.T @ matrix
    u, singular, vh = np.linalg.svd(small, full_matrices=False)
    keep = min(rank, len(singular))
    return q @ u[:, :keep], singular[:keep], vh[:keep], singular


def _stat_dict(matrix):
    total, negative, square, minimum, maximum = mass_statistics(matrix.ravel())
    return {"mass": total, "negative_absolute_mass": negative,
            "frobenius_squared": square, "minimum_entry": minimum, "maximum_entry": maximum}


def main(argv=None):
    import numba
    import stim
    from prior_model import PriorTiltDecoder
    from diagnostics.temporal_chain import sparse_edge
    from diagnostics.tensor_structure import edge_mechanisms

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--channel", choices=["P", "B0", "B1", "B2", "B3"], default="P")
    parser.add_argument("--mode", choices=["mass", "svd", "selected"], default="selected")
    parser.add_argument("--selected-left-count", type=int, default=256)
    parser.add_argument("--selected-block-size", type=int, default=64)
    parser.add_argument("--mass-epsilon", type=float, default=1e-6)
    parser.add_argument("--max-support-entries", type=int, default=5_000_000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reserve-gib", type=float, default=1.)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--factors", type=Path)
    args = parser.parse_args(argv)
    report_tag = f"selected{args.selected_left_count}" if args.mode == "selected" else args.channel
    report_path = args.report or ROOT/f"reports/target03/full_edge_{args.basis}_{report_tag}.json"
    suffix = f"selected{args.selected_left_count}" if args.mode == "selected" else ("support" if args.mode == "mass" else f"r{args.rank}")
    factors_path = args.factors or ROOT/f"data/target03/full_edge_{args.basis}_{args.channel}_{suffix}.npz"
    if report_path.exists() or factors_path.exists():
        parser.error("Output exists; choose fresh report and factor paths")
    if args.mode == "mass" and (args.channel != "P" or not 0 < args.mass_epsilon < .5):
        parser.error("Mass mode needs probability channel and epsilon in (0,1/2)")
    if args.mode == "selected" and not 1 <= args.selected_left_count <= 16384:
        parser.error("Selected left count must be1..16384")
    numba.set_num_threads(args.threads)
    circuit = stim.Circuit.from_file(str(ROOT/f"data/competition/{args.basis}_circuit.stim"))
    core = PriorTiltDecoder(circuit)
    a, b, ell, p, columns = edge_mechanisms(core)
    bits = len(a); mask = 1 << np.arange(bits)
    left, right = ((value.T @ mask).astype(np.uint16) for value in (a, b))
    logical = (ell.T @ np.asarray([1, 2, 4, 8])).astype(np.uint8)
    paths = sparse_edge(left, right, logical, p, triple_min_high=2)
    bit = None if args.channel == "P" else int(args.channel[1:])
    source_names = ["diagnostics/tensor_full_edge.py", "diagnostics/tensor_structure.py",
                    "diagnostics/temporal_chain.py", "decoder.py", "fallback.py"]
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
    started = time.perf_counter()
    report = {"purpose": "Full-order edge Fourier tail compression; diagnostic only",
              "basis": args.basis, "channel": args.channel, "complete": False,
              "bits_per_side": bits, "faults": len(p), "matrix_bytes": (1 << (2*bits))*8,
              "available_physical_bytes_before": available_physical_bytes(),
              "source_sha256_before": hashes, "model_columns": columns.astype(int).tolist(),
              "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
              "model_arrays_sha256": hashlib.sha256(core.model.h.tobytes()+core.model.logical.tobytes()+core.model.probabilities.tobytes()).hexdigest(),
              "configuration": vars(args) | {"report": str(report_path), "factors": str(factors_path)},
              "openblas_threads_environment": os.environ.get("OPENBLAS_NUM_THREADS"), "phases": {}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    def progress(phase):
        report["phases"][phase] = time.perf_counter()-started
        save(); print(args.basis, args.channel, phase, round(report["phases"][phase], 3), flush=True)
    save()
    if args.mode == "selected":
        import shutil
        order, profile = tail_row_profile(left, p, bits, paths)
        selected = order[:args.selected_left_count]
        report["left_tail_profile"] = profile
        report["selected_left"] = selected.astype(int).tolist()
        coefficient_path = factors_path.with_suffix(".npy")
        if coefficient_path.exists():
            raise FileExistsError(coefficient_path)
        coefficient_bytes = (1 << bits)*len(selected)*5*8
        required = coefficient_bytes+len(selected)*(1 << bits)*8+(256 << 20)
        report["selected_coefficient_bytes"] = coefficient_bytes
        report["available_disk_bytes"] = shutil.disk_usage(ROOT).free
        if report["available_disk_bytes"] < coefficient_bytes+(256 << 20):
            raise MemoryError("Insufficient disk for selected tail coefficient file and reserve")
        if available_physical_bytes() < required:
            raise MemoryError("Insufficient available RAM for selected output pages, channel and reserve")
        factors_path.parent.mkdir(parents=True, exist_ok=True)
        coefficients = np.lib.format.open_memmap(coefficient_path, mode="w+", dtype=np.float64,
                                                 shape=(1 << bits, len(selected), 5))
        report["channels"] = {}
        for channel in range(5):
            channel_bit = None if channel == 0 else channel-1
            channel_name = "P" if channel == 0 else f"B{channel_bit}"
            values = selected_channel(left, right, logical, p, bits, bits, selected, channel_bit,
                args.selected_block_size, lambda phase: progress(channel_name+"_"+phase))
            full_stats = _stat_dict(values)
            subtract_selected_retained(values, selected, paths, bits, channel_bit)
            tail_stats = _stat_dict(values)
            if channel == 0:
                clipping_defect = float(clip_tail_probability(values))
            else:
                probability = coefficients[:, :, 0].T
                clipping_defect = float(clip_tail_signed(values, probability))
                del probability
            coefficients[:, :, channel] = values.T
            coefficients.flush()
            report["channels"][channel_name] = {"full_selected": full_stats, "raw_tail": tail_stats,
                                                "physical_constraint_clipping_L1": clipping_defect}
            del values
            progress(channel_name+"_complete")
        coefficients.flush(); del coefficients
        metadata = {"orientation": "coefficient[b,selected_left_index,channel]", "channels": ["P", "B0", "B1", "B2", "B3"],
            "basis": args.basis, "bits_per_side": bits, "selected_left": selected.astype(int).tolist(),
            "coefficient_file": coefficient_path.name, "coefficient_bytes": coefficient_bytes,
            "circuit_sha256": report["circuit_sha256"], "model_arrays_sha256": report["model_arrays_sha256"],
            "retained_base": "weight<=2 OR weight==3 and count(p>.01)>=2",
            "added_support": "All omitted physical edge fault assignments whose LEFT syndrome is in selected_left",
            "numerical_note": "Float64 partial Fourier inversion; tiny negative P and |B|>P defects clipped and recorded."}
        np.savez_compressed(factors_path, selected_left=selected, metadata=np.asarray(json.dumps(metadata)))
        report["coefficient_file"] = str(coefficient_path)
        report["coefficient_sha256"] = file_sha256(coefficient_path)
        report["source_sha256_after"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
        report["sources_unchanged"] = report["source_sha256_after"] == hashes
        report["complete"] = True; report["factor_bytes"] = factors_path.stat().st_size
        report["factors_sha256"] = hashlib.sha256(factors_path.read_bytes()).hexdigest()
        progress("complete")
        return
    # Compile every large-array kernel on tiny arrays first.
    tiny = np.ones(4); flat_walsh(tiny); characteristic_inplace(tiny, -1.)
    scale_inplace(tiny, .25); mass_statistics(tiny)
    progress("jit_warmup_complete")
    matrix = full_channel(left, right, logical, p, bits, bit,
                          reserve_bytes=int(args.reserve_gib*(1 << 30)), progress=progress)
    report["full_channel"] = _stat_dict(matrix)
    if args.mode == "mass":
        profile = sparsity_profile(matrix)
        report["sparsity_profile"] = profile
        report["epsilon_support_candidates"] = {}
        for epsilon in (1e-4, 1e-5, 1e-6, 1e-7):
            eligible = [row for row in profile if row["one_minus_retained_raw_mass"] <= epsilon]
            report["epsilon_support_candidates"][str(epsilon)] = eligible[-1] if eligible else None
        progress("probability_sparsity_profile_complete")
        print(json.dumps(report["epsilon_support_candidates"], indent=2), flush=True)
        eligible = [row for row in profile if row["one_minus_retained_raw_mass"] <= args.mass_epsilon]
        if not eligible:
            raise ValueError("No threshold attains requested retained mass")
        chosen = eligible[-1]
        if chosen["entries"] > args.max_support_entries:
            candidates = [row for row in profile if row["entries"] <= args.max_support_entries
                          and row["one_minus_retained_raw_mass"] <= 1e-4]
            if not candidates:
                del matrix
                report["support_not_saved_reason"] = "Requested support exceeds entry cap; no support below epsilon1e-4 fits"
                report["source_sha256_after"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
                report["sources_unchanged"] = report["source_sha256_after"] == hashes
                report["complete"] = True
                progress("complete")
                return
            chosen = candidates[0]
            report["support_target_relaxed_for_entry_cap"] = True
        indices, probability = gather_support(matrix, chosen["threshold"])
        report["selected_support"] = chosen
        report["selected_support"]["uncompressed_index_probability_bytes"] = indices.nbytes+probability.nbytes
        del matrix
        metadata = {"basis": args.basis, "channel": "P", "bits_per_side": bits,
            "orientation": "flat_index=a|(b<<bits)", "threshold": chosen["threshold"],
            "raw_probability_mass": float(probability.sum()),
            "circuit_sha256": report["circuit_sha256"], "model_arrays_sha256": report["model_arrays_sha256"],
            "caveat": "Fourier float64 probability masses, no clipping; numerical defects are reported separately."}
        factors_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(factors_path, flat_indices=indices, probability=probability,
                            metadata=np.asarray(json.dumps(metadata)))
        report["source_sha256_after"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
        report["sources_unchanged"] = report["source_sha256_after"] == hashes
        report["complete"] = True; report["factor_bytes"] = factors_path.stat().st_size
        report["factors_sha256"] = hashlib.sha256(factors_path.read_bytes()).hexdigest()
        progress("complete")
        return
    subtract_retained(matrix, paths, bit)
    report["tail_channel"] = _stat_dict(matrix)
    report["retained_probability_mass"] = float(paths[3].sum())
    progress("retained_channel_subtracted")
    u, singular, vh, projected_singular = randomized_factors(matrix, args.rank, args.oversample, args.power)
    progress("randomized_svd_complete")
    square = report["tail_channel"]["frobenius_squared"]
    report["projected_singular_values"] = projected_singular.tolist()
    report["ranks"] = {}
    for rank in (4, 8, 16):
        if rank > len(singular): continue
        residual = max(0., square-float(np.sum(singular[:rank]**2)))
        report["ranks"][str(rank)] = {"relative_frobenius_residual": math.sqrt(residual/square),
            "frobenius_residual": math.sqrt(residual),
            "entrywise_l1_error_upper_bound_from_frobenius": matrix.shape[0]*math.sqrt(residual),
            "factor_bytes_float64": (matrix.shape[0]+matrix.shape[1])*rank*8}
    sample_indices = np.random.default_rng(931).choice(matrix.shape[1], min(64, matrix.shape[1]), replace=False)
    approximation = (u*singular) @ vh[:, sample_indices]
    exact = matrix[:, sample_indices]
    report["sampled_columns_diagnostics"] = {"columns": sample_indices.tolist(),
        "approximation_negative_absolute_mass": float(np.maximum(-approximation, 0.).sum()),
        "absolute_error": float(np.abs(approximation-exact).sum()),
        "exact_absolute_mass": float(np.abs(exact).sum()),
        "note": "Selected model-only columns; not a global probability or posterior bound."}
    factors_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"orientation": "tail[b,a]=(left_factor[b,:]*singular_values)@right_factor[:,a]",
                "basis": args.basis, "channel": args.channel, "bits_per_side": bits,
                "rank": len(singular), "circuit_sha256": report["circuit_sha256"],
                "model_arrays_sha256": report["model_arrays_sha256"],
                "retained_support": "weight<=2 OR weight==3 and count(p>.01)>=2"}
    np.savez_compressed(factors_path, left_factor=u, singular_values=singular, right_factor=vh,
                        metadata=np.asarray(json.dumps(metadata)))
    del matrix
    report["source_sha256_after"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
    report["sources_unchanged"] = report["source_sha256_after"] == hashes
    report["complete"] = True
    report["factor_bytes"] = factors_path.stat().st_size
    report["factors_sha256"] = hashlib.sha256(factors_path.read_bytes()).hexdigest()
    progress("complete")


if __name__ == "__main__":
    main()
