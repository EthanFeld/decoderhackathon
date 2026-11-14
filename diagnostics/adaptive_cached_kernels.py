"""Batched adaptive row selection and direct contraction from a KN5 row bank.

The selected physical model is unchanged: stable top-K input states on each
retained forward edge, sorted union, then those same rows on all three edges.
Selection computes only the probability forward messages. Contraction indexes
prevalidated bank rows directly, tiles its small accumulator, and preserves
ascending-state row sums. A complete resident bank needs no row gathering.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange, literal_unroll

from diagnostics.tensor_fast_chain import (
    _first_edge, _edge, _middle, _normalize, _terminal, _last_dot,
    _first_edge_csr, _edge_csr, _probability_edge_csr, prepare_right_layout,
)
from diagnostics.tensor_sliced_chain import _gather_first, _gather_selected


def pack_sources(fast, events):
    values = np.asarray(events)
    if values.ndim != 2 or values.shape[1] != len(fast["detector_offset"]) or not np.all((values == 0) | (values == 1)):
        raise ValueError("Expected binary detector events[shots,detectors]")
    normalized = np.asarray(values, np.uint8)^fast["detector_offset"]
    return np.ascontiguousarray(np.column_stack([
        normalized[:, region]@(1 << np.arange(len(region))) for region in fast["regions"]
    ]), dtype=np.int64)


@njit(cache=True)
def _scalar_fht(values):
    half = 1
    while half < len(values):
        for block in range(0, len(values), 2*half):
            for offset in range(half):
                left, right = block+offset, block+offset+half
                a, b = values[left], values[right]
                values[left], values[right] = a+b, a-b
        half *= 2


@njit(cache=True)
def _normalize_probability(values):
    total = 0.
    for state in range(len(values)): total += values[state]
    if total > 0 and np.isfinite(total):
        for state in range(len(values)): values[state] /= total
    return total


@njit(cache=True)
def _probability_edge(source, target, left, right, coefficients):
    target[:] = 0.
    for path in range(len(left)):
        target[right[path]] += coefficients[path, 0]*source[left[path]]


@njit(cache=True)
def _probability_middle(source, local_hat, syndrome):
    _scalar_fht(source)
    for state in range(len(source)): source[state] *= local_hat[state, 0]
    _scalar_fht(source)
    result = np.empty_like(source)
    scale = 1./len(source)
    for state in range(len(source)): result[state] = max(0., source[state^syndrome]*scale)
    return result


@njit(cache=True)
def _mark_topk(probability, row_mass, take, chosen):
    # Strict comparison preserves ascending-state order on all exact ties.
    scores = np.full(take, -np.inf)
    indices = np.full(take, -1, np.int64)
    for state in range(len(probability)):
        score = probability[state]*row_mass[state]
        if score <= scores[take-1]: continue
        position = take-1
        while position > 0 and score > scores[position-1]:
            scores[position] = scores[position-1]
            indices[position] = indices[position-1]
            position -= 1
        scores[position] = score; indices[position] = state
    for index in indices:
        if index >= 0: chosen[index] = True


@njit(cache=True)
def _select_one(source, first, middle_hat, pointers, left, coefficients, row_mass,
                take, bits, first_prefix):
    n = 1 << bits
    current = np.empty(n); other = np.empty(n); chosen = np.zeros(n, np.bool_)
    prefix = source[0]&((1 << first_prefix)-1); syndrome = source[0] >> first_prefix
    for state in range(n): current[state] = first[prefix, syndrome^state, 0]
    valid = _normalize_probability(current) > 0
    if valid: _mark_topk(current, row_mass, take, chosen)
    for edge in range(2):
        if valid:
            _probability_edge_csr(current, other, pointers, left, coefficients)
            valid = _normalize_probability(other) > 0
        if valid:
            current = _probability_middle(other, middle_hat[edge], source[edge+1])
            valid = _normalize_probability(current) > 0
            if valid: _mark_topk(current, row_mass, take, chosen)
    selected = np.full(3*take, -1, np.int64); count = 0
    if valid:
        for state in range(n):
            if chosen[state]: selected[count] = state; count += 1
    return selected, count, valid


@njit(cache=True, parallel=True)
def _select_batch(sources, first, middle_hat, pointers, left, coefficients, row_mass,
                  take, bits, first_prefix):
    selected = np.empty((len(sources), 3*take), np.int64)
    counts = np.empty(len(sources), np.int64); valid = np.empty(len(sources), np.bool_)
    for shot in prange(len(sources)):
        selected[shot], counts[shot], valid[shot] = _select_one(sources[shot], first, middle_hat,
            pointers, left, coefficients, row_mass, take, bits, first_prefix)
    return selected, counts, valid


def select_rows_sources(fast, sources, row_mass, rows_per_edge=16):
    """Return sorted states[shots,3*K] padded -1, counts and validity."""
    sources = _sources(fast, sources)
    mass = np.asarray(row_mass, float)
    n = 1 << fast["bits"]
    if mass.shape != (n,) or np.any(~np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("Need finite nonnegative omitted mass for every input state")
    if isinstance(rows_per_edge, (bool, np.bool_)) or not isinstance(rows_per_edge, (int, np.integer)) or not 1 <= rows_per_edge <= n:
        raise ValueError("rows_per_edge must be an integer within the input domain")
    pointers, right_left, right_coefficients = prepare_right_layout(fast)
    return _select_batch(sources, fast["first"], fast["middle_hat"], pointers, right_left,
        right_coefficients, np.ascontiguousarray(mass), int(rows_per_edge), fast["bits"], fast["first_prefix"])


@njit(cache=True)
def _tail_sum(work, selected_values, row_bank, positions, count):
    work[:] = 0.
    # Contiguous KN5 reads avoid a copied NK5 tensor for every syndrome.
    # Each output still sums selected rows in the exact original row order.
    for row in range(count):
        position = positions[row]
        for state in range(len(work)):
            for channel in literal_unroll((0, 1, 2, 3, 4)):
                work[state, channel] += row_bank[position, state, channel]*selected_values[row, channel]


@njit(cache=True)
def _add_tail(target, work):
    for state in range(len(target)):
        for channel in literal_unroll((0, 1, 2, 3, 4)):
            target[state, channel] += work[state, channel]


@njit(cache=True)
def _tail_block(work, selected_values, row_bank, positions, count, first, size):
    work[:] = 0.
    # A small state tile keeps all partial sums in L1 rather than repeatedly
    # reading/writing a 640 KiB output for every selected dense row.
    for row in range(count):
        position = positions[row]
        v0,v1,v2,v3,v4 = (selected_values[row,0],selected_values[row,1],selected_values[row,2],
                          selected_values[row,3],selected_values[row,4])
        for local in range(size):
            state = first+local
            work[local,0] += row_bank[position,state,0]*v0
            work[local,1] += row_bank[position,state,1]*v1
            work[local,2] += row_bank[position,state,2]*v2
            work[local,3] += row_bank[position,state,3]*v3
            work[local,4] += row_bank[position,state,4]*v4


@njit(cache=True)
def _tail_add_normalize(target, work, selected_values, row_bank, positions, count):
    total = 0.
    for first in range(0, len(target), len(work)):
        size = min(len(work),len(target)-first)
        _tail_block(work, selected_values, row_bank, positions, count, first, size)
        for local in range(size):
            state = first+local
            for channel in literal_unroll((0,1,2,3,4)):
                target[state,channel] += work[local,channel]
            # The original normalize reduction visits states in this same order.
            total += target[state,0]
    if total > 0 and np.isfinite(total):
        for state in range(len(target)):
            for channel in literal_unroll((0,1,2,3,4)):
                target[state,channel] /= total
    return total


@njit(cache=True)
def _tail_terminal(work, selected_values, row_bank, positions, count, last, prefix, syndrome):
    result = np.zeros(5)
    n = last.shape[1]
    for first in range(0,n,len(work)):
        size = min(len(work),n-first)
        _tail_block(work, selected_values, row_bank, positions, count, first, size)
        for local in range(size):
            index = syndrome^(first+local)
            for channel in literal_unroll((0,1,2,3,4)):
                result[channel] += work[local,channel]*last[prefix,index,channel]
    return result


@njit(cache=True)
def _contract_one(source, first, last, middle_hat, left, right, coefficients,
                  selected, positions, count, row_bank, bits, first_prefix,
                  logical_offset, fused_terminal, pointers, right_left, right_coefficients):
    n = 1 << bits
    current = np.empty((n, 5)); other = np.empty((n, 5)); work = np.empty((min(n,256), 5))
    first_low = source[0]&((1 << first_prefix)-1); first_high = source[0] >> first_prefix
    _first_edge_csr(current, first, first_low, first_high, pointers, right_left, right_coefficients)
    valid = _tail_add_normalize(current, work,
        _gather_first(first, first_low, first_high, selected, count), row_bank, positions, count) > 0
    if valid:
        _middle(current, middle_hat[0], source[1]); valid = _normalize(current) > 0
    if valid:
        _edge_csr(current, other, pointers, right_left, right_coefficients)
        valid = _tail_add_normalize(other, work,
            _gather_selected(current, selected, count), row_bank, positions, count) > 0
    if valid:
        _middle(other, middle_hat[1], source[2]); valid = _normalize(other) > 0
    result = np.zeros(5)
    if valid:
        prefix = source[3] >> bits; syndrome = source[3]&((1 << bits)-1)
        values = _gather_selected(other, selected, count)
        if fused_terminal:
            result = _terminal(other, last, prefix, syndrome, left, right, coefficients)
            result += _tail_terminal(work, values, row_bank, positions, count, last, prefix, syndrome)
        else:
            _edge_csr(other, current, pointers, right_left, right_coefficients)
            valid = _tail_add_normalize(current, work, values, row_bank, positions, count) > 0
            if valid: result = _last_dot(current, last, prefix, syndrome)
    valid = valid and result[0] > 0 and np.isfinite(result[0])
    probability = np.empty(4)
    for bit in range(4):
        if valid:
            correlation = min(1., max(-1., result[bit+1]/result[0]))
            if (logical_offset >> bit)&1: correlation = -correlation
            probability[bit] = .5*(1-correlation)
        else: probability[bit] = (logical_offset >> bit)&1
    return probability, valid, result[0]


@njit(cache=True, parallel=True)
def _contract_batch(sources, first, last, middle_hat, left, right, coefficients,
                    selected, positions, counts, row_bank, bits, first_prefix,
                    logical_offset, fused_terminal, pointers, right_left, right_coefficients):
    probability = np.empty((len(sources), 4)); valid = np.empty(len(sources), np.bool_)
    mass = np.empty(len(sources))
    for shot in prange(len(sources)):
        probability[shot], valid[shot], mass[shot] = _contract_one(sources[shot], first, last,
            middle_hat, left, right, coefficients, selected[shot], positions[shot], counts[shot],
            row_bank, bits, first_prefix, logical_offset, fused_terminal, pointers, right_left, right_coefficients)
    return probability, valid, mass


def _sources(fast, sources):
    values = np.asarray(sources)
    if values.ndim != 2 or values.shape[1] != 4 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Need packed source syndromes[shots,4]")
    limits = np.asarray([1 << len(region) for region in fast["regions"]])
    if np.any(values < 0) or np.any(values >= limits): raise ValueError("Packed source exceeds region width")
    return np.ascontiguousarray(values, dtype=np.int64)


def contract_selected_sources(fast, sources, selected_states, counts, row_bank, bank_positions):
    """Contract per-shot rows from prevalidated bank[K,2^bits,5] directly.

    selected_states and bank_positions share padded[shots,width] shape. Only
    their first counts[shot] entries are used. Active states must be sorted and
    unique. Coefficient finiteness/sign bounds belong to row-store validation,
    avoiding a scan of the whole disk-backed bank on each inference call.
    """
    sources = _sources(fast, sources)
    selected, positions, counts = map(np.asarray, (selected_states, bank_positions, counts))
    if selected.ndim != 2 or selected.shape[0] != len(sources) or positions.shape != selected.shape:
        raise ValueError("Need selected states and bank positions[shots,width]")
    if counts.shape != (len(sources),) or any(not np.issubdtype(x.dtype, np.integer) for x in (selected, positions, counts)):
        raise ValueError("States, positions and counts must be integer arrays")
    if np.any(counts < 0) or np.any(counts > selected.shape[1]): raise ValueError("Invalid selected count")
    bank = np.asarray(row_bank)
    n = 1 << fast["bits"]
    if bank.ndim != 3 or bank.shape[1:] != (n, 5) or bank.dtype != np.float64 or not bank.flags.c_contiguous:
        raise ValueError("Need contiguous float64 bank[K,2^bits,5], preferably a mmap")
    active = np.arange(selected.shape[1])[None] < counts[:, None]
    if np.any(selected[active] < 0) or np.any(selected[active] >= n) or np.any(positions[active] < 0) or np.any(positions[active] >= len(bank)):
        raise ValueError("Selected state or row-bank index out of range")
    pairs = np.arange(max(0, selected.shape[1]-1))[None] < (counts-1)[:, None]
    if np.any(np.diff(selected, axis=1)[pairs] <= 0): raise ValueError("Active selected states must be sorted and unique")
    if fast.get("tail_right", np.empty((0, 0))).shape[1]: raise ValueError("Combine low-rank tails explicitly")
    pointers, right_left, right_coefficients = prepare_right_layout(fast)
    return _contract_batch(sources, fast["first"], fast["last"], fast["middle_hat"], fast["left"],
        fast["right"], fast["coefficients"], np.ascontiguousarray(selected, dtype=np.int64),
        np.ascontiguousarray(positions, dtype=np.int64), np.ascontiguousarray(counts, dtype=np.int64),
        bank, fast["bits"], fast["first_prefix"], fast["logical_offset"], fast["fused_terminal"],
        pointers, right_left, right_coefficients)


def contract_resident_sources(fast, sources, selected_states, counts, resident_rows):
    """Use global state IDs directly in a full resident bank[N,N,5].

    No per-batch row gathering or packing occurs. The unrestricted shot batch
    lets the existing shot-parallel kernel use all available CPU workers.
    """
    n = 1 << fast["bits"]
    if np.shape(resident_rows) != (n,n,5): raise ValueError("Expected the complete resident row bank[N,N,5]")
    return contract_selected_sources(fast, sources, selected_states, counts,
        resident_rows, selected_states)
