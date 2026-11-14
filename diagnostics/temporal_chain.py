"""Deterministic sum-product chain with exact local blocks and sparse edges.

Four logical signed-mass channels plus probability suffice for bitwise loss.
Local blocks sum every fault order. Each inter-round edge currently enumerates
zero/one/two independent faults, rather than truncating total circuit faults.
"""
from __future__ import annotations

import hashlib
import math
from decimal import Decimal, localcontext
from pathlib import Path
import sys

import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@njit(cache=True)
def fht_rows(values):
    n = values.shape[1]
    for channel in range(values.shape[0]):
        half = 1
        while half < n:
            for block in range(0, n, 2*half):
                for offset in range(half):
                    a, b = values[channel, block+offset], values[channel, block+half+offset]
                    values[channel, block+offset] = a+b
                    values[channel, block+half+offset] = a-b
            half *= 2


@njit(cache=True)
def edge_apply(source, left, right, labels, weights):
    target = np.zeros_like(source)
    for path in range(len(weights)):
        a, b, label, weight = left[path], right[path], labels[path], weights[path]
        target[0, b] += weight*source[0, a]
        for bit in range(4):
            sign = -1. if (label >> bit) & 1 else 1.
            target[bit+1, b] += weight*sign*source[bit+1, a]
    return target


@njit(cache=True)
def normalize_messages(values):
    total = values[0].sum()
    if total > 0 and np.isfinite(total):
        values /= total
    return total


@njit(cache=True)
def middle_convolve(messages, local_hat, syndrome):
    fht_rows(messages)
    messages *= local_hat
    fht_rows(messages)
    n = messages.shape[1]
    target = np.empty_like(messages)
    for state in range(n):
        index = syndrome ^ state
        probability = max(0., messages[0, index]/n)
        target[0, state] = probability
        for bit in range(4):
            value = messages[bit+1, index]/n
            target[bit+1, state] = min(probability, max(-probability, value))
    return target


@njit(cache=True)
def decode_one(sfirst, smiddle2, smiddle3, slast, first_p, first_b, last_p, last_b,
               middle_hat, edge_left, edge_right, edge_labels, edge_weights,
               bits, first_prefix, logical_offset):
    n = 1 << bits
    current = np.empty((5, n))
    for state in range(n):
        index = sfirst ^ (state << first_prefix)
        current[0, state] = first_p[index]
        for bit in range(4):
            current[bit+1, state] = first_b[index, bit]
    valid = True
    for edge in range(3):
        current = edge_apply(current, edge_left, edge_right, edge_labels, edge_weights)
        if normalize_messages(current) <= 0:
            valid = False
            break
        if edge < 2:
            syndrome = smiddle2 if edge == 0 else smiddle3
            current = middle_convolve(current, middle_hat[edge], syndrome)
            if normalize_messages(current) <= 0:
                valid = False
                break
    probability = np.zeros(4)
    if not valid:
        for bit in range(4):
            probability[bit] = (logical_offset >> bit) & 1
        return probability, False, 0.
    total = 0.
    signed = np.zeros(4)
    for state in range(n):
        index = slast ^ state
        total += current[0, state]*last_p[index]
        for bit in range(4):
            signed[bit] += current[bit+1, state]*last_b[index, bit]
    if total <= 0 or not np.isfinite(total):
        for bit in range(4):
            probability[bit] = (logical_offset >> bit) & 1
        return probability, False, total
    for bit in range(4):
        correlation = min(1., max(-1., signed[bit]/total))
        if (logical_offset >> bit) & 1:
            correlation = -correlation
        probability[bit] = .5*(1-correlation)
    return probability, True, total


@njit(cache=True, parallel=True)
def decode_batch(sources, first_p, first_b, last_p, last_b, middle_hat,
                 edge_left, edge_right, edge_labels, edge_weights, bits, first_prefix, logical_offset):
    probability = np.empty((len(sources), 4)); valid = np.empty(len(sources), np.bool_)
    likelihood = np.empty(len(sources))
    for row in prange(len(sources)):
        probability[row], valid[row], likelihood[row] = decode_one(
            sources[row, 0], sources[row, 1], sources[row, 2], sources[row, 3],
            first_p, first_b, last_p, last_b, middle_hat, edge_left, edge_right,
            edge_labels, edge_weights, bits, first_prefix, logical_offset)
    return probability, valid, likelihood


def sparse_edge(left_masks, right_masks, logical_masks, probabilities, triple_min_high=None, high_threshold=.01):
    p = np.asarray(probabilities, float)
    n = len(p); count = 1+n+n*(n-1)//2
    high = p > high_threshold
    if triple_min_high is not None:
        if triple_min_high not in (1, 2, 3):
            raise ValueError("Selected triple paths require1,2 or3 high-prior faults")
        n_high = int(high.sum()); n_low = n-n_high
        count += sum(math.comb(n_high, k)*math.comb(n_low, 3-k)
                     for k in range(triple_min_high, 4) if k <= n_high and 3-k <= n_low)
    left, right = np.zeros(count, np.uint16), np.zeros(count, np.uint16)
    labels, weight = np.zeros(count, np.uint8), np.empty(count, float)
    p0 = float(np.exp(np.log1p(-p).sum())); odds = p/(1-p)
    weight[0] = p0
    left[1:n+1], right[1:n+1], labels[1:n+1] = left_masks, right_masks, logical_masks
    weight[1:n+1] = p0*odds
    start = n+1
    for first in range(n-1):
        stop = start+n-first-1
        left[start:stop] = left_masks[first] ^ left_masks[first+1:]
        right[start:stop] = right_masks[first] ^ right_masks[first+1:]
        labels[start:stop] = logical_masks[first] ^ logical_masks[first+1:]
        weight[start:stop] = p0*odds[first]*odds[first+1:]
        start = stop
    if triple_min_high is not None:
        high_indices = np.flatnonzero(high)
        for first in range(n-2):
            for second in range(first+1, n-1):
                needed = triple_min_high-int(high[first])-int(high[second])
                if needed > 1:
                    continue
                third = np.arange(second+1, n) if needed <= 0 else high_indices[high_indices > second]
                stop = start+len(third)
                left[start:stop] = left_masks[first] ^ left_masks[second] ^ left_masks[third]
                right[start:stop] = right_masks[first] ^ right_masks[second] ^ right_masks[third]
                labels[start:stop] = logical_masks[first] ^ logical_masks[second] ^ logical_masks[third]
                weight[start:stop] = p0*odds[first]*odds[second]*odds[third]
                start = stop
    if start != count:
        raise ValueError("Sparse edge path counting disagrees with enumeration")
    return left, right, labels, weight


def retained_probability_decimal(probabilities, triple_min_high=None, high_threshold=.01):
    """80-digit elementary-symmetric probability for the exact support rule."""
    with localcontext() as context:
        context.prec = 80
        p0 = Decimal(1)
        groups = [[Decimal(1), Decimal(0), Decimal(0), Decimal(0)] for _ in range(2)]
        total = [Decimal(1), Decimal(0), Decimal(0)]
        for value in probabilities:
            p = Decimal.from_float(float(value)); p0 *= 1-p
            odds = p/(1-p)
            for order in range(2, 0, -1):
                total[order] += odds*total[order-1]
            group = groups[int(value > high_threshold)]
            for order in range(3, 0, -1):
                group[order] += odds*group[order-1]
        coefficient = sum(total)
        if triple_min_high is not None:
            coefficient += sum(groups[1][k]*groups[0][3-k] for k in range(triple_min_high, 4))
        edge = p0*coefficient
        return str(edge), str(edge**3)


def build_chain(core, basis, max_edge_order=2, triple_min_high=None):
    from diagnostics.boundary21_tables import load_tables
    if max_edge_order != 2:
        raise ValueError("Initial chain prototype supports edge order2")
    tables, metadata = load_tables(ROOT/f"data/target03/boundary21_{basis}.npz",
        hashlib.sha256(str(core.circuit).encode()).hexdigest())
    h, logical, p = core.model.h, core.model.logical, core.model.probabilities
    coordinates = core.circuit.get_detector_coordinates()
    times = np.asarray([coordinates[i][2] for i in range(len(h))])
    signatures, edge_data, edge_supports = [], [], []
    for a, b in ((1, 2), (2, 3), (3, 4)):
        left = np.flatnonzero(times == a); right = np.flatnonzero(times == b)
        outside = np.flatnonzero(~np.isin(times, (a, b)))
        columns = np.flatnonzero(~h[outside].any(axis=0))
        columns = columns[h[left][:, columns].any(axis=0) & h[right][:, columns].any(axis=0)]
        lm = (h[left][:, columns].T @ (1 << np.arange(14))).astype(np.uint16)
        rm = (h[right][:, columns].T @ (1 << np.arange(14))).astype(np.uint16)
        label = (logical[:, columns].T @ np.asarray([1, 2, 4, 8])).astype(np.uint8)
        sorting = np.lexsort((label, rm, lm))
        signature = hashlib.sha256(lm[sorting].tobytes()+rm[sorting].tobytes()+label[sorting].tobytes()+p[columns][sorting].tobytes()).hexdigest()
        signatures.append(signature)
        edge_data.append((lm[sorting], rm[sorting], label[sorting], p[columns][sorting]))
        edge_supports.append({"times": [a, b], "model_columns": columns[sorting].astype(int).tolist(),
            "high_prior_model_columns": columns[sorting][p[columns][sorting] > .01].astype(int).tolist()})
    if len(set(signatures)) != 1:
        raise ValueError("Edge distributions differ; refusing unsupported reuse")
    edge = sparse_edge(*edge_data[0], triple_min_high=triple_min_high)
    edge_probability_decimal, joint_probability_decimal = retained_probability_decimal(edge_data[0][3], triple_min_high)
    middle = np.empty((2, 5, 1 << 14))
    for group, name in enumerate(("middle2", "middle3")):
        probability = np.maximum(tables[name+"_probability"], 0.)
        middle[group, 0] = probability
        middle[group, 1:] = np.clip(tables[name+"_bias"], -probability[:, None], probability[:, None]).T
        fht_rows(middle[group])
    first_p, last_p = np.maximum(tables["first_probability"], 0.), np.maximum(tables["last_probability"], 0.)
    chain = {"first_p": first_p, "last_p": last_p,
        "first_b": np.clip(tables["first_bias"], -first_p[:, None], first_p[:, None]),
        "last_b": np.clip(tables["last_bias"], -last_p[:, None], last_p[:, None]),
        "middle_hat": middle, "edge": edge,
        "regions": [tables[name+"_detectors"] for name in ("first", "middle2", "middle3", "last")],
        "detector_offset": core.model.detector_offset.copy(),
        "logical_offset": int(core.model.logical_offset @ np.asarray([1, 2, 4, 8])),
        "metadata": {"edge_order": 2, "extra_triples_minimum_high_prior_faults": triple_min_high,
                     "high_prior_threshold": .01, "edge_paths": len(edge[0]), "edge_probability_mass": float(edge[3].sum()),
                     "joint_retained_edge_probability_mass": float(edge[3].sum()**3), "edge_signature": signatures[0],
                     "edge_probability_mass_decimal80": edge_probability_decimal,
                     "joint_retained_probability_mass_decimal80": joint_probability_decimal,
                     "edge_supports": edge_supports,
                     "retained_event_rule": "For EACH edge independently: faultcount<=2, OR faultcount==3 and count(p_j>0.01)>=extra_triples_minimum_high_prior_faults when that parameter is notnull. Local blocks have no fault-order restriction.",
                     "circuit_sha256": hashlib.sha256(str(core.circuit).encode()).hexdigest(),
                     "model_arrays_sha256": hashlib.sha256(h.tobytes()+logical.tobytes()+p.tobytes()).hexdigest(),
                     "local_fault_orders": "all", "model": "independent DEM; each edge includes all0/1/2 faults plus explicitly selected3-fault paths"}}
    return chain


def decode_chain(chain, events):
    normalized = np.asarray(events, np.uint8)^chain["detector_offset"]
    sources = np.column_stack([normalized[:, region] @ (1 << np.arange(len(region))) for region in chain["regions"]]).astype(np.int64)
    probability, valid, likelihood = decode_batch(sources, chain["first_p"], chain["first_b"], chain["last_p"], chain["last_b"],
        chain["middle_hat"], *chain["edge"], 14, 7, chain["logical_offset"])
    return probability, {"valid": valid, "last_normalized_likelihood": likelihood, **chain["metadata"]}
