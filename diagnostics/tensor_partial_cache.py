"""Circuit-only endpoint-message caches for the retained temporal decoder.

Keys are selected from endpoint syndrome probability marginals, never from
benchmark occurrence counts or predictions. Strict first-end hits reproduce
the original operations exactly, including the syndrome-dependent order of
normalization. Bidirectional contractions are explicit opt-in because their
algebraically equivalent reassociation can change floating-point rounding.
"""
from __future__ import annotations

import hashlib

import numpy as np
from numba import njit, prange, literal_unroll, get_num_threads

from diagnostics.tensor_fast_chain import (
    _first_edge_csr, _edge_csr, _middle, _normalize, _terminal, _last_dot,
    _one_csr, _batch_csr, prepare_right_layout,
)


def _identity(fast):
    return tuple(id(fast[name]) for name in
        ("first", "last", "middle_hat", "left", "right", "coefficients"))


def _fingerprint(fast):
    digest = hashlib.sha256()
    for name in ("first", "last", "middle_hat", "left", "right", "coefficients"):
        value = fast[name]
        digest.update(name.encode("ascii"))
        digest.update(str((value.dtype.str, value.shape)).encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


@njit(cache=True)
def _fht(values):
    half = 1
    while half < len(values):
        for block in range(0, len(values), 2*half):
            for offset in range(half):
                a, b = values[block+offset], values[block+offset+half]
                values[block+offset], values[block+offset+half] = a+b, a-b
        half *= 2


@njit(cache=True, parallel=True)
def _convolve_endpoint(local, edge_hat):
    for prefix in prange(len(local)):
        values = local[prefix]
        _fht(values)
        for state in range(len(values)): values[state] *= edge_hat[state]
        _fht(values)
        for state in range(len(values)): values[state] = max(0., values[state]/len(values))


def endpoint_source_marginal(fast, endpoint):
    """Return probabilities indexed by the actual packed endpoint source key."""
    if endpoint not in ("first", "last"): raise ValueError("Use first or last endpoint")
    n = 1 << fast["bits"]
    support = fast["left"] if endpoint == "first" else fast["right"]
    marginal = np.bincount(support, weights=fast["coefficients"][:, 0], minlength=n)
    marginal = np.ascontiguousarray(marginal)
    _fht(marginal)
    local = np.ascontiguousarray(fast[endpoint][:, :, 0])
    _convolve_endpoint(local, marginal)
    # First source packs the seven prefix bits low; last source packs them high.
    return np.ascontiguousarray(local.T).ravel() if endpoint == "first" else local.ravel()


@njit(cache=True, parallel=True)
def _prepare_first(keys, first, middle_hat, pointers, left, coefficients, first_prefix, bits):
    n = 1 << bits
    messages = np.empty((len(keys), n, 5)); valid = np.ones(len(keys), np.bool_)
    for row in prange(len(keys)):
        prefix = keys[row]&((1 << first_prefix)-1)
        syndrome = keys[row] >> first_prefix
        values = messages[row]
        _first_edge_csr(values, first, prefix, syndrome, pointers, left, coefficients)
        valid[row] = _normalize(values) > 0
        if valid[row]: _middle(values, middle_hat, 0)
    return messages, valid


@njit(cache=True)
def _fill_last_backward(values, last, prefix, syndrome, left, right, coefficients):
    values[:] = 0.
    for path in range(len(left)):
        a, b = left[path], syndrome^right[path]
        for channel in literal_unroll((0, 1, 2, 3, 4)):
            values[a, channel] += coefficients[path, channel]*last[prefix, b, channel]


@njit(cache=True, parallel=True)
def _prepare_last(keys, last, middle_hat, left, right, coefficients, bits):
    n = 1 << bits
    messages = np.empty((len(keys), n, 5)); valid = np.ones(len(keys), np.bool_)
    for row in prange(len(keys)):
        prefix = keys[row] >> bits; syndrome = keys[row]&(n-1)
        values = messages[row]
        _fill_last_backward(values, last, prefix, syndrome, left, right, coefficients)
        valid[row] = _normalize(values) > 0
        if valid[row]: _middle(values, middle_hat, 0)
    return messages, valid


def prepare_partial_cache(fast, *, first_entries=128, last_entries=0,
                          mode="strict_first", max_bytes=None,
                          guard_thresholds=(.4, .5, .6), boundary_tolerance=1e-10,
                          partition_hits=True):
    """Build and attach a bounded physics-only cache at fast['partial_cache'].

    max_bytes defaults to one quarter of currently available physical memory.
    If necessary the larger requested endpoint cache is halved until it fits.
    Explicit zero/zero disables and removes an existing cache. Every value is
    binary64; no runtime sample or answer participates in cache construction.
    """
    for value in (first_entries, last_entries):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("Cache entry counts must be nonnegative integers")
    if mode not in ("strict_first", "bidirectional"): raise ValueError("Unknown partial-cache mode")
    if mode == "strict_first" and last_entries: raise ValueError("Last entries require explicit bidirectional mode")
    thresholds = np.asarray(guard_thresholds, float)
    if (thresholds.ndim != 1 or not len(thresholds) or np.any(~np.isfinite(thresholds))
            or np.any((thresholds < 0) | (thresholds > 1))):
        raise ValueError("Guard thresholds must be finite probabilities")
    if (isinstance(boundary_tolerance, (bool, np.bool_)) or not np.isfinite(boundary_tolerance)
            or boundary_tolerance < 0): raise ValueError("Boundary tolerance must be finite and nonnegative")
    if not isinstance(partition_hits, (bool, np.bool_)): raise ValueError("partition_hits must be boolean")
    if fast.get("tail_right", np.empty((0, 0))).shape[1]:
        raise ValueError("Partial caches currently require the retained sparse operator without extra tails")
    if not first_entries and not last_entries:
        fast.pop("partial_cache", None)
        return None
    if max_bytes is None:
        from diagnostics.tensor_full_edge import available_physical_bytes
        max_bytes = available_physical_bytes()//4
    if isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, (int, np.integer)) or max_bytes < 1:
        raise ValueError("Cache byte budget must be positive")
    n = 1 << fast["bits"]
    domains = {"first": fast["first"].shape[0]*n, "last": fast["last"].shape[0]*n}
    counts = {"first": min(int(first_entries), domains["first"]),
              "last": min(int(last_entries), domains["last"])}

    def size():
        return thresholds.nbytes+sum(counts[end]*(n*5*8+8)+(domains[end]*4 if counts[end] else 0) for end in ("first", "last"))

    while size() > max_bytes and any(counts.values()):
        endpoint = "first" if counts["first"] >= counts["last"] else "last"
        counts[endpoint] //= 2
    if not any(counts.values()):
        fast.pop("partial_cache", None)
        return None
    pointers, right_left, right_coefficients = prepare_right_layout(fast)
    cache = {"mode": mode, "source_identity": _identity(fast), "partition_hits": bool(partition_hits),
        "guard_thresholds": thresholds.copy(), "boundary_tolerance": float(boundary_tolerance)}
    metadata = {"mode": mode, "source_sha256": _fingerprint(fast), "max_bytes": int(max_bytes),
        "requested_first_entries": int(first_entries), "requested_last_entries": int(last_entries),
        "guard_thresholds": thresholds.tolist(), "boundary_tolerance": float(boundary_tolerance),
        "partition_hits": bool(partition_hits),
        "key_selection": "Stable top packed-source keys from circuit-derived retained endpoint XOR-convolution probabilities; no sample frequencies or answers",
        "arithmetic": "Strict first messages preserve the CSR operation sequence and normalize after actual syndrome permutation. Bidirectional messages preserve physics but reassociate binary64 operations.",
        "per_endpoint": {}}
    for endpoint in ("first", "last"):
        count = counts[endpoint]
        if not count:
            cache[endpoint+"_keys"] = np.empty(0, np.int64)
            cache[endpoint+"_lookup"] = np.empty(0, np.int32)
            cache[endpoint+"_messages"] = np.empty((0, n, 5))
            metadata["per_endpoint"][endpoint] = {"entries": 0, "predicted_retained_source_coverage": 0.}
            continue
        marginal = endpoint_source_marginal(fast, endpoint)
        keys = np.argsort(-marginal, kind="stable")[:count].astype(np.int64)
        coverage = float(marginal[keys].sum()/marginal.sum())
        lookup = np.full(domains[endpoint], -1, np.int32)
        lookup[keys] = np.arange(count, dtype=np.int32)
        if endpoint == "first":
            messages, valid = _prepare_first(keys, fast["first"], fast["middle_hat"][0],
                pointers, right_left, right_coefficients, fast["first_prefix"], fast["bits"])
        else:
            messages, valid = _prepare_last(keys, fast["last"], fast["middle_hat"][1],
                fast["left"], fast["right"], fast["coefficients"], fast["bits"])
        if not np.all(valid): raise ValueError("A selected endpoint cache key has zero retained support")
        cache[endpoint+"_keys"], cache[endpoint+"_lookup"], cache[endpoint+"_messages"] = keys, lookup, messages
        metadata["per_endpoint"][endpoint] = {"entries": count,
            "predicted_retained_source_coverage": coverage, "keys": keys.tolist()}
        del marginal
    metadata["allocated_bytes"] = sum(value.nbytes for value in cache.values() if isinstance(value, np.ndarray))
    cache["metadata"] = metadata
    fast["partial_cache"] = cache
    return cache


@njit(cache=True)
def _copy_shift(target, messages, slot, syndrome):
    for state in range(len(target)):
        for channel in literal_unroll((0, 1, 2, 3, 4)):
            target[state, channel] = messages[slot, state^syndrome, channel]


@njit(cache=True)
def _probabilities(result, logical_offset):
    probability = np.empty(4)
    valid = result[0] > 0 and np.isfinite(result[0])
    for bit in range(4):
        if valid:
            correlation = min(1., max(-1., result[bit+1]/result[0]))
            if (logical_offset >> bit)&1: correlation = -correlation
            probability[bit] = .5*(1-correlation)
        else: probability[bit] = (logical_offset >> bit)&1
    return probability, valid, result[0]


@njit(cache=True)
def _middle_edge_dot(current, endpoint, slot, syndrome, left, right, coefficients):
    result = np.zeros(5)
    for path in range(len(left)):
        a, b = left[path], right[path]^syndrome
        for channel in literal_unroll((0, 1, 2, 3, 4)):
            result[channel] += (coefficients[path, channel]*current[a, channel])*endpoint[slot, b, channel]
    return result


@njit(cache=True)
def _cached_middle_edge_dot(first_messages, first_slot, first_syndrome,
                            last_messages, last_slot, last_syndrome, left, right, coefficients):
    result = np.zeros(5)
    for path in range(len(left)):
        a, b = left[path]^first_syndrome, right[path]^last_syndrome
        for channel in literal_unroll((0, 1, 2, 3, 4)):
            result[channel] += (coefficients[path, channel]*first_messages[first_slot, a, channel])*last_messages[last_slot, b, channel]
    return result


@njit(cache=True)
def _one_partial(source, first, last, middle_hat, left, right, coefficients, tail_right, tail_left,
                 pointers, right_left, right_coefficients, bits, first_prefix, logical_offset, fused_terminal,
                 first_lookup, first_messages, last_lookup, last_messages, bidirectional):
    first_slot = first_lookup[source[0]] if len(first_lookup) else -1
    last_slot = last_lookup[source[3]] if bidirectional and len(last_lookup) else -1
    if first_slot >= 0 and last_slot >= 0:
        # The endpoint normalization is a common positive scalar and cancels
        # from every logical ratio. No full message copy/allocation is needed.
        result = _cached_middle_edge_dot(first_messages, first_slot, source[1],
            last_messages, last_slot, source[2], left, right, coefficients)
        q, valid, mass = _probabilities(result, logical_offset)
        return q, valid, mass, True, True
    if first_slot < 0 and last_slot < 0:
        q, valid, mass = _one_csr(source, first, last, middle_hat, left, right, coefficients, tail_right,
            tail_left, pointers, right_left, right_coefficients, bits, first_prefix, logical_offset, fused_terminal)
        return q, valid, mass, False, False
    n = 1 << bits
    current = np.empty((n, 5))
    if first_slot >= 0:
        _copy_shift(current, first_messages, first_slot, source[1])
        valid = _normalize(current) > 0
    else:
        _first_edge_csr(current, first, source[0]&((1 << first_prefix)-1), source[0] >> first_prefix,
            pointers, right_left, right_coefficients)
        valid = _normalize(current) > 0
        if valid:
            _middle(current, middle_hat[0], source[1]); valid = _normalize(current) > 0
    result = np.zeros(5)
    if valid and last_slot >= 0:
        # Original global path ordering, but backward elimination reassociates
        # the two local convolutions and endpoint contraction in binary64.
        result = _middle_edge_dot(current, last_messages, last_slot, source[2], left, right, coefficients)
    elif valid:
        other = np.empty_like(current)
        _edge_csr(current, other, pointers, right_left, right_coefficients)
        valid = _normalize(other) > 0
        if valid:
            _middle(other, middle_hat[1], source[2]); valid = _normalize(other) > 0
        if valid:
            prefix, syndrome = source[3] >> bits, source[3]&(n-1)
            if fused_terminal:
                result = _terminal(other, last, prefix, syndrome, left, right, coefficients)
            else:
                _edge_csr(other, current, pointers, right_left, right_coefficients)
                valid = _normalize(current) > 0
                if valid: result = _last_dot(current, last, prefix, syndrome)
    q, probability_valid, mass = _probabilities(result, logical_offset)
    return q, valid and probability_valid, mass, first_slot >= 0, last_slot >= 0


@njit(cache=True, parallel=True)
def _batch_partial(sources, first, last, middle_hat, left, right, coefficients, tail_right, tail_left,
                   pointers, right_left, right_coefficients, bits, first_prefix, logical_offset, fused_terminal,
                   first_lookup, first_messages, last_lookup, last_messages, bidirectional):
    q = np.empty((len(sources), 4)); valid = np.empty(len(sources), np.bool_); mass = np.empty(len(sources))
    first_hits = np.empty(len(sources), np.bool_); last_hits = np.empty(len(sources), np.bool_)
    for row in prange(len(sources)):
        q[row], valid[row], mass[row], first_hits[row], last_hits[row] = _one_partial(sources[row],
            first, last, middle_hat, left, right, coefficients, tail_right, tail_left, pointers, right_left,
            right_coefficients, bits, first_prefix, logical_offset, fused_terminal, first_lookup, first_messages,
            last_lookup, last_messages, bidirectional)
    return q, valid, mass, first_hits, last_hits


def decode_partial_sources(fast, sources):
    """Internal retained-decoder hook; returns q,valid,mass and hit metadata."""
    cache = fast["partial_cache"]
    if cache["source_identity"] != _identity(fast):
        raise ValueError("Partial-message cache belongs to another retained operator")
    if fast.get("tail_right", np.empty((0, 0))).shape[1]:
        raise ValueError("Partial-message cache cannot silently omit an extra tail model")
    pointers, right_left, right_coefficients = prepare_right_layout(fast)
    sources = np.ascontiguousarray(sources, dtype=np.int64)
    bidirectional = cache["mode"] == "bidirectional"
    arguments = (fast["first"], fast["last"], fast["middle_hat"], fast["left"], fast["right"],
        fast["coefficients"], fast["tail_right"], fast["tail_left"], pointers, right_left,
        right_coefficients, fast["bits"], fast["first_prefix"], fast["logical_offset"], fast["fused_terminal"])
    cache_arguments = (cache["first_lookup"], cache["first_messages"], cache["last_lookup"],
        cache["last_messages"], bidirectional)
    groups = 1
    if cache["partition_hits"] and len(sources) >= 2*get_num_threads():
        # Static prange chunks otherwise put common cheap keys and rare slow
        # misses on different workers. Uniform-cost classes retain output order.
        classes = np.zeros(len(sources), np.uint8)
        if len(cache["first_lookup"]): classes += (cache["first_lookup"][sources[:, 0]] >= 0).astype(np.uint8)
        if bidirectional and len(cache["last_lookup"]):
            classes += 2*(cache["last_lookup"][sources[:, 3]] >= 0).astype(np.uint8)
        q = np.empty((len(sources), 4)); valid = np.empty(len(sources), np.bool_)
        mass = np.empty(len(sources)); first_hits = np.empty(len(sources), np.bool_)
        last_hits = np.empty(len(sources), np.bool_)
        kinds = np.unique(classes); groups = len(kinds)
        for kind in kinds:
            rows = np.flatnonzero(classes == kind)
            result = _batch_partial(sources[rows], *arguments, *cache_arguments)
            q[rows], valid[rows], mass[rows], first_hits[rows], last_hits[rows] = result
    else:
        q, valid, mass, first_hits, last_hits = _batch_partial(sources, *arguments, *cache_arguments)
    fallback = ~valid | ~np.isfinite(mass) | (mass <= 0) | np.any(~np.isfinite(q), axis=1)
    if bidirectional:
        for threshold in cache["guard_thresholds"]:
            fallback |= np.any(np.abs(q-threshold) <= cache["boundary_tolerance"], axis=1)
    if np.any(fallback):
        q[fallback], valid[fallback], mass[fallback] = _batch_csr(sources[fallback], *arguments)
    return q, valid, mass, {"partial_cache_mode": cache["mode"],
        "partial_cache_first_hits": int(first_hits.sum()), "partial_cache_last_hits": int(last_hits.sum()),
        "partial_cache_both_hits": int(np.count_nonzero(first_hits & last_hits)),
        "partial_cache_runtime_groups": int(groups),
        "partial_cache_numeric_fallbacks": int(fallback.sum()),
        "partial_cache_bytes": cache["metadata"]["allocated_bytes"]}
