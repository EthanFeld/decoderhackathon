"""Original paper-based PriorTilt decoder for the 2026 QEC hackathon.

Algorithm: Min Ye and Nicolas Delfosse, arXiv:2611.13373v1, equations 1--7.
No code or decoder libraries from the authors' implementation are used.
"""
from __future__ import annotations

import hashlib

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import stim
from numba import njit, prange

from fallback import osd_repair


@dataclass(frozen=True)
class DecoderConfig:
    warmup_iterations: int = 12
    round_iterations: int = 6
    max_rounds: int = 256
    # Three phases (theta, pi, c1, c2), with zero-based round boundaries.
    phase_ends: tuple[int, int] = (64, 160)
    phases: tuple[tuple[float, float, float, float], ...] = (
        (2.0, 0.10, 0.5, 0.8),
        (2.0, 0.08, 0.4, 1.0),
        (2.0, 0.05, 0.3, 1.2),
    )
    osd_order: int = 2
    osd_width: int = 12
    seed: int = 20261113

    def __post_init__(self):
        integers = (self.warmup_iterations, self.round_iterations, self.max_rounds,
                    self.osd_order, self.osd_width, self.seed, *self.phase_ends)
        if any(not isinstance(v, (int, np.integer)) for v in integers):
            raise ValueError("Iteration counts, widths, boundaries and seed must be integers")
        if self.warmup_iterations < 1 or self.round_iterations < 1 or self.max_rounds < 0:
            raise ValueError("Need positive warmup/round length and nonnegative rounds")
        if len(self.phase_ends) != 2 or not 0 <= self.phase_ends[0] <= self.phase_ends[1]:
            raise ValueError("phase_ends must be two increasing nonnegative boundaries")
        if len(self.phases) != 3 or any(len(p) != 4 for p in self.phases):
            raise ValueError("phases must contain three (theta, pi, c1, c2) tuples")
        for theta, pi, c1, c2 in self.phases:
            if not np.all(np.isfinite([theta, pi, c1, c2])) or theta <= 0 or not 0 <= pi <= 1 or min(c1, c2) < 0:
                raise ValueError("Invalid PriorTilt phase parameters")
        if self.osd_order not in (0, 1, 2) or self.osd_width < 0 or not 0 <= self.seed < 2**63:
            raise ValueError("OSD order must be 0, 1, or 2; width and seed nonnegative")


@dataclass
class DetectorModel:
    h: np.ndarray
    logical: np.ndarray
    probabilities: np.ndarray
    prior: np.ndarray
    row_ptr: np.ndarray
    edge_var: np.ndarray
    col_ptr: np.ndarray
    col_edges: np.ndarray
    detector_offset: np.ndarray
    logical_offset: np.ndarray


def model_from_dem(dem: stim.DetectorErrorModel) -> DetectorModel:
    """Keep hyperedges intact and merge only identical detector+logical effects.

    Stim's separator does not make separate independent faults: all targets of
    one error instruction belong to one Bernoulli variable. Duplicate targets
    cancel over GF(2). Probabilities merge using the odd-parity probability.
    """
    m, k = dem.num_detectors, dem.num_observables
    merged = {}
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            support = detectors if target.is_relative_detector_id() else observables if target.is_logical_observable_id() else None
            if support is not None:
                value = target.val
                if value in support:
                    support.remove(value)
                else:
                    support.add(value)
        key = (tuple(sorted(detectors)), tuple(sorted(observables)))
        previous = merged.get(key, 0.0)
        merged[key] = previous + probability - 2 * previous * probability

    detector_offset = np.zeros(m, dtype=np.uint8)
    logical_offset = np.zeros(k, dtype=np.uint8)
    columns, labels, probabilities = [], [], []
    for (detectors, observables), probability in sorted(merged.items()):
        # Complement likely/certain faults, preserving their affine offsets.
        if probability > 0.5:
            detector_offset[list(detectors)] ^= 1
            logical_offset[list(observables)] ^= 1
            probability = 1.0 - probability
        if probability <= 0 or not detectors:
            continue
        columns.append(detectors)
        labels.append(observables)
        probabilities.append(probability)
    h = np.zeros((m, len(columns)), dtype=np.uint8)
    logical = np.zeros((k, len(columns)), dtype=np.uint8)
    for j, (detectors, observables) in enumerate(zip(columns, labels)):
        h[list(detectors), j] = 1
        logical[list(observables), j] = 1
    rows, variables = np.nonzero(h)
    row_ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=m))].astype(np.int32)
    edge_var = variables.astype(np.int32)
    col_edges = np.argsort(edge_var, kind="stable").astype(np.int32)
    col_ptr = np.r_[0, np.cumsum(np.bincount(variables, minlength=len(columns)))].astype(np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prior = np.log1p(-probabilities) - np.log(probabilities)
    return DetectorModel(h, logical, probabilities, prior, row_ptr, edge_var,
                         col_ptr, col_edges, detector_offset, logical_offset)


@njit(cache=True)
def _satisfies(error, syndrome, row_ptr, edge_var):
    for i in range(len(syndrome)):
        parity = 0
        for edge in range(row_ptr[i], row_ptr[i + 1]):
            parity ^= error[edge_var[edge]]
        if parity != syndrome[i]:
            return False
    return True


@njit(cache=True)
def _uniform(state):
    # Per-syndrome xorshift stream: independent of batch order and threads.
    state ^= state << np.uint64(13)
    state ^= state >> np.uint64(7)
    state ^= state << np.uint64(17)
    return state, float(state >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _bp_one(syndrome, prior, row_ptr, edge_var, col_ptr, col_edges,
            warmup, round_length, max_rounds, phase_ends, phases, seed):
    n, edge_count = len(prior), len(edge_var)
    outgoing = prior[edge_var].copy()
    incoming = np.zeros(edge_count, dtype=np.float64)
    posterior = prior.copy()
    effective = prior.copy()
    error = np.zeros(n, dtype=np.uint8)
    sums = np.zeros(n, dtype=np.float64)
    minima = np.zeros(n, dtype=np.float64)
    maxima = np.zeros(n, dtype=np.float64)
    state = np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)
    for i in range(len(syndrome)):
        if syndrome[i]:
            state ^= np.uint64(i + 1) * np.uint64(0x517CC1B727220A95)
            state, _ = _uniform(state)
    if state == 0:
        state = np.uint64(1)
    if _satisfies(error, syndrome, row_ptr, edge_var):
        return error, posterior, True, 0
    budget = warmup + round_length * max_rounds
    for iteration in range(budget):
        if iteration >= warmup and (iteration - warmup) % round_length == 0:
            round_index = (iteration - warmup) // round_length
            phase = 0 if round_index < phase_ends[0] else 1 if round_index < phase_ends[1] else 2
            theta, pi, c1, c2 = phases[phase]
            for j in range(n):
                effective[j] = prior[j]
                if sums[j] < theta * prior[j]:
                    state, random_value = _uniform(state)
                    effective[j] -= c1 * minima[j] if random_value < pi else c2 * maxima[j]
            sums[:] = 0.0
            minima[:] = 0.0
            maxima[:] = 0.0
        # Four independent exact top-two reductions shorten the critical path.
        # The sentinel cap also preserves the original comparisons for NaNs.
        for i in range(len(syndrome)):
            negative = syndrome[i] != 0
            a0, b0, a1, b1, a2, b2, a3, b3 = 1e100, 1e100, 1e100, 1e100, 1e100, 1e100, 1e100, 1e100
            begin, end = row_ptr[i], row_ptr[i + 1]
            tail = begin + ((end - begin) // 4) * 4
            for edge in range(begin, tail, 4):
                value0 = outgoing[edge]
                negative ^= value0 <= 0
                mag0 = min(1e100, abs(value0))
                larger0 = max(a0, mag0)
                a0 = min(a0, mag0)
                b0 = min(b0, larger0)
                value1 = outgoing[edge + 1]
                negative ^= value1 <= 0
                mag1 = min(1e100, abs(value1))
                larger1 = max(a1, mag1)
                a1 = min(a1, mag1)
                b1 = min(b1, larger1)
                value2 = outgoing[edge + 2]
                negative ^= value2 <= 0
                mag2 = min(1e100, abs(value2))
                larger2 = max(a2, mag2)
                a2 = min(a2, mag2)
                b2 = min(b2, larger2)
                value3 = outgoing[edge + 3]
                negative ^= value3 <= 0
                mag3 = min(1e100, abs(value3))
                larger3 = max(a3, mag3)
                a3 = min(a3, mag3)
                b3 = min(b3, larger3)
            smallest, second = 1e100, 1e100
            larger = max(smallest, a0)
            smallest = min(smallest, a0)
            second = min(second, larger)
            larger = max(smallest, b0)
            smallest = min(smallest, b0)
            second = min(second, larger)
            larger = max(smallest, a1)
            smallest = min(smallest, a1)
            second = min(second, larger)
            larger = max(smallest, b1)
            smallest = min(smallest, b1)
            second = min(second, larger)
            larger = max(smallest, a2)
            smallest = min(smallest, a2)
            second = min(second, larger)
            larger = max(smallest, b2)
            smallest = min(smallest, b2)
            second = min(second, larger)
            larger = max(smallest, a3)
            smallest = min(smallest, a3)
            second = min(second, larger)
            larger = max(smallest, b3)
            smallest = min(smallest, b3)
            second = min(second, larger)
            for edge in range(tail, end):
                value = outgoing[edge]
                negative ^= value <= 0
                magnitude = min(1e100, abs(value))
                larger = max(smallest, magnitude)
                smallest = min(smallest, magnitude)
                second = min(second, larger)
            for edge in range(begin, end):
                magnitude = second if abs(outgoing[edge]) == smallest else smallest
                incoming[edge] = -magnitude if negative ^ (outgoing[edge] <= 0) else magnitude
        collect = iteration >= max(0, warmup - round_length)
        # Sum through contiguous edge storage instead of chasing column lists.
        posterior[:] = effective
        for edge in range(edge_count):
            posterior[edge_var[edge]] += incoming[edge]
        for j in range(n):
            value = posterior[j]
            error[j] = value <= 0
            if collect:
                sums[j] += value
                minima[j] = min(minima[j], value)
                maxima[j] = max(maxima[j], value)
        for edge in range(edge_count):
            j = edge_var[edge]
            message = posterior[j] - incoming[edge]
            if abs(incoming[edge]) > 1e90:
                # Avoid losing the finite extrinsic LLR when a degree-one
                # check sends our finite representation of infinity.
                message = effective[j]
                for pos in range(col_ptr[j], col_ptr[j + 1]):
                    other = col_edges[pos]
                    if other != edge:
                        message += incoming[other]
            outgoing[edge] = message
            if collect:
                minima[j] = min(minima[j], message)
                maxima[j] = max(maxima[j], message)
        if _satisfies(error, syndrome, row_ptr, edge_var):
            return error, posterior, True, iteration + 1
    return error, posterior, False, budget


@njit(cache=True, parallel=True)
def _decode_unique(events, h, logical, prior, row_ptr, edge_var, col_ptr,
                   col_edges, warmup, round_length, max_rounds, phase_ends,
                   phases, osd_order, osd_width, seed):
    predictions = np.zeros((len(events), logical.shape[0]), dtype=np.uint8)
    iterations = np.zeros(len(events), dtype=np.int32)
    fallback = np.zeros(len(events), dtype=np.uint8)
    valid = np.ones(len(events), dtype=np.uint8)
    for shot in prange(len(events)):
        error, posterior, converged, count = _bp_one(
            events[shot], prior, row_ptr, edge_var, col_ptr, col_edges,
            warmup, round_length, max_rounds, phase_ends, phases, seed)
        iterations[shot] = count
        if not converged:
            error, repaired = osd_repair(h, events[shot], posterior, prior, osd_order, osd_width)
            fallback[shot] = 1
            valid[shot] = repaired
        for observable in range(logical.shape[0]):
            parity = 0
            for j in range(len(error)):
                parity ^= logical[observable, j] & error[j]
            predictions[shot, observable] = parity
    return predictions, iterations, fallback, valid


class PriorTiltDecoder:
    def __init__(self, circuit: stim.Circuit, config: DecoderConfig | None = None):
        self.config = config if config is not None else get_decoder_config(circuit)
        self.circuit = circuit.copy()
        # No graphlike decomposition: retain circuit-level correlated faults.
        self.model = model_from_dem(circuit.detector_error_model(approximate_disjoint_errors=True))

    def decode_with_diagnostics(self, detector_events):
        events = np.asarray(detector_events)
        if events.ndim != 2 or events.shape[1] != self.model.h.shape[0]:
            raise ValueError(f"Expected a (shots, {self.model.h.shape[0]}) detector array")
        if not np.all((events == 0) | (events == 1)):
            raise ValueError("Detector values must be binary")
        events = np.asarray(events, dtype=np.uint8) ^ self.model.detector_offset
        # Decode repeated syndromes once. No data or labels are stored between calls.
        packed = np.packbits(events, axis=1)
        if packed.shape[1]:
            # Binary row keys avoid NumPy's slower structured field-by-field
            # row sorting. Their byte order is identical to unsigned row order.
            keys = np.ascontiguousarray(packed).view(np.dtype((np.void, packed.shape[1]))).ravel()
            unique_keys, inverse = np.unique(keys, return_inverse=True)
            unique = unique_keys.view(np.uint8).reshape(-1, packed.shape[1])
        else:
            unique, inverse = np.unique(packed, axis=0, return_inverse=True)
        unique_events = np.unpackbits(unique, axis=1, count=events.shape[1]).copy()
        c, m = self.config, self.model
        predictions, iterations, fallback, valid = _decode_unique(
            unique_events, m.h, m.logical, m.prior, m.row_ptr, m.edge_var,
            m.col_ptr, m.col_edges, c.warmup_iterations, c.round_iterations,
            c.max_rounds, np.asarray(c.phase_ends, dtype=np.int32),
            np.asarray(c.phases, dtype=np.float64), c.osd_order, c.osd_width, c.seed)
        if not np.all(valid):
            raise ValueError("A detector syndrome is inconsistent with the circuit error model")
        result = predictions[inverse] ^ m.logical_offset
        return result, {"iterations": iterations[inverse],
                        "used_fallback": fallback[inverse].astype(bool),
                        "unique_syndromes": len(unique)}

    def decode_batch(self, detector_events):
        return self.decode_with_diagnostics(detector_events)[0]


@lru_cache(maxsize=8)
def _cached_decoder(circuit_text: str):
    return PriorTiltDecoder(stim.Circuit(circuit_text))


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """Hackathon interface: (shots, detectors) -> uint8 (shots, observables)."""
    from lut.frontend import decode_with_lookup

    return decode_with_lookup(circuit, detector_events, _cached_decoder, get_decoder_config)


# Frozen from independent validation before the primary holdout was decoded.
# Full precision is required: rounding phase parameters can change BP paths.
_TUNED_PARAMETERS = {'balanced': {'dc6d415ae3f0b5396a7f63ba1047f024be8ce7e44b03ace65348bcee78b3c070': {'warmup_iterations': 8,
                                                                                   'round_iterations': 4,
                                                                                   'max_rounds': 256,
                                                                                   'phase_ends': [18,
                                                                                                  145],
                                                                                   'phases': [[0.5299309467286255,
                                                                                               0.2558952904141021,
                                                                                               0.3853820477307444,
                                                                                               0.880688375277229],
                                                                                              [0.49099826214128056,
                                                                                               0.18700739612405162,
                                                                                               0.2502220172163462,
                                                                                               1.06608037045781],
                                                                                              [0.50999007,
                                                                                               0.1333403617138867,
                                                                                               0.1643728134130352,
                                                                                               1.0451415655478153]],
                                                                                   'osd_order': 2,
                                                                                   'osd_width': 12,
                                                                                   'seed': 20261113},
              '36a24a0347bbb82362af28d816d790d5ee1cb1543fef2e9af7ada70536bf9968': {'warmup_iterations': 8,
                                                                                   'round_iterations': 4,
                                                                                   'max_rounds': 256,
                                                                                   'phase_ends': [18,
                                                                                                  145],
                                                                                   'phases': [[0.5299309467286255,
                                                                                               0.2558952904141021,
                                                                                               0.3853820477307444,
                                                                                               0.880688375277229],
                                                                                              [0.49099826214128056,
                                                                                               0.18700739612405162,
                                                                                               0.2502220172163462,
                                                                                               1.06608037045781],
                                                                                              [0.50999007,
                                                                                               0.1333403617138867,
                                                                                               0.1643728134130352,
                                                                                               1.0451415655478153]],
                                                                                   'osd_order': 2,
                                                                                   'osd_width': 12,
                                                                                   'seed': 20261113}},
 'accuracy': {'dc6d415ae3f0b5396a7f63ba1047f024be8ce7e44b03ace65348bcee78b3c070': {'warmup_iterations': 8,
                                                                                   'round_iterations': 4,
                                                                                   'max_rounds': 384,
                                                                                   'phase_ends': [90,
                                                                                                  174],
                                                                                   'phases': [[0.4001895892810852,
                                                                                               0.3244911601846019,
                                                                                               0.38574560029858196,
                                                                                               0.44435068081876183],
                                                                                              [0.6312912317929403,
                                                                                               0.22415410888921514,
                                                                                               0.35428164116944383,
                                                                                               0.5087529710688727],
                                                                                              [0.40402560561710327,
                                                                                               0.1746818905149361,
                                                                                               0.2677011074916191,
                                                                                               0.7822957344693973]],
                                                                                   'osd_order': 2,
                                                                                   'osd_width': 192,
                                                                                   'seed': 20261113},
              '36a24a0347bbb82362af28d816d790d5ee1cb1543fef2e9af7ada70536bf9968': {'warmup_iterations': 8,
                                                                                   'round_iterations': 4,
                                                                                   'max_rounds': 384,
                                                                                   'phase_ends': [90,
                                                                                                  174],
                                                                                   'phases': [[0.4001895892810852,
                                                                                               0.3244911601846019,
                                                                                               0.38574560029858196,
                                                                                               0.44435068081876183],
                                                                                              [0.6312912317929403,
                                                                                               0.22415410888921514,
                                                                                               0.35428164116944383,
                                                                                               0.5087529710688727],
                                                                                              [0.40402560561710327,
                                                                                               0.1746818905149361,
                                                                                               0.2677011074916191,
                                                                                               0.7822957344693973]],
                                                                                   'osd_order': 2,
                                                                                   'osd_width': 192,
                                                                                   'seed': 20261113}}}

def get_decoder_config(circuit: stim.Circuit, profile: str = "balanced") -> DecoderConfig:
    """Return a validated preset for a recognized circuit, or generic settings.

    Routing uses the canonical public circuit only. Explicit configurations
    passed to PriorTiltDecoder override this automatic selection.
    """
    if not isinstance(profile, str) or profile not in _TUNED_PARAMETERS:
        raise ValueError("profile must be 'balanced' or 'accuracy'")
    signature = hashlib.sha256(str(circuit).encode()).hexdigest()
    parameters = _TUNED_PARAMETERS[profile].get(signature)
    if parameters is None:
        return DecoderConfig()
    values = dict(parameters)
    values["phase_ends"] = tuple(values["phase_ends"])
    values["phases"] = tuple(tuple(phase) for phase in values["phases"])
    return DecoderConfig(**values)
