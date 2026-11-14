"""Experimental adaptive all-order row repair for the two supplied circuits.

The production decoder is unchanged. This method ranks omitted edge input
states using each syndrome's retained forward messages. It computes full
all-order row distributions for those states and contracts every combination
of the repaired three edges. Unselected rows still use the retained model.

Rows are compiled from circuit probabilities only. There are no labels,
syndrome lookup answers, case IDs, teacher files, or network dependencies.
Complete row banks can be precomputed once and loaded losslessly. Without a
row bank, missing rows are compiled on demand. Population accuracy is measured
separately from implementation equivalence and cache performance.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import time

import numpy as np
import stim

from prior_model import PriorTiltDecoder
from diagnostics.temporal_chain import build_chain
from diagnostics.tensor_environment_tail import five_environments, omitted_row_masses
from diagnostics.tensor_fast_chain import build_fast_chain, decode_fast_chain
from diagnostics.tensor_full_edge import (
    selected_channel, subtract_selected_retained, clip_tail_probability,
    clip_tail_signed,
)
from diagnostics.tensor_sliced_chain import build_sliced_chain, decode_sliced_chain
from diagnostics.tensor_structure import edge_mechanisms


_BASES = {
    "dc6d415ae3f0b5396a7f63ba1047f024be8ce7e44b03ace65348bcee78b3c070": "X",
    "36a24a0347bbb82362af28d816d790d5ee1cb1543fef2e9af7ada70536bf9968": "Z",
}


def _build_physics(circuit):
    signature = hashlib.sha256(str(circuit).encode()).hexdigest()
    if signature not in _BASES:
        raise ValueError("No adaptive temporal model exists for this circuit")
    core = PriorTiltDecoder(circuit)
    chain = build_chain(core, _BASES[signature], triple_min_high=2)
    fast = build_fast_chain(chain, aggregate=True, fused_terminal=True)
    a, b, logical, probabilities, _ = edge_mechanisms(core)
    left, right = ((value.T @ (1 << np.arange(fast["bits"]))).astype(np.uint16)
                   for value in (a, b))
    labels = (logical.T @ np.asarray([1, 2, 4, 8])).astype(np.uint8)
    paths = chain["edge"]
    row_mass, numerics = omitted_row_masses(left, probabilities, paths[0], paths[3], fast["bits"])
    return fast, (left, right, labels, probabilities), paths, row_mass, numerics


class AdaptiveTensorDecoder:
    """Bounded-memory experimental row repair, with a per-instance LRU cache.

    ``uncertainty_gate=0`` repairs all syndromes. The default0.4 repairs shots
    having at least one retained marginal in[0.4,0.6]. Precomputed rows share
    a bounded workspace; full resident banks contract all chosen shots in
    parallel. On-demand compilation keeps the original eight-shot bound.
    A supplied-circuit row occupies655360bytes.
    """

    def __init__(self, circuit, *, rows_per_edge=16, uncertainty_gate=.4,
                 batch_size=8, cache_rows=512, row_store=None,
                 use_cached_kernels=True, read_workers=4,
                 pipeline=True, workspace_rows=512, cache_policy="lookahead",
                 first_cache_rows=0, last_cache_rows=0, partial_mode="strict_first",
                 retained_backend="cpu", cuda_batch_size=128, adaptive_refinement=True):
        for value, name, lower, upper in ((rows_per_edge, "rows_per_edge", 1, 32),
                (batch_size, "batch_size", 1, 512), (cache_rows, "cache_rows", 0, 512),
                (read_workers, "read_workers", 1, 4), (workspace_rows, "workspace_rows", 96, 2048)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in[{lower},{upper}]")
        if (not isinstance(uncertainty_gate, (int, float, np.integer, np.floating))
                or not np.isfinite(uncertainty_gate) or not 0 <= uncertainty_gate <= .5):
            raise ValueError("uncertainty_gate must be finite and in[0,.5]")
        self.rows_per_edge = int(rows_per_edge)
        self.uncertainty_gate = float(uncertainty_gate)
        self.batch_size = int(batch_size)
        self.cache_rows = int(cache_rows)
        self.read_workers = int(read_workers)
        if not isinstance(pipeline, (bool, np.bool_)):
            raise ValueError("pipeline must be boolean")
        self.pipeline = bool(pipeline)
        if not isinstance(adaptive_refinement, (bool, np.bool_)):
            raise ValueError("adaptive_refinement must be boolean")
        self.adaptive_refinement = bool(adaptive_refinement)
        self.workspace_rows = max(int(workspace_rows), self.cache_rows)
        if cache_policy not in ("lru", "lookahead"):
            raise ValueError("cache_policy must be lru or lookahead")
        self.cache_policy = cache_policy
        self._workspace = None
        self._slot_for_state = {}
        self._read_pool = None
        if retained_backend not in ("cpu", "cuda"):
            raise ValueError("retained_backend must be cpu or cuda")
        if retained_backend == "cuda" and (first_cache_rows or last_cache_rows):
            raise ValueError("Endpoint-message caches belong to the CPU retained backend")
        self.retained_backend = retained_backend
        self._retained_gpu = None
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        if self.num_observables != 4:
            raise ValueError("Adaptive row repair requires four observables")
        self.fast, self._edge_model, self._retained_paths, self._row_mass, self.row_mass_numerics = _build_physics(circuit)
        if first_cache_rows or last_cache_rows:
            from diagnostics.tensor_partial_cache import prepare_partial_cache
            prepare_partial_cache(self.fast, first_entries=first_cache_rows,
                last_entries=last_cache_rows, mode=partial_mode,
                guard_thresholds=(self.uncertainty_gate, .5, 1-self.uncertainty_gate))
        self._cache = OrderedDict()
        if not isinstance(use_cached_kernels, (bool, np.bool_)):
            raise ValueError("use_cached_kernels must be boolean")
        self.use_cached_kernels = bool(use_cached_kernels)
        self._row_store = self._resolve_row_store(circuit, row_store) if self.adaptive_refinement else None
        if retained_backend == "cuda":
            from diagnostics.tensor_cuda_chain import CudaRetainedDecoder
            self._retained_gpu = CudaRetainedDecoder(self.fast, batch_size=cuda_batch_size)

    def _resolve_row_store(self, circuit, row_store):
        """Bind persisted physics to this exact circuit and retained operator."""
        if row_store is False:
            return None
        from pathlib import Path

        if row_store is None:
            signature = hashlib.sha256(str(circuit).encode()).hexdigest()
            basis = _BASES.get(signature)
            if basis is None:
                return None
            path = Path(__file__).resolve().parent/f"data/target03/adaptive_rows_{basis}"
            manifest = path/"manifest.json"
            if not manifest.exists():
                return None
            import json
            if not json.loads(manifest.read_text(encoding="utf8")).get("complete", False):
                return None
            row_store = path
        from adaptive_row_store import AdaptiveRowStore, identity_for

        expected = identity_for(circuit, self._edge_model, self._retained_paths,
                                bits=self.fast["bits"])
        if isinstance(row_store, (str, Path)):
            return AdaptiveRowStore(row_store, expected_identity=expected)
        if not hasattr(row_store, "get_rows") or not hasattr(row_store, "identity"):
            raise ValueError("row_store must be a compatible reader, directory, or False")
        if row_store.identity != expected:
            raise ValueError("Precomputed rows do not match this circuit and retained operator")
        return row_store

    @property
    def row_store(self):
        return self._row_store

    @property
    def cached_rows(self):
        return len(self._cache)

    @property
    def cached_bytes(self):
        return sum(value.nbytes for value in self._cache.values())

    def clear_cache(self):
        self._cache.clear()
        self._slot_for_state.clear()

    def close(self):
        """Release worker threads and readers after the last inference call."""
        if self._read_pool is not None:
            self._read_pool.shutdown(wait=True)
            self._read_pool = None
        self.clear_cache()
        self._workspace = None
        if self._row_store is not None:
            self._row_store.close()
        if self._retained_gpu is not None:
            self._retained_gpu.close()
            self._retained_gpu = None

    def _fill_workspace_row(self, item):
        state, slot = item
        target = self._workspace[slot]
        if hasattr(self._row_store, "read_row_into"):
            self._row_store.read_row_into(state, target)
        else:
            target[:] = self._row_store.read_row(state)

    def _shared_row_bank(self, selected, next_use=None):
        """Keep cached rows in one arena; contractions use slot indices directly.

        Required rows are pinned until this batch finishes. Rows evicted from
        the logical LRU remain valid in their slots until the next batch, so a
        batch may safely use more rows than the persistent cache limit.
        """
        if len(selected) > self.workspace_rows:
            raise ValueError("Selected rows exceed the bounded row workspace")
        if self._workspace is None:
            self._workspace = np.empty((self.workspace_rows, 1 << self.fast["bits"], 5))
        if set(self._slot_for_state) != set(self._cache):
            # A caller may have switched between legacy and pipeline execution.
            self.clear_cache()
        required = set(map(int, selected))
        slots = {}
        hits = 0
        for state in selected:
            state = int(state)
            if state in self._cache:
                slots[state] = self._slot_for_state[state]
                self._cache.move_to_end(state)
                hits += 1
        occupied = set(self._slot_for_state.values())
        free = [slot for slot in range(self.workspace_rows) if slot not in occupied]
        victims = None
        if next_use is not None:
            import heapq
            victims = [(-int(next_use[state]), state) for state in self._cache if state not in required]
            heapq.heapify(victims)
        work = []
        for state in selected:
            state = int(state)
            if state in slots:
                continue
            if free:
                slot = free.pop()
            else:
                victim = (next(key for key in self._cache if key not in required)
                          if victims is None else heapq.heappop(victims)[1])
                slot = self._slot_for_state.pop(victim)
                del self._cache[victim]
            slots[state] = slot
            work.append((state, slot))
        if self.read_workers > 1 and len(work) > 1:
            from concurrent.futures import ThreadPoolExecutor, wait
            if self._read_pool is None:
                self._read_pool = ThreadPoolExecutor(max_workers=self.read_workers)
            futures = [self._read_pool.submit(self._fill_workspace_row, item) for item in work]
            wait(futures)  # No writes may outlive a failed batch and reuse its slots.
            for future in futures:
                future.result()
        else:
            for item in work:
                self._fill_workspace_row(item)
        for state, slot in work:
            self._slot_for_state[state] = slot
            self._cache[state] = self._workspace[slot]
        while len(self._cache) > self.cache_rows:
            if next_use is None:
                state, _ = self._cache.popitem(last=False)
            else:
                state = max(self._cache, key=lambda key: int(next_use[key]))
                del self._cache[state]
            self._slot_for_state.pop(state)
        positions = np.asarray([slots[int(state)] for state in selected], np.int64)
        return self._workspace, positions, hits, len(work), np.zeros(5)

    def _events(self, detector_events):
        events = np.asarray(detector_events)
        if (events.ndim != 2 or events.shape[1] != self.num_detectors
                or not np.all((events == 0) | (events == 1))):
            raise ValueError(f"Expected binary detector events[n,{self.num_detectors}]")
        return np.asarray(events, dtype=np.uint8)

    def _select_rows(self, source):
        forward, _, _, valid = five_environments(source, self.fast["first"], self.fast["last"],
            self.fast["middle_hat"], self.fast["left"], self.fast["right"], self.fast["coefficients"],
            self.fast["bits"], self.fast["first_prefix"])
        if not valid:
            raise ValueError("Retained temporal environments are invalid")
        # Backward.max()/retained_normalizer is constant within each edge.
        # Including it in a row-mass upper bound leaves this per-edge ordering unchanged.
        score = forward[:, :, 0] * self._row_mass[None]
        ranked = np.argsort(-score, axis=1, kind="stable")
        return np.unique(ranked[:, :min(self.rows_per_edge, score.shape[1])])

    def _row_bank(self, selected):
        """Compile missing rows; cached arrays own their storage after eviction."""
        n = 1 << self.fast["bits"]
        bank = np.empty((len(selected), n, 5))
        missing, positions = [], []
        hits = 0
        for index, state in enumerate(selected):
            state = int(state)
            if state in self._cache:
                bank[index] = self._cache[state]
                self._cache.move_to_end(state)
                hits += 1
            else:
                missing.append(state); positions.append(index)
        ledger = np.zeros(5)
        if missing:
            missing = np.asarray(missing, np.int64)
            positions = np.asarray(positions, np.int64)
            if self._row_store is not None:
                if (self.read_workers > 1 and len(missing) > 1
                        and hasattr(self._row_store, "read_row")):
                    from concurrent.futures import ThreadPoolExecutor
                    # The immutable reader locks only each file's short seek/read.
                    # Decompression, unshuffle and checksum work can overlap.
                    with ThreadPoolExecutor(max_workers=self.read_workers) as workers:
                        rows = workers.map(self._row_store.read_row, map(int, missing))
                        for position, row in zip(positions, rows):
                            if row.shape != (n, 5) or row.dtype != np.float64:
                                raise ValueError("Precomputed row dimensions or precision disagree")
                            bank[position] = row
                else:
                    values = self._row_store.get_rows(missing)
                    if values.shape != (len(missing), n, 5) or values.dtype != np.float64:
                        raise ValueError("Precomputed row dimensions or precision disagree")
                    bank[positions] = values
            else:
                bits = self.fast["bits"]
                left, right, labels, probabilities = self._edge_model
                for channel in range(5):
                    values = selected_channel(left, right, labels, probabilities, bits, bits, missing,
                        bit=None if channel == 0 else channel-1)
                    subtract_selected_retained(values, missing, self._retained_paths, bits,
                        None if channel == 0 else channel-1)
                    if channel == 0:
                        ledger[channel] = clip_tail_probability(values)
                    else:
                        ledger[channel] = clip_tail_signed(values, bank[positions, :, 0])
                    bank[positions, :, channel] = values
            if self.cache_rows:
                for index, state in zip(positions, missing):
                    # A view would retain the entire compile batch after LRU eviction.
                    self._cache[int(state)] = bank[index].copy(order="C")
                    self._cache.move_to_end(int(state))
                    while len(self._cache) > self.cache_rows:
                        self._cache.popitem(last=False)
        return bank, hits, len(missing), ledger

    def _pipeline_refine(self, packed, probability, chosen, counts):
        """Select independently in parallel, then contract shared immutable rows."""
        from diagnostics.adaptive_cached_kernels import select_rows_sources, contract_selected_sources

        stage = {"selection": 0., "row_loading": 0., "contraction": 0.}
        if not len(chosen):
            return 0, 0, np.zeros(5), stage
        started = time.perf_counter()
        sources = np.ascontiguousarray(packed[chosen])
        states, active_counts, valid = select_rows_sources(
            self.fast, sources, self._row_mass, self.rows_per_edge)
        if not np.all(valid):
            raise ValueError("Retained temporal environments are invalid")
        counts[chosen] = active_counts
        stage["selection"] = time.perf_counter()-started
        resident = getattr(self._row_store, "resident_rows", None)
        if resident is not None:
            from diagnostics.adaptive_cached_kernels import contract_resident_sources
            started = time.perf_counter()
            q, valid, _ = contract_resident_sources(self.fast, sources, states, active_counts, resident)
            if not np.all(valid):
                raise ValueError("Adaptive repaired contraction is invalid")
            probability[chosen] = q
            stage["contraction"] = time.perf_counter()-started
            return 0, 0, np.zeros(5), stage
        groups = []
        start = 0
        while start < len(chosen):
            stop = start
            required = set()
            while stop < min(start+self.batch_size, len(chosen)):
                next_states = set(map(int, states[stop, :active_counts[stop]]))
                if len(required | next_states) > self.workspace_rows:
                    break
                required.update(next_states)
                stop += 1
            if stop == start:
                raise ValueError("One syndrome exceeds the bounded row workspace")
            selected = np.asarray(sorted(required), np.int64)
            groups.append((start, stop, selected))
            start = stop
        next_use = None
        future, cursors = {}, {}
        if self.cache_policy == "lookahead":
            # Selection has already inspected the current call's detector data.
            # Knowing each physical row's next use lets us evict the row needed
            # furthest in the future, without caching any syndrome answers.
            next_use = np.full(1 << self.fast["bits"], len(groups), np.int64)
            for group, (_, _, selected) in enumerate(groups):
                for state in selected:
                    future.setdefault(int(state), []).append(group)
            for state, uses in future.items():
                cursors[state] = 0
                next_use[state] = uses[0]
        fetched, hits = 0, 0
        ledger = np.zeros(5)
        for group, (start, stop, selected) in enumerate(groups):
            if next_use is not None:
                for state in selected:
                    state = int(state)
                    cursor = cursors[state]+1
                    cursors[state] = cursor
                    next_use[state] = future[state][cursor] if cursor < len(future[state]) else len(groups)
            started = time.perf_counter()
            bank, bank_slots, batch_hits, batch_fetched, defects = self._shared_row_bank(selected, next_use)
            stage["row_loading"] += time.perf_counter()-started
            positions = bank_slots[np.searchsorted(selected, states[start:stop])]
            started = time.perf_counter()
            q, valid, _ = contract_selected_sources(self.fast, sources[start:stop],
                states[start:stop], active_counts[start:stop], bank, positions)
            if not np.all(valid):
                raise ValueError("Adaptive repaired contraction is invalid")
            probability[chosen[start:stop]] = q
            stage["contraction"] += time.perf_counter()-started
            hits += batch_hits
            fetched += batch_fetched
            ledger += defects
        return fetched, hits, ledger, stage

    def refine_probabilities(self, detector_events, reference_probabilities):
        """Refine already computed retained marginals without recomputing them."""
        events = self._events(detector_events)
        reference = np.asarray(reference_probabilities, dtype=np.float64)
        if (reference.shape != (len(events), 4) or np.any(~np.isfinite(reference))
                or np.any((reference < 0) | (reference > 1))):
            raise ValueError("Expected finite reference probabilities[n,4]")
        probability = reference.copy()
        gate = np.max(np.minimum(probability, 1-probability), axis=1) >= self.uncertainty_gate
        chosen = np.flatnonzero(gate)
        normalized = events ^ self.fast["detector_offset"]
        packed = np.column_stack([normalized[:, region] @ (1 << np.arange(len(region)))
            for region in self.fast["regions"]]).astype(np.int64)
        counts = np.zeros(len(events), np.int64)
        fetched, hits = 0, 0
        ledger = np.zeros(5)
        if self.pipeline and self.use_cached_kernels and self._row_store is not None:
            fetched, hits, ledger, stage = self._pipeline_refine(packed, probability, chosen, counts)
            return probability, {"refined_rows": int(gate.sum()),
                "unique_refined_syndromes": len(chosen), "compiled_rows": 0,
                "precomputed_rows_loaded": fetched, "cache_hits": hits,
                "cached_rows": self.cached_rows, "cached_bytes": self.cached_bytes,
                "selected_rows": counts, "refinement_mask": gate, "clipping_ledger": ledger,
                "precomputed_store": True, "cached_kernels": True, "pipeline": True,
                "cache_policy": self.cache_policy,
                "resident_rows": getattr(self._row_store, "resident_rows", None) is not None,
                "workspace_bytes": 0 if self._workspace is None else self._workspace.nbytes,
                "stage_seconds": stage}
        # On-demand compilation retains the original bounded eight-shot policy.
        legacy_batch = min(self.batch_size, 8)
        for start in range(0, len(chosen), legacy_batch):
            active = chosen[start:start+legacy_batch]
            if self.use_cached_kernels:
                from diagnostics.adaptive_cached_kernels import select_rows_sources, contract_selected_sources
                states, active_counts, valid = select_rows_sources(
                    self.fast, packed[active], self._row_mass, self.rows_per_edge)
                if not np.all(valid):
                    raise ValueError("Retained temporal environments are invalid")
                selected = np.unique(states[states >= 0])
                bank, batch_hits, batch_fetched, defects = self._row_bank(selected)
                positions = np.searchsorted(selected, states)
                q, valid, _ = contract_selected_sources(
                    self.fast, packed[active], states, active_counts, bank, positions)
                if not np.all(valid):
                    raise ValueError("Adaptive repaired contraction is invalid")
                probability[active] = q
                counts[active] = active_counts
            else:
                choices = [self._select_rows(packed[row]) for row in active]
                selected = np.unique(np.concatenate(choices))
                bank, batch_hits, batch_fetched, defects = self._row_bank(selected)
                for row, states in zip(active, choices):
                    positions = np.searchsorted(selected, states)
                    repaired = build_sliced_chain(self.fast, states, bank[positions], tail_layout="KN5")
                    q, info = decode_sliced_chain(repaired, events[row:row+1],
                        maximum_rows=len(states), minimum_uncertainty=0.,
                        reference_probabilities=probability[row:row+1], deduplicate=False)
                    if not np.all(info["valid"]):
                        raise ValueError("Adaptive repaired contraction is invalid")
                    probability[row] = q[0]
                    counts[row] = len(states)
            hits += batch_hits
            fetched += batch_fetched
            ledger += defects
        return probability, {"refined_rows": int(gate.sum()),
            "unique_refined_syndromes": len(chosen),
            "compiled_rows": fetched if self._row_store is None else 0,
            "precomputed_rows_loaded": fetched if self._row_store is not None else 0,
            "cache_hits": hits, "cached_rows": self.cached_rows,
            "cached_bytes": self.cached_bytes, "selected_rows": counts,
            "refinement_mask": gate, "clipping_ledger": ledger,
            "precomputed_store": self._row_store is not None,
            "cached_kernels": self.use_cached_kernels}

    def predict_with_diagnostics(self, detector_events):
        events = self._events(detector_events)
        if not len(events):
            return np.empty((0, 4)), {"refined_rows": 0, "unique_refined_syndromes": 0,
                "compiled_rows": 0, "precomputed_rows_loaded": 0,
                "cache_hits": 0, "cached_rows": self.cached_rows,
                "cached_bytes": self.cached_bytes, "selected_rows": np.empty(0, np.int64),
                "refinement_mask": np.empty(0, bool), "clipping_ledger": np.zeros(5),
                "precomputed_store": self._row_store is not None,
                "cached_kernels": self.use_cached_kernels}
        keys = np.ascontiguousarray(np.packbits(events, axis=1))
        keys = keys.view(np.dtype((np.void, keys.shape[1]))).ravel()
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        unique_events = np.ascontiguousarray(events[first])
        if self._retained_gpu is None:
            probability, details = decode_fast_chain(self.fast, unique_events, deduplicate=False)
        else:
            probability, details = self._retained_gpu.predict_with_diagnostics(unique_events)
        if not np.all(details["valid"]):
            raise ValueError("Retained temporal contraction is invalid")
        if self.adaptive_refinement:
            probability, diagnostics = self.refine_probabilities(unique_events, probability)
        else:
            diagnostics = {"refined_rows": 0, "unique_refined_syndromes": 0,
                "compiled_rows": 0, "precomputed_rows_loaded": 0, "cache_hits": 0,
                "cached_rows": 0, "cached_bytes": 0,
                "selected_rows": np.zeros(len(unique_events), np.int64),
                "refinement_mask": np.zeros(len(unique_events), bool),
                "clipping_ledger": np.zeros(5), "precomputed_store": False,
                "cached_kernels": self.use_cached_kernels}
        diagnostics["selected_rows"] = diagnostics["selected_rows"][inverse]
        diagnostics["refinement_mask"] = diagnostics["refinement_mask"][inverse]
        diagnostics["refined_rows"] = int(diagnostics["refinement_mask"].sum())
        diagnostics["retained_backend"] = self.retained_backend
        diagnostics["adaptive_refinement"] = self.adaptive_refinement
        for key, value in details.items():
            if key.startswith("partial_cache_"):
                diagnostics[key] = value
        partial = self.fast.get("partial_cache")
        if partial is not None:
            diagnostics["partial_cache_first_entries"] = len(partial["first_keys"])
            diagnostics["partial_cache_last_entries"] = len(partial["last_keys"])
        if self._retained_gpu is not None:
            diagnostics["retained_cuda_timing"] = details["timing"]
        return probability[inverse], diagnostics

    def predict_proba(self, detector_events):
        return self.predict_with_diagnostics(detector_events)[0]

    def decode(self, detector_events):
        return (self.predict_proba(detector_events) > .5).astype(np.uint8)


@lru_cache(maxsize=2)
def _decoder(circuit_text):
    return AdaptiveTensorDecoder(stim.Circuit(circuit_text))


def get_decoder(circuit):
    """Return the separately selectable experimental default profile."""
    return _decoder(str(circuit))


def predict_proba(circuit, detector_events):
    return get_decoder(circuit).predict_proba(detector_events)


def decode(circuit, detector_events):
    return get_decoder(circuit).decode(detector_events)
