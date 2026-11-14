"""Circuit-bound temporal probability decoder with a harness-shaped entry point.

All fault orders within four temporal blocks are summed exactly. Each of the
three connecting edges includes every zero/one/two-fault event and three-fault
events containing at least two mechanisms with probability above 0.01. Thus
this sums a precisely defined retained submodel, not the complete channel.

The tables are derived from circuit probabilities, with no training labels or
BP predictions. The existing decoder.py entry point remains separate.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib

import numpy as np
import stim

from prior_model import PriorTiltDecoder
from diagnostics.temporal_chain import build_chain, decode_chain


_BASES = {
    "dc6d415ae3f0b5396a7f63ba1047f024be8ce7e44b03ace65348bcee78b3c070": "X",
    "36a24a0347bbb82362af28d816d790d5ee1cb1543fef2e9af7ada70536bf9968": "Z",
}


class ProbabilityDecoder:
    def __init__(self, circuit):
        text = str(circuit)
        signature = hashlib.sha256(text.encode()).hexdigest()
        if signature not in _BASES:
            raise ValueError("No temporal probability tables exist for this circuit")
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        self.chain = build_chain(PriorTiltDecoder(circuit), _BASES[signature],
                                 triple_min_high=2)

    def _events(self, detector_events):
        events = np.asarray(detector_events)
        if events.ndim != 2 or events.shape[1] != self.num_detectors:
            raise ValueError(f"Expected an array with {self.num_detectors} detector columns")
        if not np.all((events == 0) | (events == 1)):
            raise ValueError("Detector events must contain only zero or one")
        return np.asarray(events, dtype=np.uint8)

    def predict_proba(self, detector_events):
        """Return bit marginals under the explicitly retained circuit submodel."""
        events = self._events(detector_events)
        if not len(events):
            return np.empty((0, self.num_observables), dtype=np.float64)
        packed = np.ascontiguousarray(np.packbits(events, axis=1))
        keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        probabilities, details = decode_chain(self.chain, np.ascontiguousarray(events[first]))
        if not np.all(details["valid"]):
            raise ValueError("Syndrome has no supported temporal probability calculation")
        return probabilities[inverse]

    def decode(self, detector_events):
        """Minimize bitwise MAE under the retained submodel, including ties."""
        return (self.predict_proba(detector_events) > 0.5).astype(np.uint8)


@lru_cache(maxsize=2)
def _cached(circuit_text):
    return ProbabilityDecoder(stim.Circuit(circuit_text))


def get_decoder(circuit):
    return _cached(str(circuit))


def predict_proba(circuit, detector_events):
    return get_decoder(circuit).predict_proba(detector_events)


def decode(circuit, detector_events):
    return get_decoder(circuit).decode(detector_events)
