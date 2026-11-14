"""Build the adaptive physical row bank in RAM during untimed startup.

This is the same selected-row arithmetic as the persisted lossless banks. It
uses no detector records, predictions, labels, external services or row files.
The caller must explicitly request construction; importing is inexpensive.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from adaptive_row_store import identity_for
from diagnostics.tensor_full_edge import (
    selected_channel, subtract_selected_retained,
    clip_tail_probability, clip_tail_signed, available_physical_bytes,
)


def compilation_bytes(count, bits=14):
    """Resident bank plus one channel and bounded projection/check buffers."""
    states = 1 << bits
    return int(count)*states*5*8+int(count)*states*8+(256 << 20)


def compile_selected_rows(edge_model, retained_paths, selected, *, bits=14,
                          reserve_bytes=0, available_bytes=None, progress=None):
    """Return immutable [selected,right,5] omitted rows with original arithmetic.

    Full production construction passes every left state in ascending order.
    A small selection permits bounded direct comparisons with saved rows.
    Only one contiguous channel is temporary; the probability channel already
    written into the resident bank bounds each subsequent signed channel.
    """
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)) or not 1 <= bits <= 14:
        raise ValueError("bits must be an integer from1 to14")
    raw = np.asarray(selected)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("selected must be a one-dimensional integer array")
    selected = np.ascontiguousarray(raw, np.int64)
    states = 1 << bits
    if np.any(selected < 0) or np.any(selected >= states) or len(np.unique(selected)) != len(selected):
        raise ValueError("selected states must be unique and within the physical domain")
    if isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, (int, np.integer)) or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be nonnegative")
    available = available_physical_bytes() if available_bytes is None else int(available_bytes)
    required = compilation_bytes(len(selected), bits)+int(reserve_bytes)
    if available < required:
        raise MemoryError(f"RAM row compilation needs{required} available bytes including reserve; available{available}")
    rows = np.empty((len(selected), states, 5), np.float64)
    for channel in range(5):
        started = time.perf_counter()
        bit = None if channel == 0 else channel-1
        values = selected_channel(*edge_model, bits, bits, selected, bit=bit)
        subtract_selected_retained(values, selected, retained_paths, bits, bit=bit)
        if channel == 0:
            defect = clip_tail_probability(values)
        else:
            defect = clip_tail_signed(values, rows[:, :, 0])
        # Bounded checks avoid a full-bank temporary boolean array.
        for first in range(0, len(selected), 64):
            if not np.all(np.isfinite(values[first:first+64])):
                raise ValueError("Physical row compilation produced nonfinite values")
        rows[:, :, channel] = values
        del values
        if progress:
            progress({"channel": "P" if bit is None else f"B{bit}",
                "rows": len(selected), "clipping_absolute_defect": float(defect),
                "channel_seconds": time.perf_counter()-started})
    rows.setflags(write=False)
    return rows


class CompiledMemoryRowStore:
    """Read-only all-state provider compatible with the resident contraction."""
    def __init__(self, rows, identity, construction_seconds, channel_metadata):
        states = 1 << int(identity["bits_per_side"])
        if (not isinstance(rows, np.ndarray) or rows.shape != (states, states, 5)
                or rows.dtype != np.float64 or not rows.flags.c_contiguous):
            raise ValueError("Expected a complete contiguous float64 physical row bank")
        rows.setflags(write=False)
        self.resident_rows = rows
        self.states, self.identity = states, identity
        self.resident_bytes = rows.nbytes
        self.preload_seconds = float(construction_seconds)
        self.memory_mode = "compiled_full_ram"
        self.directory = None
        self.artifact_paths = []
        self.metadata = {"identity": identity, "storage": self.memory_mode,
            "construction_seconds": self.preload_seconds,
            "channels": channel_metadata,
            "arithmetic": "selected_channel; original-order retained subtraction; probability/signed clipping; float64"}
        self.manifest = {"complete": True, "format": "startup-compiled-full-ram-v1",
                         "identity": identity, "metadata": self.metadata}
        self.manifest_sha256 = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()

    def _states(self, states):
        raw = np.asarray(states)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Expected an integer state vector")
        result = np.ascontiguousarray(raw, np.int64)
        if np.any(result < 0) or np.any(result >= self.states):
            raise ValueError("Physical row state is outside the bank")
        if self.resident_rows is None:
            raise ValueError("Compiled row bank is closed")
        return result

    def read_row(self, state):
        state = self._states([state])[0]
        return self.resident_rows[state]

    def read_row_into(self, state, output):
        value = self.read_row(state)
        if not isinstance(output, np.ndarray) or output.shape != value.shape or output.dtype != np.float64 or not output.flags.writeable:
            raise ValueError("Expected a writable float64[right,5] output")
        output[:] = value

    def get_rows(self, states):
        states = self._states(states)
        return np.ascontiguousarray(self.resident_rows[states])

    def verify_all_files(self):
        if self.resident_rows is None:
            raise ValueError("Compiled row bank is closed")
        return True

    def close(self):
        self.resident_rows = None


def compile_memory_store(circuit, edge_model, retained_paths, *, bits=14,
                         reserve_bytes=2 << 30, available_bytes=None, progress=None):
    """Explicit guarded full-bank construction; never called by module import."""
    channels = []
    def update(value):
        channels.append(value)
        if progress:
            progress(value)
    started = time.perf_counter()
    rows = compile_selected_rows(edge_model, retained_paths,
        np.arange(1 << bits, dtype=np.int64), bits=bits,
        reserve_bytes=reserve_bytes, available_bytes=available_bytes, progress=update)
    return CompiledMemoryRowStore(rows, identity_for(circuit, edge_model, retained_paths, bits),
        time.perf_counter()-started, channels)
