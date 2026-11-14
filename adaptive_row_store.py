"""Lossless, circuit-bound storage of every adaptive omitted-edge row.

Each channel has independently compressed rows for bounded random reads.
Precomputation uses the existing selected-row arithmetic and one disk-backed
channel at a time. A building manifest is never accepted by the runtime.
Completed channels survive interruption and are verified before resuming.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
import zlib

import numpy as np

from diagnostics.tensor_full_edge import (
    _selected_projection_block, _inverse_selected_right,
    subtract_selected_retained, clip_tail_probability, clip_tail_signed,
    available_physical_bytes, file_sha256,
)


CHANNELS = ("P", "B0", "B1", "B2", "B3")
FORMAT_VERSION = 1
RETAINED_RULE = "fault_count<=2 or (fault_count==3 and count(p>0.01)>=2), independently at each edge"
ROOT = Path(__file__).resolve().parent


def _array_hash(arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def identity_for(circuit, edge_model, retained_paths, bits=14):
    edge = tuple(np.asarray(value, dtype=dtype) for value, dtype in zip(edge_model,
        (np.uint16, np.uint16, np.uint8, np.float64)))
    paths = tuple(np.asarray(value, dtype=dtype) for value, dtype in zip(retained_paths,
        (np.uint16, np.uint16, np.uint8, np.float64)))
    return {"circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
        "edge_model_sha256": _array_hash(edge), "retained_paths_sha256": _array_hash(paths),
        "bits_per_side": int(bits), "retained_rule": RETAINED_RULE,
        "channels": list(CHANNELS), "storage_dtype": "<f8"}


def circuit_physics(circuit):
    """Reconstruct runtime edge order and its exactly ordered retained paths."""
    from prior_model import PriorTiltDecoder
    from diagnostics.tensor_structure import edge_mechanisms
    from diagnostics.temporal_chain import sparse_edge
    core = PriorTiltDecoder(circuit)
    a, b, logical, probabilities, _ = edge_mechanisms(core)
    bits = len(a)
    left, right = ((value.T @ (1 << np.arange(bits))).astype(np.uint16) for value in (a, b))
    labels = (logical.T @ np.asarray([1, 2, 4, 8])).astype(np.uint8)
    edge = (left, right, labels, probabilities.copy())
    order = np.lexsort((labels, right, left))
    paths = sparse_edge(left[order], right[order], labels[order], probabilities[order], triple_min_high=2)
    model_hash = hashlib.sha256(core.model.h.tobytes()+core.model.logical.tobytes()+core.model.probabilities.tobytes()).hexdigest()
    return edge, paths, {"model_arrays_sha256": model_hash, "bits_per_side": bits}


def _atomic_json(path, value):
    temporary = path.with_name(path.name+".writing")
    with temporary.open("w", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    temporary = path.with_name(path.name+".writing")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _local_path(directory, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Store filenames must stay inside their directory")
    result = (directory/name).resolve()
    if result.parent != directory.resolve():
        raise ValueError("Store path leaves its directory")
    return result


def _encode_row(row, level):
    values = np.ascontiguousarray(row, dtype="<f8")
    raw = values.tobytes()
    shuffled = np.ascontiguousarray(values.view(np.uint8).reshape(-1, 8).T).tobytes()
    return zlib.compress(shuffled, level), hashlib.sha256(raw).digest(), raw


def _decode_row(payload, checksum, states):
    try:
        shuffled = zlib.decompress(payload)
    except zlib.error as error:
        raise ValueError("Invalid compressed adaptive row") from error
    if len(shuffled) != states*8:
        raise ValueError("Compressed adaptive row has the wrong length")
    # Hash the contiguous unshuffled buffer directly. Materializing bytes here
    # made a second128KiB copy for every channel of every cache miss.
    raw = np.ascontiguousarray(np.frombuffer(shuffled, np.uint8).reshape(8, states).T)
    if hashlib.sha256(memoryview(raw).cast("B")).digest() != bytes(checksum):
        raise ValueError("Adaptive row checksum mismatch")
    values = raw.view("<f8").reshape(states)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite adaptive row")
    values.setflags(write=False)
    return values


class _ChannelReader:
    def __init__(self, directory, info, states, verify_payload=False):
        self.directory = Path(directory).resolve()
        self.info = info
        self.states = states
        index_path = _local_path(self.directory, info["index_file"])
        data_path = _local_path(self.directory, info["data_file"])
        if file_sha256(index_path) != info["index_sha256"]:
            raise ValueError("Adaptive row index checksum mismatch")
        with np.load(index_path, allow_pickle=False) as archive:
            self.offsets = archive["offsets"].copy()
            self.checksums = archive["raw_row_sha256"].copy()
        if (self.offsets.shape != (states+1,) or self.offsets.dtype != np.uint64
                or self.checksums.shape != (states, 32) or self.checksums.dtype != np.uint8
                or self.offsets[0] != 0 or np.any(self.offsets[1:] <= self.offsets[:-1])
                or int(self.offsets[-1]) != data_path.stat().st_size):
            raise ValueError("Malformed adaptive row index or truncated payload")
        if verify_payload and file_sha256(data_path) != info["data_sha256"]:
            raise ValueError("Adaptive row payload checksum mismatch")
        self.path = data_path
        self.stream = data_path.open("rb")
        self.lock = threading.RLock()

    def read_row(self, state):
        first, last = int(self.offsets[state]), int(self.offsets[state+1])
        with self.lock:
            self.stream.seek(first)
            payload = self.stream.read(last-first)
        if len(payload) != last-first:
            raise ValueError("Truncated adaptive row payload")
        return _decode_row(payload, self.checksums[state], self.states)

    def close(self):
        self.stream.close()


class AdaptiveRowStore:
    def __init__(self, directory, expected_identity=None, *, verify_payload=False):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory/"manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if (self.manifest.get("format_version") != FORMAT_VERSION or self.manifest.get("status") != "complete"
                or not self.manifest.get("complete")):
            raise ValueError("Adaptive row store is incomplete or incompatible")
        self.identity = self.manifest["identity"]
        self.metadata = self.manifest["metadata"]
        if expected_identity is not None:
            for key, value in expected_identity.items():
                if self.identity.get(key) != value:
                    raise ValueError(f"Adaptive row store identity mismatch: {key}")
        bits = self.identity.get("bits_per_side")
        if not isinstance(bits, int) or not 1 <= bits <= 14 or self.identity.get("channels") != list(CHANNELS) or self.identity.get("storage_dtype") != "<f8":
            raise ValueError("Unsupported adaptive row dimensions")
        self.states = 1 << bits
        self.manifest_sha256 = file_sha256(manifest_path)
        self._channels = []
        try:
            for channel in CHANNELS:
                self._channels.append(_ChannelReader(self.directory, self.metadata["channels"][channel], self.states, verify_payload))
        except BaseException:
            self.close()
            raise
        paths = [manifest_path]
        for channel in CHANNELS:
            info = self.metadata["channels"][channel]
            paths.extend((_local_path(self.directory, info["data_file"]), _local_path(self.directory, info["index_file"])))
        self.artifact_paths = tuple(paths)

    def read_row(self, state):
        if isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer)) or not 0 <= state < self.states:
            raise ValueError("Adaptive row state is out of range")
        row = np.empty((self.states, 5))
        return self.read_row_into(state, row)

    def read_row_into(self, state, output):
        """Fill an existing row buffer, avoiding a temporary655KiB row copy."""
        if isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer)) or not 0 <= state < self.states:
            raise ValueError("Adaptive row state is out of range")
        if (not isinstance(output, np.ndarray) or output.shape != (self.states, 5)
                or output.dtype != np.float64 or not output.flags.writeable):
            raise ValueError("Expected a writable float64[state_count,5] row buffer")
        for channel, reader in enumerate(self._channels):
            output[:, channel] = reader.read_row(int(state))
        return output

    def get_rows(self, states):
        selected = np.asarray(states)
        if selected.ndim != 1 or (len(selected) and not np.issubdtype(selected.dtype, np.integer)) or np.any(selected < 0) or np.any(selected >= self.states):
            raise ValueError("Expected a vector of valid adaptive row states")
        result = np.empty((len(selected), self.states, 5))
        for index, state in enumerate(selected):
            self.read_row_into(int(state), result[index])
        return result

    def preload_all(self, *, workers=4, reserve_bytes=2 << 30, max_bytes=None,
                    close_source=False, progress=None):
        """Explicitly load verified rows into a read-only shared RAM bank.

        One supplied-circuit bank needs10GiB. Allocation is rejected before
        touching it unless available physical memory covers the bank, worker
        buffers and reserve. This method never runs automatically on import.
        Organizers can call it during import, before their decode timer starts.
        """
        if isinstance(workers, (bool, np.bool_)) or not isinstance(workers, (int, np.integer)) or not 1 <= workers <= 48:
            raise ValueError("workers must be an integer in[1,48]")
        if isinstance(reserve_bytes, (bool, np.bool_)) or not isinstance(reserve_bytes, (int, np.integer)) or reserve_bytes < 0:
            raise ValueError("reserve_bytes must be a nonnegative integer")
        if max_bytes is not None and (isinstance(max_bytes, (bool, np.bool_))
                or not isinstance(max_bytes, (int, np.integer)) or max_bytes < 0):
            raise ValueError("max_bytes must be a nonnegative integer")
        if any(reader.stream.closed for reader in self._channels):
            raise ValueError("Adaptive row store is closed")
        required = self.states*self.states*5*np.dtype(np.float64).itemsize
        if max_bytes is not None and required > max_bytes:
            raise MemoryError(f"Full adaptive row bank needs{required} bytes; budget is{max_bytes}")
        working = int(workers)*self.states*8*9+(32 << 20)
        available = available_physical_bytes()
        if available < required+int(reserve_bytes)+working:
            raise MemoryError(f"Full adaptive row bank needs{required+int(reserve_bytes)+working} available bytes including reserve; available{available}")
        start = time.perf_counter()
        rows = np.empty((self.states, self.states, 5), np.float64)
        def fill(state):
            self.read_row_into(state, rows[state])
        # Bounded chunks prevent thousands of pending futures retaining buffers.
        if workers > 1:
            with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                for first in range(0, self.states, 128):
                    stop = min(first+128, self.states)
                    for _ in pool.map(fill, range(first, stop)):
                        pass
                    if progress:
                        progress({"completed_rows": stop, "total_rows": self.states,
                            "resident_bytes": required})
        else:
            for state in range(self.states):
                fill(state)
                if progress and (state+1) % 128 == 0:
                    progress({"completed_rows": state+1, "total_rows": self.states,
                        "resident_bytes": required})
        rows.setflags(write=False)
        result = FullMemoryRowStore(self, rows, time.perf_counter()-start)
        if close_source:
            self.close()
        return result

    def verify_all_files(self):
        for reader in self._channels:
            if file_sha256(reader.path) != reader.info["data_sha256"]:
                raise ValueError("Adaptive row payload checksum mismatch")
        return True

    def close(self):
        for reader in getattr(self, "_channels", ()):
            reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FullMemoryRowStore:
    """Verified full-row RAM bank for direct state indexing by the kernels.

    ``resident_rows[state]`` is a read-only view. Kernels can use the entire
    ``resident_rows`` array and the original state numbers as bank positions,
    avoiding row decompression, LRU work and per-batch gathers entirely.
    """

    def __init__(self, source, rows, preload_seconds=0.):
        if (not isinstance(rows, np.ndarray)
                or rows.shape != (source.states, source.states, 5)
                or rows.dtype != np.float64 or not rows.flags.c_contiguous):
            raise ValueError("Expected a contiguous float64 full adaptive row bank")
        self.directory = source.directory
        self.manifest = source.manifest
        self.identity = source.identity
        self.metadata = source.metadata
        self.states = source.states
        self.manifest_sha256 = source.manifest_sha256
        self.artifact_paths = source.artifact_paths
        rows.setflags(write=False)
        self.resident_rows = rows
        self.preload_seconds = float(preload_seconds)
        self.memory_mode = "full_ram"

    @property
    def resident_bytes(self):
        return 0 if self.resident_rows is None else self.resident_rows.nbytes

    def _rows(self):
        if self.resident_rows is None:
            raise ValueError("Full-memory adaptive row store is closed")
        return self.resident_rows

    def read_row(self, state):
        if isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer)) or not 0 <= state < self.states:
            raise ValueError("Adaptive row state is out of range")
        return self._rows()[int(state)]

    def read_row_into(self, state, output):
        if (not isinstance(output, np.ndarray) or output.shape != (self.states, 5)
                or output.dtype != np.float64 or not output.flags.writeable):
            raise ValueError("Expected a writable float64[state_count,5] row buffer")
        output[:] = self.read_row(state)
        return output

    def get_rows(self, states):
        selected = np.asarray(states)
        if selected.ndim != 1 or (len(selected) and not np.issubdtype(selected.dtype, np.integer)) or np.any(selected < 0) or np.any(selected >= self.states):
            raise ValueError("Expected a vector of valid adaptive row states")
        # Compatibility path. Performance callers should pass resident_rows
        # directly to the kernel with positions equal to the requested states.
        return np.ascontiguousarray(self._rows()[selected.astype(np.intp, copy=False)])

    def verify_all_files(self):
        for channel in CHANNELS:
            info = self.metadata["channels"][channel]
            if file_sha256(_local_path(self.directory, info["data_file"])) != info["data_sha256"]:
                raise ValueError("Adaptive row payload checksum mismatch")
        return True

    def close(self):
        self.resident_rows = None

    def __enter__(self):
        self._rows()
        return self

    def __exit__(self, *_):
        self.close()


def _check_space(directory, needed, reserve):
    available = shutil.disk_usage(directory).free
    if available < needed+reserve:
        raise OSError(f"Adaptive precompute needs {needed+reserve} free bytes including reserve; available {available}")


def _compile_scratch(path, edge_model, bits, channel, progress=None):
    """Same selected_channel arithmetic, with the result backed by disk."""
    left, right, logical = (np.asarray(value, np.int64) for value in edge_model[:3])
    probabilities = np.asarray(edge_model[3], float)
    states = 1 << bits
    selected = np.arange(states, dtype=np.int64)
    logs = np.log1p(-2*probabilities)
    total = math.fsum(logs.tolist())
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(states, states))
    for start in range(0, states, 64):
        count = min(64, states-start)
        block = _selected_projection_block(left, right, logical, logs, total, bits,
            selected, start, count, -1 if channel == 0 else channel-1)
        values[:, start:start+count] = block.T
        if progress and (start+count) % 4096 == 0:
            progress("projection", start+count)
    # Independent rows use the identical serial Walsh sequence as selected_channel.
    for start in range(0, states, 128):
        _inverse_selected_right(values[start:start+128])
    values.flush()
    return values


def precompute_store(circuit, directory, *, threads=4, compression_level=6,
                     reserve_bytes=1 << 30, resume=False, progress=None,
                     edge_model=None, retained_paths=None, physics_metadata=None):
    """Write a complete lossless store, resuming at completed-channel boundaries.

    Interrupted active channels are recomputed. Only files created within the
    matching building store are replaced; previously complete stores are immutable.
    ``edge_model``/``retained_paths`` permit tiny independent fixtures in tests.
    """
    import numba
    if not isinstance(threads, int) or not 1 <= threads <= 4 or not 0 <= compression_level <= 9:
        raise ValueError("Need1..4 threads and compression level0..9")
    numba.set_num_threads(threads)
    if edge_model is None or retained_paths is None:
        if edge_model is not None or retained_paths is not None:
            raise ValueError("Provide both edge_model and retained_paths")
        edge_model, retained_paths, physics_metadata = circuit_physics(circuit)
    physics_metadata = {} if physics_metadata is None else dict(physics_metadata)
    bits = int(physics_metadata.get("bits_per_side", 14))
    identity = identity_for(circuit, edge_model, retained_paths, bits)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/"manifest.json"
    sources = [Path(__file__), ROOT/"diagnostics/tensor_full_edge.py", ROOT/"diagnostics/temporal_chain.py",
        ROOT/"diagnostics/tensor_structure.py", ROOT/"decoder.py", ROOT/"fallback.py"]
    hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in sources}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if manifest.get("identity") != identity or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Cannot resume an incompatible adaptive row store")
        if manifest.get("status") == "complete":
            with AdaptiveRowStore(directory, identity, verify_payload=True):
                return manifest
        if not resume:
            raise ValueError("Building store exists; explicitly resume it")
        if manifest["metadata"]["source_sha256"] != hashes:
            raise ValueError("Precompute sources changed; use a new store directory")
        if manifest["metadata"]["compression_level"] != compression_level:
            raise ValueError("Resume must preserve the compression level")
    else:
        if any(directory.iterdir()):
            raise ValueError("New adaptive store directory must be empty")
        manifest = {"format_version": FORMAT_VERSION, "status": "building", "complete": False, "identity": identity,
            "metadata": {"codec": "float64-byte-shuffle-zlib", "compression_level": compression_level,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "initial_free_disk_bytes": shutil.disk_usage(directory).free,
                "initial_available_physical_bytes": available_physical_bytes(),
                "source_sha256": hashes, "physics": physics_metadata, "channels": {},
                "raw_shape": [1 << bits, 1 << bits, 5], "threads": threads,
                "resume_scope": "Completed channels are retained; an interrupted active channel is rebuilt.",
                "roundoff_scope": "Identical selected_channel arithmetic and existing float64 clipping, not interval-certified."}}
        _atomic_json(manifest_path, manifest)
    states = 1 << bits
    scratch_path = _local_path(directory, "active_channel.scratch.npy")
    # One mapped channel plus bounded projection/compression blocks; never a2GiB heap allocation.
    _check_space(directory, states*states*8+4096, reserve_bytes)
    if available_physical_bytes() < (128 << 20):
        raise MemoryError("Need128MiB available RAM for bounded precompute buffers")
    start_time = time.perf_counter()
    probability_reader = None
    for channel, name in enumerate(CHANNELS):
        if name in manifest["metadata"]["channels"]:
            reader = _ChannelReader(directory, manifest["metadata"]["channels"][name], states, True)
            reader.close()
            continue
        data_path = _local_path(directory, f"{name}.rows.zlib")
        index_path = _local_path(directory, f"{name}.index.npz")
        # These names belong exclusively to the matching incomplete store.
        for unfinished in (scratch_path, data_path, index_path):
            if unfinished.exists():
                unfinished.unlink()
        manifest["active_channel"] = name
        manifest["active_stage"] = "compiling"
        _atomic_json(manifest_path, manifest)
        elapsed = time.perf_counter()
        def stage(label, count):
            if progress:
                progress({"channel": name, "stage": label, "completed": count, "states": states})
        scratch = _compile_scratch(scratch_path, edge_model, bits, channel, stage)
        subtract_selected_retained(scratch, np.arange(states), retained_paths, bits,
            None if channel == 0 else channel-1)
        defect = 0.
        if channel == 0:
            defect = float(clip_tail_probability(scratch))
        else:
            if probability_reader is None:
                probability_reader = _ChannelReader(directory, manifest["metadata"]["channels"]["P"], states)
            for first in range(0, states, 64):
                stop = min(first+64, states)
                probability = np.asarray([probability_reader.read_row(row) for row in range(first, stop)])
                defect += float(clip_tail_signed(scratch[first:stop], probability))
        scratch.flush()
        manifest["active_stage"] = "compressing"
        _atomic_json(manifest_path, manifest)
        offsets = np.zeros(states+1, np.uint64)
        checksums = np.empty((states, 32), np.uint8)
        raw_digest = hashlib.sha256()
        payload_digest = hashlib.sha256()
        with data_path.open("wb") as stream, ThreadPoolExecutor(max_workers=threads) as workers:
            for first in range(0, states, 64):
                stop = min(first+64, states)
                _check_space(directory, (stop-first)*states*8+4096, reserve_bytes)
                encoded = workers.map(lambda row: _encode_row(row, compression_level), scratch[first:stop])
                for row, (payload, checksum, raw) in zip(range(first, stop), encoded):
                    stream.write(payload)
                    offsets[row+1] = offsets[row]+len(payload)
                    checksums[row] = np.frombuffer(checksum, np.uint8)
                    payload_digest.update(payload); raw_digest.update(raw)
                if progress and stop % 1024 == 0:
                    progress({"channel": name, "stage": "compressing", "completed": stop, "states": states,
                        "compressed_bytes": int(offsets[stop])})
            stream.flush(); os.fsync(stream.fileno())
        _atomic_npz(index_path, offsets=offsets, raw_row_sha256=checksums)
        info = {"data_file": data_path.name, "index_file": index_path.name,
            "data_sha256": payload_digest.hexdigest(), "index_sha256": file_sha256(index_path),
            "raw_rows_sha256": raw_digest.hexdigest(), "compressed_bytes": int(offsets[-1]),
            "raw_bytes": states*states*8, "clipping_absolute_defect": defect,
            "seconds": time.perf_counter()-elapsed}
        manifest["metadata"]["channels"][name] = info
        manifest["active_stage"] = "channel_complete"
        _atomic_json(manifest_path, manifest)
        del scratch
        # Only the explicitly owned scratch is removed; immutable complete channels remain.
        scratch_path.unlink()
        if progress:
            progress({"channel": name, "stage": "complete", **info})
    if probability_reader is not None:
        probability_reader.close()
    after = {str(path.relative_to(ROOT)): file_sha256(path) for path in sources}
    if hashes != after:
        raise ValueError("Precompute sources changed during the build")
    manifest["metadata"]["source_sha256_after"] = after
    manifest["metadata"]["compressed_bytes"] = sum(info["compressed_bytes"] for info in manifest["metadata"]["channels"].values())
    manifest["metadata"]["this_run_seconds"] = time.perf_counter()-start_time
    manifest.pop("active_channel", None); manifest.pop("active_stage", None)
    manifest["status"] = "complete"
    manifest["complete"] = True
    manifest["metadata"]["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    return manifest
