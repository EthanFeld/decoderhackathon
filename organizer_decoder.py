"""Import-ready CPU organizer entry for retained or adaptive decoding.

    import organizer_decoder
    predictions = organizer_decoder.decode(circuit, detector_events)

QEC_DECODER_PROFILE=retained|adaptive selects the packaged profile. Retained
never opens physical row banks. Adaptive uses existing banks when supplied;
otherwise it builds the identical all-order float64 rows in RAM during import.
That fallback requires guarded room for20GiB of rows, one2GiB scratch channel,
endpoint caches and headroom. Construction uses circuit physics only.
QEC_ROW_STORAGE=auto|ram|disk controls existing banks. Endpoint caches default
to bidirectional:4096 entries per end/basis,16384 with at least96GiB available,
or128 in bounded local mode. GPU is never selected automatically.

No labels, benchmark records, reference predictions, full-syndrome answer
tables, or network services participate. decoder.py remains separate.
Running this file with --help does not import numerical dependencies or
initialize models. Packaging and official-machine validation are separate.
"""
from __future__ import annotations

import atexit
import copy
import hashlib
import os
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
GIB = 1 << 30
AUTO_RAM_THRESHOLD = 32*GIB
RAM_HEADROOM = 2*GIB
DISK_HEADROOM = 128 << 20
MODEL_STARTUP_RESERVE = 768 << 20
MAX_THREADS = 48
POLICY = {"rows_per_edge": 16, "uncertainty_gate": .4, "batch_size": 8,
          "cache_rows": 512, "use_cached_kernels": True, "read_workers": 4,
          "pipeline": True, "workspace_rows": 512, "cache_policy": "lookahead"}
KNOWN_CIRCUITS = {
    "dc6d415ae3f0b5396a7f63ba1047f024be8ce7e44b03ace65348bcee78b3c070": "X",
    "36a24a0347bbb82362af28d816d790d5ee1cb1543fef2e9af7ada70536bf9968": "Z",
}
_MODELS = {}
_STORES = {}
_CALL_LOCK = threading.RLock()
_READY = False
_NUMBA_THREADS = None
STARTUP_METADATA = {"initialized": False, "closed": False}


def _integer_environment(name, default, cap):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value, min(value, cap)


def _memory_status():
    """Physical RAM, additionally bounded by a readable Linux cgroup limit."""
    total, available, source = None, 0, "unavailable; full preload forbidden"
    try:
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
                raise OSError("Cannot read physical memory")
            total, available = int(memory.total_physical), int(memory.available_physical)
            source = "Windows GlobalMemoryStatusEx"
        else:
            page = os.sysconf("SC_PAGE_SIZE")
            total = int(os.sysconf("SC_PHYS_PAGES")*page)
            available = int(os.sysconf("SC_AVPHYS_PAGES")*page)
            source = "sysconf physical pages"
            if sys.platform.startswith("linux"):
                info = {}
                for line in Path("/proc/meminfo").read_text().splitlines():
                    key, value = line.split(":", 1)
                    info[key] = int(value.split()[0])*1024
                available = info.get("MemAvailable", available)
                source = "Linux MemAvailable"
    except (OSError, ValueError, AttributeError):
        pass
    cgroup = None
    if sys.platform.startswith("linux"):
        pairs = (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                 ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"))
        for limit_path, used_path in pairs:
            try:
                limit = int(Path(limit_path).read_text().strip())
                used = int(Path(used_path).read_text().strip())
                if limit > 0 and (total is None or limit < total):
                    available = min(available, max(0, limit-used))
                    total = limit if total is None else min(total, limit)
                    cgroup = {"limit_bytes": limit, "used_bytes": used}
            except (OSError, ValueError):
                continue
    return {"total_effective_bytes": total, "available_effective_bytes": available,
            "source": source, "cgroup": cgroup}


def _configure_resources():
    """Set module-owned numerical imports to one BLAS thread, not48x48."""
    preimported = {name: name in sys.modules for name in ("numpy", "numba", "scipy")}
    defaults = min(MAX_THREADS, os.cpu_count() or 1)
    requested, threads = _integer_environment("NUMBA_NUM_THREADS", defaults, MAX_THREADS)
    # An initialized Numba pool cannot safely have its maximum reconfigured.
    if not preimported["numba"]:
        os.environ["NUMBA_NUM_THREADS"] = str(threads)
    blas = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "GOTO_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in blas}
    for name in blas:
        os.environ[name] = "1"
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
    workers_requested, workers = _integer_environment("QEC_PRELOAD_WORKERS", 8, MAX_THREADS)
    return {"numba_threads_requested": requested, "numba_threads_selected": threads,
            "preload_workers_requested": workers_requested, "preload_workers_selected": workers,
            "blas_thread_environment": {name: "1" for name in blas},
            "previous_blas_thread_environment": previous, "preimported_libraries": preimported,
            "blas_scope": "Limits are set before this module imports its numerical dependencies. Decoder contraction uses Numba kernels; this module starts no BLAS worker pool. Already-loaded external libraries retain their own prior configuration."}


def _worker_scratch(states, workers):
    # Same conservative worker-buffer allowance as the row-store preloader.
    return int(workers)*int(states)*8*9+(32 << 20)


def _partial_options():
    mode = os.environ.get("QEC_PARTIAL_MODE", "bidirectional").strip().lower()
    if mode not in ("strict_first", "bidirectional", "none"):
        raise ValueError("QEC_PARTIAL_MODE must be strict_first, bidirectional, or none")
    result = {"mode": mode}
    for endpoint, variable in (("first", "QEC_FIRST_CACHE_ROWS"), ("last", "QEC_LAST_CACHE_ROWS")):
        raw = os.environ.get(variable)
        if raw is None:
            result[endpoint+"_override"] = None
        else:
            try:
                value = int(raw)
            except ValueError as error:
                raise ValueError(f"{variable} must be a nonnegative integer") from error
            if value < 0:
                raise ValueError(f"{variable} must be a nonnegative integer")
            result[endpoint+"_override"] = value
    if mode == "strict_first" and result["last_override"]:
        raise ValueError("Nonzero QEC_LAST_CACHE_ROWS requires explicit bidirectional mode")
    return result


def _partial_plan(stores, backend, options):
    default_first = (16384 if _memory_status()["available_effective_bytes"] >= 96*GIB else 4096) if backend == "ram" else 128
    first = default_first if options["first_override"] is None else options["first_override"]
    last = (default_first if options["mode"] == "bidirectional" else 0) if options["last_override"] is None else options["last_override"]
    if options["mode"] == "none":
        first = last = 0
    plan = {"mode": options["mode"], "backend": backend, "per_basis": {}, "total_bytes": 0}
    for basis, store in stores.items():
        # Both known circuits have seven prefix bits at either endpoint.
        domain = store.states*(1 << 7)
        a, b = min(int(first), domain), min(int(last), domain)
        size = sum(count*(store.states*5*8+8)+(domain*4 if count else 0) for count in (a, b))
        plan["per_basis"][basis] = {"first_entries": a, "last_entries": b,
            "planned_bytes": size, "minimum_nonempty_bytes": store.states*5*8+8+domain*4}
        plan["total_bytes"] += size
    return plan


def _require_combined_ram(stores, workers, partial_plan):
    needed = sum(store.states*store.states*5*8 for store in stores.values())
    scratch = max(_worker_scratch(store.states, workers) for store in stores.values())
    memory = _memory_status()
    minimum = needed+partial_plan["total_bytes"]+RAM_HEADROOM+MODEL_STARTUP_RESERVE+scratch
    if memory["available_effective_bytes"] < minimum:
        raise MemoryError(f"Full X/Z rows and endpoint caches need at least{minimum} currently available bytes; available{memory['available_effective_bytes']}. Use QEC_ROW_STORAGE=disk or auto.")
    return needed, scratch, memory


def _prepare_partial(model, plan, mode, backend, remaining_partial_bytes):
    from diagnostics.tensor_partial_cache import prepare_partial_cache
    before = _memory_status()
    available = before["available_effective_bytes"]
    planned = int(plan["planned_bytes"])
    if backend == "ram":
        minimum = remaining_partial_bytes+RAM_HEADROOM+MODEL_STARTUP_RESERVE
        if available < minimum:
            raise MemoryError("RAM endpoint caches would consume the guarded remaining-cache/headroom budget")
        budget = planned
    else:
        # Bounded disk mode must remain usable on the small local machine.
        # The helper may halve counts; the128MiB reserve is independent of the
        # stricter2GiB reserve required for full row-bank preloading.
        budget = min(planned, available//4, max(0, available-DISK_HEADROOM))
    cache = None
    elapsed = 0.
    if planned and budget >= plan["minimum_nonempty_bytes"]:
        started = time.perf_counter()
        cache = prepare_partial_cache(model.fast,
            first_entries=plan["first_entries"], last_entries=plan["last_entries"],
            mode="strict_first" if mode == "none" else mode, max_bytes=int(budget))
        elapsed = time.perf_counter()-started
    metadata = {} if cache is None else cache["metadata"]
    actual_first = metadata.get("per_endpoint", {}).get("first", {}).get("entries", 0)
    actual_last = metadata.get("per_endpoint", {}).get("last", {}).get("entries", 0)
    if backend == "ram" and (actual_first != plan["first_entries"] or actual_last != plan["last_entries"]):
        raise MemoryError("Full-RAM endpoint cache did not retain its explicitly budgeted entry counts")
    if not actual_first and not actual_last:
        model.fast.pop("partial_cache", None)
    return {"mode": mode, "requested_first_entries": plan["first_entries"],
        "requested_last_entries": plan["last_entries"],
        "actual_first_entries": int(actual_first), "actual_last_entries": int(actual_last),
        "planned_bytes": planned, "effective_budget_bytes": int(budget),
        "actual_bytes": int(metadata.get("allocated_bytes", 0)),
        "first_predicted_retained_source_coverage": float(metadata.get("per_endpoint", {}).get("first", {}).get("predicted_retained_source_coverage", 0.)),
        "last_predicted_retained_source_coverage": float(metadata.get("per_endpoint", {}).get("last", {}).get("predicted_retained_source_coverage", 0.)),
        "construction_seconds": elapsed, "memory_before": before,
        "reduced_for_memory": bool(actual_first < plan["first_entries"] or actual_last < plan["last_entries"]),
        "key_selection": metadata.get("key_selection", "disabled or insufficient bounded memory"),
        "arithmetic": metadata.get("arithmetic", "no endpoint cache enabled")}


def _open_disk_stores(circuits, directories, row_store_module):
    stores, identities = {}, {}
    try:
        for basis in "XZ":
            edge, paths, _ = row_store_module.circuit_physics(circuits[basis])
            identity = row_store_module.identity_for(circuits[basis], edge, paths)
            stores[basis] = row_store_module.AdaptiveRowStore(
                directories[basis], expected_identity=identity)
            identities[basis] = identity
    except BaseException:
        for store in stores.values():
            store.close()
        raise
    return stores, identities


def _initialize():
    global _READY, _NUMBA_THREADS, _np, _numba, _stim
    from types import SimpleNamespace
    started = time.perf_counter()
    profile = os.environ.get("QEC_DECODER_PROFILE", "adaptive").strip().lower()
    if profile not in ("retained", "adaptive"):
        raise ValueError("QEC_DECODER_PROFILE must be retained or adaptive")
    storage = os.environ.get("QEC_ROW_STORAGE", "auto").strip().lower()
    if storage not in ("auto", "ram", "disk"):
        raise ValueError("QEC_ROW_STORAGE must be auto, ram, or disk")
    partial_options = _partial_options()
    resources = _configure_resources()
    memory_before = _memory_status()
    dependency_started = time.perf_counter()
    import numpy as np
    import numba
    import stim
    import adaptive_row_store as row_store_module
    from adaptive_tensor_decoder import AdaptiveTensorDecoder
    from diagnostics.adaptive_cached_kernels import prepare_right_layout
    _np, _numba, _stim = np, numba, stim
    _NUMBA_THREADS = min(resources["numba_threads_selected"], int(numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(_NUMBA_THREADS)
    resources["numba_threads_effective"] = _NUMBA_THREADS
    dependency_seconds = time.perf_counter()-dependency_started
    asset_root = Path(os.environ.get("QEC_ASSET_ROOT", str(ROOT))).resolve()
    directories = {basis: asset_root/f"data/target03/adaptive_rows_{basis}" for basis in "XZ"}
    circuits = {basis: stim.Circuit.from_file(str(asset_root/f"data/competition/{basis}_circuit.stim")) for basis in "XZ"}
    for basis, circuit in circuits.items():
        signature = hashlib.sha256(str(circuit).encode()).hexdigest()
        if KNOWN_CIRCUITS.get(signature) != basis:
            raise ValueError(f"Bundled {basis} circuit does not match this organizer entry")
    descriptors = {basis: SimpleNamespace(states=16384) for basis in "XZ"}
    ram_partial_plan = _partial_plan(descriptors, "ram", partial_options)
    disk_partial_plan = _partial_plan(descriptors, "disk", partial_options)
    row_bytes = 0 if profile == "retained" else 2*16384*16384*5*8
    compile_scratch = (16384*16384*8)+(256 << 20)
    workers = resources["preload_workers_selected"]
    scratch = max(_worker_scratch(16384, workers), compile_scratch if profile == "adaptive" else 0)
    full_ram_budget = row_bytes+ram_partial_plan["total_bytes"]+RAM_HEADROOM+MODEL_STARTUP_RESERVE+scratch
    auto_minimum = max(AUTO_RAM_THRESHOLD, full_ram_budget)
    available = _memory_status()["available_effective_bytes"]
    selected = "ram" if available >= auto_minimum else "disk"
    if profile == "adaptive" and storage == "ram": selected = "ram"
    if profile == "adaptive" and storage == "disk": selected = "disk"
    stores, identities = {}, {}
    preload = {basis: 0. for basis in "XZ"}
    header_seconds, fallback = 0., None
    bootstrap = False
    try:
        if profile == "adaptive":
            supplied = all((directories[basis]/"manifest.json").is_file() for basis in "XZ")
            if supplied:
                tick = time.perf_counter()
                stores, identities = _open_disk_stores(circuits, directories, row_store_module)
                header_seconds = time.perf_counter()-tick
            else:
                if storage == "disk":
                    raise FileNotFoundError("Physical row banks are absent; startup compilation requires QEC_ROW_STORAGE=auto or ram")
                if available < full_ram_budget:
                    raise MemoryError(f"Self-contained adaptive startup needs{full_ram_budget} available bytes; available{available}")
                selected, bootstrap = "ram", True
            if selected == "ram":
                try:
                    remaining = row_bytes
                    for basis in "XZ":
                        needed = 16384*16384*5*8
                        remaining -= needed
                        reserve = remaining+ram_partial_plan["total_bytes"]+RAM_HEADROOM+MODEL_STARTUP_RESERVE
                        memory = _memory_status()
                        required = needed+reserve+(compile_scratch if bootstrap else _worker_scratch(16384, workers))
                        if memory["available_effective_bytes"] < required:
                            raise MemoryError("Full-RAM startup would violate the guarded remaining-bank/cache budget")
                        if bootstrap:
                            from diagnostics.compiled_row_memory import compile_memory_store
                            edge, paths, _ = row_store_module.circuit_physics(circuits[basis])
                            stores[basis] = compile_memory_store(circuits[basis], edge, paths,
                                reserve_bytes=reserve, available_bytes=memory["available_effective_bytes"])
                            identities[basis] = stores[basis].identity
                        else:
                            stores[basis] = stores[basis].preload_all(workers=workers,
                                reserve_bytes=reserve, max_bytes=needed, close_source=True)
                        preload[basis] = stores[basis].preload_seconds
                except MemoryError as error:
                    if storage == "ram" or bootstrap:
                        raise
                    for store in stores.values(): store.close()
                    stores, identities = _open_disk_stores(circuits, directories, row_store_module)
                    selected, fallback = "disk", str(error)
        _STORES.update(stores)
        partial_plan = ram_partial_plan if selected == "ram" else disk_partial_plan
        remaining_partial = partial_plan["total_bytes"]
        per_basis = {}
        for basis in "XZ":
            if selected == "ram" and _memory_status()["available_effective_bytes"] < remaining_partial+RAM_HEADROOM+MODEL_STARTUP_RESERVE:
                raise MemoryError("RAM model/cache startup would violate the guarded headroom")
            tick = time.perf_counter()
            model = AdaptiveTensorDecoder(circuits[basis], **POLICY,
                row_store=stores[basis] if profile == "adaptive" else False,
                adaptive_refinement=profile == "adaptive",
                first_cache_rows=0, last_cache_rows=0,
                partial_mode="strict_first" if partial_options["mode"] == "none" else partial_options["mode"])
            signature = hashlib.sha256(str(circuits[basis]).encode()).hexdigest()
            _MODELS[signature] = model
            model_seconds = time.perf_counter()-tick
            partial = _prepare_partial(model, partial_plan["per_basis"][basis],
                partial_options["mode"], selected, remaining_partial)
            remaining_partial -= partial_plan["per_basis"][basis]["planned_bytes"]
            tick = time.perf_counter(); prepare_right_layout(model.fast)
            layout_seconds = time.perf_counter()-tick
            synthetic = np.zeros((1, model.num_detectors), np.uint8)
            tick = time.perf_counter(); retained = model.predict_proba(synthetic)
            retained_seconds = time.perf_counter()-tick
            adaptive_seconds = 0.
            details = {"compiled_rows": 0, "precomputed_rows_loaded": 0}
            if profile == "adaptive":
                tick = time.perf_counter()
                warmed, details = model.refine_probabilities(synthetic, np.full((1, 4), .5))
                adaptive_seconds = time.perf_counter()-tick
                if warmed.shape != (1, 4) or not np.all(np.isfinite(warmed)):
                    raise RuntimeError("Import-time adaptive warmup failed")
            if retained.shape != (1, 4) or not np.all(np.isfinite(retained)) or details["compiled_rows"] != 0:
                raise RuntimeError("Import-time fixed synthetic warmup failed")
            store = stores.get(basis)
            resident = getattr(store, "resident_rows", None)
            per_basis[basis] = {"circuit_sha256": signature,
                "store_manifest_sha256": None if store is None else store.manifest_sha256,
                "store_identity": identities.get(basis), "model_construction_seconds": model_seconds,
                "partial_cache": partial, "preload_seconds": preload[basis],
                "layout_preparation_seconds": layout_seconds,
                "retained_jit_warmup_seconds": retained_seconds,
                "adaptive_jit_warmup_seconds": adaptive_seconds, "warmup_records": 1,
                "forced_adaptive_reference_probability": .5 if profile == "adaptive" else None,
                "resident_bytes": 0 if resident is None else int(resident.nbytes),
                "resident_array_readonly": None if resident is None else not bool(resident.flags.writeable),
                "backend": "retained_only" if store is None else ("compiled_full_ram" if bootstrap else ("full_ram" if resident is not None else "compressed_disk")),
                "warmup_precomputed_rows_loaded": int(details.get("precomputed_rows_loaded", 0)),
                "warmup_compiled_rows": int(details["compiled_rows"])}
        STARTUP_METADATA.update({"initialized": True, "closed": False,
            "decoder_profile": profile, "storage_requested": storage,
            "storage_selected": "retained_only" if profile == "retained" else ("compiled_ram" if bootstrap else selected),
            "startup_compiled_physics": bootstrap, "auto_ram_threshold_bytes": AUTO_RAM_THRESHOLD,
            "full_banks_required_bytes": row_bytes, "ram_headroom_bytes": RAM_HEADROOM,
            "model_startup_reserve_bytes": MODEL_STARTUP_RESERVE,
            "startup_scratch_planned_bytes": scratch,
            "auto_ram_minimum_planned_bytes": auto_minimum,
            "full_ram_combined_planned_bytes": full_ram_budget,
            "ram_partial_plan": ram_partial_plan, "selected_partial_plan": partial_plan,
            "partial_environment": partial_options,
            "actual_partial_cache_bytes": sum(info["partial_cache"]["actual_bytes"] for info in per_basis.values()),
            "retained_backend": "cpu", "gpu_scope": "GPU is never selected automatically.",
            "auto_fallback_reason": fallback, "memory_before": memory_before,
            "memory_after": _memory_status(), "resources": resources,
            "policy": {**POLICY, "adaptive_refinement": profile == "adaptive"},
            "asset_root": str(asset_root), "store_directories": {b: str(path) for b, path in directories.items()},
            "dependency_import_seconds": dependency_seconds,
            "store_header_and_identity_seconds": header_seconds,
            "per_basis": per_basis, "startup_total_seconds": time.perf_counter()-started,
            "versions": {"numpy": np.__version__, "numba": numba.__version__, "stim": stim.__version__},
            "serialized_public_calls": True,
            "cache_scope": "Only circuit physics and physical endpoint/edge rows; no full-syndrome answers are cached.",
            "warmup_scope": "One synthetic zero record per basis; adaptive profile additionally forces one physical repair branch.",
            "row_validation": "Persisted banks verify identity/checksums; compiled banks use identical selected-channel arithmetic and finite checks."})
        _READY = True
        atexit.register(close)
    except BaseException:
        for model in _MODELS.values(): model.close()
        for store in stores.values(): store.close()
        _MODELS.clear(); _STORES.clear()
        raise


def predict_proba(circuit, detector_events):
    """Return logical bit marginals using the fully initialized physics model."""
    with _CALL_LOCK:
        if not _READY:
            raise RuntimeError("Organizer decoder is not initialized or has been closed")
        if not isinstance(circuit, _stim.Circuit):
            raise TypeError("Expected a stim.Circuit")
        signature = hashlib.sha256(str(circuit).encode()).hexdigest()
        if signature not in _MODELS:
            raise ValueError("Organizer decoder supports only the two bundled known circuits")
        _numba.set_num_threads(_NUMBA_THREADS)
        return _MODELS[signature].predict_proba(detector_events)


def decode(circuit, detector_events):
    """Harness-shaped uint8[shots,4] decision entry point."""
    return (predict_proba(circuit, detector_events) > .5).astype(_np.uint8)


def get_startup_metadata():
    """Return a detached startup timing/backend/resource description."""
    with _CALL_LOCK:
        return copy.deepcopy(STARTUP_METADATA)


def close():
    """Release resident banks, disk readers and bounded worker pools."""
    global _READY
    with _CALL_LOCK:
        for model in _MODELS.values():
            model.close()
        _MODELS.clear(); _STORES.clear()
        _READY = False
        STARTUP_METADATA["closed"] = True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print("Import organizer_decoder to initialize the organizer entry; see --help for configuration.")
else:
    _initialize()
