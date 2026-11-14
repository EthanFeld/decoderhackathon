# Optimized CPU quantum decoder

The entry point is `decoder.decode(circuit, detector_events)`, returning
`uint8[shots,4]`. `decoder.predict_proba` returns four float64 logical-bit
marginals. This archive selects the **adaptive** profile through the
`organizer_decoder` entry module. Only the two included public circuits are
supported. Call `decoder.close()` after the last inference call when available.

Use Python 3.11–3.14 and install the exact dependencies:

    python -m pip install -r requirements.txt

Keep the extracted directory structure. The 303.8 MB boundary probability
tables are circuit-derived physical model data. They contain no training
labels, benchmark answers or saved predictions. All runtime and table creation
source is included; no API key, service, network access or data download is
needed. Numba compiles architecture-specific kernels on the destination host;
compiled caches from the development machine are deliberately excluded.

The retained profile sums all local-block fault orders and retains every
zero/one/two-fault edge path plus triples containing at least two mechanisms
whose probabilities exceed 0.01. It is a specified retained probability model,
not a claim of exact full-channel maximum likelihood. It makes four separate
bitwise posterior decisions using strict `q>0.5`.

Adaptive CPU source is also included. Its policy keeps the same retained
model, repairs up to 16 selected omitted rows per edge only when a retained
bit marginal is in [0.4,0.6], and uses physics-only caches. The 14 GB compressed
row stores and 20 GiB resident row banks are not included. A selected adaptive
entry builds physical rows from the included circuit/model into guarded RAM
during import when compressed row stores are absent. The organizer's 288 GB
host can hold both 10 GiB banks. On a smaller host, disk mode can use separately
created compressed physical rows; these are not part of this archive.
Retained mode needs neither row banks nor their precomputation. No GPU package
is required. Runtime metadata records the actual selected storage/cache
configuration and initialization time.

The original model-construction source is preserved as `prior_model.py`.
Packaged dependencies import it directly to avoid a circular harness import.
Its BP/OSD reference functions are not used by the optimized entry.

To reproduce the two boundary tables into a NEW directory:

    python -m diagnostics.boundary21_tables --output data/rebuilt --report data/rebuilt/build_report.json --threads 4

A full table rebuild requires substantial memory and time. Optional adaptive
precomputation is available through `python -m diagnostics.precompute_adaptive_rows --help`.

`MANIFEST.json` lists every packaged payload hash, exact dependencies, default
profile and source-only import transformation. Validation fixtures and outputs
remain outside the archive. This is an offline source hand-in; packaging does
not upload or submit anything.
