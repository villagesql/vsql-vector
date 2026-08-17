# bench/ — local recall harness for custom KNN vector indexes

A lightweight, local (no Docker) harness to iterate on VillageSQL custom vector
indexes. Unlike MTR — which does **exact** `.result` comparison — this verifies
results the way vector search must be judged: by **recall** (what fraction of the
true k nearest neighbours the index returned), so it correctly handles the
**approximate** nature of ANN indexes.

It reports **recall@k**, **QPS**, and **index build time**, and exits non-zero if
mean recall is below a threshold (usable as a gate).

## Prerequisites

- A built VillageSQL **debug** server (the custom-index path is debug-gated).
- This extension built against it — the harness reads the server build dir from
  `../build/CMakeCache.txt` (`VillageSQL_BUILD_DIR`), i.e. the value you passed to:
  ```
  cmake .. -DVillageSQL_BUILD_DIR=<server-build-dir>
  make
  ```
- Python venv with numpy (for exact ground-truth):
  ```
  python3 -m venv .venv && . .venv/bin/activate && pip install numpy
  ```

## Usage

```bash
# 1. Start a throwaway server (fresh scratch datadir under bench/.run/).
#    Auto-detects the server build dir + bakes in the required gates
#    (preview + hypergraph optimizer + custom-index debug) as server defaults,
#    and installs the target extension.
./start_server.sh                       # prints the unix socket path

# 2. Run the recall harness (venv active).
. .venv/bin/activate
python recall_harness.py --profile knn_store --dim 4 --n 500 --queries 50 -k 10
#   → build_time_s=... qps=... recall@10=1.0000  PASS
```

### Profiles (which extension / SQL surface)

- `knn_store` — the exact brute-force test extension (`vsql_knn_store_test`,
  `KVECTOR`, **fixed 4 dims**). Recall must be ~1.0 — use it to sanity-check the
  harness and the server path.
- `vsql_vector` — this extension's ANN/HNSW index (`SVECTOR(D)`). Recall < 1.0 is
  the real ANN-quality signal; sweep `--M` / `--ef-construction`.

Each profile hard-codes that extension's **SQL surface** — the harness does not
auto-detect it, because surfaces differ in ways a type name can't reveal (e.g.
`KVECTOR::from_string` vs `SVECTOR::FROM_STRING` casing, `kvec_store_l2_distance()`
vs `L2_DISTANCE()`, fully-qualified vs bare index type, `WITH (M, ef_construction)`
vs none). **To benchmark a new extension**, add a profile entry to `PROFILES` in
`recall_harness.py` with: `extension` (install name), `coltype` (e.g. `SVECTOR({dim})`),
`vec_literal` (how a vector literal is written), `index_ddl` (the `CREATE INDEX ...
USING EXTENDED(...)`), `dist_fn` (the ORDER BY distance expression), and optionally
`fixed_dim`. Read these off the extension's own `mysql-test/t/*.test`.

### Distance metrics

`--metric l2|cosine|ip|l1` selects the distance function. It drives all three
coupled places consistently — the index build (`hnsw_l2`/`hnsw_cosine`/…), the
query (`L2_DISTANCE`/`COSINE_DISTANCE`/…), and the numpy ground truth — so recall
stays meaningful (building a cosine index but checking against L2 truth would give
bogus recall). Each profile declares which metrics its extension exposes; asking
for an unsupported one errors. Currently: `vsql_vector` = l2/cosine/l1/ip;
`knn_store` = l2/cosine.

```bash
python recall_harness.py --profile vsql_vector --metric cosine --dim 128 --n 5000
```

To add a metric to a profile, add an entry to its `metrics` map in
`recall_harness.py` (the `idx_modifier` for the index DDL + the `dist_fn` for the
query), and ensure `GROUND_TRUTH` has the matching numpy implementation.

### ef_search sweep (recall/QPS curve, one build)

`ef_search` is a query-time knob, so `--ef-search-sweep A,B,C` **builds the index
once** and re-queries at each value — much faster than N separate runs, and the
curve comes from one identical graph (no build-to-build variance). Traces the
recall-vs-QPS Pareto in a single command:

```bash
python recall_harness.py --profile vsql_vector --metric l2 --dim 32 --n 20000 \
  --M 16 --ef-construction 200 --ef-search-sweep 100,200,400,800
# build_time_s=...  (index built once; sweeping ef_search)
# ef_search  recall@10   qps
# 100        0.894       66.5
# 200        0.975       41.2  ...
```

Note: only `ef_search` is re-queryable without a rebuild. Sweeping `M`,
`ef_construction`, `dim`, or `n` requires a fresh build per value (separate runs).
Practical: `ef_search` likely needs to scale with `n` to hold recall (e.g. 100 is
enough at 5k, ~200 at 20k).

### Key flags

`--profile --metric --dim --n --queries -k --M --ef-construction --threshold
--seed --socket --mysql`. Run `python recall_harness.py --help` for all.

## Configuration / portability

Nothing is hard-coded to a particular machine. Resolution order for the server:

1. `SRV_BUILD` env var (explicit override), else
2. `VillageSQL_BUILD_DIR` from `../build/CMakeCache.txt` (auto).

The `mysql` client path is derived from that (or `--mysql`). `start_server.sh`
also honours `WORKDIR` (scratch dir, default `bench/.run`) and `EXTENSIONS`
(space-separated names to install — install only ONE vector extension at a time;
two that register the same type name make type resolution ambiguous).

## Notes / gotchas

- The custom KNN scan requires the **hypergraph optimizer**. `start_server.sh`
  sets it (and the preview + custom-index-debug gates) as server-wide defaults,
  because each `mysql` client invocation is a fresh session — per-session `SET`
  would not carry into the harness's query connections. With the classic
  optimizer the query falls back to a filesort and (currently) crashes the server.
- `bench/.run/` (scratch datadir, socket, error log) is regenerated each
  `start_server.sh` run; it should be git-ignored.
