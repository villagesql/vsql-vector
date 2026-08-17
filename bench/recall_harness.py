#!/usr/bin/env python3
"""
recall_harness.py — local recall-based tester for VillageSQL custom KNN vector
indexes. Generates random vectors, builds the custom index, runs KNN queries via
the index, and verifies APPROXIMATELY: compares the index's neighbors against
exact ground truth (computed client-side with numpy) and reports recall@k + QPS +
build time. Exits 0 if mean recall >= threshold, else 1 (usable as a gate).

Two extension "profiles" (SQL surfaces) are built in:
  knn_store : the exact brute-force test extension (recall MUST be ~1.0 — use it
              to validate the harness itself + the server path).
  vsql_vector : the real ANN/HNSW extension (recall < 1.0 = ANN quality signal).

Talks to a server started by start_server.sh over its unix socket.
"""
import argparse, subprocess, sys, time, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_SOCKET = os.path.join(HERE, ".run", "mysqld.sock")


def detect_srv_build():
    """Server build dir: $SRV_BUILD, else VillageSQL_BUILD_DIR from the
    extension's build/CMakeCache.txt (same value used to build the .veb)."""
    env = os.environ.get("SRV_BUILD")
    if env:
        return env
    cache = os.path.join(REPO, "build", "CMakeCache.txt")
    if os.path.isfile(cache):
        with open(cache) as f:
            for line in f:
                if line.startswith("VillageSQL_BUILD_DIR:"):
                    return line.split("=", 1)[1].strip()
    return None


def default_mysql():
    srv = detect_srv_build()
    return (os.path.join(srv, "runtime_output_directory", "mysql")
            if srv else "mysql")  # fall back to PATH


DEFAULT_MYSQL = default_mysql()

# Per-extension SQL surface. {dim} etc. filled at runtime.
# NOTE: vsql_knn_store_test's KVECTOR type is FIXED at exactly 4 dimensions
# (`kvector: expected exactly 4 elements`). Run this profile with --dim 4.
# A wrong --dim silently mis-encodes vectors and yields bogus (low) recall,
# so run_sql surfaces any VDF/dimension error rather than swallowing it.
# Each profile declares the extension's SQL surface. `metrics` maps a metric key
# (l2/cosine/ip/l1) to that extension's SQL for it:
#   idx_modifier : column modifier in the index DDL (or "" if metric is implicit)
#   dist_fn      : the ORDER BY distance expression (uses {qlit} = query literal)
# The SAME metric must drive the index build, the query, AND the numpy ground
# truth (GROUND_TRUTH below) — mismatching them makes recall meaningless.
PROFILES = {
    "knn_store": {
        "extension": "vsql_knn_store_test",
        "fixed_dim": 4,
        "coltype": "KVECTOR",
        "vec_literal": "KVECTOR::from_string('{lit}')",
        # brute-force store: one index type, metric selected purely by the query fn
        "index_ddl": "CREATE INDEX idx_v ON t (v) USING EXTENDED(vsql_knn_store_test.kvec_store)",
        "metrics": {
            "l2":     {"idx_modifier": "", "dist_fn": "vsql_knn_store_test.kvec_store_l2_distance(v, {qlit})"},
            "cosine": {"idx_modifier": "", "dist_fn": "vsql_knn_store_test.kvec_store_cos_distance(v, {qlit})"},
        },
    },
    "vsql_vector": {
        "extension": "vsql_vector",
        "coltype": "SVECTOR({dim})",
        # SVECTOR accepts a plain string literal (implicit conversion) — no
        # explicit FROM_STRING needed (matches the extension's own MTR tests).
        "vec_literal": "'{lit}'",
        # query-time HNSW search breadth, settable via SQL. NOTE: read only via
        # SHOW VARIABLES / not `SELECT @@global.<name>` (that read path crashes
        # for extension-namespaced dotted sysvars — separate server bug).
        "ef_search_var": "vsql_vector.ef_search",
        # HNSW: metric is chosen by BOTH the index modifier (build) and the query fn
        "index_ddl": ("CREATE INDEX idx_v ON t (v {idx_modifier}) USING EXTENDED(hnsw) "
                      "WITH (M = {M}, ef_construction = {efc})"),
        "metrics": {
            "l2":     {"idx_modifier": "hnsw_l2",            "dist_fn": "L2_DISTANCE(v, {qlit})"},
            "cosine": {"idx_modifier": "hnsw_cosine",        "dist_fn": "COSINE_DISTANCE(v, {qlit})"},
            "l1":     {"idx_modifier": "hnsw_l1",            "dist_fn": "L1_DISTANCE(v, {qlit})"},
            "ip":     {"idx_modifier": "hnsw_inner_product", "dist_fn": "INNER_PRODUCT(v, {qlit})"},
        },
    },
    # MariaDB MHNSW (13.1) — an in-database HNSW reference. VECTOR INDEX is INLINE
    # in CREATE TABLE (no separate CREATE INDEX), so this profile overrides the
    # whole table DDL via `table_ddl`. M matches vsql-vector; MariaDB's
    # ef_construction is HARDCODED at 10 (not tunable) vs vsql-vector's 200 —
    # documented, not matched. Vectors via Vec_FromText(); no gates/preview.
    "mariadb": {
        "extension": None,  # native; no INSTALL
        "coltype": "VECTOR({dim})",
        "vec_literal": "Vec_FromText('{lit}')",
        # {idx_modifier} unused; M set inline. ef_construction ignored (fixed 10).
        "table_ddl": ("CREATE TABLE t (id INT PRIMARY KEY, v VECTOR({dim}) NOT NULL, "
                      "VECTOR INDEX (v) M={M}) ENGINE=InnoDB;"),
        "no_index_table_ddl": ("CREATE TABLE t (id INT PRIMARY KEY, "
                               "v VECTOR({dim}) NOT NULL) ENGINE=InnoDB;"),
        "metrics": {
            "l2":     {"idx_modifier": "", "dist_fn": "vec_distance_euclidean(v, {qlit})"},
            "cosine": {"idx_modifier": "", "dist_fn": "vec_distance_cosine(v, {qlit})"},
        },
    },
}


# numpy exact ground truth per metric. Each returns per-row "distance" where
# SMALLER = nearer (so argsort ascending gives the true neighbours). For inner
# product, "nearest" = largest dot product, so we negate.
def _l2(data, q):     return np.linalg.norm(data - q, axis=1)
def _l1(data, q):     return np.sum(np.abs(data - q), axis=1)
def _cosine(data, q):
    dn = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-12)
    qn = q / (np.linalg.norm(q) + 1e-12)
    return 1.0 - dn @ qn
# The extension's INNER_PRODUCT returns the raw dot product and orders it
# ASCENDING (ORDER BY ... LIMIT), i.e. "nearest" = SMALLEST dot product. So
# ground truth is the raw dot (NOT negated), argsort ascending — matching the
# server's convention. (Verified against the server 2026-08-17.)
def _ip(data, q):     return data @ q

GROUND_TRUTH = {"l2": _l2, "l1": _l1, "cosine": _cosine, "ip": _ip}


def run_sql(mysql, socket, sql, want_rows=False):
    """Run SQL via the mysql client over the socket. Returns stdout lines."""
    cmd = [mysql, "--no-defaults", "-uroot", f"--socket={socket}",
           "--batch", "--raw"]
    if not want_rows:
        cmd.append("--silent")
    p = subprocess.run(cmd, input=sql, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"SQL failed (rc={p.returncode}):\n{p.stderr}\n--- sql ---\n{sql}")
    return p.stdout.strip().splitlines()


def vec_lit(v):
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="knn_store")
    ap.add_argument("--metric", choices=list(GROUND_TRUTH), default="l2",
                    help="distance metric; must be supported by the profile")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n", type=int, default=2000, help="rows in the table")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=64)
    ap.add_argument("--insert-batch", type=int, default=0,
                    help="rows per INSERT statement (0 = single statement of all n)")
    ap.add_argument("--no-index", action="store_true",
                    help="skip the custom index (SVECTOR column only) — build-cost "
                         "probe to isolate generic insert from graph maintenance; "
                         "recall will be meaningless")
    ap.add_argument("--ef-search", type=int, default=None,
                    help="query-time HNSW search breadth (profile must expose it)")
    ap.add_argument("--ef-search-sweep", default=None,
                    help="comma-separated ef_search values; builds the index ONCE "
                         "then re-queries at each (ef_search is query-time, so no "
                         "rebuild needed). Overrides --ef-search.")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--mysql", default=DEFAULT_MYSQL)
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    fixed = prof.get("fixed_dim")
    if fixed and args.dim != fixed:
        print(f"ERROR: profile '{args.profile}' requires --dim {fixed} "
              f"(its vector type is fixed at {fixed} dims); got {args.dim}.", file=sys.stderr)
        return 2
    if args.metric not in prof["metrics"]:
        print(f"ERROR: profile '{args.profile}' does not support metric "
              f"'{args.metric}'. Supported: {', '.join(prof['metrics'])}.", file=sys.stderr)
        return 2
    metric = prof["metrics"][args.metric]
    truth_fn = GROUND_TRUTH[args.metric]  # same metric drives numpy ground truth

    sweep = None
    if args.ef_search_sweep is not None:
        sweep = [int(x) for x in args.ef_search_sweep.split(",") if x.strip()]
    if (args.ef_search is not None or sweep) and "ef_search_var" not in prof:
        print(f"ERROR: profile '{args.profile}' does not expose a query-time "
              f"ef_search knob.", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    data = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    queries = rng.standard_normal((args.queries, args.dim)).astype(np.float32)

    coltype = prof["coltype"].format(dim=args.dim)
    # index_ddl only exists for the separate-CREATE-INDEX profiles; profiles with
    # a full table_ddl (e.g. mariadb) don't have it.
    index_ddl = (prof["index_ddl"].format(
        M=args.M, efc=args.ef_construction, idx_modifier=metric["idx_modifier"])
        if "index_ddl" in prof else "")

    print(f"profile={args.profile} metric={args.metric} dim={args.dim} n={args.n} queries={args.queries} "
          f"k={args.k} M={args.M} ef_construction={args.ef_construction} ef_search={args.ef_search}")

    # Gates (preview / hypergraph / custom-index debug) and extension install are
    # baked in as server defaults by start_server.sh, so no per-session SET needed.
    # --- setup: schema. --no-index skips the custom index (SVECTOR column only)
    # to isolate generic column-store row insertion from HNSW graph maintenance.
    # (Recall is meaningless without the index — --no-index is a build-cost probe.)
    # Profiles may supply a full table DDL (e.g. MariaDB's inline VECTOR INDEX);
    # otherwise use the default "CREATE TABLE" + separate index_ddl form.
    if "table_ddl" in prof:
        key = "no_index_table_ddl" if args.no_index else "table_ddl"
        schema = prof[key].format(dim=args.dim, M=args.M)
    else:
        ddl_line = "" if args.no_index else index_ddl + ";"
        schema = f"CREATE TABLE t (id INT PRIMARY KEY, v {coltype} NOT NULL);\n{ddl_line}"
    setup = f"""
DROP DATABASE IF EXISTS recall_bench;
CREATE DATABASE recall_bench; USE recall_bench;
{schema}
"""
    run_sql(args.mysql, args.socket, setup)

    def lit(v): return prof["vec_literal"].format(lit=vec_lit(v), dim=args.dim)

    # --- build: INSERT all rows (timed). ef_search is query-time, so a sweep
    # re-queries this same graph without rebuilding.
    # --insert-batch controls rows per INSERT statement (0 = one giant statement,
    # the old behaviour). Fixed-size batches keep statement-parse cost constant
    # across N, isolating server insert/index cost from harness SQL-parse cost. ---
    bs = args.insert_batch if args.insert_batch and args.insert_batch > 0 else args.n
    t0 = time.time()
    for start in range(0, args.n, bs):
        rows = ",\n".join(f"({i}, {lit(data[i])})"
                          for i in range(start, min(start + bs, args.n)))
        run_sql(args.mysql, args.socket, f"USE recall_bench; INSERT INTO t VALUES\n{rows};")
    build_s = time.time() - t0

    # Ground truth is ef_search-independent — compute once.
    truth = [set(np.argsort(truth_fn(data, queries[qi]))[:args.k].tolist())
             for qi in range(args.queries)]

    def set_ef(ef):
        if ef is None:
            return
        # Component-namespaced name UNQUOTED (backticks crash the server).
        run_sql(args.mysql, args.socket, f"SET GLOBAL {prof['ef_search_var']} = {ef};")

    def run_queries():
        # All queries in ONE connection (one process spawn) so QPS isn't
        # dominated by client startup. Sentinel SELECT delimits each query.
        parts = ["USE recall_bench;"]
        for qi in range(args.queries):
            dist = metric["dist_fn"].format(qlit=lit(queries[qi]))
            parts.append(f"SELECT '@@Q{qi}' AS m;")
            parts.append(f"SELECT id FROM t ORDER BY {dist} LIMIT {args.k};")
        t = time.time()
        out = run_sql(args.mysql, args.socket, "\n".join(parts), want_rows=True)
        qsec = time.time() - t
        per, cur = {}, None
        for line in out:
            s = line.strip()
            if s.startswith("@@Q"):
                cur = int(s[3:]); per[cur] = []
            elif cur is not None and s.isdigit():
                per[cur].append(int(s))
        hits = sum(len(set(per.get(qi, [])[:args.k]) & truth[qi])
                   for qi in range(args.queries))
        return hits / (args.queries * args.k), args.queries / qsec if qsec > 0 else float("inf")

    if sweep:
        # Build once, re-query at each ef_search value.
        print(f"build_time_s={build_s:.2f}  (index built once; sweeping ef_search)")
        print(f"{'ef_search':<10} {'recall@'+str(args.k):<12} {'qps':<8}")
        worst = 1.0
        for ef in sweep:
            set_ef(ef)
            rec, qps = run_queries()
            worst = min(worst, rec)
            print(f"{ef:<10} {rec:<12.4f} {qps:<8.1f}")
        return 0 if worst >= args.threshold else 1

    # single run
    set_ef(args.ef_search)
    recall, qps = run_queries()
    print(f"build_time_s={build_s:.2f}  qps={qps:.1f}  recall@{args.k}={recall:.4f}  "
          f"threshold={args.threshold}")
    if recall >= args.threshold:
        print("PASS"); return 0
    print("FAIL (recall below threshold)"); return 1


if __name__ == "__main__":
    sys.exit(main())
