#!/bin/bash
# start_server.sh — boot the local debug VillageSQL server on a scratch datadir
# for the vsql-vector recall harness. Idempotent-ish: wipes+reinits the scratch
# datadir each run so tests start clean. Prints the socket path on success.
#
# The server build dir is auto-detected (no hard-coded personal paths):
#   1. $SRV_BUILD if set (explicit override), else
#   2. VillageSQL_BUILD_DIR from the extension's build/CMakeCache.txt (the same
#      value passed to `cmake -DVillageSQL_BUILD_DIR=...` when building the .veb).
# So on any built vsql-vector checkout this needs zero config.
#
# Env overrides:
#   SRV_BUILD    server build dir (auto-detected from CMakeCache if unset)
#   WORKDIR      scratch dir for datadir/socket/logs (default bench/.run)
#   EXTENSIONS   space-separated extension names to install (default vsql_knn_store_test)
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$BENCH_DIR/.." && pwd)"

detect_srv_build() {
  local cache="$REPO_DIR/build/CMakeCache.txt"
  [ -f "$cache" ] || return 1
  # CMakeCache line: VillageSQL_BUILD_DIR:PATH=/abs/path/to/server/build
  sed -n 's/^VillageSQL_BUILD_DIR:[^=]*=//p' "$cache" | head -1
}

SRV_BUILD="${SRV_BUILD:-$(detect_srv_build || true)}"
if [ -z "${SRV_BUILD:-}" ]; then
  echo "ERROR: could not determine the server build dir. Set SRV_BUILD, or build" >&2
  echo "the extension first (cmake .. -DVillageSQL_BUILD_DIR=<server build>) so it" >&2
  echo "can be read from $REPO_DIR/build/CMakeCache.txt." >&2
  exit 1
fi
WORKDIR="${WORKDIR:-$BENCH_DIR/.run}"
MYSQLD="$SRV_BUILD/runtime_output_directory/mysqld"
DATADIR="$WORKDIR/data"
SOCKET="$WORKDIR/mysqld.sock"
ERRLOG="$WORKDIR/mysqld.err"
PIDFILE="$WORKDIR/mysqld.pid"

[ -x "$MYSQLD" ] || { echo "ERROR: mysqld not found/executable: $MYSQLD" >&2; exit 1; }

# Stop any prior instance on this socket.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  sleep 2
fi

rm -rf "$WORKDIR"
mkdir -p "$DATADIR"

echo "Initializing scratch datadir..." >&2
"$MYSQLD" --no-defaults --initialize-insecure \
  --basedir="$SRV_BUILD" --datadir="$DATADIR" >"$ERRLOG" 2>&1

echo "Starting mysqld (socket=$SOCKET)..." >&2
# skip-networking: harness talks over the unix socket only.
# MYSQLD_EXTRA: optional extra mysqld args (e.g. --innodb-buffer-pool-size=4G)
# for perf experiments; space-separated, word-split intentionally.
"$MYSQLD" --no-defaults \
  --basedir="$SRV_BUILD" --datadir="$DATADIR" \
  --socket="$SOCKET" --pid-file="$PIDFILE" \
  --skip-networking \
  --log-error="$ERRLOG" \
  --secure-file-priv="" \
  ${MYSQLD_EXTRA:-} \
  &

# Wait for the socket to accept connections.
MYSQL="$SRV_BUILD/runtime_output_directory/mysql"
for i in $(seq 1 60); do
  if "$MYSQL" --no-defaults -uroot --socket="$SOCKET" -e "SELECT 1" >/dev/null 2>&1; then
    # Bake the custom-KNN gates in as SERVER-WIDE DEFAULTS so every connection
    # (incl. the harness's) works without per-session SET. All three are
    # required: preview (to install/use the extension), hypergraph optimizer
    # (classic optimizer never selects the custom KNN scan → filesort → crash),
    # and the custom-index debug gate (POC path is debug-gated). Extensions to
    # install are passed space-separated via EXTENSIONS (default: the KNN test
    # extensions). Datadir is fresh each run, so (re)install here.
    echo "Applying gate defaults + installing extensions..." >&2
    "$MYSQL" --no-defaults -uroot --socket="$SOCKET" 2>>"$ERRLOG" <<SQL || true
SET PERSIST vsql_allow_preview_extensions = ON;
SET GLOBAL optimizer_switch='hypergraph_optimizer=on';
SET GLOBAL debug='+d,villagesql_custom_index_proceed';
SQL
    # NOTE: install only ONE vector extension — vsql_knn_store_test and
    # vsql_knn_mem_test (and vsql_vector) each register a KVECTOR/SVECTOR type;
    # installing two that share a type name makes type resolution ambiguous and
    # the server asserts (resolve_type_descriptor: results.size() > 1). Pass a
    # single name via EXTENSIONS to target a different one.
    for ext in ${EXTENSIONS:-vsql_knn_store_test}; do
      "$MYSQL" --no-defaults -uroot --socket="$SOCKET" \
        -e "INSTALL EXTENSION $ext;" 2>>"$ERRLOG" || \
        echo "  (note: INSTALL $ext skipped/failed — may already be installed)" >&2
    done
    echo "READY socket=$SOCKET" >&2
    echo "$SOCKET"
    exit 0
  fi
  sleep 1
done

echo "ERROR: server did not become ready; tail of $ERRLOG:" >&2
tail -30 "$ERRLOG" >&2
exit 1
