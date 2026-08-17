#!/bin/bash
# start_mariadb.sh — boot the local MariaDB debug server on a scratch datadir,
# for driving MariaDB's vector (MHNSW) index with the recall harness as an
# in-database HNSW build-time reference vs vsql-vector.
# Prints the socket path on success. Mirrors start_server.sh's scratch-dir model.
set -euo pipefail

MB="${MB:-$HOME/githome/mariadb-server/build-debug}"
SRC="${MARIADB_SRC:-$HOME/githome/mariadb-server}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.run-maria}"
MARIADBD="$MB/sql/mariadbd"
MARIADB="$MB/client/mariadb"
DATADIR="$WORKDIR/data"
SOCKET="$WORKDIR/mariadb.sock"
ERRLOG="$WORKDIR/mariadb.err"
PIDFILE="$WORKDIR/mariadb.pid"

[ -x "$MARIADBD" ] || { echo "ERROR: mariadbd not found: $MARIADBD" >&2; exit 1; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")" 2>/dev/null || true; sleep 2
fi
rm -rf "$WORKDIR"; mkdir -p "$DATADIR"

# mariadb-install-db needs the source dir for its scripts/charsets in a build tree.
echo "Initializing scratch datadir..." >&2
"$MB/scripts/mariadb-install-db" --no-defaults --srcdir="$SRC" --builddir="$MB" \
  --datadir="$DATADIR" --auth-root-authentication-method=normal >"$ERRLOG" 2>&1 \
  || "$MB/scripts/mysql_install_db" --no-defaults --srcdir="$SRC" --builddir="$MB" \
       --datadir="$DATADIR" >"$ERRLOG" 2>&1

echo "Starting mariadbd (socket=$SOCKET)..." >&2
"$MARIADBD" --no-defaults --basedir="$MB" --datadir="$DATADIR" \
  --socket="$SOCKET" --pid-file="$PIDFILE" --skip-networking \
  --log-error="$ERRLOG" --secure-file-priv="" &

for i in $(seq 1 60); do
  if "$MARIADB" --no-defaults -uroot --socket="$SOCKET" -e "SELECT 1" >/dev/null 2>&1; then
    echo "READY socket=$SOCKET" >&2
    echo "$SOCKET"
    exit 0
  fi
  sleep 1
done
echo "ERROR: mariadb did not become ready; tail $ERRLOG:" >&2
tail -30 "$ERRLOG" >&2
exit 1
