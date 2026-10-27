#!/usr/bin/env python3
# Copyright (c) 2026 VillageSQL Contributors
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License, version 2.0,
# as published by the Free Software Foundation.
"""Live browser view of an HNSW graph, backed by svector_page_dump.

Serves a small page on localhost. Each Refresh in the page runs
`svector_page_dump <ibd> <root> -i -g --graph-format dot ...` again and
redraws the result, so the graph can be re-inspected after every DML statement
without touching the shell. See README.md ("Live browser view").

Example:
  graph_view                         # from the build dir (the CMake symlink), then open the page

The page also has a Storage view: one storage page at a time (root page or
data page, via `svector_page_dump --page-json`) with a square per record slot
and links to the previous/next page and back to the root.

The page has a Browse button to pick any .ibd file on this machine and a box
for the root page number.
"""

import argparse
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse

# Look applied by the viewer on top of the tool's DOT (inserted right after the
# digraph line, so the tool's own attributes still win where they differ).
STYLE = (
    'graph [bgcolor="transparent" pad=0.3 nodesep=0.45 ranksep=0.7 '
    'style=rounded color="#c3ccd9" fontname="Helvetica" fontsize=13 '
    'fontcolor="#3b4658"];\n'
    'node [style=filled fillcolor="#ffffff" color="#3b4658" penwidth=1.3 '
    'fontcolor="#1b2330" width=0.8];\n'
    'edge [penwidth=1.3];\n')
LABEL_RE = re.compile(r'(\n\s*graph \[[^\n]*?) label="((?:[^"\\]|\\.)*)"')


def split_dot(dot):
    """Returns (dot without the graph-level label, label lines).

    The tool's header and legend text is better shown as HTML than as a
    Graphviz title, so it is cut out of the drawing and sent separately."""
    info = []

    def cut(m):
        info.extend(x for x in m.group(2).split("\\l") if x)
        return m.group(1)

    dot = LABEL_RE.sub(cut, dot, count=1)
    dot = dot.replace("digraph hnsw {\n", "digraph hnsw {\n" + STYLE, 1)
    return dot, [x.replace('\\"', '"').replace("\\\\", "\\") for x in info]


# The page lives next to the real script; the tool is looked for next to the
# path the script was started from, which is the build directory when started
# through the symlink CMake creates there.
HERE = os.path.dirname(os.path.realpath(__file__))
LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAX_NODES = 200
MAX_NODES_LIMIT = 100000
MAX_LISTED_FILES = 2000


class Handler(http.server.BaseHTTPRequestHandler):
    cfg = None  # Set in main().

    def _send(self, code, ctype, body):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _browse(self, path):
        """Directory listing for the page's file browser: sub-directories and
        .ibd files only (dot-directories are left out of the list)."""
        path = os.path.realpath(os.path.expanduser(path or self.cfg.start_dir))
        if not os.path.isdir(path):
            return None
        dirs, files = [], []
        try:
            for e in sorted(os.scandir(path), key=lambda e: e.name.lower()):
                try:
                    if e.is_dir():
                        if not e.name.startswith("."):
                            dirs.append(e.name)
                    elif e.name.endswith(".ibd") and e.is_file():
                        files.append({"name": e.name, "size": e.stat().st_size})
                except OSError:
                    pass
        except OSError as err:
            return {"path": path, "parent": os.path.dirname(path), "dirs": [],
                    "files": [], "error": str(err)}
        return {"path": path, "parent": os.path.dirname(path), "dirs": dirs,
                "files": files}

    def _valid_ibd(self, path):
        real = os.path.realpath(os.path.expanduser(path))
        if real.endswith(".ibd") and os.path.isfile(real):
            return real
        return None

    def do_GET(self):
        # The server only answers to its own name, so a web page on another
        # site cannot reach it through DNS rebinding.
        if self.headers.get("Host", "") not in self.cfg.hosts:
            return self._send(403, "text/plain", "bad Host header")
        url = urllib.parse.urlparse(self.path)
        if url.path == "/":
            with open(os.path.join(HERE, "svector_graph_viewer.html"),
                      encoding="utf-8") as f:
                page = f.read()
            defaults = json.dumps({"maxNodes": DEFAULT_MAX_NODES,
                                   "startDir": self.cfg.start_dir})
            self._send(200, "text/html",
                       page.replace("__DEFAULTS__", defaults.replace("</", "<\\/")))
        elif url.path == "/browse":
            q = urllib.parse.parse_qs(url.query)
            res = self._browse(q.get("path", [""])[0])
            if res is None:
                self._send(400, "text/plain", "not a directory")
            else:
                self._send(200, "application/json", json.dumps(res))
        elif url.path == "/graph.dot":
            self._graph(urllib.parse.parse_qs(url.query))
        elif url.path == "/page.json":
            self._page(urllib.parse.parse_qs(url.query))
        else:
            self._send(404, "text/plain", "not found")

    def _graph(self, q):
        # Only validated numbers, a flag and the path of an existing .ibd file
        # reach the command line.
        ibd = self._valid_ibd(q.get("ibd", [""])[0])
        if not ibd:
            return self._send(400, "text/plain",
                              "Choose an existing .ibd file (use Browse)")
        try:
            root = int(q.get("root", [""])[0])
            max_nodes = int(q.get("max_nodes", [DEFAULT_MAX_NODES])[0])
            level = q.get("level", [""])[0]
            level = int(level) if level != "" else None
        except ValueError:
            return self._send(400, "text/plain",
                              "root page, max nodes and level must be numbers")
        if not 1 <= root < 2**32:
            return self._send(400, "text/plain", "root page out of range")
        if not 1 <= max_nodes <= MAX_NODES_LIMIT or (level is not None and
                                                    not 0 <= level <= 255):
            return self._send(400, "text/plain", "max_nodes or level out of range")

        cmd = [self.cfg.tool, ibd, str(root), "-i", "-g",
               "--graph-format", "dot", "--max-nodes", str(max_nodes)]
        if level is not None:
            cmd += ["--graph-level", str(level)]
        if q.get("unreachable", ["1"])[0] == "1":
            cmd.append("--graph-unreachable")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            return self._send(500, "text/plain", "could not run tool: %s" % e)
        if res.returncode != 0:
            msg = res.stderr.strip() or "tool exited with %d" % res.returncode
            return self._send(500, "text/plain", msg)
        dot, info = split_dot(res.stdout)
        # The DOT text rides along with the SVG for the page's change list.
        try:
            svg = subprocess.run([self.cfg.dot, "-Tsvg"], input=dot,
                                 capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            return self._send(500, "text/plain", "could not run dot: %s" % e)
        if svg.returncode != 0:
            return self._send(500, "text/plain", svg.stderr.strip() or "dot failed")
        self._send(200, "application/json",
                   json.dumps({"dot": dot, "info": info, "svg": svg.stdout}))

    def _page(self, q):
        """One storage page as the tool's JSON (page 0 means the root)."""
        ibd = self._valid_ibd(q.get("ibd", [""])[0])
        if not ibd:
            return self._send(400, "text/plain",
                              "Choose an existing .ibd file (use Browse)")
        try:
            root = int(q.get("root", [""])[0])
            page = int(q.get("page", ["0"])[0] or 0)
        except ValueError:
            return self._send(400, "text/plain",
                              "root page and page must be numbers")
        if not 1 <= root < 2**32 or not 0 <= page < 2**32:
            return self._send(400, "text/plain", "page number out of range")
        cmd = [self.cfg.tool, ibd, str(root), "-i", "--page-json",
               str(page or root)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            return self._send(500, "text/plain", "could not run tool: %s" % e)
        if res.returncode != 0:
            msg = res.stderr.strip() or "tool exited with %d" % res.returncode
            return self._send(500, "text/plain", msg)
        self._send(200, "application/json", res.stdout)

    def log_message(self, fmt, *args):
        pass  # Keep the terminal quiet; errors surface in the page.


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tool", help="path to svector_page_dump (default: next "
                    "to this script)")
    ap.add_argument("--start-dir",
                    default=os.environ.get("SVECTOR_DATA_DIR") or
                    os.path.expanduser("~"),
                    help="directory the file browser opens in (default: "
                    "$SVECTOR_DATA_DIR, else your home directory)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dot", default=shutil.which("dot") or "",
                    help="Graphviz 'dot' binary (default: found on PATH)")
    cfg = ap.parse_args()

    if not cfg.tool:
        cfg.tool = os.path.join(LAUNCH_DIR, "svector_page_dump")
    if not os.access(cfg.tool, os.X_OK):
        sys.exit("error: %s is not an executable" % cfg.tool)
    cfg.start_dir = os.path.realpath(os.path.expanduser(cfg.start_dir))
    if not os.path.isdir(cfg.start_dir):
        sys.exit("error: %s is not a directory" % cfg.start_dir)
    if not cfg.dot:
        sys.exit("error: Graphviz 'dot' not found on PATH (use --dot)")
    cfg.hosts = {"127.0.0.1:%d" % cfg.port, "localhost:%d" % cfg.port}

    Handler.cfg = cfg
    print("Rendering: %s" % cfg.dot)
    # Localhost only: the page can read the tablespace through the tool.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", cfg.port), Handler)
    print("Serving http://127.0.0.1:%d/  (Ctrl-C to stop)" % cfg.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
