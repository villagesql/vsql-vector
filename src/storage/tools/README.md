# SVECTOR Page Dump Tool

A standalone diagnostic tool for inspecting SVECTOR column storage pages in InnoDB tablespace files.

## Overview

This tool reads and displays the internal structure of SVECTOR storage pages directly from `.ibd` (InnoDB Data File) files. It's useful for:

- **Debugging**: Verify page format and data integrity
- **Development**: Understand the storage layout during development
- **Analysis**: Examine page utilization and fragmentation
- **Troubleshooting**: Diagnose storage-related issues

## Building

The tool can be built standalone with access to the VillageSQL SDK headers:

```bash
cd villagesql/examples/vsql-svector/src/storage/tools
make
```

By default `SOURCE_ROOT` is set to `../../../../../..` (the VillageSQL source root). Override it if your layout differs:

```bash
make SOURCE_ROOT=/path/to/villagesql-source
```

This will produce the `svector_page_dump` executable.

## Usage

```bash
svector_page_dump <ibd_file> <root_page_num> [options]
```

### Arguments

- `<ibd_file>`: Path to the InnoDB tablespace file (e.g., `table.ibd`)
- `<root_page_num>`: Page number of the SVECTOR root page within the tablespace

### Options

- `-v, --verbose`: Show detailed information including free slot arrays
- `-r, --records`: Display record data (shows first 10 records per page)
- `-d, --data-page N`: Display a specific data page by page number
- `-a, --all-pages`: Traverse and display all data pages linked from root
- `-i, --index`: Treat the root page as an HNSW index root page -- decode its
  metadata as index metadata (level, entry level/points) and decode data
  page records as HNSW NeighbourEntry/OverflowEntry instead of a plain
  SVECTOR vector
- `-g, --graph`: Render the HNSW graph reachable from the root page's entry
  point as ASCII tree art, walking every level via NID links. Requires `-i`
  and a level-0 primary store root page (see "Show the HNSW Graph" below)
- `--max-nodes N`: Cap on distinct nodes rendered by `-g`/`--graph` across
  all levels combined (default: 50)
- `--graph-style tree|list`: How `-g`/`--graph` renders each level -- `tree`
  (default) draws nested tree art; `list` prints one flat adjacency line per
  node instead (see "Show the HNSW Graph" below)
- `--graph-format text|dot`: `text` (default) is the ASCII output above; `dot`
  writes the graph to stdout as a Graphviz digraph and nothing else (see
  "Draw the HNSW Graph with Graphviz" below)
- `--graph-level N`: With `--graph-format dot`, draw only level N
- `--graph-unreachable`: With `--graph-format dot`, also draw level-0 records
  that exist but cannot be reached from the entry point
- `--page-json N`: Write page N (the root page itself, or a data page of the
  store) as one JSON object on stdout and nothing else. Use with `-i` for HNSW
  stores. This feeds the live viewer's Storage view (see section 9)
- `-h, --help`: Show help message

## Examples

### 1. Show Root Page Only

```bash
./svector_page_dump table.ibd 4
```

Output shows:
- Version and creator information
- Column size (vector dimensions)
- Total/free data page counts
- Free slot array summary

### 2. Verbose Root Page Display

```bash
./svector_page_dump table.ibd 4 -v
```

Additional output:
- Complete free slot array listing
- Details of each slot's page reference

### 3. Show Specific Data Page

```bash
./svector_page_dump table.ibd 4 -d 5
```

Shows data page #5:
- Page metadata
- Free list links
- Record capacity and utilization
- Record bitmap

### 4. Show Data Page with Records

```bash
./svector_page_dump table.ibd 4 -d 5 -r
```

Additional output:
- First 10 allocated records
- Transaction references
- Vector data values

### 5. Traverse All Pages

```bash
./svector_page_dump table.ibd 4 -a -v
```

Traverses all data pages linked from the root and displays:
- Each page's metadata and statistics
- Overall storage structure

### 6. Show an HNSW Index Root Page

```bash
./svector_page_dump table.ibd 4 -i -d 5 -r
```

Each HNSW level's primary and overflow store has its own root page (see
`src/index/hnsw/storage.h`), so `<root_page_num>` names one such store.
With `-i`, root page output shows:
- The store's name, level, and (for the level-0 primary store) the graph's
  entry level and entry point
- The record layout (NeighbourEntry or OverflowEntry, with max neighbours or
  overflow capacity) implied by column size, instead of a float vector
  dimension count

and data page records decode as NeighbourEntry (owner, lower-level link,
neighbours, overflow chain) or OverflowEntry (incoming links, overflow
chain) fields instead of vector floats.

### 7. Show the HNSW Graph

```bash
./svector_page_dump table.ibd 4 -i -g
```

Renders the graph reachable from `<root_page_num>`'s entry point as ASCII
tree art, one level at a time:

```
HNSW Graph (M=16, entry level 1, showing up to 50 nodes)

Level 1 (entry)
•4:0
├─ •7:2
└─ •12:5
   └─ •4:0  ...

Level 0
•4:0
├─ •2:9
│  ├─ •9:2
│  └─ •15:4
├─ •7:3
└─ •22:0
```

Only `<root_page_num>` (the level-0 primary store) is needed -- every other
level and page is found by following NID links embedded in each node's own
record, the same way a real search descends the graph. A neighbour already
drawn elsewhere in the same level is shown as a `...` leaf rather
than re-expanded, since HNSW levels are graphs, not trees. `--max-nodes`
(default 50) bounds how many distinct nodes are fetched across the whole
walk; nodes beyond that show as `(truncated)` leaves. Add `-v` to also show
each node's owner VID as `<node>(<vid>)`.

For levels with many edges per node, `--graph-style list` prints a flat
adjacency line per node instead of nested tree art:

```bash
./svector_page_dump table.ibd 4 -i -g --graph-style list
```

```
HNSW Graph (M=16, entry level 1, showing up to 50 nodes)

Level 1 (entry)
•4:0(10:0) -> [7:2(10:3)] degree=1

Level 0
•4:0(10:0) -> [2:9(10:5), 7:3(10:3)] degree=2
•2:9(10:5) -> [4:0(10:0), 9:2(10:7), 15:4(10:8)] degree=3
•7:3(10:3) -> [4:0(10:0)] degree=1
```

Every node reachable from the level's entry point is listed exactly once
(BFS order), each showing its own outgoing neighbours and degree; there is
no `...`/`(seen above)` marker since a node is never re-expanded once
listed.

### 8. Draw the HNSW Graph with Graphviz

```bash
./svector_page_dump table.ibd 4 -i -g --graph-format dot --max-nodes 200 > g.dot
dot -Tsvg g.dot -o g.svg                 # hierarchical layout from the entry node
sfdp -Tsvg g.dot -o g.svg                # better for large, dense levels
./svector_page_dump table.ibd 4 -i -g --graph-format dot --graph-level 0 \
    --graph-unreachable --max-nodes 500 | dot -Tsvg -o level0.svg
```

Uses the same walk as `--graph-style list`: every level is walked from the
entry point via lower-level links, sharing one `--max-nodes` budget. The
output is one DOT digraph with one cluster per level; edges never cross
levels.

- **Nodes** are circles labelled `NID` over `(VID)`, both `page:slot` as in the
  ASCII output (`12:5` is the NID, the node's record in this level; `(11:5)`
  is its VID, the vertex's identity across levels). A third line, `trx N`, is the record's transaction id in decimal. A double ring marks each
  level's entry node.
- **Edges** are the node's stored outgoing neighbour slots, one arrow each.
  A bold blue double-headed arrow is a reciprocal pair (drawn once); an
  orange single arrow is one-way, i.e. the target does not list the source.
  A dashed grey arrow points at a node whose adjacency was not read
  (truncated or unresolved), so reciprocity is unknown rather than assumed.
- **Node states**: dashed grey = truncated by `--max-nodes`; dashed red =
  unresolved (referenced but not a live record, reason in the tooltip); light
  red fill + `del N` (N = deleting trx id) = delete-marked but still allocated and linked; orange
  fill + `unreachable` = see below.
- A truncated diagram says `PARTIAL GRAPH` in its header and marks the cut
  nodes; it never looks complete.

#### Unreachable vs. deleted nodes

The entry-point walk only sees what is reachable, so by itself it cannot say
whether a node that disappeared was deleted or just lost its inbound edges.
`--graph-unreachable` answers that for **level 0** by enumerating every live
record in the level-0 store through its all-pages list (independent of the
walk) and drawing those the walk did not reach, with their outgoing edges.
Compare dumps taken before and after a delete:

- node gone from the store -> it is not listed at all (deleted);
- node still allocated but orange `unreachable` -> still stored, but nothing
  reachable links to it (a connectivity break);
- node still linked but marked `del N` -> delete-marked, not yet purged;
- a neighbour that is not a live record -> dashed red `unresolved` (dangling).

Limits: the check covers level 0 only (other levels' root pages are not known
from the level-0 root); it is skipped, with a note in the diagram, if the walk
was truncated, because unvisited nodes then cannot be told from unreachable
ones; and unreachable nodes drawn are capped at `--max-nodes` (the header
gives the true count). Incoming-flagged neighbour slots and overflow chains
are other nodes' edges into this one, not outgoing edges, so they are not
drawn.

### 9. Live browser view

```bash
cd <build dir>   # the build symlinks graph_view (this script) next to the tool
./graph_view   # or: --tool /path/to/svector_page_dump
# then open http://127.0.0.1:8765/
```

Start it with no options. The page has a **Browse** button that opens a file
browser over the machine's disk (sub-directories and `.ibd` files only; it
opens in `$SVECTOR_DATA_DIR` or your home directory, or set `--start-dir`),
plus a root page box. The chosen file and each file's root page are remembered
in the browser.

The page keeps the last N distinct results (default 50, set with "keep")
in the browser's local storage, so they survive a reload. The History
drop-down lists each with its time, file:root, node count and change summary
versus the previous result (`+added −gone ~changed`); pick one to view that
snapshot (outlined in amber, with its own change list) and "Back to latest" to
return. The `‹` `›` buttons beside the list (or the Left/Right arrow keys when
no field is focused) step to the older/newer result; `3 / 10` shows the
position. Auto-refresh keeps recording but never pulls the view off a
snapshot, and a refresh that finds an identical graph only updates the
time. "Clear" forgets them all.

A small Python (stdlib only) server that re-runs the tool with
`--graph-format dot` on every **Refresh** and redraws the graph in the browser
(rendered by a local Graphviz `dot`, which must be on `PATH` or given with
`--dot`; no internet is needed). Controls: level filter, max nodes, show-unreachable, auto-refresh
every 5 s, zoom. After each refresh the page lists nodes that disappeared,
appeared or changed state (e.g. `L0 12:4: ok -> unreachable`), which is the
quickest way to see what a `DELETE` did. Only validated numbers, the
unreachable flag and the path of an existing `.ibd` file reach the
tool's command line, and requests with a foreign `Host` header are refused; the server listens on
127.0.0.1 only. Tool errors are shown in the page.

#### Storage view

The **Storage** button next to the title switches from the graph to the
storage pages of the same file and root page (the tool's `--page-json`). It
shows one page at a time: the root page first, then any data page.

- Each record slot is a small square: green `A` = allocated, red `D` =
  delete-marked, dashed `F` = free. The tiny number is the slot, and the
  line under the mark is the record's VID (`page:slot`); a plain SVECTOR
  column has no VID, so its squares show only the slot. Click a square for
  its details: transaction id, lower-level NID, outgoing neighbours (click
  one to jump to its page and slot) and the overflow NID.
- **Root page** view: column size, page counts, the all-pages list head and
  tail, the pages that still have free slots, and the store's other root
  pages, each named by its store (e.g. `HNSW-L1`, `HNSW-L0-OV`); click one to
  open it as the new root. The other roots do not point back, so the viewer
  remembers (per file, in the browser) where you opened them from and shows
  "Primary root page" on them. Every root page shows both "Primary root
  page" and "Level root page" (the level's primary store: an overflow root's
  points to its same-level store); where one is the page itself it is a
  disabled button.
- **Data page** view: links to the root page, the previous/next page in the
  all-pages list and in the free-page list, and the slot counts.
- **Back** returns to the page you came from, **Root** goes to the root page,
  `Go` jumps to any page number, and the Left/Right arrow keys step to the
  previous/next page. Refresh and auto-refresh reload the page on screen.

Run `FLUSH TABLES`/checkpoint first if the `.ibd` on disk lags behind the
server's buffer pool.

## Finding Root Page Number

The root page number is stored in the table's metadata. You can find it by:

**From Information Schema** (if available):
- TBD TODO(villagesql-indexing):

## Implementation Notes

- The tool reads pages directly from the `.ibd` file using standard file I/O
- It uses the same page format constants as defined in `../root_page.h` and `../data_page.h`
- Requires the VillageSQL SDK headers (`villagesql/experimental/storage_api.h`) for page offset constants
- Assumes default InnoDB page size of 16KB
- Safety limit of 100 pages when using `--all-pages` to prevent infinite loops

## Limitations

- Currently only supports 16KB page size (standard InnoDB default)
- Vector data displayed as float32 (actual storage may use different precision)
- Does not validate checksums or page consistency
- Cannot handle compressed tablespaces
- Read-only tool - does not modify tablespace files

## Development

The tool consists of:

- `page_reader.{h,cc}`: Low-level page reading from `.ibd` files
- `root_page_parser.{h,cc}`: Parse and display SVECTOR root pages
- `data_page_parser.{h,cc}`: Parse and display SVECTOR data pages
- `hnsw_layout.{h,cc}`: Reimplements the HNSW index's on-disk metadata and
  record layouts (see `../../index/hnsw/storage.h` and `hnsw.h`) for `-i`
  mode, independent of the HNSW index sources
- `hnsw_graph.{h,cc}`: Walks and renders the HNSW graph as ASCII tree art for
  `-g`/`--graph`, following NID links level by level via `hnsw_layout.h`
- `hnsw_dot.{h,cc}`: SDK-free graph model, breadth-first collector and
  Graphviz DOT writer for `--graph-format dot` (unit-tested by
  `unittest/hnsw_dot_test.cc`); `hnsw_graph.cc` feeds it real records
- `storage_json.{h,cc}`: JSON writer for `--page-json` (unit-tested by
  `unittest/storage_json_test.cc`)
- `svector_page_dump.cc`: Main driver program

All components use the format definitions from `../root_page.h` and `../data_page.h` to stay in sync with the server implementation. `hnsw_layout.{h,cc}` must be kept in sync by hand with `../../index/hnsw/storage.h`/`hnsw.h` the same way, since it does not link against them.
