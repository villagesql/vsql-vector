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
- `svector_page_dump.cc`: Main driver program

All components use the format definitions from `../root_page.h` and `../data_page.h` to stay in sync with the server implementation. `hnsw_layout.{h,cc}` must be kept in sync by hand with `../../index/hnsw/storage.h`/`hnsw.h` the same way, since it does not link against them.
