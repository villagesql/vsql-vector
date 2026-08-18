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
- `svector_page_dump.cc`: Main driver program

All components use the format definitions from `../root_page.h` and `../data_page.h` to stay in sync with the server implementation. `hnsw_layout.{h,cc}` must be kept in sync by hand with `../../index/hnsw/storage.h`/`hnsw.h` the same way, since it does not link against them.
