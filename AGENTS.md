# AGENTS.md

This file provides guidance to AI coding assistants (Claude Code, Gemini Code Assist, etc.) when working with code in this repository.

**Note**: Also check `AGENTS.local.md` for additional local development instructions when present.

## Project Overview

This is a vector extension for VillageSQL (a MySQL-compatible database) that provides an SVECTOR (static vector) data type with external columnar storage. The extension is built as a shared library (.so) packaged in a VEB (VillageSQL Extension Bundle) archive for installation.

## Build System

**Configure and Build:**
```bash
mkdir build && cd build
cmake .. -DVillageSQL_BUILD_DIR=/path/to/villagesql/build
make
```

**Install (optional):**
```bash
make install
```

The build process:
1. Uses CMake with `FindVillageSQL.cmake` to locate the VillageSQL Extension SDK
2. Compiles C++ source files into shared library `vsql-vector.so`
3. Packages library with `manifest.json` into `vsql_vector.veb` archive using `VEF_CREATE_VEB()`
4. Optionally installs VEB to `veb_output_directory` in the VillageSQL build tree

Also builds `svector_page_dump`, a standalone diagnostic tool for inspecting SVECTOR storage pages in InnoDB tablespace (.ibd) files. See `src/storage/tools/README.md` for usage.

**Requirements:**
- VillageSQL build tree (specified via `VillageSQL_BUILD_DIR`)
- C++17 compiler

**CMake Variables:**
- `VillageSQL_BUILD_DIR`: Path to VillageSQL build directory (required)

## Architecture

**Core Components:**
- `src/vector.cc` - All VDF implementations, type registration via `VEF_GENERATE_ENTRY_POINTS()`
- `src/native_vector.h` / `src/native_vector.cc` - Native vector type, encoding/decoding, distance functions
- `src/storage/storage.h` / `src/storage/storage.cc` - Column storage context and insert/delete/fetch operations
- `src/storage/root_page.h` / `src/storage/root_page.cc` - Root page structure, free slot management, page linking
- `src/storage/data_page.h` / `src/storage/data_page.cc` - Data page structure, record bitmap, slot management
- `src/storage/tools/` - Standalone `svector_page_dump` diagnostic tool
- `cmake/FindVillageSQL.cmake` - CMake module to locate the VillageSQL SDK

**Extension Registration:**
The extension uses the VillageSQL Extension Framework's fluent builder API to register:
- Custom `svector` type with encode/decode/compare functions and external column storage
- Vector SQL functions

**Available Functions:**
- `SVECTOR(dims)` - Create a vector with specified dimensions
- `SVECTOR_DISTANCE(v1, v2, metric)` - Compute distance between vectors (L1, L2, cosine, inner product)

**SVECTOR Type:**
- Fixed-dimension float32 vector stored in external columnar pages
- External storage enables efficient HNSW index construction and traversal
- Encoded as base64-like binary for SQL transport

**Storage Layout:**
- Root page: metadata, free page lists, column size
- Data pages: bitmap-managed slot arrays, each slot holds one vector

## Development Conventions

**Coding Style:**
- File names are lowercase with underscores
- Variables are lowercase with underscores
- Functions are lowercase with underscores
- Core storage code is in the `svector` namespace

## Testing

The extension includes a test suite using the MySQL Test Runner (MTR) framework:
- **Test Location**: `mysql-test/` directory with `.test` files and expected `.result` files

**Default: Using installed VEB**

```bash
cd /path/to/mysql-test
perl mysql-test-run.pl --suite=/path/to/vsql-vector/mysql-test
```

**Alternative: Using a VEB from the build directory**

```bash
cd /path/to/mysql-test
perl mysql-test-run.pl --suite=/path/to/vsql-vector/mysql-test --veb-source-dir=/path/to/vsql-vector/build
```

## Extension Installation

The extension registers a custom `SVECTOR` type and all functions automatically when loaded. The VEB package contains:
- `manifest.json` - Extension metadata
- `lib/vsql-vector.so` - Shared library with VDF implementations
