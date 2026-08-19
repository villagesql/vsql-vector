![VillageSQL Logo](https://villagesql.com/assets/logo-light.svg)

# VillageSQL Vector Extension

An extension for VillageSQL Server that adds a vector data type with external columnar storage (SVECTOR), enabling efficient vector operations and laying the foundation for future ANN search and indexing.

> **This extension is under active development and is not stable.** It depends on [VillageSQL experimental extension APIs](https://villagesql.com/docs/mysql-8.4/0.0.5-dev/extension-api-reference#experimental-apis) that are subject to breaking changes without notice. It is not recommended for production use.

## Features

- **SVECTOR Type**: A float32 vector type with declared dimension and external columnar storage (up to 3072 dimensions)
- **Distance Functions**: L1 (Manhattan) and L2 (Euclidean) distance metrics
- **Similarity Functions**: Inner product, and angular distance via cosine distance
- **Utility Functions**: Norm computation, dimension query, hex dump, and formatted output
- **Native InnoDB Integration**: Columnar storage implemented via VillageSQL/InnoDB exposed storage APIs, inheriting ACID guarantees, MVCC, and buffer pool–based caching
- **Efficient Storage**: External columnar storage with heap-style organization (non-clustered), providing stable vector addresses suitable for ANN index structures
- **High Performance**: C++ implementation with bitmap-managed slot arrays and insert load distribution across pages via multiple free lists

## Installation

### Build from Source

#### Prerequisites
- VillageSQL build tree (specified via `VillageSQL_BUILD_DIR`)
- CMake 3.16 or higher
- C++17 compatible compiler

#### Build Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/villagesql/vsql-vector.git
   cd vsql-vector
   ```

2. Configure and build:
   ```bash
   mkdir -p build && cd build
   cmake .. -DVillageSQL_BUILD_DIR=/path/to/villagesql/build
   make -j$(nproc)
   ```

   This produces `vsql_vector.veb` in the build directory.

   To build with debug symbols and assertions (no optimization):
   ```bash
   cmake .. -DVillageSQL_BUILD_DIR=/path/to/villagesql/build -DWITH_DEBUG=ON
   ```

3. Install the VEB into the VillageSQL build tree (optional):
   ```bash
   make install
   ```

   This copies `vsql_vector.veb` to the `veb_output_directory` configured in the VillageSQL build tree, making it discoverable by the server without specifying a full path.

## Usage

### Loading the Extension

Because this extension depends on preview APIs, you must enable preview extensions before installing:

```sql
SET PERSIST vsql_allow_preview_extensions = ON;
INSTALL EXTENSION vsql_vector;
```

To unload:

```sql
UNINSTALL EXTENSION vsql_vector;
```

### SVECTOR Type

`SVECTOR(N)` declares a fixed length float32 vector column with dimension `N`. Every row in that column stores exactly `N` floats. The maximum supported dimension is 3072.

```sql
-- Create a table with a vector column
CREATE TABLE embeddings (
    id    INT PRIMARY KEY,
    vec   SVECTOR(4) NOT NULL
) ENGINE=InnoDB;

-- Add a vector column to an existing table
ALTER TABLE embeddings ADD COLUMN vec2 SVECTOR(4) NULL;

-- Insert vectors (number of elements must match the declared dimension)
INSERT INTO embeddings VALUES (1, '[1.0, 2.0, 3.0, 4.0]', NULL);  -- reference
INSERT INTO embeddings VALUES (2, '[1.0, 2.0, 3.0, 5.0]', NULL);  -- L1 dist from id=1: 1
INSERT INTO embeddings VALUES (3, '[1.0, 2.0, 5.0, 4.0]', NULL);  -- L1 dist from id=1: 2
INSERT INTO embeddings VALUES (4, '[2.0, 4.0, 6.0, 8.0]', NULL);  -- L1 dist from id=1: 10
INSERT INTO embeddings VALUES (5, '[9.0, 8.0, 7.0, 6.0]', NULL);  -- L1 dist from id=1: 20
```

### SQL Function Reference

#### Scalar / utility functions

| Function | Returns | Description |
|---|---|---|
| `VECTOR_DIMENSION(v)` | INT | Declared dimension of the vector |
| `VECTOR_MAX_DIMENSION()` | INT | Maximum supported dimension (3072) |
| `VECTOR_NORM(v)` | REAL | L2 (Euclidean) norm |
| `VECTOR_FORMAT(v, precision)` | STRING | Vector as a fixed-precision decimal string |
| `VECTOR_HEX(v)` | STRING | Raw float bytes as uppercase hex (useful for debugging) |

#### Distance and similarity functions

| Function | Returns | Description |
|---|---|---|
| `L1_DISTANCE(v1, v2)` | REAL | L1 (Manhattan) distance — sum of absolute differences |
| `L2_DISTANCE(v1, v2)` | REAL | L2 (Euclidean) distance — square root of sum of squared differences |
| `COSINE_DISTANCE(v1, v2)` | REAL | Cosine distance — `1 - cosine_similarity`; range [0, 2] |
| `INNER_PRODUCT(v1, v2)` | REAL | Dot product (similarity, not a metric); higher means more similar |

Both arguments to a distance/similarity function must have the same dimension. All functions return NULL if either argument is NULL.

### Example Queries

```sql
-- Norm of a stored vector
SELECT id, VECTOR_NORM(vec) AS norm FROM embeddings ORDER BY id;

-- L2 distance between two stored vectors
SELECT L2_DISTANCE(a.vec, b.vec) AS dist
FROM embeddings a, embeddings b
WHERE a.id = 1 AND b.id = 2;

-- Nearest-neighbour search by L1 distance (full table scan)
-- Note: HNSW index support is planned; today this performs a sequential scan.
-- The query vector is stored in a table row and joined in as a workaround
-- for the current limitation on inline constant vectors (see Known Limitations).
SELECT id, L1_DISTANCE(vec, query.ref_vec) AS dist
FROM embeddings,
     (SELECT vec AS ref_vec FROM embeddings WHERE id = 1) AS query
ORDER BY dist ASC
LIMIT 2;
-- Expected result: id=1 dist=0, id=2 dist=1

-- TODO: once inline constant vector support is added, the intended syntax is:
-- SELECT id, L1_DISTANCE(vec, '[1.0, 2.0, 3.0, 4.0]') AS dist
-- FROM embeddings
-- ORDER BY dist ASC
-- LIMIT 2;

-- Update a vector value
UPDATE embeddings SET vec = '[0.5, 0.5, 0.5, 0.5]' WHERE id = 1;

-- Delete a row containing a vector
DELETE FROM embeddings WHERE id = 2;
```

### Known Limitations

- **Inline constant vectors**: Using `SVECTOR::FROM_STRING(...)` as a direct function argument (e.g., `SVECTOR_DISTANCE_L2(col, SVECTOR::FROM_STRING('[1,2,3,4]'))`) currently fails because constant folding for parameterized custom types is not yet supported. As a workaround, store the query vector in a table row and join against it (as shown in the nearest-neighbour example above).

## Testing

The extension includes tests using the MySQL Test Runner (MTR) framework.

### Running Tests

**Option 1 (Default): Using installed VEB**

```bash
cd /path/to/villagesql/build/mysql-test
perl mysql-test-run.pl --suite=/path/to/vsql-vector/mysql-test
```

**Option 2: Using a VEB from the build directory**

```bash
cd /path/to/villagesql/build/mysql-test
perl mysql-test-run.pl --suite=/path/to/vsql-vector/mysql-test --veb-source-dir=/path/to/vsql-vector/build
```

## Diagnostic Tools

The build also produces `svector_page_dump`, a standalone tool for inspecting SVECTOR storage pages in InnoDB tablespace files. See `src/storage/tools/README.md` for full usage.

### Example: inspecting a data page with delete-marked records

The following example shows output after inserting 5 rows into a `SVECTOR(4)` column and then deleting two of them (rows with id=2 and id=4) before the InnoDB purge thread has run. The delete-marked slots remain physically present in the page until purge.

```bash
$ svector_page_dump embeddings.ibd 4 -d 5 -r
```

```
IBD File: embeddings.ibd
Root Page Number: 4

SVECTOR Root Page
=================

Version:           1
Page Type:         1 (ROOT_PAGE)
Creator:           SVECTOR
Column Size:       16 bytes (4-dim float vector)

Data Pages:
  Total:           1
  Free:            1
  Head:            Page #5
  Tail:            Page #5

Free Slot Array:
  Max Capacity:    2048 slots
  Current Size:    1 slots
  Non-empty Slots: 1

================================================================================

SVECTOR Data Page
=================

Version:           1
Page Type:         2 (DATA_PAGE)
Free Slot Number:  0

SVECTOR Data Page Links:
  Previous:        Page #4294967295 (NULL)
  Next:            Page #4294967295 (NULL)

SVECTOR Free Page Links:
  Previous:        Page #4294967295 (NULL)
  Next:            Page #4294967295 (NULL)

Capacity:
  Max Records:     673
  Free Records:    668 (99.3%)
  Allocated:       5 (0.7%)
    Active:        3
    Deleted:       2

Record Bitmap:
  AADAD...................................
  ........................................
  (remaining 633 free slots omitted)
  (. = Free, A = Active, D = Deleted)

Records (showing from slot 0, up to 10 records):
  [  0] Trx ID:        1001 Data:[0.10, 0.20, 0.30, 0.40]
  [  1] Trx ID:        1002 Data:[0.90, 0.80, 0.70, 0.60] (DELETED)
  [  2] Trx ID:        1003 Data:[0.50, 0.50, 0.50, 0.50]
  [  3] Trx ID:        1004 Data:[3.14, 2.72, 1.41, 1.73] (DELETED)
  [  4] Trx ID:        1005 Data:[0.11, 0.22, 0.33, 0.44]
```

Key observations:
- **`DELETED` records** (slots 1 and 3) are still physically present and visible to concurrent transactions that started before the DELETE committed (MVCC). They are reclaimed by the purge thread once no active transaction can see them.
- **Record Bitmap** encodes each slot's state in 2 bits: `A` = active (occupied, not deleted), `D` = delete-marked (occupied, pending purge), `.` = free (available for insert).
- **Free Slot Number `0`** means this data page is tracked at index 0 in the root page's free slot array, making it eligible for the next insert without a root page scan.
- **Max Records `673`** is the page capacity for `SVECTOR(4)` on a 16 KB InnoDB page: `54 (header) + ⌈673×2/8⌉ (bitmap) + 673×24 (records) + 8 (trailer) = 16383 bytes`.

## Development

### Project Structure
```
vsql-vector/
├── src/
│   ├── native_vector.h      # Vector type definitions and distance functions
│   ├── native_vector.cc     # Encoding/decoding implementations
│   ├── vector.cc            # VDF implementations and extension registration
│   └── storage/
│       ├── storage.h        # ColumnStorage class declarations
│       ├── storage.cc       # Insert/delete/purge/fetch operations
│       ├── root_page.h      # Root page structure and free slot management
│       ├── root_page.cc     # Root page operations
│       ├── data_page.h      # Data page structure and slot management
│       ├── data_page.cc     # Record insert/purge and bitmap operations
│       └── tools/
│           ├── README.md             # Page dump tool documentation
│           ├── svector_page_dump.cc  # Page dump tool driver
│           ├── page_reader.h/cc      # IBD file reader
│           ├── root_page_parser.h/cc # Root page parser
│           └── data_page_parser.h/cc # Data page parser
├── cmake/
│   └── FindVillageSQL.cmake # CMake module to locate VillageSQL SDK
├── mysql-test/
│   ├── t/                   # MTR test files
│   └── r/                   # MTR expected results
├── manifest.json            # VEB package manifest
└── CMakeLists.txt           # Build configuration
```

## Roadmap

- [ ] Support for inline constant vectors and bound parameter vectors in distance functions
- [ ] HNSW index for approximate nearest-neighbour (ANN) search

## Reporting Bugs and Requesting Features

If you encounter a bug or have a feature request, please open an [issue](https://github.com/villagesql/vsql-vector/issues) using GitHub Issues.

## License

License information can be found in the [LICENSE](./LICENSE) file.

## Contributing

VillageSQL welcomes contributions from the community. For more information, please see the [VillageSQL Contributing Guide](https://github.com/villagesql/villagesql-server/blob/main/CONTRIBUTING.md).

## Contact

- File a [bug or issue](https://github.com/villagesql/vsql-vector/issues) and we will review
- Start a discussion in the project [discussions](https://github.com/villagesql/vsql-vector/discussions)
- Join the [Discord channel](https://discord.gg/KSr6whd3Fr)
